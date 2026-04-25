use super::{node::projected_schema, *};

pub(super) struct PreparedCugraphInputs {
    pub(super) execution_spec: CugraphAlgorithmSpec,
    pub(super) edge_schema: SchemaRef,
    pub(super) edge_frame: GpuDataFrame,
    pub(super) predicate_frames: Vec<CugraphGpuFrame>,
    pub(super) logical_codebook: Option<Codebook>,
}

pub(super) fn prepare_bfs_logical_gpu_inputs(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    edge_frame: GpuDataFrame,
    predicate_frames: Vec<CugraphGpuFrame>,
) -> Result<PreparedCugraphInputs> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(PreparedCugraphInputs {
            execution_spec: spec.clone(),
            edge_schema,
            edge_frame,
            predicate_frames,
            logical_codebook: None,
        });
    };
    if !bfs_uses_logical_mapping(spec, params, &edge_schema)? {
        return Ok(PreparedCugraphInputs {
            execution_spec: spec.clone(),
            edge_schema,
            edge_frame,
            predicate_frames,
            logical_codebook: None,
        });
    }
    let runtime = counted_cudf_runtime(
        context,
        "datafusion_nexus.cugraph_sql.logical_mapping.runtime",
        "query service closed before cuGraph logical mapping acquired its runtime",
    )?;
    if spec.edge_id_col.is_some() || params.predicates.include_edge_ids.is_some() {
        return Err(plan_error(
            "cuGraph BFS logical vertex-id mapping does not yet support edge_id_col or include_edge_ids predicates",
        ));
    }

    let source_edge_frame = edge_frame.clone();
    let edge_frame = edge_frame.into_frame();
    validate_bfs_logical_edge_schema(spec, &edge_schema)?;
    let mut domain_parts = vec![cudf_series(&edge_frame, &spec.src_col)?, cudf_series(&edge_frame, &spec.dst_col)?];
    let literal_source_series = bfs_literal_string_source_series(context, params)?;
    if let Some(source_series) = &literal_source_series {
        domain_parts.push(source_series.clone());
    }
    if let CugraphBfsSource::Relation(source) = &params.source {
        let source_frame = predicate_frames
            .first()
            .ok_or_else(|| plan_error("BFS source_vertices relation source was not planned"))?;
        domain_parts.push(cudf_series(source_frame.frame.frame(), &source.column)?);
    }
    for (index, frame) in predicate_frames.iter().enumerate() {
        collect_bfs_logical_predicate_domain_parts(params, index, frame, &mut domain_parts)?;
    }
    let domain_refs = domain_parts.iter().collect::<Vec<_>>();
    let domain = Series::concat(&domain_refs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to concatenate cuGraph logical BFS mapping domain on GPU",
            nexus_query_engine::error::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to concatenate cuGraph logical BFS mapping domain on GPU",
                error,
            ),
        ))
    })?;
    let codebook = domain.codebook().map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to build cuGraph logical BFS codebook on GPU",
            nexus_query_engine::error::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to build cuGraph logical BFS codebook on GPU",
                error,
            ),
        ))
    })?;

    let mapped_edge = encode_bfs_logical_edge_frame(spec, edge_frame, &codebook)?;
    let mapped_edge_schema = remap_bfs_logical_schema(&edge_schema, &[&spec.src_col, &spec.dst_col])?;
    let mut mapped_predicates = Vec::with_capacity(predicate_frames.len() + 1);
    if let Some(source_series) = literal_source_series {
        let source_code = codebook
            .encode(&source_series)
            .map_err(cudf_logical_mapping_error)?
            .into_series()
            .copy_into_runtime(&runtime)
            .map_err(cudf_logical_mapping_error)?;
        let source_frame =
            DataFrame::from_series([("__source_vertex", source_code)]).map_err(cudf_logical_mapping_error)?;
        let source_schema =
            Arc::new(Schema::new(vec![arrow::datatypes::Field::new("__source_vertex", DataType::Int32, false)]));
        mapped_predicates.push(CugraphGpuFrame {
            schema: source_schema,
            frame: retain_mapped_cugraph_frame(
                source_frame,
                &source_edge_frame,
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to retain mapped cuGraph logical source frame residency",
            )?,
        });
    }
    for (index, frame) in predicate_frames.into_iter().enumerate() {
        mapped_predicates.push(encode_bfs_logical_predicate_frame(params, index, frame, &codebook)?);
    }

    let mut execution_spec = spec.clone();
    execution_spec.output_schema = bfs_numeric_execution_schema(params)?;
    Ok(PreparedCugraphInputs {
        execution_spec,
        edge_schema: mapped_edge_schema,
        edge_frame: retain_mapped_cugraph_frame(
            mapped_edge,
            &source_edge_frame,
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to retain mapped cuGraph logical edge frame residency",
        )?,
        predicate_frames: mapped_predicates,
        logical_codebook: Some(codebook),
    })
}

pub(super) fn retain_mapped_cugraph_frame(
    frame: DataFrame,
    source: &GpuDataFrame,
    operation: &'static str,
    message: &'static str,
) -> Result<GpuDataFrame> {
    GpuDataFrame::try_new_with_source_residency(frame, source)
        .map_err(|error| DataFusionError::from(Error::from_native(operation, message, error)))
}

pub(super) fn validate_bfs_logical_edge_schema(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<()> {
    for column in [&spec.src_col, &spec.dst_col] {
        let field = edge_schema.field_with_name(column).map_err(|_| {
            plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{column}'"))
        })?;
        if !is_bfs_logical_vertex_type(field.data_type()) {
            return Err(plan_error(format!(
                "cuGraph BFS logical vertex mapping requires a string edge column '{column}', found {:?}",
                field.data_type()
            )));
        }
    }
    Ok(())
}

pub(super) fn bfs_uses_logical_mapping(
    spec: &CugraphAlgorithmSpec,
    params: &super::CugraphBfsParams,
    edge_schema: &SchemaRef,
) -> Result<bool> {
    if params.source.is_utf8() {
        return Ok(true);
    }
    if !matches!(params.source, CugraphBfsSource::Relation(_)) {
        return Ok(false);
    }
    let source_field = edge_schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{}'", spec.src_col))
    })?;
    let destination_field = edge_schema.field_with_name(&spec.dst_col).map_err(|_| {
        plan_error(format!("cuGraph BFS logical vertex mapping could not find edge column '{}'", spec.dst_col))
    })?;
    Ok(is_bfs_logical_vertex_type(source_field.data_type())
        && is_bfs_logical_vertex_type(destination_field.data_type()))
}

pub(super) fn bfs_literal_string_source_series(
    context: &ExecutionContext,
    params: &super::CugraphBfsParams,
) -> Result<Option<Series>> {
    let Some(values) = params.source.literal_values() else {
        return Ok(None);
    };
    if !values.first().is_some_and(CugraphBfsSourceVertex::is_utf8) {
        return Ok(None);
    }
    let values = values
        .iter()
        .map(|value| match value {
            CugraphBfsSourceVertex::Utf8(value) => Ok(value.as_str()),
            CugraphBfsSourceVertex::Int64(_) => Err(plan_error("string BFS source list contained a numeric value")),
        })
        .collect::<Result<Vec<_>>>()?;
    let runtime = counted_cudf_runtime(
        context,
        "datafusion_nexus.cugraph_sql.logical_mapping.runtime",
        "query service closed before cuGraph logical mapping acquired its runtime",
    )?;
    runtime.series(values.as_slice()).map(Some).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to create GPU source-vertices series for cuGraph logical BFS mapping",
            nexus_query_engine::error::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to create GPU source-vertices series for cuGraph logical BFS mapping",
                error,
            ),
        ))
    })
}

pub(super) fn is_bfs_logical_vertex_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

pub(super) fn collect_bfs_logical_predicate_domain_parts(
    params: &super::CugraphBfsParams,
    index: usize,
    frame: &CugraphGpuFrame,
    domain_parts: &mut Vec<Series>,
) -> Result<()> {
    let source_offset = bfs_planned_source_input_offset(params);
    if index < source_offset {
        return Ok(());
    }
    let index = index - source_offset;
    let mut expected_index = 0;
    if let Some(input) = &params.predicates.include_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.exclude_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.target_vertices {
        if index == expected_index {
            domain_parts.push(cudf_series(frame.frame.frame(), &input.column)?);
        }
        expected_index += 1;
    }
    if params.predicates.include_edge_ids.is_some() {
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        domain_parts.push(cudf_series(frame.frame.frame(), &input.source_col)?);
        domain_parts.push(cudf_series(frame.frame.frame(), &input.destination_col)?);
    }
    Ok(())
}

pub(super) fn encode_bfs_logical_edge_frame(
    spec: &CugraphAlgorithmSpec,
    frame: DataFrame,
    codebook: &Codebook,
) -> Result<DataFrame> {
    let frame = encode_column(frame, &spec.src_col, codebook)?;
    encode_column(frame, &spec.dst_col, codebook)
}

pub(super) fn encode_bfs_logical_predicate_frame(
    params: &super::CugraphBfsParams,
    index: usize,
    frame: CugraphGpuFrame,
    codebook: &Codebook,
) -> Result<CugraphGpuFrame> {
    let mut expected_index = 0;
    let source_frame = frame.frame.clone();
    let mut raw = frame.frame.into_frame();
    let mut encoded_columns = Vec::new();
    let source_offset = bfs_planned_source_input_offset(params);
    let is_source_frame = index < source_offset;
    if let CugraphBfsSource::Relation(input) = &params.source
        && is_source_frame
    {
        encoded_columns.push(input.column.as_str());
    }
    if is_source_frame {
        for column in &encoded_columns {
            raw = encode_column(raw, column, codebook)?;
        }
        return Ok(CugraphGpuFrame {
            schema: remap_bfs_logical_schema(&frame.schema, &encoded_columns)?,
            frame: retain_mapped_cugraph_frame(
                raw,
                &source_frame,
                "datafusion_nexus.cugraph_sql.logical_mapping",
                "failed to retain mapped cuGraph logical predicate frame residency",
            )?,
        });
    }
    let index = index.saturating_sub(source_offset);
    if let Some(input) = &params.predicates.include_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.exclude_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.target_vertices {
        if index == expected_index {
            encoded_columns.push(input.column.as_str());
        }
        expected_index += 1;
    }
    if params.predicates.include_edge_ids.is_some() {
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        encoded_columns.push(input.source_col.as_str());
        encoded_columns.push(input.destination_col.as_str());
    }
    for column in &encoded_columns {
        raw = encode_column(raw, column, codebook)?;
    }
    Ok(CugraphGpuFrame {
        schema: remap_bfs_logical_schema(&frame.schema, &encoded_columns)?,
        frame: retain_mapped_cugraph_frame(
            raw,
            &source_frame,
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "failed to retain mapped cuGraph logical predicate frame residency",
        )?,
    })
}

pub(super) fn bfs_planned_source_input_offset(params: &super::CugraphBfsParams) -> usize {
    usize::from(params.source.planned_source_vertices_input())
}

pub(super) fn encode_column(frame: DataFrame, column: &str, codebook: &Codebook) -> Result<DataFrame> {
    let encoded = codebook.encode(&cudf_series(&frame, column)?).map_err(cudf_logical_mapping_error)?.into_series();
    frame.with_column(column, encoded).map_err(cudf_logical_mapping_error)
}

pub(super) fn remap_bfs_logical_schema(schema: &SchemaRef, columns: &[&str]) -> Result<SchemaRef> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if columns.iter().any(|column| *column == field.name()) {
                arrow::datatypes::Field::new(field.name(), DataType::Int32, false)
            } else {
                field.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn bfs_numeric_execution_schema(params: &super::CugraphBfsParams) -> Result<SchemaRef> {
    let native_schema = bfs_native_result_schema(params.output.return_target_info);
    let fields: Vec<_> = match params.output.mode {
        CugraphBfsOutputMode::Raw => native_schema.fields().iter().map(|field| field.as_ref().clone()).collect(),
        CugraphBfsOutputMode::Normalized => Arc::new(Schema::new(vec![
            normalized_bfs_base_field(&native_schema, "vertex", false)?,
            normalized_bfs_base_field(&native_schema, "distance", true)?,
            normalized_bfs_base_field(&native_schema, "predecessor", true)?,
            arrow::datatypes::Field::new("reachable", DataType::Boolean, false),
        ]))
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .chain(native_schema.fields().iter().skip(3).map(|field| field.as_ref().clone()))
        .collect(),
        CugraphBfsOutputMode::Path => native_schema.fields().iter().map(|field| field.as_ref().clone()).collect(),
    };
    Ok(Arc::new(Schema::new(fields)))
}

pub(super) fn bfs_native_result_schema(return_target_info: bool) -> SchemaRef {
    let mut native_params = GraphBfsParams::new(0, None);
    native_params.return_target_info = return_target_info;
    GraphAlgorithm::Bfs.definition().result_schema_for_params(&GraphAlgorithmParams::Bfs(Box::new(native_params)))
}

pub(super) fn normalized_bfs_base_field(
    schema: &SchemaRef,
    name: &'static str,
    nullable: bool,
) -> Result<arrow::datatypes::Field> {
    let field = schema.field_with_name(name).map_err(|error| {
        DataFusionError::from(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.bfs_result_schema",
                "native BFS descriptor is missing a raw BFS base field",
            )
            .with_fact("column", name)
            .with_source(error),
        )
    })?;
    Ok(arrow::datatypes::Field::new(field.name(), field.data_type().clone(), nullable))
}

pub(super) fn decode_bfs_logical_gpu_output(
    context: &ExecutionContext,
    handle: GpuResultHandle,
    codebook: &Codebook,
    output_schema: SchemaRef,
) -> Result<GpuResultHandle> {
    let runtime = counted_cudf_runtime(
        context,
        "datafusion_nexus.cugraph_sql.logical_mapping.runtime",
        "query service closed before cuGraph logical output mapping acquired its runtime",
    )?;
    let raw_frame = handle.into_gpu_frame();
    let raw = raw_frame.frame();
    let vertex =
        decode_code_column(raw, "vertex", codebook)?.copy_into_runtime(&runtime).map_err(cudf_logical_mapping_error)?;
    let distance = cudf_series(raw, "distance")?.copy_into_runtime(&runtime).map_err(cudf_logical_mapping_error)?;
    let predecessor = decode_bfs_predecessor_code_column(raw, codebook)?
        .copy_into_runtime(&runtime)
        .map_err(cudf_logical_mapping_error)?;
    let mut columns = vec![("vertex", vertex), ("distance", distance), ("predecessor", predecessor)];
    let raw_column_names = raw.column_names();
    if raw_column_names.contains(&"reachable") {
        let reachable =
            cudf_series(raw, "reachable")?.copy_into_runtime(&runtime).map_err(cudf_logical_mapping_error)?;
        columns.push(("reachable", reachable));
    }
    if raw_column_names.contains(&"target_found") {
        let target_found =
            cudf_series(raw, "target_found")?.copy_into_runtime(&runtime).map_err(cudf_logical_mapping_error)?;
        let target_distance =
            cudf_series(raw, "target_distance")?.copy_into_runtime(&runtime).map_err(cudf_logical_mapping_error)?;
        columns.push(("target_found", target_found));
        columns.push(("target_distance", target_distance));
    }
    let frame = DataFrame::from_series(columns).map_err(cudf_logical_mapping_error)?;
    let frame = GpuDataFrame::try_new_with_source_residency(frame, &raw_frame).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_output",
            "failed to retain decoded cuGraph logical BFS output residency",
            error,
        ))
    })?;
    GpuResultHandle::new(context, frame, output_schema).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.logical_output",
            "failed to bind decoded cuGraph logical BFS output",
            error,
        ))
    })
}

pub(super) fn decode_code_column(frame: &DataFrame, column: &str, codebook: &Codebook) -> Result<Series> {
    let codes = cudf_series(frame, column)?.astype(cudf::TypeId::Int32).map_err(cudf_logical_mapping_error)?;
    codebook.decode_series(&codes).map_err(cudf_logical_mapping_error)
}

pub(super) fn decode_bfs_predecessor_code_column(frame: &DataFrame, codebook: &Codebook) -> Result<Series> {
    let codes = cudf_series(frame, "predecessor")?.astype(cudf::TypeId::Int32).map_err(cudf_logical_mapping_error)?;
    let code_frame = DataFrame::from_series([("__predecessor_code", codes)]).map_err(cudf_logical_mapping_error)?;
    let code_frame = code_frame
        .with_columns([(
            "__predecessor_code",
            cudf::ops::when(cudf::ops::col("__predecessor_code").ge(0i32))
                .then(cudf::ops::col("__predecessor_code"))
                .otherwise(cudf::ops::lit_null(cudf::DataType::new(cudf::TypeId::Int32))),
        )])
        .map_err(cudf_logical_mapping_error)?;
    codebook.decode_series(&cudf_series(&code_frame, "__predecessor_code")?).map_err(cudf_logical_mapping_error)
}

pub(super) fn apply_gpu_projection_limit(
    context: &ExecutionContext,
    metrics: &mut GraphExecutionMetrics<'_>,
    handle: GpuResultHandle,
    schema: SchemaRef,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
) -> Result<GpuResultHandle> {
    if projection.is_none() && limit.is_none() {
        return Ok(handle);
    }
    let input_name = "__datafusion_nexus_cugraph_logical_output";
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(input_name.to_string(), handle);
    let mut builder = QueryPlanBuilder::new();
    let mut root = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&schema))));
    let input_schema = Arc::clone(&schema);
    let mut root_schema = schema;
    if let Some(projection) = projection {
        let outputs = projection
            .iter()
            .map(|index| {
                let field =
                    input_schema.fields().get(*index).map(|field| field.as_ref().clone()).ok_or_else(|| {
                        plan_error(format!("cuGraph output projection index {index} is out of range"))
                    })?;
                Ok(ProjectionOutput::new(nexus_query_engine::expr::Expr::column(field.name()), field))
            })
            .collect::<Result<Vec<_>>>()?;
        root_schema = projected_schema(&input_schema, Some(projection))?;
        root = builder.add_projection(root, outputs);
    }
    if let Some(limit) = limit {
        root = builder.add_limit(root, LimitSpec::fetch(limit), Arc::clone(&root_schema));
    }
    let plan = builder.finish_gpu_resident(root);
    let outcome =
        execute_query_plan_with_gpu_inputs_observed(context, &plan, &inputs, metrics.runtime_observation_config());
    let output = finalize_graph_query_outcome(
        metrics,
        outcome,
        "datafusion_nexus.cugraph_sql.logical_output",
        "failed to apply GPU projection/limit to decoded cuGraph logical BFS output",
    )?;
    let QueryOutput::GpuResident(handle) = output else {
        return Err(plan_error("decoded cuGraph logical BFS projection did not produce a GPU output"));
    };
    Ok(handle)
}
