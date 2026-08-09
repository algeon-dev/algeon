use std::{collections::BTreeSet, mem::size_of, sync::Arc};

use super::*;

pub(super) struct PreparedCugraphInputSchemas {
    pub(super) execution_spec: CugraphAlgorithmSpec,
    pub(super) edge_schema: SchemaRef,
    pub(super) predicate_schemas: Vec<SchemaRef>,
    pub(super) logical_mapping: bool,
}

#[derive(Clone, Debug)]
pub(super) struct HostBfsCodebook {
    values: Arc<Vec<String>>,
}

impl HostBfsCodebook {
    fn from_sources(sources: &[(&StagedCugraphSource, Vec<String>)]) -> Result<Self> {
        Self::from_sources_with_limit(sources, i32::MAX as usize)
    }

    pub(super) fn from_sources_with_limit(
        sources: &[(&StagedCugraphSource, Vec<String>)],
        max_distinct: usize,
    ) -> Result<Self> {
        let mut values = BTreeSet::new();
        for (source, columns) in sources {
            let StagedCugraphSource::Host { schema, batches } = source else {
                return Err(logical_bfs_native_source_error("codebook", "native"));
            };
            for column in columns {
                let field = schema.field_with_name(column).map_err(|error| {
                    DataFusionError::from(
                        Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                            "logical BFS host mapping could not find a required vertex column",
                        )
                        .with_fact("column", column)
                        .with_source(error),
                    )
                })?;
                if !is_bfs_logical_vertex_type(field.data_type()) {
                    return Err(Error::new(
                        ErrorCode::UnsupportedType,
                        "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                        "logical BFS host mapping requires UTF8 vertex columns",
                    )
                    .with_fact("column", column)
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                    .with_fact("reason", "logical_bfs_vertex_dtype_unsupported")
                    .into());
                }
                for batch in batches {
                    let index = batch.schema().index_of(column).map_err(DataFusionError::from)?;
                    collect_string_values(batch.column(index).as_ref(), &mut values, column)?;
                }
            }
        }
        if values.len() > max_distinct {
            return Err(Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host codebook exceeds its checked Int32 domain",
            )
            .with_fact("reason", "logical_bfs_codebook_overflow")
            .with_fact("distinct_values", values.len())
            .with_fact("max_distinct_values", max_distinct)
            .into());
        }
        let values = Arc::new(values.into_iter().collect::<Vec<_>>());
        for (index, _) in values.iter().enumerate() {
            i32::try_from(index).map_err(|_| {
                Error::new(
                    ErrorCode::SizeOverflow,
                    "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                    "logical BFS host codebook index does not fit Int32",
                )
                .with_fact("reason", "logical_bfs_codebook_overflow")
                .with_fact("index", index)
            })?;
        }
        Ok(Self { values })
    }

    pub(super) fn code(&self, value: &str) -> Option<i32> {
        self.values
            .binary_search_by(|candidate| candidate.as_str().cmp(value))
            .ok()
            .and_then(|index| i32::try_from(index).ok())
    }

    pub(super) fn decode_code(&self, code: i64) -> Result<String> {
        let index = usize::try_from(code).map_err(|_| logical_bfs_unknown_code_error(code))?;
        self.values.get(index).cloned().ok_or_else(|| logical_bfs_unknown_code_error(code))
    }

    pub(super) fn memory_size(&self) -> Result<usize> {
        let vector_bytes = self.values.capacity().checked_mul(size_of::<String>());
        let string_bytes = self.values.iter().try_fold(0usize, |total, value| total.checked_add(value.capacity()));
        vector_bytes.and_then(|vector| string_bytes.and_then(|strings| vector.checked_add(strings))).ok_or_else(|| {
            Error::new(
                ErrorCode::SizeOverflow,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host codebook size overflowed usize",
            )
            .with_fact("reason", "logical_bfs_codebook_size_overflow")
            .into()
        })
    }
}

pub(super) fn prepare_bfs_logical_host_inputs(
    spec: &CugraphAlgorithmSpec,
    edge: StagedCugraphSource,
    logical_literal_source: Option<StagedCugraphSource>,
    predicates: Vec<StagedCugraphSource>,
    staged_bytes: &mut usize,
    max_staging_bytes: usize,
    reservation: &mut MemoryReservation,
) -> Result<(
    StagedCugraphSource,
    Option<StagedCugraphSource>,
    Vec<StagedCugraphSource>,
    Option<HostBfsCodebook>,
    CugraphAlgorithmSpec,
)> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok((edge, logical_literal_source, predicates, None, spec.clone()));
    };
    if !bfs_uses_logical_mapping(spec, params, &edge.schema())? {
        return Ok((edge, logical_literal_source, predicates, None, spec.clone()));
    }

    let StagedCugraphSource::Host { .. } = &edge else {
        return Err(logical_bfs_native_source_error("edge", "native"));
    };
    if let Some(source) = &logical_literal_source
        && !matches!(source, StagedCugraphSource::Host { .. })
    {
        return Err(logical_bfs_native_source_error("source_vertices", "native"));
    }
    if let Some((index, _)) =
        predicates.iter().enumerate().find(|(_, source)| matches!(source, StagedCugraphSource::Native { .. }))
    {
        return Err(logical_bfs_native_source_error("side_input", format!("native_index_{index}")));
    }

    validate_bfs_logical_edge_schema(spec, &edge.schema())?;
    let edge_columns = vec![spec.src_col.clone(), spec.dst_col.clone()];
    let literal_columns = logical_literal_source.as_ref().map(|_| vec!["__source_vertex".to_string()]);
    let predicate_columns = predicates
        .iter()
        .enumerate()
        .map(|(index, _)| bfs_logical_predicate_columns(params, index))
        .collect::<Vec<_>>();

    let mut domain_sources = Vec::with_capacity(2 + predicates.len() + usize::from(logical_literal_source.is_some()));
    domain_sources.push((&edge, edge_columns));
    if let (Some(source), Some(columns)) = (&logical_literal_source, literal_columns.as_ref()) {
        domain_sources.push((source, columns.clone()));
    }
    for (source, columns) in predicates.iter().zip(predicate_columns.iter()) {
        domain_sources.push((source, columns.clone()));
    }
    let codebook = HostBfsCodebook::from_sources(&domain_sources)?;
    reserve_logical_mapping_bytes("codebook", codebook.memory_size()?, staged_bytes, max_staging_bytes, reservation)?;

    let edge = encode_staged_source(
        edge,
        &[spec.src_col.clone(), spec.dst_col.clone()],
        &codebook,
        staged_bytes,
        max_staging_bytes,
        reservation,
    )?;
    let logical_literal_source = logical_literal_source
        .map(|source| {
            encode_staged_source(
                source,
                &["__source_vertex".to_string()],
                &codebook,
                staged_bytes,
                max_staging_bytes,
                reservation,
            )
        })
        .transpose()?;
    let predicates = predicates
        .into_iter()
        .enumerate()
        .map(|(index, source)| {
            let columns = bfs_logical_predicate_columns(params, index);
            encode_staged_source(source, &columns, &codebook, staged_bytes, max_staging_bytes, reservation)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut execution_spec = spec.clone();
    execution_spec.output_schema = bfs_numeric_execution_schema(params)?;
    Ok((edge, logical_literal_source, predicates, Some(codebook), execution_spec))
}

fn encode_staged_source(
    source: StagedCugraphSource,
    columns: &[String],
    codebook: &HostBfsCodebook,
    staged_bytes: &mut usize,
    max_staging_bytes: usize,
    reservation: &mut MemoryReservation,
) -> Result<StagedCugraphSource> {
    let StagedCugraphSource::Host { schema, batches } = source else {
        return Err(logical_bfs_native_source_error("logical_input", "native"));
    };
    let mapped_schema = remap_bfs_logical_schema(&schema, &columns.iter().map(String::as_str).collect::<Vec<_>>())?;
    let mut mapped_batches = Vec::with_capacity(batches.len());
    for batch in batches {
        let original_bytes = get_record_batch_memory_size(&batch);
        let encoded_upper_bound = original_bytes.checked_add(size_of::<RecordBatch>()).ok_or_else(|| {
            host_staging_limit_error("logical_bfs", "encoded_input", *staged_bytes, usize::MAX, max_staging_bytes)
        })?;
        let peak_bytes = staged_bytes.checked_add(encoded_upper_bound).ok_or_else(|| {
            host_staging_limit_error(
                "logical_bfs",
                "encoded_input",
                *staged_bytes,
                encoded_upper_bound,
                max_staging_bytes,
            )
        })?;
        if peak_bytes > max_staging_bytes {
            return Err(host_staging_limit_error(
                "logical_bfs",
                "encoded_input",
                *staged_bytes,
                encoded_upper_bound,
                max_staging_bytes,
            ));
        }
        reservation.try_grow(encoded_upper_bound).map_err(|source| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::MemoryReservationFailed,
                    "datafusion_nexus.cugraph_sql.host_staging_reservation",
                    "DataFusion host memory pool denied encoded logical BFS staging",
                )
                .with_fact("boundary", "encoded_input")
                .with_fact("requested_bytes", encoded_upper_bound)
                .with_fact("superseded_bytes", original_bytes)
                .with_source(source),
            )
        })?;
        let mut arrays = batch.columns().to_vec();
        for column in columns {
            let index = batch.schema().index_of(column).map_err(DataFusionError::from)?;
            arrays[index] = encode_logical_array(batch.column(index).as_ref(), codebook, column)?;
        }
        let mapped = RecordBatch::try_new(Arc::clone(&mapped_schema), arrays).map_err(DataFusionError::from)?;
        let mapped_bytes = get_record_batch_memory_size(&mapped);
        if mapped_bytes > encoded_upper_bound {
            return Err(Error::new(
                ErrorCode::CudfInvariantViolation,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS encoded batch exceeded its reserved upper bound",
            )
            .with_fact("reserved_bytes", encoded_upper_bound)
            .with_fact("actual_bytes", mapped_bytes)
            .into());
        }
        let next_bytes = staged_bytes
            .checked_sub(original_bytes)
            .and_then(|bytes| bytes.checked_add(mapped_bytes))
            .ok_or_else(|| {
                host_staging_limit_error("logical_bfs", "encoded_input", *staged_bytes, mapped_bytes, max_staging_bytes)
            })?;
        if next_bytes > max_staging_bytes {
            return Err(host_staging_limit_error(
                "logical_bfs",
                "encoded_input",
                *staged_bytes,
                mapped_bytes,
                max_staging_bytes,
            ));
        }
        // End the old batch ownership before lowering its charged staging bytes.
        drop(batch);
        reservation.shrink(original_bytes);
        reservation.shrink(encoded_upper_bound - mapped_bytes);
        *staged_bytes = next_bytes;
        mapped_batches.push(mapped);
    }
    Ok(StagedCugraphSource::Host { schema: mapped_schema, batches: mapped_batches })
}

fn reserve_logical_mapping_bytes(
    boundary: &'static str,
    requested_bytes: usize,
    staged_bytes: &mut usize,
    max_staging_bytes: usize,
    reservation: &mut MemoryReservation,
) -> Result<()> {
    let next = staged_bytes.checked_add(requested_bytes).ok_or_else(|| {
        host_staging_limit_error("logical_bfs", boundary, *staged_bytes, requested_bytes, max_staging_bytes)
    })?;
    if next > max_staging_bytes {
        return Err(host_staging_limit_error(
            "logical_bfs",
            boundary,
            *staged_bytes,
            requested_bytes,
            max_staging_bytes,
        ));
    }
    reservation.try_grow(requested_bytes).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied logical BFS mapping storage",
            )
            .with_fact("boundary", boundary)
            .with_fact("requested_bytes", requested_bytes)
            .with_fact("staged_bytes", *staged_bytes)
            .with_source(source),
        )
    })?;
    *staged_bytes = next;
    Ok(())
}

fn collect_string_values(array: &dyn Array, values: &mut BTreeSet<String>, column: &str) -> Result<()> {
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                values.insert(array.value(row).to_owned());
            }
        }
        return Ok(());
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                values.insert(array.value(row).to_owned());
            }
        }
        return Ok(());
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        for row in 0..array.len() {
            if !array.is_null(row) {
                values.insert(array.value(row).to_owned());
            }
        }
        return Ok(());
    }
    Err(Error::new(
        ErrorCode::UnsupportedType,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
        "logical BFS host mapping received a non-string vertex array",
    )
    .with_fact("column", column)
    .with_fact("data_type", format!("{:?}", array.data_type()))
    .with_fact("reason", "logical_bfs_vertex_dtype_unsupported")
    .into())
}

pub(super) fn encode_logical_array(array: &dyn Array, codebook: &HostBfsCodebook, column: &str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(None);
            continue;
        }
        let value = if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
            array.value(row)
        } else if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
            array.value(row)
        } else if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
            array.value(row)
        } else {
            return Err(Error::new(
                ErrorCode::UnsupportedType,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host mapping received a non-string vertex array",
            )
            .with_fact("column", column)
            .with_fact("data_type", format!("{:?}", array.data_type()))
            .with_fact("reason", "logical_bfs_vertex_dtype_unsupported")
            .into());
        };
        let code = codebook.code(value).ok_or_else(|| {
            Error::new(
                ErrorCode::CudfInvariantViolation,
                "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
                "logical BFS host mapping value was absent from its deterministic codebook",
            )
            .with_fact("column", column)
            .with_fact("reason", "logical_bfs_codebook_value_missing")
        })?;
        values.push(Some(code));
    }
    Ok(Arc::new(Int32Array::from(values)) as ArrayRef)
}

fn logical_bfs_native_source_error(source_kind: &'static str, source_state: impl Into<String>) -> DataFusionError {
    Error::new(
        ErrorCode::RequiredGpuOperationUnsupported,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_mapping",
        "logical BFS requires host Arrow sources before admission",
    )
    .with_fact("reason", "logical_bfs_native_source_unsupported")
    .with_fact("source_kind", source_kind)
    .with_fact("source_state", source_state.into())
    .into()
}

fn logical_bfs_unknown_code_error(code: i64) -> DataFusionError {
    Error::new(
        ErrorCode::CudfInvariantViolation,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
        "logical BFS output contained a code outside its host codebook",
    )
    .with_fact("reason", "logical_bfs_output_code_unknown")
    .with_fact("code", code)
    .into()
}

pub(super) fn prepare_bfs_logical_input_schemas(
    spec: &CugraphAlgorithmSpec,
    edge_schema: SchemaRef,
    predicate_schemas: Vec<SchemaRef>,
) -> Result<PreparedCugraphInputSchemas> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(PreparedCugraphInputSchemas {
            execution_spec: spec.clone(),
            edge_schema,
            predicate_schemas,
            logical_mapping: false,
        });
    };
    if !validate_bfs_logical_mapping_capability(spec, params, &edge_schema)? {
        return Ok(PreparedCugraphInputSchemas {
            execution_spec: spec.clone(),
            edge_schema,
            predicate_schemas,
            logical_mapping: false,
        });
    }
    validate_bfs_logical_edge_schema(spec, &edge_schema)?;
    let edge_schema = remap_bfs_logical_schema(&edge_schema, &[&spec.src_col, &spec.dst_col])?;
    let mut mapped_predicates = Vec::with_capacity(predicate_schemas.len() + usize::from(params.source.is_utf8()));
    if params.source.is_utf8() {
        mapped_predicates.push(Arc::new(Schema::new(vec![arrow::datatypes::Field::new(
            "__source_vertex",
            DataType::Int32,
            false,
        )])));
    }
    for (index, schema) in predicate_schemas.into_iter().enumerate() {
        let columns = bfs_logical_predicate_columns(params, index);
        let columns = columns.iter().map(String::as_str).collect::<Vec<_>>();
        mapped_predicates.push(remap_bfs_logical_schema(&schema, &columns)?);
    }
    let mut execution_spec = spec.clone();
    execution_spec.output_schema = bfs_numeric_execution_schema(params)?;
    Ok(PreparedCugraphInputSchemas {
        execution_spec,
        edge_schema,
        predicate_schemas: mapped_predicates,
        logical_mapping: true,
    })
}

fn validate_bfs_logical_mapping_capability(
    spec: &CugraphAlgorithmSpec,
    params: &super::CugraphBfsParams,
    edge_schema: &SchemaRef,
) -> Result<bool> {
    let logical_mapping = bfs_uses_logical_mapping(spec, params, edge_schema)?;
    super::super::validate_bfs_logical_edge_id_capability(spec, logical_mapping)?;
    Ok(logical_mapping)
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

pub(super) fn bfs_literal_string_source_batch(spec: &CugraphAlgorithmSpec) -> Result<Option<RecordBatch>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
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
    let schema = Arc::new(Schema::new(vec![arrow::datatypes::Field::new("__source_vertex", DataType::Utf8, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values)) as ArrayRef])
        .map(Some)
        .map_err(DataFusionError::from)
}

pub(super) fn is_bfs_logical_vertex_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

pub(super) fn bfs_logical_predicate_columns(params: &super::CugraphBfsParams, index: usize) -> Vec<String> {
    let source_offset = bfs_planned_source_input_offset(params);
    if index < source_offset {
        return match &params.source {
            CugraphBfsSource::Relation(input) => vec![input.column.clone()],
            _ => Vec::new(),
        };
    }
    let index = index - source_offset;
    let mut expected_index = 0;
    for input in [
        params.predicates.include_vertices.as_ref(),
        params.predicates.exclude_vertices.as_ref(),
        params.predicates.target_vertices.as_ref(),
        params.predicates.include_edge_ids.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if index == expected_index {
            return vec![input.column.clone()];
        }
        expected_index += 1;
    }
    if let Some(input) = &params.predicates.include_edges
        && index == expected_index
    {
        return vec![input.source_col.clone(), input.destination_col.clone()];
    }
    Vec::new()
}

pub(super) fn bfs_planned_source_input_offset(params: &super::CugraphBfsParams) -> usize {
    usize::from(params.source.planned_source_vertices_input())
}

pub(super) fn remap_bfs_logical_schema(schema: &SchemaRef, columns: &[&str]) -> Result<SchemaRef> {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if columns.iter().any(|column| *column == field.name()) {
                arrow::datatypes::Field::new(field.name(), DataType::Int32, field.is_nullable())
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

pub(super) fn bfs_decoded_output_schema_for_host(spec: &CugraphAlgorithmSpec) -> SchemaRef {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Arc::clone(&spec.output_schema);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Arc::clone(&spec.output_schema);
    }
    Arc::new(Schema::new(vec![
        arrow::datatypes::Field::new("vertex", DataType::Utf8, false),
        arrow::datatypes::Field::new("distance", DataType::Int64, false),
        arrow::datatypes::Field::new("predecessor", DataType::Utf8, true),
    ]))
}

pub(super) fn decode_bfs_logical_host_output(
    output: EagerCugraphHostOutput,
    codebook: &HostBfsCodebook,
    output_schema: SchemaRef,
) -> Result<EagerCugraphHostOutput> {
    let (batches, reservation) = output.into_parts();
    let old_bytes =
        batches.iter().map(get_record_batch_memory_size).try_fold(0usize, |sum, bytes| sum.checked_add(bytes));
    let old_bytes = old_bytes.ok_or_else(|| plan_error("logical BFS host output size overflowed usize"))?;
    let mut decoded = Vec::with_capacity(batches.len());
    for batch in batches.iter() {
        decoded.push(decode_bfs_logical_host_batch(batch, codebook, Arc::clone(&output_schema))?);
    }
    let new_bytes =
        decoded.iter().map(get_record_batch_memory_size).try_fold(0usize, |sum, bytes| sum.checked_add(bytes));
    let new_bytes = new_bytes.ok_or_else(|| plan_error("decoded logical BFS host output size overflowed usize"))?;
    reservation.try_grow(new_bytes).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied decoded logical BFS output staging",
            )
            .with_fact("boundary", "decoded_output")
            .with_fact("requested_bytes", new_bytes)
            .with_source(source),
        )
    })?;
    // End encoded batch ownership before lowering its charged output bytes.
    drop(batches);
    reservation.shrink(old_bytes);
    Ok(EagerCugraphHostOutput::from_shared_parts(output_schema, Arc::new(decoded), reservation))
}

pub(super) fn decode_bfs_logical_host_batch(
    batch: &RecordBatch,
    codebook: &HostBfsCodebook,
    output_schema: SchemaRef,
) -> Result<RecordBatch> {
    let mut columns = batch.columns().to_vec();
    if let Ok(index) = batch.schema().index_of("vertex") {
        columns[index] = decode_vertex_array(batch.column(index).as_ref(), codebook, "vertex")?;
    }
    if let Ok(index) = batch.schema().index_of("predecessor") {
        columns[index] = decode_predecessor_array(batch.column(index).as_ref(), codebook)?;
    }
    RecordBatch::try_new(output_schema, columns).map_err(DataFusionError::from)
}

fn decode_vertex_array(array: &dyn Array, codebook: &HostBfsCodebook, column: &'static str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(None);
        } else {
            values.push(Some(codebook.decode_code(numeric_array_value(array, row, column)?)?));
        }
    }
    Ok(Arc::new(StringArray::from(values)) as ArrayRef)
}

fn decode_predecessor_array(array: &dyn Array, codebook: &HostBfsCodebook) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(None);
            continue;
        }
        let code = numeric_array_value(array, row, "predecessor")?;
        values.push((code >= 0).then(|| codebook.decode_code(code)).transpose()?);
    }
    Ok(Arc::new(StringArray::from(values)) as ArrayRef)
}

fn numeric_array_value(array: &dyn Array, row: usize, column: &'static str) -> Result<i64> {
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(i64::from(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(array.value(row));
    }
    Err(Error::new(
        ErrorCode::UnsupportedType,
        "datafusion_nexus.cugraph_sql.logical_bfs_host_output",
        "logical BFS output vertex column was not an integral array",
    )
    .with_fact("column", column)
    .with_fact("data_type", format!("{:?}", array.data_type()))
    .into())
}
