use arrow::datatypes::Field;

use super::*;

const LOGICAL_VERTEX_COLUMN: &str = "__vertex_id_str";
const NATIVE_VERTEX_COLUMN: &str = "__vertex_id";

pub(super) struct LogicalMappingPlan {
    pub(super) execution_spec: CugraphAlgorithmSpec,
    pub(super) encoded_edge: nexus_query_engine::plan::NodeId,
    pub(super) encoded_edge_schema: SchemaRef,
    pub(super) encoded_predicates: Vec<nexus_query_engine::plan::NodeId>,
    pub(super) codebook: nexus_query_engine::plan::NodeId,
}

pub(super) fn graph_uses_logical_mapping(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<bool> {
    let source = edge_schema.field_with_name(&spec.src_col).map_err(|_| {
        plan_error(format!("cuGraph logical vertex mapping could not find edge column '{}'", spec.src_col))
    })?;
    let destination = edge_schema.field_with_name(&spec.dst_col).map_err(|_| {
        plan_error(format!("cuGraph logical vertex mapping could not find edge column '{}'", spec.dst_col))
    })?;
    Ok(vertex_id_domain_for_call(spec.algorithm, &spec.params, source.data_type(), destination.data_type())?
        .is_string_logical())
}

pub(super) fn is_logical_vertex_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

pub(super) fn add_logical_mapping_plan(
    builder: &mut QueryPlanBuilder,
    spec: &CugraphAlgorithmSpec,
    edge: nexus_query_engine::plan::NodeId,
    edge_schema: SchemaRef,
    predicate_nodes: Vec<nexus_query_engine::plan::NodeId>,
    predicate_schemas: Vec<SchemaRef>,
) -> Result<LogicalMappingPlan> {
    let mut mapped_inputs = Vec::new();
    if let Some(batch) = bfs_literal_string_source_batch(spec)? {
        let schema = batch.schema();
        let node =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
        mapped_inputs.push((node, schema, vec!["__source_vertex".to_string()]));
    }
    for (index, (node, schema)) in predicate_nodes.into_iter().zip(predicate_schemas).enumerate() {
        mapped_inputs.push((node, schema, logical_predicate_columns(spec, index)));
    }

    let mut domain_sources = vec![(edge, Arc::clone(&edge_schema), vec![spec.src_col.clone(), spec.dst_col.clone()])];
    domain_sources.extend(mapped_inputs.iter().cloned());
    let codebook_schema = Arc::new(Schema::new(vec![Field::new(LOGICAL_VERTEX_COLUMN, DataType::Utf8, false)]));
    let branches = domain_sources
        .iter()
        .flat_map(|(node, schema, columns)| {
            columns.iter().map(|column| project_logical_vertex(builder, *node, schema, column)).collect::<Vec<_>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let union = builder.add_union(branches, Arc::clone(&codebook_schema));
    let distinct = builder.add_aggregate(
        union,
        AggregateSpec::new(vec![NativeExpr::column(LOGICAL_VERTEX_COLUMN)], Vec::new()),
        Arc::clone(&codebook_schema),
    );
    let sorted = builder.add_sort(
        distinct,
        SortSpec::new(vec![SortExpr::new(NativeExpr::column(LOGICAL_VERTEX_COLUMN), false, false)]),
        Arc::clone(&codebook_schema),
    );
    let codebook_schema = Arc::new(Schema::new(vec![
        Field::new(LOGICAL_VERTEX_COLUMN, DataType::Utf8, false),
        Field::new(NATIVE_VERTEX_COLUMN, DataType::Int64, false),
    ]));
    let codebook = builder.add_window(
        sorted,
        WindowSpec::new(
            Vec::new(),
            vec![SortExpr::new(NativeExpr::column(LOGICAL_VERTEX_COLUMN), false, false)],
            vec![WindowFunction::row_number_i64(NATIVE_VERTEX_COLUMN)],
        ),
        Arc::clone(&codebook_schema),
    );

    let (encoded_edge, encoded_edge_schema) = encode_relation_columns(
        builder,
        edge,
        edge_schema,
        &[spec.src_col.clone(), spec.dst_col.clone()],
        codebook,
        false,
    )?;
    let mut encoded_predicates = Vec::with_capacity(mapped_inputs.len());
    for (node, schema, columns) in mapped_inputs {
        let (encoded, _) = encode_relation_columns(builder, node, schema, &columns, codebook, true)?;
        encoded_predicates.push(encoded);
    }

    Ok(LogicalMappingPlan {
        execution_spec: logical_execution_spec(spec)?,
        encoded_edge,
        encoded_edge_schema,
        encoded_predicates,
        codebook,
    })
}

pub(super) fn decode_logical_graph_result(
    builder: &mut QueryPlanBuilder,
    spec: &CugraphAlgorithmSpec,
    mut root: nexus_query_engine::plan::NodeId,
    mut root_schema: SchemaRef,
    codebook: nexus_query_engine::plan::NodeId,
) -> Result<(nexus_query_engine::plan::NodeId, SchemaRef)> {
    let columns: &[(&str, bool)] = match spec.algorithm {
        CugraphAlgorithm::PageRank => &[("vertex", false)],
        CugraphAlgorithm::Bfs => &[("vertex", false), ("predecessor", true)],
        _ => {
            return Err(plan_error(format!(
                "cuGraph {} does not opt in to logical string vertex mapping",
                spec.algorithm.function_name()
            )));
        },
    };
    for (column, nullable) in columns {
        (root, root_schema) = decode_relation_column(builder, root, root_schema, column, *nullable, codebook)?;
    }
    Ok((root, root_schema))
}

fn project_logical_vertex(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::plan::NodeId,
    schema: &SchemaRef,
    column: &str,
) -> Result<nexus_query_engine::plan::NodeId> {
    let field = schema
        .field_with_name(column)
        .map_err(|_| plan_error(format!("cuGraph logical vertex mapping could not find vertex column '{column}'")))?;
    let input = if field.is_nullable() {
        builder.add_filter(input, NativeExpr::is_not_null(NativeExpr::column(column)), Arc::clone(schema))
    } else {
        input
    };
    let expr = if field.data_type() == &DataType::Utf8 {
        NativeExpr::column(column)
    } else {
        NativeExpr::cast(NativeExpr::column(column), DataType::Utf8)
    };
    Ok(builder.add_projection(
        input,
        vec![ProjectionOutput::new(expr, Field::new(LOGICAL_VERTEX_COLUMN, DataType::Utf8, false))],
    ))
}

fn encode_relation_columns(
    builder: &mut QueryPlanBuilder,
    mut relation: nexus_query_engine::plan::NodeId,
    mut schema: SchemaRef,
    columns: &[String],
    codebook: nexus_query_engine::plan::NodeId,
    preserve_nulls: bool,
) -> Result<(nexus_query_engine::plan::NodeId, SchemaRef)> {
    for column in columns {
        let field = schema.field_with_name(column).map_err(|_| {
            plan_error(format!("cuGraph logical vertex mapping could not encode missing column '{column}'"))
        })?;
        let left_key = if field.data_type() == &DataType::Utf8 {
            column.clone()
        } else {
            let cast_name = format!("__logical_mapping_{column}");
            let mut outputs = schema
                .fields()
                .iter()
                .map(|field| ProjectionOutput::new(NativeExpr::column(field.name()), field.as_ref().clone()))
                .collect::<Vec<_>>();
            outputs.push(ProjectionOutput::new(
                NativeExpr::cast(NativeExpr::column(column), DataType::Utf8),
                Field::new(&cast_name, DataType::Utf8, field.is_nullable()),
            ));
            relation = builder.add_projection(relation, outputs);
            let mut fields = schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
            fields.push(Field::new(&cast_name, DataType::Utf8, field.is_nullable()));
            schema = Arc::new(Schema::new(fields));
            cast_name
        };
        let temporary_key = (left_key != *column).then_some(left_key.as_str());
        let output_schema = Arc::new(Schema::new(
            schema
                .fields()
                .iter()
                .filter(|field| Some(field.name().as_str()) != temporary_key)
                .map(|current| {
                    if current.name() == column {
                        Field::new(column, DataType::Int64, preserve_nulls)
                    } else {
                        current.as_ref().clone()
                    }
                })
                .collect::<Vec<_>>(),
        ));
        let output_projection = schema
            .fields()
            .iter()
            .filter(|current| Some(current.name().as_str()) != temporary_key)
            .map(|current| {
                if current.name() == column {
                    JoinOutputColumn::right(ColumnRef::input(1, NATIVE_VERTEX_COLUMN, NATIVE_VERTEX_COLUMN), column)
                } else {
                    JoinOutputColumn::left(ColumnRef::input(0, current.name(), current.name()), current.name())
                }
            })
            .collect::<Vec<_>>();
        let join_type = if preserve_nulls { JoinType::Left } else { JoinType::Inner };
        let spec = JoinSpec::new(
            join_type,
            vec![JoinKey::new(
                ColumnRef::input(0, &left_key, &left_key),
                ColumnRef::input(1, LOGICAL_VERTEX_COLUMN, LOGICAL_VERTEX_COLUMN),
            )],
        )
        .with_output_projection(output_projection)
        .with_build_side(JoinSide::Right);
        relation = builder.add_join(relation, codebook, spec, Arc::clone(&output_schema));
        schema = output_schema;
    }
    Ok((relation, schema))
}

fn decode_relation_column(
    builder: &mut QueryPlanBuilder,
    relation: nexus_query_engine::plan::NodeId,
    schema: SchemaRef,
    column: &str,
    nullable: bool,
    codebook: nexus_query_engine::plan::NodeId,
) -> Result<(nexus_query_engine::plan::NodeId, SchemaRef)> {
    schema.field_with_name(column).map_err(|_| {
        plan_error(format!("cuGraph logical vertex mapping could not decode missing result column '{column}'"))
    })?;
    let output_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| {
                if field.name() == column {
                    Field::new(column, DataType::Utf8, nullable)
                } else {
                    field.as_ref().clone()
                }
            })
            .collect::<Vec<_>>(),
    ));
    let output_projection = schema
        .fields()
        .iter()
        .map(|field| {
            if field.name() == column {
                JoinOutputColumn::right(ColumnRef::input(1, LOGICAL_VERTEX_COLUMN, LOGICAL_VERTEX_COLUMN), column)
            } else {
                JoinOutputColumn::left(ColumnRef::input(0, field.name(), field.name()), field.name())
            }
        })
        .collect::<Vec<_>>();
    let join_type = if nullable { JoinType::Left } else { JoinType::Inner };
    let spec = JoinSpec::new(
        join_type,
        vec![JoinKey::new(
            ColumnRef::input(0, column, column),
            ColumnRef::input(1, NATIVE_VERTEX_COLUMN, NATIVE_VERTEX_COLUMN),
        )],
    )
    .with_output_projection(output_projection)
    .with_build_side(JoinSide::Right);
    Ok((builder.add_join(relation, codebook, spec, Arc::clone(&output_schema)), output_schema))
}

fn logical_execution_spec(spec: &CugraphAlgorithmSpec) -> Result<CugraphAlgorithmSpec> {
    let mut execution_spec = spec.clone();
    execution_spec.output_schema = match &spec.params {
        CugraphAlgorithmParams::PageRank(_) => stable_result_schema(CugraphAlgorithm::PageRank),
        CugraphAlgorithmParams::Bfs(params) => bfs_numeric_execution_schema(params)?,
        _ => {
            return Err(plan_error(format!(
                "cuGraph {} does not opt in to logical string vertex mapping",
                spec.algorithm.function_name()
            )));
        },
    };
    Ok(execution_spec)
}

fn logical_predicate_columns(spec: &CugraphAlgorithmSpec, index: usize) -> Vec<String> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Vec::new();
    };
    bfs_logical_predicate_columns(params, index)
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
    let schema = Arc::new(Schema::new(vec![Field::new("__source_vertex", DataType::Utf8, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values)) as ArrayRef])
        .map(Some)
        .map_err(DataFusionError::from)
}

pub(super) fn bfs_logical_predicate_columns(params: &CugraphBfsParams, index: usize) -> Vec<String> {
    let source_offset = usize::from(params.source.planned_source_vertices_input());
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

fn bfs_numeric_execution_schema(params: &CugraphBfsParams) -> Result<SchemaRef> {
    let native_schema = bfs_native_result_schema(params.output.return_target_info);
    let fields: Vec<_> = match params.output.mode {
        CugraphBfsOutputMode::Raw | CugraphBfsOutputMode::Path => {
            native_schema.fields().iter().map(|field| field.as_ref().clone()).collect()
        },
        CugraphBfsOutputMode::Normalized => Arc::new(Schema::new(vec![
            normalized_bfs_base_field(&native_schema, "vertex", false)?,
            normalized_bfs_base_field(&native_schema, "distance", true)?,
            normalized_bfs_base_field(&native_schema, "predecessor", true)?,
            Field::new("reachable", DataType::Boolean, false),
        ]))
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .chain(native_schema.fields().iter().skip(3).map(|field| field.as_ref().clone()))
        .collect(),
    };
    Ok(Arc::new(Schema::new(fields)))
}

fn bfs_native_result_schema(return_target_info: bool) -> SchemaRef {
    let mut native_params = GraphBfsParams::new(0, None);
    native_params.return_target_info = return_target_info;
    GraphAlgorithm::Bfs.definition().result_schema_for_params(&GraphAlgorithmParams::Bfs(Box::new(native_params)))
}

fn normalized_bfs_base_field(schema: &SchemaRef, name: &'static str, nullable: bool) -> Result<Field> {
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
    Ok(Field::new(field.name(), field.data_type().clone(), nullable))
}
