use std::sync::Arc;

use arrow::{
    array::{ArrayRef, StringArray},
    datatypes::Field,
};
use query_engine::plan::NodeId;

use super::{
    AggregateSpec, ColumnRef, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBetweennessSeeds,
    CugraphBfsOutputMode, CugraphBfsParams, CugraphBfsSourceVertex, CugraphLogicalOutputMultiplicity,
    CugraphVertexSelector, DataFusionError, DataType, Error, ErrorCode, GraphAlgorithm, GraphAlgorithmParams,
    GraphBfsParams, JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, LiteralBatchSource, NativeExpr,
    PredicateSourceColumnRole, ProjectionOutput, QueryPlanBuilder, RecordBatch, Result, Schema, SchemaRef, SortExpr,
    SortSpec, SourceNode, WindowFunction, WindowSpec, plan_error, result_schema_for_params, vertex_id_domain_for_call,
};
use crate::cugraph_sql::registry::CugraphLogicalOutputColumn;

const LOGICAL_VERTEX_COLUMN: &str = "__vertex_id_str";
const NATIVE_VERTEX_COLUMN: &str = "__vertex_id";

pub(super) struct LogicalMappingPlan {
    pub(super) execution_spec: CugraphAlgorithmSpec,
    pub(super) encoded_edge: NodeId,
    pub(super) encoded_edge_schema: SchemaRef,
    pub(super) encoded_predicates: Vec<NodeId>,
    pub(super) codebook: NodeId,
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
    edge: NodeId,
    edge_schema: SchemaRef,
    predicate_nodes: Vec<NodeId>,
    predicate_schemas: Vec<SchemaRef>,
) -> Result<LogicalMappingPlan> {
    let native_id_type = spec
        .algorithm
        .definition()
        .logical_string_contract()
        .ok_or_else(|| {
            plan_error(format!(
                "cuGraph {} does not opt in to logical string vertex mapping",
                spec.algorithm.function_name()
            ))
        })?
        .native_id_type
        .data_type();
    let mut mapped_inputs = Vec::new();
    if let Some((batch, column)) = logical_literal_vertex_batch(spec)? {
        let schema = batch.schema();
        let node =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
        mapped_inputs.push((node, schema, vec![column]));
    }
    let source_specs = spec.predicate_source_specs();
    if predicate_nodes.len() != predicate_schemas.len() || predicate_nodes.len() != source_specs.len() {
        return Err(plan_error(format!(
            "cuGraph {} logical mapping side-input contract has {} nodes, {} schemas, and {} role specifications",
            spec.algorithm.function_name(),
            predicate_nodes.len(),
            predicate_schemas.len(),
            source_specs.len()
        )));
    }
    for ((node, schema), source) in predicate_nodes.into_iter().zip(predicate_schemas).zip(source_specs) {
        let columns = source
            .columns
            .into_iter()
            .filter(|column| column.role == PredicateSourceColumnRole::Vertex)
            .map(|column| column.name)
            .collect();
        mapped_inputs.push((node, schema, columns));
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
        Field::new(NATIVE_VERTEX_COLUMN, native_id_type.clone(), false),
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
        &native_id_type,
    )?;
    let mut encoded_predicates = Vec::with_capacity(mapped_inputs.len());
    for (node, schema, columns) in mapped_inputs {
        let (encoded, _) = encode_relation_columns(builder, node, schema, &columns, codebook, true, &native_id_type)?;
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
    mut root: NodeId,
    mut root_schema: SchemaRef,
    codebook: NodeId,
) -> Result<(NodeId, SchemaRef)> {
    let contract = spec.algorithm.definition().logical_string_contract().ok_or_else(|| {
        plan_error(format!(
            "cuGraph {} does not opt in to logical string vertex mapping",
            spec.algorithm.function_name()
        ))
    })?;
    let bfs_path_outputs = [
        CugraphLogicalOutputColumn {
            name: "source",
            nullable: false,
            multiplicity: CugraphLogicalOutputMultiplicity::SideInputOccurrences,
        },
        CugraphLogicalOutputColumn {
            name: "target",
            nullable: false,
            multiplicity: CugraphLogicalOutputMultiplicity::SideInputOccurrences,
        },
        CugraphLogicalOutputColumn {
            name: "vertex",
            nullable: false,
            multiplicity: CugraphLogicalOutputMultiplicity::SideInputOccurrences,
        },
    ];
    let output_columns = match &spec.params {
        CugraphAlgorithmParams::Bfs(params) if params.output.mode == CugraphBfsOutputMode::Path => {
            bfs_path_outputs.as_slice()
        },
        _ => contract.output_columns,
    };
    for column in output_columns {
        (root, root_schema) =
            decode_relation_column(builder, root, root_schema, column.name, column.nullable, codebook, true)?;
    }
    Ok((root, root_schema))
}

fn project_logical_vertex(
    builder: &mut QueryPlanBuilder,
    input: NodeId,
    schema: &SchemaRef,
    column: &str,
) -> Result<NodeId> {
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
    mut relation: NodeId,
    mut schema: SchemaRef,
    columns: &[String],
    codebook: NodeId,
    preserve_nulls: bool,
    native_id_type: &DataType,
) -> Result<(NodeId, SchemaRef)> {
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
                        Field::new(column, native_id_type.clone(), preserve_nulls)
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
        .with_build_side(JoinSide::Right)
        .with_at_most_one_match_per_bounded_row(JoinSide::Left);
        relation = builder.add_join(relation, codebook, spec, Arc::clone(&output_schema));
        schema = output_schema;
    }
    Ok((relation, schema))
}

fn decode_relation_column(
    builder: &mut QueryPlanBuilder,
    relation: NodeId,
    schema: SchemaRef,
    column: &str,
    nullable: bool,
    codebook: NodeId,
    preserves_total_payload_bound: bool,
) -> Result<(NodeId, SchemaRef)> {
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
    .with_build_side(JoinSide::Right)
    .with_at_most_one_match_per_bounded_row(JoinSide::Left);
    let spec =
        if preserves_total_payload_bound { spec.with_preserved_total_payload_bound_for_output(column) } else { spec };
    Ok((builder.add_join(relation, codebook, spec, Arc::clone(&output_schema)), output_schema))
}

fn logical_execution_spec(spec: &CugraphAlgorithmSpec) -> Result<CugraphAlgorithmSpec> {
    let mut execution_spec = spec.clone();
    execution_spec.output_schema = match &spec.params {
        CugraphAlgorithmParams::Bfs(params) => bfs_numeric_execution_schema(params)?,
        _ => result_schema_for_params(spec.algorithm, &spec.params),
    };
    Ok(execution_spec)
}

pub(super) fn logical_literal_vertex_batch(spec: &CugraphAlgorithmSpec) -> Result<Option<(RecordBatch, String)>> {
    let (column, values): (&str, Vec<&str>) = match &spec.params {
        CugraphAlgorithmParams::Bfs(params) => {
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
                    CugraphBfsSourceVertex::Int64(_) => {
                        Err(plan_error("string BFS source list contained a numeric value"))
                    },
                })
                .collect::<Result<Vec<_>>>()?;
            ("__source_vertex", values)
        },
        CugraphAlgorithmParams::Sssp(params) => match &params.source_vertex {
            CugraphVertexSelector::Utf8(value) => ("__source_vertex", vec![value.as_str()]),
            CugraphVertexSelector::Int64(_) => return Ok(None),
        },
        CugraphAlgorithmParams::BetweennessCentrality(params) => {
            let Some(seeds) = &params.seeds else { return Ok(None) };
            logical_betweenness_seed_values(seeds)?
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            let Some(seeds) = &params.seeds else { return Ok(None) };
            logical_betweenness_seed_values(seeds)?
        },
        _ => return Ok(None),
    };
    let schema = Arc::new(Schema::new(vec![Field::new(column, DataType::Utf8, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values)) as ArrayRef])
        .map(|batch| Some((batch, column.to_string())))
        .map_err(DataFusionError::from)
}

fn logical_betweenness_seed_values(seeds: &CugraphBetweennessSeeds) -> Result<(&'static str, Vec<&str>)> {
    if !seeds.vertices.first().is_some_and(CugraphVertexSelector::is_utf8) {
        return Ok(("__seed_vertex", Vec::new()));
    }
    let values = seeds
        .vertices
        .iter()
        .map(|vertex| match vertex {
            CugraphVertexSelector::Utf8(value) => Ok(value.as_str()),
            CugraphVertexSelector::Int64(_) => Err(plan_error("string betweenness seeds contained a numeric value")),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(("__seed_vertex", values))
}

fn bfs_numeric_execution_schema(params: &CugraphBfsParams) -> Result<SchemaRef> {
    let native_schema = bfs_native_result_schema(params.output.return_target_info);
    let fields: Vec<_> = match params.output.mode {
        CugraphBfsOutputMode::Raw => native_schema.fields().iter().map(|field| field.as_ref().clone()).collect(),
        CugraphBfsOutputMode::Path => vec![
            Field::new("path_index", DataType::Int64, false),
            Field::new("source", DataType::Int64, false),
            Field::new("target", DataType::Int64, false),
            Field::new("vertex", DataType::Int64, false),
            Field::new("distance", DataType::Int64, false),
        ],
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
                "algeon_datafusion.cugraph_sql.bfs_result_schema",
                "native BFS descriptor is missing a raw BFS base field",
            )
            .with_fact("column", name)
            .with_source(error),
        )
    })?;
    Ok(Field::new(field.name(), field.data_type().clone(), nullable))
}
