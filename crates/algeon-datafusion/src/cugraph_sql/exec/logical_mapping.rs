use std::sync::Arc;

use arrow::{
    array::{ArrayRef, StringArray},
    datatypes::Field,
};
use query_planning::plan::NodeId;

use super::{
    AggregateSpec, CUGRAPH_EDGE_DESTINATION_COLUMN, CUGRAPH_EDGE_SOURCE_COLUMN, ColumnRef, CugraphAlgorithmParams,
    CugraphAlgorithmSpec, CugraphBetweennessSeeds, CugraphBfsOutputMode, CugraphBfsSourceVertex,
    CugraphLogicalOutputMultiplicity, CugraphVertexSelector, DataType, JoinKey, JoinOutputColumn, JoinSide, JoinSpec,
    JoinType, LiteralBatchSource, NativeExpr, PredicateSourceColumnRole, ProjectionOutput, QueryPlanBuilder,
    RecordBatch, Result, Schema, SchemaRef, SortExpr, SortSpec, SourceNode, WindowFunction, WindowSpec,
    lowering::native_plan_error, plan, result_schema_for_params, vertex_id_domain_for_call,
};
use crate::cugraph_sql::{
    CugraphVertexIdDomain, registry::CugraphLogicalOutputColumn, types::bfs_result_schema_for_vertex_domain,
};

const LOGICAL_VERTEX_COLUMN: &str = "__vertex_id_logical";
const NATIVE_VERTEX_COLUMN: &str = "__vertex_id";

/// The per-call vertex codebook: one row per distinct vertex with its logical
/// id (the input vertex type) and its dense native id (the algorithm's native
/// vertex type).
pub(super) struct VertexCodebook {
    pub(super) node: NodeId,
    pub(super) logical_type: DataType,
}

pub(super) struct LogicalMappingPlan {
    pub(super) execution_spec: CugraphAlgorithmSpec,
    pub(super) encoded_edge: NodeId,
    pub(super) encoded_edge_schema: SchemaRef,
    pub(super) encoded_predicates: Vec<NodeId>,
    pub(super) codebook: VertexCodebook,
}

/// Whether the call's vertex domain must be mapped to native ids: string
/// vertices for every algorithm, and Int64 vertices for algorithms whose
/// native graph type is Int32.
pub(super) fn graph_uses_logical_mapping(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<bool> {
    Ok(edge_vertex_domain(spec, edge_schema)?.requires_mapping(spec.algorithm))
}

fn edge_vertex_domain(spec: &CugraphAlgorithmSpec, edge_schema: &SchemaRef) -> Result<CugraphVertexIdDomain> {
    let source = edge_schema
        .field_with_name(CUGRAPH_EDGE_SOURCE_COLUMN)
        .map_err(|_| plan("cuGraph logical vertex mapping could not find canonical edge source column 'src'"))?;
    let destination = edge_schema
        .field_with_name(CUGRAPH_EDGE_DESTINATION_COLUMN)
        .map_err(|_| plan("cuGraph logical vertex mapping could not find canonical edge destination column 'dst'"))?;
    vertex_id_domain_for_call(spec.algorithm, &spec.params, source.data_type(), destination.data_type())
}

pub(super) fn add_logical_mapping_plan(
    builder: &mut QueryPlanBuilder,
    spec: &CugraphAlgorithmSpec,
    edge: NodeId,
    edge_schema: SchemaRef,
    predicate_nodes: Vec<NodeId>,
    predicate_schemas: Vec<SchemaRef>,
) -> Result<LogicalMappingPlan> {
    let logical_type = edge_vertex_domain(spec, &edge_schema)?.output_data_type();
    let native_type = spec.algorithm.native_vertex_id_type();
    let row_number = match native_type {
        DataType::Int32 => WindowFunction::row_number_i32(NATIVE_VERTEX_COLUMN),
        DataType::Int64 => WindowFunction::row_number_i64(NATIVE_VERTEX_COLUMN),
        ref other => return Err(plan(format!("cuGraph vertex codebook cannot assign native ids of type {other:?}"))),
    };
    let mut mapped_inputs = Vec::new();
    if let Some((batch, column)) = logical_literal_vertex_batch(spec)? {
        let schema = batch.schema();
        let node =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
        mapped_inputs.push((node, schema, vec![column]));
    }
    let source_specs = spec.predicate_source_specs();
    if predicate_nodes.len() != predicate_schemas.len() || predicate_nodes.len() != source_specs.len() {
        return Err(plan(format!(
            "cuGraph {} logical mapping side-input contract has {} nodes, {} schemas, and {} role specifications",
            spec.algorithm.function_name(),
            predicate_nodes.len(),
            predicate_schemas.len(),
            source_specs.len()
        )));
    }
    for ((node, schema), source) in predicate_nodes.into_iter().zip(predicate_schemas).zip(source_specs) {
        let columns = source
            .into_iter()
            .filter(|role| {
                matches!(
                    role,
                    PredicateSourceColumnRole::Vertex
                        | PredicateSourceColumnRole::Source
                        | PredicateSourceColumnRole::Destination
                        | PredicateSourceColumnRole::FirstVertex
                        | PredicateSourceColumnRole::SecondVertex
                )
            })
            .map(PredicateSourceColumnRole::canonical_name)
            .map(str::to_string)
            .collect();
        mapped_inputs.push((node, schema, columns));
    }

    let mut domain_sources = vec![(
        edge,
        Arc::clone(&edge_schema),
        vec![CUGRAPH_EDGE_SOURCE_COLUMN.to_string(), CUGRAPH_EDGE_DESTINATION_COLUMN.to_string()],
    )];
    domain_sources.extend(mapped_inputs.iter().cloned());
    let codebook_schema = Arc::new(Schema::new(vec![Field::new(LOGICAL_VERTEX_COLUMN, logical_type.clone(), false)]));
    let branches = domain_sources
        .iter()
        .flat_map(|(node, schema, columns)| {
            columns
                .iter()
                .map(|column| project_logical_vertex(builder, *node, schema, column, &logical_type))
                .collect::<Vec<_>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let union = builder.add_union(branches, Arc::clone(&codebook_schema)).map_err(native_plan_error)?;
    let distinct = builder
        .add_aggregate(
            union,
            AggregateSpec::new(vec![NativeExpr::column(LOGICAL_VERTEX_COLUMN)], Vec::new()),
            Arc::clone(&codebook_schema),
        )
        .map_err(native_plan_error)?;
    let sorted = builder
        .add_sort(distinct, SortSpec::new(vec![SortExpr::new(NativeExpr::column(LOGICAL_VERTEX_COLUMN), false, false)]))
        .map_err(native_plan_error)?;
    let codebook_node = builder
        .add_window(
            sorted,
            WindowSpec::new(
                Vec::new(),
                vec![SortExpr::new(NativeExpr::column(LOGICAL_VERTEX_COLUMN), false, false)],
                vec![row_number],
            ),
        )
        .map_err(native_plan_error)?;
    let codebook = VertexCodebook { node: codebook_node, logical_type };

    let (encoded_edge, encoded_edge_schema) = encode_relation_columns(
        builder,
        edge,
        edge_schema,
        &[CUGRAPH_EDGE_SOURCE_COLUMN.to_string(), CUGRAPH_EDGE_DESTINATION_COLUMN.to_string()],
        &codebook,
        &native_type,
        false,
    )?;
    let mut encoded_predicates = Vec::with_capacity(mapped_inputs.len());
    for (node, schema, columns) in mapped_inputs {
        let (encoded, _) = encode_relation_columns(builder, node, schema, &columns, &codebook, &native_type, true)?;
        encoded_predicates.push(encoded);
    }

    Ok(LogicalMappingPlan {
        execution_spec: logical_execution_spec(spec),
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
    codebook: &VertexCodebook,
) -> Result<(NodeId, SchemaRef)> {
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
        _ => spec.algorithm.definition().vertex_output_columns(),
    };
    for column in output_columns {
        (root, root_schema) =
            decode_relation_column(builder, root, root_schema, column.name, column.nullable, codebook)?;
    }
    Ok((root, root_schema))
}

fn project_logical_vertex(
    builder: &mut QueryPlanBuilder,
    input: NodeId,
    schema: &SchemaRef,
    column: &str,
    logical_type: &DataType,
) -> Result<NodeId> {
    let field = schema
        .field_with_name(column)
        .map_err(|_| plan(format!("cuGraph logical vertex mapping could not find vertex column '{column}'")))?;
    let input = if field.is_nullable() {
        builder.add_filter(input, NativeExpr::is_not_null(NativeExpr::column(column))).map_err(native_plan_error)?
    } else {
        input
    };
    let expr = if field.data_type() == logical_type {
        NativeExpr::column(column)
    } else {
        NativeExpr::cast(NativeExpr::column(column), logical_type.clone())
    };
    builder
        .add_projection(
            input,
            vec![ProjectionOutput::new(expr, Field::new(LOGICAL_VERTEX_COLUMN, logical_type.clone(), false))],
        )
        .map_err(native_plan_error)
}

fn encode_relation_columns(
    builder: &mut QueryPlanBuilder,
    mut relation: NodeId,
    mut schema: SchemaRef,
    columns: &[String],
    codebook: &VertexCodebook,
    native_type: &DataType,
    preserve_nulls: bool,
) -> Result<(NodeId, SchemaRef)> {
    for column in columns {
        let field = schema
            .field_with_name(column)
            .map_err(|_| plan(format!("cuGraph logical vertex mapping could not encode missing column '{column}'")))?;
        let left_key = if field.data_type() == &codebook.logical_type {
            column.clone()
        } else {
            let cast_name = format!("__logical_mapping_{column}");
            let mut outputs = schema
                .fields()
                .iter()
                .map(|field| ProjectionOutput::new(NativeExpr::column(field.name()), field.as_ref().clone()))
                .collect::<Vec<_>>();
            outputs.push(ProjectionOutput::new(
                NativeExpr::cast(NativeExpr::column(column), codebook.logical_type.clone()),
                Field::new(&cast_name, codebook.logical_type.clone(), field.is_nullable()),
            ));
            relation = builder.add_projection(relation, outputs).map_err(native_plan_error)?;
            let mut fields = schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
            fields.push(Field::new(&cast_name, codebook.logical_type.clone(), field.is_nullable()));
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
                        Field::new(column, native_type.clone(), preserve_nulls)
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
        relation =
            builder.add_join(relation, codebook.node, spec, Arc::clone(&output_schema)).map_err(native_plan_error)?;
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
    codebook: &VertexCodebook,
) -> Result<(NodeId, SchemaRef)> {
    schema.field_with_name(column).map_err(|_| {
        plan(format!("cuGraph logical vertex mapping could not decode missing result column '{column}'"))
    })?;
    let output_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| {
                if field.name() == column {
                    Field::new(column, codebook.logical_type.clone(), nullable)
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
    Ok((
        builder.add_join(relation, codebook.node, spec, Arc::clone(&output_schema)).map_err(native_plan_error)?,
        output_schema,
    ))
}

fn logical_execution_spec(spec: &CugraphAlgorithmSpec) -> CugraphAlgorithmSpec {
    let mut execution_spec = spec.clone();
    execution_spec.output_schema = match &spec.params {
        CugraphAlgorithmParams::Bfs(params) => {
            bfs_result_schema_for_vertex_domain(params, &CugraphVertexIdDomain::Int64)
        },
        _ => result_schema_for_params(spec.algorithm, &spec.params),
    };
    execution_spec
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
                    CugraphBfsSourceVertex::Int64(_) => Err(plan("string BFS source list contained a numeric value")),
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
        .map_err(|error| plan(format!("failed to build the literal vertex batch for '{column}'")).with_source(error))
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
            CugraphVertexSelector::Int64(_) => Err(plan("string betweenness seeds contained a numeric value")),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(("__seed_vertex", values))
}
