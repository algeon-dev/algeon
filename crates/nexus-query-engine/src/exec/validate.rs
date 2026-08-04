use std::collections::BTreeSet;

use arrow::datatypes::{Field, SchemaRef};

use super::{source::local_parquet_execution_frame_schema, *};
use crate::{
    datatypes::{arrow_fields_match_except_name, is_supported_arrow_type},
    plan::{QueryNode, UnionNode},
    source::validate_literal_batch_source,
};

mod expressions;
mod join;
mod types;

use expressions::{comma_separated, validate_expr, validate_projection_output_expr};
use join::validate_join_capability;
use types::{require_boolean, require_scalar_operator_value, require_timezone_free_operator_key};

pub(super) fn validate_query_plan(plan: &QueryPlan) -> Result<()> {
    validate_reachability(plan)?;
    let mut available_columns_by_node = Vec::with_capacity(plan.node_count());
    let vector_ingress_columns = crate::vector_validation::vector_ingress_columns_by_input(plan);

    for (index, node) in plan.nodes().iter().enumerate() {
        if node.id.index() != index {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.plan.validate",
                "query plan node id must match its position",
            )
            .with_fact("node", node.id)
            .with_fact("position", index));
        }

        crate::vector_validation::validate_schema_supported_with_vector_ingress(
            node.output_schema(),
            "nexus_query_engine.plan.validate_schema",
            vector_ingress_columns.get(&node.id),
        )?;
        validate_output_names_unique(node)?;
        let available_columns = match &node.kind {
            PlanNode::Source(source) => validate_source_node(source)?,
            PlanNode::Filter(filter) => {
                let input = input_node(plan, filter.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, filter.input)?;
                let predicate_type = validate_expr(&filter.predicate, input.output_schema(), input_available)?;
                require_boolean(&predicate_type, "filter predicate")?;
                lower_filter_predicate_expr(&filter.predicate)?;
                filter_available_columns(input_available, node.output_schema())
            },
            PlanNode::Projection(projection) => {
                let input = input_node(plan, projection.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, projection.input)?;
                for output in &projection.outputs {
                    if vector_ingress_columns.get(&node.id).is_some_and(|columns| columns.contains(output.field.name()))
                    {
                        crate::vector_validation::validate_vector_ingress_projection_output(
                            output,
                            input.output_schema(),
                            input_available,
                            vector_ingress_columns.get(&projection.input),
                        )?;
                    } else {
                        validate_projection_output_expr(output, input.output_schema(), input_available)?;
                    }
                }
                validate_projection_capability(projection, &node.facts.output_schema)?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Aggregate(aggregate) => {
                let input = input_node(plan, aggregate.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, aggregate.input)?;
                for expr in &aggregate.spec.group_exprs {
                    let data_type = validate_expr(expr, input.output_schema(), input_available)?;
                    require_timezone_free_operator_key(&data_type, "aggregate group key")?;
                }
                for expr in &aggregate.spec.aggr_exprs {
                    let data_type = validate_expr(&expr.expr, input.output_schema(), input_available)?;
                    require_timezone_free_operator_key(&data_type, "aggregate input")?;
                    if let Some(filter) = &expr.filter {
                        let filter_type = validate_expr(filter, input.output_schema(), input_available)?;
                        require_boolean(&filter_type, "aggregate filter predicate")?;
                        lower_expr(filter)?;
                    }
                }
                validate_aggregate_capability(aggregate, node.output_schema())?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Window(window) => {
                let input = input_node(plan, window.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, window.input)?;
                for expr in &window.spec.partition_by {
                    let data_type = validate_expr(expr, input.output_schema(), input_available)?;
                    require_timezone_free_operator_key(&data_type, "window partition key")?;
                }
                for sort_expr in &window.spec.order_by {
                    let data_type = validate_expr(&sort_expr.expr, input.output_schema(), input_available)?;
                    require_timezone_free_operator_key(&data_type, "window order key")?;
                }
                for function in &window.spec.functions {
                    match function {
                        crate::plan::WindowFunction::Aggregate { expr, .. } => {
                            let data_type = validate_expr(expr, input.output_schema(), input_available)?;
                            require_scalar_operator_value(&data_type, "window aggregate input")?;
                        },
                        crate::plan::WindowFunction::Rank { .. } | crate::plan::WindowFunction::RowNumber { .. } => {},
                        crate::plan::WindowFunction::LagLead { expr, .. } => {
                            let data_type = validate_expr(expr, input.output_schema(), input_available)?;
                            require_scalar_operator_value(&data_type, "window LAG / LEAD input")?;
                        },
                    }
                }
                validate_window_capability(window, input.output_schema(), node.output_schema())?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Sort(sort) => {
                let input = input_node(plan, sort.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, sort.input)?;
                for sort_expr in &sort.spec.expressions {
                    let data_type = validate_expr(&sort_expr.expr, input.output_schema(), input_available)?;
                    require_timezone_free_operator_key(&data_type, "sort key")?;
                }
                lower_sort_spec(&sort.spec)?;
                filter_available_columns(input_available, node.output_schema())
            },
            PlanNode::Limit(limit) => {
                let input = input_node(plan, limit.input, node.id)?;
                if limit.spec.fetch.is_some() {
                    topk_candidate_rows(limit.spec)?;
                }
                let input_available = available_columns(&available_columns_by_node, limit.input)?;
                filter_available_columns(input_available, input.output_schema())
            },
            PlanNode::Join(join) => {
                let left = input_node(plan, join.inputs[0], node.id)?;
                let right = input_node(plan, join.inputs[1], node.id)?;
                let left_available = available_columns(&available_columns_by_node, join.inputs[0])?;
                let right_available = available_columns(&available_columns_by_node, join.inputs[1])?;
                validate_join_capability(
                    join,
                    left.output_schema(),
                    left_available,
                    right.output_schema(),
                    right_available,
                    node.output_schema(),
                )?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Union(union) => {
                validate_union_node(plan, union, node.id, node.output_schema(), &available_columns_by_node)?;
                schema_column_names(node.output_schema())
            },
            PlanNode::EdgeNormalize(edge_normalize) => {
                let input = input_node(plan, edge_normalize.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, edge_normalize.input)?;
                validate_edge_normalize_node(
                    edge_normalize,
                    input.output_schema(),
                    input_available,
                    node.output_schema(),
                )?;
                schema_column_names(node.output_schema())
            },
            PlanNode::GraphAlgorithm(graph) => {
                let input = input_node(plan, graph.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, graph.input)?;
                validate_graph_algorithm_node(graph, input.output_schema(), input_available, node.output_schema())?;
                validate_graph_algorithm_side_inputs(graph, plan, &available_columns_by_node, node.id)?;
                schema_column_names(node.output_schema())
            },
            PlanNode::VectorAlgorithm(vector) => {
                crate::vector_validation::validate_vector_algorithm_node(
                    vector,
                    plan,
                    &available_columns_by_node,
                    node.id,
                )?;
                schema_column_names(node.output_schema())
            },
        };
        available_columns_by_node.push(available_columns);
    }

    let sink_input = input_node(plan, plan.sink().input(), NodeId::from_index_for_internal_use(plan.node_count()))?;
    let sink_available = available_columns(&available_columns_by_node, plan.sink().input())?;
    if let ResultSink::HostArrow { schema, .. } = plan.sink() {
        validate_schema_supported(schema, "nexus_query_engine.plan.validate_sink_schema")?;
        if schema.fields().len() != sink_input.output_schema().fields().len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.plan.validate_sink",
                "host sink schema field count must match the root node output",
            )
            .with_fact("root_node", sink_input.id)
            .with_fact("sink_fields", schema.fields().len())
            .with_fact("root_fields", sink_input.output_schema().fields().len()));
        }
        for (index, (sink_field, root_field)) in
            schema.fields().iter().zip(sink_input.output_schema().fields().iter()).enumerate()
        {
            if !sink_available.contains(root_field.name()) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.plan.validate_sink",
                    "host sink root schema references a column not produced by the root node",
                )
                .with_fact("column", root_field.name())
                .with_fact("root_node", sink_input.id)
                .with_fact("available_columns", comma_separated(sink_available)));
            }
            if !fields_match_except_name(sink_field.as_ref(), root_field.as_ref()) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.plan.validate_sink",
                    "host sink schema field must match the root node output by ordinal except for field name",
                )
                .with_fact("root_node", sink_input.id)
                .with_fact("field_index", index)
                .with_fact("sink_field", sink_field.name())
                .with_fact("root_field", root_field.name())
                .with_fact("sink_type", format!("{:?}", sink_field.data_type()))
                .with_fact("root_type", format!("{:?}", root_field.data_type()))
                .with_fact("sink_nullable", sink_field.is_nullable())
                .with_fact("root_nullable", root_field.is_nullable())
                .with_fact("sink_metadata", format!("{:?}", sink_field.metadata()))
                .with_fact("root_metadata", format!("{:?}", root_field.metadata())));
            }
        }
    }

    Ok(())
}

fn fields_match_except_name(sink_field: &Field, root_field: &Field) -> bool {
    arrow_fields_match_except_name(sink_field, root_field)
}

fn validate_reachability(plan: &QueryPlan) -> Result<()> {
    let mut reachable = vec![false; plan.node_count()];
    let mut stack = vec![plan.sink().input()];
    while let Some(id) = stack.pop() {
        let Some(node) = plan.node(id) else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.plan.validate_reachability",
                "query plan sink reaches a missing node",
            )
            .with_fact("node", id));
        };
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        stack.extend(node.input_ids().iter().copied());
    }

    if let Some((index, _)) = reachable.iter().enumerate().find(|(_, reachable)| !**reachable) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.plan.validate_reachability",
            "query plan contains an unreachable node",
        )
        .with_fact("node", NodeId::from_index_for_internal_use(index))
        .with_fact("sink_input", plan.sink().input()));
    }
    Ok(())
}

fn validate_source_node(source: &crate::source::SourceNode) -> Result<BTreeSet<String>> {
    match source {
        crate::source::SourceNode::LocalParquet(source) => {
            validate_local_parquet_source(source)?;
            Ok(schema_column_names(&local_parquet_execution_frame_schema(source)?))
        },
        crate::source::SourceNode::GpuResident(source) => {
            if source.input_name.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu_resident_source.validate",
                    "GPU-resident source input name must not be empty",
                )
                .with_fact("reason", "empty_input_name"));
            }
            Ok(schema_column_names(&source.schema))
        },
        crate::source::SourceNode::LiteralBatch(source) => {
            validate_literal_batch_source(source)?;
            Ok(schema_column_names(&source.schema))
        },
    }
}

fn input_node(plan: &QueryPlan, input: NodeId, consumer: NodeId) -> Result<&QueryNode> {
    plan.node(input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.plan.validate",
            "query plan node references a missing input",
        )
        .with_fact("consumer", consumer)
        .with_fact("input", input)
    })
}

fn available_columns(columns_by_node: &[BTreeSet<String>], input: NodeId) -> Result<&BTreeSet<String>> {
    columns_by_node.get(input.index()).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.plan.validate",
            "query plan node references an input that has not been validated",
        )
        .with_fact("input", input)
    })
}

fn validate_schema_supported(schema: &SchemaRef, operation: &'static str) -> Result<()> {
    for field in schema.fields() {
        if !is_supported_arrow_type(field.data_type()) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                operation,
                "schema field type is not supported",
            )
            .with_fact("field", field.name())
            .with_fact("data_type", format!("{:?}", field.data_type())));
        }
    }
    Ok(())
}

fn validate_output_names_unique(node: &QueryNode) -> Result<()> {
    let output_names = node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    for (duplicate_output_index, name) in output_names.iter().enumerate() {
        if let Some(first_output_index) = output_names[..duplicate_output_index].iter().position(|seen| seen == name) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.plan.validate_output_names",
                "query plan node output schema contains duplicate executable column names",
            )
            .with_fact("node_id", node.id)
            .with_fact("node_kind", plan_node_kind(&node.kind))
            .with_fact("schema_behavior", node.facts.schema_behavior)
            .with_fact("duplicate_output_name", *name)
            .with_fact("first_output_index", first_output_index)
            .with_fact("duplicate_output_index", duplicate_output_index)
            .with_fact("output_names", output_names.join(",")));
        }
    }
    Ok(())
}

fn plan_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Window(_) => "window",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
        PlanNode::Union(_) => "union",
        PlanNode::EdgeNormalize(_) => "edge_normalize",
        PlanNode::GraphAlgorithm(_) => "graph_algorithm",
        PlanNode::VectorAlgorithm(_) => "vector_algorithm",
    }
}

fn validate_edge_normalize_node(
    edge_normalize: &crate::plan::EdgeNormalizeNode,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
    output_schema: &SchemaRef,
) -> Result<()> {
    validate_graph_vertex_column("source", &edge_normalize.spec.edge_columns.source, input_schema, input_available)?;
    validate_graph_vertex_column(
        "destination",
        &edge_normalize.spec.edge_columns.destination,
        input_schema,
        input_available,
    )?;
    if let Some(weight_col) = edge_normalize.spec.edge_columns.weight.as_deref() {
        validate_graph_weight_column(weight_col, input_schema, input_available)?;
    }
    if let Some(edge_id_col) = edge_normalize.spec.edge_columns.edge_id.as_deref() {
        validate_graph_edge_id_column(edge_id_col, input_schema, input_available)?;
    }
    let expected = [
        Some(edge_normalize.spec.edge_columns.source.as_str()),
        Some(edge_normalize.spec.edge_columns.destination.as_str()),
        edge_normalize.spec.edge_columns.weight.as_deref(),
        edge_normalize.spec.edge_columns.edge_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let actual = output_schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    if actual != expected {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.edge_normalize.validate",
            "edge normalize output schema must contain source, destination, optional weight, and optional edge-id columns in order",
        )
        .with_fact("expected_columns", expected.join(","))
        .with_fact("actual_columns", actual.join(",")));
    }
    Ok(())
}

fn validate_graph_algorithm_node(
    graph: &crate::plan::GraphAlgorithmNode,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
    output_schema: &SchemaRef,
) -> Result<()> {
    if let Some(required) = graph.spec.algorithm.definition().required_graph_directedness()
        && graph.spec.graph_options.directed != required
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.validate",
            "graph algorithm directed option is incompatible with the cuGraph algorithm",
        )
        .with_fact("reason", "unsupported_graph_directedness")
        .with_fact("algorithm", graph.spec.algorithm.as_str())
        .with_fact("directed", graph.spec.graph_options.directed)
        .with_fact("required_directed", required));
    }
    if output_schema.as_ref()
        != crate::plan::graph_result_schema_for_params(graph.spec.algorithm, &graph.spec.params).as_ref()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.validate",
            "graph algorithm output schema must match the algorithm result schema",
        )
        .with_fact("algorithm", graph.spec.algorithm.as_str())
        .with_fact("output_fields", output_schema.fields().len()));
    }
    validate_graph_vertex_column("source", &graph.spec.edge_columns.source, input_schema, input_available)?;
    validate_graph_vertex_column("destination", &graph.spec.edge_columns.destination, input_schema, input_available)?;
    let source_field = graph_edge_field("source", &graph.spec.edge_columns.source, input_schema, input_available)?;
    validate_graph_algorithm_vertex_id_support(
        graph.spec.algorithm,
        "source",
        &graph.spec.edge_columns.source,
        source_field.data_type(),
    )?;
    let destination_field =
        graph_edge_field("destination", &graph.spec.edge_columns.destination, input_schema, input_available)?;
    validate_graph_algorithm_vertex_id_support(
        graph.spec.algorithm,
        "destination",
        &graph.spec.edge_columns.destination,
        destination_field.data_type(),
    )?;
    let vertex_type = source_field.data_type().clone();
    if let Some(weight_col) = graph.spec.edge_columns.weight.as_deref() {
        validate_graph_weight_column(weight_col, input_schema, input_available)?;
    }
    if let Some(edge_id_col) = graph.spec.edge_columns.edge_id.as_deref() {
        validate_graph_edge_id_column(edge_id_col, input_schema, input_available)?;
        let edge_id_type = graph_edge_field("edge_id", edge_id_col, input_schema, input_available)?.data_type().clone();
        if edge_id_type != vertex_type {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_algorithm.validate",
                "graph edge-id column type must match the graph vertex dtype",
            )
            .with_fact("reason", "edge_id_dtype_mismatch")
            .with_fact("column", edge_id_col)
            .with_fact("edge_id_type", format!("{edge_id_type:?}"))
            .with_fact("vertex_type", format!("{vertex_type:?}")));
        }
    }
    Ok(())
}

fn validate_graph_algorithm_side_inputs(
    graph: &crate::plan::GraphAlgorithmNode,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    if graph.inputs.first().copied() != Some(graph.input) {
        return Err(graph_algorithm_error("graph algorithm input list must begin with the edge-list input")
            .with_fact("node_id", node_id)
            .with_fact("input", graph.input)
            .with_fact("first_input", graph.inputs.first().map_or_else(|| "none".to_string(), ToString::to_string)));
    }
    let edge_input = input_node(plan, graph.input, node_id)?;
    let edge_available = available_columns(available_columns_by_node, graph.input)?;
    let vertex_type =
        graph_edge_field("source", &graph.spec.edge_columns.source, edge_input.output_schema(), edge_available)?
            .data_type()
            .clone();
    let expected_side_inputs = graph.spec.params.side_input_node_ids().collect::<Vec<_>>();
    if graph.inputs[1..] != expected_side_inputs[..] {
        return Err(graph_algorithm_error("graph algorithm side-input list must match parameter input bindings")
            .with_fact("node_id", node_id)
            .with_fact("algorithm", graph.spec.algorithm.as_str())
            .with_fact(
                "actual_side_inputs",
                graph.inputs[1..].iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
            )
            .with_fact(
                "expected_side_inputs",
                expected_side_inputs.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
            ));
    }
    match &graph.spec.params {
        crate::plan::GraphAlgorithmParams::Bfs(params) => {
            validate_graph_predicate_input(
                params.source_vertices.as_ref(),
                "source_vertices",
                &vertex_type,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_graph_predicate_input(
                params.predicates.include_vertices.as_ref(),
                "include_vertices",
                &vertex_type,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_graph_predicate_input(
                params.predicates.exclude_vertices.as_ref(),
                "exclude_vertices",
                &vertex_type,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_graph_predicate_input(
                params.predicates.target_vertices.as_ref(),
                "target_vertices",
                &vertex_type,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_graph_predicate_input(
                params.predicates.include_edge_ids.as_ref(),
                "include_edge_ids",
                &vertex_type,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            if params.predicates.include_edge_ids.is_some() && graph.spec.edge_columns.edge_id.is_none() {
                return Err(graph_algorithm_error("BFS include_edge_ids requires a primary edge-id column binding")
                    .with_fact("reason", "missing_edge_id_binding")
                    .with_fact("predicate_role", "include_edge_ids")
                    .with_fact("missing_binding", "edge_id_col"));
            }
            if let Some(input) = &params.predicates.include_edges {
                validate_graph_edge_predicate_input(input, &vertex_type, plan, available_columns_by_node, node_id)?;
            }
            Ok(())
        },
        crate::plan::GraphAlgorithmParams::PersonalizedPageRank(params) => validate_graph_personalization_input(
            &params.personalization,
            &vertex_type,
            plan,
            available_columns_by_node,
            node_id,
        ),
        crate::plan::GraphAlgorithmParams::Similarity(params) => validate_graph_vertex_pair_input(
            &params.vertex_pairs,
            &vertex_type,
            plan,
            available_columns_by_node,
            node_id,
        ),
        _ => Ok(()),
    }
}

fn validate_graph_vertex_pair_input(
    input: &crate::plan::GraphVertexPairInput,
    vertex_type: &arrow::datatypes::DataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let input_node = input_node(plan, input.input, node_id)?;
    let input_available = available_columns(available_columns_by_node, input.input)?;
    let first = graph_vertex_pair_field("first", &input.first_column, input_node.output_schema(), input_available)?;
    let second = graph_vertex_pair_field("second", &input.second_column, input_node.output_schema(), input_available)?;
    if first.data_type() != second.data_type() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "vertex-pair columns must have matching physical types",
        )
        .with_fact("reason", "vertex_pair_dtype_mismatch")
        .with_fact("first_column", &input.first_column)
        .with_fact("first_type", format!("{:?}", first.data_type()))
        .with_fact("second_column", &input.second_column)
        .with_fact("second_type", format!("{:?}", second.data_type())));
    }
    if !matches!(first.data_type(), arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::Int64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "vertex-pair column type is unsupported",
        )
        .with_fact("reason", "unsupported_vertex_pair_type")
        .with_fact("pair_column_role", "first")
        .with_fact("column", &input.first_column)
        .with_fact("data_type", format!("{:?}", first.data_type())));
    }
    if first.data_type() != vertex_type {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "vertex-pair columns must match the graph vertex domain",
        )
        .with_fact("reason", "vertex_pair_graph_vertex_dtype_mismatch")
        .with_fact("first_column", &input.first_column)
        .with_fact("pair_type", format!("{:?}", first.data_type()))
        .with_fact("vertex_type", format!("{vertex_type:?}")));
    }
    Ok(())
}

fn validate_graph_personalization_input(
    input: &crate::plan::GraphPageRankPersonalizationInput,
    vertex_type: &arrow::datatypes::DataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let input_node = input_node(plan, input.input, node_id)?;
    let input_available = available_columns(available_columns_by_node, input.input)?;
    validate_graph_predicate_column(
        "personalization_vertex",
        &input.vertex,
        input_node.output_schema(),
        input_available,
        vertex_type,
    )?;
    let field = graph_edge_field("personalization_value", &input.value, input_node.output_schema(), input_available)?;
    if !matches!(field.data_type(), arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "personalized PageRank value column type is unsupported",
        )
        .with_fact("reason", "unsupported_personalization_value_type")
        .with_fact("column", &input.value)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(())
}

fn validate_graph_predicate_input(
    input: Option<&crate::plan::GraphPredicateInput>,
    role: &'static str,
    expected_type: &arrow::datatypes::DataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let Some(input) = input else {
        return Ok(());
    };
    let input_node = input_node(plan, input.input, node_id)?;
    let input_available = available_columns(available_columns_by_node, input.input)?;
    validate_graph_predicate_column(role, &input.column, input_node.output_schema(), input_available, expected_type)
}

fn validate_graph_edge_predicate_input(
    input: &crate::plan::GraphEdgePredicateInput,
    expected_type: &arrow::datatypes::DataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let input_node = input_node(plan, input.input, node_id)?;
    let input_available = available_columns(available_columns_by_node, input.input)?;
    validate_graph_predicate_column(
        "include_edges_source",
        &input.source,
        input_node.output_schema(),
        input_available,
        expected_type,
    )?;
    validate_graph_predicate_column(
        "include_edges_destination",
        &input.destination,
        input_node.output_schema(),
        input_available,
        expected_type,
    )?;
    if let Some(edge_id) = &input.edge_id {
        validate_graph_predicate_column(
            "include_edges_edge_id",
            edge_id,
            input_node.output_schema(),
            input_available,
            expected_type,
        )?;
    }
    Ok(())
}

fn validate_graph_predicate_column(
    role: &'static str,
    column: &str,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
    expected_type: &arrow::datatypes::DataType,
) -> Result<()> {
    let field = graph_edge_field(role, column, input_schema, input_available)?;
    if field.data_type() != expected_type {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "graph predicate id column type must match the graph vertex dtype",
        )
        .with_fact("reason", "predicate_dtype_mismatch")
        .with_fact("predicate_role", role)
        .with_fact("column", column)
        .with_fact("predicate_type", format!("{:?}", field.data_type()))
        .with_fact("vertex_type", format!("{expected_type:?}")));
    }
    Ok(())
}

fn validate_graph_vertex_column(
    role: &'static str,
    column: &str,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
) -> Result<()> {
    let field = graph_edge_field(role, column, input_schema, input_available)?;
    if !matches!(field.data_type(), arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::Int64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "graph edge vertex column type is unsupported",
        )
        .with_fact("reason", "unsupported_vertex_type")
        .with_fact("edge_column_role", role)
        .with_fact("column", column)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(())
}

fn validate_graph_algorithm_vertex_id_support(
    algorithm: crate::plan::GraphAlgorithm,
    role: &'static str,
    column: &str,
    data_type: &arrow::datatypes::DataType,
) -> Result<()> {
    let support = algorithm.vertex_id_type_support();
    if support.accepts_arrow_data_type(data_type) {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "nexus_query_engine.graph_algorithm.validate",
        "graph algorithm vertex column type is unsupported by the selected algorithm",
    )
    .with_fact("reason", "unsupported_vertex_dtype")
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("edge_column_role", role)
    .with_fact("column", column)
    .with_fact("data_type", format!("{data_type:?}"))
    .with_fact("vertex_id_type_support", support.as_str())
    .with_fact("supported_vertex_dtypes", support.supported_dtypes_label()))
}

fn validate_graph_weight_column(
    column: &str,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
) -> Result<()> {
    let field = graph_edge_field("weight", column, input_schema, input_available)?;
    if !matches!(field.data_type(), arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "graph edge weight column type is unsupported",
        )
        .with_fact("reason", "unsupported_weight_type")
        .with_fact("edge_column_role", "weight")
        .with_fact("column", column)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(())
}

fn validate_graph_edge_id_column(
    column: &str,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
) -> Result<()> {
    let field = graph_edge_field("edge_id", column, input_schema, input_available)?;
    if !matches!(field.data_type(), arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::Int64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate",
            "graph edge-id column type is unsupported",
        )
        .with_fact("reason", "unsupported_edge_id_type")
        .with_fact("edge_column_role", "edge_id")
        .with_fact("column", column)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(())
}

fn graph_edge_field<'a>(
    role: &'static str,
    column: &str,
    input_schema: &'a SchemaRef,
    input_available: &BTreeSet<String>,
) -> Result<&'a Field> {
    if !input_available.contains(column) {
        return Err(graph_algorithm_error("graph edge column is not produced by the input node")
            .with_fact("reason", "missing_edge_column")
            .with_fact("edge_column_role", role)
            .with_fact("column", column)
            .with_fact("available_columns", comma_separated(input_available)));
    }
    input_schema.field_with_name(column).map_err(|_| {
        graph_algorithm_error("graph edge column is not present in the input schema")
            .with_fact("reason", "missing_edge_column")
            .with_fact("edge_column_role", role)
            .with_fact("column", column)
    })
}

fn graph_vertex_pair_field<'a>(
    role: &'static str,
    column: &str,
    input_schema: &'a SchemaRef,
    input_available: &BTreeSet<String>,
) -> Result<&'a Field> {
    if !input_available.contains(column) {
        return Err(graph_algorithm_error("vertex-pair column is not produced by the input node")
            .with_fact("reason", "missing_vertex_pair_column")
            .with_fact("pair_column_role", role)
            .with_fact("column", column)
            .with_fact("available_columns", comma_separated(input_available)));
    }
    input_schema.field_with_name(column).map_err(|_| {
        graph_algorithm_error("vertex-pair column is not present in the input schema")
            .with_fact("reason", "missing_vertex_pair_column")
            .with_fact("pair_column_role", role)
            .with_fact("column", column)
    })
}

fn graph_algorithm_error(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "nexus_query_engine.graph_algorithm.validate", message)
}

fn validate_union_node(
    plan: &QueryPlan,
    union: &UnionNode,
    node_id: NodeId,
    output_schema: &SchemaRef,
    available_columns_by_node: &[BTreeSet<String>],
) -> Result<()> {
    if union.inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.union.validate",
            "union node requires at least one input",
        )
        .with_fact("node_id", node_id));
    }
    for (input_index, input) in union.inputs.iter().enumerate() {
        let input_node = input_node(plan, *input, node_id)?;
        let input_available = available_columns(available_columns_by_node, *input)?;
        if input_node.output_schema().fields().len() != output_schema.fields().len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.union.validate",
                "union input schema field count must match the union output schema",
            )
            .with_fact("node_id", node_id)
            .with_fact("input_index", input_index)
            .with_fact("input_node", *input)
            .with_fact("input_fields", input_node.output_schema().fields().len())
            .with_fact("output_fields", output_schema.fields().len()));
        }
        for (field_index, (input_field, output_field)) in
            input_node.output_schema().fields().iter().zip(output_schema.fields().iter()).enumerate()
        {
            if !input_available.contains(input_field.name()) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.union.validate",
                    "union input schema references a column not produced by the input node",
                )
                .with_fact("node_id", node_id)
                .with_fact("input_node", *input)
                .with_fact("column", input_field.name())
                .with_fact("available_columns", comma_separated(input_available)));
            }
            if input_field.as_ref() != output_field.as_ref() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.union.validate",
                    "union input field must match the union output field by ordinal",
                )
                .with_fact("node_id", node_id)
                .with_fact("input_index", input_index)
                .with_fact("input_node", *input)
                .with_fact("field_index", field_index)
                .with_fact("input_field", input_field.name())
                .with_fact("output_field", output_field.name())
                .with_fact("input_type", format!("{:?}", input_field.data_type()))
                .with_fact("output_type", format!("{:?}", output_field.data_type()))
                .with_fact("input_nullable", input_field.is_nullable())
                .with_fact("output_nullable", output_field.is_nullable()));
            }
        }
    }
    Ok(())
}

fn validate_window_capability(
    window: &crate::plan::WindowNode,
    input_schema: &SchemaRef,
    output_schema: &SchemaRef,
) -> Result<()> {
    if window.spec.functions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.window.validate",
            "window node requires at least one function",
        ));
    }
    if output_schema.fields().len() != input_schema.fields().len() + window.spec.functions.len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.window.validate",
            "window output schema must append one field per window function",
        )
        .with_fact("input_fields", input_schema.fields().len())
        .with_fact("function_count", window.spec.functions.len())
        .with_fact("output_fields", output_schema.fields().len()));
    }
    for (index, (input_field, output_field)) in input_schema.fields().iter().zip(output_schema.fields()).enumerate() {
        if input_field.as_ref() != output_field.as_ref() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.window.validate",
                "window output schema must preserve input fields before appended window outputs",
            )
            .with_fact("field_index", index)
            .with_fact("input_field", input_field.name())
            .with_fact("output_field", output_field.name()));
        }
    }
    for (index, function) in window.spec.functions.iter().enumerate() {
        let field = output_schema.fields().get(input_schema.fields().len() + index).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.window.validate",
                "window output field is missing",
            )
            .with_fact("function_index", index)
        })?;
        if field.name() != function.alias() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.window.validate",
                "window output field name must match function alias",
            )
            .with_fact("function_index", index)
            .with_fact("alias", function.alias())
            .with_fact("field", field.name()));
        }
        match function {
            crate::plan::WindowFunction::Aggregate { function, frame, .. } => {
                match frame {
                    crate::plan::WindowFrame::WholePartition => {
                        if window.spec.partition_by.is_empty() {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "nexus_query_engine.window.validate",
                                "native whole-partition aggregate windows require at least one partition key",
                            ));
                        }
                        if !window.spec.order_by.is_empty() {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "nexus_query_engine.window.validate",
                                "native whole-partition aggregate windows require no ORDER BY",
                            ));
                        }
                    },
                    crate::plan::WindowFrame::RowsUnboundedPrecedingToCurrent => {
                        if window.spec.order_by.is_empty() {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "nexus_query_engine.window.validate",
                                "native cumulative aggregate windows require ORDER BY",
                            ));
                        }
                        if !matches!(
                            function,
                            crate::expr::AggregateFunction::Sum
                                | crate::expr::AggregateFunction::Min
                                | crate::expr::AggregateFunction::Max
                        ) {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "nexus_query_engine.window.validate",
                                "native cumulative aggregate window function is unsupported",
                            )
                            .with_fact("function", format!("{function:?}")));
                        }
                    },
                }
                for expr in &window.spec.partition_by {
                    lower_column_expr(expr, "nexus_query_engine.window.validate_partition")?;
                }
            },
            crate::plan::WindowFunction::Rank { .. } => {
                if window.spec.order_by.is_empty() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedOperator,
                        "nexus_query_engine.window.validate",
                        "native rank windows require ORDER BY",
                    ));
                }
                if !matches!(field.data_type(), arrow::datatypes::DataType::UInt64) {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "nexus_query_engine.window.validate",
                        "native rank output must be UInt64",
                    )
                    .with_fact("field", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
                }
            },
            crate::plan::WindowFunction::RowNumber { .. } => {
                if !matches!(field.data_type(), arrow::datatypes::DataType::UInt64) {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "nexus_query_engine.window.validate",
                        "native row_number output must be UInt64",
                    )
                    .with_fact("field", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
                }
                if field.is_nullable() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "nexus_query_engine.window.validate",
                        "native row_number output must be non-nullable",
                    )
                    .with_fact("field", field.name()));
                }
            },
            crate::plan::WindowFunction::LagLead { expr, default, .. } => {
                if window.spec.order_by.is_empty() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedOperator,
                        "nexus_query_engine.window.validate",
                        "native LAG / LEAD windows require ORDER BY",
                    ));
                }
                for partition_expr in &window.spec.partition_by {
                    lower_column_expr(partition_expr, "nexus_query_engine.window.validate_partition")?;
                }
                // LAG / LEAD output dtype matches the input expression dtype as declared
                // by the planner. Verify the planner-declared field is nullable, because
                // the shift always produces nullable rows on partition boundaries even
                // when a non-null default is supplied (libcudf models the absence of a
                // fill scalar as a null-yielding shift).
                if let Some(value) = default {
                    if &value.data_type() != field.data_type() {
                        return Err(QueryError::new(
                            QueryErrorCode::UnsupportedType,
                            "nexus_query_engine.window.validate",
                            "native LAG / LEAD default literal type must match output field type",
                        )
                        .with_fact("field", field.name())
                        .with_fact("expected", format!("{:?}", field.data_type()))
                        .with_fact("actual", format!("{:?}", value.data_type())));
                    }
                } else if !field.is_nullable() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "nexus_query_engine.window.validate",
                        "native LAG / LEAD output without an explicit default must be nullable",
                    )
                    .with_fact("field", field.name()));
                }
                let _ = expr;
            },
        }
    }
    Ok(())
}

fn schema_column_names(schema: &SchemaRef) -> BTreeSet<String> {
    schema.fields().iter().map(|field| field.name().clone()).collect()
}

fn filter_available_columns(available: &BTreeSet<String>, output_schema: &SchemaRef) -> BTreeSet<String> {
    output_schema
        .fields()
        .iter()
        .filter(|field| available.contains(field.name()))
        .map(|field| field.name().clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateFunction, BinaryOp, LiteralValue},
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions,
            GraphEdgeColumnBindings, ProjectionOutput, QueryPlanBuilder, WindowFunction, WindowSpec,
        },
        source::{
            LiteralBatchSource, LocalParquetFile, LocalParquetSource, ParquetFieldIdReadBinding, SourceColumnRef,
            SourceNode,
        },
    };

    #[test]
    fn validate_filter_uses_filter_lowering_for_large_left_deep_boolean_predicates() {
        let schema = Arc::new(Schema::new(vec![Field::new("ca_zip", ArrowDataType::Utf8, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
        let filter = builder.add_filter(source, left_deep_zip_range_disjunction(384), Arc::clone(&schema));
        let plan = builder.finish_with_host_sink(filter);

        validate_query_plan(&plan).expect("large q08-class filter predicate should validate without generic lowering");
    }

    #[test]
    fn validate_rejects_global_whole_partition_aggregate_window() {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("v", ArrowDataType::Int64, true),
        ]));
        let window_schema = Arc::new(Schema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("v", ArrowDataType::Int64, true),
            Field::new("window_v", ArrowDataType::Int64, true),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(input_schema)));
        let window = builder.add_window(
            source,
            WindowSpec::new(
                Vec::new(),
                Vec::new(),
                vec![WindowFunction::aggregate(AggregateFunction::Sum, Expr::column("v"), "window_v")],
            ),
            window_schema,
        );
        let plan = builder.finish_with_host_sink(window);

        let error = validate_query_plan(&plan).expect_err("global whole-partition aggregate windows are unsupported");
        assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
        assert_eq!(error.operation(), "nexus_query_engine.window.validate");
    }

    #[test]
    fn validate_graph_algorithm_rejects_incompatible_directedness() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", ArrowDataType::Int64, false),
            Field::new("dst", ArrowDataType::Int64, false),
            Field::new("weight", ArrowDataType::Float64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::StronglyConnectedComponents,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions { directed: false, ..Default::default() },
            GraphAlgorithmParams::StronglyConnectedComponents(Default::default()),
        );
        let graph = builder.add_graph_algorithm(source, spec);
        let plan = builder.finish_with_host_sink(graph);

        let error = validate_query_plan(&plan).expect_err("SCC must reject undirected graph construction");
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.operation(), "nexus_query_engine.graph_algorithm.validate");
        assert_eq!(error.fact_value("reason"), Some("unsupported_graph_directedness"));
        assert_eq!(error.fact_value("algorithm"), Some("strongly_connected_components"));
        assert_eq!(error.fact_value("directed"), Some("false"));
        assert_eq!(error.fact_value("required_directed"), Some("true"));
    }

    #[test]
    fn validate_graph_algorithm_rejects_int64_for_c_api_int32_only_algorithms() {
        for (algorithm, params) in [
            (GraphAlgorithm::MinimumSpanningTree, GraphAlgorithmParams::MinimumSpanningTree(Default::default())),
            (
                GraphAlgorithm::SpectralModularityMaximization,
                GraphAlgorithmParams::SpectralModularityMaximization(Default::default()),
            ),
        ] {
            let int64_plan = graph_algorithm_validation_plan(
                algorithm,
                params.clone(),
                Arc::new(Schema::new(vec![
                    Field::new("src", ArrowDataType::Int64, false),
                    Field::new("dst", ArrowDataType::Int64, false),
                    Field::new("weight", ArrowDataType::Float64, false),
                ])),
            );
            let error =
                validate_query_plan(&int64_plan).expect_err("cuGraph C API algorithms must reject Int64 vertices");
            assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
            assert_eq!(error.operation(), "nexus_query_engine.graph_algorithm.validate");
            assert_eq!(error.fact_value("reason"), Some("unsupported_vertex_dtype"));
            assert_eq!(error.fact_value("algorithm"), Some(algorithm.as_str()));
            assert_eq!(error.fact_value("edge_column_role"), Some("source"));
            assert_eq!(error.fact_value("vertex_id_type_support"), Some("int32_only_legacy_edge_type"));
            assert_eq!(error.fact_value("supported_vertex_dtypes"), Some("Int32"));

            let int32_plan = graph_algorithm_validation_plan(
                algorithm,
                params,
                Arc::new(Schema::new(vec![
                    Field::new("src", ArrowDataType::Int32, false),
                    Field::new("dst", ArrowDataType::Int32, false),
                    Field::new("weight", ArrowDataType::Float64, false),
                ])),
            );
            validate_query_plan(&int32_plan).expect("cuGraph C API algorithms must keep accepting Int32 vertices");
        }
    }

    #[test]
    fn validate_local_parquet_source_uses_output_schema_override_for_available_columns() {
        let physical_schema = Arc::new(Schema::new(vec![
            Field::new("old_status", ArrowDataType::Int64, true),
            Field::new("status", ArrowDataType::Int64, true),
        ]));
        let logical_schema = Arc::new(Schema::new(vec![Field::new("status", ArrowDataType::Int64, true)]));
        let file = LocalParquetFile::new("/tmp/renamed.parquet").with_field_id_read_bindings(vec![
            ParquetFieldIdReadBinding::new(2, "status", "old_status", 0, true, ArrowDataType::Int64),
        ]);
        let source = LocalParquetSource::from_files(vec![file], physical_schema, vec!["old_status".to_string()])
            .with_column_roles(vec!["status".to_string()], Vec::new())
            .with_column_source_refs(vec![0], vec![SourceColumnRef::File(0)], Vec::new())
            .with_output_schema_override(Arc::clone(&logical_schema));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder.add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("status"), Field::new("status", ArrowDataType::Int64, true))],
        );
        let plan = builder.finish_with_host_sink(projection);

        validate_query_plan(&plan).expect("logical output alias should be available downstream");
    }

    fn left_deep_zip_range_disjunction(count: usize) -> Expr {
        let mut ranges = (0..count).map(zip_range_predicate);
        let first = ranges.next().expect("test predicate needs at least one range");
        ranges.fold(first, |left, right| Expr::binary(left, BinaryOp::Or, right))
    }

    fn zip_range_predicate(index: usize) -> Expr {
        let lower = format!("{:05}", 10000 + index);
        let upper = format!("{:05}", 10001 + index);
        let lower_bound =
            Expr::binary(Expr::column("ca_zip"), BinaryOp::GtEq, Expr::literal(LiteralValue::Utf8(lower)));
        let upper_bound = Expr::binary(Expr::column("ca_zip"), BinaryOp::Lt, Expr::literal(LiteralValue::Utf8(upper)));
        Expr::binary(lower_bound, BinaryOp::And, upper_bound)
    }

    fn graph_algorithm_validation_plan(
        algorithm: GraphAlgorithm,
        params: GraphAlgorithmParams,
        schema: Arc<Schema>,
    ) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            algorithm,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            algorithm.definition().default_graph_options(&GraphConstructionOptions::default()),
            params,
        );
        let graph = builder.add_graph_algorithm(source, spec);
        builder.finish_with_host_sink(graph)
    }
}
