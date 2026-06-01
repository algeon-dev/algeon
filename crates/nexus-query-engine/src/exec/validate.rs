use std::collections::BTreeSet;

use arrow::datatypes::{Field, SchemaRef};

use super::*;
use crate::{
    datatypes::is_supported_arrow_type,
    plan::{QueryNode, UnionNode},
};

mod expressions;
mod join;
mod types;

use expressions::{comma_separated, validate_expr, validate_projection_output_expr};
use join::validate_join_capability;
use types::require_boolean;

pub(super) fn validate_query_plan(plan: &QueryPlan) -> Result<()> {
    validate_reachability(plan)?;
    let mut available_columns_by_node = Vec::with_capacity(plan.node_count());

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

        validate_schema_supported(node.output_schema(), "nexus_query_engine.plan.validate_schema")?;
        validate_output_names_unique(node)?;
        let available_columns = match &node.kind {
            PlanNode::Source(source) => validate_source_node(source)?,
            PlanNode::Filter(filter) => {
                let input = input_node(plan, filter.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, filter.input)?;
                let predicate_type = validate_expr(&filter.predicate, input.output_schema(), input_available)?;
                require_boolean(&predicate_type, "filter predicate")?;
                lower_expr(&filter.predicate)?;
                filter_available_columns(input_available, node.output_schema())
            },
            PlanNode::Projection(projection) => {
                let input = input_node(plan, projection.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, projection.input)?;
                for output in &projection.outputs {
                    validate_projection_output_expr(output, input.output_schema(), input_available)?;
                }
                validate_projection_capability(projection, &node.facts.output_schema)?;
                schema_column_names(node.output_schema())
            },
            PlanNode::Aggregate(aggregate) => {
                let input = input_node(plan, aggregate.input, node.id)?;
                let input_available = available_columns(&available_columns_by_node, aggregate.input)?;
                for expr in &aggregate.spec.group_exprs {
                    validate_expr(expr, input.output_schema(), input_available)?;
                }
                for expr in &aggregate.spec.aggr_exprs {
                    validate_expr(&expr.expr, input.output_schema(), input_available)?;
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
                    validate_expr(expr, input.output_schema(), input_available)?;
                }
                for sort_expr in &window.spec.order_by {
                    validate_expr(&sort_expr.expr, input.output_schema(), input_available)?;
                }
                for function in &window.spec.functions {
                    match function {
                        crate::plan::WindowFunction::Aggregate { expr, .. } => {
                            validate_expr(expr, input.output_schema(), input_available)?;
                        },
                        crate::plan::WindowFunction::Rank { .. } | crate::plan::WindowFunction::RowNumber { .. } => {},
                        crate::plan::WindowFunction::LagLead { expr, .. } => {
                            validate_expr(expr, input.output_schema(), input_available)?;
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
                    validate_expr(&sort_expr.expr, input.output_schema(), input_available)?;
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
    sink_field.clone().with_name(root_field.name().clone()) == *root_field
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
            let mut available = BTreeSet::new();
            available.extend(source.read_columns.iter().cloned());
            available.extend(source.partition_columns.iter().map(|column| column.name.clone()));
            available.extend(source.missing_columns.iter().map(|column| column.name.clone()));
            if source.is_zero_column_count_source() {
                available.insert(ZERO_COLUMN_COUNT_COLUMN.to_string());
            }
            Ok(available)
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

pub(crate) fn validate_literal_batch_source(source: &crate::source::LiteralBatchSource) -> Result<()> {
    if source.batches.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.literal_batch_source.validate",
            "literal batch source requires at least one batch",
        )
        .with_fact("reason", "empty_batches"));
    }
    validate_schema_supported(&source.schema, "nexus_query_engine.literal_batch_source.validate_schema")?;
    for (index, batch) in source.batches.iter().enumerate() {
        if batch.schema().as_ref() != source.schema.as_ref() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.literal_batch_source.validate",
                "literal batch schema must match declared source schema",
            )
            .with_fact("batch_index", index)
            .with_fact("declared_fields", source.schema.fields().len())
            .with_fact("batch_fields", batch.schema().fields().len()));
        }
    }
    Ok(())
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
    if output_schema.as_ref() != crate::plan::graph_result_schema_for_params(&graph.spec.params).as_ref() {
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
    let vertex_type =
        graph_edge_field("source", &graph.spec.edge_columns.source, input_schema, input_available)?.data_type().clone();
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
    let crate::plan::GraphAlgorithmParams::Bfs(params) = &graph.spec.params else {
        if graph.inputs.len() > 1 {
            return Err(graph_algorithm_error("only BFS graph algorithm nodes may carry predicate side inputs")
                .with_fact("node_id", node_id)
                .with_fact("algorithm", graph.spec.algorithm.as_str())
                .with_fact("side_input_count", graph.inputs.len().saturating_sub(1)));
        }
        return Ok(());
    };
    let expected_side_inputs = params.side_input_node_ids().collect::<Vec<_>>();
    if graph.inputs[1..] != expected_side_inputs[..] {
        return Err(graph_algorithm_error(
            "BFS graph algorithm side-input list must match source/predicate input bindings",
        )
        .with_fact("node_id", node_id)
        .with_fact(
            "actual_side_inputs",
            graph.inputs[1..].iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
        )
        .with_fact(
            "expected_side_inputs",
            expected_side_inputs.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
        ));
    }

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
