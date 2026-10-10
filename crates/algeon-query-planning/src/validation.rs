use std::collections::{BTreeMap, BTreeSet, HashMap};

use arrow::datatypes::{DataType as ArrowDataType, Field, SchemaRef};

use crate::{
    backend::{AnalysisContext, ExpressionUse},
    capability::{
        require_expression_capability, validate_aggregate_expr_capability, validate_column_expr_capability,
        validate_general_expr_capability,
    },
    datatypes::{arrow_data_types_match_except_list_child_names, is_supported_arrow_implicit_cast},
    error::{QueryError, QueryErrorCode, QueryErrorKind, Result},
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue},
    ml_validation::validate_ml_algorithm_node,
    plan::{
        AggregateNode, EdgeNormalizeNode, FilterNode, GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams,
        GraphEdgePredicateInput, GraphForceAtlas2InitialPositionsInput, GraphForceAtlas2VertexAttributesInput,
        GraphPageRankPersonalizationInput, GraphPredicateInput, GraphVertexPairInput, JoinSide, LimitNode, NodeId,
        PlanNode, QueryNode, QueryPlan, SortNode, UnionNode, WindowFrame, WindowFunction, WindowNode,
        graph_result_schema_for_vertex_type,
    },
    prepared::{
        AvgStateAliases, aggregate_spec_has_avg, aggregate_spec_has_distinct,
        aggregate_spec_has_floating_statistical_aggregate, avg_declared_sum_state_type, avg_division_type,
        lower_column_expr, lower_named_column_expr, lower_sort_spec,
    },
    source::{
        SourceNode, local_parquet_execution_frame_columns, local_parquet_execution_frame_schema,
        validate_literal_batch_source, validate_local_parquet_source,
    },
    vector_validation::{
        validate_schema_supported_with_vector_ingress, validate_vector_algorithm_node,
        validate_vector_ingress_projection_output, vector_ingress_columns_by_input,
    },
};

mod aggregate;
mod expressions;
mod join;
mod types;

use aggregate::validate_aggregate_capability;
#[doc(hidden)]
pub use aggregate::{
    global_count_star_expr, global_valid_count_state_alias, validate_aggregate_output_names,
    validate_avg_output_fields, validate_global_aggregate_capability,
};
#[doc(hidden)]
pub use expressions::substring_stop_position;
pub(crate) use expressions::validate_expr;
use expressions::{comma_separated, validate_projection_output_expr};
use join::validate_join_capability;
pub(crate) use join::walk_join_residual_column_sides;
pub(crate) use types::require_boolean;
#[doc(hidden)]
pub use types::{is_integral_type, is_string_type};
use types::{require_scalar_operator_value, require_timezone_free_operator_key};

#[doc(hidden)]
pub fn projection_schema_mismatch_error(
    field_index: usize,
    field_name: &str,
    actual_type: &ArrowDataType,
    expected_type: &ArrowDataType,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedType,
        "query_engine.projection.schema_coerce",
        "native projection output cannot be coerced to its declared schema type",
    )
    .with_fact("reason", "projection_output_schema_mismatch")
    .with_fact("field_name", field_name)
    .with_fact("field_index", field_index)
    .with_fact("actual_type", format!("{actual_type:?}"))
    .with_fact("expected_type", format!("{expected_type:?}"))
}

#[doc(hidden)]
pub fn topk_candidate_rows(spec: LimitSpec) -> Result<usize> {
    let fetch = spec.fetch.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.final_sort.topk_limit",
            "offset-only limit has no finite top-k candidate bound",
        )
        .with_fact("skip", spec.skip)
        .with_fact("reason", "unbounded_fetch")
    })?;
    spec.skip.checked_add(fetch).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.final_sort.topk_limit",
            "limit skip and fetch overflow usize",
        )
        .with_fact("skip", spec.skip)
        .with_fact("fetch", fetch)
    })
}

#[doc(hidden)]
pub fn validate_query_plan(plan: &QueryPlan, ctx: AnalysisContext<'_>) -> Result<()> {
    let columns = PlanColumns::for_plan(plan);
    // An unsupported node is reported only after every node and the sink are
    // known to be well formed: a malformed plan is an error, while an
    // unsupported one is a rejection a frontend may fall back from, so the
    // first unsupported node must not hide a malformed one after it.
    let mut first_unsupported = None;

    for node in plan.nodes() {
        validate_node_references(plan, node, columns.available_by_node())?;
        if let Err(error) = check_node(plan, node, &columns, ctx) {
            defer_unsupported(error, &mut first_unsupported)?;
        }
    }
    if let Err(error) = validate_sink(plan, columns.available_by_node(), ctx) {
        defer_unsupported(error, &mut first_unsupported)?;
    }
    first_unsupported.map_or(Ok(()), Err)
}

/// Column facts every node rule reads, derived once per plan from the node
/// structure alone.
pub(crate) struct PlanColumns {
    available: Vec<BTreeSet<String>>,
    vector_ingress: BTreeMap<NodeId, BTreeSet<String>>,
}

impl PlanColumns {
    pub(crate) fn for_plan(plan: &QueryPlan) -> Self {
        let mut available = Vec::<BTreeSet<String>>::with_capacity(plan.node_count());
        for node in plan.nodes() {
            let columns = match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::LocalParquet(source) => local_parquet_execution_frame_columns(source),
                    SourceNode::GpuResident(_) | SourceNode::LiteralBatch(_) => {
                        schema_column_names(node.output_schema())
                    },
                },
                // Plan construction orders every input before its consumers.
                PlanNode::Filter(FilterNode { input, .. })
                | PlanNode::Sort(SortNode { input, .. })
                | PlanNode::Limit(LimitNode { input, .. }) => {
                    filter_available_columns(&available[input.index()], node.output_schema())
                },
                _ => schema_column_names(node.output_schema()),
            };
            available.push(columns);
        }
        Self { available, vector_ingress: vector_ingress_columns_by_input(plan) }
    }

    /// The columns each node makes available to its consumers, by node id.
    pub(crate) fn available_by_node(&self) -> &[BTreeSet<String>] {
        &self.available
    }
}

/// Returns `error` unless the backend merely cannot run the plan, in which
/// case the first such error is kept for the end of validation.
fn defer_unsupported(error: QueryError, first_unsupported: &mut Option<QueryError>) -> Result<()> {
    if error.kind() != QueryErrorKind::Unsupported {
        return Err(error);
    }
    first_unsupported.get_or_insert(error);
    Ok(())
}

/// Checks one node against the IR rules and the backend support every layer
/// applies to it. Validation and capability analysis both run it, so each
/// rule has one copy.
pub(crate) fn check_node(
    plan: &QueryPlan,
    node: &QueryNode,
    columns: &PlanColumns,
    ctx: AnalysisContext<'_>,
) -> Result<()> {
    let available_columns_by_node = columns.available_by_node();
    let vector_ingress_columns = &columns.vector_ingress;
    validate_schema_supported_with_vector_ingress(
        node.output_schema(),
        "query_engine.plan.validate_schema",
        vector_ingress_columns.get(&node.id),
        ctx,
    )?;
    validate_output_names_unique(node)?;
    match &node.kind {
        PlanNode::Source(source) => validate_source_node(source, ctx)?,
        PlanNode::Filter(filter) => {
            let input = input_node(plan, filter.input, node.id)?;
            let input_available = available_columns(available_columns_by_node, filter.input)?;
            let predicate_type = validate_expr(&filter.predicate, input.output_schema(), input_available, ctx)?;
            require_boolean(&predicate_type, "filter predicate")?;
            ctx.backend().check_expression(&filter.predicate, ExpressionUse::FilterPredicate)?;
        },
        PlanNode::Projection(projection) => {
            let input = input_node(plan, projection.input, node.id)?;
            let input_available = available_columns(available_columns_by_node, projection.input)?;
            for (field_index, output) in projection.outputs.iter().enumerate() {
                let actual_type = if vector_ingress_columns
                    .get(&node.id)
                    .is_some_and(|columns| columns.contains(output.field.name()))
                {
                    validate_vector_ingress_projection_output(
                        output,
                        input.output_schema(),
                        input_available,
                        vector_ingress_columns.get(&projection.input),
                    )?;
                    output.field.data_type().clone()
                } else {
                    validate_projection_output_expr(output, input.output_schema(), input_available, ctx)?
                };
                let expected_type = output.field.data_type();
                if !arrow_data_types_match_except_list_child_names(&actual_type, expected_type)
                    && !is_supported_arrow_implicit_cast(&actual_type, expected_type, ctx.backend())
                {
                    return Err(projection_schema_mismatch_error(
                        field_index,
                        output.field.name(),
                        &actual_type,
                        expected_type,
                    )
                    .with_fact("reason_detail", "implicit_cast_not_supported"));
                }
            }
            for output in &projection.outputs {
                ctx.backend().check_projection_output(output)?;
                require_expression_capability(validate_general_expr_capability(&output.expr), &output.expr)?;
            }
            // `outputs` is a public field, so a caller can change it after
            // construction derived the schema from it.
            if projection.outputs.len() != node.output_schema().fields().len() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.plan.validate_projection",
                    "projection outputs do not match the projection schema",
                )
                .with_fact("node_id", node.id)
                .with_fact("projection_output_count", projection.outputs.len())
                .with_fact("schema_field_count", node.output_schema().fields().len()));
            }
        },
        PlanNode::Aggregate(aggregate) => {
            let input = input_node(plan, aggregate.input, node.id)?;
            let input_available = available_columns(available_columns_by_node, aggregate.input)?;
            for expr in &aggregate.spec.group_exprs {
                let data_type = validate_expr(expr, input.output_schema(), input_available, ctx)?;
                require_timezone_free_operator_key(&data_type, "aggregate group key")?;
            }
            let mut avg_input_types = Vec::new();
            for expr in &aggregate.spec.aggr_exprs {
                let data_type = validate_expr(&expr.expr, input.output_schema(), input_available, ctx)?;
                require_timezone_free_operator_key(&data_type, "aggregate input")?;
                if let Some(filter) = &expr.filter {
                    let filter_type = validate_expr(filter, input.output_schema(), input_available, ctx)?;
                    require_boolean(&filter_type, "aggregate filter predicate")?;
                    ctx.backend().check_expression(filter, ExpressionUse::Value)?;
                }
                if expr.function == AggregateFunction::Avg {
                    avg_input_types.push((expr.alias.as_str(), data_type));
                }
            }
            validate_aggregate_capability(aggregate, node.output_schema(), ctx)?;
            // After the capability check, so an unsupported AVG output type
            // keeps that check's error.
            for (alias, input_type) in avg_input_types {
                if let Ok(output) = node.output_schema().field_with_name(alias) {
                    avg_declared_sum_state_type(&input_type, output.data_type())
                        .map_err(|error| error.with_fact("alias", alias))?;
                }
            }
        },
        PlanNode::Window(window) => {
            let input = input_node(plan, window.input, node.id)?;
            let input_available = available_columns(available_columns_by_node, window.input)?;
            for expr in &window.spec.partition_by {
                let data_type = validate_expr(expr, input.output_schema(), input_available, ctx)?;
                require_timezone_free_operator_key(&data_type, "window partition key")?;
            }
            for sort_expr in &window.spec.order_by {
                let data_type = validate_expr(&sort_expr.expr, input.output_schema(), input_available, ctx)?;
                require_timezone_free_operator_key(&data_type, "window order key")?;
            }
            for function in &window.spec.functions {
                match function {
                    WindowFunction::Aggregate { expr, .. } => {
                        let data_type = validate_expr(expr, input.output_schema(), input_available, ctx)?;
                        require_scalar_operator_value(&data_type, "window aggregate input")?;
                    },
                    WindowFunction::Rank { .. } | WindowFunction::RowNumber { .. } => {},
                    WindowFunction::LagLead { expr, .. } => {
                        let data_type = validate_expr(expr, input.output_schema(), input_available, ctx)?;
                        require_scalar_operator_value(&data_type, "window LAG / LEAD input")?;
                    },
                }
            }
            validate_window_capability(window)?;
            // The window kernel keys partitions, orderings, and LAG / LEAD
            // inputs by input columns.
            let keys = window.spec.partition_by.iter().chain(window.spec.order_by.iter().map(|key| &key.expr));
            for expr in keys {
                require_expression_capability(validate_column_expr_capability(expr), expr)?;
            }
            for function in &window.spec.functions {
                match function {
                    WindowFunction::Aggregate { function, expr, .. } => {
                        let aggregate = AggregateExpr::new(*function, expr.clone(), "__window_aggregate");
                        require_expression_capability(validate_aggregate_expr_capability(&aggregate, false), expr)?;
                    },
                    WindowFunction::LagLead { expr, .. } => {
                        require_expression_capability(validate_column_expr_capability(expr), expr)?;
                    },
                    WindowFunction::Rank { .. } | WindowFunction::RowNumber { .. } => {},
                }
            }
        },
        PlanNode::Sort(sort) => {
            let input = input_node(plan, sort.input, node.id)?;
            let input_available = available_columns(available_columns_by_node, sort.input)?;
            for sort_expr in &sort.spec.expressions {
                let data_type = validate_expr(&sort_expr.expr, input.output_schema(), input_available, ctx)?;
                require_timezone_free_operator_key(&data_type, "sort key")?;
            }
            lower_sort_spec(&sort.spec)?;
        },
        PlanNode::Limit(limit) => {
            if limit.spec.fetch.is_some() {
                topk_candidate_rows(limit.spec)?;
            }
        },
        PlanNode::Join(join) => {
            let left = input_node(plan, join.inputs[0], node.id)?;
            let right = input_node(plan, join.inputs[1], node.id)?;
            let left_available = available_columns(available_columns_by_node, join.inputs[0])?;
            let right_available = available_columns(available_columns_by_node, join.inputs[1])?;
            validate_join_capability(
                node.id,
                join,
                left.output_schema(),
                left_available,
                right.output_schema(),
                right_available,
                node.output_schema(),
                ctx,
            )?;
        },
        PlanNode::Union(union) => {
            validate_union_node(union, node.id, plan, available_columns_by_node)?;
        },
        PlanNode::EdgeNormalize(edge_normalize) => {
            let input = input_node(plan, edge_normalize.input, node.id)?;
            let input_available = available_columns(available_columns_by_node, edge_normalize.input)?;
            validate_edge_normalize_node(edge_normalize, input.output_schema(), input_available, node.output_schema())?;
        },
        PlanNode::GraphAlgorithm(graph) => {
            let input = input_node(plan, graph.edge_input(), node.id)?;
            let input_available = available_columns(available_columns_by_node, graph.edge_input())?;
            validate_graph_algorithm_node(graph, input.output_schema(), input_available, node.output_schema())?;
            validate_graph_algorithm_side_inputs(graph, plan, available_columns_by_node, node.id)?;
        },
        PlanNode::VectorAlgorithm(vector) => {
            validate_vector_algorithm_node(vector, plan, available_columns_by_node, node.id)?;
        },
        PlanNode::MlAlgorithm(ml) => {
            validate_ml_algorithm_node(ml, plan, available_columns_by_node, node.id)?;
        },
    }
    Ok(())
}

/// Checks that every column `node` reads is produced by the input it reads,
/// without asking the backend about types or operators.
fn validate_node_references(
    plan: &QueryPlan,
    node: &QueryNode,
    available_columns_by_node: &[BTreeSet<String>],
) -> Result<()> {
    let inputs = node
        .input_ids()
        .iter()
        .map(|input| {
            input_node(plan, *input, node.id)?;
            available_columns(available_columns_by_node, *input)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut reads = Vec::<(usize, String)>::new();
    let mut read = |input: usize, expr: &Expr| {
        reads.extend(expr.columns().map(|column| (input, column.name().to_string())));
    };
    match &node.kind {
        PlanNode::Filter(filter) => read(0, &filter.predicate),
        PlanNode::Projection(projection) => projection.outputs.iter().for_each(|output| read(0, &output.expr)),
        PlanNode::Aggregate(aggregate) => {
            aggregate.spec.group_exprs.iter().for_each(|expr| read(0, expr));
            for expr in &aggregate.spec.aggr_exprs {
                read(0, &expr.expr);
                expr.filter.iter().for_each(|filter| read(0, filter));
            }
        },
        PlanNode::Window(window) => {
            window.spec.partition_by.iter().for_each(|expr| read(0, expr));
            window.spec.order_by.iter().for_each(|sort_expr| read(0, &sort_expr.expr));
            for function in &window.spec.functions {
                if let WindowFunction::Aggregate { expr, .. } | WindowFunction::LagLead { expr, .. } = function {
                    read(0, expr);
                }
            }
        },
        PlanNode::Sort(sort) => sort.spec.expressions.iter().for_each(|sort_expr| read(0, &sort_expr.expr)),
        PlanNode::Join(join) => {
            for key in &join.spec.keys {
                reads.push((0, key.left.name().to_string()));
                reads.push((1, key.right.name().to_string()));
            }
            for output in &join.spec.output_projection {
                reads.push((
                    match output.side {
                        JoinSide::Left => 0,
                        JoinSide::Right => 1,
                    },
                    output.column.name().to_string(),
                ));
            }
            if let Some(residual) = &join.spec.residual_predicate {
                walk_join_residual_column_sides(
                    residual,
                    "query_engine.join.validate_residual",
                    "join residual column must be input-qualified",
                    &mut |input, name| reads.push((input, name.to_string())),
                )?;
            }
        },
        _ => {},
    }
    for (input, column) in reads {
        if !inputs[input].contains(&column) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.plan.validate_expr",
                "native expression references a column that is not produced by its input",
            )
            .with_fact("node_id", node.id)
            .with_fact("input", node.input_ids()[input])
            .with_fact("column", column)
            .with_fact("available_columns", comma_separated(inputs[input])));
        }
    }
    Ok(())
}

/// The host sink's field count and types are construction invariants; the
/// root must still produce every column the sink reads, and the backend must
/// support the sink's types.
fn validate_sink(
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    ctx: AnalysisContext<'_>,
) -> Result<()> {
    let Some(schema) = plan.sink().caller_visible_host_schema() else {
        return Ok(());
    };
    let root = plan.root_node();
    let sink_available = available_columns(available_columns_by_node, root.id)?;
    if let Some(root_field) = root.output_schema().fields().iter().find(|field| !sink_available.contains(field.name()))
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.validate_sink",
            "host sink root schema references a column not produced by the root node",
        )
        .with_fact("column", root_field.name())
        .with_fact("root_node", root.id)
        .with_fact("available_columns", comma_separated(sink_available)));
    }
    validate_schema_supported(schema, "query_engine.plan.validate_sink_schema", ctx)
}

fn validate_source_node(source: &SourceNode, ctx: AnalysisContext<'_>) -> Result<()> {
    match source {
        SourceNode::LocalParquet(source) => {
            validate_local_parquet_source(source, ctx)?;
            local_parquet_execution_frame_schema(source).map(|_| ())
        },
        SourceNode::GpuResident(source) => {
            if source.input_name.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.gpu_resident_source.validate",
                    "GPU-resident source input name must not be empty",
                )
                .with_fact("reason", "empty_input_name"));
            }
            Ok(())
        },
        SourceNode::LiteralBatch(source) => validate_literal_batch_source(source, ctx),
    }
}

/// The plan node an algorithm relation binding names by its position among
/// the algorithm node's inputs. Plan construction checks every position.
pub(crate) fn algorithm_input(plan: &QueryPlan, node_id: NodeId, ordinal: usize) -> NodeId {
    plan.nodes()[node_id.index()].input_ids()[ordinal]
}

fn input_node(plan: &QueryPlan, input: NodeId, consumer: NodeId) -> Result<&QueryNode> {
    plan.node(input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.validate",
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
            "query_engine.plan.validate",
            "query plan node references an input that has not been validated",
        )
        .with_fact("input", input)
    })
}

fn validate_schema_supported(schema: &SchemaRef, operation: &'static str, ctx: AnalysisContext<'_>) -> Result<()> {
    for field in schema.fields() {
        if !ctx.backend().supports_type(field.data_type()) {
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
    let mut first_output_indexes = HashMap::with_capacity(output_names.len());
    for (duplicate_output_index, name) in output_names.iter().enumerate() {
        if let Some(&first_output_index) = first_output_indexes.get(name) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.plan.validate_output_names",
                "query plan node output schema contains duplicate executable column names",
            )
            .with_fact("node_id", node.id)
            .with_fact("node_kind", plan_node_kind(&node.kind))
            .with_fact("duplicate_output_name", *name)
            .with_fact("first_output_index", first_output_index)
            .with_fact("duplicate_output_index", duplicate_output_index)
            .with_fact("output_names", output_names.join(",")));
        }
        first_output_indexes.insert(*name, duplicate_output_index);
    }
    Ok(())
}

#[doc(hidden)]
pub fn plan_node_kind(node: &PlanNode) -> &'static str {
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
        PlanNode::MlAlgorithm(_) => "ml_algorithm",
    }
}

fn validate_edge_normalize_node(
    edge_normalize: &EdgeNormalizeNode,
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
            "query_engine.edge_normalize.validate",
            "edge normalize output schema must contain source, destination, optional weight, and optional edge-id columns in order",
        )
        .with_fact("expected_columns", expected.join(","))
        .with_fact("actual_columns", actual.join(",")));
    }
    Ok(())
}

fn validate_graph_algorithm_node(
    graph: &GraphAlgorithmNode,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
    output_schema: &SchemaRef,
) -> Result<()> {
    if let Some(required) = graph.spec.algorithm.definition().required_graph_directedness()
        && graph.spec.graph_options.directed != required
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.validate",
            "graph algorithm directed option is incompatible with the cuGraph algorithm",
        )
        .with_fact("reason", "unsupported_graph_directedness")
        .with_fact("algorithm", graph.spec.algorithm.as_str())
        .with_fact("directed", graph.spec.graph_options.directed)
        .with_fact("required_directed", required));
    }
    validate_graph_vertex_column("source", &graph.spec.edge_columns.source, input_schema, input_available)?;
    validate_graph_vertex_column("destination", &graph.spec.edge_columns.destination, input_schema, input_available)?;
    let source_field = graph_edge_field("source", &graph.spec.edge_columns.source, input_schema, input_available)?;
    let expected_output =
        graph_result_schema_for_vertex_type(graph.spec.algorithm, &graph.spec.params, source_field.data_type());
    if output_schema.as_ref() != expected_output.as_ref() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.validate",
            "graph algorithm output schema must match the input vertex-id domain",
        )
        .with_fact("algorithm", graph.spec.algorithm.as_str())
        .with_fact("vertex_type", source_field.data_type())
        .with_fact("output_fields", output_schema.fields().len()));
    }
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
                "query_engine.graph_algorithm.validate",
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
    graph: &GraphAlgorithmNode,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let edge_input = input_node(plan, graph.edge_input(), node_id)?;
    let edge_available = available_columns(available_columns_by_node, graph.edge_input())?;
    let vertex_type =
        graph_edge_field("source", &graph.spec.edge_columns.source, edge_input.output_schema(), edge_available)?
            .data_type()
            .clone();
    match &graph.spec.params {
        GraphAlgorithmParams::Bfs(params) => {
            if let Some(path_output) = params.path_output {
                if path_output.max_output_rows == 0 {
                    return Err(graph_algorithm_error("BFS path output requires a positive row bound")
                        .with_fact("reason", "bfs_path_output_row_bound_zero"));
                }
                if params.predicates.target_vertices.is_none() {
                    return Err(graph_algorithm_error("BFS path output requires a target_vertices binding")
                        .with_fact("reason", "bfs_path_target_binding_missing"));
                }
                if params.return_target_info {
                    return Err(graph_algorithm_error("BFS path output cannot also return target metadata")
                        .with_fact("reason", "bfs_path_target_metadata_conflict"));
                }
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
        },
        GraphAlgorithmParams::PersonalizedPageRank(params) => validate_graph_personalization_input(
            &params.personalization,
            &vertex_type,
            plan,
            available_columns_by_node,
            node_id,
        ),
        GraphAlgorithmParams::Similarity(params) => validate_graph_vertex_pair_input(
            &params.vertex_pairs,
            &vertex_type,
            plan,
            available_columns_by_node,
            node_id,
        ),
        GraphAlgorithmParams::ForceAtlas2(params) => {
            if params.prevent_overlapping
                && params.vertex_attributes.as_ref().is_none_or(|input| input.radius.is_none())
            {
                return Err(graph_algorithm_error("ForceAtlas2 prevent_overlapping requires a radius binding")
                    .with_fact("reason", "force_atlas2_radius_required"));
            }
            if let Some(input) = &params.vertex_attributes {
                validate_force_atlas2_vertex_attributes(input, &vertex_type, plan, available_columns_by_node, node_id)?;
            }
            if let Some(input) = &params.initial_positions {
                validate_force_atlas2_initial_positions(input, &vertex_type, plan, available_columns_by_node, node_id)?;
            }
            Ok(())
        },
        _ => Ok(()),
    }
}

fn validate_force_atlas2_vertex_attributes(
    input: &GraphForceAtlas2VertexAttributesInput,
    vertex_type: &ArrowDataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    if input.radius.is_none() && input.mobility.is_none() && input.mass.is_none() {
        return Err(graph_algorithm_error("ForceAtlas2 vertex attributes require at least one value binding")
            .with_fact("reason", "force_atlas2_vertex_attributes_empty"));
    }
    let input_node = input_node(plan, algorithm_input(plan, node_id, input.input), node_id)?;
    let available = available_columns(available_columns_by_node, algorithm_input(plan, node_id, input.input))?;
    validate_graph_predicate_column(
        "force_atlas2_vertex_attributes_vertex",
        &input.vertex,
        input_node.output_schema(),
        available,
        vertex_type,
    )?;
    for (role, column) in [
        ("force_atlas2_radius", input.radius.as_deref()),
        ("force_atlas2_mobility", input.mobility.as_deref()),
        ("force_atlas2_mass", input.mass.as_deref()),
    ] {
        if let Some(column) = column {
            validate_force_atlas2_value_column(role, column, input_node.output_schema(), available)?;
        }
    }
    Ok(())
}

fn validate_force_atlas2_initial_positions(
    input: &GraphForceAtlas2InitialPositionsInput,
    vertex_type: &ArrowDataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let input_node = input_node(plan, algorithm_input(plan, node_id, input.input), node_id)?;
    let available = available_columns(available_columns_by_node, algorithm_input(plan, node_id, input.input))?;
    validate_graph_predicate_column(
        "force_atlas2_initial_positions_vertex",
        &input.vertex,
        input_node.output_schema(),
        available,
        vertex_type,
    )?;
    validate_force_atlas2_value_column("force_atlas2_initial_x", &input.x, input_node.output_schema(), available)?;
    validate_force_atlas2_value_column("force_atlas2_initial_y", &input.y, input_node.output_schema(), available)
}

fn validate_force_atlas2_value_column(
    role: &'static str,
    column: &str,
    input_schema: &SchemaRef,
    input_available: &BTreeSet<String>,
) -> Result<()> {
    let field = graph_edge_field(role, column, input_schema, input_available)?;
    if field.data_type() != &ArrowDataType::Float32 {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
            "ForceAtlas2 side-input value columns must be Float32",
        )
        .with_fact("reason", "force_atlas2_side_input_value_dtype_unsupported")
        .with_fact("role", role)
        .with_fact("column", column)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(())
}

fn validate_graph_vertex_pair_input(
    input: &GraphVertexPairInput,
    vertex_type: &ArrowDataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let input_node = input_node(plan, algorithm_input(plan, node_id, input.input), node_id)?;
    let input_available = available_columns(available_columns_by_node, algorithm_input(plan, node_id, input.input))?;
    let first = graph_vertex_pair_field("first", &input.first_column, input_node.output_schema(), input_available)?;
    let second = graph_vertex_pair_field("second", &input.second_column, input_node.output_schema(), input_available)?;
    if first.data_type() != second.data_type() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
            "vertex-pair columns must have matching physical types",
        )
        .with_fact("reason", "vertex_pair_dtype_mismatch")
        .with_fact("first_column", &input.first_column)
        .with_fact("first_type", format!("{:?}", first.data_type()))
        .with_fact("second_column", &input.second_column)
        .with_fact("second_type", format!("{:?}", second.data_type())));
    }
    if !matches!(first.data_type(), ArrowDataType::Int32 | ArrowDataType::Int64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
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
            "query_engine.graph_algorithm.validate",
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
    input: &GraphPageRankPersonalizationInput,
    vertex_type: &ArrowDataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let input_node = input_node(plan, algorithm_input(plan, node_id, input.input), node_id)?;
    let input_available = available_columns(available_columns_by_node, algorithm_input(plan, node_id, input.input))?;
    validate_graph_predicate_column(
        "personalization_vertex",
        &input.vertex,
        input_node.output_schema(),
        input_available,
        vertex_type,
    )?;
    let field = graph_edge_field("personalization_value", &input.value, input_node.output_schema(), input_available)?;
    if !matches!(field.data_type(), ArrowDataType::Float32 | ArrowDataType::Float64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
            "personalized PageRank value column type is unsupported",
        )
        .with_fact("reason", "unsupported_personalization_value_type")
        .with_fact("column", &input.value)
        .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(())
}

fn validate_graph_predicate_input(
    input: Option<&GraphPredicateInput>,
    role: &'static str,
    expected_type: &ArrowDataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let Some(input) = input else {
        return Ok(());
    };
    let input_node = input_node(plan, algorithm_input(plan, node_id, input.input), node_id)?;
    let input_available = available_columns(available_columns_by_node, algorithm_input(plan, node_id, input.input))?;
    validate_graph_predicate_column(role, &input.column, input_node.output_schema(), input_available, expected_type)
}

fn validate_graph_edge_predicate_input(
    input: &GraphEdgePredicateInput,
    expected_type: &ArrowDataType,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    let input_node = input_node(plan, algorithm_input(plan, node_id, input.input), node_id)?;
    let input_available = available_columns(available_columns_by_node, algorithm_input(plan, node_id, input.input))?;
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
    expected_type: &ArrowDataType,
) -> Result<()> {
    let field = graph_edge_field(role, column, input_schema, input_available)?;
    if field.data_type() != expected_type {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
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
    if !matches!(field.data_type(), ArrowDataType::Int32 | ArrowDataType::Int64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
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
    algorithm: GraphAlgorithm,
    role: &'static str,
    column: &str,
    data_type: &ArrowDataType,
) -> Result<()> {
    let support = algorithm.vertex_id_type_support();
    if support.accepts_arrow_data_type(data_type) {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "query_engine.graph_algorithm.validate",
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
    if !matches!(field.data_type(), ArrowDataType::Float32 | ArrowDataType::Float64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
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
    if !matches!(field.data_type(), ArrowDataType::Int32 | ArrowDataType::Int64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.validate",
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
    QueryError::new(QueryErrorCode::InvalidPlan, "query_engine.graph_algorithm.validate", message)
}

/// Union inputs match the declared schema by position, a construction
/// invariant; each input must still produce the columns it contributes.
fn validate_union_node(
    union: &UnionNode,
    node_id: NodeId,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
) -> Result<()> {
    for input in &union.inputs {
        let input_node = input_node(plan, *input, node_id)?;
        let input_available = available_columns(available_columns_by_node, *input)?;
        if let Some(field) =
            input_node.output_schema().fields().iter().find(|field| !input_available.contains(field.name()))
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.union.validate",
                "union input schema references a column not produced by the input node",
            )
            .with_fact("node_id", node_id)
            .with_fact("input_node", *input)
            .with_fact("column", field.name())
            .with_fact("available_columns", comma_separated(input_available)));
        }
    }
    Ok(())
}

fn validate_window_capability(window: &WindowNode) -> Result<()> {
    if window.spec.functions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.window.validate",
            "window node requires at least one function",
        ));
    }
    for function in &window.spec.functions {
        let field = function.output();
        match function {
            WindowFunction::Aggregate { function, frame, .. } => {
                match frame {
                    WindowFrame::WholePartition => {
                        if !window.spec.order_by.is_empty() {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "query_engine.window.validate",
                                "native whole-partition aggregate windows require no ORDER BY",
                            ));
                        }
                    },
                    WindowFrame::RowsUnboundedPrecedingToCurrent => {
                        if window.spec.order_by.is_empty() {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "query_engine.window.validate",
                                "native cumulative aggregate windows require ORDER BY",
                            ));
                        }
                        if !matches!(function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
                        {
                            return Err(QueryError::new(
                                QueryErrorCode::UnsupportedOperator,
                                "query_engine.window.validate",
                                "native cumulative aggregate window function is unsupported",
                            )
                            .with_fact("function", format!("{function:?}")));
                        }
                    },
                }
                for expr in &window.spec.partition_by {
                    lower_column_expr(expr, "query_engine.window.validate_partition")?;
                }
            },
            WindowFunction::Rank { .. } => {
                if window.spec.order_by.is_empty() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedOperator,
                        "query_engine.window.validate",
                        "native rank windows require ORDER BY",
                    ));
                }
                if !matches!(field.data_type(), ArrowDataType::UInt64) {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.window.validate",
                        "native rank output must be UInt64",
                    )
                    .with_fact("field", field.name())
                    .with_fact("data_type", format!("{:?}", field.data_type())));
                }
            },
            WindowFunction::RowNumber { .. } => {
                let output_type = field.data_type();
                if !matches!(output_type, ArrowDataType::UInt64 | ArrowDataType::Int64 | ArrowDataType::Int32) {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.window.validate",
                        "native row_number output must be UInt64, Int64, or Int32",
                    )
                    .with_fact("field", field.name())
                    .with_fact("data_type", format!("{output_type:?}")));
                }
                if field.is_nullable() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.window.validate",
                        "native row_number output must be non-nullable",
                    )
                    .with_fact("field", field.name()));
                }
            },
            WindowFunction::LagLead { expr, default, .. } => {
                if window.spec.order_by.is_empty() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedOperator,
                        "query_engine.window.validate",
                        "native LAG / LEAD windows require ORDER BY",
                    ));
                }
                for partition_expr in &window.spec.partition_by {
                    lower_column_expr(partition_expr, "query_engine.window.validate_partition")?;
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
                            "query_engine.window.validate",
                            "native LAG / LEAD default literal type must match output field type",
                        )
                        .with_fact("field", field.name())
                        .with_fact("expected", format!("{:?}", field.data_type()))
                        .with_fact("actual", format!("{:?}", value.data_type())));
                    }
                } else if !field.is_nullable() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.window.validate",
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
