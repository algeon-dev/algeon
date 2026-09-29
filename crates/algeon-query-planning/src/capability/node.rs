//! Per-node capability analysis.
//!
//! Produces a [`NodeCapability`] for each [`QueryNode`]: operator kind,
//! executable decision, and produced properties such as ordering and residency.
use std::collections::BTreeMap;

use arrow::datatypes::{DataType as ArrowDataType, Schema as ArrowSchema};

use super::{
    AggregateFunction, AggregateSpec, Expr, GroupingSetsSpec, JoinForm, JoinNode, NodeId, PlanNode, QueryNode,
    QueryPlan, SourceNode,
    aggregate::aggregate_state_contract,
    classify_join,
    expr::{
        expression_rejection_decision, validate_aggregate_expr_capability, validate_column_expr_capability,
        validate_general_expr_capability, validate_join_residual_expr_capability,
        validate_named_column_expr_capability, validate_post_join_residual_expr_capability,
    },
    model::{
        AggregateGrouping, CapabilityDecision, NodeCapability, NodeOperator, OrderingRequirement, ResidualRequirement,
        SourceFormatCapability,
    },
    rejection::{Reason, UnsupportedPart},
};
use crate::{
    backend::AnalysisContext,
    error::{QueryError, Result},
    ml_validation::validate_ml_algorithm_node,
    plan::{
        AggregateNode, EdgeNormalizeNode, EdgeWeightCombiner, GraphAlgorithmNode, GraphAlgorithmParams,
        GraphVertexPairInput, SortNode, graph_result_schema_for_vertex_type,
    },
    validation::{PlanColumns, algorithm_input, check_node},
    vector_validation::validate_vector_algorithm_node,
};

pub(super) fn analyze_node(
    plan: &QueryPlan,
    node: &QueryNode,
    columns: &PlanColumns,
    ctx: AnalysisContext<'_>,
) -> NodeCapability {
    let node_id = node.id;
    let kind = &node.kind;
    let operator = node_operator(kind);
    let executable = node_executable_decision(plan, node, columns, ctx);
    let aggregate_state = match kind {
        PlanNode::Aggregate(aggregate) => aggregate_state_contract(&aggregate.spec),
        _ => None,
    };
    let max_output_rows = node_max_output_rows(kind);
    NodeCapability { node_id, operator, executable, aggregate_state, max_output_rows }
}

/// Rows a node emits at most, when the operator alone bounds it: a scalar
/// aggregate (no groups and no grouping sets) emits one row; a limit with
/// fetch ≤ 1 emits at most one; a source with a ≤1 scan/max bound or a ≤1-row
/// literal emits at most one. Anything else is unbounded from the operator
/// alone; the compiler tracks tighter bounds per resource where it can prove
/// them.
fn node_max_output_rows(kind: &PlanNode) -> Option<u64> {
    match kind {
        PlanNode::Aggregate(aggregate)
            if aggregate.spec.group_exprs.is_empty() && aggregate.spec.grouping_sets.is_none() =>
        {
            Some(1)
        },
        PlanNode::Limit(limit) => limit.spec.fetch.filter(|fetch| *fetch <= 1).map(|fetch| fetch as u64),
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => source
                .scan_limit
                .filter(|limit| *limit <= 1)
                .map(|limit| limit as u64)
                .or_else(|| source.max_rows.filter(|rows| *rows <= 1).map(|rows| rows as u64)),
            SourceNode::LiteralBatch(source) => {
                let rows = source.row_count();
                (rows <= 1).then_some(rows as u64)
            },
            SourceNode::GpuResident(_) => None,
        },
        _ => None,
    }
}

pub(super) fn node_operator(kind: &PlanNode) -> NodeOperator {
    match kind {
        PlanNode::Source(source) => NodeOperator::Source { format: source_format_capability(source) },
        PlanNode::Filter(_) => NodeOperator::Filter,
        PlanNode::Projection(_) => NodeOperator::Projection,
        PlanNode::Aggregate(aggregate) => NodeOperator::Aggregate {
            grouping: if aggregate.spec.grouping_sets.is_some() {
                AggregateGrouping::GroupingSets
            } else if aggregate.spec.group_exprs.is_empty() {
                AggregateGrouping::Scalar
            } else {
                AggregateGrouping::Grouped
            },
            functions: aggregate.spec.aggr_exprs.iter().map(|expr| expr.function).collect(),
        },
        PlanNode::Window(window) => NodeOperator::Window { function_count: window.spec.functions.len() },
        PlanNode::Sort(sort) => NodeOperator::Sort {
            ordering: if sort.spec.expressions.is_empty() {
                OrderingRequirement::None
            } else {
                OrderingRequirement::Required
            },
        },
        PlanNode::Limit(_) => NodeOperator::Limit,
        PlanNode::Join(join) => NodeOperator::Join {
            join_type: join.spec.join_type.clone(),
            key_count: join.spec.keys.len(),
            residual: if join.spec.residual_predicate.is_some() {
                ResidualRequirement::Required
            } else {
                ResidualRequirement::None
            },
            build_side: join.spec.build_side,
        },
        PlanNode::Union(union) => NodeOperator::Union { input_count: union.inputs.len() },
        PlanNode::EdgeNormalize(node) => NodeOperator::EdgeNormalize {
            add_reverse: node.spec.add_reverse,
            dedup: node.spec.dedup,
            weight_combiner: node.spec.weight_combiner,
        },
        PlanNode::GraphAlgorithm(graph) => NodeOperator::GraphAlgorithm { algorithm: graph.spec.algorithm },
        PlanNode::VectorAlgorithm(vector) => NodeOperator::VectorAlgorithm { algorithm: vector.spec.algorithm },
        PlanNode::MlAlgorithm(ml) => NodeOperator::MlAlgorithm { algorithm: ml.spec.algorithm },
    }
}

pub(super) fn source_format_capability(source: &SourceNode) -> SourceFormatCapability {
    match source {
        SourceNode::LocalParquet(_) => SourceFormatCapability::LocalParquet,
        SourceNode::GpuResident(_) => SourceFormatCapability::GpuResident,
        SourceNode::LiteralBatch(_) => SourceFormatCapability::LiteralBatch,
    }
}

fn node_executable_decision(
    plan: &QueryPlan,
    node: &QueryNode,
    columns: &PlanColumns,
    ctx: AnalysisContext<'_>,
) -> CapabilityDecision {
    match &node.kind {
        PlanNode::Source(_) | PlanNode::Projection(_) | PlanNode::Window(_) | PlanNode::Limit(_) => {
            checked_node_decision(node.id, check_node(plan, node, columns, ctx))
        },
        PlanNode::Filter(filter) => expression_node_decision(
            node.id,
            Reason::UnsupportedFilterExpressionCapability,
            std::iter::once(&filter.predicate),
        ),
        PlanNode::Aggregate(aggregate) => aggregate_node_executable_decision(plan, node.id, aggregate),
        PlanNode::Sort(sort) => sort_node_executable_decision(node.id, sort),
        PlanNode::Join(join) => match join_node_executable_decision(node.id, join) {
            CapabilityDecision::Satisfied => checked_node_decision(node.id, check_node(plan, node, columns, ctx)),
            rejected => rejected,
        },
        PlanNode::Union(_) => CapabilityDecision::Satisfied,
        PlanNode::EdgeNormalize(edge_normalize) => {
            edge_normalize_node_executable_decision(plan, node.id, edge_normalize)
        },
        PlanNode::GraphAlgorithm(graph) => graph_algorithm_node_executable_decision(plan, node.id, graph),
        PlanNode::VectorAlgorithm(vector) => algorithm_node_decision(
            node.id,
            vector.spec.algorithm.as_str(),
            validate_vector_algorithm_node(vector, plan, columns.available_by_node(), node.id),
            Reason::UnsupportedVectorAlgorithmNodeCapability,
        ),
        PlanNode::MlAlgorithm(ml) => algorithm_node_decision(
            node.id,
            ml.spec.algorithm.as_str(),
            validate_ml_algorithm_node(ml, plan, columns.available_by_node(), node.id),
            Reason::UnsupportedMlAlgorithmNodeCapability,
        ),
    }
}

/// The decision for a node [`check_node`] checks: a failed rule rejects the
/// node with the reason admission gives the same validation failure.
fn checked_node_decision(node_id: NodeId, result: Result<()>) -> CapabilityDecision {
    match result {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(error) => CapabilityDecision::Rejected {
            reason: Reason::PlanValidation(UnsupportedPart::from_code(error.code())),
            facts: validation_error_facts(node_id, &error),
        },
    }
}

fn algorithm_node_decision(
    node_id: NodeId,
    algorithm: &str,
    result: Result<()>,
    reason: fn(UnsupportedPart) -> Reason,
) -> CapabilityDecision {
    match result {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(error) => {
            let mut facts = validation_error_facts(node_id, &error);
            facts.entry("algorithm").or_insert_with(|| algorithm.to_string());
            CapabilityDecision::Rejected { reason: reason(UnsupportedPart::from_code(error.code())), facts }
        },
    }
}

fn validation_error_facts(node_id: NodeId, error: &QueryError) -> BTreeMap<&'static str, String> {
    let mut facts = error.facts().iter().map(|(key, value)| (key, value.to_string())).collect::<BTreeMap<_, _>>();
    facts.entry("node_id").or_insert_with(|| node_id.to_string());
    facts.insert("validation_code", error.code().to_string());
    facts.insert("validation_operation", error.operation().to_string());
    facts
}

fn expression_node_decision<'a>(
    node_id: NodeId,
    reason: Reason,
    expressions: impl IntoIterator<Item = &'a Expr>,
) -> CapabilityDecision {
    for expression in expressions {
        if let Err(rejection) = validate_general_expr_capability(expression) {
            return expression_rejection_decision(node_id, reason, expression, rejection);
        }
    }
    CapabilityDecision::Satisfied
}

fn aggregate_node_executable_decision(
    plan: &QueryPlan,
    node_id: NodeId,
    aggregate: &AggregateNode,
) -> CapabilityDecision {
    if let Some(grouping_sets) = &aggregate.spec.grouping_sets
        && let Err(reason) = validate_grouping_sets_capability(&aggregate.spec, grouping_sets)
    {
        return CapabilityDecision::rejected(
            Reason::UnsupportedAggregateGroupingSetsCapability,
            [
                ("node_id", node_id.to_string()),
                ("reason", reason.to_string()),
                ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                ("grouping_set_count", grouping_sets.groups.len().to_string()),
            ],
        );
    }
    if aggregate.spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::Median)
        && aggregate.spec.aggr_exprs.iter().any(|expr| expr.function.is_floating_statistical())
    {
        return CapabilityDecision::rejected(
            Reason::UnsupportedAggregateExpressionCapability,
            [
                ("node_id", node_id.to_string()),
                ("reason", "median_floating_statistical_mixed_unsupported".to_string()),
                ("expression_family", "aggregate".to_string()),
                ("expression", "aggregate_spec".to_string()),
            ],
        );
    }
    let Some(input) = plan.node(aggregate.input) else {
        return CapabilityDecision::rejected(
            Reason::UnsupportedAggregateNodeCapability,
            [("node_id", node_id.to_string()), ("reason", "missing_input_node".to_string())],
        );
    };
    for expr in &aggregate.spec.group_exprs {
        if let Err(rejection) = validate_named_column_expr_capability(expr) {
            return expression_rejection_decision(
                node_id,
                Reason::UnsupportedAggregateGroupExpressionCapability,
                expr,
                rejection,
            );
        }
    }
    for aggregate_expr in &aggregate.spec.aggr_exprs {
        if let Err(rejection) =
            validate_aggregate_expr_capability(aggregate_expr, aggregate.spec.group_exprs.is_empty())
        {
            return expression_rejection_decision(
                node_id,
                Reason::UnsupportedAggregateExpressionCapability,
                &aggregate_expr.expr,
                rejection,
            );
        }
        if aggregate_expr.function == AggregateFunction::Median {
            let Some(data_type) = aggregate_expr_input_data_type(&aggregate_expr.expr, input.output_schema().as_ref())
            else {
                return CapabilityDecision::rejected(
                    Reason::UnsupportedAggregateExpressionCapability,
                    [
                        ("node_id", node_id.to_string()),
                        ("reason", "median_requires_column_input".to_string()),
                        ("expression_family", "aggregate".to_string()),
                        ("expression", aggregate_expr.expr.to_string()),
                    ],
                );
            };
            if !is_cudf_median_input_type(data_type) {
                return CapabilityDecision::rejected(
                    Reason::UnsupportedAggregateExpressionCapability,
                    [
                        ("node_id", node_id.to_string()),
                        ("reason", "unsupported_median_input_type".to_string()),
                        ("expression_family", "aggregate".to_string()),
                        ("expression", aggregate_expr.expr.to_string()),
                        ("data_type", format!("{data_type:?}")),
                    ],
                );
            }
        }
    }
    CapabilityDecision::Satisfied
}

fn aggregate_expr_input_data_type<'a>(expr: &'a Expr, input_schema: &'a ArrowSchema) -> Option<&'a ArrowDataType> {
    match expr {
        Expr::Column(column) => input_schema.field_with_name(column.name()).ok().map(|field| field.data_type()),
        Expr::Cast { target_type, .. } | Expr::TryCast { target_type, .. } => Some(target_type),
        Expr::Alias { expr, .. } => aggregate_expr_input_data_type(expr, input_schema),
        _ => None,
    }
}

pub fn is_cudf_median_input_type(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64
    )
}

fn validate_grouping_sets_capability(
    spec: &AggregateSpec,
    grouping_sets: &GroupingSetsSpec,
) -> std::result::Result<(), &'static str> {
    if spec.group_exprs.is_empty() {
        return Err("empty_grouping_sets_keys");
    }
    if spec.group_exprs.len() > 64 {
        return Err("too_many_grouping_sets_keys");
    }
    if grouping_sets.groups.is_empty() {
        return Err("empty_grouping_sets");
    }
    if grouping_sets.grouping_id_alias.is_empty() {
        return Err("empty_grouping_id_alias");
    }
    if grouping_sets.groups.iter().any(|group| group.len() != spec.group_exprs.len()) {
        return Err("grouping_set_mask_length_mismatch");
    }
    if spec.aggr_exprs.iter().any(|expr| expr.function.is_floating_statistical()) {
        return Err("grouping_sets_floating_statistical_aggregate_not_supported");
    }
    Ok(())
}

fn sort_node_executable_decision(node_id: NodeId, sort: &SortNode) -> CapabilityDecision {
    if sort.spec.expressions.is_empty() {
        return CapabilityDecision::rejected(
            Reason::UnsupportedSortNodeCapability,
            [("node_id", node_id.to_string()), ("reason", "empty_sort_expressions".to_string())],
        );
    }
    for sort_expr in &sort.spec.expressions {
        if let Err(rejection) = validate_column_expr_capability(&sort_expr.expr) {
            return expression_rejection_decision(
                node_id,
                Reason::UnsupportedSortExpressionCapability,
                &sort_expr.expr,
                rejection,
            );
        }
    }
    CapabilityDecision::Satisfied
}

/// Whether a join executor form runs the join and evaluates its residual.
/// The IR and type rules every node meets come from [`check_node`] after it.
fn join_node_executable_decision(node_id: NodeId, join: &JoinNode) -> CapabilityDecision {
    // A residual that only a post-join filter evaluates rejects a non-inner
    // join after the residual expression is checked; every other structural
    // miss rejects first.
    let form = match classify_join(node_id, join) {
        Err(rejection) if rejection.reason != Reason::PostJoinResidualFilterRequiresOneShotEquiJoin => {
            return CapabilityDecision::Rejected { reason: rejection.reason, facts: rejection.facts };
        },
        form => form,
    };
    if let Some(predicate) = &join.spec.residual_predicate {
        let residual_capability = validate_join_residual_expr_capability(predicate).or_else(|rejection| {
            if matches!(form, Ok(JoinForm::KeyedEquiPostFilter)) {
                validate_post_join_residual_expr_capability(predicate)
            } else {
                Err(rejection)
            }
        });
        if let Err(rejection) = residual_capability {
            return expression_rejection_decision(
                node_id,
                Reason::UnsupportedJoinResidualExpressionCapability,
                predicate,
                rejection,
            );
        }
    }
    match form {
        Ok(_) => CapabilityDecision::Satisfied,
        Err(rejection) => CapabilityDecision::Rejected { reason: rejection.reason, facts: rejection.facts },
    }
}

fn graph_algorithm_node_executable_decision(
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &GraphAlgorithmNode,
) -> CapabilityDecision {
    if let Some(required) = graph.spec.algorithm.definition().required_graph_directedness()
        && graph.spec.graph_options.directed != required
    {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "unsupported_graph_directedness".to_string()),
                ("algorithm", graph.spec.algorithm.as_str().to_string()),
                ("directed", graph.spec.graph_options.directed.to_string()),
                ("required_directed", required.to_string()),
            ],
        );
    }
    let Some(input) = plan.node(graph.edge_input()) else {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [("node_id", node_id.to_string()), ("reason", "missing_input_node".to_string())],
        );
    };
    for (role, column) in [
        ("source", graph.spec.edge_columns.source.as_str()),
        ("destination", graph.spec.edge_columns.destination.as_str()),
    ] {
        let Some(field) = input.output_schema().field_with_name(column).ok() else {
            return CapabilityDecision::rejected(
                Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), ArrowDataType::Int32 | ArrowDataType::Int64) {
            return CapabilityDecision::rejected(
                Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::DataType),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "unsupported_vertex_type".to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                    ("data_type", format!("{:?}", field.data_type())),
                ],
            );
        }
        let vertex_id_type_support = graph.spec.algorithm.vertex_id_type_support();
        if !vertex_id_type_support.accepts_arrow_data_type(field.data_type()) {
            return CapabilityDecision::rejected(
                Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::DataType),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "unsupported_vertex_dtype".to_string()),
                    ("algorithm", graph.spec.algorithm.as_str().to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                    ("data_type", format!("{:?}", field.data_type())),
                    ("vertex_id_type_support", vertex_id_type_support.as_str().to_string()),
                    ("supported_vertex_dtypes", vertex_id_type_support.supported_dtypes_label().to_string()),
                ],
            );
        }
    }

    if let Some(weight_col) = graph.spec.edge_columns.weight.as_deref() {
        let Some(field) = input.output_schema().field_with_name(weight_col).ok() else {
            return CapabilityDecision::rejected(
                Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", "weight".to_string()),
                    ("column", weight_col.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), ArrowDataType::Float32 | ArrowDataType::Float64) {
            return CapabilityDecision::rejected(
                Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::DataType),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "unsupported_weight_type".to_string()),
                    ("edge_column_role", "weight".to_string()),
                    ("column", weight_col.to_string()),
                    ("data_type", format!("{:?}", field.data_type())),
                ],
            );
        }
    }

    if let Some(edge_id_col) = graph.spec.edge_columns.edge_id.as_deref() {
        let Some(field) = input.output_schema().field_with_name(edge_id_col).ok() else {
            return CapabilityDecision::rejected(
                Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", "edge_id".to_string()),
                    ("column", edge_id_col.to_string()),
                ],
            );
        };
        let source_type = input
            .output_schema()
            .field_with_name(&graph.spec.edge_columns.source)
            .expect("source column was validated above")
            .data_type();
        if field.data_type() != source_type {
            return CapabilityDecision::rejected(
                Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "edge_id_dtype_mismatch".to_string()),
                    ("edge_column_role", "edge_id".to_string()),
                    ("column", edge_id_col.to_string()),
                    ("edge_id_type", format!("{:?}", field.data_type())),
                    ("vertex_type", format!("{source_type:?}")),
                ],
            );
        }
    }

    if let GraphAlgorithmParams::Bfs(params) = &graph.spec.params
        && params.predicates.include_edge_ids.is_some()
        && graph.spec.edge_columns.edge_id.is_none()
    {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "missing_edge_id_binding".to_string()),
                ("predicate_role", "include_edge_ids".to_string()),
                ("missing_binding", "edge_id_col".to_string()),
            ],
        );
    }

    if let GraphAlgorithmParams::Similarity(params) = &graph.spec.params {
        let vertex_type = input
            .output_schema()
            .field_with_name(&graph.spec.edge_columns.source)
            .expect("source column was validated above")
            .data_type();
        let pair_input_decision =
            graph_similarity_pair_input_decision(plan, node_id, &params.vertex_pairs, vertex_type);
        if !matches!(pair_input_decision, CapabilityDecision::Satisfied) {
            return pair_input_decision;
        }
    }

    let vertex_type = input
        .output_schema()
        .field_with_name(&graph.spec.edge_columns.source)
        .expect("source column was validated above")
        .data_type();
    if graph.output_schema().as_ref()
        != graph_result_schema_for_vertex_type(graph.spec.algorithm, &graph.spec.params, vertex_type).as_ref()
    {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "unexpected_output_schema".to_string()),
                ("algorithm", graph.spec.algorithm.as_str().to_string()),
                ("vertex_type", format!("{vertex_type:?}")),
                ("schema_field_count", graph.output_schema().fields().len().to_string()),
            ],
        );
    }

    CapabilityDecision::Satisfied
}

fn graph_similarity_pair_input_decision(
    plan: &QueryPlan,
    node_id: NodeId,
    pair_input: &GraphVertexPairInput,
    vertex_type: &ArrowDataType,
) -> CapabilityDecision {
    let Some(input) = plan.node(algorithm_input(plan, node_id, pair_input.input)) else {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "missing_vertex_pair_input_node".to_string()),
                ("pair_input", pair_input.input.to_string()),
            ],
        );
    };
    let Some(first) = input.output_schema().field_with_name(&pair_input.first_column).ok() else {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "missing_vertex_pair_column".to_string()),
                ("pair_column_role", "first".to_string()),
                ("column", pair_input.first_column.clone()),
            ],
        );
    };
    let Some(second) = input.output_schema().field_with_name(&pair_input.second_column).ok() else {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "missing_vertex_pair_column".to_string()),
                ("pair_column_role", "second".to_string()),
                ("column", pair_input.second_column.clone()),
            ],
        );
    };
    if first.data_type() != second.data_type() {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "vertex_pair_dtype_mismatch".to_string()),
                ("first_column", pair_input.first_column.clone()),
                ("first_type", format!("{:?}", first.data_type())),
                ("second_column", pair_input.second_column.clone()),
                ("second_type", format!("{:?}", second.data_type())),
            ],
        );
    }
    if !matches!(first.data_type(), ArrowDataType::Int32 | ArrowDataType::Int64) {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::DataType),
            [
                ("node_id", node_id.to_string()),
                ("reason", "unsupported_vertex_pair_type".to_string()),
                ("pair_column_role", "first".to_string()),
                ("column", pair_input.first_column.clone()),
                ("data_type", format!("{:?}", first.data_type())),
            ],
        );
    }
    if first.data_type() != vertex_type {
        return CapabilityDecision::rejected(
            Reason::UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart::Operator),
            [
                ("node_id", node_id.to_string()),
                ("reason", "vertex_pair_graph_vertex_dtype_mismatch".to_string()),
                ("first_column", pair_input.first_column.clone()),
                ("pair_type", format!("{:?}", first.data_type())),
                ("vertex_type", format!("{vertex_type:?}")),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

fn edge_normalize_node_executable_decision(
    plan: &QueryPlan,
    node_id: NodeId,
    edge_normalize: &EdgeNormalizeNode,
) -> CapabilityDecision {
    let Some(input) = plan.node(edge_normalize.input) else {
        return CapabilityDecision::rejected(
            Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::Operator),
            [("node_id", node_id.to_string()), ("reason", "missing_input_node".to_string())],
        );
    };
    for (role, column) in [
        ("source", edge_normalize.spec.edge_columns.source.as_str()),
        ("destination", edge_normalize.spec.edge_columns.destination.as_str()),
    ] {
        let Some(field) = input.output_schema().field_with_name(column).ok() else {
            return CapabilityDecision::rejected(
                Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::Operator),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), ArrowDataType::Int32 | ArrowDataType::Int64) {
            return CapabilityDecision::rejected(
                Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::DataType),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "unsupported_vertex_type".to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                    ("data_type", format!("{:?}", field.data_type())),
                ],
            );
        }
    }

    if matches!(
        edge_normalize.spec.weight_combiner,
        EdgeWeightCombiner::MinWeight | EdgeWeightCombiner::KeepLastInputOrder
    ) && edge_normalize.spec.edge_columns.weight.is_none()
    {
        return CapabilityDecision::rejected(
            Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::Operator),
            [("node_id", node_id.to_string()), ("reason", "weight_combiner_without_weight_column".to_string())],
        );
    }

    if let Some(weight_col) = edge_normalize.spec.edge_columns.weight.as_deref() {
        let Some(field) = input.output_schema().field_with_name(weight_col).ok() else {
            return CapabilityDecision::rejected(
                Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::Operator),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", "weight".to_string()),
                    ("column", weight_col.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), ArrowDataType::Float32 | ArrowDataType::Float64) {
            return CapabilityDecision::rejected(
                Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::DataType),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "unsupported_weight_type".to_string()),
                    ("edge_column_role", "weight".to_string()),
                    ("column", weight_col.to_string()),
                    ("data_type", format!("{:?}", field.data_type())),
                ],
            );
        }
    }

    if let Some(edge_id_col) = edge_normalize.spec.edge_columns.edge_id.as_deref() {
        let Some(field) = input.output_schema().field_with_name(edge_id_col).ok() else {
            return CapabilityDecision::rejected(
                Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::Operator),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", "edge_id".to_string()),
                    ("column", edge_id_col.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), ArrowDataType::Int32 | ArrowDataType::Int64) {
            return CapabilityDecision::rejected(
                Reason::UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart::DataType),
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "unsupported_edge_id_type".to_string()),
                    ("edge_column_role", "edge_id".to_string()),
                    ("column", edge_id_col.to_string()),
                    ("data_type", format!("{:?}", field.data_type())),
                ],
            );
        }
    }

    CapabilityDecision::Satisfied
}
