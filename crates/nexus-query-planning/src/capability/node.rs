//! Per-node capability analysis.
//!
//! Produces a [`NodeCapability`] for each [`QueryNode`]: operator kind,
//! executable decision, produced properties (ordering, partitioning, etc.),
//! local requirements on inputs, and any local reservation requirements.
use std::collections::{BTreeMap, BTreeSet};

use super::{
    AggregateExpr, AggregateFunction, AggregateSpec, Expr, GroupingSetsSpec, JoinExecutionCapability, JoinNode,
    JoinSide, KeyedHashJoinStrategy, KeyedHashJoinStrategyDecision, KeylessConditionalJoinKind, LimitNode, NodeId,
    PlanNode, QueryNode, QueryPlan, ReservationKind, SourceNode, WindowFunction,
    aggregate::{AggregateStateContract, aggregate_state_contract},
    expr::{
        expression_rejection_decision, validate_aggregate_expr_capability, validate_column_expr_capability,
        validate_general_expr_capability, validate_join_residual_expr_capability,
        validate_named_column_expr_capability, validate_post_join_residual_expr_capability, validation_result_decision,
    },
    hash_join_prebuild_reservation_unsupported_build_key_for_plan, join_node_execution_capability,
    join_residual_requires_post_join_filter, keyed_filtered_semi_anti_join_capable,
    keyed_hash_join_strategy_decision_for_plan,
    model::{
        AggregateGrouping, CapabilityDecision, ContractQuality, NodeCapability, NodeOperator, OrderingRequirement,
        ProducedProperty, ReservationRequirement, ResidualRequirement, SourceFormatCapability,
    },
    one_shot_equi_join_supported,
    shape::projection_scalar_setop_shape_decision,
    validate_local_parquet_source,
};
use crate::plan::UnionNode;

pub(super) fn analyze_node(plan: &QueryPlan, node: &QueryNode) -> NodeCapability {
    let node_id = node.id;
    let kind = &node.kind;
    let operator = node_operator(kind);
    let executable = node_executable_decision(plan, node);
    let produced_properties = produced_properties(node);
    let local_requirements = node_local_requirements(plan, node);
    NodeCapability { node_id, operator, executable, produced_properties, local_requirements }
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
            join_type: join.spec.join_type,
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
    }
}

pub(super) fn source_format_capability(source: &SourceNode) -> SourceFormatCapability {
    match source {
        SourceNode::LocalParquet(_) => SourceFormatCapability::LocalParquet,
        SourceNode::GpuResident(_) => SourceFormatCapability::GpuResident,
        SourceNode::LiteralBatch(_) => SourceFormatCapability::LiteralBatch,
    }
}

fn node_executable_decision(plan: &QueryPlan, node: &QueryNode) -> CapabilityDecision {
    match &node.kind {
        PlanNode::Source(source) => source_node_executable_decision(source),
        PlanNode::Filter(filter) => {
            // Exact replace-with-backrefs output preflight owns projection
            // column-input REGEXP_REPLACE only. Filter predicates have no
            // parallel size API yet, so admission rejects them instead
            // of launching unpreflighted string expansion under bounded tokens.
            if filter.predicate.contains_regexp_replace() {
                CapabilityDecision::rejected(
                    "unsupported_filter_expression_capability",
                    [
                        ("node_id", node.id.to_string()),
                        ("reason", "regexp_replace_filter_preflight_unavailable".to_string()),
                        ("required_contract", cudf::REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT.to_string()),
                        ("expression", filter.predicate.to_string()),
                    ],
                )
            } else {
                expression_node_decision(
                    node.id,
                    "unsupported_filter_expression_capability",
                    std::iter::once(&filter.predicate),
                )
            }
        },
        PlanNode::Projection(projection) => {
            let mut decision = expression_node_decision(
                node.id,
                "unsupported_projection_expression_capability",
                projection.expressions(),
            );
            if matches!(decision, CapabilityDecision::Satisfied)
                && projection.outputs.len() != node.output_schema().fields().len()
            {
                decision = CapabilityDecision::rejected(
                    "unsupported_projection_node_capability",
                    [
                        ("node_id", node.id.to_string()),
                        ("projection_output_count", projection.outputs.len().to_string()),
                        ("schema_field_count", node.output_schema().fields().len().to_string()),
                    ],
                );
            }
            decision
        },
        PlanNode::Aggregate(aggregate) => aggregate_node_executable_decision(plan, node.id, aggregate),
        PlanNode::Window(window) => window_node_executable_decision(node.id, window),
        PlanNode::Sort(sort) => sort_node_executable_decision(node.id, sort),
        PlanNode::Limit(limit) => limit_node_executable_decision(plan, node.id, limit),
        PlanNode::Join(join) => join_node_executable_decision(plan, node.id, join),
        PlanNode::Union(union) => union_node_executable_decision(plan, node.id, union),
        PlanNode::EdgeNormalize(edge_normalize) => {
            edge_normalize_node_executable_decision(plan, node.id, edge_normalize)
        },
        PlanNode::GraphAlgorithm(graph) => graph_algorithm_node_executable_decision(plan, node.id, graph),
        PlanNode::VectorAlgorithm(vector) => vector_algorithm_node_executable_decision(plan, node.id, vector),
    }
}

fn vector_algorithm_node_executable_decision(
    plan: &QueryPlan,
    node_id: NodeId,
    vector: &crate::plan::VectorAlgorithmNode,
) -> CapabilityDecision {
    match crate::vector_validation::validate_vector_algorithm_node_contract(vector, plan, node_id) {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(error) => {
            let mut facts =
                error.facts().iter().map(|(key, value)| (key, value.to_string())).collect::<BTreeMap<_, _>>();
            facts.entry("node_id").or_insert_with(|| node_id.to_string());
            facts.entry("algorithm").or_insert_with(|| vector.spec.algorithm.as_str().to_string());
            facts.insert("validation_code", error.code().to_string());
            facts.insert("validation_operation", error.operation().to_string());
            CapabilityDecision::Rejected { reason: "unsupported_vector_algorithm_node_capability", facts }
        },
    }
}

pub(super) fn source_node_executable_decision(source: &SourceNode) -> CapabilityDecision {
    match source {
        SourceNode::LocalParquet(source) => validation_result_decision(
            validate_local_parquet_source(source),
            "unsupported_local_parquet_source_capability",
            [("source_format", "local_parquet".to_string())],
        ),
        SourceNode::GpuResident(source) if source.input_name.is_empty() => CapabilityDecision::rejected(
            "unsupported_gpu_resident_source_capability",
            [("reason", "empty_input_name".to_string()), ("source_format", "gpu_resident".to_string())],
        ),
        SourceNode::GpuResident(_) => CapabilityDecision::Satisfied,
        SourceNode::LiteralBatch(source) => validation_result_decision(
            crate::source::validate_literal_batch_source(source),
            "unsupported_literal_batch_source_capability",
            [("source_format", "literal_batch".to_string())],
        ),
    }
}

fn expression_node_decision<'a>(
    node_id: NodeId,
    reason: &'static str,
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
    aggregate: &crate::plan::AggregateNode,
) -> CapabilityDecision {
    if let Some(grouping_sets) = &aggregate.spec.grouping_sets
        && let Err(reason) = validate_grouping_sets_capability(&aggregate.spec, grouping_sets)
    {
        return CapabilityDecision::rejected(
            "unsupported_aggregate_grouping_sets_capability",
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
            "unsupported_aggregate_expression_capability",
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
            "unsupported_aggregate_node_capability",
            [("node_id", node_id.to_string()), ("reason", "missing_input_node".to_string())],
        );
    };
    for expr in &aggregate.spec.group_exprs {
        if let Err(rejection) = validate_named_column_expr_capability(expr) {
            return expression_rejection_decision(
                node_id,
                "unsupported_aggregate_group_expression_capability",
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
                "unsupported_aggregate_expression_capability",
                &aggregate_expr.expr,
                rejection,
            );
        }
        if aggregate_expr.function == AggregateFunction::Median {
            let Some(data_type) = aggregate_expr_input_data_type(&aggregate_expr.expr, input.output_schema().as_ref())
            else {
                return CapabilityDecision::rejected(
                    "unsupported_aggregate_expression_capability",
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
                    "unsupported_aggregate_expression_capability",
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

fn aggregate_expr_input_data_type<'a>(
    expr: &'a crate::expr::Expr,
    input_schema: &'a arrow::datatypes::Schema,
) -> Option<&'a arrow::datatypes::DataType> {
    match expr {
        crate::expr::Expr::Column(column) => {
            input_schema.field_with_name(column.name()).ok().map(|field| field.data_type())
        },
        crate::expr::Expr::Cast { target_type, .. } | crate::expr::Expr::TryCast { target_type, .. } => {
            Some(target_type)
        },
        crate::expr::Expr::Alias { expr, .. } => aggregate_expr_input_data_type(expr, input_schema),
        _ => None,
    }
}

fn is_cudf_median_input_type(data_type: &arrow::datatypes::DataType) -> bool {
    matches!(
        data_type,
        arrow::datatypes::DataType::Int8
            | arrow::datatypes::DataType::Int16
            | arrow::datatypes::DataType::Int32
            | arrow::datatypes::DataType::Int64
            | arrow::datatypes::DataType::UInt8
            | arrow::datatypes::DataType::UInt16
            | arrow::datatypes::DataType::UInt32
            | arrow::datatypes::DataType::UInt64
            | arrow::datatypes::DataType::Float32
            | arrow::datatypes::DataType::Float64
    )
}

fn window_node_executable_decision(node_id: NodeId, window: &crate::plan::WindowNode) -> CapabilityDecision {
    if window.spec.functions.is_empty() {
        return CapabilityDecision::rejected(
            "unsupported_window_node_capability",
            [("node_id", node_id.to_string()), ("reason", "empty_functions".to_string())],
        );
    }
    for expr in &window.spec.partition_by {
        if let Err(rejection) = validate_column_expr_capability(expr) {
            return expression_rejection_decision(
                node_id,
                "unsupported_window_partition_expression_capability",
                expr,
                rejection,
            );
        }
    }
    for sort_expr in &window.spec.order_by {
        if let Err(rejection) = validate_column_expr_capability(&sort_expr.expr) {
            return expression_rejection_decision(
                node_id,
                "unsupported_window_order_expression_capability",
                &sort_expr.expr,
                rejection,
            );
        }
    }
    for function in &window.spec.functions {
        match function {
            crate::plan::WindowFunction::Aggregate { function, expr, frame, .. } => {
                match frame {
                    crate::plan::WindowFrame::WholePartition => {
                        if window.spec.partition_by.is_empty() {
                            return CapabilityDecision::rejected(
                                "unsupported_window_node_capability",
                                [
                                    ("node_id", node_id.to_string()),
                                    ("reason", "whole_partition_aggregate_window_requires_partition_by".to_string()),
                                ],
                            );
                        }
                        if !window.spec.order_by.is_empty() {
                            return CapabilityDecision::rejected(
                                "unsupported_window_node_capability",
                                [
                                    ("node_id", node_id.to_string()),
                                    ("reason", "whole_partition_aggregate_window_requires_no_order_by".to_string()),
                                ],
                            );
                        }
                    },
                    crate::plan::WindowFrame::RowsUnboundedPrecedingToCurrent => {
                        if window.spec.order_by.is_empty() {
                            return CapabilityDecision::rejected(
                                "unsupported_window_node_capability",
                                [
                                    ("node_id", node_id.to_string()),
                                    ("reason", "cumulative_aggregate_window_requires_order_by".to_string()),
                                ],
                            );
                        }
                        if !matches!(
                            function,
                            crate::expr::AggregateFunction::Sum
                                | crate::expr::AggregateFunction::Min
                                | crate::expr::AggregateFunction::Max
                        ) {
                            return CapabilityDecision::rejected(
                                "unsupported_window_node_capability",
                                [
                                    ("node_id", node_id.to_string()),
                                    ("reason", "cumulative_aggregate_window_function_unsupported".to_string()),
                                    ("function", format!("{function:?}")),
                                ],
                            );
                        }
                    },
                }
                let aggregate_expr = AggregateExpr::new(*function, expr.clone(), "__window_aggregate");
                if let Err(rejection) = validate_aggregate_expr_capability(&aggregate_expr, false) {
                    return expression_rejection_decision(
                        node_id,
                        "unsupported_window_aggregate_expression_capability",
                        expr,
                        rejection,
                    );
                }
            },
            crate::plan::WindowFunction::Rank { .. } if window.spec.order_by.is_empty() => {
                return CapabilityDecision::rejected(
                    "unsupported_window_node_capability",
                    [("node_id", node_id.to_string()), ("reason", "rank_requires_order_by".to_string())],
                );
            },
            crate::plan::WindowFunction::Rank { .. } => {},
            crate::plan::WindowFunction::RowNumber { .. } => {},
            crate::plan::WindowFunction::LagLead { expr, .. } if window.spec.order_by.is_empty() => {
                return CapabilityDecision::rejected(
                    "unsupported_window_node_capability",
                    [
                        ("node_id", node_id.to_string()),
                        ("reason", "lag_lead_requires_order_by".to_string()),
                        ("expression", format!("{expr}")),
                    ],
                );
            },
            crate::plan::WindowFunction::LagLead { expr, .. } => {
                if let Err(rejection) = validate_column_expr_capability(expr) {
                    return expression_rejection_decision(
                        node_id,
                        "unsupported_window_lag_lead_expression_capability",
                        expr,
                        rejection,
                    );
                }
            },
        }
    }
    CapabilityDecision::Satisfied
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

fn sort_node_executable_decision(node_id: NodeId, sort: &crate::plan::SortNode) -> CapabilityDecision {
    if sort.spec.expressions.is_empty() {
        return CapabilityDecision::rejected(
            "unsupported_sort_node_capability",
            [("node_id", node_id.to_string()), ("reason", "empty_sort_expressions".to_string())],
        );
    }
    for sort_expr in &sort.spec.expressions {
        if let Err(rejection) = validate_column_expr_capability(&sort_expr.expr) {
            return expression_rejection_decision(
                node_id,
                "unsupported_sort_expression_capability",
                &sort_expr.expr,
                rejection,
            );
        }
    }
    CapabilityDecision::Satisfied
}

fn limit_node_executable_decision(_plan: &QueryPlan, node_id: NodeId, limit: &LimitNode) -> CapabilityDecision {
    if let Some(fetch) = limit.spec.fetch
        && limit.spec.skip.checked_add(fetch).is_none()
    {
        return CapabilityDecision::rejected(
            "unsupported_limit_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "skip_fetch_overflow".to_string()),
                ("skip", limit.spec.skip.to_string()),
                ("fetch", fetch.to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

fn join_node_executable_decision(plan: &QueryPlan, node_id: NodeId, join: &JoinNode) -> CapabilityDecision {
    let Some(output_schema) = plan.node(node_id).map(|node| node.output_schema()) else {
        return CapabilityDecision::rejected(
            "unsupported_join_node_capability",
            [("node_id", node_id.to_string()), ("reason", "missing_join_node".to_string())],
        );
    };
    let Some(capability) = join_node_execution_capability(join) else {
        return CapabilityDecision::rejected(
            "unsupported_join_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("join_type", join.spec.join_type.to_string()),
                ("key_count", join.spec.keys.len().to_string()),
                ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                ("build_side", join.spec.build_side.to_string()),
            ],
        );
    };
    for (field_index, field) in output_schema.fields().iter().enumerate() {
        if !crate::datatypes::is_supported_arrow_type(field.data_type()) {
            return CapabilityDecision::rejected(
                "unsupported_join_output_dtype",
                [
                    ("node_id", node_id.to_string()),
                    ("field_index", field_index.to_string()),
                    ("field_name", field.name().to_string()),
                    ("dtype", field.data_type().to_string()),
                ],
            );
        }
    }
    if let Some(predicate) = &join.spec.residual_predicate {
        let residual_capability = validate_join_residual_expr_capability(predicate).or_else(|rejection| {
            if one_shot_equi_join_supported(join) && join_residual_requires_post_join_filter(join) {
                validate_post_join_residual_expr_capability(predicate)
            } else {
                Err(rejection)
            }
        });
        if let Err(rejection) = residual_capability {
            return expression_rejection_decision(
                node_id,
                "unsupported_join_residual_expression_capability",
                predicate,
                rejection,
            );
        }
    }
    if matches!(capability, JoinExecutionCapability::KeyedFilteredSemiAnti)
        && join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
    {
        return CapabilityDecision::rejected(
            "unsupported_join_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("join_type", join.spec.join_type.to_string()),
                ("reason", "semi_anti_projects_non_left_column".to_string()),
            ],
        );
    }
    if let Some(decision) = keyed_hash_join_strategy_decision_for_plan(plan, node_id, join) {
        match decision {
            KeyedHashJoinStrategyDecision::Rejected { reason, facts } => {
                let mut facts = facts;
                facts.insert("node_id", node_id.to_string());
                return CapabilityDecision::Rejected { reason, facts };
            },
            KeyedHashJoinStrategyDecision::Selected {
                strategy:
                    KeyedHashJoinStrategy::ReusableHashEqui
                    | KeyedHashJoinStrategy::OneShotEqui
                    | KeyedHashJoinStrategy::PartitionedHashEqui,
                ..
            } => {
                if let Some(mut facts) = hash_join_prebuild_reservation_unsupported_build_key_for_plan(plan, join) {
                    facts.insert("node_id", node_id.to_string());
                    return CapabilityDecision::Rejected {
                        reason: "unsupported_hash_join_pre_build_reservation_capability",
                        facts,
                    };
                }
            },
            KeyedHashJoinStrategyDecision::Selected { .. } => {},
        }
    }
    CapabilityDecision::Satisfied
}

fn union_node_executable_decision(plan: &QueryPlan, node_id: NodeId, union: &UnionNode) -> CapabilityDecision {
    if union.inputs.is_empty() {
        return CapabilityDecision::rejected(
            "unsupported_union_node_capability",
            [("node_id", node_id.to_string()), ("reason", "empty_inputs".to_string())],
        );
    }
    let Some(output_schema) = plan.node(node_id).map(|node| node.output_schema()) else {
        return CapabilityDecision::rejected(
            "unsupported_union_node_capability",
            [("node_id", node_id.to_string()), ("reason", "missing_union_node".to_string())],
        );
    };
    for (input_index, input) in union.inputs.iter().enumerate() {
        let Some(input_schema) = plan.node(*input).map(|node| node.output_schema()) else {
            return CapabilityDecision::rejected(
                "unsupported_union_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("input_index", input_index.to_string()),
                    ("reason", "missing_input_node".to_string()),
                ],
            );
        };
        if input_schema.as_ref() != output_schema.as_ref() {
            return CapabilityDecision::rejected(
                "unsupported_union_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("input_index", input_index.to_string()),
                    ("input_node", input.to_string()),
                    ("reason", "input_schema_mismatch".to_string()),
                ],
            );
        }
    }
    CapabilityDecision::Satisfied
}

fn graph_algorithm_node_executable_decision(
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
) -> CapabilityDecision {
    if let Some(required) = graph.spec.algorithm.definition().required_graph_directedness()
        && graph.spec.graph_options.directed != required
    {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "unsupported_graph_directedness".to_string()),
                ("algorithm", graph.spec.algorithm.as_str().to_string()),
                ("directed", graph.spec.graph_options.directed.to_string()),
                ("required_directed", required.to_string()),
            ],
        );
    }
    let Some(input) = plan.node(graph.input) else {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
            [("node_id", node_id.to_string()), ("reason", "missing_input_node".to_string())],
        );
    };
    for (role, column) in [
        ("source", graph.spec.edge_columns.source.as_str()),
        ("destination", graph.spec.edge_columns.destination.as_str()),
    ] {
        let Some(field) = input.output_schema().field_with_name(column).ok() else {
            return CapabilityDecision::rejected(
                "unsupported_graph_algorithm_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::Int64) {
            return CapabilityDecision::rejected(
                "unsupported_graph_algorithm_node_capability",
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
                "unsupported_graph_algorithm_node_capability",
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
                "unsupported_graph_algorithm_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", "weight".to_string()),
                    ("column", weight_col.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64) {
            return CapabilityDecision::rejected(
                "unsupported_graph_algorithm_node_capability",
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
                "unsupported_graph_algorithm_node_capability",
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
                "unsupported_graph_algorithm_node_capability",
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

    if let crate::plan::GraphAlgorithmParams::Bfs(params) = &graph.spec.params
        && params.predicates.include_edge_ids.is_some()
        && graph.spec.edge_columns.edge_id.is_none()
    {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "missing_edge_id_binding".to_string()),
                ("predicate_role", "include_edge_ids".to_string()),
                ("missing_binding", "edge_id_col".to_string()),
            ],
        );
    }

    if let crate::plan::GraphAlgorithmParams::Similarity(params) = &graph.spec.params {
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
        != crate::plan::graph_result_schema_for_vertex_type(graph.spec.algorithm, &graph.spec.params, vertex_type)
            .as_ref()
    {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
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
    pair_input: &crate::plan::GraphVertexPairInput,
    vertex_type: &arrow::datatypes::DataType,
) -> CapabilityDecision {
    let Some(input) = plan.node(pair_input.input) else {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "missing_vertex_pair_input_node".to_string()),
                ("pair_input", pair_input.input.to_string()),
            ],
        );
    };
    let Some(first) = input.output_schema().field_with_name(&pair_input.first_column).ok() else {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
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
            "unsupported_graph_algorithm_node_capability",
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
            "unsupported_graph_algorithm_node_capability",
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
    if !matches!(first.data_type(), arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::Int64) {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
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
            "unsupported_graph_algorithm_node_capability",
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
    edge_normalize: &crate::plan::EdgeNormalizeNode,
) -> CapabilityDecision {
    let Some(input) = plan.node(edge_normalize.input) else {
        return CapabilityDecision::rejected(
            "unsupported_edge_normalize_node_capability",
            [("node_id", node_id.to_string()), ("reason", "missing_input_node".to_string())],
        );
    };
    for (role, column) in [
        ("source", edge_normalize.spec.edge_columns.source.as_str()),
        ("destination", edge_normalize.spec.edge_columns.destination.as_str()),
    ] {
        let Some(field) = input.output_schema().field_with_name(column).ok() else {
            return CapabilityDecision::rejected(
                "unsupported_edge_normalize_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::Int64) {
            return CapabilityDecision::rejected(
                "unsupported_edge_normalize_node_capability",
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
        crate::plan::EdgeWeightCombiner::MinWeight | crate::plan::EdgeWeightCombiner::KeepLastInputOrder
    ) && edge_normalize.spec.edge_columns.weight.is_none()
    {
        return CapabilityDecision::rejected(
            "unsupported_edge_normalize_node_capability",
            [("node_id", node_id.to_string()), ("reason", "weight_combiner_without_weight_column".to_string())],
        );
    }

    if let Some(weight_col) = edge_normalize.spec.edge_columns.weight.as_deref() {
        let Some(field) = input.output_schema().field_with_name(weight_col).ok() else {
            return CapabilityDecision::rejected(
                "unsupported_edge_normalize_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", "weight".to_string()),
                    ("column", weight_col.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64) {
            return CapabilityDecision::rejected(
                "unsupported_edge_normalize_node_capability",
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
                "unsupported_edge_normalize_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "missing_edge_column".to_string()),
                    ("edge_column_role", "edge_id".to_string()),
                    ("column", edge_id_col.to_string()),
                ],
            );
        };
        if !matches!(field.data_type(), arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::Int64) {
            return CapabilityDecision::rejected(
                "unsupported_edge_normalize_node_capability",
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

pub(super) fn produced_properties(node: &QueryNode) -> Vec<ProducedProperty> {
    let mut properties = Vec::new();
    if node.facts.placement == crate::plan::DataPlacement::Device {
        properties.push(ProducedProperty::ResidentFrame);
    }
    match &node.kind {
        PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => {
            properties.push(ProducedProperty::ScalarFrame);
        },
        PlanNode::Aggregate(_) => properties.push(ProducedProperty::GroupedFrame),
        PlanNode::Window(_) => properties.push(ProducedProperty::OrderedFrame),
        PlanNode::Sort(_) => properties.push(ProducedProperty::OrderedFrame),
        _ => {},
    }
    properties
}

fn node_local_requirements(plan: &QueryPlan, node: &QueryNode) -> Vec<ReservationRequirement> {
    match &node.kind {
        PlanNode::Projection(projection)
            if plan.sink().input() == node.id
                && matches!(plan.node(projection.input).map(|input| &input.kind), Some(PlanNode::Limit(limit)) if limit.spec.fetch.is_some()) =>
        {
            let mut requirements = vec![ReservationRequirement::modeled(
                ReservationKind::OperatorOutput,
                "final_projection_output",
                "query_engine_limit_bounded_projection_output_boundary",
                ContractQuality::QueryEngineOwnedBoundary,
            )];
            if projection_contains_regexp_replace(projection) {
                requirements.extend(regexp_replace_requirements());
            }
            requirements
        },
        PlanNode::Projection(projection) if projection_is_scalar_setop_final_projection(plan, node.id) => {
            let mut requirements = vec![ReservationRequirement::modeled(
                ReservationKind::OperatorOutput,
                "final_projection_output",
                "query_engine_projection_scalar_setop_final_projection_output_boundary",
                ContractQuality::QueryEngineOwnedBoundary,
            )];
            if projection_contains_regexp_replace(projection) {
                requirements.extend(regexp_replace_requirements());
            }
            requirements
        },
        // Filter predicates that embed REGEXP_REPLACE are rare and cannot yet use the
        // column-input size preflight consumed by projection execution. Do not claim Exact.
        PlanNode::Projection(projection) if projection_contains_regexp_replace(projection) => {
            regexp_replace_requirements()
        },
        PlanNode::Filter(filter) if input_is_rank_window(plan, filter.input) => {
            vec![ReservationRequirement::modeled(
                ReservationKind::OperatorOutput,
                "rank_filter_output",
                "query_engine_rank_filter_output_boundary",
                ContractQuality::QueryEngineOwnedBoundary,
            )]
        },
        PlanNode::Aggregate(aggregate) => {
            let mut requirements = vec![ReservationRequirement::modeled(
                ReservationKind::OperatorOutput,
                "operator_output",
                "aggregate_contract",
                ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
            )];
            if matches!(
                aggregate_state_contract(&aggregate.spec),
                Ok(AggregateStateContract::DedupGroupOnly | AggregateStateContract::ExactDistinctCount)
            ) {
                requirements.push(ReservationRequirement::modeled(
                    ReservationKind::PartialState,
                    "distinct_dedup_aggregate_state",
                    "distinct_dedup_aggregate_state_materialization_contract",
                    ContractQuality::QueryEngineOwnedBoundary,
                ));
            }
            requirements
        },
        PlanNode::Window(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "operator_output",
            "window_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        )],
        PlanNode::Limit(limit)
            if limit.spec.fetch.is_some()
                && plan.sink().input() == node.id
                && final_topk_sort_before_limit(plan, limit.input).is_some() =>
        {
            vec![ReservationRequirement::modeled(
                ReservationKind::OperatorOutput,
                "final_topk_output",
                "query_engine_final_topk_output_boundary",
                ContractQuality::QueryEngineOwnedBoundary,
            )]
        },
        PlanNode::Limit(limit) if limit.spec.fetch.is_some() && plan.sink().input() == node.id => {
            vec![ReservationRequirement::modeled(
                ReservationKind::OperatorOutput,
                "global_limit_output",
                "query_engine_global_limit_output_boundary",
                ContractQuality::QueryEngineOwnedBoundary,
            )]
        },
        PlanNode::Join(join)
            if matches!(
                keyed_hash_join_strategy_decision_for_plan(plan, node.id, join),
                Some(KeyedHashJoinStrategyDecision::Selected { strategy: KeyedHashJoinStrategy::ReusableHashEqui, .. })
            ) =>
        {
            vec![ReservationRequirement::modeled(
                ReservationKind::HashTable,
                "join_hash_table",
                "hash_join_pre_build_reservation_size_bytes",
                ContractQuality::ExactPreflight,
            )]
        },
        PlanNode::Join(join)
            if matches!(
                keyed_hash_join_strategy_decision_for_plan(plan, node.id, join),
                Some(KeyedHashJoinStrategyDecision::Selected {
                    strategy: KeyedHashJoinStrategy::OneShotEqui | KeyedHashJoinStrategy::PartitionedHashEqui,
                    ..
                })
            ) =>
        {
            vec![ReservationRequirement::modeled(
                ReservationKind::HashTable,
                "join_keyed_equality_preflight_hash_table",
                "hash_join_pre_build_reservation_size_bytes",
                ContractQuality::ExactPreflight,
            )]
        },
        PlanNode::Join(join)
            if matches!(
                keyed_hash_join_strategy_decision_for_plan(plan, node.id, join),
                Some(KeyedHashJoinStrategyDecision::Selected {
                    strategy: KeyedHashJoinStrategy::PartitionedMixedKeyedResidual,
                    ..
                })
            ) =>
        {
            vec![ReservationRequirement::modeled(
                ReservationKind::MixedSizeData,
                "join_mixed_size_data",
                "mixed_keyed_residual_or_conditional_full_retained_size_data",
                ContractQuality::ObservedRetainedState,
            )]
        },
        PlanNode::Join(join) if keyed_filtered_semi_anti_join_capable(join) => vec![
            ReservationRequirement::modeled(
                ReservationKind::OperatorOutput,
                "join_filtered_build_key_deduplicate",
                "join_filtered_build_key_deduplicate_output_contract",
                ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
            ),
            ReservationRequirement::modeled(
                ReservationKind::HashTable,
                "join_hash_table",
                "hash_join_pre_build_reservation_size_bytes",
                ContractQuality::ExactPreflight,
            ),
            ReservationRequirement::modeled(
                ReservationKind::FilteredProbeState,
                "join_filtered_probe_state",
                "filtered_join_retained_probe_state",
                ContractQuality::ObservedRetainedState,
            ),
        ],
        PlanNode::Join(join)
            if matches!(KeylessConditionalJoinKind::try_from_join(join), Some(KeylessConditionalJoinKind::Full)) =>
        {
            vec![ReservationRequirement::modeled(
                ReservationKind::MixedSizeData,
                "join_mixed_size_data",
                "mixed_keyed_residual_or_conditional_full_retained_size_data",
                ContractQuality::ObservedRetainedState,
            )]
        },
        PlanNode::Join(join) if join.spec.residual_predicate.is_some() => vec![ReservationRequirement::modeled(
            ReservationKind::MixedSizeData,
            "join_mixed_size_data",
            "mixed_keyed_residual_or_conditional_full_retained_size_data",
            ContractQuality::ObservedRetainedState,
        )],
        PlanNode::Union(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "union_concat_output",
            "union_concat_output_size_preflight",
            ContractQuality::ExactPreflight,
        )],
        PlanNode::EdgeNormalize(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "edge_normalize_output",
            "edge_normalize_output_size_preflight",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        )],
        PlanNode::GraphAlgorithm(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            crate::plan::GRAPH_ALGORITHM_CAPABILITY_MEMORY_BOUNDARY,
            crate::plan::GRAPH_ALGORITHM_CAPABILITY_REQUIRED_CONTRACT,
            ContractQuality::CapabilityProof,
        )],
        PlanNode::VectorAlgorithm(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            crate::plan::VECTOR_ALGORITHM_BOUNDED_MEMORY_BOUNDARY,
            crate::plan::VECTOR_ALGORITHM_BOUNDED_REQUIRED_CONTRACT,
            ContractQuality::CapabilityProof,
        )],
        _ => Vec::new(),
    }
}

fn projection_is_scalar_setop_final_projection(plan: &QueryPlan, node_id: NodeId) -> bool {
    projection_scalar_setop_shape_decision(plan).is_ok_and(|shape| shape.final_projection_id == node_id)
}

fn final_topk_sort_before_limit(plan: &QueryPlan, limit_input: NodeId) -> Option<NodeId> {
    let mut current = limit_input;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        match plan.node(current).map(|node| &node.kind)? {
            PlanNode::Sort(_) => return Some(current),
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_) => return None,
        }
    }
}

fn input_is_rank_window(plan: &QueryPlan, input: NodeId) -> bool {
    let Some(PlanNode::Window(window)) = plan.node(input).map(|node| &node.kind) else {
        return false;
    };
    window.spec.functions.iter().any(|function| matches!(function, WindowFunction::Rank { .. }))
}

fn regexp_replace_requirements() -> Vec<ReservationRequirement> {
    vec![ReservationRequirement::modeled(
        ReservationKind::OperatorOutput,
        "regexp_replace_expression_output",
        cudf::REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT,
        ContractQuality::ExactPreflight,
    )]
}

fn projection_contains_regexp_replace(projection: &crate::plan::ProjectionNode) -> bool {
    projection.outputs.iter().any(|output| output.expr.contains_regexp_replace())
}
