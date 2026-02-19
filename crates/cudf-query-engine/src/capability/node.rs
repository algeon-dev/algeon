use super::{expr::*, model::*, *};
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
        PlanNode::GraphAlgorithm(graph) => NodeOperator::GraphAlgorithm { algorithm: graph.spec.algorithm },
    }
}

pub(super) fn source_format_capability(source: &SourceNode) -> SourceFormatCapability {
    match source {
        SourceNode::LocalParquet(_) => SourceFormatCapability::LocalParquet,
        SourceNode::GpuResident(_) => SourceFormatCapability::GpuResident,
    }
}

fn node_executable_decision(plan: &QueryPlan, node: &QueryNode) -> CapabilityDecision {
    match &node.kind {
        PlanNode::Source(source) => source_node_executable_decision(source),
        PlanNode::Filter(filter) => expression_node_decision(
            node.id,
            "unsupported_filter_expression_capability",
            std::iter::once(&filter.predicate),
        ),
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
        PlanNode::Aggregate(aggregate) => aggregate_node_executable_decision(node.id, aggregate),
        PlanNode::Window(window) => window_node_executable_decision(node.id, window),
        PlanNode::Sort(sort) => sort_node_executable_decision(node.id, sort),
        PlanNode::Limit(limit) => limit_node_executable_decision(plan, node.id, limit),
        PlanNode::Join(join) => join_node_executable_decision(plan, node.id, join),
        PlanNode::Union(union) => union_node_executable_decision(plan, node.id, union),
        PlanNode::GraphAlgorithm(graph) => graph_algorithm_node_executable_decision(plan, node.id, graph),
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

fn aggregate_node_executable_decision(node_id: NodeId, aggregate: &crate::plan::AggregateNode) -> CapabilityDecision {
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
    }
    CapabilityDecision::Satisfied
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
    if spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::StddevSamp) {
        return Err("grouping_sets_stddev_samp_not_supported");
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
    if limit.spec.skip.checked_add(limit.spec.fetch).is_none() {
        return CapabilityDecision::rejected(
            "unsupported_limit_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "skip_fetch_overflow".to_string()),
                ("skip", limit.spec.skip.to_string()),
                ("fetch", limit.spec.fetch.to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

fn join_node_executable_decision(plan: &QueryPlan, node_id: NodeId, join: &JoinNode) -> CapabilityDecision {
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
    if let Some(KeyedHashJoinStrategyDecision::Rejected { reason, facts }) =
        keyed_hash_join_strategy_decision_for_plan(plan, node_id, join)
    {
        let mut facts = facts;
        facts.insert("node_id", node_id.to_string());
        return CapabilityDecision::Rejected { reason, facts };
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
        if field.is_nullable() {
            return CapabilityDecision::rejected(
                "unsupported_graph_algorithm_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "nullable_edge_column".to_string()),
                    ("edge_column_role", role.to_string()),
                    ("column", column.to_string()),
                ],
            );
        }
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
        if field.is_nullable() {
            return CapabilityDecision::rejected(
                "unsupported_graph_algorithm_node_capability",
                [
                    ("node_id", node_id.to_string()),
                    ("reason", "nullable_edge_column".to_string()),
                    ("edge_column_role", "weight".to_string()),
                    ("column", weight_col.to_string()),
                ],
            );
        }
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

    if graph.output_schema().as_ref() != crate::plan::stable_graph_result_schema(graph.spec.algorithm).as_ref() {
        return CapabilityDecision::rejected(
            "unsupported_graph_algorithm_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "non_stable_output_schema".to_string()),
                ("algorithm", graph.spec.algorithm.as_str().to_string()),
                ("schema_field_count", graph.output_schema().fields().len().to_string()),
            ],
        );
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
        PlanNode::Aggregate(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "operator_output",
            "aggregate_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        )],
        PlanNode::Window(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "operator_output",
            "window_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        )],
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
            Vec::new()
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
        PlanNode::GraphAlgorithm(_) => vec![ReservationRequirement::blocked_by_lower_level(
            ReservationKind::OperatorOutput,
            crate::plan::GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY,
            crate::plan::GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT,
            ContractQuality::ExactPreflight,
        )],
        _ => Vec::new(),
    }
}
