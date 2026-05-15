use super::{expr::*, model::*, *};

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
            grouping: if aggregate.spec.group_exprs.is_empty() {
                AggregateGrouping::Scalar
            } else {
                AggregateGrouping::Grouped
            },
            functions: aggregate.spec.aggr_exprs.iter().map(|expr| expr.function).collect(),
        },
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
        PlanNode::Sort(sort) => sort_node_executable_decision(node.id, sort),
        PlanNode::Limit(limit) => limit_node_executable_decision(plan, node.id, limit),
        PlanNode::Join(join) => join_node_executable_decision(plan, node.id, join),
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

fn limit_node_executable_decision(plan: &QueryPlan, node_id: NodeId, limit: &LimitNode) -> CapabilityDecision {
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
    if !matches!(plan.node(limit.input).map(|node| &node.kind), Some(PlanNode::Sort(_))) {
        return CapabilityDecision::rejected(
            "unsupported_limit_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "limit_not_directly_above_sort".to_string()),
                ("input", limit.input.to_string()),
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
    if let Some(predicate) = &join.spec.residual_predicate
        && let Err(rejection) = validate_join_residual_expr_capability(predicate)
    {
        return expression_rejection_decision(
            node_id,
            "unsupported_join_residual_expression_capability",
            predicate,
            rejection,
        );
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
        _ => Vec::new(),
    }
}
