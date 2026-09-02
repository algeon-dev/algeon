use std::collections::BTreeSet;

use super::{
    AggregateNode, AggregateSpec, CapabilityDecision, Expr, FinalRollupWidthClass, JoinNode, JoinSide, JoinType,
    NodeId, PlanNode, QueryPlan, RollupRankTopKAfterAggregateJoinPushdownShape, SourceNode, WindowFunction,
    collect_reachable_node_ids_from, join_node_with_missing_reason, node_kind_for_fact, node_operator_fact,
    peel_projection_chain_ids_for_shape, sort_input_with_missing_reason, validate_bounded_join_subtree_for_capability,
};

pub(in crate::capability) fn rollup_rank_topk_after_aggregate_join_pushdown_shape_decision(
    plan: &QueryPlan,
) -> Result<RollupRankTopKAfterAggregateJoinPushdownShape, CapabilityDecision> {
    let rank_path = rank_window_path_from_sink(plan)?;
    let rank_window_id = rank_path.rank_window_id;
    require_rank_window(plan, rank_window_id)?;
    let rank_sort_id = window_input_sort(plan, rank_window_id)?;
    let rank_sort_input_id = sort_input_with_missing_reason(
        plan,
        rank_sort_id,
        "rollup_rank_topk_after_pushdown_missing_node",
        "rollup_rank_topk_after_pushdown_requires_rank_input_sort",
        "rank_input_sort",
    )?;
    let (final_grouping_aggregate_id, _) = peel_projection_chain_ids_for_shape(plan, rank_sort_input_id)?;
    let final_grouping_aggregate = grouping_sets_aggregate(
        plan,
        final_grouping_aggregate_id,
        "rollup_rank_topk_after_pushdown_requires_final_grouping_set_aggregate",
    )?;
    let final_rollup_width =
        require_supported_final_rollup_width(final_grouping_aggregate_id, final_grouping_aggregate)?;
    require_plain_column_aggregate_inputs(final_grouping_aggregate_id, &final_grouping_aggregate.spec)?;

    let (root_join_id, _) = peel_projection_chain_ids_for_shape(plan, final_grouping_aggregate.input)?;
    let root_join = join_node_with_missing_reason(
        plan,
        root_join_id,
        "rollup_rank_topk_after_pushdown_missing_node",
        "rollup_rank_topk_after_pushdown_requires_root_join_below_final_aggregate",
    )?;
    require_supported_rebuilt_rollup_join(root_join_id, root_join)?;

    let detail_pre_aggregate_input = detail_pre_aggregate_input(plan, root_join)?;
    let detail_pre_aggregate = grouped_non_grouping_sets_aggregate(
        plan,
        detail_pre_aggregate_input.node_id,
        "rollup_rank_topk_after_pushdown_requires_detail_pre_aggregate",
    )?;
    require_plain_column_aggregate_inputs(detail_pre_aggregate_input.node_id, &detail_pre_aggregate.spec)?;
    require_source_reduction_proof(
        plan,
        final_grouping_aggregate_id,
        root_join,
        detail_pre_aggregate_input,
        detail_pre_aggregate,
    )?;

    Ok(RollupRankTopKAfterAggregateJoinPushdownShape {
        detail_pre_aggregate_id: detail_pre_aggregate_input.node_id,
        root_join_id,
        final_grouping_aggregate_id,
        final_rollup_width,
        rank_sort_id,
        rank_window_id,
        rank_filter_id: rank_path.rank_filter_id,
        rank_output_residency_consumer_id: rank_path.rank_output_residency_consumer_id,
        final_sort_id: rank_path.final_sort_id,
        final_limit_id: rank_path.final_limit_id,
    })
}

#[derive(Debug, Clone, Copy)]
struct RankWindowPath {
    rank_window_id: NodeId,
    rank_filter_id: Option<NodeId>,
    rank_output_residency_consumer_id: Option<NodeId>,
    final_sort_id: Option<NodeId>,
    final_limit_id: Option<NodeId>,
}

fn rank_window_path_from_sink(plan: &QueryPlan) -> Result<RankWindowPath, CapabilityDecision> {
    let sink_input = plan.sink().input();
    let Some(node) = plan.node(sink_input) else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_missing_node",
            [("node_id", sink_input.to_string()), ("role", "sink_input".to_string())],
        ));
    };
    let (post_limit_id, final_limit_id) = match &node.kind {
        PlanNode::Limit(limit) => {
            let (final_sort_id, _) = peel_projection_chain_ids_for_shape(plan, limit.input)?;
            if !matches!(plan.node(final_sort_id).map(|node| &node.kind), Some(PlanNode::Sort(_))) {
                return Err(CapabilityDecision::rejected(
                    "rollup_rank_topk_after_pushdown_requires_final_sort_before_limit",
                    [("node_id", final_sort_id.to_string()), ("operator", node_kind_for_fact(plan, final_sort_id))],
                ));
            }
            (final_sort_id, Some(sink_input))
        },
        _ => (sink_input, None),
    };
    let (post_limit_id, _) = peel_projection_chain_ids_for_shape(plan, post_limit_id)?;
    let Some(post_limit_node) = plan.node(post_limit_id) else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_missing_node",
            [("node_id", post_limit_id.to_string()), ("role", "post_limit".to_string())],
        ));
    };
    let (rank_consumer_input_id, final_sort_id) = match &post_limit_node.kind {
        PlanNode::Sort(sort) => (sort.input, Some(post_limit_id)),
        PlanNode::Filter(_) | PlanNode::Window(_) => (post_limit_id, None),
        _ => Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_rank_filter_or_final_topk_sink",
            [
                ("node_id", post_limit_id.to_string()),
                ("operator", node_operator_fact(&post_limit_node.kind).to_string()),
            ],
        ))?,
    };

    let (rank_consumer_id, rank_consumer_projections) =
        peel_projection_chain_ids_for_shape(plan, rank_consumer_input_id)?;
    let Some(rank_consumer) = plan.node(rank_consumer_id) else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_missing_node",
            [("node_id", rank_consumer_id.to_string()), ("role", "rank_consumer".to_string())],
        ));
    };
    match &rank_consumer.kind {
        PlanNode::Filter(filter) => {
            let (rank_window_id, _) = peel_projection_chain_ids_for_shape(plan, filter.input)?;
            let rank_output_residency_consumer_id =
                rank_consumer_projections.first().copied().or(Some(rank_consumer_id));
            Ok(RankWindowPath {
                rank_window_id,
                rank_filter_id: Some(rank_consumer_id),
                rank_output_residency_consumer_id,
                final_sort_id,
                final_limit_id,
            })
        },
        PlanNode::Window(_) => Ok(RankWindowPath {
            rank_window_id: rank_consumer_id,
            rank_filter_id: None,
            rank_output_residency_consumer_id: rank_consumer_projections.first().copied(),
            final_sort_id,
            final_limit_id,
        }),
        _ => Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_rank_filter_or_rank_window",
            [
                ("node_id", rank_consumer_id.to_string()),
                ("operator", node_operator_fact(&rank_consumer.kind).to_string()),
            ],
        )),
    }
}

fn window_input_sort(plan: &QueryPlan, node_id: NodeId) -> Result<NodeId, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_missing_node",
            [("node_id", node_id.to_string()), ("role", "rank_window".to_string())],
        ));
    };
    let PlanNode::Window(window) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_rank_window",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    let (rank_sort_id, _) = peel_projection_chain_ids_for_shape(plan, window.input)?;
    Ok(rank_sort_id)
}

fn require_rank_window(plan: &QueryPlan, node_id: NodeId) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_missing_node",
            [("node_id", node_id.to_string()), ("role", "rank_window".to_string())],
        ));
    };
    let PlanNode::Window(window) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_rank_window",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    let rank_count =
        window.spec.functions.iter().filter(|function| matches!(function, WindowFunction::Rank { .. })).count();
    if rank_count == 0 {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_rank_window",
            [("node_id", node_id.to_string()), ("rank_function_count", "0".to_string())],
        ));
    }
    Ok(())
}

fn grouping_sets_aggregate<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<&'a AggregateNode, CapabilityDecision> {
    let aggregate = aggregate_node(plan, node_id, reason)?;
    if aggregate.spec.grouping_sets.is_none() {
        return Err(CapabilityDecision::rejected(
            reason,
            [("node_id", node_id.to_string()), ("grouping_sets_present", "false".to_string())],
        ));
    }
    Ok(aggregate)
}

fn grouped_non_grouping_sets_aggregate<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<&'a AggregateNode, CapabilityDecision> {
    let aggregate = aggregate_node(plan, node_id, reason)?;
    if aggregate.spec.grouping_sets.is_some() || aggregate.spec.group_exprs.is_empty() {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("node_id", node_id.to_string()),
                ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                ("grouping_sets_present", aggregate.spec.grouping_sets.is_some().to_string()),
            ],
        ));
    }
    Ok(aggregate)
}

fn require_supported_final_rollup_width(
    aggregate_id: NodeId,
    aggregate: &AggregateNode,
) -> Result<FinalRollupWidthClass, CapabilityDecision> {
    let grouping_set_count = aggregate.spec.grouping_sets.as_ref().map_or(0, |sets| sets.groups.len());
    if aggregate.spec.group_exprs.len() <= 3 && grouping_set_count <= 4 {
        return Ok(FinalRollupWidthClass::Narrow);
    }
    if aggregate.spec.group_exprs.len() == 8
        && grouping_set_count == 9
        && aggregate
            .spec
            .grouping_sets
            .as_ref()
            .is_some_and(|sets| grouping_sets_are_suffix_rollup_masks(aggregate.spec.group_exprs.len(), &sets.groups))
    {
        return Ok(FinalRollupWidthClass::Wide);
    }
    Err(CapabilityDecision::rejected(
        "rollup_rank_topk_after_pushdown_requires_bounded_final_rollup_width",
        [
            ("node_id", aggregate_id.to_string()),
            ("group_expr_count", aggregate.spec.group_exprs.len().to_string()),
            ("grouping_set_count", grouping_set_count.to_string()),
            ("max_group_expr_count", "3".to_string()),
            ("max_grouping_set_count", "4".to_string()),
            ("modeled_wide_group_expr_count", "8".to_string()),
            ("modeled_wide_grouping_set_count", "9".to_string()),
            ("modeled_wide_grouping_mask_contract", "suffix_rollup".to_string()),
        ],
    ))
}

fn grouping_sets_are_suffix_rollup_masks(group_expr_count: usize, groups: &[Vec<bool>]) -> bool {
    if groups.len() != group_expr_count + 1 {
        return false;
    }
    let actual = groups.iter().cloned().collect::<BTreeSet<_>>();
    let expected = (0..=group_expr_count)
        .map(|rolled_count| (0..group_expr_count).map(|idx| idx >= group_expr_count - rolled_count).collect::<Vec<_>>())
        .collect::<BTreeSet<_>>();
    actual == expected
}

fn aggregate_node<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    reason: &'static str,
) -> Result<&'a AggregateNode, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    let PlanNode::Aggregate(aggregate) = &node.kind else {
        return Err(CapabilityDecision::rejected(
            reason,
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        ));
    };
    Ok(aggregate)
}

fn require_plain_column_aggregate_inputs(aggregate_id: NodeId, spec: &AggregateSpec) -> Result<(), CapabilityDecision> {
    for expr in &spec.group_exprs {
        if !expr_is_plain_column_reference(expr) {
            return Err(CapabilityDecision::rejected(
                "rollup_rank_topk_after_pushdown_requires_plain_column_group_keys",
                [("node_id", aggregate_id.to_string()), ("expression", expr.to_string())],
            ));
        }
    }
    for aggregate_expr in &spec.aggr_exprs {
        if !expr_is_plain_column_reference(&aggregate_expr.expr) {
            return Err(CapabilityDecision::rejected(
                "rollup_rank_topk_after_pushdown_requires_plain_column_aggregate_inputs",
                [("node_id", aggregate_id.to_string()), ("expression", aggregate_expr.expr.to_string())],
            ));
        }
    }
    Ok(())
}

fn expr_is_plain_column_reference(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) => true,
        Expr::Alias { expr, .. } => expr_is_plain_column_reference(expr),
        _ => false,
    }
}

fn require_supported_rebuilt_rollup_join(join_id: NodeId, join: &JoinNode) -> Result<(), CapabilityDecision> {
    let supported = match join.spec.join_type {
        JoinType::Inner => {
            !join.spec.keys.is_empty()
                && join.spec.residual_predicate.is_none()
                && !join.spec.output_projection.is_empty()
        },
        JoinType::LeftSemi => {
            join.spec.build_side == JoinSide::Right
                && !join.spec.keys.is_empty()
                && join.spec.residual_predicate.is_none()
                && !join.spec.output_projection.is_empty()
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
        },
        _ => false,
    };
    if supported {
        return Ok(());
    }
    Err(CapabilityDecision::rejected(
        "rollup_rank_topk_after_pushdown_requires_supported_rebuilt_join",
        [
            ("node_id", join_id.to_string()),
            ("join_type", join.spec.join_type.to_string()),
            ("key_count", join.spec.keys.len().to_string()),
            ("has_residual", join.spec.residual_predicate.is_some().to_string()),
            ("build_side", join.spec.build_side.to_string()),
            ("output_projection_count", join.spec.output_projection.len().to_string()),
        ],
    ))
}

#[derive(Debug, Clone, Copy)]
struct DetailPreAggregateInput {
    node_id: NodeId,
    side: JoinSide,
}

fn detail_pre_aggregate_input(
    plan: &QueryPlan,
    join: &JoinNode,
) -> Result<DetailPreAggregateInput, CapabilityDecision> {
    [(join.inputs[0], JoinSide::Left), (join.inputs[1], JoinSide::Right)]
        .into_iter()
        .find_map(|(input, side)| {
            peel_detail_pre_aggregate(plan, input).ok().map(|node_id| DetailPreAggregateInput { node_id, side })
        })
        .ok_or_else(|| {
            CapabilityDecision::rejected(
                "rollup_rank_topk_after_pushdown_requires_detail_pre_aggregate",
                [("join_input_count", join.inputs.len().to_string())],
            )
        })
}

fn peel_detail_pre_aggregate(plan: &QueryPlan, node_id: NodeId) -> Result<NodeId, CapabilityDecision> {
    let (peeled, _) = peel_projection_chain_ids_for_shape(plan, node_id)?;
    let aggregate = grouped_non_grouping_sets_aggregate(
        plan,
        peeled,
        "rollup_rank_topk_after_pushdown_requires_detail_pre_aggregate",
    )?;
    validate_pre_aggregate_leaf(plan, aggregate.input)?;
    Ok(peeled)
}

fn validate_pre_aggregate_leaf(plan: &QueryPlan, node_id: NodeId) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_missing_pre_aggregate_leaf",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)) => Ok(()),
        PlanNode::Projection(projection) => validate_pre_aggregate_leaf(plan, projection.input),
        PlanNode::Filter(filter) => validate_pre_aggregate_leaf(plan, filter.input),
        PlanNode::Join(_) => {
            let mut join_count = 0;
            validate_bounded_join_subtree_for_capability(plan, node_id, &mut join_count)
        },
        _ => Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_source_reduction_leaf",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

fn require_source_reduction_proof(
    plan: &QueryPlan,
    final_grouping_aggregate_id: NodeId,
    root_join: &JoinNode,
    detail_input: DetailPreAggregateInput,
    detail_pre_aggregate: &AggregateNode,
) -> Result<(), CapabilityDecision> {
    let final_inputs = collect_reachable_node_ids_from(plan, final_grouping_aggregate_id)?;
    if !final_inputs.contains(&detail_input.node_id) {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_final_aggregate_consumes_pre_aggregate",
            [
                ("final_grouping_aggregate_id", final_grouping_aggregate_id.to_string()),
                ("detail_pre_aggregate_id", detail_input.node_id.to_string()),
            ],
        ));
    }
    prove_final_aggregate_measures_derive_from_pre_aggregate(
        plan,
        final_grouping_aggregate_id,
        root_join,
        detail_input,
        detail_pre_aggregate,
    )?;
    let detail_sources = collect_reachable_node_ids_from(plan, detail_pre_aggregate.input)?
        .into_iter()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Source(_))))
        .count();
    if detail_sources == 0 {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_source_reduction_proof",
            [("detail_pre_aggregate_id", detail_input.node_id.to_string())],
        ));
    }
    Ok(())
}

fn prove_final_aggregate_measures_derive_from_pre_aggregate(
    plan: &QueryPlan,
    final_grouping_aggregate_id: NodeId,
    root_join: &JoinNode,
    detail_input: DetailPreAggregateInput,
    detail_pre_aggregate: &AggregateNode,
) -> Result<(), CapabilityDecision> {
    let final_grouping_aggregate = grouping_sets_aggregate(
        plan,
        final_grouping_aggregate_id,
        "rollup_rank_topk_after_pushdown_requires_final_grouping_set_aggregate",
    )?;
    let pre_aggregate_aliases =
        detail_pre_aggregate.spec.aggr_exprs.iter().map(|expr| expr.alias.as_str()).collect::<BTreeSet<_>>();
    if pre_aggregate_aliases.is_empty() {
        return Err(CapabilityDecision::rejected(
            "rollup_rank_topk_after_pushdown_requires_pre_aggregate_measure_output",
            [("detail_pre_aggregate_id", detail_input.node_id.to_string())],
        ));
    }

    for final_expr in &final_grouping_aggregate.spec.aggr_exprs {
        let Expr::Column(final_column) = &final_expr.expr else {
            return Err(CapabilityDecision::rejected(
                "rollup_rank_topk_after_pushdown_requires_final_aggregate_measure_column",
                [
                    ("final_grouping_aggregate_id", final_grouping_aggregate_id.to_string()),
                    ("expression", final_expr.expr.to_string()),
                ],
            ));
        };
        let Some(join_output) =
            root_join.spec.output_projection.iter().find(|column| column.output_name == final_column.name())
        else {
            return Err(CapabilityDecision::rejected(
                "rollup_rank_topk_after_pushdown_requires_final_measure_from_root_join_output",
                [
                    ("final_grouping_aggregate_id", final_grouping_aggregate_id.to_string()),
                    ("measure_column", final_column.name().to_string()),
                ],
            ));
        };
        if join_output.side != detail_input.side || !pre_aggregate_aliases.contains(join_output.column.name()) {
            return Err(CapabilityDecision::rejected(
                "rollup_rank_topk_after_pushdown_requires_final_measure_from_pre_aggregate",
                [
                    ("final_grouping_aggregate_id", final_grouping_aggregate_id.to_string()),
                    ("detail_pre_aggregate_id", detail_input.node_id.to_string()),
                    ("measure_column", final_column.name().to_string()),
                    ("root_join_output_side", join_output.side.to_string()),
                    ("root_join_input_column", join_output.column.name().to_string()),
                ],
            ));
        }
    }
    Ok(())
}
