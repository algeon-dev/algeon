use super::*;
use crate::{
    dynamic_filter::{DynamicFilterConsumer, DynamicFilterConsumerReductionStage},
    plan::{WindowFrame, WindowNode},
    source::{LocalParquetSource, PredicateDisposition},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootTopKJoinSubcontract {
    KeyedHash,
    KeyedResidualMixed,
    KeyedFilteredSemi,
    KeylessConditional,
    KeylessOuterCrossWindow,
    ProjectedCross,
}

pub(in crate::capability) fn root_topk_join_aggregate_shape_decision(
    plan: &QueryPlan,
) -> Result<RootTopKJoinAggregateShape, CapabilityDecision> {
    let final_limit_id = plan.sink().input();
    let (final_sort_id, final_topk_projection_ids) = final_topk_sort_and_projection_chain(plan, final_limit_id)?;
    let fetch = final_topk_fetch(plan, final_limit_id)?;
    let post_root_input_id = sort_input_with_missing_reason(
        plan,
        final_sort_id,
        "root_topk_join_aggregate_missing_node",
        "root_topk_join_aggregate_requires_final_sort",
        "final_sort",
    )?;
    let Some((post_root_candidate_id, post_root_projection_ids)) =
        peel_resident_unary_chain_for_capability(plan, post_root_input_id)
    else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_unary_chain_before_sort",
            [("node_id", post_root_input_id.to_string())],
        ));
    };
    let root_reachable = collect_reachable_node_ids_from(plan, post_root_candidate_id)?;
    let has_post_root_projection = post_root_projection_ids
        .iter()
        .any(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Projection(_))));
    let direct_sort_fetch_join_input_candidate = !has_post_root_projection
        && matches!(plan.node(post_root_candidate_id).map(|node| &node.kind), Some(PlanNode::Join(_)));
    let direct_sort_fetch_aggregate_candidate = !has_post_root_projection
        && matches!(
            plan.node(post_root_candidate_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
        );
    if !has_post_root_projection && !direct_sort_fetch_join_input_candidate && !direct_sort_fetch_aggregate_candidate {
        if let Some(rejection) = unmodeled_sort_fetch_hash_subcontract_rejection(
            plan,
            &root_reachable,
            post_root_input_id,
            post_root_candidate_id,
            final_sort_id,
            final_limit_id,
            fetch,
        )? {
            return Err(rejection);
        }
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_projection_before_sort",
            [("node_id", post_root_input_id.to_string()), ("operator", node_kind_for_fact(plan, post_root_input_id))],
        ));
    }
    let whole_query_reachable = collect_reachable_node_ids(plan)?;
    let shared_producer_ids = root_topk_shared_computed_producer_ids(plan, &whole_query_reachable)?;
    let window_ids = root_topk_window_ids(plan, &root_reachable)?;
    let distinct_aggregate_ids = root_topk_distinct_aggregate_ids(plan, &root_reachable, window_ids.is_empty())?;
    let window_input_sort_ids = root_topk_window_input_sort_ids(plan, &window_ids)?;
    let join_ids = root_reachable
        .iter()
        .copied()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
        .collect::<Vec<_>>();
    let required_join_count = if direct_sort_fetch_aggregate_candidate { 1 } else { 2 };
    if join_ids.len() < required_join_count {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_join_fragment",
            [
                ("reachable_join_count", join_ids.len().to_string()),
                ("required_join_count", required_join_count.to_string()),
            ],
        ));
    }
    let root_join_id = post_root_root_join_id(plan, post_root_candidate_id, &join_ids)?;
    let mut keyless_or_cross_join_ids = Vec::new();
    let mut keyed_filtered_semi_join_ids = Vec::new();
    let mut left_output_bounded_join_ids = Vec::new();
    let mut window_full_outer_join_ids = Vec::new();
    let mut keyless_outer_cross_window_join_ids = Vec::new();
    let mut sort_fetch_hash_join_input_subcontract = false;
    let sort_fetch_hash_join_aggregate_subcontract = direct_sort_fetch_aggregate_candidate;
    for join_id in &join_ids {
        let join = join_node_with_missing_reason(
            plan,
            *join_id,
            "root_topk_join_aggregate_missing_node",
            "root_topk_join_aggregate_requires_keyed_join_fragment",
        )?;
        let allow_distinct_dedup_keyed_outer =
            root_topk_join_has_distinct_or_dedup_outer_context(plan, join, &distinct_aggregate_ids)?;
        let allow_window_keyed_full_outer = root_topk_join_has_window_full_outer_context(plan, join, &window_ids)?;
        let allow_keyless_outer_cross_window =
            root_topk_keyless_outer_cross_feeds_ordered_window(plan, *join_id, &window_ids, &window_input_sort_ids)?;
        let allow_aggregate_fan_in_left =
            root_topk_join_has_aggregate_fan_in_left_context(plan, join, &root_reachable)?;
        let allow_left_anti_grouped_aggregate_input = join.spec.join_type == JoinType::LeftAnti
            && root_topk_join_feeds_grouped_aggregate(plan, *join_id, &root_reachable)?;
        let allow_sort_fetch_aggregate_keyed_left = direct_sort_fetch_aggregate_candidate
            && join.spec.join_type == JoinType::Left
            && join.spec.build_side == JoinSide::Right;
        let allow_left_output_bounded_join = (allow_aggregate_fan_in_left || allow_left_anti_grouped_aggregate_input)
            && !allow_distinct_dedup_keyed_outer
            && !allow_window_keyed_full_outer
            && !allow_sort_fetch_aggregate_keyed_left;
        match require_supported_root_topk_join(
            *join_id,
            join,
            allow_distinct_dedup_keyed_outer,
            allow_window_keyed_full_outer,
            allow_keyless_outer_cross_window,
            allow_sort_fetch_aggregate_keyed_left,
            allow_left_output_bounded_join,
        )? {
            RootTopKJoinSubcontract::KeylessConditional | RootTopKJoinSubcontract::ProjectedCross => {
                keyless_or_cross_join_ids.push(*join_id);
            },
            RootTopKJoinSubcontract::KeylessOuterCrossWindow => {
                keyless_outer_cross_window_join_ids.push(*join_id);
            },
            RootTopKJoinSubcontract::KeyedFilteredSemi => {
                keyed_filtered_semi_join_ids.push(*join_id);
                if join.spec.join_type == JoinType::LeftAnti {
                    left_output_bounded_join_ids.push(*join_id);
                }
            },
            RootTopKJoinSubcontract::KeyedHash | RootTopKJoinSubcontract::KeyedResidualMixed => {
                if allow_left_output_bounded_join && join.spec.join_type == JoinType::Left {
                    left_output_bounded_join_ids.push(*join_id);
                }
                if direct_sort_fetch_join_input_candidate && *join_id == post_root_candidate_id {
                    sort_fetch_hash_join_input_subcontract = true;
                }
            },
        }
        if allow_window_keyed_full_outer {
            window_full_outer_join_ids.push(*join_id);
        }
    }
    keyless_or_cross_join_ids.sort_unstable();
    keyless_or_cross_join_ids.dedup();
    keyed_filtered_semi_join_ids.sort_unstable();
    keyed_filtered_semi_join_ids.dedup();
    left_output_bounded_join_ids.sort_unstable();
    left_output_bounded_join_ids.dedup();
    window_full_outer_join_ids.sort_unstable();
    window_full_outer_join_ids.dedup();
    keyless_outer_cross_window_join_ids.sort_unstable();
    keyless_outer_cross_window_join_ids.dedup();

    let (aggregate_id, aggregate_projection_id) =
        if matches!(plan.node(post_root_candidate_id).map(|node| &node.kind), Some(PlanNode::Aggregate(_))) {
            let Some(PlanNode::Aggregate(aggregate)) = plan.node(post_root_candidate_id).map(|node| &node.kind) else {
                unreachable!("post-root candidate was already checked as aggregate")
            };
            if aggregate.spec.group_exprs.is_empty() {
                return Err(CapabilityDecision::rejected(
                    "root_topk_join_aggregate_requires_grouped_aggregate_fragment",
                    [("aggregate_id", post_root_candidate_id.to_string())],
                ));
            }
            (post_root_candidate_id, post_root_input_id)
        } else {
            aggregate_projection_in_reachable_fragment(
                plan,
                &root_reachable,
                &keyless_or_cross_join_ids,
                !keyless_outer_cross_window_join_ids.is_empty(),
                window_ids.is_empty(),
            )?
        };
    let aggregate_input_reachable = aggregate_input_reachable_nodes(plan, aggregate_id)?;
    if root_topk_aggregate_requires_source_reduction(
        plan,
        aggregate_id,
        aggregate_projection_id,
        &keyless_or_cross_join_ids,
    )? {
        require_root_topk_source_reduction_proof(
            plan,
            &aggregate_input_reachable,
            aggregate_id,
            direct_sort_fetch_aggregate_candidate,
        )?;
    }
    require_root_topk_keyless_grouped_aggregate_source_reduction(plan, &keyless_or_cross_join_ids)?;
    if !left_output_bounded_join_ids.is_empty() {
        require_root_topk_all_joined_grouped_aggregates_source_reduction(
            plan,
            &root_reachable,
            &left_output_bounded_join_ids.iter().copied().collect(),
        )?;
    }

    Ok(RootTopKJoinAggregateShape {
        root_join_id,
        join_ids,
        distinct_aggregate_ids,
        keyless_or_cross_join_ids,
        keyed_filtered_semi_join_ids,
        null_filtered_left_join_ids: Vec::new(),
        window_ids,
        window_input_sort_ids,
        window_full_outer_join_ids,
        keyless_outer_cross_window_join_ids,
        shared_producer_ids,
        sort_fetch_hash_join_input_subcontract,
        sort_fetch_hash_join_aggregate_subcontract,
        aggregate_id,
        aggregate_projection_id,
        post_root_projection_ids,
        final_topk_projection_ids,
        final_sort_id,
        final_limit_id,
    })
}

fn root_topk_shared_computed_producer_ids(
    plan: &QueryPlan,
    reachable: &BTreeSet<NodeId>,
) -> Result<Vec<NodeId>, CapabilityDecision> {
    let mut shared_producers = Vec::new();
    let consumers = consumer_nodes_for_reachable(plan, reachable);
    for (producer, producer_consumers) in consumers {
        if producer_consumers.len() <= 1 {
            continue;
        }
        if matches!(plan.node(producer).map(|node| &node.kind), Some(PlanNode::Source(_))) {
            continue;
        }
        if plan.node(producer).is_none() {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", producer.to_string()), ("role", "shared_producer".to_string())],
            ));
        }
        shared_producers.push(producer);
    }
    shared_producers.sort_unstable();
    shared_producers.dedup();
    Ok(shared_producers)
}

fn unmodeled_sort_fetch_hash_subcontract_rejection(
    plan: &QueryPlan,
    root_reachable: &BTreeSet<NodeId>,
    post_root_input_id: NodeId,
    post_root_candidate_id: NodeId,
    final_sort_id: NodeId,
    final_limit_id: NodeId,
    fetch: usize,
) -> Result<Option<CapabilityDecision>, CapabilityDecision> {
    let join_ids = root_reachable
        .iter()
        .copied()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
        .collect::<Vec<_>>();
    let aggregate_ids = root_reachable
        .iter()
        .copied()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Aggregate(_))))
        .collect::<Vec<_>>();
    let grouped_aggregate_count = aggregate_ids
        .iter()
        .filter(|aggregate_id| {
            matches!(
                plan.node(**aggregate_id).map(|node| &node.kind),
                Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
            )
        })
        .count();

    match plan.node(post_root_candidate_id).map(|node| &node.kind) {
        Some(PlanNode::Join(_)) => Ok(None),
        Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty() => {
            let aggregate_input_reachable = collect_reachable_node_ids_from(plan, aggregate.input)?;
            let aggregate_input_join_count = aggregate_input_reachable
                .iter()
                .filter(|input_id| matches!(plan.node(**input_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
                .count();
            Ok(Some(CapabilityDecision::rejected(
                "root_topk_join_aggregate_rejects_sort_fetch_hash_join_aggregate_subcontract",
                [
                    ("subcontract", "sort_fetch_hash_join_aggregate".to_string()),
                    ("post_sort_input_id", post_root_input_id.to_string()),
                    ("post_sort_input_operator", node_kind_for_fact(plan, post_root_input_id)),
                    ("post_root_candidate_id", post_root_candidate_id.to_string()),
                    ("final_sort_id", final_sort_id.to_string()),
                    ("final_limit_id", final_limit_id.to_string()),
                    ("fetch", fetch.to_string()),
                    ("aggregate_id", post_root_candidate_id.to_string()),
                    ("aggregate_group_expr_count", aggregate.spec.group_exprs.len().to_string()),
                    ("aggregate_input_join_count", aggregate_input_join_count.to_string()),
                    ("join_count", join_ids.len().to_string()),
                    ("aggregate_count", aggregate_ids.len().to_string()),
                    ("grouped_aggregate_count", grouped_aggregate_count.to_string()),
                    ("dynamic_filter_route_count", plan.dynamic_filter_routes().len().to_string()),
                ],
            )))
        },
        _ => Ok(None),
    }
}

fn post_root_root_join_id(
    plan: &QueryPlan,
    post_root_candidate_id: NodeId,
    join_ids: &[NodeId],
) -> Result<NodeId, CapabilityDecision> {
    if matches!(plan.node(post_root_candidate_id).map(|node| &node.kind), Some(PlanNode::Join(_))) {
        return Ok(post_root_candidate_id);
    }
    join_ids.last().copied().ok_or_else(|| {
        CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_join_fragment",
            [("post_root_candidate_id", post_root_candidate_id.to_string())],
        )
    })
}

fn root_topk_join_has_distinct_or_dedup_outer_context(
    plan: &QueryPlan,
    join: &JoinNode,
    distinct_aggregate_ids: &[NodeId],
) -> Result<bool, CapabilityDecision> {
    if distinct_aggregate_ids.is_empty() {
        return Ok(false);
    }
    if join.spec.join_type != JoinType::Left || join.spec.build_side != JoinSide::Right {
        return Ok(false);
    }
    let left_reachable = collect_reachable_node_ids_from(plan, join.inputs[0])?;
    let right_reachable = collect_reachable_node_ids_from(plan, join.inputs[1])?;
    let left_has_distinct = distinct_aggregate_ids.iter().any(|aggregate_id| left_reachable.contains(aggregate_id));
    let right_has_distinct = distinct_aggregate_ids.iter().any(|aggregate_id| right_reachable.contains(aggregate_id));
    Ok(right_has_distinct
        || (left_has_distinct
            && root_topk_reachable_contains_non_distinct_grouped_aggregate(
                plan,
                &right_reachable,
                distinct_aggregate_ids,
            )))
}

fn root_topk_join_has_window_full_outer_context(
    plan: &QueryPlan,
    join: &JoinNode,
    window_ids: &[NodeId],
) -> Result<bool, CapabilityDecision> {
    if window_ids.is_empty() || join.spec.join_type != JoinType::Full {
        return Ok(false);
    }
    let retains_left = join.spec.output_projection.iter().any(|column| column.side == JoinSide::Left);
    let retains_right = join.spec.output_projection.iter().any(|column| column.side == JoinSide::Right);
    if !retains_left || !retains_right {
        return Ok(false);
    }
    Ok(root_topk_join_input_peels_to_window(plan, join.inputs[0], window_ids)?
        && root_topk_join_input_peels_to_window(plan, join.inputs[1], window_ids)?)
}

fn root_topk_join_has_aggregate_fan_in_left_context(
    plan: &QueryPlan,
    join: &JoinNode,
    root_reachable: &BTreeSet<NodeId>,
) -> Result<bool, CapabilityDecision> {
    if join.spec.join_type != JoinType::Left
        || join.spec.build_side != JoinSide::Right
        || join.spec.keys.is_empty()
        || join.spec.residual_predicate.is_some()
    {
        return Ok(false);
    }
    let left_reachable = collect_reachable_node_ids_from(plan, join.inputs[0])?;
    if !left_reachable.iter().any(|node_id| {
        root_reachable.contains(node_id)
            && matches!(
                plan.node(*node_id).map(|node| &node.kind),
                Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
            )
    }) {
        return Ok(false);
    }
    let Some(right_unique_keys) = grouped_aggregate_key_output_names_for_input(plan, join.inputs[1])? else {
        return Ok(false);
    };
    if right_unique_keys.len() != join.spec.keys.len() {
        return Ok(false);
    }
    Ok(join.spec.keys.iter().all(|key| {
        right_unique_keys.contains(key.right.name()) || right_unique_keys.contains(key.right.display_name())
    }))
}

fn root_topk_join_feeds_grouped_aggregate(
    plan: &QueryPlan,
    join_id: NodeId,
    root_reachable: &BTreeSet<NodeId>,
) -> Result<bool, CapabilityDecision> {
    for node_id in root_reachable {
        let Some(PlanNode::Aggregate(aggregate)) = plan.node(*node_id).map(|node| &node.kind) else {
            continue;
        };
        if aggregate.spec.group_exprs.is_empty() {
            continue;
        }
        if collect_reachable_node_ids_from(plan, aggregate.input)?.contains(&join_id) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn grouped_aggregate_key_output_names_for_input(
    plan: &QueryPlan,
    input: NodeId,
) -> Result<Option<BTreeSet<String>>, CapabilityDecision> {
    let Some((producer_id, chain_ids)) = peel_resident_unary_chain_for_capability(plan, input) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_missing_node",
            [("node_id", input.to_string()), ("role", "aggregate_fan_in_input".to_string())],
        ));
    };
    let Some(PlanNode::Aggregate(aggregate)) = plan.node(producer_id).map(|node| &node.kind) else {
        return Ok(None);
    };
    if aggregate.spec.group_exprs.is_empty() || aggregate.spec.grouping_sets.is_some() {
        return Ok(None);
    }
    let mut key_names = aggregate
        .spec
        .group_exprs
        .iter()
        .filter_map(expr_output_column_name)
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if key_names.len() != aggregate.spec.group_exprs.len() {
        return Ok(None);
    }
    for chain_id in chain_ids.iter().rev() {
        let Some(PlanNode::Projection(projection)) = plan.node(*chain_id).map(|node| &node.kind) else {
            continue;
        };
        let mut projected_names = BTreeSet::new();
        for output in &projection.outputs {
            if expr_output_column_name(&output.expr).is_some_and(|name| key_names.contains(name)) {
                projected_names.insert(output.field.name().to_string());
            }
        }
        key_names = projected_names;
        if key_names.is_empty() {
            return Ok(None);
        }
    }
    Ok(Some(key_names))
}

fn expr_output_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { expr, .. } => expr_output_column_name(expr),
        _ => None,
    }
}

fn root_topk_join_input_peels_to_window(
    plan: &QueryPlan,
    input: NodeId,
    window_ids: &[NodeId],
) -> Result<bool, CapabilityDecision> {
    let Some((producer_id, _chain_ids)) = peel_resident_unary_chain_for_capability(plan, input) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_missing_node",
            [("node_id", input.to_string()), ("role", "window_full_outer_join_input".to_string())],
        ));
    };
    Ok(window_ids.contains(&producer_id))
}

fn root_topk_keyless_outer_cross_feeds_ordered_window(
    plan: &QueryPlan,
    join_id: NodeId,
    window_ids: &[NodeId],
    window_input_sort_ids: &[NodeId],
) -> Result<bool, CapabilityDecision> {
    let ordered_window_ids = window_ids
        .iter()
        .copied()
        .filter(|window_id| {
            matches!(
                plan.node(*window_id).map(|node| &node.kind),
                Some(PlanNode::Window(window))
                    if !window.spec.order_by.is_empty() && window_input_sort_ids.contains(&window.input)
            )
        })
        .collect::<Vec<_>>();
    if ordered_window_ids.is_empty() {
        return Ok(false);
    }
    let reachable = collect_reachable_node_ids(plan)?;
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    root_topk_all_keyless_outer_cross_consumers_reach_window(
        plan,
        &consumers,
        join_id,
        &ordered_window_ids,
        window_input_sort_ids,
        &mut BTreeSet::new(),
    )
}

fn root_topk_all_keyless_outer_cross_consumers_reach_window(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    node_id: NodeId,
    window_ids: &[NodeId],
    window_input_sort_ids: &[NodeId],
    visited: &mut BTreeSet<NodeId>,
) -> Result<bool, CapabilityDecision> {
    if window_ids.contains(&node_id) {
        return Ok(true);
    }
    if !visited.insert(node_id) {
        return Err(CapabilityDecision::rejected(
            "root_topk_keyless_outer_cross_window_consumer_cycle",
            [("node_id", node_id.to_string())],
        ));
    }
    let Some(node_consumers) = consumers.get(&node_id).filter(|values| !values.is_empty()) else {
        visited.remove(&node_id);
        return Ok(false);
    };

    let mut reached_window = false;
    for consumer_id in node_consumers {
        if window_ids.contains(consumer_id) {
            reached_window = true;
            continue;
        }
        let Some(node) = plan.node(*consumer_id) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", consumer_id.to_string()), ("role", "keyless_outer_cross_window_consumer".to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Filter(_) | PlanNode::Projection(_) => {
                if root_topk_all_keyless_outer_cross_consumers_reach_window(
                    plan,
                    consumers,
                    *consumer_id,
                    window_ids,
                    window_input_sort_ids,
                    visited,
                )? {
                    reached_window = true;
                } else {
                    visited.remove(&node_id);
                    return Ok(false);
                }
            },
            PlanNode::Sort(_) if window_input_sort_ids.contains(consumer_id) => {
                if root_topk_all_keyless_outer_cross_consumers_reach_window(
                    plan,
                    consumers,
                    *consumer_id,
                    window_ids,
                    window_input_sort_ids,
                    visited,
                )? {
                    reached_window = true;
                } else {
                    visited.remove(&node_id);
                    return Ok(false);
                }
            },
            PlanNode::Source(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_)
            | PlanNode::Sort(_) => {
                visited.remove(&node_id);
                return Ok(false);
            },
        }
    }
    visited.remove(&node_id);
    Ok(reached_window)
}

fn root_topk_reachable_contains_non_distinct_grouped_aggregate(
    plan: &QueryPlan,
    reachable: &BTreeSet<NodeId>,
    distinct_aggregate_ids: &[NodeId],
) -> bool {
    reachable.iter().copied().any(|node_id| {
        if distinct_aggregate_ids.contains(&node_id) {
            return false;
        }
        matches!(
            plan.node(node_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
        )
    })
}

fn final_topk_sort_and_projection_chain(
    plan: &QueryPlan,
    final_limit_id: NodeId,
) -> Result<(NodeId, Vec<NodeId>), CapabilityDecision> {
    let Some(node) = plan.node(final_limit_id) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_missing_node",
            [("node_id", final_limit_id.to_string()), ("role", "final_limit".to_string())],
        ));
    };
    let PlanNode::Limit(limit) = &node.kind else {
        return Err(CapabilityDecision::Conditional {
            waiting_on: vec![ConstraintRef::node(final_limit_id, "root_topk_join_aggregate_requires_final_limit")],
        });
    };
    let mut current = limit.input;
    let mut projection_ids = Vec::new();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_final_topk_projection_cycle",
                [("node_id", current.to_string())],
            ));
        }
        let Some(node) = plan.node(current) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", current.to_string()), ("role", "final_sort".to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Sort(_) => return Ok((current, projection_ids)),
            PlanNode::Projection(projection) => {
                projection_ids.push(current);
                current = projection.input;
            },
            PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_) => {
                return Err(CapabilityDecision::rejected(
                    "root_topk_join_aggregate_requires_final_sort",
                    [
                        ("node_id", current.to_string()),
                        ("role", "final_sort".to_string()),
                        ("operator", node_operator_fact(&node.kind).to_string()),
                    ],
                ));
            },
        }
    }
}

fn final_topk_fetch(plan: &QueryPlan, final_limit_id: NodeId) -> Result<usize, CapabilityDecision> {
    let Some(PlanNode::Limit(limit)) = plan.node(final_limit_id).map(|node| &node.kind) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_final_limit",
            [("node_id", final_limit_id.to_string())],
        ));
    };
    limit.spec.fetch.ok_or_else(|| {
        CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_fetch_limit",
            [("node_id", final_limit_id.to_string()), ("skip", limit.spec.skip.to_string())],
        )
    })
}

fn require_supported_root_topk_join(
    join_id: NodeId,
    join: &JoinNode,
    allow_distinct_dedup_keyed_outer: bool,
    allow_window_keyed_full_outer: bool,
    allow_keyless_outer_cross_window: bool,
    allow_sort_fetch_aggregate_keyed_left: bool,
    allow_left_output_bounded_join: bool,
) -> Result<RootTopKJoinSubcontract, CapabilityDecision> {
    if join.spec.build_artifact_id.is_some() || join.spec.output_projection.is_empty() {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_supported_join_state",
            [
                ("node_id", join_id.to_string()),
                ("join_type", join.spec.join_type.to_string()),
                ("key_count", join.spec.keys.len().to_string()),
                ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                ("output_projection_count", join.spec.output_projection.len().to_string()),
                ("has_build_artifact", join.spec.build_artifact_id.is_some().to_string()),
            ],
        ));
    }

    match join_node_execution_capability(join) {
        Some(JoinExecutionCapability::KeyedHash)
            if !join.spec.keys.is_empty()
                && (join.spec.join_type == JoinType::Inner
                    || (allow_distinct_dedup_keyed_outer
                        && join.spec.join_type == JoinType::Left
                        && join.spec.build_side == JoinSide::Right)
                    || (allow_sort_fetch_aggregate_keyed_left
                        && join.spec.join_type == JoinType::Left
                        && join.spec.build_side == JoinSide::Right)
                    || (allow_left_output_bounded_join
                        && join.spec.join_type == JoinType::Left
                        && join.spec.build_side == JoinSide::Right)
                    || (allow_window_keyed_full_outer && join.spec.join_type == JoinType::Full)) =>
        {
            Ok(RootTopKJoinSubcontract::KeyedHash)
        },
        Some(JoinExecutionCapability::KeyedResidualMixed)
            if !join.spec.keys.is_empty()
                && (join.spec.join_type == JoinType::Inner
                    || (allow_distinct_dedup_keyed_outer
                        && join.spec.join_type == JoinType::Left
                        && join.spec.build_side == JoinSide::Right)
                    || (allow_sort_fetch_aggregate_keyed_left
                        && join.spec.join_type == JoinType::Left
                        && join.spec.build_side == JoinSide::Right)
                    || (allow_left_output_bounded_join
                        && join.spec.join_type == JoinType::Left
                        && join.spec.build_side == JoinSide::Right)
                    || (allow_window_keyed_full_outer && join.spec.join_type == JoinType::Full)) =>
        {
            Ok(RootTopKJoinSubcontract::KeyedResidualMixed)
        },
        Some(JoinExecutionCapability::KeyedFilteredSemiAnti)
            if (join.spec.join_type == JoinType::LeftSemi
                || (allow_left_output_bounded_join && join.spec.join_type == JoinType::LeftAnti))
                && join.spec.build_side == JoinSide::Right
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left) =>
        {
            Ok(RootTopKJoinSubcontract::KeyedFilteredSemi)
        },
        Some(JoinExecutionCapability::KeylessConditional(_)) => Ok(RootTopKJoinSubcontract::KeylessConditional),
        Some(JoinExecutionCapability::ProjectedCross) => Ok(RootTopKJoinSubcontract::ProjectedCross),
        Some(JoinExecutionCapability::KeylessOuterCross) if allow_keyless_outer_cross_window => {
            Ok(RootTopKJoinSubcontract::KeylessOuterCrossWindow)
        },
        Some(JoinExecutionCapability::KeylessOuterCross) => Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_rejects_keyless_outer_cross_subcontract",
            [
                ("node_id", join_id.to_string()),
                ("join_type", join.spec.join_type.to_string()),
                ("key_count", join.spec.keys.len().to_string()),
                ("has_residual", join.spec.residual_predicate.is_some().to_string()),
            ],
        )),
        _ => Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_keyed_hash_keyless_or_cross_join",
            [
                ("node_id", join_id.to_string()),
                ("join_type", join.spec.join_type.to_string()),
                ("key_count", join.spec.keys.len().to_string()),
                ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                ("output_projection_count", join.spec.output_projection.len().to_string()),
                ("has_build_artifact", join.spec.build_artifact_id.is_some().to_string()),
            ],
        )),
    }
}

fn root_topk_distinct_aggregate_ids(
    plan: &QueryPlan,
    reachable: &BTreeSet<NodeId>,
    allow_grouping_sets_without_window: bool,
) -> Result<Vec<NodeId>, CapabilityDecision> {
    let mut distinct_aggregate_ids = Vec::new();
    let window_input_sort_ids = root_topk_window_input_sort_ids(plan, &root_topk_window_ids(plan, reachable)?)?;
    for node_id in reachable {
        let Some(node) = plan.node(*node_id) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", node_id.to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Aggregate(aggregate) => {
                if aggregate.spec.grouping_sets.is_some() {
                    if allow_grouping_sets_without_window
                        && root_topk_rankless_grouping_sets_contract_satisfied(*node_id, aggregate)?
                    {
                        continue;
                    }
                    return Err(CapabilityDecision::rejected(
                        "root_topk_join_aggregate_rejects_rollup_subcontract",
                        [("node_id", node_id.to_string())],
                    ));
                }
                let distinct_expr_count = aggregate.spec.aggr_exprs.iter().filter(|expr| expr.distinct).count();
                if distinct_expr_count > 0
                    || (aggregate.spec.aggr_exprs.is_empty() && !aggregate.spec.group_exprs.is_empty())
                {
                    if let Some(expr) = aggregate.spec.aggr_exprs.iter().find(|expr| {
                        expr.distinct && (expr.filter.is_some() || expr.function.is_floating_statistical())
                    }) {
                        return Err(CapabilityDecision::rejected(
                            "root_topk_join_aggregate_rejects_unsupported_distinct_state",
                            [
                                ("node_id", node_id.to_string()),
                                ("aggregate_function", format!("{:?}", expr.function)),
                                ("has_filter", expr.filter.is_some().to_string()),
                            ],
                        ));
                    }
                    distinct_aggregate_ids.push(*node_id);
                }
            },
            PlanNode::Sort(_) if window_input_sort_ids.contains(node_id) => {},
            PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_) => {
                return Err(CapabilityDecision::rejected(
                    "root_topk_join_aggregate_rejects_nested_blocking_or_branch_node",
                    [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
                ));
            },
            PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Projection(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_) => {},
        }
    }
    distinct_aggregate_ids.sort_unstable();
    distinct_aggregate_ids.dedup();
    Ok(distinct_aggregate_ids)
}

fn root_topk_rankless_grouping_sets_contract_satisfied(
    aggregate_id: NodeId,
    aggregate: &AggregateNode,
) -> Result<bool, CapabilityDecision> {
    let Some(grouping_sets) = aggregate.spec.grouping_sets.as_ref() else {
        return Ok(false);
    };
    let group_expr_count = aggregate.spec.group_exprs.len();
    let grouping_set_count = grouping_sets.groups.len();
    if (1..=4).contains(&group_expr_count)
        && grouping_set_count == group_expr_count + 1
        && root_topk_grouping_sets_are_suffix_rollup_masks(group_expr_count, &grouping_sets.groups)
    {
        return Ok(true);
    }
    Err(CapabilityDecision::rejected(
        "root_topk_join_aggregate_requires_rankless_rollup_grouping_sets_contract",
        [
            ("node_id", aggregate_id.to_string()),
            ("group_expr_count", group_expr_count.to_string()),
            ("grouping_set_count", grouping_set_count.to_string()),
            ("max_rankless_rollup_group_expr_count", "4".to_string()),
            ("required_grouping_mask_contract", "suffix_rollup".to_string()),
        ],
    ))
}

fn root_topk_grouping_sets_are_suffix_rollup_masks(group_expr_count: usize, groups: &[Vec<bool>]) -> bool {
    if groups.len() != group_expr_count + 1 {
        return false;
    }
    let actual = groups.iter().cloned().collect::<BTreeSet<_>>();
    let expected = (0..=group_expr_count)
        .map(|rolled_count| (0..group_expr_count).map(|idx| idx >= group_expr_count - rolled_count).collect::<Vec<_>>())
        .collect::<BTreeSet<_>>();
    actual == expected
}

fn root_topk_window_ids(plan: &QueryPlan, reachable: &BTreeSet<NodeId>) -> Result<Vec<NodeId>, CapabilityDecision> {
    let mut window_ids = Vec::new();
    for node_id in reachable {
        let Some(node) = plan.node(*node_id) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", node_id.to_string())],
            ));
        };
        if matches!(&node.kind, PlanNode::Window(_)) {
            window_ids.push(*node_id);
        }
    }
    window_ids.sort_unstable();
    window_ids.dedup();
    Ok(window_ids)
}

fn root_topk_window_input_sort_ids(plan: &QueryPlan, window_ids: &[NodeId]) -> Result<Vec<NodeId>, CapabilityDecision> {
    let mut sort_ids = Vec::new();
    for window_id in window_ids {
        let Some(PlanNode::Window(window)) = plan.node(*window_id).map(|node| &node.kind) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", window_id.to_string()), ("role", "window".to_string())],
            ));
        };
        if matches!(plan.node(window.input).map(|node| &node.kind), Some(PlanNode::Sort(_))) {
            if !window.spec.order_by.is_empty() || root_topk_window_allows_orderless_input_sort(window) {
                sort_ids.push(window.input);
            }
        } else {
            if window.spec.order_by.is_empty() {
                continue;
            }
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_requires_ordered_window_input",
                [("node_id", window_id.to_string()), ("input_id", window.input.to_string())],
            ));
        }
    }
    sort_ids.sort_unstable();
    sort_ids.dedup();
    Ok(sort_ids)
}

fn root_topk_window_allows_orderless_input_sort(window: &WindowNode) -> bool {
    !window.spec.partition_by.is_empty()
        && !window.spec.functions.is_empty()
        && window
            .spec
            .functions
            .iter()
            .all(|function| matches!(function, WindowFunction::Aggregate { frame: WindowFrame::WholePartition, .. }))
}

fn aggregate_projection_in_reachable_fragment(
    plan: &QueryPlan,
    reachable: &BTreeSet<NodeId>,
    keyless_or_cross_join_ids: &[NodeId],
    allow_direct_source_aggregate: bool,
    allow_grouping_sets_aggregate: bool,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let mut candidates = Vec::new();
    for node_id in reachable {
        let Some(PlanNode::Projection(projection)) = plan.node(*node_id).map(|node| &node.kind) else {
            continue;
        };
        let Some(PlanNode::Aggregate(aggregate)) = plan.node(projection.input).map(|node| &node.kind) else {
            continue;
        };
        if aggregate.spec.grouping_sets.is_some() && !allow_grouping_sets_aggregate {
            continue;
        }
        let aggregate_input_reachable = collect_reachable_node_ids_from(plan, aggregate.input)?;
        let is_scalar_aggregate = aggregate.spec.group_exprs.is_empty();
        let scalar_feeds_keyless_or_cross = is_scalar_aggregate
            && root_topk_scalar_aggregate_feeds_keyless_or_cross_join(
                plan,
                projection.input,
                *node_id,
                keyless_or_cross_join_ids,
            )?;
        if is_scalar_aggregate && !scalar_feeds_keyless_or_cross {
            continue;
        }
        let aggregate_input_join_count = aggregate_input_reachable
            .iter()
            .filter(|input_id| matches!(plan.node(**input_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
            .count();
        let direct_source_grouped_aggregate =
            !is_scalar_aggregate && aggregate_input_has_local_parquet_source(&aggregate_input_reachable, plan);
        if aggregate_input_join_count > 0
            || (allow_direct_source_aggregate && direct_source_grouped_aggregate)
            || (direct_source_grouped_aggregate
                && root_topk_grouped_aggregate_feeds_keyless_conditional_join(
                    plan,
                    projection.input,
                    *node_id,
                    keyless_or_cross_join_ids,
                )?)
        {
            candidates.push((projection.input, *node_id));
        }
    }
    for node_id in reachable {
        let Some(PlanNode::Aggregate(aggregate)) = plan.node(*node_id).map(|node| &node.kind) else {
            continue;
        };
        if aggregate.spec.grouping_sets.is_some() && !allow_grouping_sets_aggregate {
            continue;
        }
        if aggregate.spec.group_exprs.is_empty() {
            continue;
        }
        let aggregate_input_reachable = collect_reachable_node_ids_from(plan, aggregate.input)?;
        let aggregate_input_join_count = aggregate_input_reachable
            .iter()
            .filter(|input_id| matches!(plan.node(**input_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
            .count();
        let direct_source_grouped_aggregate =
            aggregate_input_has_local_parquet_source(&aggregate_input_reachable, plan);
        if aggregate_input_join_count > 0
            || (allow_direct_source_aggregate && direct_source_grouped_aggregate)
            || (direct_source_grouped_aggregate
                && root_topk_grouped_aggregate_feeds_keyless_conditional_join(
                    plan,
                    *node_id,
                    *node_id,
                    keyless_or_cross_join_ids,
                )?)
        {
            candidates.push((*node_id, *node_id));
        }
    }
    match candidates.as_slice() {
        [(aggregate_id, projection_id), ..] => Ok((*aggregate_id, *projection_id)),
        [] => Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_requires_grouped_aggregate_fragment",
            [("reachable_node_count", reachable.len().to_string())],
        )),
    }
}

fn aggregate_input_has_local_parquet_source(reachable: &BTreeSet<NodeId>, plan: &QueryPlan) -> bool {
    reachable
        .iter()
        .copied()
        .any(|node_id| matches!(plan.node(node_id).map(|node| &node.kind), Some(PlanNode::Source(source)) if matches!(source.as_ref(), SourceNode::LocalParquet(_))))
}

fn root_topk_aggregate_requires_source_reduction(
    plan: &QueryPlan,
    aggregate_id: NodeId,
    aggregate_projection_id: NodeId,
    keyless_or_cross_join_ids: &[NodeId],
) -> Result<bool, CapabilityDecision> {
    let Some(PlanNode::Aggregate(aggregate)) = plan.node(aggregate_id).map(|node| &node.kind) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_missing_grouped_aggregate",
            [("aggregate_id", aggregate_id.to_string())],
        ));
    };
    if !aggregate.spec.group_exprs.is_empty() {
        return Ok(true);
    }
    Ok(!root_topk_scalar_aggregate_feeds_keyless_or_cross_join(
        plan,
        aggregate_id,
        aggregate_projection_id,
        keyless_or_cross_join_ids,
    )?)
}

fn root_topk_scalar_aggregate_feeds_keyless_or_cross_join(
    plan: &QueryPlan,
    aggregate_id: NodeId,
    aggregate_projection_id: NodeId,
    keyless_or_cross_join_ids: &[NodeId],
) -> Result<bool, CapabilityDecision> {
    for join_id in keyless_or_cross_join_ids {
        let Some(PlanNode::Join(join)) = plan.node(*join_id).map(|node| &node.kind) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", join_id.to_string()), ("role", "keyless_or_cross_join".to_string())],
            ));
        };
        for input in join.inputs {
            if input == aggregate_id || input == aggregate_projection_id {
                return Ok(true);
            }
            let Some((producer_id, chain_ids)) = peel_resident_unary_chain_for_capability(plan, input) else {
                return Err(CapabilityDecision::rejected(
                    "root_topk_join_aggregate_missing_node",
                    [("node_id", input.to_string()), ("role", "keyless_or_cross_input".to_string())],
                ));
            };
            if producer_id == aggregate_id && chain_ids.contains(&aggregate_projection_id) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn root_topk_grouped_aggregate_feeds_keyless_conditional_join(
    plan: &QueryPlan,
    aggregate_id: NodeId,
    aggregate_projection_id: NodeId,
    keyless_or_cross_join_ids: &[NodeId],
) -> Result<bool, CapabilityDecision> {
    for join_id in keyless_or_cross_join_ids {
        let Some(PlanNode::Join(join)) = plan.node(*join_id).map(|node| &node.kind) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", join_id.to_string()), ("role", "keyless_or_cross_join".to_string())],
            ));
        };
        if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeylessConditional(_))) {
            continue;
        }
        for input in join.inputs {
            if input == aggregate_id || input == aggregate_projection_id {
                return Ok(true);
            }
            let Some((producer_id, chain_ids)) = peel_resident_unary_chain_for_capability(plan, input) else {
                return Err(CapabilityDecision::rejected(
                    "root_topk_join_aggregate_missing_node",
                    [
                        ("node_id", input.to_string()),
                        ("role", "keyless_conditional_grouped_aggregate_input".to_string()),
                    ],
                ));
            };
            if producer_id == aggregate_id && chain_ids.contains(&aggregate_projection_id) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn require_root_topk_keyless_grouped_aggregate_source_reduction(
    plan: &QueryPlan,
    keyless_or_cross_join_ids: &[NodeId],
) -> Result<(), CapabilityDecision> {
    let mut aggregate_ids = BTreeSet::new();
    for join_id in keyless_or_cross_join_ids {
        let Some(PlanNode::Join(join)) = plan.node(*join_id).map(|node| &node.kind) else {
            return Err(CapabilityDecision::rejected(
                "root_topk_join_aggregate_missing_node",
                [("node_id", join_id.to_string()), ("role", "keyless_or_cross_join".to_string())],
            ));
        };
        if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeylessConditional(_))) {
            continue;
        }
        for input in join.inputs {
            let Some((producer_id, _chain_ids)) = peel_resident_unary_chain_for_capability(plan, input) else {
                return Err(CapabilityDecision::rejected(
                    "root_topk_join_aggregate_missing_node",
                    [
                        ("node_id", input.to_string()),
                        ("role", "keyless_conditional_grouped_aggregate_input".to_string()),
                    ],
                ));
            };
            if matches!(
                plan.node(producer_id).map(|node| &node.kind),
                Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
            ) {
                aggregate_ids.insert(producer_id);
            }
        }
    }

    for aggregate_id in aggregate_ids {
        let aggregate_input_reachable = aggregate_input_reachable_nodes(plan, aggregate_id)?;
        require_root_topk_source_reduction_proof(plan, &aggregate_input_reachable, aggregate_id, false)?;
    }
    Ok(())
}

fn aggregate_input_reachable_nodes(
    plan: &QueryPlan,
    aggregate_id: NodeId,
) -> Result<BTreeSet<NodeId>, CapabilityDecision> {
    let Some(PlanNode::Aggregate(aggregate)) = plan.node(aggregate_id).map(|node| &node.kind) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_missing_grouped_aggregate",
            [("aggregate_id", aggregate_id.to_string())],
        ));
    };
    collect_reachable_node_ids_from(plan, aggregate.input)
}

fn require_root_topk_source_reduction_proof(
    plan: &QueryPlan,
    aggregate_input_reachable: &BTreeSet<NodeId>,
    aggregate_id: NodeId,
    require_joined_branch_local_reduction: bool,
) -> Result<(), CapabilityDecision> {
    let has_joined_aggregate_input = aggregate_input_reachable
        .iter()
        .any(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Join(_))));
    if has_joined_aggregate_input
        && root_topk_joined_aggregate_input_has_branch_local_source_reduction(plan, aggregate_id, &BTreeSet::new())?
    {
        return Ok(());
    }
    if require_joined_branch_local_reduction && has_joined_aggregate_input {
        return Err(root_topk_source_reduction_rejection(plan, aggregate_input_reachable, aggregate_id));
    }
    for route in plan.dynamic_filter_routes() {
        if route.consumer_reduction_stage() != DynamicFilterConsumerReductionStage::SourceReader {
            continue;
        }
        let DynamicFilterConsumer::LocalParquetSource { source_node_id } = &route.consumer else {
            continue;
        };
        if aggregate_input_reachable.contains(source_node_id) {
            return Ok(());
        }
    }
    if root_topk_direct_local_parquet_aggregate_input_has_source_reduction(plan, aggregate_id)? {
        return Ok(());
    }
    for source_id in aggregate_input_reachable.iter().copied() {
        let Some(PlanNode::Source(source)) = plan.node(source_id).map(|node| &node.kind) else {
            continue;
        };
        let SourceNode::LocalParquet(source) = source.as_ref() else {
            continue;
        };
        if source
            .diagnostics
            .selected_row_group_stats
            .as_ref()
            .is_some_and(|stats| stats.selected_row_groups < stats.total_row_groups)
        {
            return Ok(());
        }
    }
    Err(root_topk_source_reduction_rejection(plan, aggregate_input_reachable, aggregate_id))
}

fn require_root_topk_all_joined_grouped_aggregates_source_reduction(
    plan: &QueryPlan,
    root_reachable: &BTreeSet<NodeId>,
    left_output_bounded_join_ids: &BTreeSet<NodeId>,
) -> Result<(), CapabilityDecision> {
    for node_id in root_reachable {
        let Some(PlanNode::Aggregate(aggregate)) = plan.node(*node_id).map(|node| &node.kind) else {
            continue;
        };
        if aggregate.spec.group_exprs.is_empty() {
            continue;
        }
        let aggregate_input_reachable = collect_reachable_node_ids_from(plan, aggregate.input)?;
        if !aggregate_input_reachable
            .iter()
            .any(|input_id| matches!(plan.node(*input_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
        {
            continue;
        }
        if root_topk_joined_aggregate_input_has_branch_local_source_reduction(
            plan,
            *node_id,
            left_output_bounded_join_ids,
        )? {
            continue;
        }
        return Err(root_topk_source_reduction_rejection(plan, &aggregate_input_reachable, *node_id));
    }
    Ok(())
}

fn root_topk_source_reduction_rejection(
    plan: &QueryPlan,
    aggregate_input_reachable: &BTreeSet<NodeId>,
    aggregate_id: NodeId,
) -> CapabilityDecision {
    CapabilityDecision::rejected(
        "root_topk_join_aggregate_requires_source_reduction_proof",
        [
            ("aggregate_id", aggregate_id.to_string()),
            ("dynamic_filter_route_count", plan.dynamic_filter_routes().len().to_string()),
            ("aggregate_input_node_count", aggregate_input_reachable.len().to_string()),
        ],
    )
}

fn root_topk_direct_local_parquet_aggregate_input_has_source_reduction(
    plan: &QueryPlan,
    aggregate_id: NodeId,
) -> Result<bool, CapabilityDecision> {
    let Some(PlanNode::Aggregate(aggregate)) = plan.node(aggregate_id).map(|node| &node.kind) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_missing_grouped_aggregate",
            [("aggregate_id", aggregate_id.to_string())],
        ));
    };
    root_topk_direct_local_parquet_pipeline_has_source_reduction(plan, aggregate.input, false, &mut BTreeSet::new())
}

fn root_topk_joined_aggregate_input_has_branch_local_source_reduction(
    plan: &QueryPlan,
    aggregate_id: NodeId,
    left_output_bounded_join_ids: &BTreeSet<NodeId>,
) -> Result<bool, CapabilityDecision> {
    let Some(PlanNode::Aggregate(aggregate)) = plan.node(aggregate_id).map(|node| &node.kind) else {
        return Err(CapabilityDecision::rejected(
            "root_topk_join_aggregate_missing_grouped_aggregate",
            [("aggregate_id", aggregate_id.to_string())],
        ));
    };
    root_topk_joined_pipeline_has_branch_local_source_reduction(
        plan,
        aggregate.input,
        false,
        left_output_bounded_join_ids,
        &mut BTreeSet::new(),
    )
}

fn root_topk_joined_pipeline_has_branch_local_source_reduction(
    plan: &QueryPlan,
    node_id: NodeId,
    has_source_adjacent_filter: bool,
    left_output_bounded_join_ids: &BTreeSet<NodeId>,
    visited: &mut BTreeSet<(NodeId, bool)>,
) -> Result<bool, CapabilityDecision> {
    if !visited.insert((node_id, has_source_adjacent_filter)) {
        return Err(CapabilityDecision::rejected(
            "root_topk_source_reduction_pipeline_cycle",
            [("node_id", node_id.to_string())],
        ));
    }
    let result = match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Projection(projection)) => root_topk_joined_pipeline_has_branch_local_source_reduction(
            plan,
            projection.input,
            has_source_adjacent_filter,
            left_output_bounded_join_ids,
            visited,
        ),
        Some(PlanNode::Filter(filter)) => root_topk_joined_pipeline_has_branch_local_source_reduction(
            plan,
            filter.input,
            true,
            left_output_bounded_join_ids,
            visited,
        ),
        Some(PlanNode::Aggregate(aggregate)) => root_topk_joined_pipeline_has_branch_local_source_reduction(
            plan,
            aggregate.input,
            false,
            left_output_bounded_join_ids,
            visited,
        ),
        Some(PlanNode::Join(join)) => {
            let [left, right] = join.inputs;
            let left_reduced = root_topk_joined_pipeline_has_branch_local_source_reduction(
                plan,
                left,
                false,
                left_output_bounded_join_ids,
                visited,
            )?;
            if left_output_bounded_join_ids.contains(&node_id) {
                return Ok(left_reduced);
            }
            let right_reduced = root_topk_joined_pipeline_has_branch_local_source_reduction(
                plan,
                right,
                false,
                left_output_bounded_join_ids,
                visited,
            )?;
            Ok(left_reduced && right_reduced)
        },
        Some(PlanNode::Source(source)) => {
            if let SourceNode::LocalParquet(source) = source.as_ref() {
                Ok(root_topk_local_parquet_source_has_reduction(plan, node_id, source, has_source_adjacent_filter))
            } else {
                Ok(false)
            }
        },
        Some(
            PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_),
        )
        | None => Ok(false),
    };
    visited.remove(&(node_id, has_source_adjacent_filter));
    result
}

fn root_topk_local_parquet_source_has_reduction(
    plan: &QueryPlan,
    source_id: NodeId,
    source: &LocalParquetSource,
    has_source_adjacent_filter: bool,
) -> bool {
    has_source_adjacent_filter
        || plan.dynamic_filter_routes().iter().any(|route| {
            route.consumer_reduction_stage() == DynamicFilterConsumerReductionStage::SourceReader
                && matches!(&route.consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if *source_node_id == source_id)
        })
        || (source.diagnostics.predicate_disposition == PredicateDisposition::ReaderPushdown
            && source
                .reader_predicate
                .as_ref()
                .is_some_and(|predicate| source.reader_predicate_is_supported(predicate)))
        || source
            .diagnostics
            .selected_row_group_stats
            .as_ref()
            .is_some_and(|stats| stats.selected_row_groups < stats.total_row_groups)
        || source
            .diagnostics
            .selected_row_group_stats
            .as_ref()
            .and_then(|stats| stats.selected_source_bytes())
            .or(source.diagnostics.known_bytes)
            .is_some_and(|bytes| {
                bytes <= plan
                    .keyed_hash_join_strategy_options()
                    .small_one_shot_probe_bytes_threshold()
            })
}

fn root_topk_direct_local_parquet_pipeline_has_source_reduction(
    plan: &QueryPlan,
    node_id: NodeId,
    has_source_adjacent_filter: bool,
    visited: &mut BTreeSet<NodeId>,
) -> Result<bool, CapabilityDecision> {
    if !visited.insert(node_id) {
        return Err(CapabilityDecision::rejected(
            "root_topk_source_reduction_pipeline_cycle",
            [("node_id", node_id.to_string())],
        ));
    }
    let result = match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Projection(projection)) => root_topk_direct_local_parquet_pipeline_has_source_reduction(
            plan,
            projection.input,
            has_source_adjacent_filter,
            visited,
        ),
        Some(PlanNode::Filter(filter)) => {
            root_topk_direct_local_parquet_pipeline_has_source_reduction(plan, filter.input, true, visited)
        },
        Some(PlanNode::Source(source)) => {
            if let SourceNode::LocalParquet(source) = source.as_ref() {
                Ok(has_source_adjacent_filter
                    || (source.diagnostics.predicate_disposition == PredicateDisposition::ReaderPushdown
                        && source
                            .reader_predicate
                            .as_ref()
                            .is_some_and(|predicate| source.reader_predicate_is_supported(predicate))))
            } else {
                Ok(false)
            }
        },
        Some(
            PlanNode::Aggregate(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_),
        )
        | None => Ok(false),
    };
    visited.remove(&node_id);
    result
}
