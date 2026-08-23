use std::collections::{BTreeMap, BTreeSet};

use super::{
    CapabilityDecision, JoinExecutionCapability, JoinSide, JoinType, KeylessConditionalJoinKind, NodeId, PlanNode,
    QueryPlan, RootFullSortJoinAggregateShape, RootFullSortJoinAggregateSubcontract, WindowFunction,
    collect_reachable_node_ids, collect_reachable_node_ids_from, consumer_nodes_for_reachable,
    join_node_execution_capability, join_node_with_missing_reason, node_operator_fact,
    peel_resident_unary_chain_for_capability,
};
use crate::plan::{WindowFrame, WindowNode};

pub(in crate::capability) fn root_full_sort_join_aggregate_shape_decision(
    plan: &QueryPlan,
) -> Result<RootFullSortJoinAggregateShape, CapabilityDecision> {
    let mut sink_input_id = plan.sink().input();
    let mut post_sort_projection_ids = Vec::new();
    while let Some(PlanNode::Projection(proj)) = plan.node(sink_input_id).map(|node| &node.kind) {
        post_sort_projection_ids.push(sink_input_id);
        sink_input_id = proj.input;
    }
    let final_sort_id = sink_input_id;
    let final_sort = match plan.node(final_sort_id).map(|node| &node.kind) {
        Some(PlanNode::Sort(sort)) => sort,
        Some(node) => {
            return Err(CapabilityDecision::rejected(
                "root_full_sort_join_aggregate_requires_sink_full_sort",
                [("node_id", final_sort_id.to_string()), ("operator", node_operator_fact(node).to_string())],
            ));
        },
        None => {
            return Err(CapabilityDecision::rejected(
                "root_full_sort_join_aggregate_missing_node",
                [("node_id", final_sort_id.to_string()), ("role", "final_sort".to_string())],
            ));
        },
    };
    let post_root_input_id = final_sort.input;
    let Some((post_root_candidate_id, post_root_chain_ids)) =
        peel_resident_unary_chain_for_capability(plan, post_root_input_id)
    else {
        return Err(CapabilityDecision::rejected(
            "root_full_sort_join_aggregate_requires_unary_chain_before_sort",
            [("node_id", post_root_input_id.to_string())],
        ));
    };

    let root_reachable = collect_reachable_node_ids_from(plan, post_root_candidate_id)?;
    let mut join_ids = root_reachable
        .iter()
        .copied()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
        .collect::<Vec<_>>();
    join_ids.sort_unstable();
    if join_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "root_full_sort_join_aggregate_requires_join_fragment",
            [("post_root_candidate_id", post_root_candidate_id.to_string())],
        ));
    }

    let (aggregate_id, aggregate_projection_id) = root_full_sort_grouped_aggregate_fragment(plan, &root_reachable)?;
    let root_join_id = root_full_sort_root_join_id(plan, post_root_candidate_id, aggregate_id, &join_ids)?;
    let keyless_or_cross_join_ids = root_full_sort_keyless_or_cross_join_ids(plan, &join_ids)?;
    let whole_query_reachable = collect_reachable_node_ids(plan)?;
    let whole_query_consumers = consumer_nodes_for_reachable(plan, &whole_query_reachable);
    let shared_producer_ids = root_full_sort_shared_computed_producer_ids(plan, &whole_query_consumers)?;
    let shared_projection_producer_ids =
        root_full_sort_shared_producers_by_operator(plan, &shared_producer_ids, |node| {
            matches!(node, PlanNode::Projection(_) | PlanNode::Filter(_))
        });
    let shared_aggregate_producer_ids =
        root_full_sort_shared_producers_by_operator(plan, &shared_producer_ids, |node| {
            matches!(node, PlanNode::Aggregate(_))
        });
    let unsupported_shared_producer_ids = root_full_sort_unsupported_shared_producer_ids(
        plan,
        &shared_producer_ids,
        &shared_projection_producer_ids,
        &shared_aggregate_producer_ids,
    );
    let shared_aggregate_consumer_join_ids =
        root_full_sort_shared_aggregate_consumer_join_ids(plan, &whole_query_consumers, &shared_aggregate_producer_ids);
    let subcontract = if root_full_sort_keyless_shared_aggregate_fanout_shape(
        plan,
        root_join_id,
        &join_ids,
        &keyless_or_cross_join_ids,
        &shared_producer_ids,
        &shared_projection_producer_ids,
        &shared_aggregate_producer_ids,
        &unsupported_shared_producer_ids,
    ) {
        RootFullSortJoinAggregateSubcontract::SharedAggregateFanout
    } else if !keyless_or_cross_join_ids.is_empty() {
        RootFullSortJoinAggregateSubcontract::KeylessOrCrossJoin
    } else if shared_producer_ids.is_empty() {
        RootFullSortJoinAggregateSubcontract::KeyedHashGroupedAggregate
    } else if shared_producer_ids == shared_projection_producer_ids {
        RootFullSortJoinAggregateSubcontract::RepeatedProjectionFanout
    } else if !unsupported_shared_producer_ids.is_empty() {
        RootFullSortJoinAggregateSubcontract::UnsupportedSharedOrFanout
    } else if !shared_aggregate_producer_ids.is_empty() {
        RootFullSortJoinAggregateSubcontract::SharedAggregateFanout
    } else {
        RootFullSortJoinAggregateSubcontract::UnsupportedSharedOrFanout
    };
    let mut post_root_projection_ids: Vec<NodeId> = post_sort_projection_ids;
    post_root_projection_ids.extend(
        post_root_chain_ids
            .into_iter()
            .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Projection(_)))),
    );

    Ok(RootFullSortJoinAggregateShape {
        root_join_id,
        join_ids,
        keyless_or_cross_join_ids,
        shared_producer_ids,
        shared_projection_producer_ids,
        shared_aggregate_producer_ids,
        unsupported_shared_producer_ids,
        shared_aggregate_consumer_join_ids,
        aggregate_id,
        aggregate_projection_id,
        post_root_projection_ids,
        final_sort_id,
        subcontract,
    })
}

#[expect(clippy::too_many_arguments, reason = "planning inputs are distinct proof and constraint values")]
fn root_full_sort_keyless_shared_aggregate_fanout_shape(
    plan: &QueryPlan,
    root_join_id: NodeId,
    join_ids: &[NodeId],
    keyless_or_cross_join_ids: &[NodeId],
    shared_producer_ids: &[NodeId],
    shared_projection_producer_ids: &[NodeId],
    shared_aggregate_producer_ids: &[NodeId],
    unsupported_shared_producer_ids: &[NodeId],
) -> bool {
    if keyless_or_cross_join_ids != [root_join_id]
        || shared_producer_ids.is_empty()
        || shared_producer_ids != shared_aggregate_producer_ids
        || !shared_projection_producer_ids.is_empty()
        || !unsupported_shared_producer_ids.is_empty()
    {
        return false;
    }
    if !shared_aggregate_producer_ids.iter().all(|producer_id| {
        matches!(
            plan.node(*producer_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
        )
    }) {
        return false;
    }

    let Some(PlanNode::Join(root_join)) = plan.node(root_join_id).map(|node| &node.kind) else {
        return false;
    };
    if !matches!(
        join_node_execution_capability(root_join),
        Some(JoinExecutionCapability::KeylessConditional(KeylessConditionalJoinKind::Inner))
    ) || !root_join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
    {
        return false;
    }
    join_ids.iter().all(|join_id| {
        if *join_id == root_join_id {
            return true;
        }
        matches!(
            plan.node(*join_id).and_then(|node| match &node.kind {
                PlanNode::Join(join) => join_node_execution_capability(join),
                _ => None,
            }),
            Some(JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
        )
    })
}

fn root_full_sort_shared_producers_by_operator(
    plan: &QueryPlan,
    shared_producer_ids: &[NodeId],
    matches_operator: impl Fn(&PlanNode) -> bool,
) -> Vec<NodeId> {
    shared_producer_ids
        .iter()
        .copied()
        .filter(|node_id| plan.node(*node_id).is_some_and(|node| matches_operator(&node.kind)))
        .collect()
}

fn root_full_sort_unsupported_shared_producer_ids(
    plan: &QueryPlan,
    shared_producer_ids: &[NodeId],
    shared_projection_producer_ids: &[NodeId],
    shared_aggregate_producer_ids: &[NodeId],
) -> Vec<NodeId> {
    shared_producer_ids
        .iter()
        .copied()
        .filter(|node_id| {
            plan.node(*node_id).is_some()
                && !shared_projection_producer_ids.contains(node_id)
                && !shared_aggregate_producer_ids.contains(node_id)
        })
        .collect()
}

fn root_full_sort_shared_aggregate_consumer_join_ids(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    shared_aggregate_producer_ids: &[NodeId],
) -> Vec<NodeId> {
    let mut consumer_join_ids = Vec::new();
    for producer in shared_aggregate_producer_ids {
        if let Some(producer_consumers) = consumers.get(producer) {
            for consumer in producer_consumers {
                root_full_sort_collect_downstream_consumer_joins(
                    plan,
                    consumers,
                    *consumer,
                    &mut BTreeSet::new(),
                    &mut consumer_join_ids,
                );
            }
        }
    }
    consumer_join_ids.sort_unstable();
    consumer_join_ids.dedup();
    consumer_join_ids
}

fn root_full_sort_collect_downstream_consumer_joins(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    node_id: NodeId,
    visited: &mut BTreeSet<NodeId>,
    consumer_join_ids: &mut Vec<NodeId>,
) {
    if !visited.insert(node_id) {
        return;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Join(_)) => consumer_join_ids.push(node_id),
        Some(PlanNode::Projection(_) | PlanNode::Filter(_) | PlanNode::Window(_)) => {
            if let Some(next_consumers) = consumers.get(&node_id) {
                for consumer in next_consumers {
                    root_full_sort_collect_downstream_consumer_joins(
                        plan,
                        consumers,
                        *consumer,
                        visited,
                        consumer_join_ids,
                    );
                }
            }
        },
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty() => {
            if let Some(next_consumers) = consumers.get(&node_id) {
                for consumer in next_consumers {
                    root_full_sort_collect_downstream_consumer_joins(
                        plan,
                        consumers,
                        *consumer,
                        visited,
                        consumer_join_ids,
                    );
                }
            }
        },
        Some(
            PlanNode::Aggregate(_)
            | PlanNode::Sort(_)
            | PlanNode::Source(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_),
        )
        | None => {},
    }
}

fn root_full_sort_root_join_id(
    plan: &QueryPlan,
    post_root_candidate_id: NodeId,
    aggregate_id: NodeId,
    join_ids: &[NodeId],
) -> Result<NodeId, CapabilityDecision> {
    if matches!(plan.node(post_root_candidate_id).map(|node| &node.kind), Some(PlanNode::Join(_))) {
        return Ok(post_root_candidate_id);
    }
    let Some(PlanNode::Aggregate(aggregate)) = plan.node(aggregate_id).map(|node| &node.kind) else {
        return join_ids.last().copied().ok_or_else(|| {
            CapabilityDecision::rejected(
                "root_full_sort_join_aggregate_requires_join_fragment",
                [("post_root_candidate_id", post_root_candidate_id.to_string())],
            )
        });
    };
    if let Some((aggregate_input, _)) = peel_resident_unary_chain_for_capability(plan, aggregate.input)
        && matches!(plan.node(aggregate_input).map(|node| &node.kind), Some(PlanNode::Join(_)))
    {
        return Ok(aggregate_input);
    }
    join_ids.last().copied().ok_or_else(|| {
        CapabilityDecision::rejected(
            "root_full_sort_join_aggregate_requires_join_fragment",
            [("aggregate_id", aggregate_id.to_string())],
        )
    })
}

fn root_full_sort_grouped_aggregate_fragment(
    plan: &QueryPlan,
    reachable: &BTreeSet<NodeId>,
) -> Result<(NodeId, NodeId), CapabilityDecision> {
    let consumers = consumer_nodes_for_reachable(plan, reachable);
    let mut aggregate_ids = Vec::new();
    for node_id in reachable {
        let Some(node) = plan.node(*node_id) else {
            return Err(CapabilityDecision::rejected(
                "root_full_sort_join_aggregate_missing_node",
                [("node_id", node_id.to_string())],
            ));
        };
        if matches!(&node.kind, PlanNode::Aggregate(aggregate) if !aggregate.spec.group_exprs.is_empty()) {
            aggregate_ids.push(*node_id);
        }
    }
    if aggregate_ids.is_empty() {
        return Err(CapabilityDecision::rejected(
            "root_full_sort_join_aggregate_requires_grouped_aggregate_fragment",
            [("reachable_node_count", reachable.len().to_string())],
        ));
    }
    let aggregate_id = root_full_sort_sink_path_grouped_aggregate_id(plan)
        .filter(|sink_aggregate_id| aggregate_ids.contains(sink_aggregate_id))
        .unwrap_or(aggregate_ids[0]);
    let aggregate_projection_id = consumers
        .get(&aggregate_id)
        .and_then(|consumer_ids| {
            consumer_ids.iter().copied().find(|consumer_id| {
                matches!(plan.node(*consumer_id).map(|consumer| &consumer.kind), Some(PlanNode::Projection(_)))
            })
        })
        .unwrap_or(aggregate_id);
    Ok((aggregate_id, aggregate_projection_id))
}

fn root_full_sort_sink_path_grouped_aggregate_id(plan: &QueryPlan) -> Option<NodeId> {
    let mut sink_input_id = plan.sink().input();
    while let Some(PlanNode::Projection(proj)) = plan.node(sink_input_id).map(|node| &node.kind) {
        sink_input_id = proj.input;
    }
    let mut current = match plan.node(sink_input_id).map(|node| &node.kind) {
        Some(PlanNode::Sort(sort)) => sort.input,
        _ => return None,
    };
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let node = plan.node(current)?;
        match &node.kind {
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Sort(sort) => current = sort.input,
            PlanNode::Window(window) if root_full_sort_window_allows_sink_path_grouped_aggregate_search(window) => {
                current = window.input;
            },
            PlanNode::Aggregate(aggregate) if !aggregate.spec.group_exprs.is_empty() => return Some(current),
            _ => return None,
        }
    }
}

pub(in crate::capability) fn root_full_sort_window_allows_sink_path_grouped_aggregate_search(
    window: &WindowNode,
) -> bool {
    window.spec.order_by.is_empty()
        && !window.spec.partition_by.is_empty()
        && !window.spec.functions.is_empty()
        && window
            .spec
            .functions
            .iter()
            .all(|function| matches!(function, WindowFunction::Aggregate { frame: WindowFrame::WholePartition, .. }))
}

fn root_full_sort_keyless_or_cross_join_ids(
    plan: &QueryPlan,
    join_ids: &[NodeId],
) -> Result<Vec<NodeId>, CapabilityDecision> {
    let mut keyless_or_cross_join_ids = Vec::new();
    for join_id in join_ids {
        let join = join_node_with_missing_reason(
            plan,
            *join_id,
            "root_full_sort_join_aggregate_missing_node",
            "root_full_sort_join_aggregate_requires_join_fragment",
        )?;
        if join.spec.keys.is_empty() || join.spec.join_type == JoinType::Cross {
            keyless_or_cross_join_ids.push(*join_id);
        }
    }
    Ok(keyless_or_cross_join_ids)
}

fn root_full_sort_shared_computed_producer_ids(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
) -> Result<Vec<NodeId>, CapabilityDecision> {
    let mut shared_producers = Vec::new();
    for (producer, producer_consumers) in consumers {
        if producer_consumers.len() <= 1 {
            continue;
        }
        if matches!(plan.node(*producer).map(|node| &node.kind), Some(PlanNode::Source(_))) {
            continue;
        }
        if plan.node(*producer).is_none() {
            return Err(CapabilityDecision::rejected(
                "root_full_sort_join_aggregate_missing_node",
                [("node_id", producer.to_string()), ("role", "shared_producer".to_string())],
            ));
        }
        shared_producers.push(*producer);
    }
    shared_producers.sort_unstable();
    shared_producers.dedup();
    Ok(shared_producers)
}
