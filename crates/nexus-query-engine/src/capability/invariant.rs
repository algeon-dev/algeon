//! Subgraph-invariant analysis (currently focused on join shape invariants).
//!
//! Inspects edge contracts plus producer/consumer node facts to verify
//! invariants that span more than a single node — for example, "build side of
//! this join must dominate the resident-resource reservation gate."

use super::{edge::*, graph::*, model::*, node::*, shape::*, *};
use crate::{plan::WindowFunction, source::LocalParquetSource};

const LARGE_BLOCKING_ROLLUP_RANK_SOURCE_BYTES_THRESHOLD: u64 = 512 << 20;

pub(super) fn analyze_invariants(plan: &QueryPlan, edges: &[EdgeContract]) -> Vec<SubgraphInvariant> {
    let mut invariants = Vec::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let build_input = join.inputs[if join.spec.build_side == JoinSide::Left { 0 } else { 1 }];
        let probe_input = join.inputs[if join.spec.build_side == JoinSide::Left { 1 } else { 0 }];
        let invariant_id = InvariantId(invariants.len());
        let decision = solve_build_before_probe_invariant(plan, node.id, build_input, probe_input);
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::BuildBeforeProbe { join_id: node.id },
            nodes: vec![build_input, probe_input, node.id],
            required_schedule: ScheduleConstraint::BuildBeforeProbe,
            required_resources: vec![
                ResidentResource::JoinBuildInput { join_id: node.id, input: build_input },
                ResidentResource::JoinProbeInput { join_id: node.id, input: probe_input },
            ],
            reservation_requirements: Vec::new(),
            decision,
        });
        if join.spec.residual_predicate.is_some() {
            let invariant_id = InvariantId(invariants.len());
            let decision = solve_residual_uses_both_inputs_invariant(node.id, join);
            invariants.push(SubgraphInvariant {
                invariant_id,
                kind: SubgraphInvariantKind::ResidualUsesBothInputs { join_id: node.id },
                nodes: vec![join.inputs[0], join.inputs[1], node.id],
                required_schedule: ScheduleConstraint::None,
                required_resources: Vec::new(),
                reservation_requirements: Vec::new(),
                decision,
            });
        }
    }
    append_shared_resident_subplan_invariants(plan, edges, &mut invariants);
    append_aggregate_derivation_invariants(plan, &mut invariants);
    append_projection_transparent_residency_invariants(plan, &mut invariants);
    append_resident_unary_chain_preserves_residency_invariants(plan, &mut invariants);
    append_final_barrier_order_invariants(plan, &mut invariants);
    append_limit_bounded_projection_invariants(plan, &mut invariants);
    append_mixed_residual_post_root_stage_order_invariant(plan, &mut invariants);
    append_hash_join_resident_grouped_aggregate_topk_post_root_stage_order_invariant(plan, &mut invariants);
    append_rollup_rank_topk_after_aggregate_join_pushdown_post_root_stage_order_invariant(plan, &mut invariants);
    append_multi_join_aggregate_over_join_subplan_topk_post_root_stage_order_invariant(plan, &mut invariants);
    append_root_topk_join_aggregate_post_root_stage_order_invariant(plan, &mut invariants);
    append_root_full_sort_join_aggregate_post_root_stage_order_invariant(plan, &mut invariants);
    append_keyless_aggregate_over_join_subplan_inputs_post_root_stage_order_invariant(plan, &mut invariants);
    append_statistical_join_final_sort_exactness_invariant(plan, &mut invariants);
    append_blocking_rollup_rank_full_scan_invariant(plan, &mut invariants);

    if reachable_join_count_from_sink(plan).unwrap_or(0) > 1 {
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::TopologicalBuildProbeSchedule,
            nodes: plan
                .nodes()
                .iter()
                .filter_map(|node| matches!(node.kind, PlanNode::Join(_)).then_some(node.id))
                .collect(),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            reservation_requirements: Vec::new(),
            decision: solve_topological_build_probe_schedule_invariant(plan),
        });
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ResidentLifetimeDominatesConsumers,
            nodes: collect_reachable_node_ids(plan).map(|nodes| nodes.into_iter().collect()).unwrap_or_default(),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            reservation_requirements: Vec::new(),
            decision: solve_resident_lifetime_invariant(plan, edges),
        });
    }

    if matches!(closed_multi_join_subgraph_structure_decision(plan), CapabilityDecision::Satisfied)
        && let Ok(root_join_id) = post_root_join_id(plan)
    {
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ClosedJoinSubtree { root_join_id },
            nodes: collect_reachable_node_ids_from(plan, root_join_id)
                .map(|nodes| nodes.into_iter().collect())
                .unwrap_or_else(|_| vec![root_join_id]),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            reservation_requirements: Vec::new(),
            decision: CapabilityDecision::Satisfied,
        });
    }
    invariants
}

#[derive(Debug, Clone)]
struct StatisticalJoinFinalSortExactnessShape {
    aggregate_ids: Vec<NodeId>,
    join_ids: Vec<NodeId>,
    final_sort_id: NodeId,
}

fn append_statistical_join_final_sort_exactness_invariant(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    let Some(shape) = statistical_join_final_sort_exactness_shape(plan) else {
        return;
    };
    let mut nodes = shape.aggregate_ids.clone();
    nodes.extend(shape.join_ids.iter().copied());
    nodes.push(shape.final_sort_id);
    nodes.sort_unstable();
    nodes.dedup();

    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::StatisticalJoinFinalSortExactness,
        nodes,
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.final_sort_id)],
        reservation_requirements: vec![
            ReservationRequirement::modeled(
                ReservationKind::CapabilityContract,
                "datafusion_exact_host_welford_input_materialization",
                "host_welford_input_materialization_preflight",
                ContractQuality::QueryEngineOwnedBoundary,
            ),
            ReservationRequirement::modeled(
                ReservationKind::CapabilityContract,
                "datafusion_exact_host_welford_output_upload",
                "host_welford_output_upload_reservation",
                ContractQuality::QueryEngineOwnedBoundary,
            ),
            ReservationRequirement::modeled(
                ReservationKind::CapabilityContract,
                "final_full_sort_output_cardinality",
                "final_full_sort_output_cardinality_bound",
                ContractQuality::CapabilityProof,
            ),
        ],
        decision: CapabilityDecision::Satisfied,
    });
}

fn statistical_join_final_sort_exactness_shape(plan: &QueryPlan) -> Option<StatisticalJoinFinalSortExactnessShape> {
    let final_sort_id = plan.sink().input();
    match plan.node(final_sort_id).map(|node| &node.kind)? {
        PlanNode::Sort(_) => {},
        _ => return None,
    };
    let reachable = collect_reachable_node_ids(plan).ok()?;
    let mut aggregate_ids = Vec::new();
    let mut join_ids = Vec::new();

    for node_id in &reachable {
        let node = plan.node(*node_id)?;
        match &node.kind {
            PlanNode::Aggregate(aggregate)
                if aggregate.spec.aggr_exprs.iter().any(|expr| expr.function.is_floating_statistical())
                    && collect_reachable_node_ids_from(plan, aggregate.input).ok()?.iter().any(|input_id| {
                        matches!(plan.node(*input_id).map(|input| &input.kind), Some(PlanNode::Join(_)))
                    }) =>
            {
                aggregate_ids.push(*node_id);
            },
            PlanNode::Join(_) => join_ids.push(*node_id),
            _ => {},
        }
    }

    if aggregate_ids.is_empty() || join_ids.is_empty() {
        return None;
    }

    Some(StatisticalJoinFinalSortExactnessShape { aggregate_ids, join_ids, final_sort_id })
}

#[derive(Debug, Clone, Copy)]
struct BlockingRollupRankTopKFullScanShape {
    aggregate_id: NodeId,
    grouping_key_count: usize,
    grouping_set_count: usize,
    window_id: NodeId,
    rank_function_count: usize,
    rank_sort_id: Option<NodeId>,
    rank_filter_id: Option<NodeId>,
    final_topk_sort_id: Option<NodeId>,
    final_topk_limit_id: Option<NodeId>,
    final_topk_fetch: Option<usize>,
    large_source_id: NodeId,
    large_source_selected_bytes: u64,
    large_source_total_bytes: Option<u64>,
    large_source_selected_row_groups: Option<usize>,
    large_source_total_row_groups: Option<usize>,
    large_source_file_count: usize,
    large_source_read_column_count: usize,
    source_count: usize,
    join_count: usize,
}

fn append_blocking_rollup_rank_full_scan_invariant(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    let Some(shape) = blocking_rollup_rank_full_scan_shape(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    let mut nodes = vec![shape.large_source_id, shape.aggregate_id, shape.window_id];
    if let Some(rank_sort_id) = shape.rank_sort_id {
        nodes.push(rank_sort_id);
    }
    if let Some(rank_filter_id) = shape.rank_filter_id {
        nodes.push(rank_filter_id);
    }
    if let Some(final_topk_sort_id) = shape.final_topk_sort_id {
        nodes.push(final_topk_sort_id);
    }
    if let Some(final_topk_limit_id) = shape.final_topk_limit_id {
        nodes.push(final_topk_limit_id);
    }
    let mut required_resources =
        vec![ResidentResource::NodeOutput(shape.aggregate_id), ResidentResource::NodeOutput(shape.window_id)];
    if let Some(rank_sort_id) = shape.rank_sort_id {
        required_resources.push(ResidentResource::NodeOutput(rank_sort_id));
    }
    if let Some(final_topk_sort_id) = shape.final_topk_sort_id {
        required_resources.push(ResidentResource::NodeOutput(final_topk_sort_id));
    }
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::BlockingRollupRankTopKFullScan,
        nodes,
        required_schedule: ScheduleConstraint::Barrier,
        required_resources,
        reservation_requirements: Vec::new(),
        decision: CapabilityDecision::rejected(
            BLOCKING_ROLLUP_RANK_FULL_SCAN_REQUIRES_EXACT_CONTRACTS,
            [
                ("aggregate_id", shape.aggregate_id.to_string()),
                ("grouping_key_count", shape.grouping_key_count.to_string()),
                ("grouping_set_count", shape.grouping_set_count.to_string()),
                ("window_id", shape.window_id.to_string()),
                ("rank_function_count", shape.rank_function_count.to_string()),
                ("rank_sort_id", optional_node_fact(shape.rank_sort_id)),
                ("rank_filter_id", optional_node_fact(shape.rank_filter_id)),
                ("final_topk_sort_id", optional_node_fact(shape.final_topk_sort_id)),
                ("final_topk_limit_id", optional_node_fact(shape.final_topk_limit_id)),
                ("final_topk_fetch", optional_usize_fact(shape.final_topk_fetch)),
                ("large_source_id", shape.large_source_id.to_string()),
                ("large_source_selected_bytes", shape.large_source_selected_bytes.to_string()),
                ("large_source_total_bytes", optional_u64_fact(shape.large_source_total_bytes)),
                ("large_source_selected_row_groups", optional_usize_fact(shape.large_source_selected_row_groups)),
                ("large_source_total_row_groups", optional_usize_fact(shape.large_source_total_row_groups)),
                ("large_source_threshold_bytes", LARGE_BLOCKING_ROLLUP_RANK_SOURCE_BYTES_THRESHOLD.to_string()),
                ("large_source_file_count", shape.large_source_file_count.to_string()),
                ("large_source_read_column_count", shape.large_source_read_column_count.to_string()),
                ("source_count", shape.source_count.to_string()),
                ("join_count", shape.join_count.to_string()),
                ("required_remediation", "source_reduction_or_exact_blocking_stage_preflight".to_string()),
                (
                    "unmodeled_contracts",
                    "aggregate_contract;rank_sort_contract;window_contract;optional_final_topk_contract".to_string(),
                ),
            ],
        ),
    });
}

fn blocking_rollup_rank_full_scan_shape(plan: &QueryPlan) -> Option<BlockingRollupRankTopKFullScanShape> {
    let reachable = collect_reachable_node_ids(plan).ok()?;
    let mut source_count = 0;
    let mut join_count = 0;
    let mut grouping_aggregates = Vec::new();
    let mut rank_windows = Vec::new();
    let mut final_topks = Vec::new();
    let mut filters = Vec::new();
    let mut large_sources = Vec::new();

    for node_id in &reachable {
        let node = plan.node(*node_id)?;
        match &node.kind {
            PlanNode::Source(source) => {
                source_count += 1;
                let SourceNode::LocalParquet(source) = source.as_ref() else {
                    continue;
                };
                let Some(source_size) = blocking_rollup_rank_topk_source_size(source) else {
                    continue;
                };
                if source_size.selected_bytes < LARGE_BLOCKING_ROLLUP_RANK_SOURCE_BYTES_THRESHOLD {
                    continue;
                }
                large_sources.push((*node_id, source_size, source.file_count(), source.read_column_count()));
            },
            PlanNode::Join(_) => join_count += 1,
            PlanNode::Aggregate(aggregate) => {
                if let Some(grouping_sets) = &aggregate.spec.grouping_sets {
                    grouping_aggregates.push((
                        *node_id,
                        aggregate.spec.group_exprs.len(),
                        grouping_sets.groups.len(),
                        grouping_set_input_sources_reduced_by_prior_aggregate(plan, aggregate.input),
                    ));
                }
            },
            PlanNode::Window(window) => {
                let rank_function_count = window
                    .spec
                    .functions
                    .iter()
                    .filter(|function| matches!(function, WindowFunction::Rank { .. }))
                    .count();
                if rank_function_count > 0 {
                    rank_windows.push((*node_id, rank_function_count));
                }
            },
            PlanNode::Limit(limit) => {
                if let Some(fetch) = limit.spec.fetch
                    && matches!(plan.node(limit.input).map(|node| &node.kind), Some(PlanNode::Sort(_)))
                {
                    final_topks.push((limit.input, *node_id, fetch));
                }
            },
            PlanNode::Filter(filter) => filters.push((*node_id, filter.input)),
            PlanNode::Projection(_)
            | PlanNode::Sort(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => {},
        }
    }

    if join_count == 0 {
        return None;
    }

    let mut selected = None;
    for (window_id, rank_function_count) in &rank_windows {
        let window_inputs = collect_reachable_node_ids_from(plan, *window_id).ok()?;
        let rank_sort_id = match plan.node(*window_id).map(|node| &node.kind) {
            Some(PlanNode::Window(window))
                if matches!(plan.node(window.input).map(|node| &node.kind), Some(PlanNode::Sort(_))) =>
            {
                Some(window.input)
            },
            _ => None,
        };
        let rank_filter_id = filters.iter().find_map(|(filter_id, filter_input)| {
            collect_reachable_node_ids_from(plan, *filter_input)
                .ok()
                .is_some_and(|filter_inputs| filter_inputs.contains(window_id))
                .then_some(*filter_id)
        });
        let final_topk = final_topks.iter().copied().find(|(_, final_limit_id, _)| {
            collect_reachable_node_ids_from(plan, *final_limit_id)
                .ok()
                .is_some_and(|topk_inputs| topk_inputs.contains(window_id))
        });
        for (aggregate_id, grouping_key_count, grouping_set_count, reduced_source_ids) in &grouping_aggregates {
            if !window_inputs.contains(aggregate_id) {
                continue;
            }
            let aggregate_inputs = collect_reachable_node_ids_from(plan, *aggregate_id).ok()?;
            for (large_source_id, source_size, large_source_file_count, large_source_read_column_count) in
                &large_sources
            {
                if !aggregate_inputs.contains(large_source_id) {
                    continue;
                }
                if reduced_source_ids.contains(large_source_id) {
                    continue;
                }
                if selected.is_none_or(|shape: BlockingRollupRankTopKFullScanShape| {
                    source_size.selected_bytes > shape.large_source_selected_bytes
                }) {
                    selected = Some(BlockingRollupRankTopKFullScanShape {
                        aggregate_id: *aggregate_id,
                        grouping_key_count: *grouping_key_count,
                        grouping_set_count: *grouping_set_count,
                        window_id: *window_id,
                        rank_function_count: *rank_function_count,
                        rank_sort_id,
                        rank_filter_id,
                        final_topk_sort_id: final_topk.map(|(sort_id, _, _)| sort_id),
                        final_topk_limit_id: final_topk.map(|(_, limit_id, _)| limit_id),
                        final_topk_fetch: final_topk.map(|(_, _, fetch)| fetch),
                        large_source_id: *large_source_id,
                        large_source_selected_bytes: source_size.selected_bytes,
                        large_source_total_bytes: source_size.total_bytes,
                        large_source_selected_row_groups: source_size.selected_row_groups,
                        large_source_total_row_groups: source_size.total_row_groups,
                        large_source_file_count: *large_source_file_count,
                        large_source_read_column_count: *large_source_read_column_count,
                        source_count,
                        join_count,
                    });
                }
            }
        }
    }

    let shape = selected?;
    Some(BlockingRollupRankTopKFullScanShape { source_count, join_count, ..shape })
}

fn grouping_set_input_sources_reduced_by_prior_aggregate(
    plan: &QueryPlan,
    input: NodeId,
) -> std::collections::BTreeSet<NodeId> {
    let Some(aggregate_inputs) = collect_reachable_node_ids_from(plan, input).ok() else {
        return std::collections::BTreeSet::new();
    };
    aggregate_inputs
        .iter()
        .filter_map(|node_id| plan.node(*node_id))
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(prior) if prior.spec.grouping_sets.is_none() => {
                collect_reachable_node_ids_from(plan, prior.input).ok()
            },
            _ => None,
        })
        .flatten()
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct BlockingRollupRankTopKSourceSize {
    selected_bytes: u64,
    total_bytes: Option<u64>,
    selected_row_groups: Option<usize>,
    total_row_groups: Option<usize>,
}

fn blocking_rollup_rank_topk_source_size(source: &LocalParquetSource) -> Option<BlockingRollupRankTopKSourceSize> {
    if let Some(stats) = source.diagnostics.selected_row_group_stats.as_ref() {
        let selected_bytes = stats.selected_source_bytes()?;
        return Some(BlockingRollupRankTopKSourceSize {
            selected_bytes,
            total_bytes: stats.total_source_bytes(),
            selected_row_groups: Some(stats.selected_row_groups),
            total_row_groups: Some(stats.total_row_groups),
        });
    }
    source.diagnostics.known_bytes.map(|known_bytes| BlockingRollupRankTopKSourceSize {
        selected_bytes: known_bytes,
        total_bytes: Some(known_bytes),
        selected_row_groups: source.diagnostics.selected_row_groups,
        total_row_groups: None,
    })
}

fn optional_u64_fact(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

fn optional_usize_fact(value: Option<usize>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

fn optional_node_fact(value: Option<NodeId>) -> String {
    value.map_or_else(|| "absent".to_string(), |value| value.to_string())
}

pub(super) fn append_shared_resident_subplan_invariants(
    plan: &QueryPlan,
    edges: &[EdgeContract],
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return;
    };
    for (producer, consumers) in consumer_nodes_for_reachable(plan, &reachable) {
        if consumers.len() <= 1 {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        let mut nodes = Vec::with_capacity(consumers.len() + 1);
        nodes.push(producer);
        nodes.extend(consumers.iter().copied());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::SharedResidentSubplan,
            nodes,
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(producer)],
            reservation_requirements: Vec::new(),
            decision: solve_shared_resident_subplan_invariant(plan, edges, producer, &consumers),
        });
    }
}

pub(super) fn append_aggregate_derivation_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    for node in plan.nodes() {
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            continue;
        };
        if !aggregate.spec.group_exprs.is_empty() {
            continue;
        }
        let Some((shared_producer, _projection_count)) = peel_projection_chain_for_capability(plan, aggregate.input)
        else {
            continue;
        };
        if consumers.get(&shared_producer).map_or(0, Vec::len) <= 1 {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::AggregateDerivesFromSharedProducer,
            nodes: vec![shared_producer, aggregate.input, node.id],
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(shared_producer)],
            reservation_requirements: Vec::new(),
            decision: solve_aggregate_derives_from_shared_producer_invariant(plan, shared_producer, node.id),
        });
    }
}

pub(super) fn append_projection_transparent_residency_invariants(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    for node in plan.nodes() {
        if !matches!(node.kind, PlanNode::Projection(_)) {
            continue;
        }
        let Some((peeled_producer, projection_count)) = peel_projection_chain_for_capability(plan, node.id) else {
            continue;
        };
        if projection_count == 0 || peeled_producer == node.id {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ProjectionTransparentResidency,
            nodes: vec![peeled_producer, node.id],
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(peeled_producer)],
            reservation_requirements: Vec::new(),
            decision: solve_projection_transparent_residency_invariant(plan, peeled_producer, node.id),
        });
    }
}

pub(super) fn append_resident_unary_chain_preserves_residency_invariants(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    for node in plan.nodes() {
        if !matches!(node.kind, PlanNode::Projection(_) | PlanNode::Filter(_)) {
            continue;
        }
        let Some((peeled_producer, chain_ids)) = peel_resident_unary_chain_for_capability(plan, node.id) else {
            continue;
        };
        if chain_ids.is_empty() || peeled_producer == node.id {
            continue;
        }
        let mut invariant_nodes = Vec::with_capacity(chain_ids.len() + 1);
        invariant_nodes.push(peeled_producer);
        invariant_nodes.extend(chain_ids.iter().rev().copied());
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ResidentUnaryChainPreservesResidency,
            nodes: invariant_nodes,
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(peeled_producer)],
            reservation_requirements: Vec::new(),
            decision: solve_resident_unary_chain_preserves_residency_invariant(plan, peeled_producer, &chain_ids),
        });
    }
}

pub(super) fn append_final_barrier_order_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    for node in plan.nodes() {
        let PlanNode::Sort(sort) = &node.kind else {
            continue;
        };
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::FinalBarrierOrder,
            nodes: vec![sort.input, node.id],
            required_schedule: ScheduleConstraint::Barrier,
            required_resources: vec![ResidentResource::NodeOutput(sort.input)],
            reservation_requirements: Vec::new(),
            decision: solve_final_barrier_order_invariant(plan, node.id, sort.input),
        });
    }
}

pub(super) fn append_limit_bounded_projection_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    for projection_node in plan.nodes() {
        let PlanNode::Projection(projection) = &projection_node.kind else {
            continue;
        };
        let Some(limit_node) = plan.node(projection.input) else {
            continue;
        };
        let PlanNode::Limit(limit) = &limit_node.kind else {
            continue;
        };
        if limit.spec.fetch.is_none() {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::LimitBoundedProjection,
            nodes: vec![projection.input, projection_node.id],
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(projection.input)],
            reservation_requirements: Vec::new(),
            decision: solve_limit_bounded_projection_invariant(plan, projection.input, projection_node.id),
        });
    }
}

fn solve_limit_bounded_projection_invariant(
    plan: &QueryPlan,
    limit_id: NodeId,
    projection_id: NodeId,
) -> CapabilityDecision {
    let Some(limit_node) = plan.node(limit_id) else {
        return CapabilityDecision::rejected(
            "limit_bounded_projection_missing_limit",
            [("limit_id", limit_id.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    let Some(projection_node) = plan.node(projection_id) else {
        return CapabilityDecision::rejected(
            "limit_bounded_projection_missing_projection",
            [("limit_id", limit_id.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    let PlanNode::Limit(limit) = &limit_node.kind else {
        return CapabilityDecision::rejected(
            "limit_bounded_projection_requires_limit_input",
            [("limit_id", limit_id.to_string()), ("operator", node_operator_fact(&limit_node.kind).to_string())],
        );
    };
    if limit.spec.fetch.is_none() {
        return CapabilityDecision::rejected(
            "limit_bounded_projection_requires_fetch_limit",
            [("limit_id", limit_id.to_string()), ("skip", limit.spec.skip.to_string())],
        );
    }
    let PlanNode::Projection(projection) = &projection_node.kind else {
        return CapabilityDecision::rejected(
            "limit_bounded_projection_requires_projection_consumer",
            [
                ("projection_id", projection_id.to_string()),
                ("operator", node_operator_fact(&projection_node.kind).to_string()),
            ],
        );
    };
    if projection.input != limit_id {
        return CapabilityDecision::rejected(
            "limit_bounded_projection_requires_direct_limit_input",
            [("limit_id", limit_id.to_string()), ("projection_input", projection.input.to_string())],
        );
    }
    let limit_resident = produced_properties(limit_node).contains(&ProducedProperty::ResidentFrame);
    let projection_resident = produced_properties(projection_node).contains(&ProducedProperty::ResidentFrame);
    if !limit_resident || !projection_resident {
        return CapabilityDecision::rejected(
            "limit_bounded_projection_requires_resident_frames",
            [
                ("limit_id", limit_id.to_string()),
                ("projection_id", projection_id.to_string()),
                ("limit_resident", limit_resident.to_string()),
                ("projection_resident", projection_resident.to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn append_mixed_residual_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes: vec![
            shape.root_join_id,
            shape.post_root_projection_id,
            shape.post_root_aggregate_id,
            shape.final_projection_id,
        ],
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        reservation_requirements: Vec::new(),
        decision: CapabilityDecision::Satisfied,
    });
}

pub(super) fn append_hash_join_resident_grouped_aggregate_topk_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    let mut nodes = vec![shape.root_join_id];
    if shape.post_root_projection_id != shape.root_join_id {
        nodes.push(shape.post_root_projection_id);
    }
    nodes.extend([shape.post_root_aggregate_id, shape.final_projection_id, shape.final_sort_id, shape.final_limit_id]);
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes,
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        reservation_requirements: Vec::new(),
        decision: CapabilityDecision::Satisfied,
    });
}

pub(super) fn append_rollup_rank_topk_after_aggregate_join_pushdown_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = rollup_rank_topk_after_aggregate_join_pushdown_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    let mut nodes = vec![
        shape.detail_pre_aggregate_id,
        shape.root_join_id,
        shape.final_grouping_aggregate_id,
        shape.rank_sort_id,
        shape.rank_window_id,
    ];
    if let Some(rank_filter_id) = shape.rank_filter_id {
        nodes.push(rank_filter_id);
    }
    if let Some(consumer_id) = shape.rank_output_residency_consumer_id
        && !nodes.contains(&consumer_id)
    {
        nodes.push(consumer_id);
    }
    if let Some(final_sort_id) = shape.final_sort_id {
        nodes.push(final_sort_id);
    }
    if let Some(final_limit_id) = shape.final_limit_id {
        nodes.push(final_limit_id);
    }
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes,
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        reservation_requirements: rollup_rank_topk_after_aggregate_join_pushdown_subcontract_requirements(&shape),
        decision: CapabilityDecision::Satisfied,
    });
}

fn rollup_rank_topk_after_aggregate_join_pushdown_subcontract_requirements(
    shape: &RollupRankTopKAfterAggregateJoinPushdownShape,
) -> Vec<ReservationRequirement> {
    match shape.final_rollup_width {
        FinalRollupWidthClass::Narrow => Vec::new(),
        FinalRollupWidthClass::Wide => vec![ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "rollup_rank_topk_wide_final_rollup_subcontract",
            "rollup_rank_topk_wide_final_rollup_subcontract",
            ContractQuality::CapabilityProof,
        )],
    }
}

pub(super) fn append_multi_join_aggregate_over_join_subplan_topk_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = multi_join_aggregate_over_join_subplan_topk_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes: {
            let mut nodes = vec![shape.root_join_id];
            nodes.extend(shape.post_root_join_ids);
            if shape.post_root_projection_id != shape.root_join_id {
                nodes.push(shape.post_root_projection_id);
            }
            nodes.extend([shape.final_sort_id, shape.final_limit_id]);
            nodes
        },
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        reservation_requirements: Vec::new(),
        decision: CapabilityDecision::Satisfied,
    });
}

pub(super) fn append_root_topk_join_aggregate_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = root_topk_join_aggregate_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes: {
            let mut nodes = vec![shape.root_join_id];
            nodes.extend(shape.window_input_sort_ids.iter().copied());
            nodes.extend(shape.window_ids.iter().copied());
            if shape.sort_fetch_hash_join_aggregate_subcontract {
                nodes.push(shape.aggregate_id);
            }
            nodes.extend(shape.post_root_projection_ids.iter().rev().copied());
            nodes.push(shape.final_sort_id);
            nodes.extend(shape.final_topk_projection_ids.iter().rev().copied());
            nodes.push(shape.final_limit_id);
            nodes
        },
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        reservation_requirements: root_topk_join_aggregate_subcontract_requirements(&shape),
        decision: CapabilityDecision::Satisfied,
    });
}

fn root_topk_join_aggregate_subcontract_requirements(
    shape: &RootTopKJoinAggregateShape,
) -> Vec<ReservationRequirement> {
    let mut requirements = Vec::new();
    if shape.sort_fetch_hash_join_input_subcontract {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_sort_fetch_hash_join_input_subcontract",
            "root_topk_sort_fetch_hash_join_input_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if shape.sort_fetch_hash_join_aggregate_subcontract {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if !shape.distinct_aggregate_ids.is_empty() {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_distinct_aggregate_subcontract",
            "root_topk_distinct_aggregate_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if !shape.keyed_filtered_semi_join_ids.is_empty() {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_keyed_filtered_semi_join_subcontract",
            "root_topk_keyed_filtered_semi_join_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if !shape.null_filtered_left_join_ids.is_empty() {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_null_filtered_left_join_fan_in_subcontract",
            "root_topk_null_filtered_left_join_fan_in_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if !shape.keyless_or_cross_join_ids.is_empty() {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_keyless_or_cross_join_subcontract",
            "root_topk_keyless_or_cross_join_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if !shape.window_ids.is_empty() {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_window_subcontract",
            "root_topk_window_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if !shape.window_full_outer_join_ids.is_empty() {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_window_full_outer_join_subcontract",
            "root_topk_window_full_outer_join_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    if !shape.keyless_outer_cross_window_join_ids.is_empty() {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_topk_keyless_outer_cross_window_subcontract",
            "root_topk_keyless_outer_cross_window_subcontract",
            ContractQuality::CapabilityProof,
        ));
    }
    requirements
}

pub(super) fn append_root_full_sort_join_aggregate_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = root_full_sort_join_aggregate_shape_decision(plan) else {
        return;
    };
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes: {
            let mut nodes = vec![shape.root_join_id];
            nodes.extend(shape.post_root_projection_ids.iter().rev().copied());
            nodes.push(shape.final_sort_id);
            nodes
        },
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        reservation_requirements: root_full_sort_join_aggregate_subcontract_requirements(&shape),
        decision: CapabilityDecision::Satisfied,
    });
}

fn root_full_sort_join_aggregate_subcontract_requirements(
    shape: &RootFullSortJoinAggregateShape,
) -> Vec<ReservationRequirement> {
    match shape.subcontract {
        RootFullSortJoinAggregateSubcontract::KeyedHashGroupedAggregate => Vec::new(),
        RootFullSortJoinAggregateSubcontract::KeylessOrCrossJoin => vec![ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_full_sort_keyless_or_cross_join_subcontract",
            "root_full_sort_keyless_or_cross_join_subcontract",
            ContractQuality::CapabilityProof,
        )],
        RootFullSortJoinAggregateSubcontract::RepeatedProjectionFanout => vec![ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_full_sort_repeated_projection_fanout_subcontract",
            "root_full_sort_repeated_projection_fanout_subcontract",
            ContractQuality::CapabilityProof,
        )],
        RootFullSortJoinAggregateSubcontract::SharedAggregateFanout => vec![ReservationRequirement::modeled(
            ReservationKind::CapabilityContract,
            "root_full_sort_shared_aggregate_fanout_subcontract",
            "root_full_sort_shared_aggregate_fanout_subcontract",
            ContractQuality::CapabilityProof,
        )],
        RootFullSortJoinAggregateSubcontract::UnsupportedSharedOrFanout => Vec::new(),
    }
}

pub(super) fn append_keyless_aggregate_over_join_subplan_inputs_post_root_stage_order_invariant(
    plan: &QueryPlan,
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(shape) = keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision(plan) else {
        return;
    };
    let mut nodes = Vec::new();
    nodes.push(shape.root_join_id);
    if let Some(projection_id) = shape.post_root_projection_id {
        nodes.push(projection_id);
    }
    if let Some(sort_id) = shape.final_sort_id {
        nodes.push(sort_id);
    }
    let invariant_id = InvariantId(invariants.len());
    invariants.push(SubgraphInvariant {
        invariant_id,
        kind: SubgraphInvariantKind::PostRootStageOrder { root_join_id: shape.root_join_id },
        nodes,
        required_schedule: ScheduleConstraint::Barrier,
        required_resources: vec![ResidentResource::NodeOutput(shape.root_join_id)],
        reservation_requirements: Vec::new(),
        decision: CapabilityDecision::Satisfied,
    });
}

pub(super) fn solve_build_before_probe_invariant(
    plan: &QueryPlan,
    join_id: NodeId,
    build_input: NodeId,
    probe_input: NodeId,
) -> CapabilityDecision {
    let build_nodes = match collect_reachable_node_ids_from(plan, build_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    let probe_nodes = match collect_reachable_node_ids_from(plan, probe_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    if build_nodes.contains(&join_id) || probe_nodes.contains(&join_id) {
        return CapabilityDecision::rejected(
            "build_before_probe_cycle",
            [
                ("join_id", join_id.to_string()),
                ("build_input", build_input.to_string()),
                ("probe_input", probe_input.to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_topological_build_probe_schedule_invariant(plan: &QueryPlan) -> CapabilityDecision {
    let mut scheduler = GraphScheduleProof::new(plan);
    scheduler.schedule_from(plan.sink().input()).map_or_else(|decision| decision, |()| CapabilityDecision::Satisfied)
}

pub(super) fn solve_resident_lifetime_invariant(plan: &QueryPlan, edges: &[EdgeContract]) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    let mut consumers = consumer_nodes_for_reachable(plan, &reachable);
    consumers.entry(plan.sink().input()).or_default();

    for producer in reachable {
        let Some(node) = plan.node(producer) else {
            return CapabilityDecision::rejected("resident_lifetime_missing_node", [("node_id", producer.to_string())]);
        };
        if !produced_properties(node).contains(&ProducedProperty::ResidentFrame) {
            return CapabilityDecision::rejected(
                "resident_lifetime_requires_resident_frame",
                [("node_id", producer.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
            );
        }
        for edge in edges.iter().filter(|edge| edge.producer == producer) {
            if let EdgeConsumer::Node(consumer) = edge.consumer
                && !consumers.get(&producer).is_some_and(|values| values.contains(&consumer))
            {
                continue;
            }
            if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
                return CapabilityDecision::rejected(
                    "resident_lifetime_edge_rejected",
                    [("producer", producer.to_string()), ("consumer", edge_consumer_fact(edge.consumer))],
                );
            }
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_shared_resident_subplan_invariant(
    plan: &QueryPlan,
    edges: &[EdgeContract],
    producer: NodeId,
    consumers: &[NodeId],
) -> CapabilityDecision {
    let Some(node) = plan.node(producer) else {
        return CapabilityDecision::rejected(
            "shared_resident_subplan_missing_node",
            [("node_id", producer.to_string())],
        );
    };
    if !produced_properties(node).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "shared_resident_subplan_requires_resident_frame",
            [("node_id", producer.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        );
    }
    for consumer in consumers {
        let Some(edge) =
            edges.iter().find(|edge| edge.producer == producer && edge.consumer == EdgeConsumer::Node(*consumer))
        else {
            return CapabilityDecision::rejected(
                "shared_resident_subplan_missing_edge",
                [("producer", producer.to_string()), ("consumer", consumer.to_string())],
            );
        };
        if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
            return CapabilityDecision::rejected(
                "shared_resident_subplan_edge_rejected",
                [("producer", producer.to_string()), ("consumer", consumer.to_string())],
            );
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_aggregate_derives_from_shared_producer_invariant(
    plan: &QueryPlan,
    shared_producer: NodeId,
    aggregate_id: NodeId,
) -> CapabilityDecision {
    let Some(producer) = plan.node(shared_producer) else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_missing_node",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    let Some(aggregate_node) = plan.node(aggregate_id) else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_missing_aggregate",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_consumer_not_aggregate",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_requires_resident_frame",
            [("shared_producer", shared_producer.to_string())],
        );
    }

    let Some((peeled_input, _projection_count)) = peel_projection_chain_for_capability(plan, aggregate.input) else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_input_lineage_unresolved",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    if peeled_input != shared_producer {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_requires_dataflow_identity",
            [
                ("shared_producer", shared_producer.to_string()),
                ("aggregate_id", aggregate_id.to_string()),
                ("peeled_input", peeled_input.to_string()),
            ],
        );
    }

    let producer_lineages = match output_lineages_for_node(plan, shared_producer) {
        Ok(lineages) => lineages,
        Err(decision) => return decision,
    };
    for (aggr_index, expr) in aggregate.spec.aggr_exprs.iter().enumerate() {
        let input_lineage = match expression_lineage_for_input(plan, aggregate.input, &expr.expr) {
            Ok(lineage) => lineage,
            Err(decision) => return decision,
        };
        if !producer_lineages.contains(&input_lineage) {
            return CapabilityDecision::rejected(
                "aggregate_shared_producer_requires_input_lineage",
                [
                    ("shared_producer", shared_producer.to_string()),
                    ("aggregate_id", aggregate_id.to_string()),
                    ("aggr_index", aggr_index.to_string()),
                ],
            );
        }
    }
    CapabilityDecision::Satisfied
}

fn output_lineages_for_node(plan: &QueryPlan, node_id: NodeId) -> Result<Vec<OutputLineage>, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "aggregate_shared_producer_missing_node",
            [("shared_producer", node_id.to_string())],
        ));
    };
    (0..node.output_schema().fields().len()).map(|field_index| output_lineage(plan, node_id, field_index)).collect()
}

pub(super) fn solve_projection_transparent_residency_invariant(
    plan: &QueryPlan,
    peeled_producer: NodeId,
    projection_id: NodeId,
) -> CapabilityDecision {
    let Some(producer) = plan.node(peeled_producer) else {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_missing_producer",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    let Some(projection) = plan.node(projection_id) else {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_missing_projection",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    if !matches!(projection.kind, PlanNode::Projection(_)) {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_consumer_not_projection",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    }
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame)
        || !produced_properties(projection).contains(&ProducedProperty::ResidentFrame)
    {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_requires_resident_frames",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_resident_unary_chain_preserves_residency_invariant(
    plan: &QueryPlan,
    peeled_producer: NodeId,
    chain_ids_from_consumer: &[NodeId],
) -> CapabilityDecision {
    let Some(producer) = plan.node(peeled_producer) else {
        return CapabilityDecision::rejected(
            "resident_unary_chain_missing_producer",
            [("peeled_producer", peeled_producer.to_string())],
        );
    };
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "resident_unary_chain_requires_resident_producer",
            [("peeled_producer", peeled_producer.to_string())],
        );
    }
    let mut expected_input = peeled_producer;
    for chain_id in chain_ids_from_consumer.iter().rev() {
        let Some(chain_node) = plan.node(*chain_id) else {
            return CapabilityDecision::rejected(
                "resident_unary_chain_missing_node",
                [("node_id", chain_id.to_string()), ("peeled_producer", peeled_producer.to_string())],
            );
        };
        let actual_input = match &chain_node.kind {
            PlanNode::Projection(projection) => projection.input,
            PlanNode::Filter(filter) => filter.input,
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => {
                return CapabilityDecision::rejected(
                    "resident_unary_chain_requires_filter_projection_nodes",
                    [("node_id", chain_id.to_string()), ("operator", node_operator_fact(&chain_node.kind).to_string())],
                );
            },
        };
        if actual_input != expected_input {
            return CapabilityDecision::rejected(
                "resident_unary_chain_requires_connected_nodes",
                [
                    ("node_id", chain_id.to_string()),
                    ("expected_input", expected_input.to_string()),
                    ("actual_input", actual_input.to_string()),
                ],
            );
        }
        if !produced_properties(chain_node).contains(&ProducedProperty::ResidentFrame) {
            return CapabilityDecision::rejected(
                "resident_unary_chain_requires_resident_nodes",
                [("node_id", chain_id.to_string()), ("peeled_producer", peeled_producer.to_string())],
            );
        }
        expected_input = *chain_id;
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_final_barrier_order_invariant(
    plan: &QueryPlan,
    sort_id: NodeId,
    sort_input: NodeId,
) -> CapabilityDecision {
    let input_nodes = match collect_reachable_node_ids_from(plan, sort_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    if input_nodes.contains(&sort_id) {
        return CapabilityDecision::rejected(
            "final_barrier_order_cycle",
            [("sort_id", sort_id.to_string()), ("sort_input", sort_input.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

pub(super) fn solve_residual_uses_both_inputs_invariant(join_id: NodeId, join: &JoinNode) -> CapabilityDecision {
    let Some(predicate) = &join.spec.residual_predicate else {
        return CapabilityDecision::Satisfied;
    };
    let references_left = expr_references_input(predicate, 0);
    let references_right = expr_references_input(predicate, 1);
    if references_left && references_right {
        return CapabilityDecision::Satisfied;
    }
    CapabilityDecision::rejected(
        "residual_does_not_reference_both_inputs",
        [
            ("join_id", join_id.to_string()),
            ("references_left", references_left.to_string()),
            ("references_right", references_right.to_string()),
            ("expression", predicate.to_string()),
        ],
    )
}

pub(super) fn expr_references_input(expr: &Expr, input_index: usize) -> bool {
    match expr {
        Expr::Column(column) => {
            matches!(column.qualifier(), Some(ColumnQualifier::Input { index }) if *index == input_index)
        },
        Expr::Literal(_) => false,
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Alias { expr, .. } => expr_references_input(expr, input_index),
        Expr::Binary { left, right, .. } => {
            expr_references_input(left, input_index) || expr_references_input(right, input_index)
        },
        Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. } => expr_references_input(expr, input_index),
        Expr::Concat(exprs) => exprs.iter().any(|expr| expr_references_input(expr, input_index)),
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            expr_references_input(left, input_index) || expr_references_input(right, input_index)
        },
        Expr::InList { expr, list, .. } => {
            expr_references_input(expr, input_index) || list.iter().any(|item| expr_references_input(item, input_index))
        },
        Expr::Case { branches, else_expr } => {
            branches.iter().any(|branch| {
                expr_references_input(&branch.when, input_index) || expr_references_input(&branch.then, input_index)
            }) || expr_references_input(else_expr, input_index)
        },
    }
}
