//! Plan-shape admission for strict bounded capability analysis.
//!
//! Walks already-analyzed node/edge/invariant facts and decides whether the
//! whole [`QueryPlan`] forms a recognized strict-bounded shape (root join,
//! keyless conditional, mixed residual, etc.) or must be rejected with a
//! local reason.

use std::collections::{BTreeMap, BTreeSet};

use super::{graph::*, model::*, shape::*, *};
use crate::ResultSink;

pub(super) fn analyze_root_join_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    match root_join_shape_decision(plan) {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(decision) => decision,
    }
}

pub(super) fn analyze_union_pipeline_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if !reachable.iter().any(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Union(_)))) {
        return CapabilityDecision::rejected(
            "union_pipeline_requires_union_node",
            [("sink_input", plan.sink().input().to_string())],
        );
    }
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_single_join_post_root_pipeline_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    match single_join_post_root_pipeline_shape_decision(plan) {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(decision) => decision,
    }
}

pub(super) fn analyze_closed_multi_join_subgraph_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let structure_decision = closed_multi_join_subgraph_structure_decision(plan);
    if matches!(structure_decision, CapabilityDecision::Rejected { .. }) {
        return structure_decision;
    }
    for kind in [
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
        InvariantSelector::ClosedJoinSubtree,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_closed_multi_join_subgraph_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_multi_join_aggregate_over_join_subplan_topk_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match multi_join_aggregate_over_join_subplan_topk_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.aggregate_id,
            consumer_id: shape.aggregate_projection_id,
        },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_multi_join_aggregate_over_join_subplan_topk_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_root_topk_join_aggregate_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match root_topk_join_aggregate_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    let mut required = shape
        .join_ids
        .iter()
        .copied()
        .map(|join_id| InvariantSelector::BuildBeforeProbe { join_id })
        .collect::<Vec<_>>();
    required.extend(
        shape.shared_producer_ids.iter().copied().map(|producer| InvariantSelector::SharedResidentSubplan { producer }),
    );
    if shape.aggregate_id != shape.aggregate_projection_id {
        required.push(InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.aggregate_id,
            consumer_id: shape.aggregate_projection_id,
        });
    }
    required.extend(
        shape.window_input_sort_ids.iter().copied().map(|sort_id| InvariantSelector::FinalBarrierOrder { sort_id }),
    );
    required.extend([
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
    ]);
    if shape.join_ids.len() > 1 {
        required.extend([
            InvariantSelector::TopologicalBuildProbeSchedule,
            InvariantSelector::ResidentLifetimeDominatesConsumers,
        ]);
    }

    for kind in required {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_root_topk_join_aggregate_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_root_full_sort_join_aggregate_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match root_full_sort_join_aggregate_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };

    if !root_full_sort_subcontract_has_cardinality_proof(plan, &shape) {
        return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
    }
    for join_id in &shape.join_ids {
        let join = match join_node_with_missing_reason(
            plan,
            *join_id,
            "root_full_sort_join_aggregate_missing_node",
            "root_full_sort_join_aggregate_requires_keyed_hash_join",
        ) {
            Ok(join) => join,
            Err(decision) => return decision,
        };
        let Some(capability) = join_node_execution_capability(join) else {
            return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
        };
        if !root_full_sort_join_capability_satisfies_subcontract(&shape, *join_id, join, capability) {
            return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
        }
    }

    let mut required = shape
        .join_ids
        .iter()
        .copied()
        .map(|join_id| InvariantSelector::BuildBeforeProbe { join_id })
        .collect::<Vec<_>>();
    for join_id in &shape.join_ids {
        let Some(PlanNode::Join(join)) = plan.node(*join_id).map(|node| &node.kind) else {
            return root_full_sort_unsupported_subcontract_rejection(plan, &shape);
        };
        if join.spec.residual_predicate.is_some() {
            required.push(InvariantSelector::ResidualUsesBothInputs { join_id: *join_id });
        }
    }
    if shape.aggregate_id != shape.aggregate_projection_id {
        required.push(InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.aggregate_id,
            consumer_id: shape.aggregate_projection_id,
        });
    }
    if matches!(shape.subcontract, RootFullSortJoinAggregateSubcontract::SharedAggregateFanout) {
        required.extend(
            shape
                .shared_aggregate_producer_ids
                .iter()
                .copied()
                .map(|producer| InvariantSelector::SharedResidentSubplan { producer }),
        );
        required.extend(root_full_sort_shared_scalar_aggregate_derivation_invariants(plan, &shape));
    }
    required.extend([
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
    ]);
    if shape.join_ids.len() > 1 {
        required.extend([
            InvariantSelector::TopologicalBuildProbeSchedule,
            InvariantSelector::ResidentLifetimeDominatesConsumers,
        ]);
    }

    for kind in required {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_root_full_sort_join_aggregate_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }

    CapabilityDecision::Satisfied
}

pub(super) fn analyze_exact_statistical_join_final_full_sort_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    match analyze_root_full_sort_join_aggregate_admission(plan, nodes, edges, invariants) {
        CapabilityDecision::Satisfied => {},
        CapabilityDecision::Rejected { reason, facts } => {
            return CapabilityDecision::Rejected { reason, facts };
        },
        CapabilityDecision::Conditional { waiting_on } => {
            return CapabilityDecision::Conditional { waiting_on };
        },
    }
    let Some(invariant) =
        invariants.iter().find(|invariant| invariant.kind == SubgraphInvariantKind::StatisticalJoinFinalSortExactness)
    else {
        return CapabilityDecision::rejected(
            "exact_statistical_join_final_full_sort_requires_shape",
            [("shape", "absent".to_string())],
        );
    };
    match &invariant.decision {
        CapabilityDecision::Satisfied => {},
        CapabilityDecision::Rejected { reason, facts } => {
            return CapabilityDecision::Rejected { reason, facts: facts.clone() };
        },
        CapabilityDecision::Conditional { waiting_on } => {
            return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
        },
    }
    for required_contract in [
        "host_welford_input_materialization_preflight",
        "host_welford_output_upload_reservation",
        "final_full_sort_output_cardinality_bound",
    ] {
        if !invariant.reservation_requirements.iter().any(|requirement| {
            requirement.required_contract == required_contract && requirement.status == ContractStatus::Modeled
        }) {
            return CapabilityDecision::rejected(
                "exact_statistical_join_final_full_sort_missing_contract_gate",
                [("required_contract", required_contract.to_string())],
            );
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_projection_global_limit_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match projection_global_limit_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for join_id in &shape.join_ids {
        let join = match join_node_with_missing_reason(
            plan,
            *join_id,
            "projection_global_limit_missing_node",
            "projection_global_limit_requires_supported_join_fragment",
        ) {
            Ok(join) => join,
            Err(decision) => return decision,
        };
        if join_node_execution_capability(join).is_none() {
            return CapabilityDecision::rejected(
                "projection_global_limit_requires_supported_join_fragment",
                [
                    ("node_id", join_id.to_string()),
                    ("join_type", join.spec.join_type.to_string()),
                    ("key_count", join.spec.keys.len().to_string()),
                    ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                ],
            );
        }
    }

    let mut required = shape
        .join_ids
        .iter()
        .copied()
        .map(|join_id| InvariantSelector::BuildBeforeProbe { join_id })
        .collect::<Vec<_>>();
    if let Some(post_limit_projection_id) = shape.post_limit_projection_id {
        required.push(InvariantSelector::LimitBoundedProjection {
            limit_id: shape.global_limit_id,
            projection_id: post_limit_projection_id,
        });
    }
    if shape.join_ids.len() > 1 {
        required.extend([
            InvariantSelector::TopologicalBuildProbeSchedule,
            InvariantSelector::ResidentLifetimeDominatesConsumers,
        ]);
    }

    for kind in required {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_projection_global_limit_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }

    CapabilityDecision::Satisfied
}

pub(super) fn analyze_projection_scalar_setop_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    _invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let shape = match projection_scalar_setop_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) if decision.rejection_reason() == Some("projection_scalar_setop_rejects_distinct_aggregate") => {
            let mut facts = decision.rejection_facts().cloned().unwrap_or_default();
            facts.insert("blocked_reason", "projection_scalar_setop_rejects_distinct_aggregate".to_string());
            facts.insert("required_remediation", "projection_scalar_setop_cardinality_contract".to_string());
            return CapabilityDecision::Rejected { reason: "projection_scalar_setop_contract_not_implemented", facts };
        },
        Err(decision) => return decision,
    };
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    if projection_scalar_setop_scalar_fanout_has_cardinality_proof(plan, &shape)
        || projection_scalar_setop_setop_dedup_scalar_has_contract(plan, &shape)
        || projection_scalar_setop_scalar_join_aggregate_input_has_contract(plan, &shape)
        || projection_scalar_setop_scalar_join_grouped_aggregate_input_has_contract(plan, &shape)
    {
        return CapabilityDecision::Satisfied;
    }
    CapabilityDecision::rejected(
        "projection_scalar_setop_contract_not_implemented",
        [
            ("final_projection_id", shape.final_projection_id.to_string()),
            ("join_node_ids", node_ids_fact(&shape.join_ids)),
            ("aggregate_node_ids", node_ids_fact(&shape.aggregate_ids)),
            ("shape_kind", shape.kind.as_str().to_string()),
            ("required_remediation", "projection_scalar_setop_cardinality_contract".to_string()),
        ],
    )
}

fn projection_scalar_setop_scalar_fanout_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::ScalarFanout {
        return false;
    }
    if !shape.aggregate_ids.iter().all(|aggregate_id| {
        matches!(
            plan.node(*aggregate_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty()
        )
    }) {
        return false;
    }
    projection_scalar_setop_node_output_at_most_one_row(plan, shape.final_projection_id, &mut BTreeSet::new())
}

fn projection_scalar_setop_node_output_at_most_one_row(
    plan: &QueryPlan,
    node_id: NodeId,
    visiting: &mut BTreeSet<NodeId>,
) -> bool {
    if !visiting.insert(node_id) {
        return false;
    }
    let Some(node) = plan.node(node_id) else {
        visiting.remove(&node_id);
        return false;
    };
    let proven = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                source.scan_limit.is_some_and(|limit| limit <= 1) || source.max_rows.is_some_and(|rows| rows <= 1)
            },
            SourceNode::LiteralBatch(source) => source.row_count() <= 1,
            SourceNode::GpuResident(_) => false,
        },
        PlanNode::Projection(projection) => {
            projection_scalar_setop_node_output_at_most_one_row(plan, projection.input, visiting)
        },
        PlanNode::Filter(filter) => projection_scalar_setop_node_output_at_most_one_row(plan, filter.input, visiting),
        PlanNode::Aggregate(aggregate) => aggregate.spec.group_exprs.is_empty(),
        PlanNode::Limit(limit) => {
            limit.spec.fetch.is_some_and(|fetch| fetch <= 1)
                || projection_scalar_setop_node_output_at_most_one_row(plan, limit.input, visiting)
        },
        PlanNode::Join(join) => projection_scalar_setop_join_output_at_most_one_row(plan, join, visiting),
        PlanNode::Sort(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
    };
    visiting.remove(&node_id);
    proven
}

fn projection_scalar_setop_join_output_at_most_one_row(
    plan: &QueryPlan,
    join: &JoinNode,
    visiting: &mut BTreeSet<NodeId>,
) -> bool {
    if !matches!(join.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Cross) {
        return false;
    }
    let mut left_visiting = visiting.clone();
    let mut right_visiting = visiting.clone();
    projection_scalar_setop_node_output_at_most_one_row(
        plan,
        join.inputs[join_side_index(JoinSide::Left)],
        &mut left_visiting,
    ) && projection_scalar_setop_node_output_at_most_one_row(
        plan,
        join.inputs[join_side_index(JoinSide::Right)],
        &mut right_visiting,
    )
}

fn projection_scalar_setop_setop_dedup_scalar_has_contract(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::SetopDedupScalar {
        return false;
    }
    let Some(final_scalar_aggregate_id) =
        projection_scalar_setop_final_projection_input_scalar_aggregate_id(plan, shape.final_projection_id)
    else {
        return false;
    };

    let mut saw_grouped_dedup_aggregate = false;
    let mut grouped_dedup_aggregate_ids = BTreeSet::new();
    for aggregate_id in &shape.aggregate_ids {
        let Some(PlanNode::Aggregate(aggregate)) = plan.node(*aggregate_id).map(|node| &node.kind) else {
            return false;
        };
        if *aggregate_id == final_scalar_aggregate_id {
            if !aggregate.spec.group_exprs.is_empty() {
                return false;
            }
            continue;
        }
        if aggregate.spec.grouping_sets.is_some()
            || !aggregate.spec.aggr_exprs.is_empty()
            || aggregate.spec.group_exprs.is_empty()
        {
            return false;
        }
        saw_grouped_dedup_aggregate = true;
        grouped_dedup_aggregate_ids.insert(*aggregate_id);
    }

    let mut saw_left_anti_join = false;
    for join_id in &shape.join_ids {
        let Some(PlanNode::Join(join)) = plan.node(*join_id).map(|node| &node.kind) else {
            return false;
        };
        match join.spec.join_type {
            JoinType::Inner => {
                if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeyedHash)) {
                    return false;
                }
            },
            JoinType::LeftAnti => {
                if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeyedFilteredSemiAnti))
                {
                    return false;
                }
                let [left, right] = join.inputs;
                if !grouped_dedup_aggregate_ids.contains(&left) || !grouped_dedup_aggregate_ids.contains(&right) {
                    return false;
                }
                saw_left_anti_join = true;
            },
            JoinType::Left | JoinType::Full | JoinType::LeftSemi | JoinType::Cross | JoinType::LeftMark => {
                return false;
            },
        }
    }

    saw_grouped_dedup_aggregate && saw_left_anti_join
}

fn projection_scalar_setop_scalar_join_aggregate_input_has_contract(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::ScalarJoinAggregateInput {
        return false;
    }
    let Some(final_scalar_aggregate_id) =
        projection_scalar_setop_final_projection_input_scalar_aggregate_id(plan, shape.final_projection_id)
    else {
        return false;
    };
    if shape.aggregate_ids.as_slice() != [final_scalar_aggregate_id] {
        return false;
    }
    shape.join_ids.iter().all(|join_id| {
        matches!(
            plan.node(*join_id).map(|node| &node.kind),
            Some(PlanNode::Join(join))
                if matches!(
                    join_node_execution_capability(join),
                    Some(JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
                )
        )
    })
}

fn projection_scalar_setop_scalar_join_grouped_aggregate_input_has_contract(
    plan: &QueryPlan,
    shape: &ProjectionScalarSetopShape,
) -> bool {
    if shape.kind != ProjectionScalarSetopKind::ScalarJoinGroupedAggregateInput {
        return false;
    }
    if shape.join_ids.len() != 1 {
        return false;
    }
    let Some(final_scalar_aggregate_id) =
        projection_scalar_setop_final_projection_input_scalar_aggregate_id(plan, shape.final_projection_id)
    else {
        return false;
    };
    let grouped_aggregate_ids = shape
        .aggregate_ids
        .iter()
        .copied()
        .filter(|aggregate_id| *aggregate_id != final_scalar_aggregate_id)
        .collect::<Vec<_>>();
    if grouped_aggregate_ids.is_empty() {
        return false;
    }
    if !grouped_aggregate_ids.iter().all(|aggregate_id| {
        matches!(
            plan.node(*aggregate_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate))
                if !aggregate.spec.group_exprs.is_empty()
                    && aggregate.spec.grouping_sets.is_none()
                    && !aggregate.spec.aggr_exprs.is_empty()
        )
    }) {
        return false;
    }
    if !grouped_aggregate_ids
        .iter()
        .all(|aggregate_id| projection_scalar_setop_grouped_aggregate_feeds_join(plan, *aggregate_id, &shape.join_ids))
    {
        return false;
    }
    shape.join_ids.iter().all(|join_id| {
        matches!(
            plan.node(*join_id).map(|node| &node.kind),
            Some(PlanNode::Join(join))
                if matches!(
                    join_node_execution_capability(join),
                    Some(JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
                )
        )
    })
}

fn projection_scalar_setop_grouped_aggregate_feeds_join(
    plan: &QueryPlan,
    aggregate_id: NodeId,
    join_ids: &[NodeId],
) -> bool {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return false;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    consumers.get(&aggregate_id).is_some_and(|consumer_ids| {
        consumer_ids.iter().any(|consumer_id| {
            projection_scalar_setop_consumer_chain_reaches_join(
                plan,
                &consumers,
                *consumer_id,
                join_ids,
                &mut BTreeSet::new(),
            )
        })
    })
}

fn projection_scalar_setop_consumer_chain_reaches_join(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    node_id: NodeId,
    join_ids: &[NodeId],
    visited: &mut BTreeSet<NodeId>,
) -> bool {
    if !visited.insert(node_id) {
        return false;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Join(_)) => join_ids.contains(&node_id),
        Some(PlanNode::Projection(_) | PlanNode::Filter(_)) => consumers.get(&node_id).is_some_and(|next| {
            next.iter().any(|consumer_id| {
                projection_scalar_setop_consumer_chain_reaches_join(plan, consumers, *consumer_id, join_ids, visited)
            })
        }),
        Some(
            PlanNode::Aggregate(_)
            | PlanNode::Source(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_),
        )
        | None => false,
    }
}

fn projection_scalar_setop_final_projection_input_scalar_aggregate_id(
    plan: &QueryPlan,
    final_projection_id: NodeId,
) -> Option<NodeId> {
    let mut current = match plan.node(final_projection_id).map(|node| &node.kind)? {
        PlanNode::Projection(projection) => projection.input,
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
            PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => return Some(current),
            PlanNode::Aggregate(_)
            | PlanNode::Source(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Join(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => return None,
        }
    }
}

fn root_full_sort_subcontract_has_cardinality_proof(plan: &QueryPlan, shape: &RootFullSortJoinAggregateShape) -> bool {
    match shape.subcontract {
        RootFullSortJoinAggregateSubcontract::KeyedHashGroupedAggregate => {
            shape.shared_producer_ids.is_empty()
                && (root_full_sort_aggregate_is_on_sink_path(plan, shape)
                    || root_full_sort_root_join_has_filtered_aggregate_branch(plan, shape))
        },
        RootFullSortJoinAggregateSubcontract::KeylessOrCrossJoin => {
            root_full_sort_keyless_scalar_projection_fanout_has_cardinality_proof(plan, shape)
        },
        RootFullSortJoinAggregateSubcontract::RepeatedProjectionFanout => {
            !shape.shared_producer_ids.is_empty()
                && shape.keyless_or_cross_join_ids.is_empty()
                && root_full_sort_shared_producers_are_projection_fanout(plan, &shape.shared_producer_ids)
        },
        RootFullSortJoinAggregateSubcontract::SharedAggregateFanout => {
            root_full_sort_shared_aggregate_fanout_has_cardinality_proof(plan, shape)
        },
        RootFullSortJoinAggregateSubcontract::UnsupportedSharedOrFanout => false,
    }
}

fn root_full_sort_aggregate_is_on_sink_path(plan: &QueryPlan, shape: &RootFullSortJoinAggregateShape) -> bool {
    let mut current = shape.final_sort_id;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return false;
        }
        let Some(node) = plan.node(current) else {
            return false;
        };
        match &node.kind {
            PlanNode::Sort(sort) => current = sort.input,
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Window(window) if root_full_sort_window_allows_sink_path_grouped_aggregate_search(window) => {
                current = window.input;
            },
            PlanNode::Aggregate(_) => return current == shape.aggregate_id,
            PlanNode::Source(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => return false,
        }
    }
}

fn root_full_sort_root_join_has_filtered_aggregate_branch(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    let Some(PlanNode::Join(join)) = plan.node(shape.root_join_id).map(|node| &node.kind) else {
        return false;
    };
    join.inputs.iter().any(|input| root_full_sort_branch_filters_aggregate(plan, *input, shape.aggregate_id))
}

fn root_full_sort_branch_filters_aggregate(plan: &QueryPlan, mut current: NodeId, aggregate_id: NodeId) -> bool {
    let mut visited = BTreeSet::new();
    let mut saw_filter = false;
    loop {
        if !visited.insert(current) {
            return false;
        }
        let Some(node) = plan.node(current) else {
            return false;
        };
        match &node.kind {
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Filter(filter) => {
                saw_filter = true;
                current = filter.input;
            },
            PlanNode::Aggregate(_) => return current == aggregate_id && saw_filter,
            _ => return false,
        }
    }
}

fn root_full_sort_keyless_scalar_projection_fanout_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    if shape.keyless_or_cross_join_ids.as_slice() != [shape.root_join_id] {
        return false;
    }
    if shape.shared_producer_ids.is_empty()
        || !root_full_sort_shared_producers_are_projection_fanout(plan, &shape.shared_producer_ids)
    {
        return false;
    }
    let Some(PlanNode::Join(join)) = plan.node(shape.root_join_id).map(|node| &node.kind) else {
        return false;
    };
    if !matches!(join_node_execution_capability(join), Some(JoinExecutionCapability::KeylessConditional(_))) {
        return false;
    }
    let Some(scalar_side) = root_full_sort_keyless_join_scalar_side(plan, join) else {
        return false;
    };
    join.spec.output_projection.iter().all(|column| column.side != scalar_side)
}

fn root_full_sort_keyless_join_scalar_side(plan: &QueryPlan, join: &JoinNode) -> Option<JoinSide> {
    [JoinSide::Left, JoinSide::Right]
        .into_iter()
        .find(|side| root_full_sort_input_is_scalar_aggregate(plan, join.inputs[join_side_index(*side)]))
}

fn root_full_sort_input_is_scalar_aggregate(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some((peeled, _)) = peel_resident_unary_chain_for_capability(plan, node_id) else {
        return false;
    };
    matches!(
        plan.node(peeled).map(|node| &node.kind),
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty()
    )
}

fn root_full_sort_shared_producers_are_projection_fanout(plan: &QueryPlan, shared_producer_ids: &[NodeId]) -> bool {
    shared_producer_ids.iter().all(|node_id| {
        matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Projection(_) | PlanNode::Filter(_)))
    })
}

fn root_full_sort_shared_aggregate_fanout_has_cardinality_proof(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    if !shape.keyless_or_cross_join_ids.is_empty()
        || !shape.unsupported_shared_producer_ids.is_empty()
        || !shape.shared_projection_producer_ids.is_empty()
        || shape.shared_aggregate_producer_ids.is_empty()
        || shape.shared_producer_ids != shape.shared_aggregate_producer_ids
        || shape.shared_aggregate_consumer_join_ids.is_empty()
        || !shape.shared_aggregate_consumer_join_ids.iter().all(|join_id| shape.join_ids.contains(join_id))
        || !root_full_sort_all_shared_aggregate_consumers_feed_joins(plan, shape)
    {
        return false;
    }

    shape.shared_aggregate_producer_ids.iter().all(|node_id| {
        matches!(
            plan.node(*node_id).map(|node| &node.kind),
            Some(PlanNode::Aggregate(aggregate)) if !aggregate.spec.group_exprs.is_empty()
        )
    })
}

fn root_full_sort_all_shared_aggregate_consumers_feed_joins(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> bool {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return false;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    shape.shared_aggregate_producer_ids.iter().all(|producer| {
        consumers.get(producer).is_some_and(|producer_consumers| {
            producer_consumers.iter().all(|consumer| {
                root_full_sort_consumer_chain_reaches_join(plan, &consumers, *consumer, &mut BTreeSet::new())
            })
        })
    })
}

fn root_full_sort_consumer_chain_reaches_join(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    node_id: NodeId,
    visited: &mut BTreeSet<NodeId>,
) -> bool {
    if !visited.insert(node_id) {
        return false;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Join(_)) => true,
        Some(PlanNode::Projection(_) | PlanNode::Filter(_) | PlanNode::Window(_)) => {
            consumers.get(&node_id).is_some_and(|next_consumers| {
                next_consumers
                    .iter()
                    .all(|consumer| root_full_sort_consumer_chain_reaches_join(plan, consumers, *consumer, visited))
            })
        },
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty() => {
            consumers.get(&node_id).is_some_and(|next_consumers| {
                next_consumers
                    .iter()
                    .all(|consumer| root_full_sort_consumer_chain_reaches_join(plan, consumers, *consumer, visited))
            })
        },
        Some(
            PlanNode::Aggregate(_)
            | PlanNode::Sort(_)
            | PlanNode::Source(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_),
        )
        | None => false,
    }
}

fn root_full_sort_shared_scalar_aggregate_derivation_invariants(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> Vec<InvariantSelector> {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return Vec::new();
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    let mut pairs = BTreeSet::new();
    for shared_producer in &shape.shared_aggregate_producer_ids {
        if let Some(producer_consumers) = consumers.get(shared_producer) {
            for consumer in producer_consumers {
                root_full_sort_collect_shared_scalar_aggregate_derivations(
                    plan,
                    &consumers,
                    *shared_producer,
                    *consumer,
                    &mut BTreeSet::new(),
                    &mut pairs,
                );
            }
        }
    }
    pairs
        .into_iter()
        .map(|(shared_producer, aggregate_id)| InvariantSelector::AggregateDerivesFromSharedProducer {
            shared_producer,
            aggregate_id,
        })
        .collect()
}

fn root_full_sort_collect_shared_scalar_aggregate_derivations(
    plan: &QueryPlan,
    consumers: &BTreeMap<NodeId, Vec<NodeId>>,
    shared_producer: NodeId,
    node_id: NodeId,
    visited: &mut BTreeSet<NodeId>,
    pairs: &mut BTreeSet<(NodeId, NodeId)>,
) {
    if !visited.insert(node_id) {
        return;
    }
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Aggregate(aggregate)) if aggregate.spec.group_exprs.is_empty() => {
            pairs.insert((shared_producer, node_id));
            if let Some(next_consumers) = consumers.get(&node_id) {
                for consumer in next_consumers {
                    root_full_sort_collect_shared_scalar_aggregate_derivations(
                        plan,
                        consumers,
                        shared_producer,
                        *consumer,
                        visited,
                        pairs,
                    );
                }
            }
        },
        Some(PlanNode::Projection(_) | PlanNode::Filter(_) | PlanNode::Window(_)) => {
            if let Some(next_consumers) = consumers.get(&node_id) {
                for consumer in next_consumers {
                    root_full_sort_collect_shared_scalar_aggregate_derivations(
                        plan,
                        consumers,
                        shared_producer,
                        *consumer,
                        visited,
                        pairs,
                    );
                }
            }
        },
        Some(PlanNode::Join(_))
        | Some(
            PlanNode::Aggregate(_)
            | PlanNode::Sort(_)
            | PlanNode::Source(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_),
        )
        | None => {},
    }
}

fn root_full_sort_join_capability_satisfies_subcontract(
    shape: &RootFullSortJoinAggregateShape,
    join_id: NodeId,
    join: &JoinNode,
    capability: JoinExecutionCapability,
) -> bool {
    match shape.subcontract {
        RootFullSortJoinAggregateSubcontract::KeyedHashGroupedAggregate => {
            matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
                || root_full_sort_keyed_filtered_semi_anti_satisfies_grouped_aggregate_subcontract(join, capability)
        },
        RootFullSortJoinAggregateSubcontract::KeylessOrCrossJoin => {
            if shape.keyless_or_cross_join_ids.contains(&join_id) {
                matches!(
                    capability,
                    JoinExecutionCapability::KeylessConditional(_)
                        | JoinExecutionCapability::KeylessOuterCross
                        | JoinExecutionCapability::ProjectedCross
                )
            } else {
                matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
            }
        },
        RootFullSortJoinAggregateSubcontract::RepeatedProjectionFanout => {
            matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
        },
        RootFullSortJoinAggregateSubcontract::SharedAggregateFanout => {
            matches!(capability, JoinExecutionCapability::KeyedHash | JoinExecutionCapability::KeyedResidualMixed)
        },
        RootFullSortJoinAggregateSubcontract::UnsupportedSharedOrFanout => false,
    }
}

fn root_full_sort_keyed_filtered_semi_anti_satisfies_grouped_aggregate_subcontract(
    join: &JoinNode,
    capability: JoinExecutionCapability,
) -> bool {
    capability == JoinExecutionCapability::KeyedFilteredSemiAnti
        && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
        && join.spec.build_side == JoinSide::Right
        && !join.spec.keys.is_empty()
        && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left)
}

fn root_full_sort_unsupported_subcontract_rejection(
    plan: &QueryPlan,
    shape: &RootFullSortJoinAggregateShape,
) -> CapabilityDecision {
    CapabilityDecision::Rejected {
        reason: "root_full_sort_join_aggregate_rejects_unsupported_subcontract",
        facts: BTreeMap::from([
            ("subcontract", shape.subcontract.as_str().to_string()),
            ("final_sort_id", shape.final_sort_id.to_string()),
            ("sink_kind", root_full_sort_sink_kind_fact(plan.sink()).to_string()),
            ("sink_input_id", plan.sink().input().to_string()),
            ("host_sink_path", format!("{}->{}", shape.final_sort_id, root_full_sort_sink_kind_fact(plan.sink()))),
            ("root_join_id", shape.root_join_id.to_string()),
            ("join_node_ids", node_ids_fact(&shape.join_ids)),
            ("keyless_or_cross_join_node_ids", node_ids_fact(&shape.keyless_or_cross_join_ids)),
            ("shared_producer_node_ids", node_ids_fact(&shape.shared_producer_ids)),
            ("shared_projection_producer_node_ids", node_ids_fact(&shape.shared_projection_producer_ids)),
            ("shared_aggregate_producer_node_ids", node_ids_fact(&shape.shared_aggregate_producer_ids)),
            ("unsupported_shared_producer_node_ids", node_ids_fact(&shape.unsupported_shared_producer_ids)),
            ("shared_aggregate_consumer_join_node_ids", node_ids_fact(&shape.shared_aggregate_consumer_join_ids)),
            ("aggregate_id", shape.aggregate_id.to_string()),
            ("aggregate_projection_id", shape.aggregate_projection_id.to_string()),
            ("required_remediation", "root_full_sort_subcontract_cardinality_proof".to_string()),
        ]),
    }
}

fn root_full_sort_sink_kind_fact(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::GpuResident { .. } => "gpu_resident",
        ResultSink::HostArrow { .. } => "host_arrow",
    }
}

pub(super) fn analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.root_join_id },
        InvariantSelector::ProjectionTransparentResidency {
            peeled_producer: shape.join_subtree_root_id,
            projection_id: shape.join_subtree_projection_id,
        },
        InvariantSelector::ProjectionTransparentResidency {
            peeled_producer: shape.grouped_aggregate_id,
            projection_id: shape.grouped_aggregate_projection_id,
        },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_mixed_residual_resident_grouped_aggregate_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

fn node_ids_fact(node_ids: &[NodeId]) -> String {
    if node_ids.is_empty() {
        return "none".to_string();
    }
    node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

pub(super) fn analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.grouped_aggregate_id,
            consumer_id: shape.grouped_aggregate_projection_id,
        },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_hash_join_resident_grouped_aggregate_topk_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_rollup_rank_topk_after_aggregate_join_pushdown_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match rollup_rank_topk_after_aggregate_join_pushdown_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    let mut required = vec![
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.rank_sort_id },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
    ];
    if let Some(consumer_id) = shape.rank_output_residency_consumer_id {
        required.push(InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.rank_window_id,
            consumer_id,
        });
    }
    if let Some(final_sort_id) = shape.final_sort_id {
        required.push(InvariantSelector::FinalBarrierOrder { sort_id: final_sort_id });
    }
    for kind in required {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_rollup_rank_topk_after_aggregate_join_pushdown_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_keyed_hash_join_resident_subplan_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match keyed_hash_join_resident_subplan_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.joined_grouped_branch_join_id },
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::SharedResidentSubplan { producer: shape.shared_grouped_aggregate_id },
        InvariantSelector::AggregateDerivesFromSharedProducer {
            shared_producer: shape.shared_grouped_aggregate_id,
            aggregate_id: shape.scalar_aggregate_id,
        },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_keyed_hash_resident_subplan_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_keyless_conditional_join_input_subplan_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match keyless_conditional_join_input_subplan_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.root_join_id },
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None if matches!(
                kind,
                InvariantSelector::TopologicalBuildProbeSchedule
                    | InvariantSelector::ResidentLifetimeDominatesConsumers
            ) && shape.reachable_join_count <= 1 => {},
            None => {
                return CapabilityDecision::rejected(
                    "missing_keyless_conditional_input_subplan_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    let mut required = vec![
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.root_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.grouped_aggregate_id,
            consumer_id: shape.grouped_aggregate_projection_id,
        },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.scalar_aggregate_id,
            consumer_id: shape.scalar_aggregate_projection_id,
        },
        InvariantSelector::PostRootStageOrder { root_join_id: shape.root_join_id },
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ];
    if let Some(final_sort_id) = shape.final_sort_id {
        required.push(InvariantSelector::FinalBarrierOrder { sort_id: final_sort_id });
    }
    if let Some(shared_producer) = shape.scalar_aggregate_shared_producer_id {
        required.push(InvariantSelector::AggregateDerivesFromSharedProducer {
            shared_producer,
            aggregate_id: shape.scalar_aggregate_id,
        });
    }

    for kind in required {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_keyless_conditional_aggregate_over_join_subplan_inputs_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

pub(super) fn analyze_multi_join_resident_grouped_aggregate_subquery_filter_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let shape = match multi_join_resident_grouped_aggregate_subquery_filter_shape_decision(plan) {
        Ok(shape) => shape,
        Err(decision) => return decision,
    };
    for kind in [
        InvariantSelector::BuildBeforeProbe { join_id: shape.root_join_id },
        InvariantSelector::BuildBeforeProbe { join_id: shape.residual_join_id },
        InvariantSelector::ResidualUsesBothInputs { join_id: shape.residual_join_id },
        InvariantSelector::ResidentUnaryChainPreservesResidency {
            peeled_producer: shape.grouped_aggregate_id,
            consumer_id: shape.grouped_aggregate_projection_id,
        },
        InvariantSelector::FinalBarrierOrder { sort_id: shape.final_sort_id },
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_multi_join_resident_grouped_aggregate_subquery_filter_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

#[derive(Debug, Clone, Copy)]
enum InvariantSelector {
    BuildBeforeProbe {
        join_id: NodeId,
    },
    TopologicalBuildProbeSchedule,
    /// Mirrors `SubgraphInvariantKind::ResidentLifetimeDominatesConsumers`: all
    /// reachable resident producers must outlive every admitted consumer edge.
    ResidentLifetimeDominatesConsumers,
    ClosedJoinSubtree,
    SharedResidentSubplan {
        producer: NodeId,
    },
    /// Mirrors `SubgraphInvariantKind::AggregateDerivesFromSharedProducer` for
    /// a scalar aggregate whose inputs must trace back to one shared producer.
    AggregateDerivesFromSharedProducer {
        shared_producer: NodeId,
        aggregate_id: NodeId,
    },
    FinalBarrierOrder {
        sort_id: NodeId,
    },
    PostRootStageOrder {
        root_join_id: NodeId,
    },
    /// Mirrors `SubgraphInvariantKind::ProjectionTransparentResidency`: the
    /// named projection may be peeled only if it preserves resident output.
    ProjectionTransparentResidency {
        peeled_producer: NodeId,
        projection_id: NodeId,
    },
    /// Mirrors `SubgraphInvariantKind::ResidentUnaryChainPreservesResidency`:
    /// the selected unary chain must stay connected and resident end to end.
    ResidentUnaryChainPreservesResidency {
        peeled_producer: NodeId,
        consumer_id: NodeId,
    },
    ResidualUsesBothInputs {
        join_id: NodeId,
    },
    LimitBoundedProjection {
        limit_id: NodeId,
        projection_id: NodeId,
    },
}

impl InvariantSelector {
    const fn as_str(self) -> &'static str {
        match self {
            Self::BuildBeforeProbe { .. } => "build_before_probe",
            Self::TopologicalBuildProbeSchedule => "topological_build_probe_schedule",
            Self::ResidentLifetimeDominatesConsumers => "resident_lifetime_dominates_consumers",
            Self::ClosedJoinSubtree => "closed_join_subtree",
            Self::SharedResidentSubplan { .. } => "shared_resident_subplan",
            Self::AggregateDerivesFromSharedProducer { .. } => "aggregate_derives_from_shared_producer",
            Self::FinalBarrierOrder { .. } => "final_barrier_order",
            Self::PostRootStageOrder { .. } => "post_root_stage_order",
            Self::ProjectionTransparentResidency { .. } => "projection_transparent_residency",
            Self::ResidentUnaryChainPreservesResidency { .. } => "resident_unary_chain_preserves_residency",
            Self::ResidualUsesBothInputs { .. } => "residual_uses_both_inputs",
            Self::LimitBoundedProjection { .. } => "limit_bounded_projection",
        }
    }
}

fn solved_invariant_decision(
    invariants: &[SubgraphInvariant],
    selector: InvariantSelector,
) -> Option<&CapabilityDecision> {
    invariants
        .iter()
        .find(|invariant| match (&invariant.kind, selector) {
            (
                SubgraphInvariantKind::BuildBeforeProbe { join_id: left },
                InvariantSelector::BuildBeforeProbe { join_id: right },
            ) => *left == right,
            (
                SubgraphInvariantKind::TopologicalBuildProbeSchedule,
                InvariantSelector::TopologicalBuildProbeSchedule,
            ) => true,
            (
                SubgraphInvariantKind::ResidentLifetimeDominatesConsumers,
                InvariantSelector::ResidentLifetimeDominatesConsumers,
            ) => true,
            (SubgraphInvariantKind::ClosedJoinSubtree { .. }, InvariantSelector::ClosedJoinSubtree) => true,
            (SubgraphInvariantKind::SharedResidentSubplan, InvariantSelector::SharedResidentSubplan { producer }) => {
                invariant.nodes.first().is_some_and(|node| *node == producer)
            },
            (
                SubgraphInvariantKind::AggregateDerivesFromSharedProducer,
                InvariantSelector::AggregateDerivesFromSharedProducer { shared_producer, aggregate_id },
            ) => {
                invariant.nodes.first().is_some_and(|node| *node == shared_producer)
                    && invariant.nodes.last().is_some_and(|node| *node == aggregate_id)
            },
            (SubgraphInvariantKind::FinalBarrierOrder, InvariantSelector::FinalBarrierOrder { sort_id }) => {
                invariant.nodes.last().is_some_and(|node| *node == sort_id)
            },
            (
                SubgraphInvariantKind::PostRootStageOrder { root_join_id: left },
                InvariantSelector::PostRootStageOrder { root_join_id: right },
            ) => *left == right,
            (
                SubgraphInvariantKind::ProjectionTransparentResidency,
                InvariantSelector::ProjectionTransparentResidency { peeled_producer, projection_id },
            ) => invariant.nodes.as_slice() == [peeled_producer, projection_id],
            (
                SubgraphInvariantKind::ResidentUnaryChainPreservesResidency,
                InvariantSelector::ResidentUnaryChainPreservesResidency { peeled_producer, consumer_id },
            ) => {
                invariant.nodes.first().is_some_and(|node| *node == peeled_producer)
                    && invariant.nodes.last().is_some_and(|node| *node == consumer_id)
            },
            (
                SubgraphInvariantKind::ResidualUsesBothInputs { join_id: left },
                InvariantSelector::ResidualUsesBothInputs { join_id: right },
            ) => *left == right,
            (
                SubgraphInvariantKind::LimitBoundedProjection,
                InvariantSelector::LimitBoundedProjection { limit_id, projection_id },
            ) => invariant.nodes.as_slice() == [limit_id, projection_id],
            _ => false,
        })
        .map(|invariant| &invariant.decision)
}

pub(super) fn first_reachable_node_rejection(
    nodes: &[NodeCapability],
    reachable: &BTreeSet<NodeId>,
) -> Option<CapabilityDecision> {
    nodes.iter().find_map(|node| {
        if !reachable.contains(&node.node_id) {
            return None;
        }
        match &node.executable {
            CapabilityDecision::Rejected { reason, facts } => {
                Some(CapabilityDecision::Rejected { reason, facts: facts.clone() })
            },
            _ => None,
        }
    })
}

pub(super) fn first_reachable_edge_rejection(
    edges: &[EdgeContract],
    reachable: &BTreeSet<NodeId>,
) -> Option<CapabilityDecision> {
    edges.iter().find_map(|edge| {
        if !reachable.contains(&edge.producer) {
            return None;
        }
        if let EdgeConsumer::Node(consumer) = edge.consumer
            && !reachable.contains(&consumer)
        {
            return None;
        }
        match &edge.decision {
            CapabilityDecision::Rejected { reason, facts } => {
                Some(CapabilityDecision::Rejected { reason, facts: facts.clone() })
            },
            _ => None,
        }
    })
}
