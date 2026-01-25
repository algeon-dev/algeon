use super::{admission::*, compatibility::*, edge::*, graph::*, invariant::*, model::*, node::*, *};

impl QueryPlan {
    /// Runs capability analysis without changing bounded admission or execution.
    pub fn capability_analysis(&self) -> CapabilityAnalysis {
        analyze_query_capabilities(self)
    }
}

/// Runs capability analysis without changing bounded admission or execution.
pub fn analyze_query_capabilities(plan: &QueryPlan) -> CapabilityAnalysis {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));
    mark_shared_producer_edge_contracts(plan, &mut edges);

    let invariants = analyze_invariants(plan, &edges);
    let requirements = collect_requirements(&nodes, &edges);
    let closed_multi_join_subgraph_admission =
        analyze_closed_multi_join_subgraph_admission(plan, &nodes, &edges, &invariants);
    let multi_join_aggregate_over_join_subplan_topk_admission =
        analyze_multi_join_aggregate_over_join_subplan_topk_admission(plan, &nodes, &edges, &invariants);
    let mixed_keyed_residual_join_resident_grouped_aggregate_input_admission =
        analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission(plan, &nodes, &edges, &invariants);
    let hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission =
        analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission(
            plan,
            &nodes,
            &edges,
            &invariants,
        );
    let keyed_hash_join_resident_subplan_admission =
        analyze_keyed_hash_join_resident_subplan_admission(plan, &nodes, &edges, &invariants);
    CapabilityAnalysis {
        nodes,
        edges,
        invariants,
        requirements,
        closed_multi_join_subgraph_admission,
        multi_join_aggregate_over_join_subplan_topk_admission,
        mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
        hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
        keyed_hash_join_resident_subplan_admission,
        compatibility_family_projection: compatibility_family_projection(plan),
    }
}

/// Runs the closed multi-join proof without compatibility-family projection.
///
/// Bounded admission uses this helper during the Phase 6 migration. Keeping it
/// projection-free avoids routing capability-based admission back through the
/// transitional family catalogue.
pub(crate) fn closed_multi_join_subgraph_admission_decision(plan: &QueryPlan) -> CapabilityDecision {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));

    let invariants = analyze_invariants(plan, &edges);
    analyze_closed_multi_join_subgraph_admission(plan, &nodes, &edges, &invariants)
}

pub(crate) fn admits_closed_multi_join_subgraph(plan: &QueryPlan) -> bool {
    closed_multi_join_subgraph_admission_decision(plan) == CapabilityDecision::Satisfied
}

/// Runs the multi-join aggregate-over-join subplan TopK proof without
/// compatibility-family projection.
pub(crate) fn multi_join_aggregate_over_join_subplan_topk_admission_decision(plan: &QueryPlan) -> CapabilityDecision {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));
    mark_shared_producer_edge_contracts(plan, &mut edges);

    let invariants = analyze_invariants(plan, &edges);
    analyze_multi_join_aggregate_over_join_subplan_topk_admission(plan, &nodes, &edges, &invariants)
}

pub(crate) fn admits_multi_join_aggregate_over_join_subplan_topk_pipeline(plan: &QueryPlan) -> bool {
    multi_join_aggregate_over_join_subplan_topk_admission_decision(plan) == CapabilityDecision::Satisfied
}

/// Runs the mixed residual/resident grouped-aggregate proof without
/// compatibility-family projection.
pub(crate) fn mixed_keyed_residual_join_resident_grouped_aggregate_input_admission_decision(
    plan: &QueryPlan,
) -> CapabilityDecision {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));
    mark_shared_producer_edge_contracts(plan, &mut edges);

    let invariants = analyze_invariants(plan, &edges);
    analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission(plan, &nodes, &edges, &invariants)
}

pub(crate) fn admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan: &QueryPlan) -> bool {
    mixed_keyed_residual_join_resident_grouped_aggregate_input_admission_decision(plan) == CapabilityDecision::Satisfied
}

/// Runs the resident grouped-aggregate TopK proof without compatibility-family
/// projection.
pub(crate) fn hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission_decision(
    plan: &QueryPlan,
) -> CapabilityDecision {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));
    mark_shared_producer_edge_contracts(plan, &mut edges);

    let invariants = analyze_invariants(plan, &edges);
    analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission(
        plan,
        &nodes,
        &edges,
        &invariants,
    )
}

pub(crate) fn admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(
    plan: &QueryPlan,
) -> bool {
    hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission_decision(plan)
        == CapabilityDecision::Satisfied
}

/// Runs the keyed hash join resident-subplan proof without compatibility-family
/// projection.
pub(crate) fn keyed_hash_join_resident_subplan_admission_decision(plan: &QueryPlan) -> CapabilityDecision {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));
    mark_shared_producer_edge_contracts(plan, &mut edges);

    let invariants = analyze_invariants(plan, &edges);
    analyze_keyed_hash_join_resident_subplan_admission(plan, &nodes, &edges, &invariants)
}

pub(crate) fn admits_keyed_hash_join_resident_subplan_pipeline(plan: &QueryPlan) -> bool {
    keyed_hash_join_resident_subplan_admission_decision(plan) == CapabilityDecision::Satisfied
}
