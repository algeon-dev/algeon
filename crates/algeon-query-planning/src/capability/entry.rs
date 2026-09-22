use super::{
    QueryPlan,
    edge::{analyze_input_edges, analyze_sink_edge, mark_shared_producer_edge_contracts},
    graph::collect_reachable_node_ids,
    invariant::analyze_invariants,
    model::CapabilityAnalysis,
    node::analyze_node,
};
use crate::backend::AnalysisContext;

impl QueryPlan {
    /// Runs bounded capability analysis for diagnostics, report projection, and
    /// native bounded-join admission.
    pub fn capability_analysis(&self, ctx: AnalysisContext<'_>) -> CapabilityAnalysis {
        analyze_query_capabilities(self, ctx)
    }
}

/// Runs bounded capability analysis for diagnostics, report projection, and
/// native bounded-join admission.
pub(crate) fn analyze_query_capabilities(plan: &QueryPlan, ctx: AnalysisContext<'_>) -> CapabilityAnalysis {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node, ctx)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));
    mark_shared_producer_edge_contracts(plan, &mut edges);

    let invariants = analyze_invariants(plan, &edges);
    // A malformed plan whose reachability walk fails keeps every node, so the
    // local-rejection projection stays unfiltered exactly as before.
    let reachable_node_ids =
        collect_reachable_node_ids(plan).unwrap_or_else(|_| plan.nodes().iter().map(|node| node.id).collect());
    CapabilityAnalysis { nodes, edges, invariants, policy: ctx.policy(), reachable_node_ids }
}
