use super::{
    QueryPlan,
    admission::{
        analyze_closed_multi_join_subgraph_admission, analyze_exact_statistical_join_final_full_sort_admission,
        analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
        analyze_keyed_hash_join_resident_subplan_admission,
        analyze_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
        analyze_keyless_conditional_join_input_subplan_admission,
        analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
        analyze_multi_join_aggregate_over_join_subplan_topk_admission,
        analyze_multi_join_resident_grouped_aggregate_subquery_filter_admission,
        analyze_projection_global_limit_admission, analyze_projection_scalar_setop_admission,
        analyze_rollup_rank_topk_after_aggregate_join_pushdown_admission,
        analyze_root_full_sort_join_aggregate_admission, analyze_root_join_admission,
        analyze_root_topk_join_aggregate_admission, analyze_single_join_post_root_pipeline_admission,
        analyze_union_pipeline_admission,
    },
    analyze_aggregate_strategies,
    edge::{analyze_input_edges, analyze_sink_edge, mark_shared_producer_edge_contracts},
    graph::collect_requirements,
    invariant::analyze_invariants,
    model::CapabilityAnalysis,
    node::analyze_node,
};

impl QueryPlan {
    /// Runs bounded capability analysis for diagnostics, report projection, and
    /// native bounded-join admission.
    pub fn capability_analysis(&self) -> CapabilityAnalysis {
        analyze_query_capabilities(self)
    }
}

/// Runs bounded capability analysis for diagnostics, report projection, and
/// native bounded-join admission.
pub fn analyze_query_capabilities(plan: &QueryPlan) -> CapabilityAnalysis {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));
    mark_shared_producer_edge_contracts(plan, &mut edges);

    let invariants = analyze_invariants(plan, &edges);
    let requirements = collect_requirements(&nodes, &edges, &invariants);
    let aggregate_strategies = analyze_aggregate_strategies(plan);
    let root_join_admission = analyze_root_join_admission(plan, &nodes, &edges);
    let single_join_post_root_pipeline_admission =
        analyze_single_join_post_root_pipeline_admission(plan, &nodes, &edges);
    let closed_multi_join_subgraph_admission =
        analyze_closed_multi_join_subgraph_admission(plan, &nodes, &edges, &invariants);
    let multi_join_aggregate_over_join_subplan_topk_admission =
        analyze_multi_join_aggregate_over_join_subplan_topk_admission(plan, &nodes, &edges, &invariants);
    let root_topk_join_aggregate_admission =
        analyze_root_topk_join_aggregate_admission(plan, &nodes, &edges, &invariants);
    let root_full_sort_join_aggregate_admission =
        analyze_root_full_sort_join_aggregate_admission(plan, &nodes, &edges, &invariants);
    let exact_statistical_join_final_full_sort_admission =
        analyze_exact_statistical_join_final_full_sort_admission(plan, &nodes, &edges, &invariants);
    let projection_global_limit_admission =
        analyze_projection_global_limit_admission(plan, &nodes, &edges, &invariants);
    let projection_scalar_setop_admission =
        analyze_projection_scalar_setop_admission(plan, &nodes, &edges, &invariants);
    let mixed_keyed_residual_join_resident_grouped_aggregate_input_admission =
        analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission(plan, &nodes, &edges, &invariants);
    let hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission =
        analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission(
            plan,
            &nodes,
            &edges,
            &invariants,
        );
    let rollup_rank_topk_after_aggregate_join_pushdown_admission =
        analyze_rollup_rank_topk_after_aggregate_join_pushdown_admission(plan, &nodes, &edges, &invariants);
    let keyed_hash_join_resident_subplan_admission =
        analyze_keyed_hash_join_resident_subplan_admission(plan, &nodes, &edges, &invariants);
    let keyless_conditional_join_input_subplan_admission =
        analyze_keyless_conditional_join_input_subplan_admission(plan, &nodes, &edges, &invariants);
    let keyless_conditional_join_aggregate_over_join_subplan_inputs_admission =
        analyze_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission(
            plan,
            &nodes,
            &edges,
            &invariants,
        );
    let multi_join_resident_grouped_aggregate_subquery_filter_admission =
        analyze_multi_join_resident_grouped_aggregate_subquery_filter_admission(plan, &nodes, &edges, &invariants);
    let union_pipeline_admission = analyze_union_pipeline_admission(plan, &nodes, &edges);
    CapabilityAnalysis {
        nodes,
        edges,
        invariants,
        requirements,
        aggregate_strategies,
        root_join_admission,
        single_join_post_root_pipeline_admission,
        closed_multi_join_subgraph_admission,
        multi_join_aggregate_over_join_subplan_topk_admission,
        root_topk_join_aggregate_admission,
        root_full_sort_join_aggregate_admission,
        exact_statistical_join_final_full_sort_admission,
        projection_global_limit_admission,
        projection_scalar_setop_admission,
        mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
        hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
        rollup_rank_topk_after_aggregate_join_pushdown_admission,
        keyed_hash_join_resident_subplan_admission,
        keyless_conditional_join_input_subplan_admission,
        keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
        multi_join_resident_grouped_aggregate_subquery_filter_admission,
        union_pipeline_admission,
    }
}
