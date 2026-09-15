//! Plan-shape admission for bounded capability analysis.
//!
//! Walks already-analyzed node/edge/invariant facts and decides whether the
//! whole [`QueryPlan`] forms a recognized bounded shape (root join,
//! keyless conditional, mixed residual, etc.) or must be rejected with a
//! local reason.

use super::{
    JoinExecutionCapability, JoinNode, JoinSide, JoinType, KeylessConditionalJoinKind, NodeId, PlanNode, QueryPlan,
    SourceNode,
    graph::{collect_reachable_node_ids, consumer_nodes_for_reachable},
    join_node_execution_capability,
    model::{CapabilityDecision, EdgeConsumer, EdgeContract, NodeCapability, SubgraphInvariant, SubgraphInvariantKind},
    shape::{
        ProjectionScalarSetopKind, ProjectionScalarSetopShape, RootFullSortJoinAggregateShape,
        RootFullSortJoinAggregateSubcontract, closed_multi_join_subgraph_structure_decision,
        hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision,
        join_node_with_missing_reason, join_side_index, keyed_hash_join_resident_subplan_shape_decision,
        keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision,
        keyless_conditional_join_input_subplan_shape_decision,
        mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision,
        multi_join_aggregate_over_join_subplan_topk_shape_decision,
        multi_join_resident_grouped_aggregate_subquery_filter_shape_decision, peel_resident_unary_chain_for_capability,
        projection_global_limit_shape_decision, projection_scalar_setop_shape_decision,
        rollup_rank_topk_after_aggregate_join_pushdown_shape_decision, root_full_sort_join_aggregate_shape_decision,
        root_full_sort_window_allows_sink_path_grouped_aggregate_search, root_join_shape_decision,
        root_topk_join_aggregate_shape_decision, single_join_post_root_pipeline_shape_decision,
    },
};
use crate::plan::ResultSink;

mod basic;
mod projection;
mod rejections;
mod resident;
mod root_full_sort;
mod root_topk;

// Preserve original pub(super) API for capability/* consumers (`use super::admission::*`).
pub(super) use basic::{
    analyze_closed_multi_join_subgraph_admission, analyze_multi_join_aggregate_over_join_subplan_topk_admission,
    analyze_root_join_admission, analyze_single_join_post_root_pipeline_admission, analyze_union_pipeline_admission,
};
pub(super) use projection::{analyze_projection_global_limit_admission, analyze_projection_scalar_setop_admission};
pub(super) use rejections::reachable_local_capability_rejection;
pub(in crate::capability::admission) use rejections::{InvariantSelector, node_ids_fact, solved_invariant_decision};
pub(super) use resident::{
    analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
    analyze_keyed_hash_join_resident_subplan_admission,
    analyze_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
    analyze_keyless_conditional_join_input_subplan_admission,
    analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
    analyze_multi_join_resident_grouped_aggregate_subquery_filter_admission,
    analyze_rollup_rank_topk_after_aggregate_join_pushdown_admission,
};
pub(super) use root_full_sort::{
    analyze_exact_statistical_join_final_full_sort_admission, analyze_root_full_sort_join_aggregate_admission,
};
pub(super) use root_topk::analyze_root_topk_join_aggregate_admission;
