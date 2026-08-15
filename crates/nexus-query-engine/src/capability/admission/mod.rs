//! Plan-shape admission for bounded capability analysis.
//!
//! Walks already-analyzed node/edge/invariant facts and decides whether the
//! whole [`QueryPlan`] forms a recognized bounded shape (root join,
//! keyless conditional, mixed residual, etc.) or must be rejected with a
//! local reason.

use std::collections::{BTreeMap, BTreeSet};

use super::{graph::*, model::*, shape::*, *};
use crate::plan::ResultSink;

mod basic;
mod projection;
mod rejections;
mod resident;
mod root_full_sort;
mod root_topk;

#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::capability::admission) use basic::*;
// Preserve original pub(super) API for capability/* consumers (`use super::admission::*`).
pub(super) use basic::{
    analyze_closed_multi_join_subgraph_admission, analyze_multi_join_aggregate_over_join_subplan_topk_admission,
    analyze_root_join_admission, analyze_single_join_post_root_pipeline_admission, analyze_union_pipeline_admission,
};
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::capability::admission) use projection::*;
pub(super) use projection::{analyze_projection_global_limit_admission, analyze_projection_scalar_setop_admission};
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::capability::admission) use rejections::*;
pub(super) use rejections::{first_reachable_edge_rejection, first_reachable_node_rejection};
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::capability::admission) use resident::*;
pub(super) use resident::{
    analyze_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission,
    analyze_keyed_hash_join_resident_subplan_admission,
    analyze_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
    analyze_keyless_conditional_join_input_subplan_admission,
    analyze_mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
    analyze_multi_join_resident_grouped_aggregate_subquery_filter_admission,
    analyze_rollup_rank_topk_after_aggregate_join_pushdown_admission,
};
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::capability::admission) use root_full_sort::*;
pub(super) use root_full_sort::{
    analyze_exact_statistical_join_final_full_sort_admission, analyze_root_full_sort_join_aggregate_admission,
};
pub(super) use root_topk::analyze_root_topk_join_aggregate_admission;
#[allow(unused_imports)] // sibling resolution via use super::*
pub(in crate::capability::admission) use root_topk::*;
