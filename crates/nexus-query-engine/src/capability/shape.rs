//! Plan-shape catalogues used by strict bounded admission.
//!
//! Each submodule recognises one named shape family (root join, mixed
//! residual, keyless conditional, etc.) and exposes its recognisers to
//! [`super::admission`].

use std::collections::BTreeSet;

use super::{edge::*, expr::*, graph::*, invariant::expr_references_input, model::*, *};

mod access;
mod branch;
mod keyed_hash_resident;
mod keyless_conditional;
mod mixed_residual;
mod post_join_topk;
mod root_join;
mod single_join_post_root;
mod subquery_filter;
mod types;

pub(in crate::capability) use access::*;
pub(in crate::capability) use branch::*;
pub(in crate::capability) use keyed_hash_resident::*;
pub(in crate::capability) use keyless_conditional::*;
pub(in crate::capability) use mixed_residual::*;
pub(in crate::capability) use post_join_topk::*;
pub(in crate::capability) use root_join::*;
pub(in crate::capability) use single_join_post_root::*;
pub(in crate::capability) use subquery_filter::*;
pub(in crate::capability) use types::*;

pub(crate) fn strict_bounded_shape_contract_projection_ids(plan: &QueryPlan) -> BTreeSet<NodeId> {
    let mut ids = BTreeSet::new();

    if let Ok(shape) = mixed_keyed_residual_join_resident_grouped_aggregate_input_shape_decision(plan) {
        ids.insert(shape.join_subtree_projection_id);
        ids.insert(shape.grouped_aggregate_projection_id);
    }
    if let Ok(shape) = hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_shape_decision(plan) {
        ids.insert(shape.grouped_aggregate_projection_id);
    }
    if let Ok(shape) = multi_join_aggregate_over_join_subplan_topk_shape_decision(plan) {
        ids.insert(shape.aggregate_projection_id);
    }
    if let Ok(shape) = keyless_conditional_join_aggregate_over_join_subplan_inputs_shape_decision(plan) {
        ids.insert(shape.grouped_aggregate_projection_id);
        ids.insert(shape.scalar_aggregate_projection_id);
    }
    if let Ok(shape) = multi_join_resident_grouped_aggregate_subquery_filter_shape_decision(plan) {
        ids.insert(shape.grouped_aggregate_projection_id);
    }

    ids
}
