//! Shared helper for locating the lowered native `QueryPlan` inside a
//! DataFusion physical plan tree.
//!
//! Both the ClickBench operator-stage probe and the shadow-estimate report
//! need the same DataFusion-free native IR that `NexusNativeExec` and
//! `NexusNativeComposedExec` own, so this lookup lives in one place.

use datafusion::physical_plan::ExecutionPlan;
use datafusion_nexus::native_exec::{NexusNativeComposedExec, NexusNativeExec};
use nexus_query_engine::plan::QueryPlan;

/// Walks a DataFusion physical plan and returns the native `QueryPlan` owned
/// by the first `NexusNativeExec`/`NexusNativeComposedExec` node found.
///
/// Returns `None` when the plan never lowered to native execution (for
/// example, a full DataFusion CPU fallback).
pub(crate) fn peel_native_query_plan(plan: &dyn ExecutionPlan) -> Option<QueryPlan> {
    if let Some(native) = plan.downcast_ref::<NexusNativeExec>() {
        return Some(native.native_plan_for_internal_consumer().clone());
    }
    if let Some(native) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(native.native_plan_for_internal_consumer().clone());
    }
    for child in plan.children() {
        if let Some(found) = peel_native_query_plan(child.as_ref()) {
            return Some(found);
        }
    }
    None
}
