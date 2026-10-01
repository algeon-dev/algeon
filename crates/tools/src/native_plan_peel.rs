//! Shared helper for locating the admitted native artifact inside a
//! DataFusion physical plan tree.
//!
//! Both the ClickBench operator-stage probe and the shadow-estimate report
//! read the artifact that `AlgeonNativeExec` and `AlgeonNativeComposedExec`
//! execute, so this lookup lives in one place.

use std::sync::Arc;

use algeon_datafusion::native_exec::{AlgeonNativeComposedExec, AlgeonNativeExec};
use datafusion::physical_plan::ExecutionPlan;
use query_engine::admission::AdmittedPlan;

/// Walks a DataFusion physical plan and returns the admitted artifact of the
/// first `AlgeonNativeExec`/`AlgeonNativeComposedExec` node found.
///
/// Returns `None` when the plan never lowered to native execution (for
/// example, a full DataFusion CPU fallback).
pub(crate) fn peel_native_admitted_plan(plan: &dyn ExecutionPlan) -> Option<Arc<AdmittedPlan>> {
    if let Some(native) = plan.downcast_ref::<AlgeonNativeExec>() {
        return Some(Arc::clone(native.admitted_plan_for_internal_consumer()));
    }
    if let Some(native) = plan.downcast_ref::<AlgeonNativeComposedExec>() {
        return Some(Arc::clone(native.admitted_plan_for_internal_consumer()));
    }
    for child in plan.children() {
        if let Some(found) = peel_native_admitted_plan(child.as_ref()) {
            return Some(found);
        }
    }
    None
}
