use std::{fmt::Write as _, sync::Arc};

use algeon_datafusion::native_exec::AlgeonNativeExec;
use datafusion::physical_plan::ExecutionPlan;
use query_engine::admission::AdmittedPlan;

/// Appends the compiled pipeline every native execution wrapper in a physical
/// plan executes, as planning admitted it.
pub(crate) fn append_compiled_pipeline_diagnostic(output: &mut String, plan: &Arc<dyn ExecutionPlan>) {
    let mut count = 0;
    append_compiled_pipelines(output, plan, &mut count);
    if count == 0 {
        output.push_str("none: no native execution-plan wrapper found\n");
    }
}

fn append_compiled_pipelines(output: &mut String, plan: &Arc<dyn ExecutionPlan>, count: &mut usize) {
    let admitted = plan
        .as_ref()
        .downcast_ref::<AlgeonNativeExec>()
        .filter(|native| native.materializes_host())
        .map(AlgeonNativeExec::admitted_plan_for_internal_consumer);
    if let Some(admitted) = admitted {
        *count += 1;
        append_compiled_pipeline(output, *count, admitted);
    }
    for child in plan.children() {
        append_compiled_pipelines(output, child, count);
    }
}

fn append_compiled_pipeline(output: &mut String, index: usize, admitted: &AdmittedPlan) {
    let _ = writeln!(output, "wrapper={index}");
    let _ = writeln!(output, "{}", admitted.pipeline().debug_render());
}
