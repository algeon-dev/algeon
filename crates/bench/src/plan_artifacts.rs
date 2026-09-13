use std::{fmt::Write as _, sync::Arc};

use algeon_datafusion::native_exec::{AlgeonNativeComposedExec, AlgeonNativeExec};
use datafusion::physical_plan::ExecutionPlan;
use query_engine::{
    admission::bounded_pipeline_decision, mode::StatisticalAggregateExecutionMode, plan::QueryPlan,
    rapids_backend::rapids_analysis_context,
};

/// Appends a fresh bounded-pipeline diagnostic for every native execution
/// wrapper in a physical plan.
///
/// The diagnostic intentionally recompiles the immutable query plan with the
/// default statistical mode. It is evidence for an artifact, not the pipeline
/// used by execution, so it must not mutate or replace the plan being run.
pub(crate) fn append_compiled_pipeline_diagnostic(output: &mut String, plan: &Arc<dyn ExecutionPlan>) {
    let mut count = 0;
    append_compiled_pipelines(output, plan, &mut count);
    if count == 0 {
        output.push_str("none: no native execution-plan wrapper found\n");
    }
}

fn append_compiled_pipelines(output: &mut String, plan: &Arc<dyn ExecutionPlan>, count: &mut usize) {
    let native_plan = plan
        .as_ref()
        .downcast_ref::<AlgeonNativeExec>()
        .map(AlgeonNativeExec::native_plan_for_internal_consumer)
        .or_else(|| {
            plan.as_ref()
                .downcast_ref::<AlgeonNativeComposedExec>()
                .map(AlgeonNativeComposedExec::native_plan_for_internal_consumer)
        });
    if let Some(native_plan) = native_plan {
        *count += 1;
        append_compiled_pipeline(output, *count, native_plan);
    }
    for child in plan.children() {
        append_compiled_pipelines(output, child, count);
    }
}

fn append_compiled_pipeline(output: &mut String, index: usize, plan: &QueryPlan) {
    let _ = writeln!(output, "wrapper={index}");
    let decision =
        bounded_pipeline_decision(plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context());
    match decision {
        Ok(decision) => {
            if let Some(admission) = decision.admission() {
                let _ = writeln!(output, "{}", admission.pipeline().debug_render());
            } else if let Some(rejection) = decision.rejection_projection() {
                let _ = writeln!(output, "not_admitted={rejection:?}");
            } else {
                output.push_str("no_compiled_pipeline\n");
            }
        },
        Err(error) => {
            let _ = writeln!(output, "diagnostic_compile_error={error}");
        },
    }
}
