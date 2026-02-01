//! Compiled bounded-pipeline execution facade.
//!
//! Phase D.1.1 wires the compiled graph into bounded runtime dispatch without
//! executing tasks yet. The support checker is intentionally graph-driven: it
//! inspects `CompiledPipeline` task, edge, barrier, sink, and gate contracts and
//! does not route by legacy bounded join family/path vocabulary.

pub(in crate::exec::bounded) mod join_strategy;
mod reservations;
mod scheduler;
mod state;
mod support;
mod tasks;

pub use scheduler::{CompiledPipelineScheduler, CompiledTaskExecutor, CompiledTaskOutput};
pub use state::{CompiledPipelineState, ReleasedCompiledResource};
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};
pub use tasks::{CompiledGpuResource, CompiledGpuTaskExecutor};

use crate::{
    error::Result,
    exec::types::{GpuResidentInputMap, QueryOutput},
    metrics::QueryMetrics,
    pipeline::compile_pipeline,
    plan::{QueryPlan, ResultSink},
    runtime::ExecutionContext,
};

pub(in crate::exec) fn try_execute_compiled_bounded_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<Option<QueryOutput>> {
    if !plan.contains_join() {
        return Ok(None);
    }
    if !matches!(plan.sink(), ResultSink::HostArrow { .. }) {
        return Ok(None);
    }

    let analysis = plan.capability_analysis();
    if analysis.has_rejections() {
        return Ok(None);
    }

    let pipeline = compile_pipeline(plan, &analysis)?;
    match compiled_bounded_query_plan_support(plan, &pipeline)? {
        CompiledBoundedPipelineSupport::Unsupported => Ok(None),
        CompiledBoundedPipelineSupport::Supported => {
            let scheduler = CompiledPipelineScheduler::new(&pipeline)?;
            metrics.record_compiled_pipeline_attempt();
            let (state, output) = {
                let mut executor = CompiledGpuTaskExecutor::new(context, plan, &pipeline, gpu_inputs, metrics)?;
                let state = scheduler.run(&mut executor)?;
                let output = executor.take_host_output()?;
                debug_assert_eq!(executor.reserved_bytes(), 0);
                (state, output)
            };
            metrics.record_compiled_pipeline_tasks_executed(state.completed_task_count() as u64);
            metrics.record_compiled_pipeline_host_arrow_sink();
            Ok(Some(QueryOutput::HostArrow(output)))
        },
    }
}
