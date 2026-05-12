//! Shared compiled bounded-pipeline runtime support.
//!
//! The support checker is intentionally graph-driven: it inspects
//! `CompiledPipeline` task, edge, barrier, sink, and gate contracts and does
//! not route by legacy bounded join family/path vocabulary. Unsupported
//! graph envelopes fail deterministically instead of falling back to legacy
//! whole-query wrappers.

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
    pipeline::CompiledPipeline,
    plan::QueryPlan,
    runtime::ExecutionContext,
};

pub(in crate::exec::bounded) fn execute_compiled_pipeline_to_host(
    context: &ExecutionContext,
    plan: &QueryPlan,
    pipeline: &CompiledPipeline,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let scheduler = CompiledPipelineScheduler::new(pipeline)?;
    metrics.record_compiled_pipeline_attempt();
    let (state, output) = {
        let mut executor = CompiledGpuTaskExecutor::new(context, plan, pipeline, gpu_inputs, metrics)?;
        let state = scheduler.run(&mut executor)?;
        let output = executor.take_host_output()?;
        debug_assert_eq!(executor.reserved_bytes(), 0);
        (state, output)
    };
    metrics.record_compiled_pipeline_tasks_executed(state.completed_task_count() as u64);
    metrics.record_compiled_pipeline_host_arrow_sink();
    Ok(QueryOutput::HostArrow(output))
}
