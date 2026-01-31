//! Compiled bounded-pipeline execution facade.
//!
//! Phase D.1.1 wires the compiled graph into bounded runtime dispatch without
//! executing tasks yet. The support checker is intentionally graph-driven: it
//! inspects `CompiledPipeline` task, edge, barrier, sink, and gate contracts and
//! does not route by legacy bounded join family/path vocabulary.

mod reservations;
mod scheduler;
mod state;
mod support;
mod tasks;

pub use scheduler::{CompiledPipelineScheduler, CompiledTaskExecutor, CompiledTaskOutput};
pub use state::{CompiledPipelineState, ReleasedCompiledResource};
pub use support::{CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support};
pub use tasks::{CompiledGpuResource, CompiledGpuTaskExecutor};

use crate::{
    error::Result,
    exec::types::{GpuResidentInputMap, QueryOutput},
    metrics::QueryMetrics,
    pipeline::compile_pipeline,
    plan::QueryPlan,
    runtime::ExecutionContext,
};

pub(in crate::exec) fn try_execute_compiled_bounded_pipeline(
    _context: &ExecutionContext,
    plan: &QueryPlan,
    _gpu_inputs: &GpuResidentInputMap,
    _metrics: &mut QueryMetrics,
) -> Result<Option<QueryOutput>> {
    if !plan.contains_join() {
        return Ok(None);
    }

    let analysis = plan.capability_analysis();
    if analysis.has_rejections() {
        return Ok(None);
    }

    let pipeline = compile_pipeline(plan, &analysis)?;
    match compiled_bounded_pipeline_support(&pipeline)? {
        CompiledBoundedPipelineSupport::Unsupported => Ok(None),
        CompiledBoundedPipelineSupport::Supported => {
            // D.1.2 installs graph scheduling and resource-lifetime state.
            // D.1.3 starts executing source/unary tasks, so the production
            // facade still declines the graph before any runtime side effect.
            Ok(None)
        },
    }
}
