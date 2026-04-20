//! Shared compiled bounded-pipeline runtime support.
//!
//! The support checker is intentionally graph-driven: it inspects
//! `CompiledPipeline` task, edge, barrier, sink, and gate contracts and does
//! not route by legacy bounded join family/path vocabulary. Unsupported
//! graph envelopes fail deterministically instead of falling back to legacy
//! whole-query wrappers.

use std::collections::BTreeSet;

mod reservations;
mod scheduler;
mod state;
mod tasks;

pub use scheduler::{
    CompiledPipelineScheduler, CompiledSchedulingCycle, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput,
};
pub use state::{CompiledPipelineState, ReleasedCompiledResource};
pub(in crate::exec::bounded) use tasks::ResidentFrameStreamPayload;
#[allow(unused_imports)]
pub(in crate::exec::bounded) use tasks::ResidentFrameStreamSource;
pub use tasks::{
    CompiledGpuResource, CompiledGpuTaskExecutor, ResidentFrameStreamProducerTestHarness,
    YieldedFrameCounterTestHarness,
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    exec::output::QueryOutput,
    metrics::QueryMetrics,
    pipeline::{CompiledPipeline, PipelineTaskId, PipelineTaskKind},
    plan::{JoinType, PlanNode, QueryPlan, ResultSink},
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

pub(in crate::exec::bounded) fn execute_compiled_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    pipeline: &CompiledPipeline,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    validate_pipeline_sink_matches_plan(plan, pipeline)?;
    let scheduler = CompiledPipelineScheduler::new_with_prioritized_join_builds(
        pipeline,
        prioritized_join_build_tasks(plan, pipeline),
    )?;
    metrics.record_compiled_pipeline_attempt();
    let (completed_task_count, output, host_sink) = {
        let mut executor = CompiledGpuTaskExecutor::new(context, plan, pipeline, gpu_inputs, metrics)?;
        let mut state = scheduler.run(&mut executor)?;
        let (output, host_sink) = match plan.sink() {
            ResultSink::HostArrow { .. } => (QueryOutput::HostArrow(executor.take_host_output()?), true),
            ResultSink::GpuResident { .. } => (QueryOutput::GpuResident(executor.take_gpu_output(&mut state)?), false),
        };
        debug_assert_eq!(executor.reserved_bytes(), 0);
        (state.completed_task_count(), output, host_sink)
    };
    metrics.record_compiled_pipeline_tasks_executed(completed_task_count as u64);
    if host_sink {
        metrics.record_compiled_pipeline_host_arrow_sink();
    }
    Ok(output)
}

fn validate_pipeline_sink_matches_plan(plan: &QueryPlan, pipeline: &CompiledPipeline) -> Result<()> {
    let expected = match plan.sink() {
        ResultSink::HostArrow { .. } => crate::pipeline::SinkBoundaryKind::HostArrowMaterialization,
        ResultSink::GpuResident { .. } => crate::pipeline::SinkBoundaryKind::GpuResidentOutput,
    };
    if pipeline.sink().kind() == expected {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.bounded.compiled.sink_boundary",
        "compiled pipeline sink boundary does not match query plan sink",
    )
    .with_fact("plan_sink_kind", result_sink_kind(plan.sink()))
    .with_fact("pipeline_sink_boundary_kind", pipeline.sink().kind().as_str())
    .with_fact("sink_boundary_task_id", pipeline.sink().input())
    .with_fact("plan_sink_node_id", plan.sink().input()))
}

fn result_sink_kind(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::HostArrow { .. } => "host_arrow",
        ResultSink::GpuResident { .. } => "gpu_resident",
    }
}

fn prioritized_join_build_tasks(plan: &QueryPlan, pipeline: &CompiledPipeline) -> BTreeSet<PipelineTaskId> {
    pipeline
        .tasks()
        .iter()
        .filter(|task| task.kind() == PipelineTaskKind::JoinBuild)
        .filter_map(|task| {
            let node_id = task.plan_node()?;
            let node = plan.node(node_id)?;
            let PlanNode::Join(join) = &node.kind else {
                return None;
            };
            if join.spec.residual_predicate.is_none()
                && matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark)
            {
                Some(task.id())
            } else {
                None
            }
        })
        .collect()
}
