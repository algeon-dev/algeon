//! Shared compiled bounded-pipeline runtime support.
//!
//! The support checker is intentionally graph-driven: it inspects
//! `CompiledPipeline` task, edge, barrier, sink, and gate contracts and does
//! not route by legacy bounded join family/path vocabulary. Unsupported
//! graph envelopes fail deterministically instead of falling back to legacy
//! whole-query wrappers.

use std::collections::BTreeSet;

pub(in crate::exec::bounded) mod join_strategy;
mod reservations;
mod scheduler;
mod state;
mod support;
mod tasks;

pub use scheduler::{CompiledPipelineScheduler, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput};
pub use state::{CompiledPipelineState, ReleasedCompiledResource};
pub use support::{
    CompiledBoundedPipelineSupport, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
};
pub(in crate::exec::bounded) use tasks::ResidentFrameStreamPayload;
#[allow(unused_imports)]
pub(in crate::exec::bounded) use tasks::ResidentFrameStreamSource;
pub use tasks::{
    CompiledGpuResource, CompiledGpuTaskExecutor, ResidentFrameStreamProducerTestHarness,
    YieldedFrameCounterTestHarness,
};

use crate::{
    error::Result,
    exec::types::{GpuResidentInputMap, QueryOutput},
    metrics::QueryMetrics,
    pipeline::{CompiledPipeline, PipelineTaskId, PipelineTaskKind},
    plan::{JoinType, PlanNode, QueryPlan},
    runtime::ExecutionContext,
};

pub(in crate::exec::bounded) fn execute_compiled_pipeline_to_host(
    context: &ExecutionContext,
    plan: &QueryPlan,
    pipeline: &CompiledPipeline,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let scheduler = CompiledPipelineScheduler::new_with_prioritized_join_builds(
        pipeline,
        prioritized_join_build_tasks(plan, pipeline),
    )?;
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
