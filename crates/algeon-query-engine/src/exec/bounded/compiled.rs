//! Shared compiled bounded-pipeline runtime support.
//!
//! The support checker is intentionally graph-driven: it inspects
//! `CompiledPipeline` task, edge, barrier, and sink contracts and does
//! not route by named bounded join family/path vocabulary. Unsupported graph
//! envelopes fail deterministically instead of falling back to whole-query
//! wrappers.

use std::{collections::BTreeSet, mem};

mod executor;
mod frame_stream;
mod helpers;
pub(in crate::exec::bounded) mod lanes;
mod resource;
mod scheduler;
mod state;

// These unit tests use private admission hooks owned by compiled execution.
#[cfg(test)]
mod execution_tests;

use executor::CompiledGpuTaskExecutor;
pub(in crate::exec::bounded) use frame_stream::{ResidentFrameStreamPayload, ResidentFrameStreamSource};
use lanes::{
    LaneDispatchSupport, is_resident_aggregate_lane_pipeline, is_resident_sort_lane_pipeline, run_resident_lane_borrow,
};
use query_planning::{
    error::Result,
    pipeline::{CompiledPipeline, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentLifetime},
    plan::{QueryPlan, ResultSink},
    policy::PlanningPolicy,
};
#[cfg(test)]
use query_runtime::observability::assert_live_topology_references_resolve;
use query_runtime::{
    observability::{
        BarrierDefinitionRow, CompiledEdgeRow, RuntimeBarrierId, RuntimeExecutionObservationScope,
        RuntimeObservationRecordClass, RuntimePipelineEdgeId, RuntimePlanNodeId, RuntimeResourceDomain,
        RuntimeResourceId, RuntimeResourceRow, RuntimeSinkId, RuntimeTaskId, RuntimeTopologyRecord, SinkDefinitionRow,
        TaskDefinitionRow,
    },
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};
pub(in crate::exec::bounded) use resource::CompiledGpuResource;
use scheduler::CompiledPipelineScheduler;

use crate::exec::{ExecutionMetrics, output::QueryOutput};

/// Runs one compiled pipeline to its sink output on the calling thread.
///
/// The caller is the attempt's root `QueryExecution` pool job, so this blocks
/// that worker for the whole run. A resident-sink pipeline eligible for lane
/// borrowing takes the lane path, which moves `gpu_inputs` into the shared
/// dispatch support; every other pipeline runs the serial interpreter. On
/// `Err` the scheduler has already released every resident payload.
pub(in crate::exec::bounded) fn execute_compiled_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    policy: PlanningPolicy,
    pipeline: &CompiledPipeline,
    mut gpu_inputs: GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    if let Some(scope) = observation_scope {
        record_compiled_pipeline_topology(scope, pipeline, context.runtime()?.device().ordinal());
    }
    let scheduler =
        CompiledPipelineScheduler::new_with_prioritized_join_builds(pipeline, prioritized_join_build_tasks(pipeline))?;
    metrics.record_compiled_pipeline_attempt();
    let lane_support = lane_dispatch_support(context, plan, policy, pipeline, &mut gpu_inputs, metrics);
    let (completed_task_count, output, host_sink) = if let Some(support) = lane_support.as_ref() {
        let (completed, output) = run_resident_lane_borrow(&scheduler, observation_scope, support)?;
        (completed, QueryOutput::GpuResident(output), false)
    } else {
        let mut executor = CompiledGpuTaskExecutor::new_owned(context, plan, policy, pipeline, gpu_inputs, metrics)?;
        let mut state = match observation_scope {
            Some(scope) => scheduler.run_observed(&mut executor, scope)?,
            None => scheduler.run(&mut executor)?,
        };
        let (output, host_sink) = match plan.sink() {
            ResultSink::HostArrow { .. } => (QueryOutput::HostArrow(executor.take_host_output()?), true),
            ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => {
                (QueryOutput::GpuResident(executor.take_gpu_output(&mut state)?), false)
            },
        };
        let completed_task_count = state.completed_task_count();
        (completed_task_count, output, host_sink)
    };
    metrics.record_compiled_pipeline_tasks_executed(completed_task_count as u64);
    if host_sink {
        metrics.record_compiled_pipeline_host_arrow_sink();
    }
    Ok(output)
}

/// Lane dispatch support for eligible pipelines admitted with a
/// fork-stream route. Returns `None` for pipelines that keep the serial
/// scheduler path and for contexts without a route.
fn lane_dispatch_support(
    context: &ExecutionContext,
    plan: &QueryPlan,
    policy: PlanningPolicy,
    pipeline: &CompiledPipeline,
    gpu_inputs: &mut GpuResidentInputMap,
    metrics: &ExecutionMetrics,
) -> Option<LaneDispatchSupport> {
    if !is_resident_aggregate_lane_pipeline(pipeline) && !is_resident_sort_lane_pipeline(pipeline) {
        return None;
    }
    let route = context.attempt_stream_provider_route()?;
    Some(LaneDispatchSupport::new(
        route,
        context.clone(),
        plan.clone(),
        policy,
        pipeline.clone(),
        mem::take(gpu_inputs),
        metrics.clone(),
    ))
}

/// Publishes the task, edge, barrier, sink and resource rows before the first
/// task runs, so lifecycle events recorded during the run resolve against
/// them. Ids are plan-execution scoped; two plans may reuse local task ids.
fn record_compiled_pipeline_topology(
    scope: &RuntimeExecutionObservationScope,
    pipeline: &CompiledPipeline,
    device_ordinal: usize,
) {
    let plan_execution_id = scope.plan_execution_id().clone();
    for task in pipeline.tasks() {
        scope.observation().record_topology(RuntimeObservationRecordClass::TaskDefinition, || {
            RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
                plan_execution_id: plan_execution_id.clone(),
                task_id: RuntimeTaskId::new(task.id().to_string()),
                plan_node_id: task.plan_node().map(|node_id| RuntimePlanNodeId::new(node_id.to_string())),
                task_kind: task.kind().as_str().to_string(),
            })
        });
    }
    for edge in pipeline.edges() {
        scope.observation().record_topology(RuntimeObservationRecordClass::CompiledEdge, || {
            RuntimeTopologyRecord::CompiledEdge(CompiledEdgeRow {
                plan_execution_id: plan_execution_id.clone(),
                edge_id: RuntimePipelineEdgeId::new(edge.id().to_string()),
                from_task_id: RuntimeTaskId::new(edge.from().to_string()),
                to_task_id: RuntimeTaskId::new(edge.to().to_string()),
                edge_kind: edge.kind().as_str().to_string(),
                resource_ids: edge
                    .resources()
                    .iter()
                    .map(|resource_id| RuntimeResourceId::new(resource_id.to_string()))
                    .collect(),
            })
        });
    }
    for barrier in pipeline.barriers() {
        scope.observation().record_topology(RuntimeObservationRecordClass::BarrierDefinition, || {
            RuntimeTopologyRecord::BarrierDefinition(BarrierDefinitionRow {
                plan_execution_id: plan_execution_id.clone(),
                barrier_id: RuntimeBarrierId::new(barrier.id().to_string()),
                barrier_kind: barrier.kind().as_str().to_string(),
                upstream_task_ids: barrier
                    .upstream_tasks()
                    .iter()
                    .map(|task_id| RuntimeTaskId::new(task_id.to_string()))
                    .collect(),
                downstream_task_ids: barrier
                    .downstream_tasks()
                    .iter()
                    .map(|task_id| RuntimeTaskId::new(task_id.to_string()))
                    .collect(),
            })
        });
    }
    let sink_id = RuntimeSinkId::new("sink");
    scope.observation().record_topology(RuntimeObservationRecordClass::SinkDefinition, || {
        RuntimeTopologyRecord::SinkDefinition(SinkDefinitionRow {
            plan_execution_id: plan_execution_id.clone(),
            sink_id: sink_id.clone(),
            sink_kind: pipeline.sink().kind().as_str().to_string(),
            input_task_id: RuntimeTaskId::new(pipeline.sink().input().to_string()),
        })
    });
    for resource in pipeline.resources() {
        let lifetime = match resource.lifetime() {
            ResidentLifetime::TaskLocal { task } => format!("task_local:{}", task.index()),
            ResidentLifetime::BetweenTasks { producer } => format!("between_tasks:{}", producer.index()),
            ResidentLifetime::SinkOutput { producer } => format!("sink_output:{}", producer.index()),
        };
        scope.observation().record_topology(RuntimeObservationRecordClass::Resource, || {
            RuntimeTopologyRecord::Resource(RuntimeResourceRow {
                plan_execution_id: Some(plan_execution_id.clone()),
                resource_id: RuntimeResourceId::new(resource.id().to_string()),
                resource_kind: "resident_frame".to_string(),
                domain: RuntimeResourceDomain::Device {
                    device_ordinal: u32::try_from(device_ordinal).unwrap_or(u32::MAX),
                },
                producer_task_id: Some(RuntimeTaskId::new(resource.producer().to_string())),
                consumer_task_ids: resource
                    .consumers()
                    .iter()
                    .map(|task_id| RuntimeTaskId::new(task_id.to_string()))
                    .collect(),
                lifetime,
            })
        });
    }
}

/// Join builds whose form requires the whole build side resident before any
/// probe work; the scheduler ranks these ahead of other ready tasks.
fn prioritized_join_build_tasks(pipeline: &CompiledPipeline) -> BTreeSet<PipelineTaskId> {
    pipeline
        .tasks()
        .iter()
        .filter(|task| task.kind() == PipelineTaskKind::JoinBuild)
        .filter(|task| task.join_plan().is_some_and(|plan| plan.form().builds_first()))
        .map(PipelineTask::id)
        .collect()
}

#[cfg(test)]
mod topology_tests {
    use std::collections::BTreeSet;

    use query_planning::pipeline::{
        CompiledPipeline, PipelineEdge, PipelineEdgeId, PipelineEdgeKind, PipelineTask, PipelineTaskId,
        ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind, SourceReadPlan,
        TaskPhysical,
    };
    use query_runtime::observability::{
        PlanExecutionId, RuntimeObservationConfig, RuntimeObservationOwner, RuntimeQueryLifecycle,
        RuntimeTopologyRecord,
    };

    use super::*;

    fn topology_pipeline() -> CompiledPipeline {
        let producer = PipelineTaskId::new(0);
        let consumer = PipelineTaskId::new(1);
        let resource = ResidentResourceId::new(0);
        CompiledPipeline::from_parts_unvalidated(
            vec![
                PipelineTask::new(producer, TaskPhysical::SourceRead(SourceReadPlan::FRAME))
                    .with_output_resources(vec![resource]),
                PipelineTask::new(consumer, TaskPhysical::SinkMaterialize).with_input_resources(vec![resource]),
            ],
            vec![
                PipelineEdge::new(PipelineEdgeId::new(0), producer, consumer, PipelineEdgeKind::SinkInput)
                    .with_resources(vec![resource]),
            ],
            vec![],
            vec![ResidentResource::new(
                resource,
                producer,
                vec![consumer],
                ResidentLifetime::BetweenTasks { producer },
            )],
            SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, consumer),
        )
    }

    #[test]
    fn compiled_topology_references_resolve_and_local_ids_are_plan_scoped() {
        let pipeline = topology_pipeline();
        pipeline.validate().unwrap();
        let owner =
            RuntimeObservationOwner::new("topology-query", None, RuntimeObservationConfig::bounded(128, 128 * 1024));
        let first = RuntimeQueryLifecycle::from_owner(owner.clone(), PlanExecutionId::new("plan-a"));
        let second = RuntimeQueryLifecycle::from_owner(owner.clone(), PlanExecutionId::new("plan-b"));
        record_compiled_pipeline_topology(&first.execution_scope(), &pipeline, 0);
        record_compiled_pipeline_topology(&second.execution_scope(), &pipeline, 0);
        let snapshot = owner.live_snapshot().unwrap();
        assert_live_topology_references_resolve(&snapshot);

        let tasks = snapshot
            .topology()
            .filter_map(|row| match row {
                RuntimeTopologyRecord::TaskDefinition(row) => {
                    Some((row.plan_execution_id.clone(), row.task_id.clone()))
                },
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(tasks.len(), 4, "two plans may reuse local task ids without collision");
        for row in snapshot.topology() {
            match row {
                RuntimeTopologyRecord::CompiledEdge(row) => {
                    assert!(tasks.contains(&(row.plan_execution_id.clone(), row.from_task_id.clone())));
                    assert!(tasks.contains(&(row.plan_execution_id.clone(), row.to_task_id.clone())));
                },
                RuntimeTopologyRecord::Resource(row) => {
                    let producer = row.producer_task_id.clone().expect("resource producer");
                    let plan_execution_id = row.plan_execution_id.clone().expect("compiled resource plan scope");
                    assert!(tasks.contains(&(plan_execution_id.clone(), producer)));
                    for consumer in &row.consumer_task_ids {
                        assert!(tasks.contains(&(plan_execution_id.clone(), consumer.clone())));
                    }
                    assert_eq!(row.lifetime, "between_tasks:0");
                },
                _ => {},
            }
        }
    }
}
