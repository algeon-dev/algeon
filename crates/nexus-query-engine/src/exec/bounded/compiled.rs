//! Shared compiled bounded-pipeline runtime support.
//!
//! The support checker is intentionally graph-driven: it inspects
//! `CompiledPipeline` task, edge, barrier, sink, and gate contracts and does
//! not route by named bounded join family/path vocabulary. Unsupported graph
//! envelopes fail deterministically instead of falling back to whole-query
//! wrappers.

use std::collections::BTreeSet;

mod scheduler;
mod state;
mod tasks;

pub use scheduler::{
    CompiledPipelineScheduler, CompiledSchedulingCycle, CompiledTaskAdmission, CompiledTaskExecutor, CompiledTaskOutput,
};
pub use state::{CompiledPipelineState, ReleasedCompiledResource};
pub use tasks::{CompiledGpuResource, CompiledGpuTaskExecutor};
pub(in crate::exec::bounded) use tasks::{ResidentFrameStreamPayload, ResidentFrameStreamSource};
#[cfg(test)]
pub(crate) use tasks::{ResidentFrameStreamProducerTestHarness, YieldedFrameCounterTestHarness};

use crate::{
    error::Result,
    exec::output::QueryOutput,
    metrics::QueryMetrics,
    observability::{
        BarrierDefinitionRow, CompiledEdgeRow, ReservationGateRow, RuntimeBarrierId, RuntimePipelineEdgeId,
        RuntimePlanNodeId, RuntimeReservationGateId, RuntimeResourceDomain, RuntimeResourceId, RuntimeResourceRow,
        RuntimeSinkId, RuntimeTaskId, RuntimeTopologyRecord, SinkDefinitionRow, TaskDefinitionRow,
    },
    pipeline::{CompiledPipeline, PipelineTaskId, PipelineTaskKind, ReservationGateOwner, ResidentLifetime},
    plan::{JoinType, PlanNode, QueryPlan, ResultSink},
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

pub(in crate::exec::bounded) fn execute_compiled_pipeline(
    context: &ExecutionContext,
    plan: &QueryPlan,
    pipeline: &CompiledPipeline,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    if let Some(scope) = observation_scope {
        record_compiled_pipeline_topology(scope, pipeline, context.runtime()?.device().ordinal());
    }
    let scheduler = CompiledPipelineScheduler::new_with_prioritized_join_builds(
        pipeline,
        prioritized_join_build_tasks(plan, pipeline),
    )?;
    metrics.record_compiled_pipeline_attempt();
    let (completed_task_count, output, host_sink) = {
        let mut executor =
            CompiledGpuTaskExecutor::new(context, plan, pipeline, gpu_inputs, metrics, observation_scope)?;
        let mut state = match observation_scope {
            Some(scope) => scheduler.run_observed(&mut executor, scope)?,
            None => scheduler.run(&mut executor)?,
        };
        let (output, host_sink) = match plan.sink() {
            ResultSink::HostArrow { .. } => (QueryOutput::HostArrow(executor.take_host_output()?), true),
            ResultSink::GpuResident { .. } => (QueryOutput::GpuResident(executor.take_gpu_output(&mut state)?), false),
        };
        let completed_task_count = state.completed_task_count();
        let reservations = executor.into_reservations();
        debug_assert_eq!(reservations.reserved_bytes(), 0);
        (completed_task_count, output, host_sink)
    };
    metrics.record_compiled_pipeline_tasks_executed(completed_task_count as u64);
    if host_sink {
        metrics.record_compiled_pipeline_host_arrow_sink();
    }
    Ok(output)
}

fn record_compiled_pipeline_topology(
    scope: &crate::observability::RuntimeExecutionObservationScope,
    pipeline: &CompiledPipeline,
    device_ordinal: usize,
) {
    let plan_execution_id = scope.plan_execution_id().clone();
    for task in pipeline.tasks() {
        scope.observation().record_topology(
            crate::observability::RuntimeObservationRecordClass::TaskDefinition,
            || {
                RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
                    plan_execution_id: plan_execution_id.clone(),
                    task_id: RuntimeTaskId::new(task.id().to_string()),
                    plan_node_id: task.plan_node().map(|node_id| RuntimePlanNodeId::new(node_id.to_string())),
                    task_kind: task.kind().as_str().to_string(),
                })
            },
        );
    }
    for edge in pipeline.edges() {
        scope.observation().record_topology(crate::observability::RuntimeObservationRecordClass::CompiledEdge, || {
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
                gate_ids: edge
                    .gates()
                    .iter()
                    .map(|gate_id| RuntimeReservationGateId::new(gate_id.to_string()))
                    .collect(),
            })
        });
    }
    for barrier in pipeline.barriers() {
        scope.observation().record_topology(
            crate::observability::RuntimeObservationRecordClass::BarrierDefinition,
            || {
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
                    gate_ids: barrier
                        .gates()
                        .iter()
                        .map(|gate_id| RuntimeReservationGateId::new(gate_id.to_string()))
                        .collect(),
                })
            },
        );
    }
    let sink_id = RuntimeSinkId::new("sink");
    scope.observation().record_topology(crate::observability::RuntimeObservationRecordClass::SinkDefinition, || {
        RuntimeTopologyRecord::SinkDefinition(SinkDefinitionRow {
            plan_execution_id: plan_execution_id.clone(),
            sink_id: sink_id.clone(),
            sink_kind: pipeline.sink().kind().as_str().to_string(),
            input_task_id: RuntimeTaskId::new(pipeline.sink().input().to_string()),
            gate_ids: pipeline
                .sink()
                .gates()
                .iter()
                .map(|gate_id| RuntimeReservationGateId::new(gate_id.to_string()))
                .collect(),
        })
    });
    for resource in pipeline.resources() {
        let lifetime = match resource.lifetime() {
            ResidentLifetime::TaskLocal { task } => format!("task_local:{}", task.index()),
            ResidentLifetime::BetweenTasks { producer } => format!("between_tasks:{}", producer.index()),
            ResidentLifetime::SinkOutput { producer } => format!("sink_output:{}", producer.index()),
        };
        scope.observation().record_topology(crate::observability::RuntimeObservationRecordClass::Resource, || {
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
    for gate in pipeline.gates() {
        let (owner_kind, owner_id) = match gate.owner() {
            ReservationGateOwner::Task(task) => ("task", task.to_string()),
            ReservationGateOwner::Edge(edge) => ("edge", edge.to_string()),
            ReservationGateOwner::Barrier(barrier) => ("barrier", barrier.to_string()),
            ReservationGateOwner::SinkBoundary => ("sink", sink_id.to_string()),
        };
        scope.observation().record_topology(
            crate::observability::RuntimeObservationRecordClass::ReservationGate,
            || {
                RuntimeTopologyRecord::ReservationGate(ReservationGateRow {
                    plan_execution_id: plan_execution_id.clone(),
                    gate_id: RuntimeReservationGateId::new(gate.id().to_string()),
                    owner_kind: owner_kind.to_string(),
                    owner_id,
                    reservation_kind: gate.reservation_kind().as_str().to_string(),
                    requirement_scope: gate.requirement().scope.as_str().to_string(),
                    memory_boundary: gate.memory_boundary().to_string(),
                    required_contract: gate.required_contract().to_string(),
                    contract_status: gate.status().as_str().to_string(),
                    contract_quality: gate.quality().as_str().to_string(),
                })
            },
        );
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

#[cfg(test)]
mod topology_tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        capability::{ContractQuality, ContractStatus, ReservationRequirementScope},
        observability::{PlanExecutionId, RuntimeObservationConfig, RuntimeObservationOwner, RuntimeQueryLifecycle},
        pipeline::{
            PipelineEdge, PipelineEdgeId, PipelineEdgeKind, PipelineTask, ReservationGate, ReservationGateId,
            ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
        },
        runtime::ReservationKind,
    };

    fn topology_pipeline() -> CompiledPipeline {
        let producer = PipelineTaskId::new(0);
        let consumer = PipelineTaskId::new(1);
        let resource = ResidentResourceId::new(0);
        let gate = ReservationGateId::new(0);
        CompiledPipeline::new(
            vec![
                PipelineTask::new(producer, PipelineTaskKind::SourceRead).with_output_resources(vec![resource]),
                PipelineTask::new(consumer, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![resource]),
            ],
            vec![
                PipelineEdge::new(PipelineEdgeId::new(0), producer, consumer, PipelineEdgeKind::SinkInput)
                    .with_resources(vec![resource])
                    .with_gates(vec![gate]),
            ],
            vec![],
            vec![ResidentResource::new(
                resource,
                producer,
                vec![consumer],
                ResidentLifetime::BetweenTasks { producer },
            )],
            vec![ReservationGate::new(
                gate,
                ReservationGateOwner::Edge(PipelineEdgeId::new(0)),
                ReservationKind::SourceRead,
                ReservationRequirementScope::Edge,
                "source_read",
                "bounded_read",
                ContractStatus::Modeled,
                ContractQuality::ExactPreflight,
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
        crate::observability::assert_live_topology_references_resolve(&snapshot);

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
                RuntimeTopologyRecord::ReservationGate(row) if row.owner_kind == "edge" => {
                    assert!(snapshot.topology().any(|candidate| matches!(
                        candidate,
                        RuntimeTopologyRecord::CompiledEdge(edge)
                            if edge.plan_execution_id == row.plan_execution_id
                                && edge.edge_id.as_str() == row.owner_id
                    )));
                },
                _ => {},
            }
        }
    }
}
