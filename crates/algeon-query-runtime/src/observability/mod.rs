//! Native typed runtime observations and bounded query-local capture.

mod attempt_report;
mod cache_fill_event;
mod clock;
pub mod graph_observation;
mod jsonl;
mod lifecycle;
mod metrics_recorder;
mod metrics_summary;
mod operator_stage;
mod recorder;
mod runtime_observation;
mod schedule_projection;
mod task_identity;

#[cfg(test)]
pub(crate) use attempt_report::AttemptFailureOrigin;
pub use attempt_report::{
    AttemptAdmissionFact, AttemptAdmissionWaitFact, AttemptAllocatorEventSource, AttemptCapacityDisposition,
    AttemptCleanupDisposition, AttemptDisposition, AttemptEvidenceProvenance, AttemptFailureFactKey,
    AttemptFailureFactValue, AttemptFailureOperation, AttemptFailureStatus, AttemptFailureSummary,
    AttemptFinalDomainUsage, AttemptHealthTransition, AttemptQueryId, AttemptRetryAdvice, AttemptRetryPrerequisite,
    AttemptRetryRecovery, AttemptTimestamps, AttemptTraceContext, CleanupFailurePhase, CleanupFailureSummary,
    MAX_ATTEMPT_QUERY_ID_BYTES, MAX_ATTEMPT_TRACE_CONTEXT_BYTES, QUERY_ATTEMPT_REPORT_SCHEMA, QueryAttemptReport,
    RuntimeTerminalOutcome, SEALED_ATTEMPT_PROJECTION_SCHEMA, SealedAttemptLabels, SealedAttemptProjection,
};
pub(crate) use attempt_report::{
    AttemptAllocatorEvent, AttemptAllocatorEventAction, AttemptAllocatorPoolUsage, AttemptAlternateDeviceEligibility,
    AttemptDeviceHealth, AttemptDeviceHealthStatus, AttemptEvidenceField, AttemptFaultIsolation, AttemptNativeFault,
    AttemptNativeFaultEvidence, AttemptNativeFaultKind, AttemptNativeStatus, AttemptResourceEvidence,
    AttemptTeardownHealth, AttemptTerminalCommit, AttemptTimedDomainUsage, AttemptUnattributedAllocatorEvent,
    has_proven_device_local_fault_evidence,
};
pub use cache_fill_event::LakehouseCacheFillEventSnapshot;
#[doc(hidden)]
pub use cache_fill_event::{LakehouseCacheFillErrorEvent, LakehouseCacheFillEvent, LakehouseCacheFillOutcome};
pub use clock::DeterministicRuntimeObservationClock;
pub(crate) use clock::{RuntimeObservationClock, SystemRuntimeObservationClock};
pub use jsonl::{
    RuntimeObservationExportError, RuntimeObservationExportErrorKind, RuntimeObservationExportErrorStatus,
    write_runtime_observation_jsonl,
};
pub use lifecycle::{
    RuntimeExecutionObservationScope, RuntimePlanExecutionLifecycle, RuntimeQueryLifecycle, RuntimeTaskLifecycle,
    task_attempt_id,
};
pub(crate) use metrics_recorder::ExecutionProgress;
pub use metrics_recorder::MetricsRecorder;
#[cfg(test)]
pub(crate) use metrics_summary::QueryMetricsSummary;
pub(crate) use metrics_summary::{MetricsCaptureMode, MetricsCompleteness, QueryMetricsCheckpoint};
pub use metrics_summary::{SourceReadFact, SourceReadTotals};
pub(crate) use operator_stage::RuntimeExecutionStageContext;
pub use operator_stage::{RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, stage_context_from_execution};
#[cfg(test)]
pub(crate) use recorder::RuntimeObservationDeliveryGate;
pub(crate) use recorder::{
    AttemptReportSealProvider, BudgetWindow, CleanupFailureTally, QueryCleanupSealDelivery,
    in_runtime_observation_delivery,
};
pub use recorder::{
    BudgetWindowPeaks, LiveRuntimeObservationSnapshot, RuntimeEventContext, RuntimeObservationConfig,
    RuntimeObservationHandle, RuntimeObservationOwner, RuntimeObservationRecordDisposition, RuntimeObservationSession,
    SealedRuntimeObservationSnapshot,
};
pub(crate) use runtime_observation::{
    AdmissionLifecycleState, JobDefinitionRow, RuntimeAdmissionObservation, RuntimeClockDomain, RuntimeJobId,
    RuntimeObservationClassLoss, RuntimeObservationFooter, RuntimeObservationManifest, RuntimeObservationReasonLoss,
    RuntimeOperatorStageObservation, RuntimeSourceLocationId, RuntimeStreamLeaseObservation,
    RuntimeTaskLifecycleObservation, StreamLeaseAcquisition,
};
pub use runtime_observation::{
    AdmissionRequestId, BarrierDefinitionRow, CompiledEdgeRow, GpuJobLifecycleState, ObservationSequence, PlanEdgeRow,
    PlanExecutionBoundaryKind, PlanExecutionId, PlanNodeRow, QueryBoundaryKind, RUNTIME_OBSERVATION_JSONL_SCHEMA,
    RuntimeBarrierId, RuntimeCleanupFailureObservation, RuntimeGpuJobObservation, RuntimeObservationLimits,
    RuntimeObservationLossReason, RuntimeObservationPayload, RuntimeObservationRecord, RuntimeObservationRecordClass,
    RuntimeOperatorStageKind, RuntimeOperatorStageState, RuntimePipelineEdgeId, RuntimePlanNodeId,
    RuntimeReservationAction, RuntimeReservationDisposition, RuntimeReservationId, RuntimeReservationObservation,
    RuntimeResourceDomain, RuntimeResourceId, RuntimeResourceRow, RuntimeSinkId, RuntimeSourceReadObservation,
    RuntimeSpanId, RuntimeStreamLeaseScope, RuntimeTimelineEvent, RuntimeTopologyRecord, RuntimeWorkerId,
    SinkDefinitionRow, SourceReadLifecycleState, StreamLeaseLifecycleState, TaskDefinitionRow, TaskLifecycleState,
    WorkerDefinitionRow,
};
pub use schedule_projection::{AttemptComposedScheduleProjection, AttemptId, COMPOSED_SCHEDULE_PROJECTION_SCHEMA};
pub use task_identity::RuntimeTaskId;
pub(crate) use task_identity::{RuntimeAttemptId, TaskAttemptId};

#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub fn assert_live_topology_references_resolve(snapshot: &LiveRuntimeObservationSnapshot) {
    use std::collections::BTreeSet;

    let mut plan_nodes = BTreeSet::new();
    let mut tasks = BTreeSet::new();
    let mut edges = BTreeSet::new();
    let mut barriers = BTreeSet::new();
    let mut sinks = BTreeSet::new();
    let mut jobs = BTreeSet::new();
    let mut workers = BTreeSet::new();
    let mut resources = BTreeSet::new();
    for record in snapshot.topology() {
        let inserted = match record {
            RuntimeTopologyRecord::PlanNode(row) => {
                plan_nodes.insert((row.plan_execution_id.clone(), row.node_id.clone()))
            },
            RuntimeTopologyRecord::TaskDefinition(row) => {
                tasks.insert((row.plan_execution_id.clone(), row.task_id.clone()))
            },
            RuntimeTopologyRecord::CompiledEdge(row) => {
                edges.insert((row.plan_execution_id.clone(), row.edge_id.clone()))
            },
            RuntimeTopologyRecord::BarrierDefinition(row) => {
                barriers.insert((row.plan_execution_id.clone(), row.barrier_id.clone()))
            },
            RuntimeTopologyRecord::SinkDefinition(row) => {
                sinks.insert((row.plan_execution_id.clone(), row.sink_id.clone()))
            },
            RuntimeTopologyRecord::JobDefinition(row) => {
                jobs.insert((row.plan_execution_id.clone(), row.job_id.clone()))
            },
            RuntimeTopologyRecord::WorkerDefinition(row) => {
                workers.insert((row.plan_execution_id.clone(), row.worker_id.clone()))
            },
            RuntimeTopologyRecord::Resource(row) => {
                resources.insert((row.plan_execution_id.clone(), row.resource_id.clone()))
            },
            RuntimeTopologyRecord::PlanEdge(_) => true,
        };
        assert!(inserted, "topology identity must be unique within its plan-execution scope: {record:?}");
    }

    for record in snapshot.topology() {
        match record {
            RuntimeTopologyRecord::PlanEdge(row) => {
                assert!(plan_nodes.contains(&(row.plan_execution_id.clone(), row.from_node_id.clone())));
                assert!(plan_nodes.contains(&(row.plan_execution_id.clone(), row.to_node_id.clone())));
            },
            RuntimeTopologyRecord::TaskDefinition(row) => {
                if let Some(node_id) = &row.plan_node_id {
                    assert!(plan_nodes.contains(&(row.plan_execution_id.clone(), node_id.clone())));
                }
            },
            RuntimeTopologyRecord::CompiledEdge(row) => {
                assert!(tasks.contains(&(row.plan_execution_id.clone(), row.from_task_id.clone())));
                assert!(tasks.contains(&(row.plan_execution_id.clone(), row.to_task_id.clone())));
                for resource_id in &row.resource_ids {
                    assert!(resources.contains(&(Some(row.plan_execution_id.clone()), resource_id.clone())));
                }
            },
            RuntimeTopologyRecord::BarrierDefinition(row) => {
                for task_id in row.upstream_task_ids.iter().chain(&row.downstream_task_ids) {
                    assert!(tasks.contains(&(row.plan_execution_id.clone(), task_id.clone())));
                }
            },
            RuntimeTopologyRecord::SinkDefinition(row) => {
                assert!(tasks.contains(&(row.plan_execution_id.clone(), row.input_task_id.clone())));
            },
            RuntimeTopologyRecord::Resource(row) => {
                if let Some(plan_execution_id) = &row.plan_execution_id {
                    if let Some(producer) = &row.producer_task_id {
                        assert!(tasks.contains(&(plan_execution_id.clone(), producer.clone())));
                    }
                    for consumer in &row.consumer_task_ids {
                        assert!(tasks.contains(&(plan_execution_id.clone(), consumer.clone())));
                    }
                }
            },
            RuntimeTopologyRecord::PlanNode(_)
            | RuntimeTopologyRecord::JobDefinition(_)
            | RuntimeTopologyRecord::WorkerDefinition(_) => {},
        }
    }

    for event in snapshot.events() {
        match &event.payload {
            RuntimeObservationPayload::TaskLifecycle(observation) => {
                let plan = event.plan_execution_id.clone().expect("task event has plan execution scope");
                assert!(tasks.contains(&(plan, observation.attempt_id.task_id.clone())));
            },
            RuntimeObservationPayload::GpuJob(observation) => {
                assert!(jobs.contains(&(event.plan_execution_id.clone(), observation.job_id.clone())));
                if let Some(worker_id) = &observation.worker_id {
                    assert!(workers.contains(&(event.plan_execution_id.clone(), worker_id.clone())));
                }
            },
            RuntimeObservationPayload::StreamLease(observation) => {
                assert!(resources.contains(&(event.plan_execution_id.clone(), observation.lease_id.clone())));
                if let Some(task_id) = &observation.scope.task_id {
                    let plan = event.plan_execution_id.clone().expect("task-correlated stream has plan scope");
                    assert!(tasks.contains(&(plan, task_id.clone())));
                }
                if let Some(worker_id) = &observation.scope.source_worker_id {
                    assert!(workers.iter().any(|(_, candidate)| candidate == worker_id));
                }
                if let Some(job_id) = &observation.scope.gpu_job_id {
                    assert!(jobs.iter().any(|(_, candidate)| candidate == job_id));
                }
                if let Some(worker_id) = &observation.scope.gpu_worker_id {
                    assert!(workers.iter().any(|(_, candidate)| candidate == worker_id));
                }
            },
            RuntimeObservationPayload::ResourceMeasurement(observation) => {
                assert!(resources.contains(&(event.plan_execution_id.clone(), observation.resource_id.clone())));
            },
            RuntimeObservationPayload::SourceRead(observation) => {
                let plan = event.plan_execution_id.clone().expect("source event has plan execution scope");
                assert!(tasks.contains(&(plan.clone(), observation.task_id.clone())));
                assert!(workers.contains(&(Some(plan), observation.worker_id.clone())));
            },
            RuntimeObservationPayload::Reservation(observation) => {
                if let Some(attempt) = &observation.task_attempt_id {
                    let plan = event.plan_execution_id.clone().expect("task reservation has plan execution scope");
                    assert!(tasks.contains(&(plan, attempt.task_id.clone())));
                }
            },
            RuntimeObservationPayload::QueryBoundary { .. }
            | RuntimeObservationPayload::PlanExecutionBoundary { .. }
            | RuntimeObservationPayload::Admission(_)
            | RuntimeObservationPayload::CleanupFailure(_)
            | RuntimeObservationPayload::OperatorStage(_) => {},
        }
    }
}
