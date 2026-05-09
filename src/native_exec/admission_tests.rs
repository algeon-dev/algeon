use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use arrow::{
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::StreamExt;
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof,
    error::QueryErrorCode,
    observability::{
        QueryBoundaryKind, RuntimeObservationConfig, RuntimeObservationOwner, RuntimeObservationPayload,
        RuntimeTerminalOutcome,
    },
    plan::QueryPlanBuilder,
    runtime::ExecutionCancellation,
    source::{GpuResidentSource, SourceNode},
};

use super::*;
use crate::{
    backend::{InstalledTestGpuBackend, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
    error::DataFusionErrorExt,
    native_exec::NexusNativeExecutionConfig,
};

#[derive(Clone, Copy, Debug)]
enum LifecyclePlanBehavior {
    Success,
    ExecuteError,
    StreamError,
    CancelledStream,
}

#[derive(Debug)]
struct LifecycleExec {
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    behavior: LifecyclePlanBehavior,
    observed_ticket: Arc<Mutex<Option<u64>>>,
}

impl LifecycleExec {
    fn new(behavior: LifecyclePlanBehavior, observed_ticket: Arc<Mutex<Option<u64>>>) -> Self {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self { schema, properties, behavior, observed_ticket }
    }

    fn typed_error(behavior: LifecyclePlanBehavior) -> DataFusionError {
        let (code, operation, message) = match behavior {
            LifecyclePlanBehavior::ExecuteError => (
                crate::error::ErrorCode::Internal,
                "datafusion_nexus.test.composed_producer",
                "synthetic composed producer failed",
            ),
            LifecyclePlanBehavior::StreamError => (
                crate::error::ErrorCode::Internal,
                "datafusion_nexus.test.composed_consumer",
                "synthetic composed consumer failed",
            ),
            LifecyclePlanBehavior::CancelledStream => (
                crate::error::ErrorCode::Cancelled,
                "datafusion_nexus.test.composed_consumer_cancelled",
                "synthetic composed consumer was cancelled",
            ),
            LifecyclePlanBehavior::Success => unreachable!("success does not produce a typed error"),
        };
        DataFusionError::from(crate::error::Error::new(code, operation, message))
    }
}

impl DisplayAs for LifecycleExec {
    fn fmt_as(&self, _format: DisplayFormatType, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LifecycleExec")
    }
}

impl ExecutionPlan for LifecycleExec {
    fn name(&self) -> &str {
        "LifecycleExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Execution("lifecycle test plan has no children".to_string()))
        }
    }

    fn execute(
        &self,
        _partition: usize,
        task_context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if let Some(ticket) = query_admission_ticket_from_task_context(&task_context) {
            *self.observed_ticket.lock().expect("test ticket observation mutex") = Some(ticket.id());
        }
        match self.behavior {
            LifecyclePlanBehavior::Success => {
                let stream = futures::stream::empty::<DataFusionResult<RecordBatch>>();
                Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
            },
            LifecyclePlanBehavior::ExecuteError => Err(Self::typed_error(self.behavior)),
            LifecyclePlanBehavior::StreamError | LifecyclePlanBehavior::CancelledStream => {
                let error = Self::typed_error(self.behavior);
                let stream = futures::stream::once(async move { Err::<RecordBatch, DataFusionError>(error) });
                Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
            },
        }
    }
}

fn lifecycle_owner(query_id: &'static str) -> RuntimeObservationOwner {
    RuntimeObservationOwner::new(query_id, None, RuntimeObservationConfig::bounded(32, 4096))
}

fn lifecycle_exec(behavior: LifecyclePlanBehavior, observed_ticket: Arc<Mutex<Option<u64>>>) -> Arc<dyn ExecutionPlan> {
    Arc::new(LifecycleExec::new(behavior, observed_ticket))
}

fn assert_terminal_boundary_once(owner: &RuntimeObservationOwner, expected: RuntimeTerminalOutcome) {
    let snapshot = owner.sealed_snapshot().expect("lifecycle owner must be sealed");
    assert_eq!(snapshot.footer().terminal, Some(expected));
    let terminal_count = snapshot
        .events()
        .iter()
        .filter(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::QueryBoundary {
                    boundary: QueryBoundaryKind::Succeeded
                        | QueryBoundaryKind::Failed
                        | QueryBoundaryKind::Cancelled
                        | QueryBoundaryKind::Dropped
                        | QueryBoundaryKind::Panicked
                }
            )
        })
        .count();
    assert_eq!(terminal_count, 1, "terminal lifecycle boundary must be recorded exactly once");
}

fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
    backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority")
}

fn explicit_sink_capability_proof() -> QueryPlanCapabilityProof {
    use arrow::datatypes::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    let plan = builder.finish_with_host_sink(source);
    QueryPlanCapabilityProof::for_query_plan(&plan).expect("an explicit host sink must expose a capability requirement")
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_forwards_opaque_evidence_and_releases_real_ledger_grant_for_successive_admission() {
    let backend = InstalledTestGpuBackend::new();
    let ticket = QueryAdmissionTicket::new(admission_source(&backend), None);
    assert!(ticket.attempt_capability_proof().is_none());

    let context = ticket
        .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
        .await
        .expect("installed backend must acquire a real admission context");
    let facts = context.admission_facts();
    assert!(facts.admission_sequence > 0);
    assert!(facts.query_grant_bytes > 0);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);

    drop(context);
    let next_context = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ticket.acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new()),
    )
    .await
    .expect("dropping the context must eventually return the real ledger grant")
    .expect("released capacity must admit the next context");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);
    drop(next_context);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_propagates_cancellation_to_admission() {
    let backend = InstalledTestGpuBackend::new();
    let ticket = QueryAdmissionTicket::new(admission_source(&backend), None);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), cancellation)
        .await
        .expect_err("cancelled admission must not acquire a real handle");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_failure_clears_round_state_and_allows_a_retry() {
    let backend = InstalledTestGpuBackend::new();
    let ticket = QueryAdmissionTicket::new(admission_source(&backend), None);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), cancellation)
        .await
        .expect_err("cancelled admission must fail before publishing a handle");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);

    {
        let state = ticket.state.lock().expect("ticket state mutex");
        assert!(state.admitting.is_empty(), "a failed admission must not strand its in-flight marker");
        assert_eq!(state.rounds.len(), 1, "the failed task round is retained only until the next acquisition");
        let round = state.rounds.values().next().expect("failed round is registered");
        assert_eq!(round.active_contexts, 0);
        assert_eq!(round.active_streams, 0);
        assert!(round.handle.is_none(), "a rejected admission cannot retain a query handle");
    }

    let retry = tokio::time::timeout(
        Duration::from_secs(5),
        ticket.acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new()),
    )
    .await
    .expect("a retry must not wait on the rejected round")
    .expect("the same ticket can admit after a rejected attempt");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);
    drop(retry);

    let state = ticket.state.lock().expect("ticket state mutex");
    assert!(state.admitting.is_empty());
    assert!(state.rounds.is_empty(), "retry cleanup must remove the completed round exactly once");
}

#[tokio::test(flavor = "current_thread")]
async fn pending_context_cancellation_keeps_the_shared_grant_until_the_last_context_drops() {
    let backend = InstalledTestGpuBackend::new();
    let owner = lifecycle_owner("pending-context-real-grant");
    let ticket = QueryAdmissionTicket::new_with_observation(admission_source(&backend), None, owner);
    let task_context = Arc::new(TaskContext::default());
    let first_context = ticket
        .acquire_context(Arc::clone(&task_context), ExecutionCancellation::new())
        .await
        .expect("the first context must be admitted");

    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();
    let error = ticket
        .acquire_context(Arc::clone(&task_context), cancellation)
        .await
        .expect_err("a cancelled secondary context must not consume another context lease");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);
    {
        let state = ticket.state.lock().expect("ticket state mutex");
        let round = state.rounds.values().next().expect("the shared task round remains live");
        assert_eq!(round.active_contexts, 1, "the failed secondary acquire must release only its own context slot");
        assert_eq!(round.active_streams, 0);
    }

    drop(first_context);
    let replacement = tokio::time::timeout(
        Duration::from_secs(5),
        ticket.acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new()),
    )
    .await
    .expect("the grant must be returned after the last context drops")
    .expect("the ticket must admit a replacement context");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);
    drop(replacement);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_deadline_cancels_a_waiter_without_leaking_round_state_or_capacity() {
    let backend = InstalledTestGpuBackend::with_config(
        NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1),
        NexusGpuDeviceProfile::new(0)
            .with_memory_limit(nexus_query_engine::runtime::GpuDeviceMemoryLimit::Explicit {
                limit_bytes: 256 * 1024 * 1024,
            })
            .with_max_active_attempts(1),
        NexusGpuQueuePolicy::default().with_max_queue_wait(Duration::from_secs(1)),
    );
    let holder = QueryAdmissionTicket::new(admission_source(&backend), None);
    let held_context = holder
        .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
        .await
        .expect("the holder must consume the device-local admission slot");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);

    let waiter = QueryAdmissionTicket::new(admission_source(&backend), None);
    let deadline = Instant::now() + Duration::from_millis(100);
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        waiter.acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::with_deadline(Some(deadline))),
    )
    .await
    .expect("a queued waiter must terminate at its deadline")
    .expect_err("deadline expiry must reject the waiter");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    let snapshot = backend.backend().admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1, "the holder still owns the only grant");
    assert_eq!(snapshot.queued_queries, 0, "a cancelled waiter must be removed from the queue");
    {
        let state = waiter.state.lock().expect("waiter ticket state mutex");
        assert!(state.admitting.is_empty(), "deadline cleanup must clear the in-flight marker");
        assert!(state.rounds.values().all(|round| round.handle.is_none()));
    }

    drop(held_context);
    let replacement = tokio::time::timeout(
        Duration::from_secs(5),
        waiter.acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new()),
    )
    .await
    .expect("released capacity must be reusable after waiter cancellation")
    .expect("waiter ticket can retry after deadline cleanup");
    drop(replacement);
}

#[tokio::test(flavor = "current_thread")]
async fn planner_capability_evidence_rejects_at_the_native_ticket_boundary_before_admission() {
    let unavailable = explicit_sink_capability_proof();
    let backend = InstalledTestGpuBackend::with_config(
        NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1),
        NexusGpuDeviceProfile::new(0)
            .with_memory_limit(nexus_query_engine::runtime::GpuDeviceMemoryLimit::Explicit {
                limit_bytes: 256 * 1024 * 1024,
            })
            .with_device_capabilities(nexus_query_engine::runtime::DeviceCapabilities::only(Vec::new())),
        NexusGpuQueuePolicy::default(),
    );
    let config = crate::native_exec::NexusNativeExecutionConfig::bounded(256, 512, 1, 1, 1)
        .with_query_admission_source(admission_source(&backend));
    let ticket = config
        .new_plan_query_admission_ticket_with_graph_model_ask_and_fragment_requirements(None, vec![unavailable.clone()])
        .expect("fragment requirements aggregate into a ticket")
        .expect("configured admission source creates a ticket");

    assert_eq!(ticket.attempt_capability_proof(), Some(&unavailable));
    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()), ExecutionCancellation::new())
        .await
        .expect_err("an aggregated unsupported capability must reject before admission");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
    assert_eq!(
        error.fact_value("required_capabilities"),
        Some("materialization:query_engine_explicit_sink_materialization_reservation")
    );
    assert_eq!(error.fact_value("configured_device_count"), Some("1"));
    assert_eq!(error.fact_value("compatible_device_count"), Some("0"));
    assert_eq!(
        backend.backend().admission_snapshot().unwrap().active_queries,
        0,
        "rejection must precede grant mutation"
    );
    assert_eq!(
        backend.backend().admission_snapshot().unwrap().queued_queries,
        0,
        "rejection must not become a capacity waiter"
    );
}

#[test]
fn standalone_ticket_drop_seals_local_observation() {
    let backend = InstalledTestGpuBackend::new();
    let owner = RuntimeObservationOwner::new("ticket-drop", None, RuntimeObservationConfig::bounded(16, 4096));
    let ticket = QueryAdmissionTicket::new_with_observation(admission_source(&backend), None, owner.clone());
    drop(ticket);
    assert_eq!(owner.sealed_snapshot().unwrap().footer().terminal, Some(RuntimeTerminalOutcome::Dropped));
}

#[tokio::test(flavor = "current_thread")]
async fn composed_producer_consumer_success_seals_after_consumer_eof() {
    let owner = lifecycle_owner("composed-success");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = NativeAdmissionExec::new(
        lifecycle_exec(LifecyclePlanBehavior::Success, Arc::clone(&observed_ticket)),
        ticket.clone(),
    );
    let task_context = Arc::new(TaskContext::default());

    let mut stream = exec.execute(0, Arc::clone(&task_context)).expect("synthetic composed consumer executes");
    assert!(stream.next().await.is_none(), "consumer reaches host EOF");
    assert!(stream.next().await.is_none(), "polling an already-complete consumer remains EOF");
    assert!(owner.sealed_snapshot().is_none(), "round guard remains until the stream is dropped");
    drop(stream);

    assert_eq!(*observed_ticket.lock().expect("test ticket observation mutex"), Some(ticket.id()));
    assert_terminal_boundary_once(&owner, RuntimeTerminalOutcome::Succeeded);
}

#[tokio::test(flavor = "current_thread")]
async fn ambient_task_ticket_owns_the_wrapper_round_without_a_second_admission() {
    let wrapper_owner = lifecycle_owner("wrapper-ticket-unused");
    let active_owner = lifecycle_owner("ambient-ticket-active");
    let wrapper_ticket = QueryAdmissionTicket::new_for_lifecycle_test(wrapper_owner.clone());
    let active_ticket = QueryAdmissionTicket::new_for_lifecycle_test(active_owner.clone());
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = NativeAdmissionExec::new(
        lifecycle_exec(LifecyclePlanBehavior::Success, Arc::clone(&observed_ticket)),
        Arc::clone(&wrapper_ticket),
    );
    let task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&active_ticket));

    let mut stream = exec.execute(0, task_context).expect("ambient ticket stream starts");
    assert!(stream.next().await.is_none());
    drop(stream);

    assert_eq!(
        *observed_ticket.lock().expect("test ticket observation mutex"),
        Some(active_ticket.id()),
        "the active invocation ticket owns the execution round"
    );
    assert_terminal_boundary_once(&active_owner, RuntimeTerminalOutcome::Succeeded);
    assert!(wrapper_owner.sealed_snapshot().is_none(), "the wrapper ticket must not receive a second round");

    drop(exec);
    drop(wrapper_ticket);
    assert_terminal_boundary_once(&wrapper_owner, RuntimeTerminalOutcome::Dropped);
}

#[test]
fn composed_producer_execute_error_releases_round_and_seals_failure() {
    let owner = lifecycle_owner("composed-producer-error");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = NativeAdmissionExec::new(
        lifecycle_exec(LifecyclePlanBehavior::ExecuteError, Arc::clone(&observed_ticket)),
        ticket.clone(),
    );

    let error = match exec.execute(0, Arc::new(TaskContext::default())) {
        Ok(_) => panic!("producer error must reach the wrapper"),
        Err(error) => error,
    };
    assert!(error.find_nexus_error().is_some(), "producer failure remains typed at the DataFusion boundary");
    assert_eq!(*observed_ticket.lock().expect("test ticket observation mutex"), Some(ticket.id()));
    let snapshot = owner.sealed_snapshot().expect("producer failure seals the round");
    assert!(matches!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Failed { .. })));
    assert_terminal_boundary_once(&owner, snapshot.footer().terminal.clone().expect("terminal outcome recorded"));
}

#[tokio::test(flavor = "current_thread")]
async fn composed_consumer_error_seals_one_failed_terminal_boundary() {
    let owner = lifecycle_owner("composed-consumer-error");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = NativeAdmissionExec::new(
        lifecycle_exec(LifecyclePlanBehavior::StreamError, Arc::clone(&observed_ticket)),
        ticket.clone(),
    );

    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("consumer stream starts");
    let error = stream.next().await.expect("consumer emits one error").expect_err("consumer error is preserved");
    assert!(error.find_nexus_error().is_some());
    drop(stream);

    assert_eq!(*observed_ticket.lock().expect("test ticket observation mutex"), Some(ticket.id()));
    let snapshot = owner.sealed_snapshot().expect("consumer failure seals the round");
    assert!(matches!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Failed { .. })));
    assert_terminal_boundary_once(&owner, snapshot.footer().terminal.clone().expect("terminal outcome recorded"));
}

#[tokio::test(flavor = "current_thread")]
async fn composed_consumer_cancellation_is_terminalized_as_cancelled() {
    let owner = lifecycle_owner("composed-cancelled");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = NativeAdmissionExec::new(
        lifecycle_exec(LifecyclePlanBehavior::CancelledStream, Arc::clone(&observed_ticket)),
        ticket.clone(),
    );

    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("cancelled consumer stream starts");
    let error = stream.next().await.expect("consumer emits cancellation").expect_err("cancellation is an error item");
    assert_eq!(
        error.find_nexus_error().expect("cancelled error remains typed").code(),
        crate::error::ErrorCode::Cancelled
    );
    drop(stream);

    assert_eq!(*observed_ticket.lock().expect("test ticket observation mutex"), Some(ticket.id()));
    assert_terminal_boundary_once(&owner, RuntimeTerminalOutcome::Cancelled);
}

#[test]
fn dropping_task_context_before_stream_does_not_drop_the_admission_round() {
    let owner = lifecycle_owner("task-drop");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = NativeAdmissionExec::new(lifecycle_exec(LifecyclePlanBehavior::Success, observed_ticket), ticket);
    let task_context = Arc::new(TaskContext::default());

    let stream = exec.execute(0, Arc::clone(&task_context)).expect("synthetic stream starts");
    drop(task_context);
    assert!(owner.sealed_snapshot().is_none(), "stream owns the round after its task context is dropped");
    drop(stream);
    assert_terminal_boundary_once(&owner, RuntimeTerminalOutcome::Dropped);
}

#[test]
fn dropping_stream_without_polling_records_dropped_once() {
    let owner = lifecycle_owner("stream-drop");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let exec =
        NativeAdmissionExec::new(lifecycle_exec(LifecyclePlanBehavior::Success, Arc::new(Mutex::new(None))), ticket);

    let stream = exec.execute(0, Arc::new(TaskContext::default())).expect("synthetic stream starts");
    drop(stream);
    assert_terminal_boundary_once(&owner, RuntimeTerminalOutcome::Dropped);
}

#[tokio::test(flavor = "current_thread")]
async fn one_context_ticket_keeps_composed_round_alive_until_all_streams_drop() {
    let owner = lifecycle_owner("context-ticket");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = NativeAdmissionExec::new(
        lifecycle_exec(LifecyclePlanBehavior::Success, Arc::clone(&observed_ticket)),
        ticket.clone(),
    );
    let task_context = Arc::new(TaskContext::default());

    let mut first = exec.execute(0, Arc::clone(&task_context)).expect("first composed stream starts");
    let mut second = exec.execute(0, Arc::clone(&task_context)).expect("second composed stream reuses the ticket");
    assert_eq!(*observed_ticket.lock().expect("test ticket observation mutex"), Some(ticket.id()));

    assert!(first.next().await.is_none());
    drop(first);
    assert!(owner.sealed_snapshot().is_none(), "one live stream keeps the shared execution round alive");

    assert!(second.next().await.is_none());
    drop(second);
    assert_terminal_boundary_once(&owner, RuntimeTerminalOutcome::Succeeded);
}

#[test]
fn context_ticket_guard_release_waits_for_the_execution_round() {
    let owner = lifecycle_owner("context-guard");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let task_context = Arc::new(TaskContext::default());
    let round = ticket.begin_execution_round(Arc::clone(&task_context)).expect("execution round starts");

    {
        let mut state = ticket.state.lock().expect("ticket state mutex");
        let ticket_round = state.rounds.get_mut(&round.task_key).expect("round is registered");
        assert_eq!(ticket_round.active_streams, 1);
        ticket_round.active_contexts = 1;
    }
    let context_guard = QueryAdmissionTicketContextGuard {
        ticket: Arc::clone(&ticket),
        task_key: round.task_key,
        epoch: round.epoch,
        released: false,
    };
    drop(context_guard);

    assert!(owner.sealed_snapshot().is_none(), "context release cannot finish a still-live execution round");
    {
        let state = ticket.state.lock().expect("ticket state mutex");
        let ticket_round = state.rounds.get(&round.task_key).expect("stream still owns the round");
        assert_eq!(ticket_round.active_contexts, 0);
        assert_eq!(ticket_round.active_streams, 1);
    }
    drop(round);
    assert_terminal_boundary_once(&owner, RuntimeTerminalOutcome::Dropped);
}

#[test]
fn pre_cancelled_pending_context_seals_cancelled_and_releases_its_round() {
    let owner = lifecycle_owner("pending-context-cancelled");
    let ticket = QueryAdmissionTicket::new_for_lifecycle_test(owner.clone());
    let task_context = Arc::new(TaskContext::default());
    let round = ticket.begin_execution_round(Arc::clone(&task_context)).expect("execution round starts");
    let observation = ticket
        .state
        .lock()
        .expect("ticket state mutex")
        .rounds
        .get(&round.task_key)
        .expect("round is registered")
        .observation
        .clone();

    let pending_guard = QueryAdmissionPendingContextGuard {
        ticket: Arc::clone(&ticket),
        task_key: round.task_key,
        epoch: round.epoch,
        released: false,
        observation,
    };
    drop(pending_guard);
    drop(round);

    assert_terminal_boundary_once(&owner, RuntimeTerminalOutcome::Cancelled);
}
