use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
    task::Poll,
    time::{Duration, Instant},
};

use arrow::{
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{
    FutureExt, StreamExt,
    future::join,
    stream::{empty, once, poll_fn},
};
use query_engine::{
    admission::{AttemptCapabilityEvidence, AttemptExecutionManifest, AttemptExecutionOccurrence, ExecutionCapability},
    error::QueryErrorCode,
    observability::{QueryAttemptReport, RuntimeTerminalOutcome},
    plan::{QueryPlan, QueryPlanBuilder},
    runtime::{DeviceCapabilities, GpuDeviceMemoryLimit},
    source::{GpuResidentSource, SourceNode},
};
use tokio::time::timeout;

use super::*;
use crate::{
    attempt::{AttemptOptions, attempt_ticket_for_test, attempt_ticket_with_options_for_test},
    backend::{AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, InstalledTestGpuBackend},
    error::{DataFusionErrorExt, Error, ErrorCode},
    native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
};

#[derive(Clone, Copy, Debug)]
enum LifecyclePlanBehavior {
    Success,
    ExecuteError,
    StreamError,
    CancelledStream,
    PanickingStream,
    PanickingExecute,
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
            LifecyclePlanBehavior::ExecuteError => {
                (ErrorCode::Internal, "algeon_datafusion.test.composed_producer", "synthetic composed producer failed")
            },
            LifecyclePlanBehavior::StreamError => {
                (ErrorCode::Internal, "algeon_datafusion.test.composed_consumer", "synthetic composed consumer failed")
            },
            LifecyclePlanBehavior::CancelledStream => (
                ErrorCode::Cancelled,
                "algeon_datafusion.test.composed_consumer_cancelled",
                "synthetic composed consumer was cancelled",
            ),
            LifecyclePlanBehavior::Success
            | LifecyclePlanBehavior::PanickingStream
            | LifecyclePlanBehavior::PanickingExecute => {
                unreachable!("this behavior does not produce a typed error")
            },
        };
        DataFusionError::from(Error::new(code, operation, message))
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
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
                let stream = empty::<DataFusionResult<RecordBatch>>();
                Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
            },
            LifecyclePlanBehavior::ExecuteError => Err(Self::typed_error(self.behavior)),
            LifecyclePlanBehavior::PanickingExecute => panic!("injected input execute panic"),
            LifecyclePlanBehavior::PanickingStream => {
                let stream = poll_fn(|_cx| -> Poll<Option<DataFusionResult<RecordBatch>>> {
                    panic!("injected inner stream panic")
                });
                Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
            },
            LifecyclePlanBehavior::StreamError | LifecyclePlanBehavior::CancelledStream => {
                let error = Self::typed_error(self.behavior);
                let stream = once(async move { Err::<RecordBatch, DataFusionError>(error) });
                Ok(Box::pin(RecordBatchStreamAdapter::new(Arc::clone(&self.schema), stream)))
            },
        }
    }
}

fn lifecycle_exec(behavior: LifecyclePlanBehavior, observed_ticket: Arc<Mutex<Option<u64>>>) -> Arc<dyn ExecutionPlan> {
    Arc::new(LifecycleExec::new(behavior, observed_ticket))
}

/// Roots `input` in the admission node the final optimizer pass places.
fn admission_exec(backend: &InstalledTestGpuBackend, input: Arc<dyn ExecutionPlan>) -> NativeAdmissionExec {
    NativeAdmissionExec::new(input, admission_source(backend), AttemptCapabilityEvidence::empty())
}

/// The report of the one internal attempt that executions without an
/// attempt of their own began on `backend`.
fn sole_internal_attempt_report(backend: &InstalledTestGpuBackend) -> Arc<QueryAttemptReport> {
    let reports = backend.backend().take_abandoned_attempt_reports(8);
    assert_eq!(reports.len(), 1, "one execution must publish exactly one attempt report");
    Arc::clone(&reports[0])
}

fn admission_source(backend: &InstalledTestGpuBackend) -> QueryAdmissionSource {
    backend
        .optimizer_config()
        .native_execution_config()
        .query_admission_source()
        .expect("installed backend must provide admission authority")
}

fn explicit_sink_capability_evidence() -> AttemptCapabilityEvidence {
    use arrow::datatypes::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    let plan = builder.finish_with_host_sink(source);
    AttemptCapabilityEvidence::for_query_plan(&plan)
}

fn resident_host_plan(input_name: &str) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source)
}

#[test]
fn host_occurrence_dependencies_complete_once_and_fail_closed_on_drop() {
    let backend = InstalledTestGpuBackend::new();
    let first_plan = resident_host_plan("first");
    let second_plan = resident_host_plan("second");
    let manifest = AttemptExecutionManifest::new(vec![
        AttemptExecutionOccurrence::host_arrow("gpu-first", &first_plan, vec![], false),
        AttemptExecutionOccurrence::host_arrow("gpu-second", &second_plan, vec!["gpu-first".to_string()], false),
    ])
    .unwrap();
    let evidence = AttemptCapabilityEvidence::empty().with_execution_manifest(manifest).unwrap();
    let (_attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), Some(evidence));
    let task_context = task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&ticket));

    let error = ticket
        .begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan)
        .expect_err("a dependent occurrence cannot begin first");
    assert_eq!(error.fact_value("reason"), Some("host_occurrence_predecessor_incomplete"));

    drop(ticket.begin_host_occurrence(&task_context, "gpu-first", "gpu-first", &first_plan).unwrap().unwrap());
    assert!(
        ticket.begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan).is_err(),
        "dropping an unfinished occurrence must not satisfy its dependents"
    );

    ticket.begin_host_occurrence(&task_context, "gpu-first", "gpu-first", &first_plan).unwrap().unwrap().complete();
    ticket.begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan).unwrap().unwrap().complete();
    let duplicate = ticket
        .begin_host_occurrence(&task_context, "gpu-second", "gpu-second", &second_plan)
        .expect_err("one final-plan occurrence cannot execute twice");
    assert_eq!(duplicate.fact_value("reason"), Some("host_occurrence_duplicate_execution"));

    let next_task_context =
        task_context_with_query_admission_ticket(Arc::new(TaskContext::default()), Arc::clone(&ticket));
    ticket
        .begin_host_occurrence(&next_task_context, "gpu-first", "gpu-first", &first_plan)
        .expect("a new invocation round has independent occurrence state")
        .unwrap()
        .complete();
}

#[tokio::test(flavor = "current_thread")]
async fn fragments_bound_after_the_claim_join_its_grant() {
    let backend = InstalledTestGpuBackend::new();
    let first_plan = resident_host_plan("first-fragment");
    let second_plan = resident_host_plan("second-fragment");
    let (_attempt, ticket) =
        attempt_ticket_for_test(admission_source(&backend), Some(AttemptCapabilityEvidence::empty()));

    // Before the claim, each fragment adds its delta to the claim's evidence.
    ticket.bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&first_plan)]).unwrap();
    ticket.bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&second_plan)]).unwrap();
    let evidence = ticket.attempt_capability_evidence().unwrap();
    assert!(evidence.execution_scopes().iter().any(|scope| scope.matches(&first_plan)));
    assert!(evidence.execution_scopes().iter().any(|scope| scope.matches(&second_plan)));

    let context = ticket.acquire_context(Arc::new(TaskContext::default())).await.expect("the combined evidence admits");
    let admitted = backend.backend().admission_snapshot().unwrap();
    ticket
        .bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&first_plan)])
        .expect("evidence the claim covers binds nothing new");
    let third_plan = resident_host_plan("third-fragment");
    ticket
        .bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&third_plan)])
        .expect("a fragment planned after the claim joins its grant");
    assert_eq!(
        backend.backend().admission_snapshot().unwrap(),
        admitted,
        "joining the grant changes neither the queue nor the ledger"
    );
    let execution = context.context().context();
    execution.validate_admitted_plan_capabilities(&third_plan).expect("the joined fragment may execute");
    let never_bound = resident_host_plan("never-bound-fragment");
    let error = execution
        .validate_admitted_plan_capabilities(&never_bound)
        .expect_err("a plan no fragment bound stays outside the admitted identities");
    assert_eq!(error.fact_value("reason"), Some("admitted_capability_scope_mismatch"));

    // The grant belongs to the attempt, not to the fragment that acquired
    // the context: once that context ends, a later fragment still joins it.
    drop(context);
    let fourth_plan = resident_host_plan("fourth-fragment");
    ticket
        .bind_fragment_evidence([AttemptCapabilityEvidence::for_query_plan(&fourth_plan)])
        .expect("a finished fragment does not end the attempt's grant");
    let context =
        ticket.acquire_context(Arc::new(TaskContext::default())).await.expect("the attempt's grant is reused");
    context
        .context()
        .context()
        .validate_admitted_plan_capabilities(&fourth_plan)
        .expect("the joined fragment may execute");
    assert_eq!(backend.backend().admission_snapshot().unwrap(), admitted);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_forwards_opaque_evidence_and_releases_real_ledger_grant_for_successive_admission() {
    // One active attempt per device, so the second admission can only succeed after the first grant is released.
    let backend = InstalledTestGpuBackend::with_config(
        AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
        ),
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
            .with_max_active_attempts(1),
        AlgeonGpuQueuePolicy::default(),
    );
    let (attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    assert!(ticket.attempt_capability_evidence().is_none());

    let context = ticket
        .acquire_context(Arc::new(TaskContext::default()))
        .await
        .expect("installed backend must acquire a real admission context");
    let facts = context.admission_facts();
    assert!(facts.admission_sequence > 0);
    assert!(facts.query_grant_bytes > 0);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);

    drop(context);
    assert_eq!(
        backend.backend().admission_snapshot().unwrap().active_queries,
        1,
        "the grant belongs to the attempt, not to the context"
    );
    drop(attempt);
    let (_next_attempt, next_ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    let next_context =
        timeout(std::time::Duration::from_secs(5), next_ticket.acquire_context(Arc::new(TaskContext::default())))
            .await
            .expect("ending the attempt must eventually return the real ledger grant")
            .expect("released capacity must admit the next attempt");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);
    drop(next_context);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_ticket_propagates_attempt_cancellation_to_admission() {
    let backend = InstalledTestGpuBackend::new();
    let (attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    attempt.cancellation().cancel();

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()))
        .await
        .expect_err("cancelled admission must not acquire a real handle");

    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn a_failed_admission_is_the_outcome_every_later_fragment_gets() {
    let backend = InstalledTestGpuBackend::new();
    let (attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    attempt.cancellation().cancel();

    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()))
        .await
        .expect_err("cancelled admission must fail before publishing a handle");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);

    {
        let admission = ticket.admission.lock().expect("ticket admission mutex");
        assert!(!admission.admitting, "a failed admission must not strand its in-flight marker");
        assert!(matches!(admission.outcome, Some(Err(_))), "a failed admission records its error, not a handle");
    }

    let retry = timeout(Duration::from_secs(5), ticket.acquire_context(Arc::new(TaskContext::default())))
        .await
        .expect("a later fragment must not wait on the resolved admission")
        .expect_err("an attempt admits at most once");
    assert_eq!(
        (retry.code(), retry.operation(), retry.message(), retry.facts()),
        (error.code(), error.operation(), error.message(), error.facts()),
        "a later fragment gets the admission's own error, not a second claim's"
    );
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 0);
}

/// Fragments waiting while another fragment claims the attempt's admission
/// get that admission's outcome; none of them claims again.
#[tokio::test(flavor = "current_thread")]
async fn concurrent_fragments_share_the_one_admission_outcome() {
    let backend = InstalledTestGpuBackend::with_config(
        AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
        ),
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
            .with_max_active_attempts(1),
        AlgeonGpuQueuePolicy::default(),
    );
    let (holder_attempt, holder) = attempt_ticket_for_test(admission_source(&backend), None);
    let held_context = holder
        .acquire_context(Arc::new(TaskContext::default()))
        .await
        .expect("the holder must consume the device-local admission slot");

    // The claim queues behind the holder until the waiting attempt's
    // deadline rejects it, while a second fragment waits on that claim.
    let (_waiting_attempt, waiting) = attempt_ticket_with_options_for_test(
        admission_source(&backend),
        None,
        AttemptOptions { deadline: Some(Instant::now() + Duration::from_millis(200)), ..AttemptOptions::default() },
    );
    let (claiming, waiting_fragment) = timeout(
        Duration::from_secs(5),
        join(
            waiting.acquire_context(Arc::new(TaskContext::default())),
            waiting.acquire_context(Arc::new(TaskContext::default())),
        ),
    )
    .await
    .expect("both fragments resolve at the claim's deadline");
    let claiming = claiming.expect_err("the claim is rejected at its deadline");
    let waiting_fragment = waiting_fragment.expect_err("the waiting fragment gets the claim's outcome");
    assert_eq!(claiming.fact_value("reason"), Some("deadline_expired"));
    assert_eq!(
        (waiting_fragment.code(), waiting_fragment.operation(), waiting_fragment.message(), waiting_fragment.facts()),
        (claiming.code(), claiming.operation(), claiming.message(), claiming.facts()),
        "the waiting fragment gets the same typed error instead of claiming again"
    );
    let snapshot = backend.backend().admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1, "only the holder owns a grant");
    assert_eq!(snapshot.queued_queries, 0);
    assert_eq!(snapshot.queue_high_watermark, 1, "the attempt queued one admission request");

    drop(held_context);
    drop(holder_attempt);
}

#[tokio::test(flavor = "current_thread")]
async fn pending_context_cancellation_keeps_the_shared_grant_until_the_attempt_ends() {
    let backend = InstalledTestGpuBackend::with_config(
        AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
        ),
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
            .with_max_active_attempts(1),
        AlgeonGpuQueuePolicy::default(),
    );
    let (attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    let task_context = Arc::new(TaskContext::default());
    let first_context =
        ticket.acquire_context(Arc::clone(&task_context)).await.expect("the first context must be admitted");

    attempt.cancellation().cancel();
    let error = ticket
        .acquire_context(Arc::clone(&task_context))
        .await
        .expect_err("a cancelled secondary context must not consume another context lease");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);

    drop(first_context);
    assert_eq!(
        backend.backend().admission_snapshot().unwrap().active_queries,
        1,
        "a cancelled attempt keeps its grant until its outcome is committed"
    );
    drop(attempt);
    let (_replacement_attempt, replacement_ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    let replacement =
        timeout(Duration::from_secs(5), replacement_ticket.acquire_context(Arc::new(TaskContext::default())))
            .await
            .expect("the grant must be returned once the attempt ends")
            .expect("released capacity must admit a replacement attempt");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);
    drop(replacement);
}

#[tokio::test(flavor = "current_thread")]
async fn admission_deadline_cancels_a_waiter_without_leaking_admission_state_or_capacity() {
    let backend = InstalledTestGpuBackend::with_config(
        AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
        ),
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
            .with_max_active_attempts(1),
        AlgeonGpuQueuePolicy::default().with_max_queue_wait(Duration::from_secs(1)),
    );
    let (holder_attempt, holder) = attempt_ticket_for_test(admission_source(&backend), None);
    let held_context = holder
        .acquire_context(Arc::new(TaskContext::default()))
        .await
        .expect("the holder must consume the device-local admission slot");
    assert_eq!(backend.backend().admission_snapshot().unwrap().active_queries, 1);

    let deadline = Instant::now() + Duration::from_millis(100);
    let (_waiter_attempt, waiter) = attempt_ticket_with_options_for_test(
        admission_source(&backend),
        None,
        AttemptOptions { deadline: Some(deadline), ..AttemptOptions::default() },
    );
    let error = timeout(Duration::from_secs(5), waiter.acquire_context(Arc::new(TaskContext::default())))
        .await
        .expect("a queued waiter must terminate at its deadline")
        .expect_err("deadline expiry must reject the waiter");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    let snapshot = backend.backend().admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 1, "the holder still owns the only grant");
    assert_eq!(snapshot.queued_queries, 0, "a cancelled waiter must be removed from the queue");
    {
        let admission = waiter.admission.lock().expect("waiter ticket admission mutex");
        assert!(!admission.admitting, "deadline cleanup must clear the in-flight marker");
        assert!(matches!(admission.outcome, Some(Err(_))));
    }

    drop(held_context);
    drop(holder_attempt);
    let (_replacement_attempt, replacement_ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    let replacement =
        timeout(Duration::from_secs(5), replacement_ticket.acquire_context(Arc::new(TaskContext::default())))
            .await
            .expect("released capacity must be reusable after waiter cancellation")
            .expect("a new attempt admits after the waiter's deadline cleanup");
    drop(replacement);
}

#[tokio::test(flavor = "current_thread")]
async fn planner_capability_evidence_rejects_at_the_native_ticket_boundary_before_admission() {
    let mut unavailable = explicit_sink_capability_evidence();
    unavailable.required_capabilities = vec![ExecutionCapability::new("unavailable")];
    let backend = InstalledTestGpuBackend::with_config(
        AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
        ),
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 })
            .with_device_capabilities(DeviceCapabilities::only(Vec::new())),
        AlgeonGpuQueuePolicy::default(),
    );
    let (_attempt, ticket) = attempt_ticket_for_test(admission_source(&backend), Some(unavailable.clone()));

    assert_eq!(ticket.attempt_capability_evidence(), Some(unavailable));
    let error = ticket
        .acquire_context(Arc::new(TaskContext::default()))
        .await
        .expect_err("an aggregated unsupported capability must reject before admission");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.fact_value("reason"), Some("required_device_capability_unavailable"));
    assert_eq!(error.fact_value("required_capabilities"), Some("unavailable"));
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

#[tokio::test(flavor = "current_thread")]
async fn panic_in_the_inner_stream_poll_reports_panicked_once() {
    let backend = InstalledTestGpuBackend::new();
    let exec =
        admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::PanickingStream, Arc::new(Mutex::new(None))));
    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("synthetic stream starts");

    let unwind = AssertUnwindSafe(stream.next()).catch_unwind().await;

    assert!(unwind.is_err(), "the inner panic must keep unwinding");
    drop(stream);
    assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Panicked);
}

#[test]
fn panic_in_the_input_execute_reports_panicked_once() {
    let backend = InstalledTestGpuBackend::new();
    let exec =
        admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::PanickingExecute, Arc::new(Mutex::new(None))));

    let unwind = catch_unwind(AssertUnwindSafe(|| exec.execute(0, Arc::new(TaskContext::default()))));

    assert!(unwind.is_err(), "the input panic must keep unwinding");
    drop(exec);
    assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Panicked);
}

/// An attempt mints its one admission ticket while planning; executing the
/// plan admits under that ticket without minting another.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_attempt_invocation_mints_one_admission_ticket() {
    use arrow::array::{Int64Array, StringArray};
    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{ParquetReadOptions, SessionContext},
    };
    use parquet::arrow::ArrowWriter;

    let backend = InstalledTestGpuBackend::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.parquet");
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Utf8, false), Field::new("v", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["a", "b", "a"])), Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let ctx = SessionContext::new_with_state(
        backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build(),
    );
    ctx.register_parquet("rows", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let next_ticket_id = || attempt_ticket_for_test(admission_source(&backend), None).1.id();

    let before = next_ticket_id();
    let attempt = backend.backend().begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = attempt.sql(&ctx, "SELECT k, SUM(v) FROM rows GROUP BY k").await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = attempt.completion().await;
    let after = next_ticket_id();

    assert!(report.admission.is_some(), "the aggregate executes natively under the attempt");
    assert_eq!(after - before - 1, 1, "the attempt minted exactly one admission ticket");
}

#[tokio::test(flavor = "current_thread")]
async fn internal_attempt_reports_success_once_at_consumer_eof() {
    let backend = InstalledTestGpuBackend::new();
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::clone(&observed_ticket)));

    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("synthetic consumer executes");
    assert!(stream.next().await.is_none(), "consumer reaches host EOF");
    assert!(stream.next().await.is_none(), "polling an already-complete consumer remains EOF");
    drop(stream);

    assert!(
        observed_ticket.lock().expect("test ticket observation mutex").is_some(),
        "the internal attempt's ticket owns the execution round"
    );
    assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Succeeded);
}

#[tokio::test(flavor = "current_thread")]
async fn ambient_attempt_ticket_owns_the_round_without_an_internal_attempt() {
    let backend = InstalledTestGpuBackend::new();
    let (_attempt, active_ticket) = attempt_ticket_for_test(admission_source(&backend), None);
    let observed_ticket = Arc::new(Mutex::new(None));
    let exec = admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::clone(&observed_ticket)));
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
    assert_eq!(
        backend.backend().attempt_counters_snapshot().attempts_begun,
        1,
        "an execution inside an attempt begins no internal attempt"
    );
}

#[test]
fn internal_attempt_input_execute_error_reports_one_failure() {
    let backend = InstalledTestGpuBackend::new();
    let exec =
        admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::ExecuteError, Arc::new(Mutex::new(None))));

    let error = match exec.execute(0, Arc::new(TaskContext::default())) {
        Ok(_) => panic!("input error must reach the admission node"),
        Err(error) => error,
    };
    assert!(error.find_algeon_error().is_some(), "input failure remains typed at the DataFusion boundary");
    assert!(matches!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Failed { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn internal_attempt_stream_error_reports_one_failure() {
    let backend = InstalledTestGpuBackend::new();
    let exec = admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::StreamError, Arc::new(Mutex::new(None))));

    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("consumer stream starts");
    let error = stream.next().await.expect("consumer emits one error").expect_err("consumer error is preserved");
    assert!(error.find_algeon_error().is_some());
    drop(stream);

    assert!(matches!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Failed { .. }));
}

#[tokio::test(flavor = "current_thread")]
async fn internal_attempt_stream_cancellation_reports_cancelled() {
    let backend = InstalledTestGpuBackend::new();
    let exec =
        admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::CancelledStream, Arc::new(Mutex::new(None))));

    let mut stream = exec.execute(0, Arc::new(TaskContext::default())).expect("cancelled consumer stream starts");
    let error = stream.next().await.expect("consumer emits cancellation").expect_err("cancellation is an error item");
    assert_eq!(error.find_algeon_error().expect("cancelled error remains typed").code(), ErrorCode::Cancelled);
    drop(stream);

    assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Cancelled);
}

#[test]
fn dropping_task_context_before_stream_does_not_end_the_internal_attempt() {
    let backend = InstalledTestGpuBackend::new();
    let exec = admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::new(Mutex::new(None))));
    let task_context = Arc::new(TaskContext::default());

    let stream = exec.execute(0, Arc::clone(&task_context)).expect("synthetic stream starts");
    drop(task_context);
    assert!(
        backend.backend().take_abandoned_attempt_reports(8).is_empty(),
        "the stream owns the attempt after its task context is dropped"
    );
    drop(stream);
    assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Dropped);
}

#[test]
fn dropping_stream_without_polling_reports_dropped_once() {
    let backend = InstalledTestGpuBackend::new();
    let exec = admission_exec(&backend, lifecycle_exec(LifecyclePlanBehavior::Success, Arc::new(Mutex::new(None))));

    let stream = exec.execute(0, Arc::new(TaskContext::default())).expect("synthetic stream starts");
    drop(stream);
    assert_eq!(sole_internal_attempt_report(&backend).outcome, RuntimeTerminalOutcome::Dropped);
}
