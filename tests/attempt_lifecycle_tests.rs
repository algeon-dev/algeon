//! GPU-lane integration contracts for the explicit attempt adapter.

use std::sync::OnceLock;

use datafusion::{physical_plan::RecordBatchStream, prelude::SessionContext};
use datafusion_nexus::{
    attempt::{AttemptOptions, RuntimeObservationConfig},
    backend::{GpuMemoryOwnership, NexusGpuBackend, NexusMemoryPolicy},
};
use futures::StreamExt;
use nexus_query_engine::observability::{AttemptDisposition, RuntimeTerminalOutcome};

fn backend() -> &'static NexusGpuBackend {
    static BACKEND: OnceLock<NexusGpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 256 * 1024 * 1024,
                max_source_chunk_bytes: 64 * 1024 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
            .build()
            .expect("attempt lifecycle tests require CUDA device 0")
    })
}

fn context() -> SessionContext {
    let state = backend()
        .install_on(datafusion::execution::SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .build();
    SessionContext::new_with_state(state)
}

/// A memory-source table never lowers to native execution, so a scan of it is
/// a deterministic CPU-only (`cpu_fallback`) invocation.
fn cpu_only_context() -> SessionContext {
    cpu_only_context_for(backend())
}

fn cpu_only_context_for(backend: &NexusGpuBackend) -> SessionContext {
    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::datasource::MemTable;
    let state = backend
        .install_on(datafusion::execution::SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .build();
    let ctx = SessionContext::new_with_state(state);
    let schema = std::sync::Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![std::sync::Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )
    .expect("mem table batch");
    let table = MemTable::try_new(schema, vec![vec![batch]]).expect("mem table");
    ctx.register_table("cpu_rows", std::sync::Arc::new(table)).expect("register mem table");
    ctx
}

#[tokio::test]
async fn correlation_is_bounded_and_cpu_attempt_waits_for_eof() {
    let oversized_query = AttemptOptions { query_id: Some("q".repeat(257)), ..Default::default() };
    assert!(backend().begin_attempt(oversized_query).is_err());
    let oversized_trace = AttemptOptions { trace_context: Some("t".repeat(1025)), ..Default::default() };
    assert!(backend().begin_attempt(oversized_trace).is_err());

    let attempt = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let first_id = attempt.attempt_id();
    let mut stream = attempt.sql(&cpu_only_context(), "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    assert!(!attempt.completion().is_ready(), "success is terminal only at EOF");
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = attempt.completion().await;
    assert_eq!(report.attempt_id, first_id);
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.timestamps.physical_execution_started_ns.is_some());
    assert!(report.admission.is_none());

    let second = backend().begin_attempt(AttemptOptions::default()).unwrap();
    assert_ne!(first_id, second.attempt_id());
}

#[tokio::test]
async fn planning_failure_cancellation_drop_and_disabled_observation_publish_reports() {
    let planning = backend().begin_attempt(AttemptOptions::default()).unwrap();
    assert!(planning.sql(&context(), "SELECT FROM").await.is_err());
    let planning_report = planning.completion().await;
    assert_eq!(planning_report.disposition, AttemptDisposition::Unplanned);
    assert!(matches!(planning_report.outcome, RuntimeTerminalOutcome::Failed { .. }));
    assert!(planning_report.timestamps.physical_execution_started_ns.is_none());

    let cancelled = backend().begin_attempt(AttemptOptions::default()).unwrap();
    cancelled.cancellation().cancel();
    assert!(cancelled.sql(&context(), "SELECT 1").await.is_err());
    assert!(matches!(cancelled.completion().await.outcome, RuntimeTerminalOutcome::Cancelled));

    let dropped = backend()
        .begin_attempt(AttemptOptions { observation: RuntimeObservationConfig::disabled(), ..Default::default() })
        .unwrap();
    let stream = dropped.sql(&cpu_only_context(), "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    drop(stream);
    let report = dropped.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
}

/// Shared serialized construction for tests that need their own backend
/// instance (close/queue-config cases must not share the static backend).
fn own_backend(queue_capacity: Option<usize>) -> NexusGpuBackend {
    own_backend_with_policy(queue_capacity, datafusion_nexus::backend::NexusAdmissionPolicy::default())
}

fn own_backend_with_policy(
    queue_capacity: Option<usize>,
    admission: datafusion_nexus::backend::NexusAdmissionPolicy,
) -> NexusGpuBackend {
    use std::sync::{Mutex, MutexGuard};
    static SERIAL: Mutex<()> = Mutex::new(());
    let _guard: MutexGuard<'_, ()> = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut builder = NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: 256 * 1024 * 1024,
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 2,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .admission(admission);
    if let Some(capacity) = queue_capacity {
        builder = builder.with_attempt_completion_queue_capacity(capacity);
    }
    builder.build().expect("own attempt backend")
}

async fn native_context(backend: &NexusGpuBackend) -> SessionContext {
    use arrow::{
        array::{Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use parquet::arrow::ArrowWriter;
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]));
    static FIXTURE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "nexus-attempt-lifecycle-{}-{}.parquet",
        std::process::id(),
        FIXTURE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    {
        let batch = RecordBatch::try_new(
            std::sync::Arc::clone(&schema),
            vec![
                std::sync::Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
                std::sync::Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            ],
        )
        .expect("fixture batch");
        let file = std::fs::File::create(&path).expect("fixture file");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
        writer.write(&batch).expect("write fixture");
        writer.close().expect("close fixture");
    }
    let state = backend
        .install_on(datafusion::execution::SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet(
        "native_rows",
        path.to_str().expect("utf8 path"),
        datafusion::prelude::ParquetReadOptions::default(),
    )
    .await
    .expect("register parquet");
    ctx
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_invocation_shares_one_admission_and_distinct_boundaries() {
    let ctx = native_context(backend()).await;
    let attempt = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = attempt.sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = attempt.completion().await;
    assert!(
        matches!(report.disposition, AttemptDisposition::Native | AttemptDisposition::PartialNative),
        "native aggregate over parquet must execute natively: {:?}",
        report.disposition
    );
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert_eq!(report.device_ordinal, Some(0), "the admission assigns exactly one device");
    let admission = report.admission.as_ref().expect("native execution records admission facts");
    assert_eq!(admission.device_ordinal, 0);
    assert!(admission.grant_bytes > 0);
    let timestamps = &report.timestamps;
    let physical = timestamps.physical_execution_started_ns.expect("physical start present");
    let gpu = timestamps.gpu_execution_started_ns.expect("gpu start present");
    assert!(gpu >= physical, "gpu start never precedes physical start");
    assert!(timestamps.admitted_ns.is_some());
    assert!(timestamps.cleanup_complete_ns.is_some(), "admitted attempts record cleanup completion");
    assert!(timestamps.report_ready_ns.is_some());
    assert!(report.cleanup_ns().is_some());
    assert_eq!(report.cleanup_disposition, nexus_query_engine::observability::AttemptCleanupDisposition::Completed);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reused_plans_mint_fresh_attempts_and_never_observe_plan_lifetime_attempts() {
    let ctx = native_context(backend()).await;
    let planner = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let plan = planner.plan_sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    let planner_id = planner.attempt_id();

    // The planning attempt executes its own plan.
    let mut stream = planner.execute_physical_plan(std::sync::Arc::clone(&plan), &ctx).unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = planner.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));

    // A second attempt reuses the same plan and receives a fresh identity and
    // its own admission facts; nothing records into the first attempt.
    let second = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let second_id = second.attempt_id();
    assert_ne!(planner_id, second_id);
    let mut stream = second.execute_physical_plan(std::sync::Arc::clone(&plan), &ctx).unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = second.completion().await;
    assert_eq!(report.attempt_id, second_id);
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_some(), "reused plan admits under the new attempt");
    assert!(
        report.timestamps.planning_started_ns.is_none(),
        "an execute-only attempt does not synthesize planning boundaries"
    );

    // A third invocation reuses the SQL text and still mints a fresh id.
    let third = backend().begin_attempt(AttemptOptions::default()).unwrap();
    assert_ne!(third.attempt_id(), planner_id);
    assert_ne!(third.attempt_id(), second_id);

    // The same attempt-bound plan executed through a raw session without the
    // invocation extension is rejected: fragments cannot observe a
    // plan-lifetime attempt.
    let raw_state = backend()
        .install_on(datafusion::execution::SessionStateBuilder::new_with_default_features())
        .expect("install backend")
        .build();
    let raw_ctx = SessionContext::new_with_state(raw_state);
    let error = match datafusion::physical_plan::execute_stream(plan, raw_ctx.task_ctx()) {
        Ok(_) => panic!("raw execution of an attempt-bound plan must be rejected"),
        Err(error) => error,
    };
    let nexus_error =
        datafusion_nexus::error::DataFusionErrorExt::find_nexus_error(&error).expect("structured nexus error");
    assert_eq!(nexus_error.fact_value("reason"), Some("attempt_execution_required"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn panic_in_result_stream_is_terminal_exactly_once_and_unwinds() {
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{
        execution::TaskContext,
        physical_expr::EquivalenceProperties,
        physical_plan::{
            DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
            execution_plan::{Boundedness, EmissionType},
            stream::RecordBatchStreamAdapter,
        },
    };

    /// Test-only plan that panics on the first poll inside the attempt
    /// stream boundary.
    #[derive(Debug)]
    struct PanicExec {
        properties: std::sync::Arc<PlanProperties>,
        schema: std::sync::Arc<Schema>,
    }
    impl DisplayAs for PanicExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("PanicExec")
        }
    }
    impl ExecutionPlan for PanicExec {
        fn name(&self) -> &str {
            "PanicExec"
        }
        fn properties(&self) -> &std::sync::Arc<PlanProperties> {
            &self.properties
        }
        fn children(&self) -> Vec<&std::sync::Arc<dyn ExecutionPlan>> {
            vec![]
        }
        fn with_new_children(
            self: std::sync::Arc<Self>,
            _children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
        ) -> datafusion::common::Result<std::sync::Arc<dyn ExecutionPlan>> {
            Ok(self)
        }
        fn execute(
            &self,
            _partition: usize,
            _context: std::sync::Arc<TaskContext>,
        ) -> datafusion::common::Result<datafusion::physical_plan::SendableRecordBatchStream> {
            let stream = futures::stream::once(async {
                panic!("injected attempt stream panic");
            });
            Ok(Box::pin(RecordBatchStreamAdapter::new(std::sync::Arc::clone(&self.schema), stream)))
        }
    }

    let schema = std::sync::Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let plan: std::sync::Arc<dyn ExecutionPlan> = std::sync::Arc::new(PanicExec {
        properties: std::sync::Arc::new(PlanProperties::new(
            EquivalenceProperties::new(std::sync::Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        )),
        schema,
    });

    let attempt = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let mut stream = attempt.execute_physical_plan(plan, &context()).unwrap();
    let caught = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(stream.next())).await;
    assert!(caught.is_err(), "the panic unwinds through the nexus boundary");
    drop(stream);
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Panicked));
    assert!(matches!(report.disposition, AttemptDisposition::CpuFallback));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_reports_are_best_effort_delivery_loss_only() {
    let own = own_backend(Some(1));

    let cpu_ctx = cpu_only_context_for(&own);
    let first = own.begin_attempt(AttemptOptions::default()).unwrap();
    let first_id = first.attempt_id();
    drop(first.completion());
    let mut stream = first.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = first.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));

    let drained = own.take_abandoned_attempt_reports(8);
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].attempt_id, first_id);
    let counters = own.attempt_counters_snapshot();
    assert_eq!(counters.completion_queue_pushed, 1);
    assert_eq!(counters.completion_queue_overflowed, 0);

    // Two further abandoned reports overflow the capacity-one queue.
    for _ in 0..2 {
        let attempt = own.begin_attempt(AttemptOptions::default()).unwrap();
        drop(attempt.completion());
        let mut stream = attempt.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
        while let Some(batch) = stream.next().await {
            batch.unwrap();
        }
        let _ = attempt.completion().await;
    }
    let counters = own.attempt_counters_snapshot();
    assert_eq!(counters.completion_queue_pushed, 2);
    assert_eq!(counters.completion_queue_overflowed, 1);
    assert_eq!(own.take_abandoned_attempt_reports(8).len(), 1);
    own.close_and_wait(std::time::Duration::from_secs(10)).expect("own backend closes");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn begin_after_close_is_backend_closing_and_reports_outlive_quiescence() {
    let own = own_backend(None);
    let ctx = native_context(&own).await;

    let attempt = own.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let mut stream = attempt.sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));

    let close = own.close_and_wait(std::time::Duration::from_secs(10)).expect("close completes");
    assert!(close.gpu_resource_quiescent);
    // The published report remains readable after GPU-resource quiescence:
    // observation state never retains the device lease.
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.timestamps.cleanup_complete_ns.is_some());

    let error = own.begin_attempt(AttemptOptions::default()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("backend_closing"), "begin after close is backend_closing: {error:?}");
    let counters = own.attempt_counters_snapshot();
    assert_eq!(counters.attempts_begun, 1, "a losing begin creates no counted attempt");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_before_execute_and_mid_drain_are_cooperative() {
    // Cancellation before execution starts: no physical start boundary.
    let planned = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let plan = planned.plan_sql(&cpu_only_context(), "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    planned.cancellation().cancel();
    let _error = match planned.execute_physical_plan(plan, &cpu_only_context()) {
        Ok(_) => panic!("a cancelled attempt must not execute"),
        Err(error) => error,
    };
    let report = planned.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
    assert!(report.timestamps.physical_execution_started_ns.is_none());

    // Cancellation during CPU result drain terminalizes the attempt.
    let draining = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = draining.sql(&cpu_only_context(), "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    draining.cancellation().cancel();
    let mut saw_error = false;
    while let Some(batch) = stream.next().await {
        saw_error = saw_error || batch.is_err();
    }
    let report = draining.completion().await;
    assert!(saw_error, "the drain surfaces the cancellation");
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_invocation_and_partial_drain_drop_terminalize() {
    // DataFrame path: same lifecycle coverage through the explicit adapter.
    let ctx = cpu_only_context();
    let attempt = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let df = ctx.sql("SELECT approx_distinct(v) FROM cpu_rows").await.expect("aggregate dataframe");
    let mut stream = attempt.execute_dataframe(df).await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = attempt.completion().await;
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_none());

    // Dropping after a partial drain terminalizes as dropped exactly once.
    let partial = backend().begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = partial.sql(&cpu_only_context(), "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    let first = stream.next().await;
    assert!(matches!(first, Some(Ok(_)) | None));
    drop(stream);
    let report = partial.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deadline_expiry_cancels_planning_and_drain_with_typed_reason() {
    // Expired before planning: the attempt terminalizes cancelled, unplanned.
    let expired = backend()
        .begin_attempt(AttemptOptions {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_millis(1)),
            ..Default::default()
        })
        .unwrap();
    let error = expired.sql(&cpu_only_context(), "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap_err();
    let nexus_error = datafusion_nexus::error::DataFusionErrorExt::find_nexus_error(&error).expect("typed error");
    assert_eq!(nexus_error.fact_value("reason"), Some("deadline_expired"));
    let report = expired.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
    assert_eq!(report.disposition, AttemptDisposition::Unplanned);
    assert!(report.timestamps.physical_execution_started_ns.is_none());

    // Expiring during result drain terminalizes the attempt as cancelled.
    let draining = backend()
        .begin_attempt(AttemptOptions {
            deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(500)),
            ..Default::default()
        })
        .unwrap();
    let mut stream = draining.sql(&cpu_only_context(), "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(700)).await;
    let mut saw_error = false;
    while let Some(batch) = stream.next().await {
        saw_error = saw_error || batch.is_err();
    }
    let report = draining.completion().await;
    assert!(saw_error, "the drain surfaces the expired deadline");
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_rejection_reports_the_typed_rejection_not_a_synthetic_drop() {
    use datafusion_nexus::backend::NexusAdmissionPolicy;
    let own = own_backend_with_policy(
        None,
        NexusAdmissionPolicy::default()
            .with_max_concurrent_queries(1)
            .with_wait_timeout(std::time::Duration::from_millis(50)),
    );
    let ctx = native_context(&own).await;

    // Hold the single admission slot so the attempt's admission must queue
    // and time out instead of being placed.
    let held = own.admit_query().expect("slot holder admits");
    let attempt = own.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let mut stream = attempt.sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    let mut saw_error = false;
    while let Some(batch) = stream.next().await {
        if let Err(error) = batch {
            saw_error = true;
            let message = format!("{error:?}");
            assert!(message.contains("admission") || message.contains("overloaded"), "{message}");
        }
    }
    assert!(saw_error, "the queued admission rejects");
    let report = completion.await;
    assert!(
        matches!(
            &report.outcome,
            RuntimeTerminalOutcome::Failed { error_code, .. } if error_code == "service_overloaded"
        ),
        "the report carries the typed rejection, not a synthetic drop: {:?}",
        report.outcome
    );
    assert!(report.timestamps.admission_requested_ns.is_some(), "queue facts are present");
    assert!(report.timestamps.admitted_ns.is_none());
    assert!(report.timestamps.physical_execution_started_ns.is_none());
    assert!(report.timestamps.cleanup_complete_ns.is_none());
    drop(held);
    own.close_and_wait(std::time::Duration::from_secs(10)).expect("own backend closes");
}
