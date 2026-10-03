//! GPU-lane integration contracts for the explicit attempt adapter.

use std::ops::Deref;

use algeon_datafusion::{
    attempt::{AttemptOptions, RuntimeObservationConfig},
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    error::DataFusionErrorExt,
    native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig, SourceChunking, below_native_admission},
    native_report::NativeDataFusionExecutionMetrics,
    planner::{AlgeonGpuExecutionMode, FinalPlanDisposition, plan_with_native_diagnostics},
};
use datafusion::{
    common::{Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::SessionStateBuilder,
    physical_expr::PhysicalExpr,
    physical_plan::{ExecutionPlan, SendableRecordBatchStream, collect, displayable, execute_stream},
    prelude::SessionContext,
};
use futures::{StreamExt, stream::once};
use query_planning::error::{AttemptFailureCode, QueryErrorCode};
use query_runtime::observability::{
    AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptFailureFactValue,
    AttemptHealthTransition, QueryAttemptReport, RuntimeTerminalOutcome,
};
use tokio::{sync::OwnedSemaphorePermit, time::sleep};

/// Keeps process-wide backend ownership serialized until the backend has
/// closed and released every resource that can retain its owner lease.
struct BackendFixture {
    backend: Option<AlgeonGpuBackend>,
    _guard: OwnedSemaphorePermit,
}

impl BackendFixture {
    fn new(queue_capacity: Option<usize>, resource_policy: AlgeonGpuQueuePolicy) -> Self {
        Self::with_execution_mode(queue_capacity, resource_policy, AlgeonGpuExecutionMode::NativePreferred)
    }

    fn with_execution_mode(
        queue_capacity: Option<usize>,
        resource_policy: AlgeonGpuQueuePolicy,
        execution_mode: AlgeonGpuExecutionMode,
    ) -> Self {
        let guard = super::root_gpu_fixture::lock();
        let profile = AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 });
        let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
        let mut builder = AlgeonGpuBackend::builder()
            .with_native_execution_config(AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ))
            .devices([device])
            .queue_policy(resource_policy)
            .with_execution_mode(execution_mode);
        if let Some(capacity) = queue_capacity {
            builder = builder.with_attempt_completion_queue_capacity(capacity);
        }
        let backend = builder.build().expect("attempt lifecycle tests require CUDA device 0");
        Self { backend: Some(backend), _guard: guard }
    }
}

impl Deref for BackendFixture {
    type Target = AlgeonGpuBackend;

    fn deref(&self) -> &Self::Target {
        self.backend.as_ref().expect("backend fixture remains live")
    }
}

impl Drop for BackendFixture {
    fn drop(&mut self) {
        let Some(backend) = self.backend.take() else {
            return;
        };
        let close = backend.close_and_wait(std::time::Duration::from_secs(10));
        drop(backend);
        if let Err(error) = close
            && !std::thread::panicking()
        {
            panic!("attempt lifecycle backend closes before releasing ownership: {error}");
        }
    }
}

fn backend() -> BackendFixture {
    BackendFixture::new(None, AlgeonGpuQueuePolicy::default())
}

fn context(backend: &AlgeonGpuBackend) -> SessionContext {
    let state = backend.install_on(SessionStateBuilder::new_with_default_features()).expect("install backend").build();
    SessionContext::new_with_state(state)
}

/// A memory-source table never lowers to native execution, so a scan of it is
/// a deterministic CPU-only (`cpu_fallback`) invocation.
fn cpu_only_context(backend: &AlgeonGpuBackend) -> SessionContext {
    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::datasource::MemTable;
    let state = backend.install_on(SessionStateBuilder::new_with_default_features()).expect("install backend").build();
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
    let backend = backend();
    let cpu_ctx = cpu_only_context(&backend);
    let oversized_query = AttemptOptions { query_id: Some("q".repeat(257)), ..Default::default() };
    assert!(backend.begin_attempt(oversized_query).is_err());
    let oversized_trace = AttemptOptions { trace_context: Some("t".repeat(1025)), ..Default::default() };
    assert!(backend.begin_attempt(oversized_trace).is_err());

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let first_id = attempt.attempt_id();
    let mut stream = attempt.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
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

    let second = backend.begin_attempt(AttemptOptions::default()).unwrap();
    assert_ne!(first_id, second.attempt_id());
}

#[tokio::test]
async fn planning_failure_cancellation_drop_and_disabled_observation_publish_reports() {
    let backend = backend();
    let ctx = context(&backend);
    let cpu_ctx = cpu_only_context(&backend);
    let planning = backend.begin_attempt(AttemptOptions::default()).unwrap();
    assert!(planning.sql(&ctx, "SELECT FROM").await.is_err());
    let planning_report = planning.completion().await;
    assert_eq!(planning_report.disposition, AttemptDisposition::Unplanned);
    assert!(matches!(planning_report.outcome, RuntimeTerminalOutcome::Failed { .. }));
    assert!(planning_report.timestamps.physical_execution_started_ns.is_none());

    let cancelled = backend.begin_attempt(AttemptOptions::default()).unwrap();
    cancelled.cancellation().cancel();
    assert!(cancelled.sql(&ctx, "SELECT 1").await.is_err());
    assert!(matches!(cancelled.completion().await.outcome, RuntimeTerminalOutcome::Cancelled));

    let dropped = backend
        .begin_attempt(AttemptOptions { observation: RuntimeObservationConfig::disabled(), ..Default::default() })
        .unwrap();
    let stream = dropped.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    drop(stream);
    let report = dropped.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
}

async fn native_context(backend: &AlgeonGpuBackend) -> SessionContext {
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
        "algeon-attempt-lifecycle-{}-{}.parquet",
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
    let state = backend.install_on(SessionStateBuilder::new_with_default_features()).expect("install backend").build();
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
    let backend = backend();
    let ctx = native_context(&backend).await;
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
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
    assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed);
    assert_eq!(
        report.capacity_disposition,
        AttemptCapacityDisposition::Released,
        "outer EOF terminalization precedes the native round cleanup commit"
    );
    assert_eq!(report.health_transition, Some(AttemptHealthTransition::HealthyReuse));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reused_plans_mint_fresh_attempts_and_never_observe_plan_lifetime_attempts() {
    let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
    let ctx = native_context(&backend).await;
    let planner = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let plan = planner.plan_sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    let planner_id = planner.attempt_id();
    assert!(
        below_native_admission(&plan)
            .metrics()
            .is_none_or(|metrics| metrics.sum_by_name("attempt_report_rows").is_none()),
        "retry advice cannot appear in DataFusion metrics before the attempt seals"
    );

    // The planning attempt executes its own plan.
    let mut stream = planner.execute_physical_plan(std::sync::Arc::clone(&plan), &ctx).unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = planner.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    let metrics = below_native_admission(&plan).metrics().expect("native root exposes DataFusion metrics");
    let execution = planner.sealed_native_execution_snapshot().expect("sealed native execution snapshot");
    assert_eq!(execution.attempt_id, planner_id);
    assert_eq!(execution.coverage.disposition(), FinalPlanDisposition::Native);
    assert_eq!(execution.coverage.native_fragment_count(), 1);
    assert_eq!(execution.coverage.explicit_gpu_island_count(), 0);
    assert_eq!(execution.coverage.datafusion_boundary_count(), 0);
    assert_eq!(execution.terminal_host_materialization_count, 1);
    assert_eq!(execution.intermediate_device_to_host_materialization_count, 0);
    assert_eq!(execution.metrics, NativeDataFusionExecutionMetrics::from_metrics(&metrics));
    assert_eq!(metrics.sum_by_name("attempt_report_rows").unwrap().as_usize(), 1);
    let sealed_rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == "attempt_report_rows")
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "retry_recovery"))
        .collect::<Vec<_>>();
    assert_eq!(sealed_rows.len(), 1, "one elected root publishes one post-seal row per attempt");
    let labels = sealed_rows[0]
        .labels()
        .iter()
        .map(|label| (label.name(), label.value()))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(labels.get("retry_recovery"), Some(&"no_retry_needed"));
    assert_eq!(labels.get("retry_prerequisite"), Some(&"none"));
    assert!(!labels.contains_key("attempt_id"));
    assert!(!labels.contains_key("device_ordinal"));

    // A second attempt reuses the same plan and receives a fresh identity and
    // its own admission facts; nothing records into the first attempt.
    let second = backend.begin_attempt(AttemptOptions::default()).unwrap();
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
    assert_eq!(
        below_native_admission(&plan).metrics().unwrap().sum_by_name("attempt_report_rows").unwrap().as_usize(),
        2,
        "reused wrappers project one counter row per sealed attempt rather than sharing terminal state"
    );
    assert!(
        report.timestamps.planning_started_ns.is_none(),
        "an execute-only attempt does not synthesize planning boundaries"
    );

    // A third invocation reuses the SQL text and still mints a fresh id.
    let third = backend.begin_attempt(AttemptOptions::default()).unwrap();
    assert_ne!(third.attempt_id(), planner_id);
    assert_ne!(third.attempt_id(), second_id);

    // The same plan executed through a raw session without an attempt runs
    // as an internal attempt of its own: fragments cannot observe a
    // plan-lifetime attempt.
    let raw_state =
        backend.install_on(SessionStateBuilder::new_with_default_features()).expect("install backend").build();
    let raw_ctx = SessionContext::new_with_state(raw_state);
    drain(execute_stream(plan, raw_ctx.task_ctx()).expect("raw execution starts")).await;
    let report = sole_abandoned_report(&backend).await;
    assert!(![planner_id, second_id].contains(&report.attempt_id), "raw execution begins a fresh attempt");
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_some(), "the internal attempt admits the raw execution");
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
        ) -> DataFusionResult<std::sync::Arc<dyn ExecutionPlan>> {
            Ok(self)
        }
        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&std::sync::Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }
        fn execute(
            &self,
            _partition: usize,
            _context: std::sync::Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            let stream = once(async {
                panic!("injected attempt stream panic");
            });
            Ok(Box::pin(RecordBatchStreamAdapter::new(std::sync::Arc::clone(&self.schema), stream)))
        }
    }

    let backend = backend();
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

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let mut stream = attempt.execute_physical_plan(plan, &context(&backend)).unwrap();
    let caught = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(stream.next())).await;
    assert!(caught.is_err(), "the panic unwinds through the algeon boundary");
    drop(stream);
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Panicked));
    assert!(matches!(report.disposition, AttemptDisposition::CpuFallback));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_reports_are_best_effort_delivery_loss_only() {
    let backend = BackendFixture::new(Some(1), AlgeonGpuQueuePolicy::default());

    let cpu_ctx = cpu_only_context(&backend);
    let first = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let first_id = first.attempt_id();
    drop(first.completion());
    let mut stream = first.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = first.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));

    let drained = backend.take_abandoned_attempt_reports(8);
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].attempt_id, first_id);
    let counters = backend.attempt_counters_snapshot();
    assert_eq!(counters.completion_queue_pushed, 1);
    assert_eq!(counters.completion_queue_overflowed, 0);

    // Two further abandoned reports overflow the capacity-one queue.
    for _ in 0..2 {
        let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
        drop(attempt.completion());
        let mut stream = attempt.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
        while let Some(batch) = stream.next().await {
            batch.unwrap();
        }
        let _ = attempt.completion().await;
    }
    let counters = backend.attempt_counters_snapshot();
    assert_eq!(counters.completion_queue_pushed, 2);
    assert_eq!(counters.completion_queue_overflowed, 1);
    assert_eq!(backend.take_abandoned_attempt_reports(8).len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_before_planning_seals_backend_closing_without_admission() {
    let backend = backend();
    let cpu_ctx = cpu_only_context(&backend);
    let attempt = backend.begin_attempt(AttemptOptions::default()).expect("attempt wins the begin gate before close");
    let completion = attempt.completion();

    assert!(backend.close(), "the close signal wins after the attempt starts");
    let error = attempt
        .plan_sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows")
        .await
        .expect_err("a pre-planning attempt cannot create a plan after close");
    let algeon_error = DataFusionErrorExt::find_algeon_error(&error).expect("structured close error");
    assert_eq!(algeon_error.fact_value("reason"), Some("backend_closing"));

    let report = completion.await;
    assert!(matches!(
        report.outcome,
        RuntimeTerminalOutcome::Failed { error_code: AttemptFailureCode::BackendClosing, .. }
    ));
    assert_eq!(report.disposition, AttemptDisposition::Unplanned);
    assert!(report.admission.is_none());
    assert!(report.timestamps.planning_started_ns.is_none());
    assert!(report.timestamps.admission_requested_ns.is_none());
    assert!(report.timestamps.physical_execution_started_ns.is_none());
    assert_eq!(report.primary_failure.as_ref().map(|summary| summary.code()), Some(AttemptFailureCode::BackendClosing));

    let close = backend
        .close_and_wait(std::time::Duration::from_secs(10))
        .expect("unplanned close report releases observation scope");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_during_cpu_result_drain_times_out_then_recovers_after_one_report() {
    let backend = backend();
    let cpu_ctx = cpu_only_context(&backend);
    let attempt = backend.begin_attempt(AttemptOptions::default()).expect("attempt begins before close");
    let completion = attempt.completion();
    let mut stream =
        attempt.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.expect("CPU result stream starts");

    assert!(backend.close(), "the result drain starts before backend close");
    let timeout = backend
        .close_and_wait(std::time::Duration::from_millis(25))
        .expect_err("the live result stream retains its observation scope during close");
    assert_eq!(timeout.fact_value("reason"), Some("backend_close_timeout"));
    assert_eq!(timeout.fact_value("gpu_resource_quiescent"), Some("true"));
    assert_eq!(timeout.fact_value("observation_quiescent"), Some("false"));

    while let Some(batch) = stream.next().await {
        batch.expect("CPU result drain remains usable after close starts");
    }
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
    assert!(report.timestamps.physical_execution_started_ns.is_some());
    assert!(report.timestamps.report_ready_ns.is_some());

    let close =
        backend.close_and_wait(std::time::Duration::from_secs(10)).expect("EOF releases the final result-drain scope");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_installed_context_does_not_block_close_and_reports_outlive_quiescence() {
    let backend = backend();
    let ctx = native_context(&backend).await;

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let mut stream = attempt.sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));

    // An idle installed SessionContext does not block close or quiescence:
    let close = backend.close_and_wait(std::time::Duration::from_secs(10)).expect("close completes");
    assert!(close.gpu_resource_quiescent);
    // The published report remains readable after GPU-resource quiescence:
    // observation state never retains the device lease.
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.timestamps.cleanup_complete_ns.is_some());

    let error = backend.begin_attempt(AttemptOptions::default()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("backend_closing"), "begin after close is backend_closing: {error:?}");
    let counters = backend.attempt_counters_snapshot();
    assert_eq!(counters.attempts_begun, 1, "a losing begin creates no counted attempt");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_before_execute_and_mid_drain_are_cooperative() {
    let backend = backend();
    let cpu_ctx = cpu_only_context(&backend);
    // Cancellation before execution starts: no physical start boundary.
    let planned = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let plan = planned.plan_sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    planned.cancellation().cancel();
    let _error = match planned.execute_physical_plan(plan, &cpu_ctx) {
        Ok(_) => panic!("a cancelled attempt must not execute"),
        Err(error) => error,
    };
    let report = planned.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
    assert!(report.timestamps.physical_execution_started_ns.is_none());

    // Cancellation during CPU result drain terminalizes the attempt.
    let draining = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = draining.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    draining.cancellation().cancel();
    let cancellation = stream.next().await.expect("cancellation produces one terminal stream item");
    assert!(cancellation.is_err(), "the drain surfaces the cancellation");
    assert!(stream.next().await.is_none(), "the cancellation error exhausts the wrapper stream");
    assert!(stream.next().await.is_none(), "later polls must not re-enter the inner stream");
    let report = draining.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));

    // Flight cancellation may discard the response stream before its next
    // poll. Once cancellation has propagated, `AttemptStream::drop` must
    // retain the cancelled terminal outcome instead of rewriting it as a
    // client-drop outcome.
    let cancelled_then_dropped = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let stream = cancelled_then_dropped.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    cancelled_then_dropped.cancellation().cancel();
    drop(stream);
    let report = cancelled_then_dropped.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_invocation_and_partial_drain_drop_terminalize() {
    let backend = backend();
    // DataFrame path: same lifecycle coverage through the explicit adapter.
    let ctx = cpu_only_context(&backend);
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
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
    let partial = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = partial.sql(&ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    let first = stream.next().await;
    assert!(matches!(first, Some(Ok(_)) | None));
    drop(stream);
    let report = partial.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deadline_expiry_cancels_planning_and_drain_with_typed_reason() {
    let backend = backend();
    let cpu_ctx = cpu_only_context(&backend);
    // Expired before planning: the attempt terminalizes cancelled, unplanned.
    let expired = backend
        .begin_attempt(AttemptOptions {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_millis(1)),
            ..Default::default()
        })
        .unwrap();
    let error = expired.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap_err();
    let algeon_error = DataFusionErrorExt::find_algeon_error(&error).expect("typed error");
    assert_eq!(algeon_error.fact_value("reason"), Some("deadline_expired"));
    let report = expired.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
    assert_eq!(
        report.primary_failure.as_ref().and_then(|summary| summary.facts().first()).map(|fact| fact.value()),
        Some(AttemptFailureFactValue::DeadlineExpired)
    );
    assert_eq!(report.disposition, AttemptDisposition::Unplanned);
    assert!(report.timestamps.physical_execution_started_ns.is_none());

    // Expiring during result drain terminalizes the attempt as cancelled.
    let draining = backend
        .begin_attempt(AttemptOptions {
            deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(500)),
            ..Default::default()
        })
        .unwrap();
    let mut stream = draining.sql(&cpu_ctx, "SELECT approx_distinct(v) FROM cpu_rows").await.unwrap();
    sleep(std::time::Duration::from_millis(700)).await;
    let mut saw_error = false;
    while let Some(batch) = stream.next().await {
        saw_error = saw_error || batch.is_err();
    }
    let report = draining.completion().await;
    assert!(saw_error, "the drain surfaces the expired deadline");
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled));
    assert_eq!(
        report.primary_failure.as_ref().and_then(|summary| summary.facts().first()).map(|fact| fact.value()),
        Some(AttemptFailureFactValue::DeadlineExpired)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_rejection_reports_the_typed_rejection_not_a_synthetic_drop() {
    use AlgeonGpuQueuePolicy;
    let backend = BackendFixture::new(
        None,
        AlgeonGpuQueuePolicy::default().with_max_queue_wait(std::time::Duration::from_millis(50)),
    );
    let ctx = native_context(&backend).await;

    // Hold both admission slots with live suspended executions so the
    // attempt's admission must queue and time out instead of being placed.
    // Each holder stream is polled once to admit, then left suspended: the
    // slots stay occupied until the streams are dropped, never reaching EOF.
    let holder = backend.begin_attempt(AttemptOptions::default()).expect("slot holder admits");
    let mut holder_stream = holder.sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    holder_stream.next().await.expect("holder admits on first poll").expect("holder batch");
    let second_holder = backend.begin_attempt(AttemptOptions::default()).expect("second slot holder admits");
    let mut second_holder_stream =
        second_holder.sql(&ctx, "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k").await.unwrap();
    second_holder_stream.next().await.expect("second holder admits on first poll").expect("holder batch");
    let _held = (holder, holder_stream, second_holder, second_holder_stream);
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
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
            RuntimeTerminalOutcome::Failed { error_code, .. } if *error_code == AttemptFailureCode::Query(QueryErrorCode::ServiceOverloaded)
        ),
        "the report carries the typed rejection, not a synthetic drop: {:?}",
        report.outcome
    );
    assert!(report.timestamps.admission_requested_ns.is_some(), "queue facts are present");
    assert!(report.timestamps.admitted_ns.is_none());
    assert!(report.timestamps.physical_execution_started_ns.is_none());
    assert!(report.timestamps.cleanup_complete_ns.is_none());
    drop(_held);
}

/// DataFusion `EXPLAIN` runs the explained query through the native stages,
/// so the attempt records that query's diagnostics, and executing the
/// explanation requests no admission.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explain_inside_an_attempt_records_the_query_diagnostics_without_admission() {
    let backend = backend();
    let ctx = native_context(&backend).await;
    let query = "SELECT k, SUM(v) FROM native_rows GROUP BY k";
    let logical = ctx.state().create_logical_plan(query).await.unwrap();
    let (_, expected) = plan_with_native_diagnostics(&ctx.state(), &logical).await.unwrap();
    assert!(expected.report.selected_count() > 0, "the explained query plans natively");

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = attempt.sql(&ctx, &format!("EXPLAIN {query}")).await.unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    let diagnostics = attempt.planning_diagnostics().expect("EXPLAIN planning records diagnostics");
    assert_eq!(diagnostics.input_plan_root, expected.input_plan_root);
    assert_eq!(diagnostics.report.outcomes(), expected.report.outcomes());

    let report = attempt.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_none());
}

/// `algeon_explain_coverage` plans its query without the calling attempt: a
/// CPU-only query over coverage rows stays `cpu_fallback` even when the
/// validated query would run natively.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn coverage_planned_inside_an_attempt_keeps_the_outer_disposition() {
    use arrow::array::AsArray;
    let backend = backend();
    let ctx = native_context(&backend).await;
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = attempt
        .sql(
            &ctx,
            "SELECT gpu_path FROM algeon_explain_coverage('SELECT k, SUM(v) FROM native_rows GROUP BY k') \
             WHERE row_kind = 'summary'",
        )
        .await
        .unwrap();
    let mut gpu_paths = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        gpu_paths.extend(batch.column(0).as_string::<i32>().iter().map(|path| path.unwrap().to_string()));
    }
    assert_eq!(gpu_paths, ["native"], "the validated query plans natively");

    let report = attempt.completion().await;
    assert_eq!(report.disposition, AttemptDisposition::CpuFallback);
    assert!(report.admission.is_none());
}

async fn drain(mut stream: SendableRecordBatchStream) -> usize {
    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        rows += batch.expect("stream batch").num_rows();
    }
    rows
}

/// Waits for `count` reports of attempts no caller awaits, such as the
/// internal attempt of a plain `collect`. An admitted attempt publishes its
/// report after GPU cleanup, which can finish after its result stream ends.
async fn abandoned_reports(backend: &AlgeonGpuBackend, count: usize) -> Vec<std::sync::Arc<QueryAttemptReport>> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut reports = backend.take_abandoned_attempt_reports(8);
    while reports.len() < count {
        assert!(std::time::Instant::now() < deadline, "{} of {count} attempt reports published", reports.len());
        sleep(std::time::Duration::from_millis(5)).await;
        reports.extend(backend.take_abandoned_attempt_reports(8));
    }
    assert_eq!(reports.len(), count, "each execution publishes exactly one attempt report");
    reports
}

async fn sole_abandoned_report(backend: &AlgeonGpuBackend) -> std::sync::Arc<QueryAttemptReport> {
    abandoned_reports(backend, 1).await.remove(0)
}

const NATIVE_AGGREGATE_SQL: &str = "SELECT k, SUM(v) FROM native_rows GROUP BY k ORDER BY k";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_collect_runs_as_one_internal_attempt() {
    let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
    let ctx = native_context(&backend).await;
    let begun = backend.attempt_counters_snapshot().attempts_begun;

    let batches = ctx.sql(NATIVE_AGGREGATE_SQL).await.unwrap().collect().await.unwrap();

    assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 2);
    assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun + 1);
    let report = sole_abandoned_report(&backend).await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert_eq!(report.disposition, AttemptDisposition::Native);
    assert!(report.admission.is_some(), "the internal attempt admits the native plan once");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repeated_plain_execution_of_one_plan_runs_one_attempt_each_time() {
    let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
    let ctx = native_context(&backend).await;
    let plan = ctx.sql(NATIVE_AGGREGATE_SQL).await.unwrap().create_physical_plan().await.unwrap();

    let mut attempt_ids = Vec::new();
    for _ in 0..2 {
        assert_eq!(drain(execute_stream(std::sync::Arc::clone(&plan), ctx.task_ctx()).unwrap()).await, 2);
        let report = sole_abandoned_report(&backend).await;
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
        assert!(report.admission.is_some());
        attempt_ids.push(report.attempt_id);
    }
    assert_ne!(attempt_ids[0], attempt_ids[1], "each execution is its own attempt");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_plain_executions_on_one_context_run_one_attempt_each() {
    let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
    let ctx = native_context(&backend).await;
    let plan = ctx.sql(NATIVE_AGGREGATE_SQL).await.unwrap().create_physical_plan().await.unwrap();

    let (first, second) = tokio::join!(
        drain(execute_stream(std::sync::Arc::clone(&plan), ctx.task_ctx()).unwrap()),
        drain(execute_stream(std::sync::Arc::clone(&plan), ctx.task_ctx()).unwrap()),
    );

    assert_eq!((first, second), (2, 2));
    let reports = abandoned_reports(&backend, 2).await;
    assert_ne!(reports[0].attempt_id, reports[1].attempt_id);
    for report in &reports {
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
        assert!(report.admission.is_some());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plans_that_read_no_column_data_begin_no_attempt_unless_native_is_required() {
    // A DDL result's empty relation, `SELECT 1`'s placeholder row, and a
    // relation DataFusion folds to empty carry no column data a GPU could
    // accelerate, so NativePreferred keeps them on DataFusion instead of
    // queueing for a grant.
    {
        let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
        let ctx = native_context(&backend).await;
        let begun = backend.attempt_counters_snapshot().attempts_begun;
        ctx.sql("CREATE VIEW plain_view AS SELECT k FROM native_rows").await.unwrap().collect().await.unwrap();
        assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun, "a view's DDL result begins no attempt");
        let batches = ctx.sql("SELECT 1").await.unwrap().collect().await.unwrap();
        assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 1);
        assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun, "SELECT 1 begins no attempt");
        for sql in ["SELECT k FROM native_rows WHERE false", "SELECT k FROM native_rows LIMIT 0"] {
            let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 0);
            assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun, "{sql} begins no attempt");
        }
    }

    // NativeRequired leaves no DataFusion fallback, so the literal stays native.
    let backend = BackendFixture::with_execution_mode(
        Some(8),
        AlgeonGpuQueuePolicy::default(),
        AlgeonGpuExecutionMode::NativeRequired,
    );
    let ctx = native_context(&backend).await;
    let begun = backend.attempt_counters_snapshot().attempts_begun;
    let batches = ctx.sql("SELECT 1").await.unwrap().collect().await.unwrap();
    assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 1);
    assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun + 1, "a required native literal runs");
}

/// Every native fragment of `plan`.
fn native_fragments(plan: &dyn ExecutionPlan) -> usize {
    usize::from(plan.downcast_ref::<AlgeonNativeExec>().is_some())
        + plan.children().into_iter().map(|child| native_fragments(child.as_ref())).sum::<usize>()
}

/// Each union branch keeps its `approx_distinct` on DataFusion over a native
/// fragment of the filtered scan.
const TWO_FRAGMENT_SQL: &str = "SELECT approx_distinct(v) AS d FROM native_rows WHERE v > 15 \
                                UNION ALL SELECT approx_distinct(v) AS d FROM native_rows WHERE v < 35";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_fragments_under_datafusion_operators_share_the_attempts_one_grant() {
    let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
    let ctx = native_context(&backend).await;
    let plan = ctx.sql(TWO_FRAGMENT_SQL).await.unwrap().create_physical_plan().await.unwrap();
    assert_eq!(native_fragments(plan.as_ref()), 2, "{}", displayable(plan.as_ref()).indent(false));
    let begun = backend.attempt_counters_snapshot().attempts_begun;

    let batches = collect(plan, ctx.task_ctx()).await.unwrap();

    assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 2);
    assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun + 1);
    let report = sole_abandoned_report(&backend).await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded), "{:?}", report.outcome);
    assert_eq!(report.disposition, AttemptDisposition::PartialNative);
    assert!(report.admission.is_some(), "both fragments run under the attempt's one admission");
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
}

/// A DataFusion operator that fails above a native fragment fails the
/// attempt, which still releases its grant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failure_of_a_datafusion_operator_over_a_native_fragment_fails_the_attempt() {
    let backend = backend();
    let ctx = native_context(&backend).await;
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    // The aggregate runs as a native fragment. Its keys "a" and "b" do not
    // parse as timestamps, so the DataFusion projection above it fails.
    let mut stream =
        attempt.sql(&ctx, "SELECT to_timestamp(k) AS t, SUM(v) AS s FROM native_rows GROUP BY k").await.unwrap();
    let mut failed = false;
    while let Some(batch) = stream.next().await {
        failed |= batch.is_err();
    }
    assert!(failed, "the DataFusion operator's error ends the stream");

    let report = attempt.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Failed { .. }), "{:?}", report.outcome);
    assert_eq!(report.disposition, AttemptDisposition::PartialNative);
    assert!(report.admission.is_some(), "the native fragment ran under the attempt's grant");
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
}

/// Cancelling while DataFusion operators read native fragments ends the
/// attempt as cancelled and releases its grant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_of_datafusion_operators_over_native_fragments_releases_the_grant() {
    let backend = backend();
    let ctx = native_context(&backend).await;
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = attempt.sql(&ctx, TWO_FRAGMENT_SQL).await.unwrap();
    stream.next().await.expect("the first union branch emits a row").unwrap();
    attempt.cancellation().cancel();
    let cancellation = stream.next().await.expect("cancellation produces one terminal stream item");
    assert!(cancellation.is_err(), "the drain surfaces the cancellation");
    assert!(stream.next().await.is_none());

    let report = attempt.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled), "{:?}", report.outcome);
    assert_eq!(report.disposition, AttemptDisposition::PartialNative);
    assert!(report.admission.is_some(), "a fragment ran before the cancellation");
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
}

/// A limit on a DataFusion filter stops reading the native fragment below it
/// once it has its row; the attempt succeeds and releases its grant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limit_over_a_native_fragment_stops_early_and_succeeds() {
    let backend = backend();
    let ctx = native_context(&backend).await;
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let mut stream = attempt.sql(&ctx, "SELECT k FROM native_rows WHERE k ILIKE 'A' LIMIT 1").await.unwrap();
    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        rows += batch.unwrap().num_rows();
    }
    assert_eq!(rows, 1);

    let report = attempt.completion().await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded), "{:?}", report.outcome);
    assert_eq!(report.disposition, AttemptDisposition::PartialNative);
    assert!(report.admission.is_some());
    assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ctas_inner_plan_runs_inside_one_attempt() {
    let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
    let ctx = native_context(&backend).await;
    let begun = backend.attempt_counters_snapshot().attempts_begun;

    // DataFusion executes the inner plan of a CTAS while it builds the outer
    // statement; without an attempt, that execution is the internal attempt.
    ctx.sql(&format!("CREATE TABLE plain_totals AS {NATIVE_AGGREGATE_SQL}")).await.unwrap();
    assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun + 1);
    let report = sole_abandoned_report(&backend).await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_some());

    // Planned inside an attempt, the inner plan admits under that attempt's
    // ticket and begins no internal attempt. The admission belongs to the
    // attempt, not to the inner plan's stream: the attempt stays live until
    // the statement's own result stream ends.
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let plan = attempt.plan_sql(&ctx, &format!("CREATE TABLE attempt_totals AS {NATIVE_AGGREGATE_SQL}")).await.unwrap();
    assert!(!completion.is_ready(), "the inner plan's end does not end the attempt");
    let mut stream = attempt.execute_physical_plan(plan, &ctx).unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    drop(stream);
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
    assert!(report.admission.is_some(), "the inner plan admits under the attempt");
    assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun + 2);
    assert!(backend.take_abandoned_attempt_reports(8).is_empty());
    let rows = ctx.sql("SELECT COUNT(*) FROM attempt_totals").await.unwrap().collect().await.unwrap();
    assert_eq!(rows[0].num_rows(), 1, "the table exists after the statement succeeds");

    // A cancellation between planning and the result reaches the report.
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let plan =
        attempt.plan_sql(&ctx, &format!("CREATE TABLE cancelled_totals AS {NATIVE_AGGREGATE_SQL}")).await.unwrap();
    attempt.cancellation().cancel();
    let error = attempt.execute_physical_plan(plan, &ctx).err();
    drop(error);
    drop(attempt);
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Cancelled), "{:?}", report.outcome);
    assert!(report.admission.is_some(), "the cancelled attempt had admitted its inner plan");
}

/// Planning again in an attempt whose result stream already started cannot
/// replace the executing native root as the target of the sealed metrics.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn planning_after_the_result_stream_started_keeps_the_sealed_metrics_target() {
    let backend = backend();
    let ctx = native_context(&backend).await;
    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let mut stream = attempt.sql(&ctx, NATIVE_AGGREGATE_SQL).await.unwrap();
    attempt.plan_sql(&ctx, NATIVE_AGGREGATE_SQL).await.expect("the attempt still plans");
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    drop(stream);
    completion.await;
    let sealed = attempt.sealed_native_execution_snapshot().expect("the executed native root seals its metrics");
    assert!(sealed.metrics.parquet_files > 0, "the sealed metrics are the executed root's: {:?}", sealed.metrics);
    assert!(sealed.metrics.device_to_host_materializations > 0, "{:?}", sealed.metrics);
}

/// Holds the attempt's grant until the root stream ends: an operator above
/// the native fragment that drops its input early (as `SortExec`, TopK, and
/// `LimitStream` do) must neither release the grant nor seal the report, and
/// a failure after that drop is the attempt's outcome.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn internal_attempt_holds_its_grant_until_the_root_stream_ends() {
    use datafusion::{
        arrow::record_batch::RecordBatch,
        common::DataFusionError,
        execution::TaskContext,
        physical_plan::{
            DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
            execution_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions},
            stream::RecordBatchStreamAdapter,
        },
    };
    use futures::{TryStreamExt, stream::iter};

    /// Drains and drops its input before it emits anything, then emits the
    /// input's batches, then optionally fails.
    #[derive(Debug)]
    struct DrainThenEmitExec {
        input: std::sync::Arc<dyn ExecutionPlan>,
        fail_after: bool,
    }
    impl DisplayAs for DrainThenEmitExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("DrainThenEmitExec")
        }
    }
    impl ExecutionPlan for DrainThenEmitExec {
        fn name(&self) -> &str {
            "DrainThenEmitExec"
        }
        fn properties(&self) -> &std::sync::Arc<PlanProperties> {
            self.input.properties()
        }
        fn children(&self) -> Vec<&std::sync::Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }
        fn with_new_children(
            self: std::sync::Arc<Self>,
            children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<std::sync::Arc<dyn ExecutionPlan>> {
            let [input] = children.try_into().expect("one child");
            Ok(std::sync::Arc::new(Self { input, fail_after: self.fail_after }))
        }
        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&std::sync::Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }
        fn execute(
            &self,
            partition: usize,
            context: std::sync::Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            let input = self.input.execute(partition, context)?;
            let schema = input.schema();
            let fail_after = self.fail_after;
            let stream = once(async move {
                let batches = input.try_collect::<Vec<RecordBatch>>().await?;
                let mut items = batches.into_iter().map(Ok).collect::<Vec<_>>();
                if fail_after {
                    items.push(Err(DataFusionError::Execution("injected failure after the native input".into())));
                }
                Ok::<_, DataFusionError>(iter(items))
            })
            .try_flatten();
            Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
        }
    }

    let backend = BackendFixture::new(Some(8), AlgeonGpuQueuePolicy::default());
    let ctx = native_context(&backend).await;
    let planned = ctx.sql(NATIVE_AGGREGATE_SQL).await.unwrap().create_physical_plan().await.unwrap();
    assert_eq!(planned.name(), "NativeAdmissionExec");
    let native = std::sync::Arc::clone(below_native_admission(&planned));

    for fail_after in [false, true] {
        let drain = std::sync::Arc::new(DrainThenEmitExec { input: std::sync::Arc::clone(&native), fail_after });
        let plan = std::sync::Arc::clone(&planned)
            .replace_children(vec![drain], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap();
        // No attempt: the admission root begins an internal attempt.
        let mut stream = execute_stream(plan, ctx.task_ctx()).unwrap();
        stream.next().await.expect("the drained rows are emitted").expect("the first batch succeeds");
        assert_eq!(
            backend.admission_snapshot().unwrap().active_queries,
            1,
            "the native fragment ended, but the root stream still holds the attempt's grant"
        );
        assert!(backend.take_abandoned_attempt_reports(8).is_empty(), "no report before the root stream ends");
        let last = stream.next().await;
        if fail_after {
            assert!(matches!(last, Some(Err(_))), "the injected failure ends the root stream");
        } else {
            assert!(last.is_none(), "the root stream ends after its one batch");
        }
        drop(stream);
        let report = sole_abandoned_report(&backend).await;
        if fail_after {
            assert!(matches!(report.outcome, RuntimeTerminalOutcome::Failed { .. }), "{:?}", report.outcome);
        } else {
            assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded), "{:?}", report.outcome);
        }
    }
}

/// Under NativeRequired the CTAS result relation, a zero-column literal, is
/// native as well. It is planned after the inner plan claimed the attempt's
/// admission, so it joins that grant rather than claiming a second one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_required_ctas_result_joins_the_inner_plans_grant() {
    let backend = BackendFixture::with_execution_mode(
        Some(8),
        AlgeonGpuQueuePolicy::default(),
        AlgeonGpuExecutionMode::NativeRequired,
    );
    let ctx = native_context(&backend).await;
    let begun = backend.attempt_counters_snapshot().attempts_begun;

    let attempt = backend.begin_attempt(AttemptOptions::default()).unwrap();
    let completion = attempt.completion();
    let plan = attempt
        .plan_sql(&ctx, &format!("CREATE TABLE required_totals AS {NATIVE_AGGREGATE_SQL}"))
        .await
        .expect("the result relation planned after the claim joins the attempt's grant");
    let mut stream = attempt.execute_physical_plan(plan, &ctx).unwrap();
    assert!(!completion.is_ready(), "the report waits for the statement's own result stream");
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    drop(stream);
    let report = completion.await;
    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded), "{:?}", report.outcome);
    assert!(report.admission.is_some(), "both plans run under the attempt's one admission");
    assert_eq!(backend.attempt_counters_snapshot().attempts_begun, begun + 1, "no plan begins an internal attempt");
    let rows = ctx.sql("SELECT COUNT(*) FROM required_totals").await.unwrap().collect().await.unwrap();
    assert_eq!(rows[0].num_rows(), 1, "the table exists after the statement succeeds");
}
