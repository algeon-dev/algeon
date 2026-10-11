//! Resident chunk-lane execution and source-work ownership regressions.
//!
//! These cases own the whole device while native event and stream hooks are
//! active; the `gpu_exclusive` module name selects the exclusive nextest lane.

use std::{
    cell::RefCell,
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::testing::{event::EventSynchronizeFault, stream::StreamSynchronizeBlock};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_planning::{
    admission::{AttemptCapabilityEvidence, admit},
    error::{QueryErrorCode, Result},
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, LimitSpec},
    mode::StatisticalAggregateExecutionMode,
    pipeline::{CompiledPipeline, PipelineTaskKind, ResidentDataForm},
    plan::{ProjectionOutput, QueryPlan, QueryPlanBuilder},
    policy::PlanningPolicy,
    source::{LiteralBatchSource, LocalParquetSource, SourceNode},
};
use query_runtime::{
    attempt::{AttemptOwner, BeginAttemptRequest},
    metrics::QueryMetrics,
    observability::{
        AttemptDisposition, AttemptQueryId, PlanExecutionId, RuntimeObservationConfig, RuntimeObservationOwner,
        RuntimeObservationPayload, RuntimePlanExecutionLifecycle, RuntimeTaskId, RuntimeTerminalOutcome,
        RuntimeTimelineEvent, RuntimeWorkerId, StreamLeaseLifecycleState,
    },
    runtime::{
        AdmittedQueryContext, ExecutionContext, GpuBackendConfig, GpuBackendCore, GpuBackendDeviceConfig,
        GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuExecutionJobKind, GpuMemoryOwnership,
        QueryAdmissionRequest, QueryExecutionConfig, QueryHandle, QueueBounds, resident::GpuResidentInputMap,
    },
};
use tokio::runtime::Builder as TokioRuntimeBuilder;

use super::{LaneDispatchSupport, run_resident_lane_borrow};
use crate::{
    exec::{
        ExecutionMetrics, QueryExecutionRequest, admit_query_plan,
        bounded::compiled::{CompiledGpuTaskExecutor, CompiledPipelineScheduler},
        execute_query_plan, materialize_gpu_result_to_host,
        output::QueryOutput,
    },
    rapids_backend::rapids_analysis_context,
};

static LANE_BACKEND: Mutex<Option<GpuBackendCore>> = Mutex::new(None);

fn build_lane_backend() -> Result<GpuBackendCore> {
    let policy = QueryExecutionConfig {
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
        ..QueryExecutionConfig::default()
    };
    // Two forks per attempt back two execution lanes; two execution workers,
    // beside the worker reserved for lifecycle jobs, let both lanes run at once
    // instead of queueing behind one worker.
    let device = GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(0)
            .with_max_active_attempts(1)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(4, 3, 2))
            .with_source_work_per_attempt(2),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    GpuBackendCore::new(
        GpuBackendConfig::new(vec![device], policy, QueueBounds::default()).validate().expect("valid backend config"),
    )
}

fn ensure_lane_backend(backend: &mut Option<GpuBackendCore>) -> Result<()> {
    if backend.is_none() {
        *backend = Some(build_lane_backend()?);
    }
    Ok(())
}

struct LaneBackendGuard {
    backend: RefCell<MutexGuard<'static, Option<GpuBackendCore>>>,
}

impl LaneBackendGuard {
    fn acquire() -> Result<Self> {
        let backend = LANE_BACKEND.lock().expect("lane backend lock is not poisoned");
        Ok(Self { backend: RefCell::new(backend) })
    }

    fn admit(&self, plan: &QueryPlan, cap_bytes: Option<u64>) -> Result<LaneTestContext> {
        admit_query_plan(plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default())?;
        let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
        let mut backend = self.backend.borrow_mut();
        ensure_lane_backend(&mut backend)?;
        let mut request = QueryAdmissionRequest::test_attempt().with_query_plan_capability_evidence(evidence);
        if let Some(cap) = cap_bytes {
            request = request.with_query_cap_bytes(cap);
        }
        let handle = backend
            .as_ref()
            .expect("lane backend is installed")
            .session_capability()
            .admit_query_with_request(request)?;
        let context = handle.acquire_context()?;
        Ok(LaneTestContext { context, _handle: handle, _attempt: None })
    }

    /// Admits `plan` under an attempt identified as `query_id`, so a test's own
    /// observation scope for that query id applies to the admitted context.
    fn admit_as(&self, plan: &QueryPlan, query_id: &str) -> Result<LaneTestContext> {
        admit_query_plan(plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default())?;
        let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
        let mut backend = self.backend.borrow_mut();
        ensure_lane_backend(&mut backend)?;
        let session = backend.as_ref().expect("lane backend is installed").session_capability();
        let attempt = session
            .begin_attempt(BeginAttemptRequest {
                query_id: Some(AttemptQueryId::new(query_id)?),
                ..BeginAttemptRequest::default()
            })?
            .owner;
        attempt.handle().set_disposition(AttemptDisposition::Native).expect("a new attempt accepts a disposition");
        let request =
            QueryAdmissionRequest::new(attempt.admission_binding()).with_query_plan_capability_evidence(evidence);
        let handle = session.admit_query_with_request(request)?;
        let context = handle.acquire_context()?;
        Ok(LaneTestContext { context, _handle: handle, _attempt: Some(attempt) })
    }
}

struct LaneTestContext {
    context: AdmittedQueryContext,
    _handle: QueryHandle,
    _attempt: Option<AttemptOwner>,
}

fn int32_values(batch: &RecordBatch, column: usize) -> Vec<Option<i32>> {
    let array = batch.column(column).as_any().downcast_ref::<Int32Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn int64_values(batch: &RecordBatch, column: usize) -> Vec<Option<i64>> {
    let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn write_key_value_parquet(path: &Path, keys: &[i32], values: &[i64]) {
    assert_eq!(keys.len(), values.len());
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys.to_vec())), Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Splits this run's acquired stream leases into lane leases (acquired with
/// no source worker in scope) and nested source-worker leases.
fn count_acquired_stream_leases(events: &[RuntimeTimelineEvent]) -> (usize, usize) {
    let mut lane_leases = 0usize;
    let mut source_worker_leases = 0usize;
    for event in events {
        if let RuntimeObservationPayload::StreamLease(lease) = &event.payload
            && matches!(lease.lifecycle, StreamLeaseLifecycleState::Acquired { .. })
        {
            if lease.scope.source_worker_id.is_some() {
                source_worker_leases += 1;
            } else {
                lane_leases += 1;
            }
        }
    }
    (lane_leases, source_worker_leases)
}

/// Writes one branch as several row groups of `rows_per_row_group` rows each,
/// so chunk planning emits one chunk per row group and each chunk is large
/// enough that generic source batching cannot merge them.
fn write_row_grouped_key_value_parquet(path: &Path, keys: &[i32], values: &[i64], rows_per_row_group: usize) {
    assert_eq!(keys.len(), values.len());
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys.to_vec())), Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_row_group)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn literal_union_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int32, false)]));
    let left_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("left literal batch");
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3])) as ArrayRef])
        .expect("right literal batch");
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![left_batch])));
    let left = builder
        .add_projection(
            left,
            vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
        )
        .unwrap();
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let right = builder
        .add_projection(
            right,
            vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
        )
        .unwrap();
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    builder.finish_with_host_sink(union).unwrap()
}

fn parquet_union_plan(left_path: &Path, right_path: &Path, schema: &Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let left = builder
        .add_projection(
            left,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
            ],
        )
        .unwrap();
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let right = builder
        .add_projection(
            right,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
            ],
        )
        .unwrap();
    let union = builder.add_union(vec![left, right], Arc::clone(schema)).unwrap();
    let limit = builder.add_limit(union, LimitSpec::fetch(10)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

fn admitted_pipeline(plan: &QueryPlan) -> CompiledPipeline {
    let decision = admit(
        plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("lane union uses the compiled admission");
    admission.pipeline().clone()
}

/// A tiny immutable cap fails the query with a typed error, and the next
/// query on the same backend still admits and executes: the failure neither
/// commits partial outputs nor poisons shared device state.
#[test]
fn lane_small_cap_typed_failure_isolates_the_next_query() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-lane-small-cap");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_keys: Vec<i32> = (0..100_000).collect();
    let left_values: Vec<i64> = (0..100_000).map(|key| i64::from(key) * 10).collect();
    let right_keys: Vec<i32> = (100_000..200_000).collect();
    let right_values: Vec<i64> = (100_000..200_000).map(|key| i64::from(key) * 10).collect();
    // Each source chunk must exceed the cap before the trailing limit can reduce it.
    write_row_grouped_key_value_parquet(&left_path, &left_keys, &left_values, left_keys.len());
    write_row_grouped_key_value_parquet(&right_path, &right_keys, &right_values, right_keys.len());
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = parquet_union_plan(&left_path, &right_path, &schema);

    // The capped attempt drops here: its ledger grant releases with the
    // handle, so the next admission proves the failure left nothing held.
    let error = {
        let capped = backend.admit(&plan, Some(65_536)).expect("tiny cap still admits");
        let admitted_plan = admit_query_plan(
            &plan,
            capped.context.context().statistical_aggregate_execution_mode(),
            capped.context.context().planning_policy(),
        )
        .expect("lane union admits");
        match execute_query_plan(capped.context.context(), QueryExecutionRequest::admitted(&admitted_plan))
            .into_parts()
            .0
        {
            Ok(_) => panic!("tiny cap must fail lane execution"),
            Err(error) => error,
        }
    };
    assert!(
        matches!(
            error.code(),
            QueryErrorCode::QueryMemoryLimitExceeded
                | QueryErrorCode::MemoryReservationFailed
                | QueryErrorCode::CudfTemporaryFailure
        ),
        "small-cap failure is typed, got {:?}",
        error.code()
    );

    let healthy = backend.admit(&plan, None).expect("next query still admits after a lane failure");
    let admitted_plan = admit_query_plan(
        &plan,
        healthy.context.context().statistical_aggregate_execution_mode(),
        healthy.context.context().planning_policy(),
    )
    .expect("next query plan admits");
    let output = execute_query_plan(healthy.context.context(), QueryExecutionRequest::admitted(&admitted_plan))
        .into_parts()
        .0
        .expect("next query executes after a lane failure");
    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 10, "the isolated query executes with its limit after a lane failure");
}

/// The serial compiled executor reports typed cancellation when its query epoch is retired.
#[test]
fn serial_compiled_released_attempt_refuses_execution() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let pipeline = admitted_pipeline(&plan);

    let context = {
        let test_context = backend.admit(&plan, None).expect("lane union admits");
        let context: ExecutionContext = test_context.context.context().clone();
        context
    };
    // The handle dropped with the block above; the cloned view outlives it.

    let metrics = ExecutionMetrics::detached("lane-released");
    let metrics_context = metrics.scope_context(&context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(
        &metrics_context,
        &plan,
        PlanningPolicy::default(),
        &pipeline,
        &gpu_inputs,
        &metrics,
    )
    .unwrap();

    match scheduler.run(&mut executor) {
        Ok(_) => panic!("released attempt must not dispatch"),
        Err(error) => {
            assert_eq!(error.code(), QueryErrorCode::Cancelled, "a retired query epoch reports cancellation");
        },
    }
}

/// Cancellation before execution prevents the serial scheduler from running a task.
#[test]
fn serial_compiled_cancelled_attempt_stops_before_any_task_runs() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let pipeline = admitted_pipeline(&plan);

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    context.cancellation().cancel();

    let metrics = ExecutionMetrics::detached("lane-cancel");
    let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(
        &metrics_context,
        &plan,
        PlanningPolicy::default(),
        &pipeline,
        &gpu_inputs,
        &metrics,
    )
    .unwrap();

    let error = match scheduler.run(&mut executor) {
        Ok(_) => panic!("cancelled attempt must not dispatch"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::Cancelled, "cancellation is typed");
}

/// The production permit chain, run exactly as DataFusion runs it: a root
/// query job holds its own root-class pool permit for the whole plan while
/// the lanes underneath it run as auxiliary work, each on one attempt work
/// lease.
///
/// The backend's `S_d` equals the lane width, so the run has exactly enough
/// attempt auxiliary capacity for its lanes and none to spare, and its root
/// class has exactly one slot. A lane that took a root permit, or any work
/// under a lane that took a second attempt auxiliary permit, would fail
/// `ServiceOverloaded` or deadlock instead of completing.
///
/// The stream-lease observation pins the permit shape without timing: every
/// lease this run acquires is a lane lease, which `acquire_attempt_work_lease`
/// records with no source worker in its scope.
#[test]
fn root_query_job_dispatches_lanes_within_the_attempt_auxiliary_envelope() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-resident-root-chain");
    let path = dir.join("input.parquet");
    write_row_grouped_key_value_parquet(
        &path,
        &(0..800).collect::<Vec<i32>>(),
        &(0..800).map(i64::from).collect::<Vec<i64>>(),
        200,
    );
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = row_grouped_count_plan(&path, &schema).clone_with_gpu_resident_sink();
    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let route =
        test_context.context.context().attempt_stream_provider_route().expect("admitted context carries a fork route");
    assert_eq!(route.lane_capacity(), 2, "the fork pool backs exactly the lane width this test pins");

    // `execute_query_plan` records the plan execution into the attempt's
    // own observation, after the root lease its context binding acquired.
    let context: ExecutionContext = test_context.context.context().clone();
    let attempt_observation = context.runtime_observation().clone();
    let run_start = attempt_observation.live_snapshot().expect("observation is live").events().len();

    let admitted_plan =
        admit_query_plan(&plan, context.statistical_aggregate_execution_mode(), context.planning_policy())
            .expect("lane root chain plan admits");
    let executor = context.gpu_executor().expect("admitted context exposes its device executor");
    let observation = context.runtime_observation().clone();
    let runtime = TokioRuntimeBuilder::new_current_thread().enable_all().build().expect("test runtime");
    // Exactly as DataFusion runs it: the root query job holds a root-class
    // permit for the whole plan while the lanes run underneath it.
    let output = runtime
        .block_on(executor.run_observed_scoped(
            GpuExecutionJobKind::QueryExecution,
            observation,
            move |scope| async move {
                let context = context.with_runtime_gpu_stream_scope(scope);
                execute_query_plan(&context, QueryExecutionRequest::admitted(&admitted_plan)).into_parts().0
            },
        ))
        .expect("the root query job runs")
        .expect("the lanes run inside the attempt envelope instead of saturating a job class");

    let QueryOutput::GpuResident(output) = output else {
        panic!("resident root retains its output");
    };
    let batch =
        materialize_gpu_result_to_host(output, &mut QueryMetrics::default()).expect("resident root materializes");
    assert_eq!(int64_values(&batch, 0), vec![Some(800)], "resident chunks count every source row once");

    let snapshot = attempt_observation.live_snapshot().expect("observation remains live");
    let (lane_leases, source_worker_leases) = count_acquired_stream_leases(&snapshot.events()[run_start..]);
    assert_eq!(lane_leases, 2, "one attempt work lease per lane backs the whole run");
    assert_eq!(source_worker_leases, 0, "no lane leases a nested attempt auxiliary permit of its own attempt");
}

/// Cancellation while two resident chunk workers hold completed outputs drains both.
/// No chunk is committed after the cancellation check.
#[test]
fn resident_chunk_cancellation_drains_workers_without_committing() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-resident-chunk-cancel");
    let path = dir.join("input.parquet");
    write_row_grouped_key_value_parquet(
        &path,
        &(0..800).collect::<Vec<i32>>(),
        &(0..800).map(i64::from).collect::<Vec<i64>>(),
        200,
    );
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = row_grouped_count_plan(&path, &schema).clone_with_gpu_resident_sink();
    let pipeline = admitted_pipeline(&plan);

    let test_context = backend.admit_as(&plan, "lane-running-cancel").expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let width = route.lane_capacity();
    assert_eq!(width, 2, "this test gates both lanes of a width-2 run");

    let owner =
        RuntimeObservationOwner::new("lane-running-cancel", None, RuntimeObservationConfig::bounded(512, 512 * 1024));
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner, PlanExecutionId::new("lane-cancel-plan"));
    let scope = lifecycle.execution_scope();
    let observed_context = context.with_runtime_execution_scope(scope.clone()).expect("observation scope applies");

    let metrics = ExecutionMetrics::detached("lane-running-cancel");
    let metrics_context = metrics.scope_context(&observed_context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let entered = Arc::new(std::sync::Barrier::new(width + 1));
    let release = Arc::new(std::sync::Barrier::new(width + 1));
    let gated_chunks = Arc::new(std::sync::atomic::AtomicUsize::new(width));
    let gate_entered = Arc::clone(&entered);
    let gate_release = Arc::clone(&release);
    let support = LaneDispatchSupport::new(
        route,
        metrics_context,
        plan,
        PlanningPolicy::default(),
        pipeline.clone(),
        gpu_inputs,
        metrics.clone(),
    )
    .with_aggregate_chunk_completion_gate(Arc::new(move |_ordinal, _lane, _fork_index| {
        if gated_chunks.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) > 0 {
            gate_entered.wait();
            gate_release.wait();
        }
    }));

    let cancellation = context.cancellation().clone();
    let canceller = std::thread::spawn(move || {
        entered.wait();
        cancellation.cancel();
        release.wait();
    });

    let error = match run_resident_lane_borrow(&scheduler, Some(&scope), &support) {
        Ok(_) => panic!("a cancelled attempt must not commit its lane completions"),
        Err(error) => error,
    };
    canceller.join().expect("the cancelling thread joins");
    assert_eq!(error.code(), QueryErrorCode::Cancelled, "cancellation is typed");
    assert_eq!(metrics.snapshot().source_chunks_committed, 0, "cancelled chunk outputs never commit");
    lifecycle.finish(RuntimeTerminalOutcome::Cancelled);
}

/// A source read issued from an attempt work lease runs inline on that
/// lease's own context and stream instead of leasing a nested worker.
///
/// This is the invariant that keeps the permit chain acyclic: the caller
/// already holds this attempt's auxiliary permit behind its lease, so a
/// nested `acquire_attempt_work_lease` would make a permit holder block on a
/// second permit of the same attempt. At full lane width that nesting
/// deadlocks outright; this test deliberately holds only one of the
/// backend's two attempt auxiliary permits, so a nested lease would still
/// succeed and the assertion below fails on the observation rather than on a
/// timeout.
///
/// The plan is a grouped aggregate, because that is the shape whose chunk
/// execution reaches `ScheduledLocalParquetRead`. The work-lease context is what
/// a lane hands its task executor, so the guard is exercised at the boundary
/// that owns it.
///
/// The stream-lease observation separates the two dispatch shapes without
/// timing: a work lease taken for a nested source worker records that worker
/// in its scope, so an inline read leaves zero source-worker leases while a
/// leased worker per chunk would leave one each.
#[test]
fn source_read_from_an_attempt_work_lease_runs_inline_instead_of_leasing_a_nested_worker() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-lane-inline-source-read");
    let path = dir.join("keys.parquet");
    // One chunk per row group (the backend caps `max_row_groups_per_chunk`
    // at 1), and each row group must exceed the 1 MiB source-batch target on
    // its own or batching would merge them into a single direct read.
    const ROWS_PER_ROW_GROUP: usize = 200_000;
    const ROW_GROUPS: usize = 3;
    const ROWS: usize = ROWS_PER_ROW_GROUP * ROW_GROUPS;
    const GROUPS: i32 = 4;
    let keys: Vec<i32> = (0..ROWS).map(|row| (row as i32) % GROUPS).collect();
    let values: Vec<i64> = (0..ROWS).map(|row| row as i64).collect();
    write_row_grouped_key_value_parquet(&path, &keys, &values, ROWS_PER_ROW_GROUP);

    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        &path,
        Arc::clone(&schema),
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let test_context = backend.admit_as(&plan, "lane-inline-source-read").expect("grouped aggregate admits");
    let route =
        test_context.context.context().attempt_stream_provider_route().expect("admitted context carries a fork route");
    let lane_capacity = route.lane_capacity();
    assert_eq!(lane_capacity, 2, "this backend backs exactly two attempt work leases");

    let owner = RuntimeObservationOwner::new(
        "lane-inline-source-read",
        None,
        RuntimeObservationConfig::bounded(4096, 4 * 1024 * 1024),
    );
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("lane-inline-plan"));
    let context: ExecutionContext = test_context
        .context
        .context()
        .with_runtime_execution_scope(lifecycle.execution_scope())
        .expect("observation scope applies");

    // Hold one lane's work lease, exactly as the dispatcher hands one to a
    // lane job. The attempt's remaining auxiliary permit is left free on
    // purpose: a regression that leased a nested worker would acquire it and
    // be observed, instead of blocking forever.
    let lane_lease = context.acquire_attempt_work_lease(None).expect("lane work lease acquires");
    let lane_context = lane_lease.context().clone();
    assert!(lane_context.is_attempt_work_lease_context(), "a lane context is a work-lease context");

    let admitted_plan =
        admit_query_plan(&plan, lane_context.statistical_aggregate_execution_mode(), lane_context.planning_policy())
            .expect("lane source read plan admits");
    let outcome = execute_query_plan(&lane_context, QueryExecutionRequest::admitted(&admitted_plan));
    let (result, metrics, _observation) = outcome.into_parts();
    let metrics = metrics.snapshot();
    let output = result.expect("a source read from a work lease completes inline instead of deadlocking");
    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut totals: Vec<(Option<i32>, Option<i64>)> =
        int32_values(&batch, 0).into_iter().zip(int64_values(&batch, 1)).collect();
    totals.sort();
    let expected: Vec<(Option<i32>, Option<i64>)> = (0..GROUPS)
        .map(|group| {
            let total = (0..ROWS).filter(|row| (*row as i32) % GROUPS == group).map(|row| row as i64).sum();
            (Some(group), Some(total))
        })
        .collect();
    assert_eq!(totals, expected, "every row of every chunk reaches the aggregate exactly once");

    // Batched chunk work has one source read, so logical row-group coverage
    // comes from the source facts rather than source-worker lifecycle events.
    assert_eq!(metrics.source_chunks_attempted, ROW_GROUPS as u64);
    assert_eq!(metrics.source_chunks_committed, ROW_GROUPS as u64);
    let snapshot = owner.live_snapshot().expect("observation remains live");
    let (_, source_worker_leases) = count_acquired_stream_leases(snapshot.events());
    assert_eq!(
        source_worker_leases, 0,
        "a read issued from a work lease runs inline on that lease instead of leasing a nested worker"
    );
    lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
    drop(lane_lease);
}

/// The reviewer's scenario at the owner level: a source-worker-style lease
/// held live while the dispatcher's lane leases are taken.
///
/// Before every alias claimed a fork for its lifetime, a source worker and a
/// second island's fragment context attached with no claim at all and were
/// served forks dynamically from whatever looked unclaimed; a lane attaching
/// afterwards was then handed a fork whose kernels that live caller had
/// launched moments earlier. Permit counting could not exclude it, because
/// the unregistered holders consumed permits without consuming forks.
///
/// Now every live caller owns one fixed fork: the source-worker lease, each
/// lane lease, and the attempt root all hold pairwise-distinct streams, and
/// none of the lease forks is the root's reserved one. The backend's `S_d` is
/// the lane width, so the source worker and the lanes together exhaust the
/// auxiliary pool and the assertion covers the full width.
#[test]
fn every_live_attempt_caller_holds_a_distinct_fork_including_the_root() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let capacity = route.lane_capacity();
    assert_eq!(capacity, 2, "this backend backs exactly two attempt auxiliary callers");
    assert_eq!(
        context.attempt_claimed_fork_index(),
        None,
        "the attempt root claims no auxiliary fork; it runs on the route's reserved one"
    );

    // A scheduled source read takes its lease exactly this way, naming the
    // worker it runs for. Before this fix that call registered no claim.
    let source_worker_lease = context
        .acquire_attempt_work_lease(Some((
            RuntimeTaskId::new("fork-isolation-source-task"),
            RuntimeWorkerId::new("fork-isolation-source-worker"),
        )))
        .expect("the source-worker lease acquires");
    let lane_lease = context
        .try_acquire_attempt_work_lease()
        .expect("a lane lease acquires without error")
        .expect("the attempt's second auxiliary permit is free");

    let held: Vec<usize> = [source_worker_lease.context(), lane_lease.context()]
        .iter()
        .map(|context| context.attempt_claimed_fork_index().expect("every attempt work lease holds one auxiliary fork"))
        .collect();
    assert_eq!(
        held.iter().collect::<std::collections::BTreeSet<_>>().len(),
        held.len(),
        "a source worker and a lane never share a fork, saw {held:?}"
    );

    let root_fork = route.reserved_root_fork_identity();
    let lease_forks: Vec<usize> = held
        .iter()
        .map(|index| route.auxiliary_fork_identity(*index).expect("a claimed index is inside the auxiliary pool"))
        .collect();
    assert!(
        lease_forks.iter().all(|fork| *fork != root_fork),
        "no lease is served the root's reserved fork, saw {lease_forks:?} against {root_fork}"
    );

    // Releasing one lease frees exactly its fork, and the next caller takes
    // it back without ever colliding with the still-live holder.
    let surviving = held[0];
    drop(lane_lease);
    let replacement = context
        .try_acquire_attempt_work_lease()
        .expect("the freed permit is reusable")
        .expect("the released permit is free again");
    assert_ne!(
        replacement.context().attempt_claimed_fork_index(),
        Some(surviving),
        "a freed index never overlaps the fork the live source-worker lease still holds"
    );

    drop(replacement);
    drop(source_worker_lease);
}

fn union_output_data_form(pipeline: &CompiledPipeline, task_kind: PipelineTaskKind) -> ResidentDataForm {
    let producer = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == task_kind)
        .expect("pipeline contains the asserted task")
        .id();
    let outputs = pipeline
        .resources()
        .iter()
        .filter(|resource| resource.producer() == producer)
        .map(|resource| resource.data_form())
        .collect::<Vec<ResidentDataForm>>();
    assert_eq!(outputs.len(), 1, "the asserted task produces exactly one resource");
    outputs[0]
}

fn literal_union_count_plan() -> QueryPlan {
    use query_planning::expr::{AggregateExpr, AggregateFunction, AggregateSpec};
    let schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int32, false)]));
    let left_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("left literal batch");
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3])) as ArrayRef])
        .expect("right literal batch");
    let count_schema = Arc::new(Schema::new(vec![Field::new("rows", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![left_batch])));
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    let aggregate = builder
        .add_aggregate(
            union,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_key"), "rows")],
            ),
            count_schema,
        )
        .unwrap();
    builder.finish_with_host_sink(aggregate).unwrap()
}

/// A union feeding a streaming aggregate declares frame-stream output and preserves its count.
/// A union feeding a frame-only limit retains the frame output contract.
#[test]
fn union_streaming_output_contract_matches_its_consumer_form() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_count_plan();
    let pipeline = admitted_pipeline(&plan);

    assert_eq!(
        union_output_data_form(&pipeline, PipelineTaskKind::Union),
        ResidentDataForm::FrameStream,
        "a union with one stream-consuming aggregate must not declare a full concat output"
    );

    let limited = {
        let dir = unique_temp_dir("query-engine-union-frame-contract");
        let left_path = dir.join("left.parquet");
        let right_path = dir.join("right.parquet");
        write_key_value_parquet(&left_path, &[1, 2], &[10, 20]);
        write_key_value_parquet(&right_path, &[3, 4], &[30, 40]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int64, true),
        ]));
        parquet_union_plan(&left_path, &right_path, &schema)
    };
    assert_eq!(
        union_output_data_form(&admitted_pipeline(&limited), PipelineTaskKind::Union),
        ResidentDataForm::Frame,
        "a union feeding a frame-only limit keeps the concat output"
    );

    let test_context = backend.admit(&plan, None).expect("lane union aggregate admits");
    let context: &ExecutionContext = test_context.context.context();
    let run = || {
        let metrics = ExecutionMetrics::detached("lane-union-stream");
        let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &metrics,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).expect("streamed union aggregate executes serially");
        assert_eq!(state.live_resource_count(), 0);
        let batch = executor.take_host_output().expect("compiled host sink should materialize");
        int64_values(&batch, 0)
    };
    let expected = vec![Some(3)];
    assert_eq!(run(), expected, "serial baseline counts every branch row once");
}

fn row_grouped_count_plan(path: &Path, schema: &Arc<Schema>) -> QueryPlan {
    use query_planning::expr::{AggregateExpr, AggregateFunction, AggregateSpec};
    let count_schema = Arc::new(Schema::new(vec![Field::new("rows", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder
        .add_aggregate(
            scan,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "rows")],
            ),
            count_schema,
        )
        .unwrap();
    builder.finish_with_host_sink(aggregate).unwrap()
}

/// A streaming union feeds wide branch frames into a narrow aggregate without a concat output.
#[test]
fn wide_union_branches_stream_into_narrow_aggregate_without_concat() {
    use arrow::array::StringArray;
    use query_planning::{
        expr::{AggregateExpr, AggregateFunction, AggregateSpec},
        pipeline::{PipelineTaskKind, ResidentDataForm},
    };

    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("w1", DataType::Int64, true),
        Field::new("w2", DataType::Int64, true),
        Field::new("w3", DataType::Utf8, true),
        Field::new("w4", DataType::Int64, true),
        Field::new("w5", DataType::Int64, true),
        Field::new("w6", DataType::Utf8, true),
        Field::new("w7", DataType::Int64, true),
    ]));
    let branch = |keys: Vec<i32>| {
        let n = keys.len();
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(keys)) as ArrayRef,
                Arc::new(Int64Array::from(vec![1_i64; n])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2_i64; n])) as ArrayRef,
                Arc::new(StringArray::from(vec!["wide-unused-column-payload"; n])) as ArrayRef,
                Arc::new(Int64Array::from(vec![3_i64; n])) as ArrayRef,
                Arc::new(Int64Array::from(vec![4_i64; n])) as ArrayRef,
                Arc::new(StringArray::from(vec!["another-unused-wide-payload"; n])) as ArrayRef,
                Arc::new(Int64Array::from(vec![5_i64; n])) as ArrayRef,
            ],
        )
        .expect("wide literal branch")
    };
    let count_schema = Arc::new(Schema::new(vec![Field::new("rows", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![branch(vec![1, 2])])));
    let right = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![branch(vec![3, 4])])));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    let aggregate = builder
        .add_aggregate(
            union,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "rows")],
            ),
            count_schema,
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();
    let pipeline = admitted_pipeline(&plan);

    assert_eq!(
        union_output_data_form(&pipeline, PipelineTaskKind::Union),
        ResidentDataForm::FrameStream,
        "wide union branches must chain, not concat, for a stream consumer"
    );

    let test_context = backend.admit(&plan, None).expect("wide union aggregate admits");
    let context: &ExecutionContext = test_context.context.context();
    let run = || {
        let metrics = ExecutionMetrics::detached("lane-wide-union-stream");
        let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &metrics,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).expect("streamed union aggregate executes serially");
        assert_eq!(state.live_resource_count(), 0);
        let batch = executor.take_host_output().expect("compiled host sink should materialize");
        int64_values(&batch, 0)
    };
    let expected = vec![Some(4)];
    assert_eq!(run(), expected);
}

/// A failed resident chunk readiness event reports its ordinal and lane before any chunk commits.
/// The next admitted query completes after cleanup.
#[test]
fn chunk_readiness_sync_failure_attributes_head_ordinal_and_commits_nothing() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-chunk-readiness-failure");
    let path = dir.join("input.parquet");
    write_row_grouped_key_value_parquet(
        &path,
        &(0..800).collect::<Vec<i32>>(),
        &(0..800).map(i64::from).collect::<Vec<i64>>(),
        200,
    );
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = row_grouped_count_plan(&path, &schema).clone_with_gpu_resident_sink();
    let pipeline = admitted_pipeline(&plan);

    let error = {
        let test_context = backend.admit(&plan, None).expect("lane aggregate admits");
        let context: ExecutionContext = test_context.context.context().clone();
        let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
        assert_eq!(route.lane_capacity(), 2);

        let metrics = ExecutionMetrics::detached("lane-readiness-failure");
        let metrics_context = metrics.scope_context(&context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context,
            plan.clone(),
            PlanningPolicy::default(),
            pipeline.clone(),
            gpu_inputs,
            metrics.clone(),
        )
        .with_requested_lanes(2);
        let _fault = EventSynchronizeFault::inject();
        let error = match run_resident_lane_borrow(&scheduler, None, &support) {
            Ok(_) => panic!("a failed head readiness must abort the lane run"),
            Err(error) => error,
        };
        assert_eq!(metrics.snapshot().source_chunks_committed, 0, "readiness fails before any chunk commits");
        error
    };
    // The failed attempt's grant releases with its handle, so the next admission proves the
    // failure left nothing held.
    assert_eq!(
        error.operation(),
        "query_engine.compiled.chunk_readiness_sync",
        "late lane failure is attributed at completion, got {}",
        error.operation()
    );
    assert_eq!(error.fact_value("chunk_ordinal"), Some("0"), "the head ordinal fails first");
    assert!(error.fact_value("execution_lane").is_some(), "the failing lane is attached");

    // The failure neither commits partials nor poisons the backend: the next admission runs.
    let healthy = backend.admit(&plan, None).expect("next query still admits after a readiness failure");
    let admitted_plan = admit_query_plan(
        &plan,
        healthy.context.context().statistical_aggregate_execution_mode(),
        healthy.context.context().planning_policy(),
    )
    .expect("next query plan admits");
    let output = execute_query_plan(healthy.context.context(), QueryExecutionRequest::admitted(&admitted_plan))
        .into_parts()
        .0
        .expect("next query executes after a readiness failure");
    let QueryOutput::GpuResident(output) = output else {
        panic!("resident sink retains the healthy output");
    };
    let batch = materialize_gpu_result_to_host(output, &mut QueryMetrics::default())
        .expect("healthy resident output materializes");
    assert_eq!(int64_values(&batch, 0), vec![Some(800)]);
}

/// Resident chunk workers record readiness and reach completion while stream synchronization is blocked.
/// The coordinator completes after the test releases that synchronization hook.
#[test]
fn lane_workers_complete_chunks_without_stream_synchronize() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-lane-no-worker-sync");
    let path = dir.join("input.parquet");
    write_row_grouped_key_value_parquet(
        &path,
        &(0..800).collect::<Vec<i32>>(),
        &(0..800).map(i64::from).collect::<Vec<i64>>(),
        200,
    );
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = row_grouped_count_plan(&path, &schema).clone_with_gpu_resident_sink();
    let pipeline = admitted_pipeline(&plan);

    let test_context = backend.admit(&plan, None).expect("lane aggregate admits");
    let owned_context: ExecutionContext = test_context.context.context().clone();
    let route = owned_context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let arrivals = Arc::new(Mutex::new(Vec::new()));
    let arrivals_in = Arc::clone(&arrivals);
    let support_route = Arc::clone(&route);
    let _block = StreamSynchronizeBlock::engage();
    let worker = std::thread::spawn(move || {
        let metrics = ExecutionMetrics::detached("lane-no-worker-sync");
        let metrics_context = metrics.scope_context(&owned_context).expect("metrics scope applies");
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let support = LaneDispatchSupport::new(
            support_route,
            metrics_context,
            plan.clone(),
            PlanningPolicy::default(),
            pipeline.clone(),
            GpuResidentInputMap::new(),
            metrics,
        )
        .with_requested_lanes(2)
        .with_aggregate_chunk_completion_gate(Arc::new(move |ordinal, lane, fork| {
            arrivals_in.lock().expect("arrival mutex").push((ordinal, lane, fork));
        }));
        let (completed, output) =
            run_resident_lane_borrow(&scheduler, None, &support).expect("resident chunks complete after unblock");
        assert_eq!(completed, pipeline.tasks().len());
        let batch =
            materialize_gpu_result_to_host(output, &mut QueryMetrics::default()).expect("resident output materializes");
        int64_values(&batch, 0)
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        {
            let observed = arrivals.lock().expect("arrival mutex");
            if observed.len() >= 2 {
                break;
            }
        }
        if std::time::Instant::now() > deadline {
            panic!(
                "both lane workers must record readiness while stream synchronize is blocked; \
                 a worker stuck in a stream synchronize means the old host-fence protocol is back"
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(_block);
    let rows = worker.join().expect("lane run thread joins");
    assert_eq!(rows, vec![Some(800)], "every source row counts exactly once");
    let observed = arrivals.lock().expect("arrival mutex");
    assert!(observed.len() >= 2);
    assert_ne!(observed[0].0, observed[1].0, "completions carry distinct chunk ordinals");
}

/// The serial scheduler exposes host output after its sink commits.
/// The attempt retains its grant, and its auxiliary leases remain available.
#[test]
fn host_sink_handoff_never_reports_success_early_or_releases_its_grant() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_count_plan();
    let pipeline = admitted_pipeline(&plan);

    let test_context = backend.admit(&plan, None).expect("lane union aggregate admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let capacity = route.lane_capacity();
    assert_eq!(capacity, 2);

    let metrics = ExecutionMetrics::detached("lane-sink-handoff");
    let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(
        &metrics_context,
        &plan,
        PlanningPolicy::default(),
        &pipeline,
        &gpu_inputs,
        &metrics,
    )
    .unwrap();
    assert!(executor.take_host_output().is_err(), "host output is absent before the sink task commits it");

    let state = scheduler.run(&mut executor).expect("lane run should execute");
    assert_eq!(state.live_resource_count(), 0);
    let batch = executor.take_host_output().expect("host output exists only after the sink commits");
    assert_eq!(int64_values(&batch, 0), vec![Some(3)]);
    drop(executor);
    drop(state);
    // The serial host run leaves the attempt auxiliary capacity available.
    let mut reacquired = Vec::new();
    for _ in 0..capacity {
        reacquired.push(
            context
                .try_acquire_attempt_work_lease()
                .expect("lease reacquire reports no error")
                .expect("a released lane permit is reusable"),
        );
    }
    assert_eq!(reacquired.len(), capacity);
}
