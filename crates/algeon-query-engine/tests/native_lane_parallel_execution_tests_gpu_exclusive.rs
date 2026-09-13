//! Lane-parallel execution acceptance: width-2 dispatch, the production
//! root -> lane -> source chain, running-work cancellation, the overlap
//! workload, and failure isolation.
//!
//! This target builds its own process-owned GPU backend with
//! `WholeDeviceExclusive` memory ownership and two fork streams per attempt,
//! so its `_gpu_exclusive` suffix is load-bearing: `.config/nextest.toml`
//! runs this binary alone, and no other GPU test may share the device while
//! it owns the whole thing.

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
use cudf_sys::test_support::{
    cudf_test_event_allow_synchronize, cudf_test_event_fail_synchronize, cudf_test_stream_block_synchronize,
    cudf_test_stream_unblock_synchronize,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_engine::{
    admission::{AttemptCapabilityEvidence, bounded_pipeline_decision},
    error::{QueryErrorCode, Result},
    exec::{
        CompiledGpuTaskExecutor, CompiledPipelineScheduler, ExecutionMetrics, LaneDispatchSupport,
        QueryExecutionRequest, execute_query_plan, is_lane_family_pipeline, output::QueryOutput,
        preflight_bounded_query_plan, run_lane_dispatched,
    },
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, LimitSpec},
    mode::StatisticalAggregateExecutionMode,
    observability::{
        LiveRuntimeObservationSnapshot, PlanExecutionId, RuntimeObservationConfig, RuntimeObservationOwner,
        RuntimeObservationPayload, RuntimePlanExecutionLifecycle, RuntimeTaskId, RuntimeTerminalOutcome,
        RuntimeWorkerId, StreamLeaseLifecycleState, TaskLifecycleState,
    },
    pipeline::{CompiledPipeline, PipelineTaskKind, ResidentDataForm},
    plan::{ProjectionOutput, QueryPlan, QueryPlanBuilder},
    rapids_backend::rapids_analysis_context,
    runtime::{
        AdmittedQueryContext, AttemptWorkLease, DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus,
        GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft, GpuBackendDeviceConfig,
        GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuExecutionJobKind, GpuMemoryOwnership,
        KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryAdmissionRequest, QueryExecutionConfig, QueryHandle,
        QueueBounds, resident::GpuResidentInputMap,
    },
    source::{LiteralBatchSource, LocalParquetSource, SourceNode},
};
use tokio::runtime::Builder as TokioRuntimeBuilder;

struct LaneBackend {
    core: GpuBackendCore,
}

static LANE_BACKEND: Mutex<Option<LaneBackend>> = Mutex::new(None);

fn build_lane_backend() -> Result<GpuBackendCore> {
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let policy = QueryExecutionConfig {
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
        ..QueryExecutionConfig::default()
    };
    let controller = GpuBackendControllerDraft::new(policy, QueueBounds::default());
    // Two forks per attempt back two execution lanes; two blocking workers let
    // both lanes run at once instead of queueing behind one worker.
    let device = GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(0)
            .with_max_active_attempts(1)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(4, 2, 2))
            .with_source_work_per_attempt(2),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller, deployment)).build()
}

fn ensure_lane_backend(backend: &mut Option<LaneBackend>) -> Result<()> {
    if backend.is_none() {
        *backend = Some(LaneBackend { core: build_lane_backend()? });
    }
    Ok(())
}

struct LaneBackendGuard {
    backend: RefCell<MutexGuard<'static, Option<LaneBackend>>>,
}

impl LaneBackendGuard {
    fn acquire() -> Result<Self> {
        let backend = LANE_BACKEND.lock().expect("lane backend lock is not poisoned");
        Ok(Self { backend: RefCell::new(backend) })
    }

    fn admit(&self, plan: &QueryPlan, cap_bytes: Option<u64>) -> Result<LaneTestContext> {
        preflight_bounded_query_plan(plan, StatisticalAggregateExecutionMode::default())?;
        let evidence = AttemptCapabilityEvidence::for_query_plan(plan);
        let mut backend = self.backend.borrow_mut();
        ensure_lane_backend(&mut backend)?;
        let mut request = QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence);
        if let Some(cap) = cap_bytes {
            request = request.with_query_cap_bytes(cap);
        }
        let handle = backend
            .as_ref()
            .expect("lane backend is installed")
            .core
            .session_capability()
            .admit_query_with_request(request)?;
        let context = handle.acquire_context()?;
        Ok(LaneTestContext { context, _handle: handle })
    }
}

struct LaneTestContext {
    context: AdmittedQueryContext,
    _handle: QueryHandle,
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
fn count_acquired_stream_leases(snapshot: &LiveRuntimeObservationSnapshot) -> (usize, usize) {
    let mut lane_leases = 0usize;
    let mut source_worker_leases = 0usize;
    for event in snapshot.events() {
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

fn write_big_key_value_parquet(path: &Path, keys: Vec<i32>, values: Vec<i64>) {
    assert_eq!(keys.len(), values.len());
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
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
    let left = builder.add_projection(
        left,
        vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
    );
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let right = builder.add_projection(
        right,
        vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    builder.finish_with_host_sink(union)
}

fn parquet_union_plan(left_path: &Path, right_path: &Path, schema: &Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let left = builder.add_projection(
        left,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
        ],
    );
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let right = builder.add_projection(
        right,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
        ],
    );
    let union = builder.add_union(vec![left, right], Arc::clone(schema));
    let limit = builder.add_limit(union, LimitSpec::fetch(10), Arc::clone(schema));
    builder.finish_with_host_sink(limit)
}

fn admitted_pipeline(plan: &QueryPlan) -> CompiledPipeline {
    let decision =
        bounded_pipeline_decision(plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
            .expect("bounded admission should not error");
    let admission = decision.admission().expect("lane union uses the compiled admission");
    admission.pipeline().clone()
}

fn sorted_host_values(batch: &RecordBatch) -> Vec<Option<i32>> {
    let mut values = int32_values(batch, 0);
    values.sort();
    values
}

/// Width 2 engages two pool lanes on disjoint fork subsets; width 1 runs the
/// same protocol inline; both agree with the serial scheduler.
#[test]
fn lane_width_2_agrees_with_serial_and_inline_on_union_branches() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    assert_eq!(route.lane_capacity(), 2, "two-fork backend backs two lanes, one fork each");

    let run = |requested_lanes: Option<usize>| {
        let metrics = ExecutionMetrics::detached("lane-width-2");
        let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        match requested_lanes {
            None => {
                let state = scheduler.run(&mut executor).expect("serial union branches should execute");
                assert_eq!(state.live_resource_count(), 0, "serial run releases every resource");
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane-dispatched union branches should execute");
                assert_eq!(state.live_resource_count(), 0, "lane run releases every resource");
            },
        }
        let batch = executor.take_host_output().expect("compiled host sink should materialize");
        sorted_host_values(&batch)
    };

    let expected = vec![Some(1), Some(2), Some(3)];
    assert_eq!(run(None), expected, "serial baseline computes every branch input once");
    assert_eq!(run(Some(1)), expected, "width 1 runs the same lane protocol inline");
    assert_eq!(run(Some(2)), expected, "width 2 computes every branch input exactly once");
}

/// Two materialized parquet branches under union and limit, through the
/// production query path at width 2. This is the Nsight overlap workload.
#[test]
fn lane_width_2_executes_parquet_union_on_production_path() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-lane-width-2-parquet");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_key_value_parquet(&left_path, &[1, 2], &[10, 20]);
    write_key_value_parquet(&right_path, &[3, 4], &[30, 40]);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = parquet_union_plan(&left_path, &right_path, &schema);
    let pipeline = admitted_pipeline(&plan);
    assert!(
        pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::Union),
        "both branches compile into one pipeline behind a union task"
    );
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let outcome = execute_query_plan(test_context.context.context(), QueryExecutionRequest::new(&plan));
    let (result, _metrics, _observation) = outcome.into_parts();
    let output = result.expect("width-2 lane union should execute");
    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut pairs: Vec<(Option<i32>, Option<i64>)> =
        int32_values(&batch, 0).into_iter().zip(int64_values(&batch, 1)).collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![(Some(1), Some(10)), (Some(2), Some(20)), (Some(3), Some(30)), (Some(4), Some(40))],
        "every parquet branch row survives width-2 dispatch exactly once"
    );
}

/// Overlap workload for the Nsight trace: two half-million-row parquet
/// branches under union with no limit, so decode, projection, and the full
/// concat kernels are long enough to observe overlapping on disjoint fork
/// subsets. CI asserts exact row survival; the trace asserts stream overlap.
#[test]
fn lane_width_2_parquet_overlap_workload() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-lane-overlap");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_keys: Vec<i32> = (0..500_000).collect();
    let left_values: Vec<i64> = (0..500_000).map(|key| i64::from(key) * 10).collect();
    let right_keys: Vec<i32> = (500_000..1_000_000).collect();
    let right_values: Vec<i64> = (500_000..1_000_000).map(|key| i64::from(key) * 10).collect();
    write_big_key_value_parquet(&left_path, left_keys, left_values);
    write_big_key_value_parquet(&right_path, right_keys, right_values);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = parquet_union_plan_no_limit(&left_path, &right_path, &schema);
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    assert_eq!(route.lane_capacity(), 2, "overlap run engages two lanes");
    let outcome = execute_query_plan(context, QueryExecutionRequest::new(&plan));
    let (result, _metrics, _observation) = outcome.into_parts();
    let output = result.expect("width-2 overlap union should execute");
    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1_000_000, "every row of both branches survives width-2 dispatch");
    let mut keys = int32_values(&batch, 0);
    keys.sort();
    assert_eq!(keys.first(), Some(&Some(0)));
    assert_eq!(keys.last(), Some(&Some(999_999)));
}

fn parquet_union_plan_no_limit(left_path: &Path, right_path: &Path, schema: &Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let left = builder.add_projection(
        left,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
        ],
    );
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let right = builder.add_projection(
        right,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
        ],
    );
    let union = builder.add_union(vec![left, right], Arc::clone(schema));
    builder.finish_with_host_sink(union)
}

/// A tiny immutable cap fails the lane run with a typed error, and the next
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
    write_big_key_value_parquet(&left_path, left_keys, left_values);
    write_big_key_value_parquet(&right_path, right_keys, right_values);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = parquet_union_plan(&left_path, &right_path, &schema);
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    // The capped attempt drops here: its ledger grant releases with the
    // handle, so the next admission proves the failure left nothing held.
    let error = {
        let capped = backend.admit(&plan, Some(65_536)).expect("tiny cap still admits");
        match execute_query_plan(capped.context.context(), QueryExecutionRequest::new(&plan)).into_parts().0 {
            Ok(_) => panic!("tiny cap must fail lane execution"),
            Err(error) => error,
        }
    };
    assert!(
        matches!(
            error.code(),
            QueryErrorCode::QueryMemoryLimitExceeded
                | QueryErrorCode::MemoryReservationFailed
                | QueryErrorCode::CudfResourceExhausted
                | QueryErrorCode::CudfTemporaryFailure
        ),
        "small-cap failure is typed, got {:?}",
        error.code()
    );

    let healthy = backend.admit(&plan, None).expect("next query still admits after a lane failure");
    let output = execute_query_plan(healthy.context.context(), QueryExecutionRequest::new(&plan))
        .into_parts()
        .0
        .expect("next query executes after a lane failure");
    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 10, "the isolated query executes with its limit after a lane failure");
}

/// A released attempt refuses lane dispatch: dropping the consumer handle
/// retires the query epoch, which the runtime reports as cancellation, and
/// the dispatcher fails closed before claiming any task instead of running
/// against released admission.
#[test]
fn lane_released_attempt_refuses_dispatch() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let pipeline = admitted_pipeline(&plan);

    let (context, route) = {
        let test_context = backend.admit(&plan, None).expect("lane union admits");
        let context: ExecutionContext = test_context.context.context().clone();
        let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
        (context, route)
    };
    // The handle dropped with the block above; the cloned view outlives it.

    let metrics = ExecutionMetrics::detached("lane-released");
    let metrics_context = metrics.scope_context(&context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
    let support = LaneDispatchSupport::new(
        route,
        metrics_context.clone(),
        plan.clone(),
        pipeline.clone(),
        gpu_inputs.clone(),
        metrics.clone(),
    );
    match run_lane_dispatched(&scheduler, &mut executor, None, &support) {
        Ok(_) => panic!("released attempt must not dispatch"),
        Err(error) => {
            assert_eq!(error.code(), QueryErrorCode::Cancelled, "a retired query epoch reports cancellation");
        },
    }
}

/// A cancelled attempt refuses lane dispatch before any task runs.
#[test]
fn lane_cancelled_attempt_stops_dispatch_before_any_task_runs() {
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
    let mut executor = CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let support = LaneDispatchSupport::new(
        route,
        metrics_context.clone(),
        plan.clone(),
        pipeline.clone(),
        gpu_inputs.clone(),
        metrics.clone(),
    );
    let error = match run_lane_dispatched(&scheduler, &mut executor, None, &support) {
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
    let dir = unique_temp_dir("query-engine-lane-root-chain");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_keys: Vec<i32> = vec![1, 2];
    let left_values: Vec<i64> = vec![10, 20];
    let right_keys: Vec<i32> = vec![3, 4];
    let right_values: Vec<i64> = vec![30, 40];
    write_key_value_parquet(&left_path, &left_keys, &left_values);
    write_key_value_parquet(&right_path, &right_keys, &right_values);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = parquet_union_plan(&left_path, &right_path, &schema);
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let route =
        test_context.context.context().attempt_stream_provider_route().expect("admitted context carries a fork route");
    assert_eq!(route.lane_capacity(), 2, "the fork pool backs exactly the lane width this test pins");

    let owner = RuntimeObservationOwner::new(
        "lane-root-chain-inline",
        None,
        RuntimeObservationConfig::bounded(4096, 4 * 1024 * 1024),
    );
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("lane-root-chain"));
    let context: ExecutionContext = test_context
        .context
        .context()
        .with_runtime_execution_scope(lifecycle.execution_scope())
        .expect("observation scope applies");

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
                execute_query_plan(&context, QueryExecutionRequest::new(&plan)).into_parts().0
            },
        ))
        .expect("the root query job runs")
        .expect("the lanes run inside the attempt envelope instead of saturating a job class");

    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut pairs: Vec<(Option<i32>, Option<i64>)> =
        int32_values(&batch, 0).into_iter().zip(int64_values(&batch, 1)).collect();
    pairs.sort();
    let expected: Vec<(Option<i32>, Option<i64>)> =
        left_keys.iter().chain(right_keys.iter()).map(|key| (Some(*key), Some(i64::from(*key) * 10))).collect();
    assert_eq!(pairs, expected, "every parquet branch row survives the root -> lane chain exactly once");

    let snapshot = owner.live_snapshot().expect("observation remains live");
    let (lane_leases, source_worker_leases) = count_acquired_stream_leases(&snapshot);
    assert_eq!(lane_leases, 2, "one attempt work lease per lane backs the whole run");
    assert_eq!(source_worker_leases, 0, "no lane leases a nested attempt auxiliary permit of its own attempt");
    lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
}

/// Acquires an attempt work lease holding auxiliary fork `fork_index` of
/// `context`'s attempt pool.
///
/// The route, not the caller, picks a claim's index: it takes the lowest
/// unclaimed one. A test that needs a specific fork held therefore claims
/// every index up to it and releases the ones it does not want, driving the
/// choice through acquisition order rather than naming an index.
fn hold_lane_fork(context: &ExecutionContext, fork_index: usize) -> AttemptWorkLease {
    let mut leases: Vec<_> = (0..=fork_index)
        .map(|_| context.acquire_attempt_work_lease(None).expect("the other holder takes its permit"))
        .collect();
    let held = leases.pop().expect("the range is non-empty");
    assert_eq!(
        held.context().attempt_claimed_fork_index(),
        Some(fork_index),
        "claims take the lowest free index in order, so the last of {} holds fork {fork_index}",
        fork_index + 1
    );
    drop(leases);
    held
}

/// Lanes never wait for an attempt auxiliary permit. Another holder of this
/// attempt's auxiliary envelope -- a second native island's fragment context,
/// say -- may itself be waiting on work the dispatching root job owns, so a
/// lane that blocked for a permit would deadlock the attempt while holding its
/// root-class permit. With one of the two permits already taken, the run
/// narrows to the single lease it can take without waiting and completes
/// inline instead.
///
/// Which fork the other holder ended up with must not matter: the dispatcher
/// asks the route for an exclusive fork rather than naming one, so it is
/// served whichever index is still free. A dispatcher that hardcoded lane
/// `i`'s index would fail permanently when the holder occupies fork 0.
#[test]
fn lane_dispatch_narrows_instead_of_waiting_for_a_held_attempt_auxiliary_permit() {
    for held_fork in [0usize, 1] {
        lane_dispatch_narrows_with_held_fork(held_fork);
    }
}

fn lane_dispatch_narrows_with_held_fork(held_fork: usize) {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let dir = unique_temp_dir("query-engine-lane-permit-contention");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let left_keys: Vec<i32> = vec![1, 2];
    let left_values: Vec<i64> = vec![10, 20];
    let right_keys: Vec<i32> = vec![3, 4];
    let right_values: Vec<i64> = vec![30, 40];
    write_key_value_parquet(&left_path, &left_keys, &left_values);
    write_key_value_parquet(&right_path, &right_keys, &right_values);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let plan = parquet_union_plan(&left_path, &right_path, &schema);
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let route =
        test_context.context.context().attempt_stream_provider_route().expect("admitted context carries a fork route");
    assert_eq!(route.lane_capacity(), 2, "the fork pool backs exactly the lane width this test pins");

    let owner = RuntimeObservationOwner::new(
        "lane-permit-contention",
        None,
        RuntimeObservationConfig::bounded(4096, 4 * 1024 * 1024),
    );
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner, PlanExecutionId::new("lane-permit-contention"));
    let context: ExecutionContext = test_context
        .context
        .context()
        .with_runtime_execution_scope(lifecycle.execution_scope())
        .expect("observation scope applies");

    // Another auxiliary holder of this attempt owns one of the two permits
    // for the whole run, leaving the dispatcher one -- and one fork, which is
    // whichever of the two the holder did not take.
    let held = hold_lane_fork(&context, held_fork);

    let executor = context.gpu_executor().expect("admitted context exposes its device executor");
    let observation = context.runtime_observation().clone();
    let dispatch_context = context;
    let dispatch_plan = plan;
    let runtime = TokioRuntimeBuilder::new_current_thread().enable_all().build().expect("test runtime");
    let output = runtime
        .block_on(executor.run_observed_scoped(
            GpuExecutionJobKind::QueryExecution,
            observation,
            move |scope| async move {
                let context = dispatch_context.with_runtime_gpu_stream_scope(scope);
                execute_query_plan(&context, QueryExecutionRequest::new(&dispatch_plan)).into_parts().0
            },
        ))
        .expect("the root query job runs")
        .unwrap_or_else(|error| {
            panic!("the run must narrow onto the free fork while fork {held_fork} is held, got {error}")
        });

    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut pairs: Vec<(Option<i32>, Option<i64>)> =
        int32_values(&batch, 0).into_iter().zip(int64_values(&batch, 1)).collect();
    pairs.sort();
    let expected: Vec<(Option<i32>, Option<i64>)> =
        left_keys.iter().chain(right_keys.iter()).map(|key| (Some(*key), Some(i64::from(*key) * 10))).collect();
    assert_eq!(pairs, expected, "a narrowed run still produces every row exactly once");

    drop(held);
    lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
}

/// The single lease a narrowed run obtains before falling back to the inline
/// arm must not stay held for that inline run: it is unusable at width 1 (the
/// inline arm runs on the coordinator's own context and stream instead), so
/// it has to be released before the first inline task runs, not just by the
/// time the whole dispatch returns. A regression that only dropped it at the
/// end of `run_lane_dispatched` (for example by shadowing the lease vector
/// without actually releasing the old one) would keep the lane's attempt
/// auxiliary permit, service stream, and fork claim charged for the entire
/// inline run, so a probe from inside a lane-completion callback -- which
/// fires while `run_lane_dispatched` is still on the stack -- would find that
/// permit still unavailable.
#[test]
fn lane_dispatch_releases_a_single_narrowed_lease_before_the_inline_run() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    assert_eq!(route.lane_capacity(), 2, "the fork pool backs exactly the lane width this test pins");

    // Another auxiliary holder of this attempt owns one of the two permits for
    // the whole run, exactly as in the narrowing test above: the dispatcher
    // can only ever obtain one lease, so the run narrows to width 1 and falls
    // back to the inline arm.
    let held = hold_lane_fork(context, 1);

    let owner =
        RuntimeObservationOwner::new("lane-inline-release", None, RuntimeObservationConfig::bounded(512, 512 * 1024));
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner, PlanExecutionId::new("lane-inline-release-plan"));
    let scope = lifecycle.execution_scope();
    let observed_context = context.with_runtime_execution_scope(scope.clone()).expect("observation scope applies");

    let metrics = ExecutionMetrics::detached("lane-inline-release");
    let metrics_context = metrics.scope_context(&observed_context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();

    let probed_context = context.clone();
    let observed_available = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let gate_available = Arc::clone(&observed_available);
    let support = LaneDispatchSupport::new(
        route,
        metrics_context.clone(),
        plan.clone(),
        pipeline.clone(),
        gpu_inputs.clone(),
        metrics.clone(),
    )
    .with_requested_lanes(2)
    .with_lane_completion_gate(Arc::new(move |_lane, _fork_index| {
        // The dispatcher's own narrowed lease claimed the attempt's one free
        // permit and fork. If it was released before this inline task ran,
        // a work lease is obtainable right now; if it is still held for the
        // whole run, no permit and no fork are free and this fails.
        if let Ok(Some(reacquired)) = probed_context.try_acquire_attempt_work_lease() {
            gate_available.store(true, std::sync::atomic::Ordering::Release);
            drop(reacquired);
        }
    }));

    let state = run_lane_dispatched(&scheduler, &mut executor, Some(&scope), &support)
        .expect("the narrowed run completes inline");
    assert_eq!(state.live_resource_count(), 0, "inline run releases every resource");
    let batch = executor.take_host_output().expect("inline run commits its sink output");
    assert_eq!(
        sorted_host_values(&batch),
        vec![Some(1), Some(2), Some(3)],
        "a narrowed inline run still unions every branch"
    );

    assert!(
        observed_available.load(std::sync::atomic::Ordering::Acquire),
        "the dispatcher's single narrowed lease must release its attempt auxiliary permit, \
         service stream, and fork claim before the inline run starts, not only after the whole dispatch returns"
    );

    drop(held);
    lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
}

/// With the whole auxiliary envelope free, both lanes obtain a lease and the
/// route serves each one a fork of its own: the run stays at width 2 and no
/// two lanes share a stream. The route picks the indices, so the test asserts
/// distinctness and pool membership rather than a fixed lane-to-fork mapping.
#[test]
fn a_full_width_run_serves_every_lane_a_distinct_fork() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let capacity = route.lane_capacity();
    assert_eq!(capacity, 2, "the fork pool backs exactly the lane width this test pins");

    let owner =
        RuntimeObservationOwner::new("lane-full-width", None, RuntimeObservationConfig::bounded(512, 512 * 1024));
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner, PlanExecutionId::new("lane-full-width-plan"));
    let scope = lifecycle.execution_scope();
    let observed_context = context.with_runtime_execution_scope(scope.clone()).expect("observation scope applies");

    let metrics = ExecutionMetrics::detached("lane-full-width");
    let metrics_context = metrics.scope_context(&observed_context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();

    let observed_forks = Arc::new(std::sync::Mutex::new(Vec::<(usize, Option<usize>)>::new()));
    let gate_forks = Arc::clone(&observed_forks);
    let support = LaneDispatchSupport::new(
        route,
        metrics_context.clone(),
        plan.clone(),
        pipeline.clone(),
        gpu_inputs.clone(),
        metrics.clone(),
    )
    .with_requested_lanes(capacity)
    .with_lane_completion_gate(Arc::new(move |lane, fork_index| {
        gate_forks.lock().expect("the fork record is uncontended by panics").push((lane, fork_index));
    }));

    let state =
        run_lane_dispatched(&scheduler, &mut executor, Some(&scope), &support).expect("the full-width run completes");
    assert_eq!(state.live_resource_count(), 0, "a full-width run releases every resource");
    let batch = executor.take_host_output().expect("the run commits its sink output");
    assert_eq!(sorted_host_values(&batch), vec![Some(1), Some(2), Some(3)], "a full-width run unions every branch");

    let records = observed_forks.lock().expect("the fork record is uncontended by panics").clone();
    let lanes: std::collections::BTreeSet<usize> = records.iter().map(|(lane, _)| *lane).collect();
    assert_eq!(lanes.len(), capacity, "both lanes of the width-2 run execute, saw {records:?}");
    let forks: std::collections::BTreeSet<usize> = records
        .iter()
        .map(|(lane, fork)| fork.unwrap_or_else(|| panic!("lane {lane} runs on a fork-claiming alias")))
        .collect();
    assert_eq!(forks.len(), capacity, "each lane holds a fork of its own, saw {records:?}");
    assert!(forks.iter().all(|fork| *fork < capacity), "every claimed fork is inside the pool, saw {records:?}");

    lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
}

/// Cancelling an attempt while its lane workers hold completed outputs must
/// not commit them. Both lanes of the first round are gated after their task
/// succeeded, the attempt is cancelled, the gate is released, and the
/// dispatcher drains both workers. Because the coordinator re-checks
/// cancellation after the joins, neither drained task is ever marked
/// finished: both outputs are released through the same accounting a failed
/// round uses, and the run fails with the cancellation error.
#[test]
fn lane_cancellation_while_workers_run_drains_without_committing() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_plan();
    let pipeline = admitted_pipeline(&plan);

    let test_context = backend.admit(&plan, None).expect("lane union admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let width = route.lane_capacity();
    assert_eq!(width, 2, "this test gates both lanes of a width-2 run");

    let owner =
        RuntimeObservationOwner::new("lane-running-cancel", None, RuntimeObservationConfig::bounded(512, 512 * 1024));
    let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("lane-cancel-plan"));
    let scope = lifecycle.execution_scope();
    let observed_context = context.with_runtime_execution_scope(scope.clone()).expect("observation scope applies");

    let metrics = ExecutionMetrics::detached("lane-running-cancel");
    let metrics_context = metrics.scope_context(&observed_context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();

    // Both lane workers of the first round park on the barrier after their
    // task has produced its output, so the attempt is provably cancelled
    // while two completed but uncommitted outputs are in flight; the release
    // barrier then lets the workers return them to be drained. Only the first
    // round is gated, so the cancellation the coordinator must observe is the
    // one raised after those joins.
    let entered = Arc::new(std::sync::Barrier::new(width + 1));
    let release = Arc::new(std::sync::Barrier::new(width + 1));
    let gate_rounds = Arc::new(std::sync::atomic::AtomicUsize::new(width));
    let gate_entered = Arc::clone(&entered);
    let gate_release = Arc::clone(&release);
    let support = LaneDispatchSupport::new(
        route,
        metrics_context.clone(),
        plan.clone(),
        pipeline.clone(),
        gpu_inputs.clone(),
        metrics.clone(),
    )
    .with_lane_completion_gate(Arc::new(move |_lane, _fork_index| {
        if gate_rounds.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) > 0 {
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

    let error = match run_lane_dispatched(&scheduler, &mut executor, Some(&scope), &support) {
        Ok(_) => panic!("a cancelled attempt must not commit its lane completions"),
        Err(error) => error,
    };
    canceller.join().expect("the cancelling thread joins");
    assert_eq!(error.code(), QueryErrorCode::Cancelled, "cancellation is typed");
    assert!(executor.take_host_output().is_err(), "no sink output is committed on a cancelled run");

    let finished = owner
        .live_snapshot()
        .expect("observation remains live")
        .events()
        .iter()
        .filter(|event| {
            matches!(
                &event.payload,
                RuntimeObservationPayload::TaskLifecycle(task) if task.lifecycle == TaskLifecycleState::Finished
            )
        })
        .count();
    assert_eq!(finished, 0, "a coordinator that re-checks cancellation after draining commits no lane completion");
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
/// execution reaches `ScheduledLocalParquetRead`; the lane family's own task
/// kinds read their chunks directly. The work-lease context is exactly what
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
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let test_context = backend.admit(&plan, None).expect("grouped aggregate admits");
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

    let outcome = execute_query_plan(&lane_context, QueryExecutionRequest::new(&plan));
    let (result, metrics, _observation) = outcome.into_parts();
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
    let (_, source_worker_leases) = count_acquired_stream_leases(&snapshot);
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

/// RAII restore for CUDA test-support fault hooks: the fault clears even when the test
/// panics, so the serial exclusive binary never leaks a global hook into the next test.
struct EventSyncFaultGuard;

impl EventSyncFaultGuard {
    fn fail() -> Self {
        unsafe { cudf_test_event_fail_synchronize() };
        Self
    }
}

impl Drop for EventSyncFaultGuard {
    fn drop(&mut self) {
        unsafe { cudf_test_event_allow_synchronize() };
    }
}

struct StreamSyncBlockGuard;

impl StreamSyncBlockGuard {
    fn block() -> Self {
        unsafe { cudf_test_stream_block_synchronize() };
        Self
    }
}

impl Drop for StreamSyncBlockGuard {
    fn drop(&mut self) {
        unsafe { cudf_test_stream_unblock_synchronize() };
    }
}

fn union_output_data_form(pipeline: &CompiledPipeline, task_kind: PipelineTaskKind) -> ResidentDataForm {
    use ResidentDataForm;
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
    use query_engine::expr::{AggregateExpr, AggregateFunction, AggregateSpec};
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
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let aggregate = builder.add_aggregate(
        union,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_key"), "rows")],
        ),
        count_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

/// A union whose consumer is a streaming aggregate declares a frame-stream output, while a
/// union whose consumer is a frame-only limit keeps the concatenated frame; both contract
/// assertions fail when the union declaration is reverted. The streamed form is then executed
/// serially, inline, and with two lanes, and every run must produce the same count and release
/// every resource. The test observes planning contracts and result parity; it does not measure
/// allocations.
#[test]
fn union_streaming_output_contract_matches_its_consumer_form() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_count_plan();
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));
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
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let run = |requested_lanes: Option<usize>| {
        let metrics = ExecutionMetrics::detached("lane-union-stream");
        let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        match requested_lanes {
            None => {
                let state = scheduler.run(&mut executor).expect("serial union aggregate should execute");
                assert_eq!(state.live_resource_count(), 0, "serial run releases every resource");
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane-dispatched union aggregate should execute");
                assert_eq!(state.live_resource_count(), 0, "lane run releases every resource");
            },
        }
        let batch = executor.take_host_output().expect("compiled host sink should materialize");
        int64_values(&batch, 0)
    };
    let expected = vec![Some(3)];
    assert_eq!(run(None), expected, "serial baseline counts every branch row once");
    assert_eq!(run(Some(1)), expected, "width 1 streams through the same protocol");
    assert_eq!(run(Some(2)), expected, "width 2 agrees with the serial baseline");
}

fn row_grouped_count_plan(path: &Path, schema: &Arc<Schema>) -> QueryPlan {
    use query_engine::expr::{AggregateExpr, AggregateFunction, AggregateSpec};
    let count_schema = Arc::new(Schema::new(vec![Field::new("rows", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        scan,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "rows")]),
        count_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

/// E5: a union of wide branches feeding a narrow aggregate streams the branch frames without
/// a full-width concat copy. Unused wide columns ride the stream payloads and are dropped by
/// the downstream partial aggregate per frame; no second materialization is created and every
/// row counts exactly once at every width.
#[test]
fn wide_union_branches_stream_into_narrow_aggregate_without_concat() {
    use arrow::array::StringArray;
    use query_engine::{
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
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let aggregate = builder.add_aggregate(
        union,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("key"), "rows")]),
        count_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));
    assert_eq!(
        union_output_data_form(&pipeline, PipelineTaskKind::Union),
        ResidentDataForm::FrameStream,
        "wide union branches must chain, not concat, for a stream consumer"
    );

    let test_context = backend.admit(&plan, None).expect("wide union aggregate admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let run = |requested_lanes: Option<usize>| {
        let metrics = ExecutionMetrics::detached("lane-wide-union-stream");
        let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        match requested_lanes {
            None => {
                scheduler.run(&mut executor).expect("serial wide union aggregate should execute");
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane wide union aggregate should execute");
            },
        }
        let batch = executor.take_host_output().expect("compiled host sink should materialize");
        int64_values(&batch, 0)
    };
    let expected = vec![Some(4)];
    assert_eq!(run(None), expected);
    assert_eq!(run(Some(1)), expected);
    assert_eq!(run(Some(2)), expected);
}

/// E1: failing the head chunk's readiness event aborts the run before any commit, with the
/// chunk ordinal and lane attached. No partial enters coordinator state and no host output
/// is produced.
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
    let plan = row_grouped_count_plan(&path, &schema);
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let error = {
        let test_context = backend.admit(&plan, None).expect("lane aggregate admits");
        let context: ExecutionContext = test_context.context.context().clone();
        let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
        assert_eq!(route.lane_capacity(), 2);

        let metrics = ExecutionMetrics::detached("lane-readiness-failure");
        let metrics_context = metrics.scope_context(&context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context.clone(),
            plan.clone(),
            pipeline.clone(),
            gpu_inputs.clone(),
            metrics.clone(),
        )
        .with_requested_lanes(2);
        let _fault = EventSyncFaultGuard::fail();
        let error = match run_lane_dispatched(&scheduler, &mut executor, None, &support) {
            Ok(_) => panic!("a failed head readiness must abort the lane run"),
            Err(error) => error,
        };
        assert!(executor.take_host_output().is_err(), "no host output is produced when readiness fails before commit");
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
    let output = execute_query_plan(healthy.context.context(), QueryExecutionRequest::new(&plan))
        .into_parts()
        .0
        .expect("next query executes after a readiness failure");
    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int64_values(&batch, 0), vec![Some(800)]);
}

/// E1: lane workers finish chunk work without host-synchronizing their streams. With every
/// stream synchronize blocked, both lane workers still reach completion (they only record
/// readiness); the coordinator's device-side commit and the terminal D2H proceed after
/// unblock. On the old worker-sync protocol neither worker could complete while blocked.
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
    let plan = row_grouped_count_plan(&path, &schema);
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane aggregate admits");
    let owned_context: ExecutionContext = test_context.context.context().clone();
    let route = owned_context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let arrivals = Arc::new(Mutex::new(Vec::new()));
    let arrivals_in = Arc::clone(&arrivals);
    let support_route = Arc::clone(&route);
    let _block = StreamSyncBlockGuard::block();
    let worker = std::thread::spawn(move || {
        let metrics = ExecutionMetrics::detached("lane-no-worker-sync");
        let metrics_context = metrics.scope_context(&owned_context).expect("metrics scope applies");
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let gpu_inputs = GpuResidentInputMap::new();
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        let support = LaneDispatchSupport::new(
            support_route,
            metrics_context.clone(),
            plan.clone(),
            pipeline.clone(),
            GpuResidentInputMap::new(),
            metrics.clone(),
        )
        .with_requested_lanes(2)
        .with_aggregate_chunk_completion_gate(Arc::new(move |ordinal, lane, fork| {
            arrivals_in.lock().expect("arrival mutex").push((ordinal, lane, fork));
        }));
        let state =
            run_lane_dispatched(&scheduler, &mut executor, None, &support).expect("lane run completes after unblock");
        let batch = executor.take_host_output().expect("host sink materializes");
        (state.live_resource_count(), int64_values(&batch, 0))
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
    let (live_resources, rows) = worker.join().expect("lane run thread joins");
    assert_eq!(live_resources, 0, "lane run releases every resource");
    assert_eq!(rows, vec![Some(800)], "every source row counts exactly once");
    let observed = arrivals.lock().expect("arrival mutex");
    assert!(observed.len() >= 2);
    assert_ne!(observed[0].0, observed[1].0, "completions carry distinct chunk ordinals");
}

/// E3: the host sink never reports success early and never releases its grant before the
/// device-to-host copy finishes. This is a prohibition pin, not an overlap proof: host output
/// is absent until the sink task commits it, and the attempt's full lane capacity is reusable
/// only after the run returns its leases, so a future change that emits rows early or drops
/// the grant at sink entry fails here.
#[test]
fn host_sink_handoff_never_reports_success_early_or_releases_its_grant() {
    let backend = LaneBackendGuard::acquire().expect("lane backend lock acquires");
    let plan = literal_union_count_plan();
    let pipeline = admitted_pipeline(&plan);
    assert!(is_lane_family_pipeline(&pipeline));

    let test_context = backend.admit(&plan, None).expect("lane union aggregate admits");
    let context: &ExecutionContext = test_context.context.context();
    let route = context.attempt_stream_provider_route().expect("admitted context carries a fork route");
    let capacity = route.lane_capacity();
    assert_eq!(capacity, 2);

    let metrics = ExecutionMetrics::detached("lane-sink-handoff");
    let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
    assert!(executor.take_host_output().is_err(), "host output is absent before the sink task commits it");
    let support = LaneDispatchSupport::new(
        Arc::clone(&route),
        metrics_context.clone(),
        plan.clone(),
        pipeline.clone(),
        gpu_inputs.clone(),
        metrics.clone(),
    )
    .with_requested_lanes(2);
    let state = run_lane_dispatched(&scheduler, &mut executor, None, &support).expect("lane run should execute");
    assert_eq!(state.live_resource_count(), 0);
    let batch = executor.take_host_output().expect("host output exists only after the sink commits");
    assert_eq!(int64_values(&batch, 0), vec![Some(3)]);
    drop(executor);
    drop(state);
    // Every lane lease returned with the run: the attempt serves its full capacity again.
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
