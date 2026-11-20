//! Reads kept ahead of an aggregate on an attempt with one auxiliary stream.
//!
//! Every case runs a grouped `SUM` over a file of one chunk per row group on a backend whose
//! attempts have one auxiliary permit, so the aggregate runs its chunks inline and its read
//! ahead takes that permit.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_planning::{
    admission::AttemptCapabilityEvidence,
    error::{QueryErrorCode, Result},
    execution_contracts::SourceChunkPlanningMode,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr},
    mode::StatisticalAggregateExecutionMode,
    pipeline::AggregateInputHints,
    plan::{NodeId, ProjectionOutput, QueryPlan, QueryPlanBuilder},
    policy::PlanningPolicy,
    source::{LocalParquetSource, SourceNode},
};
use query_runtime::{
    metrics::QueryMetrics,
    observability::{RuntimeObservationPayload, StreamLeaseLifecycleState},
    runtime::{
        AdmittedQueryContext, ExecutionCancellation, ExecutionContext, GpuBackendConfig, GpuBackendCore,
        GpuBackendDeviceConfig, GpuBackendDeviceProfile, GpuMemoryOwnership, QueryAdmissionRequest,
        QueryExecutionConfig, QueryHandle, QueueBounds, SourceSchedulerPolicy,
    },
};

use super::{AGGREGATE_READ_AHEAD_BOUNDARY, SourceReadAhead, read_ahead_chunk_batch};
use crate::exec::{
    ExecutionMetrics, QueryExecutionRequest, admit_query_plan,
    bounded::{
        BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, chunks::source_read_ahead_admitted,
        plan_bounded_aggregate_input_local_parquet_source_stream,
    },
    execute_query_plan,
    output::QueryOutput,
    source::read_fault::{self, ReadFault},
};

/// One chunk per row group. Each row group exceeds the 1 MiB source byte target, so no batch
/// merges two chunks and the aggregate takes the per-chunk stream path.
const ROWS_PER_ROW_GROUP: usize = 200_000;
const ROW_GROUPS: usize = 4;
const ROWS: usize = ROWS_PER_ROW_GROUP * ROW_GROUPS;
const GROUPS: i32 = 4;

fn backend(max_source_reads_in_flight: usize) -> GpuBackendCore {
    backend_with_chunking(max_source_reads_in_flight, 1024 * 1024, 1)
}

fn backend_with_chunking(
    max_source_reads_in_flight: usize,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
) -> GpuBackendCore {
    let policy = QueryExecutionConfig {
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
        source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
        source_scheduler_policy: SourceSchedulerPolicy::new(max_source_reads_in_flight),
        ..QueryExecutionConfig::default()
    };
    let device = GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(0).with_max_active_attempts(1).with_source_work_per_attempt(1),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    GpuBackendCore::new(
        GpuBackendConfig::new(vec![device], policy, QueueBounds::default()).validate().expect("valid backend config"),
    )
    .expect("backend builds")
}

struct Admitted {
    context: AdmittedQueryContext,
    _handle: QueryHandle,
}

fn admit(backend: &GpuBackendCore, plan: &QueryPlan, cancellation: ExecutionCancellation) -> Admitted {
    admit_with_cap(backend, plan, cancellation, None)
}

fn admit_with_cap(
    backend: &GpuBackendCore,
    plan: &QueryPlan,
    cancellation: ExecutionCancellation,
    cap_bytes: Option<u64>,
) -> Admitted {
    let mut request = QueryAdmissionRequest::test_attempt()
        .with_query_plan_capability_evidence(AttemptCapabilityEvidence::for_query_plan(plan));
    if let Some(cap_bytes) = cap_bytes {
        request = request.with_query_cap_bytes(cap_bytes);
    }
    let handle = backend.session_capability().admit_query_with_request(request).expect("aggregate admits");
    let context = handle.acquire_context_with_cancellation(cancellation).expect("root context acquires");
    Admitted { context, _handle: handle }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

fn key_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]))
}

fn write_source(dir: &Path) -> PathBuf {
    write_source_with_row_groups(dir, ROWS_PER_ROW_GROUP)
}

fn write_source_with_row_groups(dir: &Path, rows_per_row_group: usize) -> PathBuf {
    let path = dir.join("keys.parquet");
    let keys = (0..ROWS).map(|row| (row as i32) % GROUPS).collect::<Vec<_>>();
    let values = (0..ROWS).map(|row| row as i64).collect::<Vec<_>>();
    let schema = key_value_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_row_group)).build();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

fn grouped_sum_plan(path: &Path) -> QueryPlan {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            output_schema,
        )
        .unwrap();
    builder.finish_with_host_sink(aggregate).unwrap()
}

/// `grouped_sum_plan` with a projection between the source and the aggregate, the plan shape
/// that may take the partitioned source-adjacent path.
fn projected_grouped_sum_plan(path: &Path) -> QueryPlan {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(stream_source(path)));
    let projection = builder
        .add_projection(
            source,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
            ],
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            projection,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            output_schema,
        )
        .unwrap();
    builder.finish_with_host_sink(aggregate).unwrap()
}

fn expected_totals() -> Vec<(Option<i32>, Option<i64>)> {
    (0..GROUPS)
        .map(|group| {
            let total = (0..ROWS).filter(|row| (*row as i32) % GROUPS == group).map(|row| row as i64).sum();
            (Some(group), Some(total))
        })
        .collect()
}

fn totals(output: QueryOutput) -> Vec<(Option<i32>, Option<i64>)> {
    let QueryOutput::HostArrow(batch) = output else {
        panic!("host sink should materialize one RecordBatch");
    };
    let keys = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let values = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let mut totals = (0..batch.num_rows())
        .map(|row| ((!keys.is_null(row)).then(|| keys.value(row)), (!values.is_null(row)).then(|| values.value(row))))
        .collect::<Vec<_>>();
    totals.sort();
    totals
}

fn run(context: &ExecutionContext, plan: &QueryPlan) -> (Result<QueryOutput>, QueryMetrics) {
    let admitted_plan =
        admit_query_plan(plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default()).unwrap();
    let (result, metrics, _observation) =
        execute_query_plan(context, QueryExecutionRequest::admitted(&admitted_plan)).into_parts();
    (result, metrics.snapshot())
}

/// Source-read stream leases this attempt acquired and released since `from`.
fn source_worker_stream_leases(context: &ExecutionContext, from: usize) -> (usize, usize) {
    let snapshot = context.runtime_observation().live_snapshot().expect("observation is live");
    let mut acquired = 0;
    let mut released = 0;
    for event in &snapshot.events()[from..] {
        if let RuntimeObservationPayload::StreamLease(lease) = &event.payload
            && lease.scope.source_worker_id.is_some()
        {
            match lease.lifecycle {
                StreamLeaseLifecycleState::Acquired { .. } => acquired += 1,
                StreamLeaseLifecycleState::Released => released += 1,
                _ => {},
            }
        }
    }
    (acquired, released)
}

fn planned_stream(context: &ExecutionContext, path: &Path) -> BoundedJoinLeafLocalParquetSourceStream {
    planned_stream_with_read_ahead(context, path, true)
}

fn planned_stream_with_read_ahead(
    context: &ExecutionContext,
    path: &Path,
    read_ahead: bool,
) -> BoundedJoinLeafLocalParquetSourceStream {
    plan_bounded_aggregate_input_local_parquet_source_stream(
        context,
        &stream_source(path),
        NodeId::from_index_for_test(0),
        false,
        AggregateInputHints { adjacent_stages: false, grouped_string_key: false },
        1,
        read_ahead,
        &ExecutionMetrics::detached("read-ahead-planning"),
    )
    .expect("stream plans")
}

fn event_count(context: &ExecutionContext) -> usize {
    context.runtime_observation().live_snapshot().expect("observation is live").events().len()
}

#[test]
fn aggregate_reads_each_chunk_after_the_first_ahead_of_its_partial() {
    let dir = unique_temp_dir("query-engine-read-ahead-aggregate");
    let plan = grouped_sum_plan(&write_source(&dir));
    let backend = backend(2);
    let admitted = admit(&backend, &plan, ExecutionCancellation::new());
    let context = admitted.context.context();
    let events_before = event_count(context);

    let (result, metrics) = run(context, &plan);

    assert_eq!(totals(result.expect("aggregate completes")), expected_totals());
    assert_eq!(metrics.source_chunks_planned, ROW_GROUPS as u64);
    assert_eq!(metrics.source_chunks_committed, ROW_GROUPS as u64);
    assert_eq!(metrics.source_chunks_attempted, ROW_GROUPS as u64, "a read ahead replaces the chunk's own read");
    assert_eq!(metrics.source_read_prefetches_started, (ROW_GROUPS - 1) as u64);
    assert_eq!(metrics.source_read_prefetches_completed, (ROW_GROUPS - 1) as u64);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    let (acquired, released) = source_worker_stream_leases(context, events_before);
    assert_eq!((acquired, released), (ROW_GROUPS - 1, ROW_GROUPS - 1), "every read ahead returns its stream");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn one_source_read_in_flight_keeps_no_read_ahead() {
    let dir = unique_temp_dir("query-engine-read-ahead-disabled");
    let plan = grouped_sum_plan(&write_source(&dir));
    let backend = backend(1);
    let admitted = admit(&backend, &plan, ExecutionCancellation::new());

    let (result, metrics) = run(admitted.context.context(), &plan);

    assert_eq!(totals(result.expect("aggregate completes")), expected_totals());
    assert_eq!(metrics.source_chunks_committed, ROW_GROUPS as u64);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    fs::remove_dir_all(dir).unwrap();
}

/// A read ahead rejected by the query cap is dropped and its chunk read inline, and the
/// aggregate still sees every row once.
#[test]
fn read_ahead_rejected_by_the_query_cap_is_read_inline() {
    let dir = unique_temp_dir("query-engine-read-ahead-memory-failure");
    let plan = grouped_sum_plan(&write_source(&dir));
    let backend = backend(2);
    let admitted = admit(&backend, &plan, ExecutionCancellation::new());
    let context = admitted.context.context();
    let events_before = event_count(context);
    let _fault = read_fault::arm(AGGREGATE_READ_AHEAD_BOUNDARY, ReadFault::QueryCapRejection);

    let (result, metrics) = run(context, &plan);

    assert_eq!(totals(result.expect("a failed read ahead does not fail the aggregate")), expected_totals());
    assert_eq!(metrics.source_chunks_committed, ROW_GROUPS as u64);
    assert_eq!(metrics.source_read_prefetches_started, (ROW_GROUPS - 1) as u64);
    assert_eq!(metrics.source_read_prefetches_completed, (ROW_GROUPS - 2) as u64);
    assert_eq!(metrics.source_read_prefetches_abandoned, 1);
    assert_eq!(metrics.source_chunks_attempted, ROW_GROUPS as u64 + 1, "the dropped chunk is read again");
    let (acquired, released) = source_worker_stream_leases(context, events_before);
    assert_eq!(acquired, released, "the failed read ahead returns its stream");
    fs::remove_dir_all(dir).unwrap();
}

/// With the attempt's only auxiliary permit held elsewhere, no read ahead starts and every
/// chunk is read by the aggregate itself.
#[test]
fn read_ahead_without_a_free_permit_is_skipped() {
    let dir = unique_temp_dir("query-engine-read-ahead-no-permit");
    let plan = grouped_sum_plan(&write_source(&dir));
    let backend = backend(2);
    let admitted = admit(&backend, &plan, ExecutionCancellation::new());
    let context = admitted.context.context();
    let held = context
        .try_acquire_attempt_work_lease(|| None)
        .expect("lease acquisition reports no error")
        .expect("the attempt's auxiliary permit is free");

    let (result, metrics) = run(context, &plan);

    assert_eq!(totals(result.expect("aggregate completes without a read ahead")), expected_totals());
    assert_eq!(metrics.source_chunks_committed, ROW_GROUPS as u64);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, (ROW_GROUPS - 1) as u64);
    drop(held);
    fs::remove_dir_all(dir).unwrap();
}

/// Admission refuses a read ahead whose planned bytes and scratch do not fit the space left
/// under the attempt's allocation limit.
#[test]
fn admission_refuses_a_read_ahead_that_does_not_fit_the_space_left() {
    let dir = unique_temp_dir("query-engine-read-ahead-admission");
    let path = write_source(&dir);
    let plan = grouped_sum_plan(&path);
    let backend = backend(2);
    let admitted = admit(&backend, &plan, ExecutionCancellation::new());
    let context = admitted.context.context();
    let stream = planned_stream(context, &path);
    let usage = context.attempt_allocation_domain_usage().unwrap().expect("admitted attempts report usage");
    let free = usage.allocation_limit_bytes - usage.current_aligned_bytes;
    let batch_with_planned_bytes = |bytes: u64| {
        let mut batch = read_ahead_chunk_batch(&stream.chunks()[1]);
        batch.chunks[0].estimated_decoded_bytes = Some(bytes);
        batch
    };

    assert!(source_read_ahead_admitted(context, &batch_with_planned_bytes(free / 4), None));
    assert!(!source_read_ahead_admitted(context, &batch_with_planned_bytes(free / 2 + 1), None));
    assert!(!source_read_ahead_admitted(context, &batch_with_planned_bytes(1), Some(free)));
    fs::remove_dir_all(dir).unwrap();
}

/// Cancellation that reaches a read ahead fails the query as cancelled, and the read's stream
/// and memory are returned.
#[test]
fn cancelled_read_ahead_returns_its_stream_and_memory() {
    let dir = unique_temp_dir("query-engine-read-ahead-cancelled");
    let plan = grouped_sum_plan(&write_source(&dir));
    let backend = backend(2);
    let cancellation = ExecutionCancellation::new();
    let admitted = admit(&backend, &plan, cancellation.clone());
    let context = admitted.context.context();
    let events_before = event_count(context);
    let usage_before = context.attempt_allocation_domain_usage().unwrap().expect("usage").current_bytes;
    let _fault = read_fault::arm(AGGREGATE_READ_AHEAD_BOUNDARY, ReadFault::Cancel(cancellation));

    let (result, metrics) = run(context, &plan);

    let Err(error) = result else {
        panic!("a cancelled query fails");
    };
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(metrics.source_read_prefetches_started, 1);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0, "a cancelled read ahead is not dropped and re-read");
    let (acquired, released) = source_worker_stream_leases(context, events_before);
    assert_eq!((acquired, released), (1, 1), "the cancelled read ahead returns its stream");
    let usage_after = context.attempt_allocation_domain_usage().unwrap().expect("usage").current_bytes;
    assert_eq!(usage_after, usage_before, "no frame of the cancelled query stays allocated");
    fs::remove_dir_all(dir).unwrap();
}

/// Dropping a read ahead with a read still pending joins and discards that read, so an owner
/// that returns early on any path leaves no worker, stream or permit behind.
#[test]
fn dropping_a_read_ahead_joins_its_pending_read() {
    let dir = unique_temp_dir("query-engine-read-ahead-drop");
    let path = write_source(&dir);
    let plan = grouped_sum_plan(&path);
    let backend = backend(2);
    let admitted = admit(&backend, &plan, ExecutionCancellation::new());
    let context = admitted.context.context();
    let stream = planned_stream(context, &path);
    let metrics = ExecutionMetrics::detached("read-ahead-drop");
    let metrics_context = metrics.scope_context(context).expect("metrics scope applies");
    let execution = BoundedExecutionAttempt::new(&metrics_context, metrics_context.device_budget_bytes(), &metrics);
    let read_ahead = SourceReadAhead::for_stream(&execution, &stream, AGGREGATE_READ_AHEAD_BOUNDARY)
        .expect("a root context with two reads in flight reads ahead");
    read_ahead
        .start(&execution, &stream_source(&path), read_ahead_chunk_batch(&stream.chunks()[1]), 0)
        .expect("the read ahead starts");
    assert_eq!(metrics.snapshot().source_read_prefetches_started, 1);

    drop(read_ahead);

    assert_eq!(metrics.snapshot().source_read_prefetches_abandoned, 1, "the pending read was joined and dropped");
    let lease = context.try_acquire_attempt_work_lease(|| None).expect("lease acquisition reports no error");
    assert!(lease.is_some(), "the read ahead's auxiliary permit is free again");
    fs::remove_dir_all(dir).unwrap();
}

fn stream_source(path: &Path) -> LocalParquetSource {
    LocalParquetSource::new(path, key_value_schema(), vec!["k".into(), "v".into()])
}

/// A grant-bound aggregate scan plans its chunks for one more live chunk when it reads ahead:
/// the target drops from a quarter of the grant to a fifth, and the scan reads more chunks.
#[test]
fn read_ahead_slot_lowers_the_chunk_target_of_a_grant_bound_scan() {
    const CAP_BYTES: u64 = 16 * 1024 * 1024;
    let dir = unique_temp_dir("query-engine-read-ahead-chunk-target");
    let path = write_source_with_row_groups(&dir, ROWS / 16);
    let plan = grouped_sum_plan(&path);
    let backend = backend_with_chunking(2, u64::MAX, 1024);
    let admitted = admit_with_cap(&backend, &plan, ExecutionCancellation::new(), Some(CAP_BYTES));
    let context = admitted.context.context();

    let serial = planned_stream_with_read_ahead(context, &path, false);
    let read_ahead = planned_stream_with_read_ahead(context, &path, true);

    let grant = context.device_budget_bytes();
    assert_eq!(serial.planned_chunk_target_bytes(), Some(grant / 4));
    assert_eq!(read_ahead.planned_chunk_target_bytes(), Some(grant / 5));
    eprintln!(
        "grant-bound scan of 16 row groups under a {grant}-byte grant: {} chunks without read ahead, {} with it",
        serial.chunk_count(),
        read_ahead.chunk_count()
    );
    assert!(read_ahead.chunk_count() > serial.chunk_count());
    fs::remove_dir_all(dir).unwrap();
}

/// Chunks planned for `plan` under a grant of `cap_mib` MiB whose source-byte target binds,
/// with `max_source_reads_in_flight` reads in flight.
fn grant_bound_chunks_planned(plan: &QueryPlan, cap_mib: u64, max_source_reads_in_flight: usize) -> u64 {
    let backend = backend_with_chunking(max_source_reads_in_flight, u64::MAX, 1024);
    let admitted = admit_with_cap(&backend, plan, ExecutionCancellation::new(), Some(cap_mib * 1024 * 1024));
    let (result, metrics) = run(admitted.context.context(), plan);
    assert_eq!(totals(result.expect("aggregate completes")), expected_totals());
    // A process holds one live backend; the next call builds another.
    drop(admitted);
    backend.close_and_wait(Duration::from_secs(30)).expect("backend closes");
    metrics.source_chunks_planned
}

/// The read-ahead slot is planned only for a scan whose aggregate can read ahead: a grouped
/// aggregate behind a projection may take the partitioned path, which reads no chunk ahead, so
/// its chunk count does not change with two reads in flight.
///
/// Row groups of 100,000 rows decode to about 1.2 MB. The grants are picked so the slot would
/// change the count: a 16 MiB grant packs three row groups per chunk at four slots and two at
/// five, and an 18 MiB grant packs two at seven slots (a projection adds three) and one at eight.
#[test]
fn partitioned_shape_aggregate_plans_no_read_ahead_slot() {
    let dir = unique_temp_dir("query-engine-read-ahead-partitioned-shape");
    let path = write_source_with_row_groups(&dir, ROWS / 8);

    let plain = grouped_sum_plan(&path);
    assert!(
        grant_bound_chunks_planned(&plain, 16, 2) > grant_bound_chunks_planned(&plain, 16, 1),
        "a scan that reads ahead plans the slot"
    );
    let projected = projected_grouped_sum_plan(&path);
    assert_eq!(grant_bound_chunks_planned(&projected, 18, 2), grant_bound_chunks_planned(&projected, 18, 1));
    fs::remove_dir_all(dir).unwrap();
}
