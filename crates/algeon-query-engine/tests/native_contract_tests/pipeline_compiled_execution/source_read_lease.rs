use std::{
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use ::cudf::Runtime as CudfRuntime;
use query_engine::{
    error::QueryErrorCode,
    observability::{
        DeterministicRuntimeObservationClock, RuntimeObservationOwner, RuntimeObservationPayload,
        RuntimeObservationSession, RuntimeTimelineEvent, SealedRuntimeObservationSnapshot, SourceReadLifecycleState,
        StreamLeaseLifecycleState,
    },
    runtime::{
        ExecutionCancellation, MemoryControlPolicy, RuntimeObservedExecutionContext, ServiceStreamPool,
        StandaloneConfig,
    },
};

use super::*;

type ExecutionParts = (Result<QueryOutput>, SealedRuntimeObservationSnapshot);

/// A hash join over two local Parquet leaves, executing on a standalone context
/// whose one-stream service pool the test has exhausted. The join's first source
/// read is blocked in its stream-lease wait once `start` returns.
struct JoinBehindStreamLease {
    owner: RuntimeObservationOwner,
    execution: JoinHandle<ExecutionParts>,
}

impl JoinBehindStreamLease {
    fn start(
        runtime: Arc<CudfRuntime>,
        pool: &ServiceStreamPool,
        cancellation: ExecutionCancellation,
        session: RuntimeObservationSession,
    ) -> Self {
        let dir = unique_temp_dir("query-engine-source-read-lease");
        let left_path = dir.join("left.parquet");
        let right_path = dir.join("right.parquet");
        write_join_parquets(&left_path, &right_path);
        let plan = root_join_data_plan(&left_path, &right_path);
        let observed = RuntimeObservedExecutionContext::standalone(
            StandaloneConfig {
                runtime: Some(runtime),
                service_stream_pool: Some(pool.clone()),
                memory_control_policy: MemoryControlPolicy {
                    device_budget_bytes: 256 * 1024 * 1024,
                    // A byte target below any two row groups keeps each row group its own
                    // read batch, so the multi-chunk leaves read through the source scheduler.
                    max_source_chunk_bytes: 1,
                    max_row_groups_per_chunk: 1,
                    max_retry_attempts: 1,
                    min_retry_chunk_row_groups: 1,
                },
                cancellation,
                ..StandaloneConfig::default()
            },
            session,
        )
        .expect("standalone observed context");
        let owner = observed.observation().clone();
        let admitted = admit_query_plan(
            &plan,
            observed.context().statistical_aggregate_execution_mode(),
            observed.context().planning_policy(),
        )
        .expect("join plan admits");
        let execution = thread::spawn(move || {
            let (result, _metrics, snapshot) =
                execute_engine_query_plan(observed.context(), QueryExecutionRequest::admitted(&admitted)).into_parts();
            (result, snapshot.expect("a standalone observation owner seals its snapshot into the outcome"))
        });
        pool.wait_until_acquire_queued_for_test(Instant::now() + Duration::from_secs(30));
        Self { owner, execution }
    }

    fn finish(self) -> ExecutionParts {
        self.execution.join().expect("join execution thread")
    }
}

fn source_read_states(events: &[RuntimeTimelineEvent]) -> Vec<SourceReadLifecycleState> {
    events
        .iter()
        .filter_map(|event| match &event.payload {
            RuntimeObservationPayload::SourceRead(read) => Some(read.lifecycle.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn source_read_lease_wait_precedes_enqueue_and_is_not_queue_time() {
    let runtime = Arc::new(CudfRuntime::new().expect("source read lease test requires a CUDA runtime"));
    let pool = ServiceStreamPool::new(runtime.device(), 1).unwrap();
    let held = pool.acquire().unwrap();
    let clock = Arc::new(DeterministicRuntimeObservationClock::new(None));
    let session = RuntimeObservationSession::with_clock(
        "source-read-lease-clock",
        None,
        RuntimeObservationConfig::native_execution(),
        Arc::clone(&clock) as _,
    );
    let join = JoinBehindStreamLease::start(runtime, &pool, ExecutionCancellation::new(), session);

    let waiting = join.owner.live_snapshot().unwrap();
    assert!(source_read_states(waiting.events()).is_empty(), "lease contention is not source worker queue time");
    clock.advance(100);
    drop(held);
    let (result, snapshot) = join.finish();

    let QueryOutput::HostArrow(batch) = result.expect("join executes once the lease is granted") else {
        panic!("host sink materializes one RecordBatch");
    };
    let mut rows = int32_values(&batch, 0).into_iter().zip(int32_values(&batch, 1)).collect::<Vec<_>>();
    rows.sort();
    assert_eq!(rows, vec![(Some(2), Some(200)), (Some(4), Some(400))]);
    let states = source_read_states(snapshot.events());
    assert_eq!(states.first(), Some(&SourceReadLifecycleState::Enqueued));
    let queue_waits = states
        .iter()
        .filter_map(|state| match state {
            SourceReadLifecycleState::WorkerStarted { queue_wait_ns } => Some(*queue_wait_ns),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!queue_waits.is_empty(), "started source reads are retained");
    assert!(queue_waits.iter().all(|wait| *wait == 0), "the 100 ns lease wait is not queue time: {queue_waits:?}");
    let worker_id = snapshot
        .events()
        .iter()
        .find_map(|event| match &event.payload {
            RuntimeObservationPayload::SourceRead(read) => Some(read.worker_id.clone()),
            _ => None,
        })
        .expect("source read events are retained");
    assert!(
        snapshot.events().iter().any(|event| matches!(
            &event.payload,
            RuntimeObservationPayload::StreamLease(stream) if stream.scope.source_worker_id.as_ref() == Some(&worker_id)
        )),
        "the stream lease carries its source worker correlation"
    );
}

#[test]
fn cancelled_source_read_lease_wait_records_no_source_enqueue() {
    let runtime = Arc::new(CudfRuntime::new().expect("source read lease test requires a CUDA runtime"));
    let pool = ServiceStreamPool::new(runtime.device(), 1).unwrap();
    let held = pool.acquire().unwrap();
    let cancellation = ExecutionCancellation::new();
    let session =
        RuntimeObservationSession::new("source-read-lease-cancel", None, RuntimeObservationConfig::native_execution());
    let join = JoinBehindStreamLease::start(runtime, &pool, cancellation.clone(), session);

    cancellation.cancel();
    // The lease stays held until the execution ends so the wait can only end by cancellation.
    let (result, snapshot) = join.finish();
    drop(held);

    let error = result.err().expect("cancelled lease wait fails the execution");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert!(source_read_states(snapshot.events()).is_empty(), "a cancelled lease wait never enqueues source work");
    assert!(snapshot.events().iter().any(|event| matches!(
        &event.payload,
        RuntimeObservationPayload::StreamLease(stream) if stream.lifecycle == StreamLeaseLifecycleState::Cancelled
    )));
}
