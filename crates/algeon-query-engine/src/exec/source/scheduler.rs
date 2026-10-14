//! Scheduled local Parquet chunk reads and their join protocol.
//!
//! `ScheduledLocalParquetRead::spawn` starts one chunk read and the consumer ends it with either
//! `finish` (take the decoded frame at the commit point) or `abandon` (join and discard).
//! A read runs in one of two arms. From an attempt's root context with an attempt auxiliary
//! permit free, it is submitted as a `SupervisedWorker` job on the device executor's blocking
//! lane, holding an `AttemptWorkLease` (that permit plus a leased stream) until it is joined.
//! Otherwise it runs inline on the caller's context and stream: under an attempt work lease, such
//! as an execution lane, or on a root context whose permits are all held. The caller never waits
//! for a permit, because the root coordinator may itself hold every one through its lanes.
//!
//! This module does not count reads outstanding. The join-leaf scheduler keeps at most one
//! optional read ahead of the required one, and only when `max_source_reads_in_flight` exceeds 1;
//! for an admitted attempt, its auxiliary permits also bound the worker arm. The worker does not
//! host-synchronize the decode; see `ReadinessEvent`. Design owner:
//! `website/docs/design/data-path.mdx` and
//! `website/docs/design/admission.mdx#device-profiles-and-grants`.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use cudf::runtime::event::CudaEvent;
use query_runtime::{
    observability::{
        CleanupFailurePhase, CleanupFailureSummary, MetricsRecorder, RuntimeEventContext, RuntimeObservationHandle,
        RuntimeObservationPayload, RuntimeObservationRecordClass, RuntimePlanNodeId, RuntimeSourceReadObservation,
        RuntimeSpanId, RuntimeTaskId, RuntimeTopologyRecord, RuntimeWorkerId, SourceReadFact, SourceReadLifecycleState,
        TaskDefinitionRow, WorkerDefinitionRow,
    },
    runtime::{AttemptWorkLease, ExecutionCancellation, GpuExecutorLease, SupervisedWorker, SupervisedWorkerJoin},
};

use super::{
    BoundedExecutionAttempt, ExecutionContext, ExecutionMetrics, GpuDataFrame, LocalParquetSource, QueryError,
    QueryErrorCode, Result, SourceChunk, attach_row_group_facts, cudf_query_error, observed_chunk_row_group_count,
    parquet_chunk_path_fact, read_local_parquet_chunk,
};

/// Process-wide span ids for source-read observations. The plan span is id 1 and operator-stage
/// spans start at `1 << 32`, so this range sits between them.
static NEXT_SOURCE_READ_SPAN_ID: AtomicU64 = AtomicU64::new(10_000);

/// One started chunk read, owned by the consumer that spawned it.
///
/// The consumer must end it with `finish` or `abandon`. Both join the worker, observe the
/// readiness event and record the consumer-join lifecycle events; a plain drop does none of
/// that. `prefetch` only labels the read for the caller and does not change how it runs.
pub(in crate::exec) struct ScheduledLocalParquetRead {
    chunk: SourceChunk,
    state: ScheduledLocalParquetReadState,
    prefetch: bool,
}

enum ScheduledLocalParquetReadState {
    /// A leased worker read scheduled from the attempt's root context.
    ///
    /// `source_read_lease` keeps the worker's stream and auxiliary permit held until the
    /// consumer has joined the worker.
    Running {
        worker: SupervisedWorker<ScheduledLocalParquetReadResult>,
        source_read_lease: Box<AttemptWorkLease>,
        observation: Option<Arc<SourceReadObservation>>,
    },
    /// A read that already ran on the caller's context and stream, either an
    /// attempt work lease or a root context with no auxiliary permit free.
    Inline { result: Box<ScheduledLocalParquetReadResult>, observation: Option<Arc<SourceReadObservation>> },
}

/// A joined chunk read whose readiness event has been synchronized. `finish` reports
/// `observed_bytes` as 0; it does not measure device bytes.
pub(in crate::exec) struct ScheduledLocalParquetReadOutput {
    pub(in crate::exec) chunk: SourceChunk,
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) observed_bytes: u64,
    pub(in crate::exec) observed_row_group_count: Option<u64>,
}

/// What a read body returns. The readiness event is `None` only when recording it failed and the
/// worker host-synchronized its stream instead, so the frame is already complete.
struct ScheduledLocalParquetReadResult {
    frame: Result<(GpuDataFrame, Option<ReadinessEvent>)>,
    observed_row_group_count: Option<u64>,
}

/// Device-side readiness for one decoded source chunk.
///
/// The worker records this on its own stream after the last decode enqueue and does not
/// host-synchronize, so the worker thread recycles while decode drains. The consumer
/// synchronizes the event at its join point before committing the chunk; cross-stream
/// consumers additionally order device-side through the retained provenance fences, and
/// host-visible uses (D2H, scalars) keep their own stream synchronizes.
struct ReadinessEvent {
    event: CudaEvent,
}

impl ReadinessEvent {
    /// Records a CUDA event on the context's stream. It must run on the worker, after the read
    /// body's last enqueue, so the event orders after every decode kernel of the chunk.
    fn record_on_worker_stream(context: &ExecutionContext) -> Result<Self> {
        let event = context.runtime()?.record_completion_event().map_err(|error| {
            cudf_query_error(
                "query_engine.source_scheduler.readiness_record",
                "failed to record scheduled source read readiness on the worker stream",
                error,
            )
        })?;
        Ok(Self { event })
    }

    /// Blocks the calling thread until the decode enqueued before the event has finished. A
    /// failure is a deferred decode error from the chunk's own stream, reported with the chunk's
    /// path, index and row-group facts.
    fn synchronize_for_commit(self, chunk: &SourceChunk) -> Result<()> {
        self.event.synchronize().map_err(|error| {
            attach_row_group_facts(
                cudf_query_error(
                    "query_engine.source_scheduler.readiness_sync",
                    "scheduled source read readiness failed before consumer commit",
                    error,
                )
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("path", parquet_chunk_path_fact(chunk)),
                chunk,
            )
        })
    }
}

/// Observes one discarded read's readiness without consuming its frame.
///
/// Abandonment releases the frame, but a late decode failure must still be attributed to its
/// chunk and must not surface at an unrelated later fence. Failures are secondary to the
/// abandonment itself, so they are recorded as cleanup failures, never raised.
fn observe_abandoned_readiness(
    result: Result<(GpuDataFrame, Option<ReadinessEvent>)>,
    chunk: &SourceChunk,
    execution: &BoundedExecutionAttempt<'_>,
) {
    let Ok((_, readiness)) = result else { return };
    let Some(readiness) = readiness else { return };
    if let Err(error) = readiness.synchronize_for_commit(chunk) {
        if let Some(scope) = execution.context().runtime_execution_scope() {
            let summary = CleanupFailureSummary::from_native_error(CleanupFailurePhase::TaskExecution, &error);
            scope.observation().record_attempt_cleanup_failure(summary);
        } else {
            eprintln!("[query_engine::source_scheduler] warning: abandoned source read readiness failed: {error}");
        }
    }
}

/// Observation identity of one scheduled read: a `source_read_worker` task and a `source_read`
/// worker that exist in the topology for each read, including inline reads, so that lifecycle
/// events resolve to definition rows. Shared by the submitter and the worker closure.
struct SourceReadObservation {
    observation: RuntimeObservationHandle,
    metrics: MetricsRecorder,
    context: RuntimeEventContext,
    task_id: RuntimeTaskId,
    worker_id: RuntimeWorkerId,
    chunk_index: usize,
    file_index: usize,
}

impl SourceReadObservation {
    /// Returns `None` when the context has no admitted plan-execution scope, in which case the
    /// read runs unobserved. Writes the task and worker definition rows before any lifecycle
    /// event can reference them. `attempt` is a caller-supplied ordinal that is part of the task
    /// id, so two reads of one chunk get distinct tasks.
    fn new(context: &ExecutionContext, chunk: &SourceChunk, attempt: usize) -> Option<Arc<Self>> {
        let scope = context.runtime_execution_scope()?;
        let task_id = RuntimeTaskId::new(format!(
            "source-read-node-{}-chunk-{}-attempt-{attempt}",
            chunk.source_node_id, chunk.chunk_index
        ));
        let worker_id = RuntimeWorkerId::new(format!("{}-worker", task_id.as_str()));
        let span_id = RuntimeSpanId::new(NEXT_SOURCE_READ_SPAN_ID.fetch_add(1, Ordering::Relaxed));
        let observation = Arc::new(Self {
            observation: scope.observation().clone(),
            metrics: scope.metrics().clone(),
            context: RuntimeEventContext::for_plan(scope.plan_execution_id().clone())
                .with_span(span_id, Some(scope.plan_span_id())),
            task_id: task_id.clone(),
            worker_id: worker_id.clone(),
            chunk_index: chunk.chunk_index,
            file_index: chunk.file_index,
        });
        observation.observation.record_topology(RuntimeObservationRecordClass::TaskDefinition, || {
            RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
                plan_execution_id: scope.plan_execution_id().clone(),
                task_id,
                plan_node_id: Some(RuntimePlanNodeId::new(chunk.source_node_id.to_string())),
                task_kind: "source_read_worker".to_string(),
            })
        });
        observation.observation.record_topology(RuntimeObservationRecordClass::WorkerDefinition, || {
            let device_ordinal = u32::try_from(context.device().ordinal()).ok();
            RuntimeTopologyRecord::WorkerDefinition(WorkerDefinitionRow {
                plan_execution_id: Some(scope.plan_execution_id().clone()),
                worker_id,
                worker_kind: "source_read".to_string(),
                device_ordinal,
            })
        });
        Some(observation)
    }

    fn source_worker_scope(&self) -> (RuntimeTaskId, RuntimeWorkerId) {
        (self.task_id.clone(), self.worker_id.clone())
    }

    /// Returns the recorder's monotonic timestamp of the `Enqueued` event, or `None` when the
    /// recorder dropped it (capture exhausted or disabled).
    fn enqueue(&self) -> Option<u64> {
        self.observation.record_with_context_timestamped(self.context.clone(), |_monotonic_ns| {
            RuntimeObservationPayload::SourceRead(Box::new(RuntimeSourceReadObservation {
                task_id: self.task_id.clone(),
                worker_id: self.worker_id.clone(),
                chunk_index: self.chunk_index,
                file_index: self.file_index,
                lifecycle: SourceReadLifecycleState::Enqueued,
            }))
        })
    }

    fn worker_started(&self, enqueued_monotonic_ns: u64) -> Option<u64> {
        let worker_started_monotonic_ns =
            self.observation.record_with_context_timestamped(self.context.clone(), |worker_started_monotonic_ns| {
                let queue_wait_ns = worker_started_monotonic_ns.saturating_sub(enqueued_monotonic_ns);
                RuntimeObservationPayload::SourceRead(Box::new(RuntimeSourceReadObservation {
                    task_id: self.task_id.clone(),
                    worker_id: self.worker_id.clone(),
                    chunk_index: self.chunk_index,
                    file_index: self.file_index,
                    lifecycle: SourceReadLifecycleState::WorkerStarted { queue_wait_ns },
                }))
            })?;
        Some(worker_started_monotonic_ns.saturating_sub(enqueued_monotonic_ns))
    }

    fn record(&self, lifecycle: SourceReadLifecycleState) {
        self.observation.record_with_context(self.context.clone(), || {
            RuntimeObservationPayload::SourceRead(Box::new(RuntimeSourceReadObservation {
                task_id: self.task_id.clone(),
                worker_id: self.worker_id.clone(),
                chunk_index: self.chunk_index,
                file_index: self.file_index,
                lifecycle,
            }))
        });
    }

    /// Folds one source-read fact into this worker's plan execution totals.
    fn record_source_read(&self, fact: SourceReadFact) {
        self.metrics.record_source_read(fact);
    }
}

/// Start of one read's queue-wait measurement. It carries both the recorder timestamp and a
/// local `Instant`, because trace capture can drop the `Enqueued` event while the queue-wait
/// metric must still be reported.
struct SourceReadQueueStart {
    recorder_monotonic_ns: Option<u64>,
    fallback_started_at: Instant,
}

impl SourceReadQueueStart {
    fn new(observation: Option<&SourceReadObservation>) -> Self {
        let fallback_started_at = Instant::now();
        let recorder_monotonic_ns = observation.and_then(SourceReadObservation::enqueue);
        Self { recorder_monotonic_ns, fallback_started_at }
    }

    // The sole recording site for queue wait. The totals must see the same
    // measurement whether or not trace capture retained the worker-started
    // event, so the local clock feeds the fact whenever the recorder clock is
    // unavailable.
    fn worker_started(self, observation: Option<&SourceReadObservation>) -> Duration {
        let fallback = self.fallback_started_at.elapsed();
        let duration = match self.recorder_monotonic_ns {
            Some(enqueued_monotonic_ns) => observation
                .and_then(|observation| observation.worker_started(enqueued_monotonic_ns))
                .map(Duration::from_nanos)
                .unwrap_or(fallback),
            None => fallback,
        };
        if let Some(observation) = observation {
            observation.record_source_read(SourceReadFact::QueueWait(duration));
        }
        duration
    }
}

/// Records the worker-start event and queue wait, then runs `worker`. Used by both arms so that
/// an inline read reports a (near zero) queue wait with the event shape of a leased read.
fn run_source_read_worker<T, F>(
    queue_start: SourceReadQueueStart,
    observation: Option<Arc<SourceReadObservation>>,
    worker: F,
) -> T
where
    F: FnOnce(Duration, Option<Arc<SourceReadObservation>>) -> T,
{
    let queue_wait = queue_start.worker_started(observation.as_deref());
    worker(queue_wait, observation)
}

fn spawn_registered_source_read_worker<T, F>(
    queue_start: SourceReadQueueStart,
    observation: Option<Arc<SourceReadObservation>>,
    cancellation: ExecutionCancellation,
    executor: &GpuExecutorLease,
    worker: F,
) -> Result<SupervisedWorker<T>>
where
    T: Send + 'static,
    F: FnOnce(Duration, Option<Arc<SourceReadObservation>>) -> T + Send + 'static,
{
    // Worker submission, cancellation, and process-exit drain are owned by
    // the runtime device executor; the engine only produces the source work
    // closure with its observation semantics. The single submission entry
    // owns registration, pool admission, and the handle atomically. The
    // closure runs on the executor's blocking lane under an execution-class
    // job permit, never on the host runtime.
    let pool_observation = observation
        .as_ref()
        .map(|observation| observation.observation.clone())
        .unwrap_or_else(RuntimeObservationHandle::disabled);
    SupervisedWorker::submit(cancellation, executor, pool_observation, move || {
        run_source_read_worker(queue_start, observation, worker)
    })
}

/// The one source-read body, run either on a leased worker or inline on the
/// caller's own context and stream.
///
/// The returned closure never fails as a whole; a read error travels inside the result so that
/// `finish` can attach the chunk facts at the join. It records one terminal lifecycle
/// event: `ReadFinished`, `Cancelled` for `QueryErrorCode::Cancelled`, `ReadFailed` otherwise, or
/// `Panicked` from `SourceReadWorkerGuard::drop` on unwind.
fn read_local_parquet_chunk_worker(
    context: ExecutionContext,
    source: LocalParquetSource,
    chunk: SourceChunk,
    budget_bytes: u64,
    budget_boundary: &'static str,
    metrics: ExecutionMetrics,
) -> impl FnOnce(Duration, Option<Arc<SourceReadObservation>>) -> ScheduledLocalParquetReadResult + Send + 'static {
    move |_queue_wait, observation| {
        let read_started_at = Instant::now();
        let mut observation_guard = SourceReadWorkerGuard::new(observation);
        let frame = read_local_parquet_chunk(&context, &source, &chunk, budget_bytes, budget_boundary, &metrics)
            .and_then(|frame| {
                // Device-side consumers order through the frame's retained provenance fences;
                // the worker records readiness and does not block on its stream, so the pool
                // slot recycles while decode drains. When the event cannot be recorded, the
                // record failure is reported on stderr and the worker host-fences its stream
                // before the frame ships, so success is never reported for an unverified frame.
                match ReadinessEvent::record_on_worker_stream(&context) {
                    Ok(readiness) => Ok((frame, Some(readiness))),
                    Err(record_error) => {
                        eprintln!(
                            "[query_engine::source_scheduler] warning: failed to record source read readiness on the worker stream, falling back to host synchronization: {record_error}"
                        );
                        context.runtime()?.synchronize().map_err(|error| {
                            attach_row_group_facts(
                                cudf_query_error(
                                    "query_engine.source_scheduler.worker_sync",
                                    "failed to synchronize scheduled source read worker stream",
                                    error,
                                ),
                                &chunk,
                            )
                        })?;
                        Ok((frame, None))
                    },
                }
            });
        // Comes from the chunk's metadata, or from the local file footer when the chunk reads
        // every row group; no metrics counter delta is involved.
        // The read succeeded only when `frame` is `Ok`.
        let observed_row_group_count =
            frame.is_ok().then(|| observed_chunk_row_group_count(&chunk)).flatten().map(|count| count as u64);
        let read_duration_ns = read_started_at.elapsed().as_nanos() as u64;
        match &frame {
            Ok(_) => observation_guard.terminal(SourceReadLifecycleState::ReadFinished { read_duration_ns }),
            Err(error) if error.code() == QueryErrorCode::Cancelled => {
                observation_guard.terminal(SourceReadLifecycleState::Cancelled)
            },
            Err(error) => observation_guard.terminal(SourceReadLifecycleState::ReadFailed {
                error_code: error.code().to_string(),
                read_duration_ns,
            }),
        }
        ScheduledLocalParquetReadResult { frame, observed_row_group_count }
    }
}

/// Records at most one terminal lifecycle event for a read body. `terminal` covers the normal
/// exits; `Drop` adds `Panicked` only when it runs during unwinding and no terminal event was
/// recorded.
struct SourceReadWorkerGuard {
    observation: Option<Arc<SourceReadObservation>>,
    terminal_recorded: bool,
}

impl SourceReadWorkerGuard {
    fn new(observation: Option<Arc<SourceReadObservation>>) -> Self {
        Self { observation, terminal_recorded: false }
    }

    fn terminal(&mut self, lifecycle: SourceReadLifecycleState) {
        if self.terminal_recorded {
            return;
        }
        if let Some(observation) = &self.observation {
            observation.record(lifecycle);
        }
        self.terminal_recorded = true;
    }
}

impl Drop for SourceReadWorkerGuard {
    fn drop(&mut self) {
        if !self.terminal_recorded && std::thread::panicking() {
            self.terminal(SourceReadLifecycleState::Panicked);
        }
    }
}

/// Takes an attempt work lease for one read only if an attempt auxiliary permit is free now.
///
/// With the permit held, deriving the lease can still wait, cancellably, for a service stream.
/// The read's observation is registered only once the permit is held, so a read that never
/// starts leaves no task or worker definition behind for a later read of the same chunk to
/// collide with. Callers start the queue-wait clock after this returns, so the queue-wait metric
/// excludes the stream wait.
fn try_acquire_source_read_lease(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    attempt: usize,
) -> Result<Option<(AttemptWorkLease, Option<Arc<SourceReadObservation>>)>> {
    let mut observation = None;
    let lease = context.try_acquire_attempt_work_lease(|| {
        observation = SourceReadObservation::new(context, chunk, attempt);
        observation.as_ref().map(|observation| observation.source_worker_scope())
    })?;
    Ok(lease.map(|lease| (lease, observation)))
}

impl ScheduledLocalParquetRead {
    /// Starts a required read of `chunk`.
    ///
    /// The calling thread never waits for an attempt auxiliary permit. From an attempt's root
    /// context with a permit free, the read takes an attempt work lease, is submitted to the
    /// device executor, and this returns without waiting for it. Otherwise the read completes on
    /// the calling context and stream before this returns, with a read error deferred to `finish`:
    /// that covers an attempt work lease context, which already holds a permit, and a root
    /// context whose permits are all held, possibly by lanes this same thread leased. Errors from
    /// lease acquisition or submission, including a submission refused during process exit, are
    /// returned here and nothing is left running. `budget_bytes` and `budget_boundary` are
    /// forwarded to the bounded chunk read unchanged.
    pub(in crate::exec) fn spawn(
        context: ExecutionContext,
        metrics: ExecutionMetrics,
        source: LocalParquetSource,
        chunk: SourceChunk,
        budget_bytes: u64,
        budget_boundary: &'static str,
        attempt: usize,
    ) -> Result<Self> {
        if !context.is_attempt_work_lease_context()
            && let Some((lease, observation)) = try_acquire_source_read_lease(&context, &chunk, attempt)?
        {
            return Self::submit_on_lease(
                lease,
                observation,
                metrics,
                source,
                chunk,
                budget_bytes,
                budget_boundary,
                false,
            );
        }
        let observation = SourceReadObservation::new(&context, &chunk, attempt);
        Ok(Self::run_inline(context, observation, metrics, source, chunk, budget_bytes, budget_boundary, false))
    }

    /// Starts an optional read of `chunk` ahead of its consumer, or returns `None` when that
    /// would hold the calling thread.
    ///
    /// From an attempt's root context the read runs only on a free attempt auxiliary permit,
    /// as in `spawn`; with none free nothing is started or recorded and the caller reads the
    /// chunk later through `spawn`. From an attempt work lease context the read runs inline, as
    /// in `spawn`.
    pub(in crate::exec) fn spawn_prefetch(
        context: ExecutionContext,
        metrics: ExecutionMetrics,
        source: LocalParquetSource,
        chunk: SourceChunk,
        budget_bytes: u64,
        budget_boundary: &'static str,
        attempt: usize,
    ) -> Result<Option<Self>> {
        if context.is_attempt_work_lease_context() {
            let observation = SourceReadObservation::new(&context, &chunk, attempt);
            return Ok(Some(Self::run_inline(
                context,
                observation,
                metrics,
                source,
                chunk,
                budget_bytes,
                budget_boundary,
                true,
            )));
        }
        let Some((lease, observation)) = try_acquire_source_read_lease(&context, &chunk, attempt)? else {
            return Ok(None);
        };
        Self::submit_on_lease(lease, observation, metrics, source, chunk, budget_bytes, budget_boundary, true).map(Some)
    }

    // A read issued from an attempt work lease already holds that attempt's auxiliary permit,
    // and a root context with no permit free may be the holder of every one through its own
    // lanes. Leasing, or waiting for, another permit could then make a permit holder wait on
    // itself, so the read runs on the caller's own context and stream instead.
    #[expect(
        clippy::too_many_arguments,
        reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
    )]
    fn run_inline(
        context: ExecutionContext,
        observation: Option<Arc<SourceReadObservation>>,
        metrics: ExecutionMetrics,
        source: LocalParquetSource,
        chunk: SourceChunk,
        budget_bytes: u64,
        budget_boundary: &'static str,
        prefetch: bool,
    ) -> Self {
        let queue_start = SourceReadQueueStart::new(observation.as_deref());
        let result = run_source_read_worker(
            queue_start,
            observation.clone(),
            read_local_parquet_chunk_worker(context, source, chunk.clone(), budget_bytes, budget_boundary, metrics),
        );
        Self {
            chunk,
            state: ScheduledLocalParquetReadState::Inline { result: Box::new(result), observation },
            prefetch,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
    )]
    fn submit_on_lease(
        source_read_lease: AttemptWorkLease,
        observation: Option<Arc<SourceReadObservation>>,
        metrics: ExecutionMetrics,
        source: LocalParquetSource,
        chunk: SourceChunk,
        budget_bytes: u64,
        budget_boundary: &'static str,
        prefetch: bool,
    ) -> Result<Self> {
        let queue_start = SourceReadQueueStart::new(observation.as_deref());
        let thread_context = source_read_lease.context().clone();
        let worker_cancellation = thread_context.cancellation().clone();
        let worker_executor = thread_context.gpu_executor()?;
        let worker = spawn_registered_source_read_worker(
            queue_start,
            observation.clone(),
            worker_cancellation,
            &worker_executor,
            read_local_parquet_chunk_worker(
                thread_context,
                source,
                chunk.clone(),
                budget_bytes,
                budget_boundary,
                metrics,
            ),
        )?;
        Ok(Self {
            chunk,
            state: ScheduledLocalParquetReadState::Running {
                worker,
                source_read_lease: Box::new(source_read_lease),
                observation,
            },
            prefetch,
        })
    }

    pub(in crate::exec) fn is_prefetch(&self) -> bool {
        self.prefetch
    }

    /// Joins the read and returns its frame once the decode has completed on the device.
    ///
    /// Blocks the calling thread until the worker returns, then until the readiness event fires.
    /// A worker panic maps to `QueryErrorCode::Internal` and a worker lost to shutdown to
    /// `QueryErrorCode::Cancelled`; every error carries the chunk's path, index and row-group
    /// facts. The attempt work lease is released before this returns, on success and on error.
    pub(in crate::exec) fn finish(self) -> Result<ScheduledLocalParquetReadOutput> {
        let ScheduledLocalParquetRead { chunk, state, .. } = self;
        let result = match state {
            ScheduledLocalParquetReadState::Inline { result, observation } => {
                // The inline read already completed on the caller's stream;
                // record the degenerate join interval so the source-read
                // lifecycle stays a single shape for both arms.
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
                    observation.record(SourceReadLifecycleState::ConsumerJoinFinished { join_duration_ns: 0 });
                }
                *result
            },
            ScheduledLocalParquetReadState::Running { worker, source_read_lease, observation } => {
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
                }
                let join_started_at = Instant::now();
                let joined = worker.join();
                let join_duration_ns = join_started_at.elapsed().as_nanos() as u64;
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinFinished { join_duration_ns });
                }
                let result = match joined {
                    Some(SupervisedWorkerJoin::Completed(result)) => result,
                    Some(SupervisedWorkerJoin::Panicked) => {
                        return Err(attach_row_group_facts(
                            QueryError::new(
                                QueryErrorCode::Internal,
                                "query_engine.source_scheduler.join",
                                "native source scheduler read worker panicked",
                            )
                            .with_fact("chunk_index", chunk.chunk_index)
                            .with_fact("file_index", chunk.file_index)
                            .with_fact("path", parquet_chunk_path_fact(&chunk)),
                            &chunk,
                        ));
                    },
                    // `Cancelled` covers pool shutdown as well as process exit; `None` means the
                    // process-exit owner already joined the worker. Both report one reason.
                    Some(SupervisedWorkerJoin::Cancelled) | None => {
                        return Err(attach_row_group_facts(
                            QueryError::new(
                                QueryErrorCode::Cancelled,
                                "query_engine.source_scheduler.join",
                                "source read worker was stopped by process exit before consumer join",
                            )
                            .with_fact("reason", "source_worker_process_exit")
                            .with_fact("chunk_index", chunk.chunk_index)
                            .with_fact("file_index", chunk.file_index)
                            .with_fact("path", parquet_chunk_path_fact(&chunk)),
                            &chunk,
                        ));
                    },
                };
                // The lease held the worker's stream registered through the
                // join above; it releases at the end of this arm.
                let _source_read_lease = source_read_lease;
                result
            },
        };
        let (frame, readiness) = result.frame?;
        // Synchronizing the readiness event at the commit point is the coordinator's
        // completion observation for this chunk, so a late decode failure is reported
        // with this chunk's facts and not at an unrelated later host synchronize.
        if let Some(readiness) = readiness {
            readiness.synchronize_for_commit(&chunk)?;
        }
        Ok(ScheduledLocalParquetReadOutput {
            chunk,
            frame,
            observed_bytes: 0,
            observed_row_group_count: result.observed_row_group_count,
        })
    }

    /// Joins and discards the read, then counts a `PrefetchAbandoned` source-read fact.
    ///
    /// Blocks until the worker returns, so the lease's stream is not released while the worker
    /// still uses it. A read error or worker panic is not returned; a late decode failure is
    /// recorded as an attempt cleanup failure by `observe_abandoned_readiness`.
    pub(in crate::exec) fn abandon(self, execution: &mut BoundedExecutionAttempt<'_>) {
        let ScheduledLocalParquetRead { chunk, state, .. } = self;
        match state {
            // The inline read already completed and reported itself; an
            // abandoned read's frame is discarded unconsumed.
            ScheduledLocalParquetReadState::Inline { result, observation, .. } => {
                observe_abandoned_readiness(result.frame, &chunk, execution);
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
                    observation.record(SourceReadLifecycleState::ConsumerJoinFinished { join_duration_ns: 0 });
                }
            },
            ScheduledLocalParquetReadState::Running { worker, observation, .. } => {
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
                }
                let join_started_at = Instant::now();
                // The worker already reported its read to the plan execution;
                // an abandoned read's frame is discarded unconsumed, but its
                // readiness is still observed so a late decode failure is
                // attributed to its chunk and does not leak to an unrelated later fence.
                if let Some(SupervisedWorkerJoin::Completed(result)) = worker.join() {
                    observe_abandoned_readiness(result.frame, &chunk, execution);
                }
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinFinished {
                        join_duration_ns: join_started_at.elapsed().as_nanos() as u64,
                    });
                }
            },
        }
        execution.record_source_read(SourceReadFact::PrefetchAbandoned);
    }
}

#[cfg(test)]
mod observation_tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        time::Duration,
    };

    use query_runtime::observability::{
        DeterministicRuntimeObservationClock, PlanExecutionId, RuntimeObservationConfig, RuntimeObservationOwner,
        RuntimeObservationRecordClass, RuntimeObservationSession, RuntimeTerminalOutcome, RuntimeTopologyRecord,
        TaskDefinitionRow, WorkerDefinitionRow, assert_live_topology_references_resolve,
    };

    use super::*;

    fn test_observation(owner: &RuntimeObservationOwner) -> Arc<SourceReadObservation> {
        let plan_execution_id = PlanExecutionId::new("source-plan");
        let task_id = RuntimeTaskId::new("source-task");
        let worker_id = RuntimeWorkerId::new("source-worker");
        owner.handle().record_topology(RuntimeObservationRecordClass::TaskDefinition, || {
            RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
                plan_execution_id: plan_execution_id.clone(),
                task_id: task_id.clone(),
                plan_node_id: None,
                task_kind: "source_read_worker".to_string(),
            })
        });
        owner.handle().record_topology(RuntimeObservationRecordClass::WorkerDefinition, || {
            RuntimeTopologyRecord::WorkerDefinition(WorkerDefinitionRow {
                plan_execution_id: Some(plan_execution_id.clone()),
                worker_id: worker_id.clone(),
                worker_kind: "source_read".to_string(),
                device_ordinal: Some(0),
            })
        });
        Arc::new(SourceReadObservation {
            observation: owner.handle(),
            metrics: MetricsRecorder::new(owner.handle(), plan_execution_id.clone()),
            context: RuntimeEventContext::for_plan(plan_execution_id)
                .with_span(RuntimeSpanId::new(3), Some(RuntimeSpanId::new(1))),
            task_id,
            worker_id,
            chunk_index: 2,
            file_index: 4,
        })
    }

    #[test]
    fn queue_wait_projection_excludes_consumer_join_duration() {
        let owner =
            RuntimeObservationOwner::new("source-timing", None, RuntimeObservationConfig::bounded(32, 32 * 1024));
        let observation = test_observation(&owner);
        observation.record(SourceReadLifecycleState::Enqueued);
        observation.record(SourceReadLifecycleState::WorkerStarted { queue_wait_ns: 17 });
        observation.record(SourceReadLifecycleState::ReadFinished { read_duration_ns: 23 });
        observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
        observation.record(SourceReadLifecycleState::ConsumerJoinFinished { join_duration_ns: 101 });
        observation.record_source_read(SourceReadFact::QueueWait(Duration::from_nanos(17)));

        let checkpoint = observation.metrics.checkpoint().unwrap();
        assert_eq!(checkpoint.summary.source_read.source_read_queue_wait_time, Duration::from_nanos(17));
        let snapshot = owner.live_snapshot().unwrap();
        assert_live_topology_references_resolve(&snapshot);
        let states = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::SourceRead(observation) => Some(observation.lifecycle.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            states,
            vec![
                SourceReadLifecycleState::Enqueued,
                SourceReadLifecycleState::WorkerStarted { queue_wait_ns: 17 },
                SourceReadLifecycleState::ReadFinished { read_duration_ns: 23 },
                SourceReadLifecycleState::ConsumerJoinStarted,
                SourceReadLifecycleState::ConsumerJoinFinished { join_duration_ns: 101 },
            ]
        );
    }

    #[test]
    fn source_worker_unwind_records_one_panic_terminal() {
        let owner =
            RuntimeObservationOwner::new("source-panic", None, RuntimeObservationConfig::bounded(16, 16 * 1024));
        let observation = test_observation(&owner);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = SourceReadWorkerGuard::new(Some(observation));
            panic!("source worker panic");
        }));
        assert!(result.is_err());

        let panic_count = owner
            .live_snapshot()
            .unwrap()
            .events()
            .iter()
            .filter(|event| {
                matches!(
                    &event.payload,
                    RuntimeObservationPayload::SourceRead(observation)
                        if observation.lifecycle == SourceReadLifecycleState::Panicked
                )
            })
            .count();
        assert_eq!(panic_count, 1);
    }

    #[test]
    fn exhausted_capture_keeps_source_queue_timing_in_recorder_clock_domain() {
        let clock = Arc::new(DeterministicRuntimeObservationClock::new(None));
        let owner = RuntimeObservationOwner::from_session(RuntimeObservationSession::with_clock(
            "source-queue-exhausted-clock",
            None,
            RuntimeObservationConfig::bounded(3, 64 * 1024),
            Arc::clone(&clock) as _,
        ));
        let observation = test_observation(&owner);
        clock.set_monotonic_ns(1_000);
        let queue_start = SourceReadQueueStart::new(Some(&observation));
        clock.advance(37);

        assert_eq!(queue_start.worker_started(Some(&observation)), Duration::from_nanos(37));
        let snapshot = owner.live_snapshot().unwrap();
        assert!(snapshot.footer().truncated);
        assert!(
            snapshot.events().iter().all(|event| !matches!(&event.payload, RuntimeObservationPayload::SourceRead(_)))
        );
        owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded));
    }
}
