#[cfg(test)]
use std::sync::Barrier;
#[cfg(test)]
use std::thread::{self, JoinHandle};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use cudf::runtime::event::CudaEvent;

use super::{
    BoundedExecutionAttempt, ExecutionContext, ExecutionMetrics, GpuDataFrame, LocalParquetSource, QueryError,
    QueryErrorCode, Result, SourceChunk, attach_row_group_facts, cudf_query_error, observed_chunk_row_group_count,
    parquet_chunk_path_fact, read_local_parquet_chunk,
};
use crate::{
    exec::{
        bounded::chunks::{record_chunk_retry_error, source_read_budget_correction_from_error},
        retry::backoff_before_retry,
    },
    observability::{
        CleanupFailurePhase, CleanupFailureSummary, MetricsRecorder, RuntimeEventContext, RuntimeObservationHandle,
        RuntimeObservationPayload, RuntimeObservationRecordClass, RuntimePlanNodeId, RuntimeSourceReadObservation,
        RuntimeSpanId, RuntimeTaskId, RuntimeTopologyRecord, RuntimeWorkerId, SourceReadFact, SourceReadLifecycleState,
        TaskDefinitionRow, WorkerDefinitionRow,
    },
    runtime::{AttemptWorkLease, ExecutionCancellation, GpuExecutorLease, SupervisedWorker, SupervisedWorkerJoin},
};

static NEXT_SOURCE_READ_SPAN_ID: AtomicU64 = AtomicU64::new(10_000);

pub(in crate::exec) struct ScheduledLocalParquetRead {
    chunk: SourceChunk,
    state: ScheduledLocalParquetReadState,
    budget_bytes: u64,
    budget_boundary: &'static str,
    prefetch: bool,
}

enum ScheduledLocalParquetReadState {
    /// A leased worker read scheduled from the attempt's root context.
    Running {
        worker: SupervisedWorker<ScheduledLocalParquetReadResult>,
        source_read_lease: Box<AttemptWorkLease>,
        source: Box<LocalParquetSource>,
        attempt: usize,
        max_attempts: usize,
        observation: Option<Arc<SourceReadObservation>>,
    },
    /// A read already issued from an attempt work lease, which ran inline on
    /// that lease's context and stream.
    ///
    /// The caller holds the attempt auxiliary permit behind that lease, so
    /// leasing a second one here would let a holder of one permit block on
    /// another permit of the same attempt.
    Inline {
        result: Box<ScheduledLocalParquetReadResult>,
        source: Box<LocalParquetSource>,
        attempt: usize,
        max_attempts: usize,
        observation: Option<Arc<SourceReadObservation>>,
    },
}

pub(in crate::exec) struct ScheduledLocalParquetReadOutput {
    pub(in crate::exec) chunk: SourceChunk,
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) observed_bytes: u64,
    pub(in crate::exec) observed_row_group_count: Option<u64>,
}

struct ScheduledLocalParquetReadResult {
    frame: Result<(GpuDataFrame, Option<ReadinessEvent>)>,
    observed_row_group_count: Option<u64>,
}

/// Device-side readiness for one decoded source chunk.
///
/// The worker records this on its own stream after the last decode enqueue instead of
/// host-synchronizing, so the worker thread recycles while decode drains. The consumer
/// synchronizes the event at its join point before committing the chunk; cross-stream
/// consumers additionally order device-side through the retained provenance fences, and
/// host-visible uses (D2H, scalars) keep their own stream synchronizes.
struct ReadinessEvent {
    event: CudaEvent,
}

impl ReadinessEvent {
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
/// chunk instead of surfacing at an unrelated later fence. Failures are secondary to the
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

    // The sole recording site for queue wait: the totals must see the same
    // measurement whether or not trace capture retained the worker-started
    // event, so the fallback clock feeds the fact exactly as it feeds the
    // caller's return value.
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

#[cfg(test)]
struct SourceReadSpawnTestHook {
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

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

#[cfg(test)]
fn spawn_source_read_worker<T, F>(
    queue_start: SourceReadQueueStart,
    observation: Option<Arc<SourceReadObservation>>,
    #[cfg(test)] before_spawn: Option<SourceReadSpawnTestHook>,
    worker: F,
) -> JoinHandle<T>
where
    T: Send + 'static,
    F: FnOnce(Duration, Option<Arc<SourceReadObservation>>) -> T + Send + 'static,
{
    #[cfg(test)]
    if let Some(before_spawn) = before_spawn {
        before_spawn.entered.wait();
        before_spawn.release.wait();
    }
    thread::spawn(move || run_source_read_worker(queue_start, observation, worker))
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
    // owns registration, pool admission, and the handle atomically.
    let pool_observation = observation
        .as_ref()
        .map(|observation| observation.observation.clone())
        .unwrap_or_else(RuntimeObservationHandle::disabled);
    SupervisedWorker::submit(cancellation, executor, pool_observation, move || {
        run_source_read_worker(queue_start, observation, worker)
    })
}

/// The one source-read body, run either on a leased worker or inline on an
/// attempt work lease's own context and stream.
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
                // the worker records readiness instead of blocking on its stream so the pool
                // slot recycles while decode drains. When the event cannot be recorded, the
                // record failure is reported and the worker host-fences its stream before the
                // frame ships, so success is never reported for an unverified frame.
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
        // A pure function of the chunk's own metadata, not a metrics counter
        // delta: the read succeeded only when `frame` is `Ok`.
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

fn acquire_source_read_lease_then_enqueue(
    context: &ExecutionContext,
    observation: Option<&Arc<SourceReadObservation>>,
) -> Result<(AttemptWorkLease, SourceReadQueueStart)> {
    let lease = context.acquire_attempt_work_lease(observation.map(|observation| observation.source_worker_scope()))?;
    let queue_start = SourceReadQueueStart::new(observation.map(Arc::as_ref));
    Ok((lease, queue_start))
}

impl ScheduledLocalParquetRead {
    #[expect(
        clippy::too_many_arguments,
        reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
    )]
    pub(in crate::exec) fn spawn(
        context: ExecutionContext,
        metrics: ExecutionMetrics,
        source: LocalParquetSource,
        chunk: SourceChunk,
        budget_bytes: u64,
        budget_boundary: &'static str,
        prefetch: bool,
        attempt: usize,
        max_attempts: usize,
    ) -> Result<Self> {
        let observation = SourceReadObservation::new(&context, &chunk, attempt);
        // A read issued from an attempt work lease already holds that
        // attempt's auxiliary permit. Leasing another one here would make a
        // permit holder wait on a second permit of the same attempt, so the
        // read runs inline on the lease's own context and stream instead.
        if context.is_attempt_work_lease_context() {
            let queue_start = SourceReadQueueStart::new(observation.as_deref());
            let result = run_source_read_worker(
                queue_start,
                observation.clone(),
                read_local_parquet_chunk_worker(
                    context,
                    source.clone(),
                    chunk.clone(),
                    budget_bytes,
                    budget_boundary,
                    metrics,
                ),
            );
            return Ok(Self {
                chunk,
                state: ScheduledLocalParquetReadState::Inline {
                    result: Box::new(result),
                    source: Box::new(source),
                    attempt,
                    max_attempts,
                    observation,
                },
                budget_bytes,
                budget_boundary,
                prefetch,
            });
        }
        let (source_read_lease, queue_start) = acquire_source_read_lease_then_enqueue(&context, observation.as_ref())?;
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
                source.clone(),
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
                source: Box::new(source),
                attempt,
                max_attempts,
                observation,
            },
            budget_bytes,
            budget_boundary,
            prefetch,
        })
    }

    pub(in crate::exec) fn is_prefetch(&self) -> bool {
        self.prefetch
    }

    pub(in crate::exec) const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub(in crate::exec) fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<ScheduledLocalParquetReadOutput> {
        let ScheduledLocalParquetRead { chunk, state, budget_boundary, .. } = self;
        let (result, retry_context) = match state {
            ScheduledLocalParquetReadState::Inline { result, source, attempt, max_attempts, observation } => {
                // The inline read already completed on the caller's stream;
                // record the degenerate join interval so the source-read
                // lifecycle stays a single shape for both arms.
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
                    observation.record(SourceReadLifecycleState::ConsumerJoinFinished { join_duration_ns: 0 });
                }
                (*result, Some((*source, attempt, max_attempts)))
            },
            ScheduledLocalParquetReadState::Running {
                worker,
                source_read_lease,
                source,
                attempt,
                max_attempts,
                observation,
            } => {
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
                (result, Some((*source, attempt, max_attempts)))
            },
        };
        match result.frame {
            Ok((frame, readiness)) => {
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
            },
            Err(error) => {
                if let Some((source, attempt, max_attempts)) = retry_context
                    && let Some(retry) = try_finish_scheduled_source_read_with_corrected_budget(
                        execution,
                        &source,
                        &chunk,
                        budget_boundary,
                        &error,
                        attempt,
                        max_attempts,
                    )?
                {
                    return Ok(ScheduledLocalParquetReadOutput {
                        chunk,
                        frame: retry.frame,
                        observed_bytes: 0,
                        observed_row_group_count: retry.observed_row_group_count,
                    });
                }
                Err(error)
            },
        }
    }

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
                // attributed instead of leaking to an unrelated later fence.
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

fn try_finish_scheduled_source_read_with_corrected_budget(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    budget_boundary: &'static str,
    error: &QueryError,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<ScheduledLocalParquetReadRetryOutput>> {
    execution.context().check_cancellation("query_engine.source_scheduler.budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(execution, error) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, execution.metrics());
    execution.record_source_read(SourceReadFact::ChunkRetried);
    backoff_before_retry(execution.context(), "query_engine.source_scheduler.budget_retry")?;
    execution.record_source_read(SourceReadFact::ChunkAttempted);
    let frame = read_local_parquet_chunk(
        execution.context(),
        source,
        chunk,
        correction.corrected_budget_bytes(),
        budget_boundary,
        execution.metrics(),
    )
    .map_err(|error| correction.attach_to_error(error))?;
    Ok(Some(ScheduledLocalParquetReadRetryOutput {
        frame,
        observed_row_group_count: observed_chunk_row_group_count(chunk).map(|count| count as u64),
    }))
}

struct ScheduledLocalParquetReadRetryOutput {
    frame: GpuDataFrame,
    observed_row_group_count: Option<u64>,
}

#[cfg(test)]
mod observation_tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{Barrier, mpsc},
        thread,
        time::Duration,
    };

    use super::*;
    use crate::{
        observability::{
            DeterministicRuntimeObservationClock, PlanExecutionId, RuntimeObservationConfig, RuntimeObservationOwner,
            RuntimeObservationRecordClass, RuntimeObservationSession, RuntimePlanExecutionLifecycle,
            RuntimeTerminalOutcome, RuntimeTopologyRecord, StreamLeaseLifecycleState, TaskDefinitionRow,
            WorkerDefinitionRow, assert_live_topology_references_resolve,
        },
        runtime::{ExecutionCancellation, ExecutionContext, ServiceStreamPool},
    };

    fn observed_source_states(owner: &RuntimeObservationOwner) -> Vec<SourceReadLifecycleState> {
        owner
            .live_snapshot()
            .unwrap()
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::SourceRead(observation) => Some(observation.lifecycle.clone()),
                _ => None,
            })
            .collect()
    }

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
    fn contended_source_lease_wait_precedes_source_queue_enqueue() {
        let runtime = cudf::Runtime::new().unwrap();
        let pool = ServiceStreamPool::new(runtime.device(), 1).unwrap();
        let held = pool.acquire().unwrap();
        let observed_context = ExecutionContext::builder()
            .with_runtime(runtime)
            .with_service_stream_pool(pool.clone())
            .with_runtime_observation_session(RuntimeObservationSession::new(
                "source-lease-contention",
                None,
                RuntimeObservationConfig::bounded(64, 64 * 1024),
            ))
            .build_with_runtime_observation()
            .unwrap();
        let owner = observed_context.observation().clone();
        let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("source-plan"));
        let context = observed_context.context().with_runtime_execution_scope(lifecycle.execution_scope()).unwrap();
        let observation = test_observation(&owner);
        let worker_observation = Arc::clone(&observation);
        let waiter = thread::spawn(move || acquire_source_read_lease_then_enqueue(&context, Some(&worker_observation)));

        pool.wait_until_acquire_queued_for_test(Instant::now() + Duration::from_secs(5));
        assert!(observed_source_states(&owner).is_empty(), "lease contention is not source worker queue time");
        drop(held);
        let (lease, _queue_start) = waiter.join().unwrap().unwrap();
        assert_eq!(observed_source_states(&owner), vec![SourceReadLifecycleState::Enqueued]);

        let stream_event = owner
            .live_snapshot()
            .unwrap()
            .events()
            .iter()
            .find_map(|event| match &event.payload {
                RuntimeObservationPayload::StreamLease(stream)
                    if stream.scope.task_id.as_ref().is_some_and(|id| id.as_str() == observation.task_id.as_str()) =>
                {
                    Some((event.plan_execution_id.clone(), stream.scope.clone()))
                },
                _ => None,
            })
            .expect("source stream correlation is retained");
        assert_eq!(stream_event.0.as_ref().map(|id| id.as_str()), Some("source-plan"));
        assert_eq!(
            stream_event.1.source_worker_id.as_ref().map(|id| id.as_str()),
            Some(observation.worker_id.as_str())
        );
        drop(lease);
        lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
    }

    #[test]
    fn production_worker_queue_interval_uses_recorder_clock_and_excludes_lease_wait_and_join() {
        let runtime = cudf::Runtime::new().unwrap();
        let pool = ServiceStreamPool::new(runtime.device(), 1).unwrap();
        let held = pool.acquire().unwrap();
        let clock = Arc::new(DeterministicRuntimeObservationClock::new(None));
        let observed_context = ExecutionContext::builder()
            .with_runtime(runtime)
            .with_service_stream_pool(pool.clone())
            .with_runtime_observation_session(RuntimeObservationSession::with_clock(
                "source-queue-clock",
                None,
                RuntimeObservationConfig::bounded(64, 64 * 1024),
                Arc::clone(&clock) as _,
            ))
            .build_with_runtime_observation()
            .unwrap();
        let owner = observed_context.observation().clone();
        let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("source-plan"));
        let context = observed_context.context().with_runtime_execution_scope(lifecycle.execution_scope()).unwrap();
        let observation = test_observation(&owner);
        let worker_observation = Arc::clone(&observation);
        let lease_waiter =
            thread::spawn(move || acquire_source_read_lease_then_enqueue(&context, Some(&worker_observation)));

        pool.wait_until_acquire_queued_for_test(Instant::now() + Duration::from_secs(5));
        clock.advance(100);
        drop(held);
        let (lease, queue_start) = lease_waiter.join().unwrap().unwrap();
        assert_eq!(observed_source_states(&owner), vec![SourceReadLifecycleState::Enqueued]);

        let before_spawn_entered = Arc::new(Barrier::new(2));
        let before_spawn_release = Arc::new(Barrier::new(2));
        let worker_release = Arc::new(Barrier::new(2));
        let (started_sender, started_receiver) = mpsc::sync_channel(0);
        let spawn_observation = observation;
        let hook = SourceReadSpawnTestHook {
            entered: Arc::clone(&before_spawn_entered),
            release: Arc::clone(&before_spawn_release),
        };
        let worker_release_clone = Arc::clone(&worker_release);
        let spawn_setup = thread::spawn(move || {
            spawn_source_read_worker(queue_start, Some(spawn_observation), Some(hook), move |queue_wait, _| {
                started_sender.send(queue_wait).unwrap();
                worker_release_clone.wait();
                queue_wait
            })
        });

        before_spawn_entered.wait();
        clock.advance(37);
        before_spawn_release.wait();
        let worker = spawn_setup.join().unwrap();
        assert_eq!(started_receiver.recv().unwrap(), Duration::from_nanos(37));
        clock.advance(89);
        worker_release.wait();
        assert_eq!(worker.join().unwrap(), Duration::from_nanos(37));
        assert_eq!(
            observed_source_states(&owner),
            vec![SourceReadLifecycleState::Enqueued, SourceReadLifecycleState::WorkerStarted { queue_wait_ns: 37 },]
        );

        drop(lease);
        lifecycle.finish(RuntimeTerminalOutcome::Succeeded);
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

    #[test]
    fn cancelled_source_lease_wait_does_not_orphan_source_enqueue() {
        let runtime = cudf::Runtime::new().unwrap();
        let pool = ServiceStreamPool::new(runtime.device(), 1).unwrap();
        let held = pool.acquire().unwrap();
        let cancellation = ExecutionCancellation::new();
        let observed_context = ExecutionContext::builder()
            .with_runtime(runtime)
            .with_service_stream_pool(pool.clone())
            .with_cancellation(cancellation.clone())
            .with_runtime_observation_session(RuntimeObservationSession::new(
                "source-lease-cancel",
                None,
                RuntimeObservationConfig::bounded(64, 64 * 1024),
            ))
            .build_with_runtime_observation()
            .unwrap();
        let owner = observed_context.observation().clone();
        let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("source-plan"));
        let context = observed_context.context().with_runtime_execution_scope(lifecycle.execution_scope()).unwrap();
        let observation = test_observation(&owner);
        let worker_observation = observation;
        let waiter = thread::spawn(move || acquire_source_read_lease_then_enqueue(&context, Some(&worker_observation)));

        pool.wait_until_acquire_queued_for_test(Instant::now() + Duration::from_secs(5));
        cancellation.cancel();
        let error = match waiter.join().unwrap() {
            Ok((_lease, _queue_start)) => panic!("cancelled source lease unexpectedly acquired"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert!(observed_source_states(&owner).is_empty());
        assert!(owner.live_snapshot().unwrap().events().iter().any(|event| matches!(
            &event.payload,
            RuntimeObservationPayload::StreamLease(observation)
                if observation.lifecycle == StreamLeaseLifecycleState::Cancelled
        )));
        drop(held);
        lifecycle.finish(RuntimeTerminalOutcome::Cancelled);
    }
}
