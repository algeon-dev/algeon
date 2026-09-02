#[cfg(test)]
use std::sync::Barrier;
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use parquet::file::reader::FileReader;

use super::*;
use crate::{
    exec::bounded::{
        ActiveReservation,
        chunks::{
            SourceReadReservationContractQuality, record_chunk_retry_error, source_read_budget_correction_from_error,
            try_grow_source_read_reservation_for_budget_correction,
        },
    },
    observability::{
        RuntimeEventContext, RuntimeObservationHandle, RuntimeObservationPayload, RuntimePlanNodeId,
        RuntimeSourceReadObservation, RuntimeSpanId, RuntimeTaskId, RuntimeTopologyRecord, RuntimeWorkerId,
        SourceReadLifecycleState, TaskDefinitionRow, WorkerDefinitionRow,
    },
    runtime::{
        ExecutionCancellation, ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown,
        register_cuda_process_exit_shutdown,
    },
};

static NEXT_SOURCE_READ_SPAN_ID: AtomicU64 = AtomicU64::new(10_000);

pub(in crate::exec) struct ScheduledLocalParquetRead {
    chunk: SourceChunk,
    state: ScheduledLocalParquetReadState,
    budget_boundary: &'static str,
    prefetch: bool,
}

enum ScheduledLocalParquetReadState {
    Running {
        reservation: Box<ActiveReservation>,
        worker: SourceReadWorker<ScheduledLocalParquetReadResult>,
        source_read_lease: Box<crate::runtime::SourceReadStreamLease>,
        source: Box<LocalParquetSource>,
        attempt: usize,
        max_attempts: usize,
        observation: Option<Arc<SourceReadObservation>>,
    },
}

pub(in crate::exec) struct ScheduledLocalParquetReadOutput {
    pub(in crate::exec) chunk: SourceChunk,
    pub(in crate::exec) frame: GpuDataFrame,
    pub(in crate::exec) reservation: ActiveReservation,
    pub(in crate::exec) observed_bytes: u64,
    pub(in crate::exec) observed_row_group_count: Option<u64>,
}

struct ScheduledLocalParquetReadResult {
    frame: Result<GpuDataFrame>,
    metrics: QueryMetrics,
    observed_row_group_count: Option<u64>,
    queue_wait: Duration,
}

/// Owns one detached source-read worker until it has been joined.
///
/// The normal scheduler consumer owns this wrapper, but process exit also
/// retains the same supervisor through the process-exit registry. That closes
/// the gap where `std::process::exit` skips the consumer's `finish` or
/// `abandon` call while the worker is still synchronizing a leased CUDA stream.
struct SourceReadWorker<T> {
    supervisor: Arc<SourceReadWorkerSupervisor<T>>,
}

struct SourceReadWorkerSupervisor<T> {
    cancellation: ExecutionCancellation,
    state: Mutex<SourceReadWorkerSupervisorState<T>>,
    state_changed: Condvar,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

struct SourceReadWorkerSupervisorState<T> {
    process_exit_started: bool,
    worker: Option<JoinHandle<T>>,
    joining: bool,
    joined: bool,
}

impl<T> SourceReadWorkerSupervisor<T> {
    fn new(cancellation: ExecutionCancellation) -> Arc<Self> {
        Arc::new(Self {
            cancellation,
            state: Mutex::new(SourceReadWorkerSupervisorState {
                process_exit_started: false,
                worker: None,
                joining: false,
                joined: false,
            }),
            state_changed: Condvar::new(),
            process_exit_registration: Mutex::new(None),
        })
    }

    fn request_process_exit(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.process_exit_started = true;
        drop(state);
        // Source reads check this token before entering the local-Parquet
        // boundary. It cannot interrupt a CUDA call already in flight, so the
        // subsequent join remains the terminal safety boundary.
        self.cancellation.cancel();
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    fn finish_join(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.joining = false;
        state.joined = true;
        self.state_changed.notify_all();
        drop(state);
        // Keep the registry entry live until the OS thread has stopped. This
        // is also safe when the process-exit snapshot already owns the entry.
        self.unregister_process_exit();
    }

    fn take_or_wait_for_join(&self) -> Option<JoinHandle<T>> {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(worker) = state.worker.take() {
                state.joining = true;
                return Some(worker);
            }
            if state.joining {
                state = self.state_changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
                continue;
            }
            return None;
        }
    }

    fn join(&self) -> Option<std::thread::Result<T>> {
        let worker = self.take_or_wait_for_join()?;
        let joined = worker.join();
        self.finish_join();
        Some(joined)
    }

    #[cfg(test)]
    fn joined_for_test(&self) -> bool {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).joined
    }
}

impl<T: Send + 'static> SourceReadWorkerSupervisor<T> {
    fn activate_process_exit(self: &Arc<Self>) {
        let shutdown: Arc<dyn ProcessExitShutdown> = self.clone();
        let registration = register_cuda_process_exit_shutdown(shutdown);
        *self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = registration;
    }

    fn spawn<F>(
        self: &Arc<Self>,
        queue_start: SourceReadQueueStart,
        observation: Option<Arc<SourceReadObservation>>,
        worker: F,
    ) -> Result<SourceReadWorker<T>>
    where
        F: FnOnce(Duration, Option<Arc<SourceReadObservation>>) -> T + Send + 'static,
    {
        // Hold this mutex across `thread::spawn` and handle publication. A
        // process-exit callback either observes the fully owned JoinHandle or
        // sets the terminal boundary first, in which case no CUDA-capable
        // worker may begin.
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.process_exit_started {
            drop(state);
            self.unregister_process_exit();
            return Err(QueryError::new(
                QueryErrorCode::Cancelled,
                "nexus_query_engine.source_scheduler.spawn",
                "source read worker cannot start after process exit begins",
            )
            .with_fact("reason", "process_exit_started"));
        }
        let handle =
            match thread::Builder::new().spawn(move || run_source_read_worker(queue_start, observation, worker)) {
                Ok(handle) => handle,
                Err(error) => {
                    drop(state);
                    self.unregister_process_exit();
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.source_scheduler.spawn",
                        "failed to spawn source read worker",
                    )
                    .with_fact("io_error", error));
                },
            };
        state.worker = Some(handle);
        drop(state);
        Ok(SourceReadWorker { supervisor: Arc::clone(self) })
    }
}

impl<T> SourceReadWorker<T> {
    fn join(self) -> Option<std::thread::Result<T>> {
        self.supervisor.join()
    }
}

impl<T: Send + 'static> ProcessExitShutdown for SourceReadWorkerSupervisor<T> {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::SourceReadWorker
    }

    fn shutdown_for_process_exit(&self) {
        self.request_process_exit();
        let _ = self.join();
        // `join` unregisters after an owned worker stops. If exit won the
        // race before spawning, there is no worker to join but the terminal
        // boundary still makes removing this registration safe.
        self.unregister_process_exit();
    }
}

struct SourceReadObservation {
    observation: RuntimeObservationHandle,
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
            context: RuntimeEventContext::for_plan(scope.plan_execution_id().clone())
                .with_span(span_id, Some(scope.plan_span_id())),
            task_id: task_id.clone(),
            worker_id: worker_id.clone(),
            chunk_index: chunk.chunk_index,
            file_index: chunk.file_index,
        });
        observation.observation.record_topology(
            crate::observability::RuntimeObservationRecordClass::TaskDefinition,
            || {
                RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
                    plan_execution_id: scope.plan_execution_id().clone(),
                    task_id,
                    plan_node_id: Some(RuntimePlanNodeId::new(chunk.source_node_id.to_string())),
                    task_kind: "source_read_worker".to_string(),
                })
            },
        );
        observation.observation.record_topology(
            crate::observability::RuntimeObservationRecordClass::WorkerDefinition,
            || {
                let device_ordinal = match context.device_placement() {
                    crate::runtime::DevicePlacement::SingleDevice { device_ordinal } => {
                        u32::try_from(device_ordinal).ok()
                    },
                    _ => None,
                };
                RuntimeTopologyRecord::WorkerDefinition(WorkerDefinitionRow {
                    plan_execution_id: Some(scope.plan_execution_id().clone()),
                    worker_id,
                    worker_kind: "source_read".to_string(),
                    device_ordinal,
                })
            },
        );
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

    fn worker_started(self, observation: Option<&SourceReadObservation>) -> Duration {
        let fallback = self.fallback_started_at.elapsed();
        let Some(enqueued_monotonic_ns) = self.recorder_monotonic_ns else {
            return fallback;
        };
        observation
            .and_then(|observation| observation.worker_started(enqueued_monotonic_ns))
            .map(Duration::from_nanos)
            .unwrap_or(fallback)
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
    worker: F,
) -> Result<SourceReadWorker<T>>
where
    T: Send + 'static,
    F: FnOnce(Duration, Option<Arc<SourceReadObservation>>) -> T + Send + 'static,
{
    let supervisor = SourceReadWorkerSupervisor::new(cancellation);
    // Register before spawning, not after: `std::process::exit` may otherwise
    // take its registry snapshot while this worker is already executing CUDA.
    supervisor.activate_process_exit();
    supervisor.spawn(queue_start, observation, worker)
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
) -> Result<(crate::runtime::SourceReadStreamLease, SourceReadQueueStart)> {
    let lease = context.acquire_source_read_lease(observation.map(|observation| observation.source_worker_scope()))?;
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
        source: LocalParquetSource,
        chunk: SourceChunk,
        reservation: ActiveReservation,
        budget_bytes: u64,
        budget_boundary: &'static str,
        prefetch: bool,
        attempt: usize,
        max_attempts: usize,
    ) -> Result<Self> {
        let thread_chunk = chunk.clone();
        let observation = SourceReadObservation::new(&context, &chunk, attempt);
        let (source_read_lease, queue_start) = acquire_source_read_lease_then_enqueue(&context, observation.as_ref())?;
        let thread_context = source_read_lease.context().clone();
        let thread_source = source.clone();
        let thread_observation = observation.clone();
        let worker_cancellation = thread_context.cancellation().clone();
        let worker = spawn_registered_source_read_worker(
            queue_start,
            thread_observation,
            worker_cancellation,
            move |queue_wait, thread_observation| {
                let read_started_at = Instant::now();
                let mut observation_guard = SourceReadWorkerGuard::new(thread_observation);
                let mut metrics = QueryMetrics::default();
                let selected_row_groups_before_read = metrics.row_groups_selected;
                let frame = read_local_parquet_chunk(
                    &thread_context,
                    &thread_source,
                    &thread_chunk,
                    budget_bytes,
                    budget_boundary,
                    &mut metrics,
                )
                .and_then(|frame| {
                    thread_context.runtime()?.synchronize().map_err(|error| {
                        QueryError::from_cudf(
                            "nexus_query_engine.source_scheduler.worker_sync",
                            "failed to synchronize scheduled source read worker stream",
                            error,
                        )
                    })?;
                    Ok(frame)
                });
                let observed_row_group_count = metrics
                    .row_groups_selected
                    .checked_sub(selected_row_groups_before_read)
                    .filter(|row_groups| *row_groups > 0);
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
                ScheduledLocalParquetReadResult { frame, metrics, observed_row_group_count, queue_wait }
            },
        )?;
        Ok(Self {
            chunk,
            state: ScheduledLocalParquetReadState::Running {
                reservation: Box::new(reservation),
                worker,
                source_read_lease: Box::new(source_read_lease),
                source: Box::new(source),
                attempt,
                max_attempts,
                observation,
            },
            budget_boundary,
            prefetch,
        })
    }

    pub(in crate::exec) fn is_prefetch(&self) -> bool {
        self.prefetch
    }

    pub(in crate::exec) fn budget_boundary(&self) -> &'static str {
        self.budget_boundary
    }

    pub(in crate::exec) fn reservation_bytes(&self) -> u64 {
        let ScheduledLocalParquetReadState::Running { reservation, .. } = &self.state;
        reservation.bytes()
    }

    pub(in crate::exec) fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<ScheduledLocalParquetReadOutput> {
        let ScheduledLocalParquetRead { chunk, state, budget_boundary, prefetch: _ } = self;
        let (mut reservation, worker, retry_context, observation) = match state {
            ScheduledLocalParquetReadState::Running {
                reservation,
                worker,
                source_read_lease,
                source,
                attempt,
                max_attempts,
                observation,
            } => (*reservation, worker, Some((*source_read_lease, *source, attempt, max_attempts)), observation),
        };
        if let Some(observation) = &observation {
            observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
        }
        let join_started_at = Instant::now();
        let joined = worker.join();
        let join_duration_ns = join_started_at.elapsed().as_nanos() as u64;
        if let Some(observation) = &observation {
            observation.record(SourceReadLifecycleState::ConsumerJoinFinished { join_duration_ns });
        }
        let result = joined
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Cancelled,
                    "nexus_query_engine.source_scheduler.join",
                    "source read worker was stopped by process exit before consumer join",
                )
                .with_fact("reason", "source_worker_process_exit")
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("path", parquet_chunk_path_fact(&chunk))
                .with_row_group_facts(&chunk)
            })?
            .map_err(|_panic| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.source_scheduler.join",
                    "native source scheduler read worker panicked",
                )
                .with_fact("chunk_index", chunk.chunk_index)
                .with_fact("file_index", chunk.file_index)
                .with_fact("path", parquet_chunk_path_fact(&chunk))
                .with_row_group_facts(&chunk)
            })?;
        execution.metrics_mut().record_source_read_queue_wait(result.queue_wait);
        execution.metrics_mut().merge_from(result.metrics);
        match result.frame {
            Ok(frame) => Ok(ScheduledLocalParquetReadOutput {
                chunk,
                frame,
                reservation,
                observed_bytes: 0,
                observed_row_group_count: result.observed_row_group_count,
            }),
            Err(error) => {
                if let Some((_source_read_lease, source, attempt, max_attempts)) = retry_context
                    && let Some(retry) = try_finish_scheduled_source_read_with_corrected_budget(
                        execution,
                        &source,
                        &chunk,
                        &mut reservation,
                        budget_boundary,
                        &error,
                        attempt,
                        max_attempts,
                    )?
                {
                    return Ok(ScheduledLocalParquetReadOutput {
                        chunk,
                        frame: retry.frame,
                        reservation,
                        observed_bytes: 0,
                        observed_row_group_count: retry.observed_row_group_count,
                    });
                }
                execution.release(reservation);
                Err(error)
            },
        }
    }

    pub(in crate::exec) fn abandon(self, execution: &mut BoundedExecutionAttempt<'_>) {
        let ScheduledLocalParquetRead { state, .. } = self;
        let result = match state {
            ScheduledLocalParquetReadState::Running { reservation, worker, observation, .. } => {
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinStarted);
                }
                let join_started_at = Instant::now();
                let result = worker.join();
                if let Some(observation) = &observation {
                    observation.record(SourceReadLifecycleState::ConsumerJoinFinished {
                        join_duration_ns: join_started_at.elapsed().as_nanos() as u64,
                    });
                }
                execution.release(*reservation);
                result
            },
        };
        execution.metrics_mut().record_source_read_prefetch_abandoned();
        if let Some(Ok(result)) = result {
            execution.metrics_mut().record_source_read_queue_wait(result.queue_wait);
            execution.metrics_mut().merge_from(result.metrics);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "source-read inputs (file/chunk identity, reader state, budgets) are distinct domain values"
)]
fn try_finish_scheduled_source_read_with_corrected_budget(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservation: &mut ActiveReservation,
    budget_boundary: &'static str,
    error: &QueryError,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<ScheduledLocalParquetReadRetryOutput>> {
    execution.context().check_cancellation("nexus_query_engine.source_scheduler.budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(execution, error, reservation) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, execution.metrics_mut());
    execution.metrics_mut().record_source_chunk_retry();
    crate::exec::retry::backoff_before_retry(execution.context(), "nexus_query_engine.source_scheduler.budget_retry")?;
    let request = source_read_request(
        ReservationKind::SourceRead,
        budget_boundary,
        0,
        execution.reservations().budget(),
        chunk.source_node_id,
        chunk,
        attempt + 1,
        max_attempts,
    )
    .with_source_read_contract(
        SourceReadReservationContractQuality::CorrectedLowerLevelBudget.as_str(),
        Some(correction.previous_budget_bytes()),
        Some(correction.corrected_budget_bytes()),
    );
    try_grow_source_read_reservation_for_budget_correction(execution, reservation, correction, request)?;
    execution.metrics_mut().record_source_chunk_attempt();
    let selected_row_groups_before_read = execution.metrics_mut().row_groups_selected;
    let frame = read_local_parquet_chunk(
        execution.context(),
        source,
        chunk,
        correction.corrected_budget_bytes(),
        budget_boundary,
        execution.metrics_mut(),
    )
    .map_err(|error| correction.attach_to_error(error))?;
    Ok(Some(ScheduledLocalParquetReadRetryOutput {
        frame,
        observed_row_group_count: scheduled_source_chunk_read_row_groups(
            chunk,
            selected_row_groups_before_read,
            execution.metrics_mut(),
        ),
    }))
}

struct ScheduledLocalParquetReadRetryOutput {
    frame: GpuDataFrame,
    observed_row_group_count: Option<u64>,
}

fn scheduled_source_chunk_read_row_groups(
    chunk: &SourceChunk,
    selected_row_groups_before_read: u64,
    metrics: &QueryMetrics,
) -> Option<u64> {
    if let Some(row_groups) = chunk.known_read_row_group_count() {
        return Some(row_groups as u64);
    }
    let observed_delta = metrics.row_groups_selected.saturating_sub(selected_row_groups_before_read);
    if observed_delta > 0 {
        return Some(observed_delta);
    }
    if chunk.reads_all_row_groups && chunk.path.is_file() {
        return parquet::file::reader::SerializedFileReader::try_from(chunk.path.as_path())
            .ok()
            .map(|reader| reader.metadata().num_row_groups() as u64);
    }
    None
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
    use crate::observability::{
        DeterministicRuntimeObservationClock, PlanExecutionId, RuntimeObservationConfig, RuntimeObservationOwner,
        RuntimeObservationSession, RuntimePlanExecutionLifecycle, RuntimeTerminalOutcome, RuntimeTopologyRecord,
        TaskDefinitionRow, WorkerDefinitionRow,
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
        owner.handle().record_topology(crate::observability::RuntimeObservationRecordClass::TaskDefinition, || {
            RuntimeTopologyRecord::TaskDefinition(TaskDefinitionRow {
                plan_execution_id: plan_execution_id.clone(),
                task_id: task_id.clone(),
                plan_node_id: None,
                task_kind: "source_read_worker".to_string(),
            })
        });
        owner.handle().record_topology(crate::observability::RuntimeObservationRecordClass::WorkerDefinition, || {
            RuntimeTopologyRecord::WorkerDefinition(WorkerDefinitionRow {
                plan_execution_id: Some(plan_execution_id.clone()),
                worker_id: worker_id.clone(),
                worker_kind: "source_read".to_string(),
                device_ordinal: Some(0),
            })
        });
        Arc::new(SourceReadObservation {
            observation: owner.handle(),
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
        let mut metrics = QueryMetrics::default();
        metrics.record_source_read_queue_wait(Duration::from_nanos(17));

        assert_eq!(metrics.source_read_queue_wait_time, Duration::from_nanos(17));
        let snapshot = owner.live_snapshot().unwrap();
        crate::observability::assert_live_topology_references_resolve(&snapshot);
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
    fn process_exit_joins_source_worker_and_prevents_late_spawn() {
        let cancellation = ExecutionCancellation::new();
        let supervisor = SourceReadWorkerSupervisor::new(cancellation.clone());
        let (started_sender, started_receiver) = mpsc::sync_channel(0);
        let worker_cancellation = cancellation.clone();
        let worker = supervisor
            .spawn(SourceReadQueueStart::new(None), None, move |_queue_wait, _observation| {
                started_sender.send(()).unwrap();
                while !worker_cancellation.is_cancelled() {
                    thread::yield_now();
                }
                7_u8
            })
            .unwrap();
        started_receiver.recv_timeout(Duration::from_secs(5)).unwrap();

        supervisor.shutdown_for_process_exit();

        assert!(cancellation.is_cancelled());
        assert!(supervisor.joined_for_test());
        assert!(worker.join().is_none(), "process-exit owner already joined the worker");

        let late_spawn = supervisor.spawn(SourceReadQueueStart::new(None), None, |_queue_wait, _observation| 9_u8);
        let error = match late_spawn {
            Ok(_) => panic!("source worker unexpectedly spawned after process exit"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("process_exit_started"));
    }

    #[test]
    fn contended_source_lease_wait_precedes_source_queue_enqueue() {
        let runtime = cudf::Runtime::new().unwrap();
        let pool = crate::runtime::ServiceStreamPool::new(runtime.device(), 1).unwrap();
        let held = pool.acquire().unwrap();
        let owner = RuntimeObservationOwner::new(
            "source-lease-contention",
            None,
            RuntimeObservationConfig::bounded(64, 64 * 1024),
        );
        let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("source-plan"));
        let context = crate::runtime::ExecutionContext::builder()
            .with_runtime(runtime)
            .with_service_stream_pool(pool.clone())
            .build()
            .unwrap()
            .with_runtime_observation_owner(owner.clone())
            .unwrap()
            .with_runtime_execution_scope(lifecycle.execution_scope())
            .unwrap();
        let observation = test_observation(&owner);
        let worker_observation = observation.clone();
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
        let pool = crate::runtime::ServiceStreamPool::new(runtime.device(), 1).unwrap();
        let held = pool.acquire().unwrap();
        let clock = Arc::new(DeterministicRuntimeObservationClock::new(None));
        let owner = RuntimeObservationOwner::from_session(RuntimeObservationSession::with_clock(
            "source-queue-clock",
            None,
            RuntimeObservationConfig::bounded(64, 64 * 1024),
            clock.clone(),
        ));
        let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("source-plan"));
        let context = crate::runtime::ExecutionContext::builder()
            .with_runtime(runtime)
            .with_service_stream_pool(pool.clone())
            .build()
            .unwrap()
            .with_runtime_observation_owner(owner.clone())
            .unwrap()
            .with_runtime_execution_scope(lifecycle.execution_scope())
            .unwrap();
        let observation = test_observation(&owner);
        let worker_observation = observation.clone();
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
        let spawn_observation = observation.clone();
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
            clock.clone(),
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
        let pool = crate::runtime::ServiceStreamPool::new(runtime.device(), 1).unwrap();
        let held = pool.acquire().unwrap();
        let cancellation = crate::runtime::ExecutionCancellation::new();
        let owner =
            RuntimeObservationOwner::new("source-lease-cancel", None, RuntimeObservationConfig::bounded(64, 64 * 1024));
        let lifecycle = RuntimePlanExecutionLifecycle::from_owner(owner.clone(), PlanExecutionId::new("source-plan"));
        let context = crate::runtime::ExecutionContext::builder()
            .with_runtime(runtime)
            .with_service_stream_pool(pool.clone())
            .with_cancellation(cancellation.clone())
            .build()
            .unwrap()
            .with_runtime_observation_owner(owner.clone())
            .unwrap()
            .with_runtime_execution_scope(lifecycle.execution_scope())
            .unwrap();
        let observation = test_observation(&owner);
        let worker_observation = observation.clone();
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
                if observation.lifecycle == crate::observability::StreamLeaseLifecycleState::Cancelled
        )));
        drop(held);
        lifecycle.finish(RuntimeTerminalOutcome::Cancelled);
    }
}
