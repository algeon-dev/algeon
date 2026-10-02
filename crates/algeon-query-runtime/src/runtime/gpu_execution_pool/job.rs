//! Per-job observation, counters, and scoped execution guards for the GPU execution pool.

use std::{
    cell::Cell,
    future::Future,
    hint, panic,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync,
    sync::{
        Arc,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    thread,
};

use futures::FutureExt;
use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::task::{JoinError, JoinHandle};

use super::{
    BUILD_OPERATION, CURRENT_POOL_ID, CURRENT_WORKER_INDEX, GpuExecutionJobKind, GpuExecutionQueryTracker, PoolDrain,
    RESERVED_LIFECYCLE_WORKERS, RUN_OPERATION, inner::WorkerSeat,
};
use crate::observability::{
    GpuJobLifecycleState, JobDefinitionRow, RuntimeGpuJobObservation, RuntimeJobId, RuntimeObservationHandle,
    RuntimeObservationPayload, RuntimeObservationRecordClass, RuntimeStreamLeaseScope, RuntimeTopologyRecord,
    RuntimeWorkerId, WorkerDefinitionRow,
};

pub(super) struct JobObservation {
    pub(super) observation: RuntimeObservationHandle,
    pub(super) job_id: RuntimeJobId,
    pub(super) job_kind: GpuExecutionJobKind,
    pub(super) device_ordinal: u32,
    pub(super) lifecycle: AtomicU8,
}

pub(super) const JOB_LIFECYCLE_PENDING: u8 = 0;
const JOB_LIFECYCLE_CANCELLATION_RECORDING: u8 = 1;
const JOB_LIFECYCLE_CANCELLATION_REQUESTED: u8 = 2;
const JOB_LIFECYCLE_TERMINAL: u8 = 3;

impl JobObservation {
    pub(super) fn enqueued(&self) {
        self.observation.record_topology(RuntimeObservationRecordClass::JobDefinition, || {
            RuntimeTopologyRecord::JobDefinition(JobDefinitionRow {
                plan_execution_id: None,
                job_id: self.job_id.clone(),
                job_kind: self.job_kind.as_str().to_string(),
                device_ordinal: self.device_ordinal,
            })
        });
        self.record(GpuJobLifecycleState::Enqueued, None);
    }

    pub(super) fn started(&self, worker_index: Option<usize>) -> RuntimeWorkerId {
        let worker_id = RuntimeWorkerId::new(format!(
            "{}-worker-{}",
            self.job_id.as_str(),
            worker_index.map_or_else(|| "inline".to_string(), |index| index.to_string())
        ));
        self.observation.record_topology(RuntimeObservationRecordClass::WorkerDefinition, || {
            RuntimeTopologyRecord::WorkerDefinition(WorkerDefinitionRow {
                plan_execution_id: None,
                worker_id: worker_id.clone(),
                worker_kind: "gpu_execution_pool".to_string(),
                device_ordinal: Some(self.device_ordinal),
            })
        });
        self.record(GpuJobLifecycleState::Started, Some(worker_id.clone()));
        worker_id
    }

    /// Requests cancellation unless a terminal transition already won.
    ///
    /// Returns whether the child may still need to be aborted. The transient
    /// recording state prevents a racing terminal event from being retained
    /// before its preceding cancellation request.
    pub(super) fn cancellation_requested(&self) -> bool {
        loop {
            match self.lifecycle.load(Ordering::Acquire) {
                JOB_LIFECYCLE_TERMINAL => return false,
                JOB_LIFECYCLE_CANCELLATION_REQUESTED => return true,
                JOB_LIFECYCLE_CANCELLATION_RECORDING => hint::spin_loop(),
                JOB_LIFECYCLE_PENDING => {
                    if self
                        .lifecycle
                        .compare_exchange(
                            JOB_LIFECYCLE_PENDING,
                            JOB_LIFECYCLE_CANCELLATION_RECORDING,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        self.record(GpuJobLifecycleState::CancellationRequested, None);
                        self.lifecycle.store(JOB_LIFECYCLE_CANCELLATION_REQUESTED, Ordering::Release);
                        return true;
                    }
                },
                unexpected => unreachable!("invalid GPU job lifecycle state {unexpected}"),
            }
        }
    }

    pub(super) fn terminal(&self, lifecycle: GpuJobLifecycleState) {
        loop {
            let current = self.lifecycle.load(Ordering::Acquire);
            match current {
                JOB_LIFECYCLE_TERMINAL => return,
                JOB_LIFECYCLE_CANCELLATION_RECORDING => hint::spin_loop(),
                JOB_LIFECYCLE_PENDING | JOB_LIFECYCLE_CANCELLATION_REQUESTED => {
                    if self
                        .lifecycle
                        .compare_exchange(current, JOB_LIFECYCLE_TERMINAL, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        self.record(lifecycle, None);
                        return;
                    }
                },
                unexpected => unreachable!("invalid GPU job lifecycle state {unexpected}"),
            }
        }
    }

    pub(super) fn is_terminal(&self) -> bool {
        self.lifecycle.load(Ordering::Acquire) == JOB_LIFECYCLE_TERMINAL
    }

    pub(super) fn has_cancellation_request(&self) -> bool {
        matches!(
            self.lifecycle.load(Ordering::Acquire),
            JOB_LIFECYCLE_CANCELLATION_RECORDING | JOB_LIFECYCLE_CANCELLATION_REQUESTED
        )
    }

    fn record(&self, lifecycle: GpuJobLifecycleState, worker_id: Option<RuntimeWorkerId>) {
        self.observation.record(|| {
            RuntimeObservationPayload::GpuJob(Box::new(RuntimeGpuJobObservation {
                job_id: self.job_id.clone(),
                job_kind: self.job_kind.as_str().to_string(),
                device_ordinal: self.device_ordinal,
                worker_id,
                lifecycle,
            }))
        });
    }
}

#[derive(Debug, Default)]
pub(super) struct PoolCounters {
    pub(super) active: AtomicUsize,
    pub(super) peak_active: AtomicUsize,
    pub(super) queued: AtomicUsize,
    pub(super) peak_queued: AtomicUsize,
    pub(super) active_queries: AtomicUsize,
    pub(super) peak_active_queries: AtomicUsize,
}

#[derive(Debug, Default)]
pub(super) struct QueryCounters {
    pub(super) active: AtomicUsize,
    pub(super) peak: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobCounterState {
    Queued,
    Active,
}

pub(super) struct JobCounterGuard {
    kind: GpuExecutionJobKind,
    counters: Arc<PoolCounters>,
    drain: Arc<PoolDrain>,
    query_tracker: GpuExecutionQueryTracker,
    state: JobCounterState,
    pub(super) terminal: JobTerminalGuard,
    completion: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl JobCounterGuard {
    pub(super) fn new(
        kind: GpuExecutionJobKind,
        counters: Arc<PoolCounters>,
        drain: Arc<PoolDrain>,
        query_tracker: GpuExecutionQueryTracker,
        observation: Arc<JobObservation>,
    ) -> Self {
        increment_with_peak(&counters.queued, &counters.peak_queued);
        Self {
            kind,
            counters,
            drain,
            query_tracker,
            state: JobCounterState::Queued,
            terminal: JobTerminalGuard::spawned(observation),
            completion: None,
        }
    }

    pub(super) fn new_with_completion(
        kind: GpuExecutionJobKind,
        counters: Arc<PoolCounters>,
        drain: Arc<PoolDrain>,
        query_tracker: GpuExecutionQueryTracker,
        observation: Arc<JobObservation>,
        completion: Box<dyn FnOnce() + Send + 'static>,
    ) -> Self {
        let mut guard = Self::new(kind, counters, drain, query_tracker, observation);
        guard.completion = Some(completion);
        guard
    }

    pub(super) fn activate(&mut self) -> RuntimeStreamLeaseScope {
        debug_assert_eq!(self.state, JobCounterState::Queued);
        decrement(&self.counters.queued);
        let worker_id = self.terminal.observation.started(current_worker_index());
        increment_with_peak(&self.counters.active, &self.counters.peak_active);
        if self.kind == GpuExecutionJobKind::QueryExecution {
            increment_with_peak(&self.counters.active_queries, &self.counters.peak_active_queries);
            self.query_tracker.start_query();
        }
        self.state = JobCounterState::Active;
        RuntimeStreamLeaseScope::default().with_gpu_worker(self.terminal.observation.job_id.clone(), worker_id)
    }
}

impl Drop for JobCounterGuard {
    fn drop(&mut self) {
        // Publish the terminal transition before counters expose pool
        // quiescence. Callers use those counters as a completion fence and
        // must never observe an active count of zero with a nonterminal job.
        self.terminal.finish_drop();

        // The completion hook can acquire arbitrary locks or block. Run it
        // before publishing pool quiescence, but never while retaining the
        // drain mutex that serializes the zero-counter transition.
        let completion_panic =
            self.completion.take().and_then(|completion| panic::catch_unwind(AssertUnwindSafe(completion)).err());

        {
            // Synchronize the zero-counter transition with drain waiters.
            // Completion above releases the job's final resource anchors, so
            // a waiter that observes zero after this decrement may proceed.
            let _drain_state = self.drain.changed.lock().unwrap_or_else(sync::PoisonError::into_inner);
            match self.state {
                JobCounterState::Queued => decrement(&self.counters.queued),
                JobCounterState::Active => {
                    if self.kind == GpuExecutionJobKind::QueryExecution {
                        self.query_tracker.finish_query();
                        decrement(&self.counters.active_queries);
                    }
                    decrement(&self.counters.active);
                },
            }
            self.drain.wake.notify_all();
        }

        if let Some(panic) = completion_panic {
            panic::resume_unwind(panic);
        }
    }
}

pub(super) struct JobTerminalGuard {
    observation: Arc<JobObservation>,
    cancellation_on_drop: bool,
}

impl JobTerminalGuard {
    pub(super) fn spawned(observation: Arc<JobObservation>) -> Self {
        Self { observation, cancellation_on_drop: false }
    }

    pub(super) fn inline(observation: Arc<JobObservation>) -> Self {
        Self { observation, cancellation_on_drop: true }
    }

    pub(super) fn terminal(&mut self, lifecycle: GpuJobLifecycleState) {
        self.observation.terminal(lifecycle);
    }

    pub(super) fn finish_drop(&mut self) {
        if self.observation.is_terminal() {
            return;
        }
        if thread::panicking() {
            self.observation.terminal(GpuJobLifecycleState::Panicked);
        } else if self.cancellation_on_drop {
            self.observation.cancellation_requested();
            self.observation.terminal(GpuJobLifecycleState::Cancelled);
        } else if self.observation.has_cancellation_request() {
            self.observation.terminal(GpuJobLifecycleState::Cancelled);
        } else {
            self.observation.terminal(GpuJobLifecycleState::Dropped);
        }
    }
}

impl Drop for JobTerminalGuard {
    fn drop(&mut self) {
        self.finish_drop();
    }
}

pub(super) async fn run_counted_job<F, T>(
    mut guard: JobCounterGuard,
    _seat: WorkerSeat,
    future: F,
) -> std::result::Result<T, ()>
where
    F: Future<Output = T>,
{
    let _scope = guard.activate();
    match AssertUnwindSafe(future).catch_unwind().await {
        Ok(value) => {
            guard.terminal.terminal(GpuJobLifecycleState::Finished);
            Ok(value)
        },
        Err(_panic) => {
            guard.terminal.terminal(GpuJobLifecycleState::Panicked);
            Err(())
        },
    }
}

pub(super) async fn run_counted_job_scoped<B, F, T>(
    mut guard: JobCounterGuard,
    _seat: WorkerSeat,
    build_future: B,
) -> std::result::Result<T, ()>
where
    B: FnOnce(RuntimeStreamLeaseScope) -> F,
    F: Future<Output = T>,
{
    let scope = guard.activate();
    match AssertUnwindSafe(build_future(scope)).catch_unwind().await {
        Ok(value) => {
            guard.terminal.terminal(GpuJobLifecycleState::Finished);
            Ok(value)
        },
        Err(_panic) => {
            guard.terminal.terminal(GpuJobLifecycleState::Panicked);
            Err(())
        },
    }
}

pub(super) fn current_pool_id() -> Option<u64> {
    CURRENT_POOL_ID.with(Cell::get)
}

pub(super) fn current_worker_index() -> Option<usize> {
    CURRENT_WORKER_INDEX.with(Cell::get)
}

pub(super) struct AbortOnDropJoinHandle<T> {
    handle: JoinHandle<T>,
    observation: Arc<JobObservation>,
    completed: bool,
}

impl<T> AbortOnDropJoinHandle<T> {
    pub(super) fn new(handle: JoinHandle<T>, observation: Arc<JobObservation>) -> Self {
        Self { handle, observation, completed: false }
    }
}

impl<T> Future for AbortOnDropJoinHandle<T> {
    type Output = std::result::Result<T, JoinError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let result = Pin::new(&mut self.handle).poll(context);
        if result.is_ready() {
            self.completed = true;
        }
        result
    }
}

impl<T> Drop for AbortOnDropJoinHandle<T> {
    fn drop(&mut self) {
        if !self.completed && self.observation.cancellation_requested() {
            self.handle.abort();
        }
    }
}

pub(super) fn validate_worker_count(device_ordinal: usize, worker_count: usize) -> Result<()> {
    if worker_count <= RESERVED_LIFECYCLE_WORKERS {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            BUILD_OPERATION,
            "GPU execution pool worker count must exceed its reserved lifecycle workers",
        )
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("worker_count", worker_count)
        .with_fact("reserved_lifecycle_workers", RESERVED_LIFECYCLE_WORKERS)
        .with_fact("reason", "below_lifecycle_reserve"));
    }
    Ok(())
}

pub(super) fn job_panic_error(device_ordinal: usize, kind: GpuExecutionJobKind) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, RUN_OPERATION, "GPU execution pool job panicked")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("job_kind", kind.as_str())
        .with_fact("reason", "job_panicked")
}

pub(super) fn job_join_panic_error(device_ordinal: usize, kind: GpuExecutionJobKind, error: JoinError) -> QueryError {
    job_panic_error(device_ordinal, kind).with_source(error)
}

pub(super) fn job_cancelled_error(device_ordinal: usize, kind: GpuExecutionJobKind, error: JoinError) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, RUN_OPERATION, "GPU execution pool job was cancelled")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("job_kind", kind.as_str())
        .with_fact("reason", "job_cancelled")
        .with_source(error)
}

pub(super) fn increment_with_peak(active: &AtomicUsize, peak: &AtomicUsize) {
    let current = active.fetch_add(1, Ordering::AcqRel).saturating_add(1);
    peak.fetch_max(current, Ordering::AcqRel);
}

pub(super) fn decrement(counter: &AtomicUsize) {
    let previous = counter.fetch_sub(1, Ordering::AcqRel);
    debug_assert!(previous > 0, "GPU execution pool counter underflow");
}
