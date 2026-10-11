//! Dedicated asynchronous executors for synchronous GPU work.
//!
//! GPU calls are commonly synchronous from Rust's point of view. Running them
//! on a serving runtime therefore pins serving workers until the GPU call
//! returns. A `GpuExecutionPool` gives each device a separately sized Tokio
//! runtime and keeps that blocking work away from the serving runtime.
//!
//! Work reaches the device along two paths that share one admission layer.
//! A job-class permit is taken at submission and bounds how many accepted
//! jobs exist; a saturated class fails with `ServiceOverloaded` rather than
//! queueing, except for the admission envelope of a placed attempt, which is
//! the one bounded waiter. The capacities derive from the device profile
//! (`A_d` active attempts, `S_d` auxiliary slots per attempt); the design
//! contract lives in `website/docs/design/admission.mdx`.
//!
//! An async job (`run*`, `spawn*`) is an `async` block whose body is
//! synchronous GPU work. The runtime serves as a fixed-width thread pool for
//! it: the spawned task awaits one of `worker_count` seats and holds it until
//! the body returns, so at most `worker_count` async job bodies run at once
//! and the number of seat waiters is bounded by the class capacities.
//! Blocking work (`submit_blocking*`) runs on the runtime's blocking threads
//! instead and takes no seat; its concurrency is bounded only by its class
//! capacity. Chunk lanes and source reads use this path so a decode never
//! occupies an async worker. Concurrent GPU calls per device are therefore
//! bounded by `worker_count` plus the class capacities, with the class
//! permits as the shared limit.
//!
//! Shutdown closes submissions, asks the supervisor to drop the runtime,
//! waits for the job census to drain, and joins the supervisor thread, each
//! under the caller's deadline. The process-exit hook runs the same sequence
//! without a deadline so no CUDA-owning task outlives libcudart.

use std::{
    cell::Cell,
    future::Future,
    panic::AssertUnwindSafe,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};
#[cfg(any(test, feature = "test-hooks"))]
use std::{
    collections::HashMap,
    sync::{OnceLock, Weak},
};

use futures::FutureExt;
use query_planning::error::{QueryError, QueryErrorCode, Result};
use serde::Serialize;
#[cfg(any(test, feature = "test-hooks"))]
use tokio::runtime::EnterGuard;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use super::{ExecutionCancellation, context::QueryServiceResourceEpoch};
use crate::observability::{GpuJobLifecycleState, RuntimeJobId, RuntimeObservationHandle, RuntimeStreamLeaseScope};

mod inner;
mod job;
mod lease;
mod supervisor;

use inner::{GpuExecutionPoolInner, PoolDrain, pool_closed_error};
use job::{
    AbortOnDropJoinHandle, JOB_LIFECYCLE_PENDING, JobCounterGuard, JobObservation, JobTerminalGuard, PoolCounters,
    QueryCounters, current_pool_id, current_worker_index, decrement, increment_with_peak, job_cancelled_error,
    job_join_panic_error, job_panic_error, run_counted_job, run_counted_job_scoped, validate_worker_count,
};
use lease::{BlockingCompletionNotify, BlockingWorkOutcome};
pub use lease::{GpuExecutorLease, PoolBlockingWork};
use supervisor::RuntimeSupervisor;

const BUILD_OPERATION: &str = "query_engine.gpu_execution_pool.build";
const RUN_OPERATION: &str = "query_engine.gpu_execution_pool.run";
#[cfg(any(test, feature = "test-hooks"))]
const SHARED_OPERATION: &str = "query_engine.gpu_execution_pool.shared";
const SHUTDOWN_OPERATION: &str = "query_engine.gpu_execution_pool.shutdown";

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static CURRENT_POOL_ID: Cell<Option<u64>> = const { Cell::new(None) };
    static CURRENT_WORKER_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Classifies work submitted to the backend-owned GPU execution pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuExecutionJobKind {
    /// CUDA and execution-context initialization.
    ContextSetup,
    /// One admitted attempt's root query execution job.
    QueryExecution,
    /// Auxiliary work an admitted attempt issues underneath its root job:
    /// execution lanes and scheduled source reads. Every such submission is
    /// backed by the caller's attempt auxiliary permit.
    AttemptWork,
    /// Resource synchronization and destruction detached from request serving.
    Cleanup,
}

/// Workers each pool keeps for lifecycle jobs (context setup and release
/// cleanup). Query execution and its auxiliary work run only on the other
/// workers, so a placed admission's envelope and a finished attempt's cleanup
/// never wait behind running queries.
pub(crate) const RESERVED_LIFECYCLE_WORKERS: usize = 1;

/// Resource class partitioning for bounded worker admission.
///
/// For active-attempt limit `A_d` and per-attempt source-work bound `S_d`,
/// each device reserves `A_d` lifecycle job slots and `A_d * S_d` execution
/// job slots. Additional fragment setup is execution work. Child work can
/// never consume a future lifecycle slot: class permits are acquired before a
/// job occupies a worker thread, and no hidden fallback pool exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GpuExecutionJobClass {
    /// Per-attempt lifecycle work: initial context setup and cleanup.
    Lifecycle,
    /// Bounded query execution and additional fragment setup.
    Execution,
}

impl GpuExecutionJobClass {
    /// Stable diagnostic label for this job class (`root` or `auxiliary`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lifecycle => "root",
            Self::Execution => "auxiliary",
        }
    }
}

impl GpuExecutionJobKind {
    /// Stable diagnostic label for this job kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextSetup => "context_setup",
            Self::QueryExecution => "query_execution",
            Self::AttemptWork => "attempt_work",
            Self::Cleanup => "cleanup",
        }
    }

    /// Resource class admitted for this job kind.
    ///
    /// A root query job is per-attempt lifecycle work, like the context setup
    /// it follows; only the auxiliary work that job issues underneath itself
    /// is execution-class. Context setup is the one kind whose class depends
    /// on the attempt's own claim rather than on the kind, so it is selected
    /// explicitly through [`GpuExecutionPool::run_observed_scoped_in_class`].
    pub(crate) const fn job_class(self) -> GpuExecutionJobClass {
        match self {
            Self::ContextSetup | Self::QueryExecution | Self::Cleanup => GpuExecutionJobClass::Lifecycle,
            Self::AttemptWork => GpuExecutionJobClass::Execution,
        }
    }

    /// Whether the job runs only on workers outside
    /// [`RESERVED_LIFECYCLE_WORKERS`].
    pub(crate) const fn runs_outside_lifecycle_reserve(self) -> bool {
        matches!(self, Self::QueryExecution | Self::AttemptWork)
    }
}

/// Point-in-time counters for one device's GPU execution pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GpuExecutionPoolSnapshot {
    /// Opaque process-local identity of the backend-owned pool instance.
    ///
    /// Clones of the same pool report the same value. A newly constructed pool
    /// reports a distinct value, including after a backend generation changes.
    pub pool_id: u64,
    /// CUDA device ordinal associated with the pool.
    pub device_ordinal: usize,
    /// Maximum number of concurrently active jobs.
    pub worker_count: usize,
    /// Jobs that have started and not yet completed.
    pub active_jobs: usize,
    /// Highest observed value of [`Self::active_jobs`].
    pub(crate) peak_active_jobs: usize,
    /// Accepted jobs that have not started.
    pub queued_jobs: usize,
    /// Highest observed value of [`Self::queued_jobs`].
    pub(crate) peak_queued_jobs: usize,
    /// Active query-execution jobs on this device.
    pub active_query_jobs: usize,
    /// Highest observed device-local query-execution concurrency.
    pub peak_active_query_jobs: usize,
    /// Configured lifecycle-class job slots (`root` diagnostic label).
    pub(crate) root_job_capacity: usize,
    /// Configured execution-class job slots (`auxiliary` diagnostic label).
    pub(crate) aux_job_capacity: usize,
    /// Lifecycle-class job slots currently free (`root` diagnostic label).
    pub(crate) root_jobs_available: usize,
    /// Execution-class job slots currently free (`auxiliary` diagnostic label).
    pub(crate) aux_jobs_available: usize,
}

/// Point-in-time counters shared by a group of GPU execution pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GpuExecutionQuerySnapshot {
    /// Query-execution jobs active across all attached pools.
    pub(crate) active_query_jobs: usize,
    /// Highest exact concurrent query-execution count across attached pools.
    pub peak_active_query_jobs: usize,
}

/// Cloneable aggregate counter for service-wide query execution concurrency.
///
/// Give the same tracker to each per-device pool through
/// [`GpuExecutionPool::new_with_tracker`] to obtain a temporally correct peak.
/// Summing device-local peaks is not equivalent because those peaks may have
/// occurred at different times.
#[derive(Debug, Clone, Default)]
pub(crate) struct GpuExecutionQueryTracker {
    counters: Arc<QueryCounters>,
}

impl GpuExecutionQueryTracker {
    /// Creates an empty aggregate query tracker.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the current and peak query concurrency across attached pools.
    pub(crate) fn snapshot(&self) -> GpuExecutionQuerySnapshot {
        let active_query_jobs = self.counters.active.load(Ordering::Acquire);
        let peak_active_query_jobs = self.counters.peak.load(Ordering::Acquire).max(active_query_jobs);
        GpuExecutionQuerySnapshot { active_query_jobs, peak_active_query_jobs }
    }

    fn start_query(&self) {
        increment_with_peak(&self.counters.active, &self.counters.peak);
    }

    fn finish_query(&self) {
        decrement(&self.counters.active);
    }
}

/// Fixed-width, per-device executor for work that can block on the GPU.
///
/// Clones share one runtime and one set of counters. Dropping the last clone
/// closes submissions and asks the supervisor to drop the runtime; an owner
/// that needs bounded, observed teardown calls the explicit shutdown methods
/// before that.
#[derive(Clone)]
pub(crate) struct GpuExecutionPool {
    inner: Arc<GpuExecutionPoolInner>,
    /// Query-service epoch this clone is bound to, when it was obtained
    /// through an admitted context. Submissions are refused once the epoch
    /// closes, and each spawned task retains the epoch so the service cannot
    /// reach idle while a job it admitted is still running.
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
}

impl GpuExecutionPool {
    /// Enters this pool's Tokio runtime, so a future polled on a blocking
    /// thread can register its timers with that runtime.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn enter_runtime(&self) -> EnterGuard<'_> {
        self.inner.handle.enter()
    }

    /// Creates a pool with an independent query-concurrency tracker.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn new(device_ordinal: usize, worker_count: usize) -> Result<Self> {
        Self::new_with_tracker(device_ordinal, worker_count, GpuExecutionQueryTracker::new())
    }

    /// Creates a pool attached to a caller-provided aggregate query tracker.
    ///
    /// Direct constructors use the documented default class partition:
    /// `worker_count` lifecycle job slots and `worker_count` execution job slots.
    /// Backend-owned pools pass explicit `A_d`/`A_d * S_d` capacities from
    /// the device resource profile via [`Self::new_with_tracker_and_classes`].
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn new_with_tracker(
        device_ordinal: usize,
        worker_count: usize,
        query_tracker: GpuExecutionQueryTracker,
    ) -> Result<Self> {
        Self::new_with_tracker_and_classes(device_ordinal, worker_count, worker_count, worker_count, query_tracker)
    }

    /// Creates a pool with explicit lifecycle/execution job-slot capacities.
    pub(crate) fn new_with_tracker_and_classes(
        device_ordinal: usize,
        worker_count: usize,
        lifecycle_job_capacity: usize,
        execution_job_capacity: usize,
        query_tracker: GpuExecutionQueryTracker,
    ) -> Result<Self> {
        validate_worker_count(device_ordinal, worker_count)?;
        for (field, capacity) in
            [("root_job_capacity", lifecycle_job_capacity), ("aux_job_capacity", execution_job_capacity)]
        {
            if capacity == 0 {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidConfiguration,
                    "query_engine.gpu.execution_pool",
                    "GPU execution pool job-class capacity must be greater than zero",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("field", field)
                .with_fact("reason", "zero"));
            }
        }

        let id = NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed);
        let next_worker_index = Arc::new(AtomicUsize::new(0));
        let (handle, supervisor) = RuntimeSupervisor::start(device_ordinal, worker_count, id, next_worker_index)?;

        let inner = Arc::new(GpuExecutionPoolInner {
            id,
            device_ordinal,
            worker_count,
            handle,
            supervisor: Arc::clone(&supervisor),
            submissions_closed: AtomicBool::new(false),
            submission_gate: Mutex::new(()),
            concurrency: Arc::new(Semaphore::new(worker_count)),
            execution_workers: Arc::new(Semaphore::new(worker_count - RESERVED_LIFECYCLE_WORKERS)),
            lifecycle_job_permits: Arc::new(Semaphore::new(lifecycle_job_capacity)),
            execution_job_permits: Arc::new(Semaphore::new(execution_job_capacity)),
            // Every admitted attempt can transition to exactly one release
            // cleanup. Keep that bounded lifecycle tail separate from
            // caller-submitted lifecycle jobs so an unrelated saturated
            // lifecycle job cannot force CUDA-bound envelope destruction onto
            // its caller.
            // The task still uses the same fixed-width device runtime.
            release_cleanup_permits: Arc::new(Semaphore::new(lifecycle_job_capacity)),
            lifecycle_job_capacity,
            execution_job_capacity,
            counters: Arc::new(PoolCounters::default()),
            drain: Arc::new(PoolDrain::default()),
            query_tracker,
            next_job_id: AtomicU64::new(1),
        });
        supervisor.register_process_exit(Arc::downgrade(&inner));
        Ok(Self { inner, resource_epoch: None })
    }

    #[cfg(test)]
    fn class_semaphore(&self, kind: GpuExecutionJobKind) -> &Arc<Semaphore> {
        self.class_semaphore_for_class(kind.job_class())
    }

    fn class_semaphore_for_class(&self, class: GpuExecutionJobClass) -> &Arc<Semaphore> {
        match class {
            GpuExecutionJobClass::Lifecycle => &self.inner.lifecycle_job_permits,
            GpuExecutionJobClass::Execution => &self.inner.execution_job_permits,
        }
    }

    #[cfg(test)]
    fn class_saturated_error(&self, kind: GpuExecutionJobKind) -> QueryError {
        self.class_saturated_error_for_class(kind, kind.job_class())
    }

    fn class_saturated_error_for_class(&self, kind: GpuExecutionJobKind, class: GpuExecutionJobClass) -> QueryError {
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.gpu.execution_pool",
            "GPU execution pool job-class capacity is saturated",
        )
        .with_fact("device_ordinal", self.inner.device_ordinal)
        .with_fact("job_kind", kind.as_str())
        .with_fact("job_class", class.as_str())
        .with_fact("reason", "gpu_execution_pool_class_saturated")
    }

    fn release_cleanup_saturated_error(&self) -> QueryError {
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.gpu.execution_pool",
            "GPU execution pool bounded release-cleanup capacity is saturated",
        )
        .with_fact("device_ordinal", self.inner.device_ordinal)
        .with_fact("job_kind", GpuExecutionJobKind::Cleanup.as_str())
        .with_fact("job_class", "release_cleanup")
        .with_fact("reason", "gpu_execution_pool_release_cleanup_saturated")
    }

    /// Returns the process-global fallback pool for one CUDA device.
    ///
    /// A live fallback pool fixes the worker count for its device. Requesting a
    /// different width is rejected rather than silently constructing another
    /// executor for the same device. Once the last clone is dropped, a later
    /// call may install a new width.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn shared_for_device(device_ordinal: usize, worker_count: usize) -> Result<Self> {
        validate_worker_count(device_ordinal, worker_count)?;

        static SHARED_POOLS: OnceLock<Mutex<HashMap<usize, Weak<GpuExecutionPoolInner>>>> = OnceLock::new();
        let shared_pools = SHARED_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut shared_pools = shared_pools.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                SHARED_OPERATION,
                "GPU execution pool registry mutex was poisoned",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("worker_count", worker_count)
            .with_fact("reason", "registry_poisoned")
        })?;

        if let Some(inner) = shared_pools.get(&device_ordinal).and_then(Weak::upgrade) {
            if inner.worker_count != worker_count {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    SHARED_OPERATION,
                    "CUDA device already has a shared GPU execution pool with a different worker count",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("existing_worker_count", inner.worker_count)
                .with_fact("requested_worker_count", worker_count)
                .with_fact("reason", "worker_count_conflict"));
            }
            return Ok(Self { inner, resource_epoch: None });
        }

        let pool = Self::new(device_ordinal, worker_count)?;
        shared_pools.insert(device_ordinal, Arc::downgrade(&pool.inner));
        Ok(pool)
    }

    /// Runs one job on the device pool and awaits its result.
    ///
    /// Dropping the returned future aborts the submitted task. Awaiting it
    /// joins the task and converts task panic or cancellation into a structured
    /// native error. Calls made from a job already running on this same pool
    /// execute inline, which permits width-one setup/execution composition.
    pub async fn run<F, T>(&self, kind: GpuExecutionJobKind, future: F) -> Result<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.run_observed(kind, RuntimeObservationHandle::disabled(), future).await
    }

    /// Runs one job while emitting host-side queue and worker lifecycle evidence.
    ///
    /// These events describe executor submission and host execution only. They
    /// do not claim CUDA kernel duration.
    pub(crate) async fn run_observed<F, T>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        future: F,
    ) -> Result<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.clone().run_observed_scoped(kind, observation, move |_scope| future).await
    }

    /// Runs one observed job and passes its explicit job/worker correlation to
    /// work that may acquire a service-stream lease.
    pub fn run_observed_scoped<B, F, T>(
        self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        build_future: B,
    ) -> impl Future<Output = Result<T>> + Send + 'static
    where
        B: FnOnce(RuntimeStreamLeaseScope) -> F + Send + 'static,
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.run_observed_scoped_in_class(kind, kind.job_class(), observation, build_future)
    }

    /// Runs one observed job through an explicit bounded resource class.
    ///
    /// Context setup retains its stable observation kind even when a later
    /// fragment uses an auxiliary stream; its class is selected by the
    /// admitted attempt's context reservation rather than inferred from that
    /// observability label. Initial setup uses the lifecycle class; additional
    /// fragment setup uses the execution class.
    pub(crate) async fn run_observed_scoped_in_class<B, F, T>(
        self,
        kind: GpuExecutionJobKind,
        class: GpuExecutionJobClass,
        observation: RuntimeObservationHandle,
        build_future: B,
    ) -> Result<T>
    where
        B: FnOnce(RuntimeStreamLeaseScope) -> F + Send + 'static,
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.check_resource_epoch(RUN_OPERATION)?;
        // A job-class permit bounds every accepted awaited submission, not
        // only detached `spawn` work. Acquiring asynchronously here would
        // create an unbounded semaphore waiter queue containing arbitrary
        // caller futures. Same-pool composition uses the same bounded permit
        // below and executes inline only after that accounting succeeds.
        let class_permit = Arc::clone(self.class_semaphore_for_class(class))
            .try_acquire_owned()
            .map_err(|_error| self.class_saturated_error_for_class(kind, class))?;
        self.run_observed_scoped_with_class_permit(kind, class_permit, observation, build_future).await
    }

    /// Waits for a free slot of `class`, or returns the typed error of
    /// `cancellation` once it is requested or its deadline passes. A free slot
    /// is taken without a timer, so only a caller that has to wait needs a
    /// Tokio context.
    ///
    /// Only work whose own count the class width already bounds may wait
    /// here, so the waiter queue stays bounded: the admission envelope of
    /// each placed attempt, whose lane caps them at the lifecycle width.
    pub(crate) async fn wait_for_class_permit(
        &self,
        kind: GpuExecutionJobKind,
        class: GpuExecutionJobClass,
        cancellation: &ExecutionCancellation,
    ) -> Result<OwnedSemaphorePermit> {
        self.check_resource_epoch(RUN_OPERATION)?;
        let semaphore = self.class_semaphore_for_class(class);
        match Arc::clone(semaphore).try_acquire_owned() {
            Ok(permit) => return Ok(permit),
            Err(TryAcquireError::Closed) => return Err(self.class_saturated_error_for_class(kind, class)),
            Err(TryAcquireError::NoPermits) => {},
        }
        tokio::select! {
            biased;
            permit = Arc::clone(self.class_semaphore_for_class(class)).acquire_owned() => {
                permit.map_err(|_closed| self.class_saturated_error_for_class(kind, class))
            },
            error = cancellation.terminated(RUN_OPERATION) => Err(error),
        }
    }

    /// Runs one observed job under a class permit the caller already holds.
    pub(crate) async fn run_observed_scoped_with_class_permit<B, F, T>(
        self,
        kind: GpuExecutionJobKind,
        class_permit: OwnedSemaphorePermit,
        observation: RuntimeObservationHandle,
        build_future: B,
    ) -> Result<T>
    where
        B: FnOnce(RuntimeStreamLeaseScope) -> F + Send + 'static,
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let job = self.new_job_observation(kind, observation);
        // A submission from a worker of this same pool runs inline on that
        // worker. Spawning would wait for a second seat while the caller
        // keeps its own, which deadlocks at width one and otherwise spends a
        // worker on waiting. The class permit is still held so the inline
        // run counts against its class like any other job.
        if current_pool_id() == Some(self.inner.id) {
            let _class_permit = class_permit;
            let worker_id = job.started(current_worker_index());
            let mut terminal = JobTerminalGuard::inline(Arc::clone(&job));
            let scope = RuntimeStreamLeaseScope::default().with_gpu_worker(job.job_id.clone(), worker_id);
            return match AssertUnwindSafe(build_future(scope)).catch_unwind().await {
                Ok(value) => {
                    terminal.terminal(GpuJobLifecycleState::Finished);
                    Ok(value)
                },
                Err(_panic) => {
                    terminal.terminal(GpuJobLifecycleState::Panicked);
                    Err(job_panic_error(self.inner.device_ordinal, kind))
                },
            };
        }

        // The class permit is held before the job enters the worker runtime,
        // so bounded child work can never consume a future root slot while it
        // waits. Once accepted, a job may wait only behind the fixed-width
        // worker semaphore; the number of such accepted jobs is bounded by
        // its class capacity.
        let job_guard = JobCounterGuard::new(
            kind,
            Arc::clone(&self.inner.counters),
            Arc::clone(&self.inner.drain),
            self.inner.query_tracker.clone(),
            Arc::clone(&job),
        );
        let seats = self.inner.worker_seats();
        let resource_epoch = self.resource_epoch.clone();
        let task = self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _class_permit = class_permit;
            let seat = seats.occupy(kind).await;
            run_counted_job_scoped(job_guard, seat, build_future).await
        })?;

        match AbortOnDropJoinHandle::new(task, Arc::clone(&job)).await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(_panic)) => {
                job.terminal(GpuJobLifecycleState::Panicked);
                Err(job_panic_error(self.inner.device_ordinal, kind))
            },
            Err(error) if error.is_cancelled() => {
                job.cancellation_requested();
                job.terminal(GpuJobLifecycleState::Cancelled);
                Err(job_cancelled_error(self.inner.device_ordinal, kind, error))
            },
            Err(error) => {
                job.terminal(GpuJobLifecycleState::Failed { error_code: QueryErrorCode::Internal.to_string() });
                Err(job_join_panic_error(self.inner.device_ordinal, kind, error))
            },
        }
    }

    /// Detaches one job onto the device pool.
    ///
    /// This is intended for cleanup that must not block a serving worker. The
    /// pool still accounts for the job, and its RAII counters are released if
    /// it completes, panics, or is cancelled during runtime shutdown.
    #[cfg(test)]
    pub(crate) fn spawn<F>(&self, kind: GpuExecutionJobKind, future: F) -> Result<()>
    where F: Future<Output = ()> + Send + 'static {
        self.spawn_observed(kind, RuntimeObservationHandle::disabled(), future)
    }

    /// Detaches one observed job onto the device pool.
    #[cfg(test)]
    pub(crate) fn spawn_observed<F>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        future: F,
    ) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.check_resource_epoch(RUN_OPERATION)?;
        let class_permit = Arc::clone(self.class_semaphore(kind))
            .try_acquire_owned()
            .map_err(|_error| self.class_saturated_error(kind))?;
        let job = self.new_job_observation(kind, observation);
        let job_guard = JobCounterGuard::new(
            kind,
            Arc::clone(&self.inner.counters),
            Arc::clone(&self.inner.drain),
            self.inner.query_tracker.clone(),
            job,
        );
        let seats = self.inner.worker_seats();
        let resource_epoch = self.resource_epoch.clone();
        self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _class_permit = class_permit;
            let seat = seats.occupy(kind).await;
            let _ = run_counted_job(job_guard, seat, future).await;
        })?;
        Ok(())
    }

    /// Detaches one observed job and invokes `completion` after its terminal
    /// job event, including cancellation or panic cleanup.
    #[cfg(test)]
    pub(crate) fn spawn_observed_with_completion<F, C>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        future: F,
        completion: C,
    ) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        self.check_resource_epoch(RUN_OPERATION)?;
        let class_permit = Arc::clone(self.class_semaphore(kind))
            .try_acquire_owned()
            .map_err(|_error| self.class_saturated_error(kind))?;
        let job = self.new_job_observation(kind, observation);
        let job_guard = JobCounterGuard::new_with_completion(
            kind,
            Arc::clone(&self.inner.counters),
            Arc::clone(&self.inner.drain),
            self.inner.query_tracker.clone(),
            job,
            Box::new(completion),
        );
        let seats = self.inner.worker_seats();
        let resource_epoch = self.resource_epoch.clone();
        self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _class_permit = class_permit;
            let seat = seats.occupy(kind).await;
            let _ = run_counted_job(job_guard, seat, future).await;
        })?;
        Ok(())
    }

    /// Detaches one admission-owned CUDA cleanup task onto this device pool.
    ///
    /// A live admitted attempt has exactly one release path, so this uses a
    /// dedicated `A_d`-sized permit set rather than waiting in an unbounded
    /// semaphore queue or competing with arbitrary root submissions. The
    /// cleanup still waits only behind the fixed-width device runtime.
    pub(crate) fn spawn_release_cleanup_observed_with_completion<F, C>(
        &self,
        observation: RuntimeObservationHandle,
        future: F,
        completion: C,
    ) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        self.check_resource_epoch(RUN_OPERATION)?;
        let release_permit = Arc::clone(&self.inner.release_cleanup_permits)
            .try_acquire_owned()
            .map_err(|_error| self.release_cleanup_saturated_error())?;
        let kind = GpuExecutionJobKind::Cleanup;
        let job = self.new_job_observation(kind, observation);
        let job_guard = JobCounterGuard::new_with_completion(
            kind,
            Arc::clone(&self.inner.counters),
            Arc::clone(&self.inner.drain),
            self.inner.query_tracker.clone(),
            job,
            Box::new(completion),
        );
        let seats = self.inner.worker_seats();
        let resource_epoch = self.resource_epoch.clone();
        self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _release_permit = release_permit;
            let seat = seats.occupy(kind).await;
            let _ = run_counted_job(job_guard, seat, future).await;
        })?;
        Ok(())
    }

    /// Submits one blocking worker onto the device pool without occupying an
    /// async worker thread.
    ///
    /// Blocking source reads and other host-bound device work run on tokio's
    /// blocking threads, never on the fixed-width async workers, so serving
    /// workers cannot starve behind a decode. Admission still rides the same
    /// bounded class permit as every other submission: saturation fails fast
    /// with a typed error instead of queueing unbounded waiters, and at most
    /// `execution_job_capacity` blocking workers run concurrently, so no
    /// second arbitrarily-sized thread pool appears behind this entry.
    ///
    /// The returned handle holds no pool permits. A coordinator blocked in
    /// [`PoolBlockingWork::join`] therefore cannot starve the worker it waits
    /// for. Capacity is released before completion is delivered, so a resolved
    /// join implies the slot is reusable. Pool shutdown drops the completion
    /// sender and the join reports a typed cancellation instead of hanging.
    pub(crate) fn submit_blocking<F, T>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        work: F,
    ) -> Result<PoolBlockingWork<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.submit_blocking_inner(kind, observation, work, || {})
    }

    /// Submits blocking work with a terminal notification, including panic.
    /// The callback runs after the outcome can be joined and does not hold a
    /// pool slot; a coordinator can use it to receive workers in finish order.
    pub(crate) fn submit_blocking_with_completion<F, T, C>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        work: F,
        completion: C,
    ) -> Result<PoolBlockingWork<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        self.submit_blocking_inner(kind, observation, work, completion)
    }

    fn submit_blocking_inner<F, T, C>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        work: F,
        completion: C,
    ) -> Result<PoolBlockingWork<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        self.check_resource_epoch(RUN_OPERATION)?;
        let class = kind.job_class();
        let class_permit = Arc::clone(self.class_semaphore_for_class(class))
            .try_acquire_owned()
            .map_err(|_error| self.class_saturated_error_for_class(kind, class))?;
        let job = self.new_job_observation(kind, observation);
        let guard = JobCounterGuard::new(
            kind,
            Arc::clone(&self.inner.counters),
            Arc::clone(&self.inner.drain),
            self.inner.query_tracker.clone(),
            Arc::clone(&job),
        );
        let (sender, receiver) = mpsc::channel();
        let device_ordinal = self.inner.device_ordinal;
        let resource_epoch = self.resource_epoch.clone();
        {
            let _submission = self.inner.submission_gate.lock().map_err(|_| {
                QueryError::new(QueryErrorCode::Internal, RUN_OPERATION, "GPU execution submission gate was poisoned")
                    .with_fact("device_ordinal", device_ordinal)
                    .with_fact("reason", "gpu_execution_pool_submission_gate_poisoned")
            })?;
            if self.inner.submissions_closed.load(Ordering::Acquire) {
                return Err(pool_closed_error(device_ordinal));
            }
            let completion = BlockingCompletionNotify::new(completion);
            self.inner.handle.spawn_blocking(move || {
                let mut completion = completion;
                let _resource_epoch = resource_epoch;
                let mut guard = guard;
                let _scope = guard.activate();
                let outcome = match std::panic::catch_unwind(AssertUnwindSafe(work)) {
                    Ok(value) => {
                        guard.terminal.terminal(GpuJobLifecycleState::Finished);
                        BlockingWorkOutcome::Completed(value)
                    },
                    Err(_) => {
                        guard.terminal.terminal(GpuJobLifecycleState::Panicked);
                        BlockingWorkOutcome::Panicked
                    },
                };
                // The class slot, resource epoch, and job counters must be free
                // before the completion reaches the waiter, so a resolved join
                // implies a reusable slot rather than one still charged to this
                // worker.
                drop(class_permit);
                drop(_resource_epoch);
                drop(guard);
                let _ = sender.send(outcome);
                completion.notify();
            });
        }
        Ok(PoolBlockingWork { receiver, job, device_ordinal, kind })
    }

    /// Allocates the job's identity and records it as enqueued. Job ids are
    /// unique per pool instance, so a backend generation change yields fresh
    /// ids rather than continuing the previous pool's sequence.
    fn new_job_observation(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
    ) -> Arc<JobObservation> {
        let sequence = self.inner.next_job_id.fetch_add(1, Ordering::Relaxed);
        let job = Arc::new(JobObservation {
            observation,
            job_id: RuntimeJobId::new(format!("gpu-pool-{}-job-{sequence}", self.inner.id)),
            job_kind: kind,
            device_ordinal: u32::try_from(self.inner.device_ordinal).unwrap_or(u32::MAX),
            lifecycle: AtomicU8::new(JOB_LIFECYCLE_PENDING),
        });
        job.enqueued();
        job
    }

    /// Returns device-local execution counters.
    pub(crate) fn snapshot(&self) -> GpuExecutionPoolSnapshot {
        let counters = &self.inner.counters;
        let active_jobs = counters.active.load(Ordering::Acquire);
        let queued_jobs = counters.queued.load(Ordering::Acquire);
        let active_query_jobs = counters.active_queries.load(Ordering::Acquire);
        GpuExecutionPoolSnapshot {
            pool_id: self.inner.id,
            device_ordinal: self.inner.device_ordinal,
            worker_count: self.inner.worker_count,
            root_job_capacity: self.inner.lifecycle_job_capacity,
            aux_job_capacity: self.inner.execution_job_capacity,
            root_jobs_available: self.inner.lifecycle_job_permits.available_permits(),
            aux_jobs_available: self.inner.execution_job_permits.available_permits(),
            active_jobs,
            peak_active_jobs: counters.peak_active.load(Ordering::Acquire).max(active_jobs),
            queued_jobs,
            peak_queued_jobs: counters.peak_queued.load(Ordering::Acquire).max(queued_jobs),
            active_query_jobs,
            peak_active_query_jobs: counters.peak_active_queries.load(Ordering::Acquire).max(active_query_jobs),
        }
    }

    /// Returns the aggregate query tracker attached to this pool.
    #[cfg(test)]
    pub(crate) fn query_tracker(&self) -> GpuExecutionQueryTracker {
        self.inner.query_tracker.clone()
    }

    /// Prevents every later submission from being accepted by this pool.
    ///
    /// The submission gate makes the close transition linearizable with task
    /// spawning: a task either enters the runtime before this call wins or is
    /// rejected as closed.
    pub(crate) fn close_submissions(&self) -> Result<()> {
        self.inner.close_submissions()
    }

    /// Asks the owning supervisor to drop its runtime and cancel any async work
    /// that did not drain after submissions were closed.
    pub(crate) fn request_supervisor_exit(&self) -> Result<()> {
        self.inner.request_supervisor_exit()
    }

    /// Waits for every accepted job guard to leave the queued or active state.
    pub(crate) fn wait_for_drain_until(&self, deadline: Instant) -> Result<()> {
        self.inner.wait_for_drain_until(deadline)
    }

    /// Waits for the runtime-owning supervisor to publish its terminal state.
    pub(crate) fn wait_for_supervisor_exit_until(&self, deadline: Instant) -> Result<()> {
        self.inner.supervisor.wait_for_exit_until(deadline)
    }

    /// Reclaims the exited supervisor thread and reports join failure as a
    /// permanent cleanup failure.
    pub(crate) fn join_supervisor(&self) -> Result<()> {
        self.inner.supervisor.join()
    }

    /// Compatibility composition for cleanup coordinators that have not yet
    /// split worker-pool cleanup into independently observed phases.
    pub(crate) fn close_and_wait_until(&self, deadline: Instant) -> Result<()> {
        self.close_submissions()?;
        self.request_supervisor_exit()?;
        let drain_result = self.wait_for_drain_until(deadline);
        let exit_result = self.wait_for_supervisor_exit_until(deadline);
        let exit_observed = match &exit_result {
            Ok(()) => true,
            Err(error) => matches!(
                error.fact_value("reason"),
                Some("runtime_supervisor_panicked" | "runtime_supervisor_build_failed")
            ),
        };
        let join_result = if exit_observed { self.join_supervisor() } else { Ok(()) };

        let mut first_deadline = None;
        for result in [drain_result, exit_result, join_result] {
            match result {
                Ok(()) => {},
                Err(error) if error.is_cleanup_deadline_elapsed() => {
                    if first_deadline.is_none() {
                        first_deadline = Some(error);
                    }
                },
                Err(error) => return Err(error),
            }
        }
        first_deadline.map_or(Ok(()), Err)
    }

    /// Terminal process-exit path. It has no timeout because returning while
    /// the Tokio runtime can still execute a CUDA-owning job would hand that
    /// job to libcudart teardown.
    fn close_and_join_for_process_exit(&self) {
        // Use the same gate as ordinary shutdown. Otherwise a submitter that
        // already checked `submissions_closed` could still enqueue a task
        // after this terminal path starts draining the runtime.
        let submission = self.inner.submission_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.inner.submissions_closed.store(true, Ordering::Release);
        // Rejected submitters take the gate to report closure; holding it
        // through the drain below would block them behind running jobs.
        drop(submission);
        self.inner.supervisor.request_exit_from_drop();
        let mut state = self.inner.drain.changed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while self.inner.counters.active.load(Ordering::Acquire) != 0
            || self.inner.counters.queued.load(Ordering::Acquire) != 0
        {
            state = self.inner.drain.wake.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        // Job guards take the drain lock in their `Drop`; the supervisor join
        // below waits for those drops to finish.
        drop(state);
        self.inner.supervisor.join_for_process_exit();
    }

    /// Returns a clone whose submissions are additionally bound to
    /// `resource_epoch`. Admitted contexts use this so jobs they submit fail
    /// once their query service closes, while the device pool itself can
    /// outlive that service.
    pub(crate) fn with_resource_epoch(&self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        Self { inner: Arc::clone(&self.inner), resource_epoch: Some(resource_epoch) }
    }

    /// Rejects a closing pool or a closed epoch before any permit is taken.
    ///
    /// Both reads are unsynchronized. Pool closure is re-checked under the
    /// submission gate (`spawn_task` and the blocking submission path), which
    /// is what makes "closed" authoritative. The epoch is checked only here;
    /// the spawned task then retains it, so an epoch that closes after this
    /// point waits for the job to finish instead of cancelling it.
    fn check_resource_epoch(&self, operation: &'static str) -> Result<()> {
        if self.inner.submissions_closed.load(Ordering::Acquire) {
            return Err(QueryError::new(QueryErrorCode::Cancelled, operation, "GPU execution pool is closing")
                .with_fact("device_ordinal", self.inner.device_ordinal)
                .with_fact("reason", "gpu_execution_pool_closing"));
        }
        self.resource_epoch.as_ref().map_or(Ok(()), |epoch| epoch.check_open(operation))
    }
}

impl std::fmt::Debug for GpuExecutionPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("GpuExecutionPool").field("snapshot", &self.snapshot()).finish()
    }
}

#[cfg(test)]
mod tests;
