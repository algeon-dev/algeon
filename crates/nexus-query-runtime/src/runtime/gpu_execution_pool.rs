//! Dedicated asynchronous executors for synchronous GPU work.
//!
//! GPU calls are commonly synchronous from Rust's point of view. Running them
//! on a serving runtime therefore pins serving workers until the GPU call
//! returns. A `GpuExecutionPool` gives each device a separately sized Tokio
//! runtime and keeps that blocking work away from the serving runtime.

use std::{
    any::Any,
    cell::Cell,
    collections::HashMap,
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    thread,
    time::Instant,
};

use futures::FutureExt;
use nexus_query_planning::error::{QueryError, QueryErrorCode, Result};
use serde::Serialize;
use tokio::{
    runtime::Handle,
    sync::{OwnedSemaphorePermit, Semaphore},
    task::{JoinError, JoinHandle},
};

use crate::{
    observability::{
        GpuJobLifecycleState, JobDefinitionRow, RuntimeGpuJobObservation, RuntimeJobId, RuntimeObservationHandle,
        RuntimeObservationPayload, RuntimeStreamLeaseScope, RuntimeTopologyRecord, RuntimeWorkerId,
        WorkerDefinitionRow,
    },
    runtime::{
        context::QueryServiceResourceEpoch,
        process_exit::{
            ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
        },
    },
};

const BUILD_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.build";
const RUN_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.run";
const SHARED_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.shared";
const SHUTDOWN_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.shutdown";

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
    /// Foreground query execution.
    QueryExecution,
    /// Resource synchronization and destruction detached from request serving.
    Cleanup,
}

/// Resource class partitioning for bounded worker admission.
///
/// For active-attempt limit `A_d` and per-attempt source-work bound `S_d`,
/// each device reserves `A_d` lifecycle job slots and `A_d * S_d` execution
/// job slots. Additional fragment setup is execution work. Child work can
/// never consume a future lifecycle slot: class permits are acquired before a
/// job occupies a worker thread, and no hidden fallback pool exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuExecutionJobClass {
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
            Self::Cleanup => "cleanup",
        }
    }

    /// Resource class admitted for this job kind.
    #[doc(hidden)]
    pub const fn job_class(self) -> GpuExecutionJobClass {
        match self {
            Self::ContextSetup | Self::Cleanup => GpuExecutionJobClass::Lifecycle,
            Self::QueryExecution => GpuExecutionJobClass::Execution,
        }
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
    pub peak_active_jobs: usize,
    /// Accepted jobs that have not started.
    pub queued_jobs: usize,
    /// Highest observed value of [`Self::queued_jobs`].
    pub peak_queued_jobs: usize,
    /// Active query-execution jobs on this device.
    pub active_query_jobs: usize,
    /// Highest observed device-local query-execution concurrency.
    pub peak_active_query_jobs: usize,
    /// Configured lifecycle-class job slots (`root` diagnostic label).
    pub root_job_capacity: usize,
    /// Configured execution-class job slots (`auxiliary` diagnostic label).
    pub aux_job_capacity: usize,
    /// Lifecycle-class job slots currently free (`root` diagnostic label).
    pub root_jobs_available: usize,
    /// Execution-class job slots currently free (`auxiliary` diagnostic label).
    pub aux_jobs_available: usize,
}

/// Point-in-time counters shared by a group of GPU execution pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GpuExecutionQuerySnapshot {
    /// Query-execution jobs active across all attached pools.
    pub active_query_jobs: usize,
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
#[doc(hidden)]
pub struct GpuExecutionQueryTracker {
    counters: Arc<QueryCounters>,
}

impl GpuExecutionQueryTracker {
    /// Creates an empty aggregate query tracker.
    #[doc(hidden)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current and peak query concurrency across attached pools.
    #[doc(hidden)]
    pub fn snapshot(&self) -> GpuExecutionQuerySnapshot {
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

/// Opaque lifecycle-bound capability for synchronous GPU work.
///
/// This lease deliberately exposes execution but not pool construction,
/// detached spawning, shutdown, counters, or raw resource-epoch access.
#[derive(Clone)]
pub struct GpuExecutorLease {
    pool: GpuExecutionPool,
}

impl GpuExecutorLease {
    #[doc(hidden)]
    pub fn from_pool(pool: GpuExecutionPool) -> Self {
        Self { pool }
    }

    /// Runs one job on the lease's dedicated device executor.
    pub async fn run<F, T>(&self, kind: GpuExecutionJobKind, future: F) -> Result<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.pool.run(kind, future).await
    }

    /// Runs one observed job and provides explicit job/worker correlation to
    /// work that may acquire a service-stream lease.
    pub fn run_observed_scoped<B, F, T>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        build_future: B,
    ) -> impl Future<Output = Result<T>> + Send + 'static + use<B, F, T>
    where
        B: FnOnce(RuntimeStreamLeaseScope) -> F + Send + 'static,
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.pool.clone().run_observed_scoped(kind, observation, build_future)
    }
}

impl std::fmt::Debug for GpuExecutorLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("GpuExecutorLease").finish_non_exhaustive()
    }
}

/// Fixed-width, per-device executor for work that can block on the GPU.
#[derive(Clone)]
#[doc(hidden)]
pub struct GpuExecutionPool {
    inner: Arc<GpuExecutionPoolInner>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
}

impl GpuExecutionPool {
    /// Creates a pool with an independent query-concurrency tracker.
    #[doc(hidden)]
    pub fn new(device_ordinal: usize, worker_count: usize) -> Result<Self> {
        Self::new_with_tracker(device_ordinal, worker_count, GpuExecutionQueryTracker::new())
    }

    /// Creates a pool attached to a caller-provided aggregate query tracker.
    ///
    /// Direct constructors use the documented default class partition:
    /// `worker_count` lifecycle job slots and `worker_count` execution job slots.
    /// Backend-owned pools pass explicit `A_d`/`A_d * S_d` capacities from
    /// the device resource profile via [`Self::new_with_tracker_and_classes`].
    #[doc(hidden)]
    pub fn new_with_tracker(
        device_ordinal: usize,
        worker_count: usize,
        query_tracker: GpuExecutionQueryTracker,
    ) -> Result<Self> {
        Self::new_with_tracker_and_classes(device_ordinal, worker_count, worker_count, worker_count, query_tracker)
    }

    /// Creates a pool with explicit lifecycle/execution job-slot capacities.
    #[doc(hidden)]
    pub fn new_with_tracker_and_classes(
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
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.execution_pool",
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
            "nexus_query_engine.gpu.execution_pool",
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
            "nexus_query_engine.gpu.execution_pool",
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
    #[doc(hidden)]
    pub fn shared_for_device(device_ordinal: usize, worker_count: usize) -> Result<Self> {
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
    pub async fn run_observed<F, T>(
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
    #[doc(hidden)]
    pub async fn run_observed_scoped_in_class<B, F, T>(
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
        let class_permit = self
            .class_semaphore_for_class(class)
            .clone()
            .try_acquire_owned()
            .map_err(|_error| self.class_saturated_error_for_class(kind, class))?;
        let job = self.new_job_observation(kind, observation);
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
        let concurrency = Arc::clone(&self.inner.concurrency);
        let resource_epoch = self.resource_epoch.clone();
        let task = self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _class_permit = class_permit;
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            run_counted_job_scoped(job_guard, permit, build_future).await
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
    pub fn spawn<F>(&self, kind: GpuExecutionJobKind, future: F) -> Result<()>
    where F: Future<Output = ()> + Send + 'static {
        self.spawn_observed(kind, RuntimeObservationHandle::disabled(), future)
    }

    /// Detaches one observed job onto the device pool.
    #[cfg(test)]
    pub fn spawn_observed<F>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        future: F,
    ) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.check_resource_epoch(RUN_OPERATION)?;
        let class_permit = self
            .class_semaphore(kind)
            .clone()
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
        let concurrency = Arc::clone(&self.inner.concurrency);
        let resource_epoch = self.resource_epoch.clone();
        self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _class_permit = class_permit;
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            let _ = run_counted_job(job_guard, permit, future).await;
        })?;
        Ok(())
    }

    /// Detaches one observed job and invokes `completion` after its terminal
    /// job event, including cancellation or panic cleanup.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn spawn_observed_with_completion<F, C>(
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
        let class_permit = self
            .class_semaphore(kind)
            .clone()
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
        let concurrency = Arc::clone(&self.inner.concurrency);
        let resource_epoch = self.resource_epoch.clone();
        self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _class_permit = class_permit;
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            let _ = run_counted_job(job_guard, permit, future).await;
        })?;
        Ok(())
    }

    /// Detaches one admission-owned CUDA cleanup task onto this device pool.
    ///
    /// A live admitted attempt has exactly one release path, so this uses a
    /// dedicated `A_d`-sized permit set rather than waiting in an unbounded
    /// semaphore queue or competing with arbitrary root submissions. The
    /// cleanup still waits only behind the fixed-width device runtime.
    #[doc(hidden)]
    pub fn spawn_release_cleanup_observed_with_completion<F, C>(
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
        let release_permit = self
            .inner
            .release_cleanup_permits
            .clone()
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
        let concurrency = Arc::clone(&self.inner.concurrency);
        let resource_epoch = self.resource_epoch.clone();
        self.inner.spawn_task(async move {
            let _resource_epoch = resource_epoch;
            let _release_permit = release_permit;
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            let _ = run_counted_job(job_guard, permit, future).await;
        })?;
        Ok(())
    }

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
    pub fn snapshot(&self) -> GpuExecutionPoolSnapshot {
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
    pub fn query_tracker(&self) -> GpuExecutionQueryTracker {
        self.inner.query_tracker.clone()
    }

    /// Prevents every later submission from being accepted by this pool.
    ///
    /// The submission gate makes the close transition linearizable with task
    /// spawning: a task either enters the runtime before this call wins or is
    /// rejected as closed.
    #[doc(hidden)]
    pub fn close_submissions(&self) -> Result<()> {
        self.inner.close_submissions()
    }

    /// Asks the owning supervisor to drop its runtime and cancel any async work
    /// that did not drain after submissions were closed.
    #[doc(hidden)]
    pub fn request_supervisor_exit(&self) -> Result<()> {
        self.inner.request_supervisor_exit()
    }

    /// Waits for every accepted job guard to leave the queued or active state.
    #[doc(hidden)]
    pub fn wait_for_drain_until(&self, deadline: Instant) -> Result<()> {
        self.inner.wait_for_drain_until(deadline)
    }

    /// Waits for the runtime-owning supervisor to publish its terminal state.
    #[doc(hidden)]
    pub fn wait_for_supervisor_exit_until(&self, deadline: Instant) -> Result<()> {
        self.inner.supervisor.wait_for_exit_until(deadline)
    }

    /// Reclaims the exited supervisor thread and reports join failure as a
    /// permanent cleanup failure.
    #[doc(hidden)]
    pub fn join_supervisor(&self) -> Result<()> {
        self.inner.supervisor.join()
    }

    /// Compatibility composition for cleanup coordinators that have not yet
    /// split worker-pool cleanup into independently observed phases.
    #[doc(hidden)]
    pub fn close_and_wait_until(&self, deadline: Instant) -> Result<()> {
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
        drop(submission);
        self.inner.supervisor.request_exit_from_drop();
        let mut state = self.inner.drain.changed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while self.inner.counters.active.load(Ordering::Acquire) != 0
            || self.inner.counters.queued.load(Ordering::Acquire) != 0
        {
            state = self.inner.drain.wake.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        drop(state);
        self.inner.supervisor.join_for_process_exit();
    }

    #[doc(hidden)]
    pub fn with_resource_epoch(&self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        Self { inner: Arc::clone(&self.inner), resource_epoch: Some(resource_epoch) }
    }

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

struct GpuExecutionPoolInner {
    id: u64,
    device_ordinal: usize,
    worker_count: usize,
    handle: Handle,
    supervisor: Arc<RuntimeSupervisor>,
    submissions_closed: AtomicBool,
    submission_gate: Mutex<()>,
    concurrency: Arc<Semaphore>,
    lifecycle_job_permits: Arc<Semaphore>,
    execution_job_permits: Arc<Semaphore>,
    release_cleanup_permits: Arc<Semaphore>,
    lifecycle_job_capacity: usize,
    execution_job_capacity: usize,
    counters: Arc<PoolCounters>,
    drain: Arc<PoolDrain>,
    query_tracker: GpuExecutionQueryTracker,
    next_job_id: AtomicU64,
}

impl Drop for GpuExecutionPoolInner {
    fn drop(&mut self) {
        self.submissions_closed.store(true, Ordering::Release);
        self.supervisor.request_exit_from_drop();
    }
}

impl GpuExecutionPoolInner {
    fn spawn_task<F>(&self, future: F) -> Result<JoinHandle<F::Output>>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let _submission = self.submission_gate.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, RUN_OPERATION, "GPU execution submission gate was poisoned")
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "gpu_execution_pool_submission_gate_poisoned")
        })?;
        if self.submissions_closed.load(Ordering::Acquire) {
            return Err(pool_closed_error(self.device_ordinal));
        }
        Ok(self.handle.spawn(future))
    }

    fn close_submissions(&self) -> Result<()> {
        let _submission = self.submission_gate.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, SHUTDOWN_OPERATION, "GPU execution submission gate was poisoned")
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "gpu_execution_pool_submission_gate_poisoned")
        })?;
        self.submissions_closed.store(true, Ordering::Release);
        Ok(())
    }

    fn request_supervisor_exit(&self) -> Result<()> {
        if !self.submissions_closed.load(Ordering::Acquire) {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution supervisor exit was requested before submissions were closed",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "gpu_execution_pool_exit_before_close"));
        }
        self.supervisor.request_exit()
    }

    fn wait_for_drain_until(&self, deadline: Instant) -> Result<()> {
        self.drain.wait_until(deadline, self.device_ordinal, self.counters.as_ref())
    }
}

fn pool_closed_error(device_ordinal: usize) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, RUN_OPERATION, "GPU execution pool is closing")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("reason", "gpu_execution_pool_closing")
}

fn pool_drain_timeout(device_ordinal: usize, counters: &PoolCounters) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        SHUTDOWN_OPERATION,
        "timed out waiting for accepted GPU execution jobs to drain",
    )
    .with_fact("device_ordinal", device_ordinal)
    .with_fact("active_jobs", counters.active.load(Ordering::Acquire))
    .with_fact("queued_jobs", counters.queued.load(Ordering::Acquire))
    .with_fact("reason", "gpu_execution_pool_drain_timeout")
    .with_cleanup_deadline_elapsed()
}

#[derive(Debug, Default)]
struct PoolDrain {
    changed: Mutex<()>,
    wake: Condvar,
}

impl PoolDrain {
    fn wait_until(&self, deadline: Instant, device_ordinal: usize, counters: &PoolCounters) -> Result<()> {
        let mut state =
            crate::runtime::deadline::mutex_lock_until(&self.changed, deadline).map_err(|error| match error {
                crate::runtime::deadline::DeadlineMutexLockError::Poisoned => QueryError::new(
                    QueryErrorCode::Internal,
                    SHUTDOWN_OPERATION,
                    "GPU execution pool drain mutex was poisoned",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("reason", "gpu_execution_pool_drain_poisoned"),
                crate::runtime::deadline::DeadlineMutexLockError::TimedOut => {
                    pool_drain_timeout(device_ordinal, counters)
                },
            })?;
        while counters.active.load(Ordering::Acquire) != 0 || counters.queued.load(Ordering::Acquire) != 0 {
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return Err(pool_drain_timeout(device_ordinal, counters));
            };
            let (next, timeout) = self.wake.wait_timeout(state, remaining).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    SHUTDOWN_OPERATION,
                    "GPU execution pool drain condition variable was poisoned",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("reason", "gpu_execution_pool_drain_poisoned")
            })?;
            state = next;
            if timeout.timed_out()
                && (counters.active.load(Ordering::Acquire) != 0 || counters.queued.load(Ordering::Acquire) != 0)
            {
                return Err(pool_drain_timeout(device_ordinal, counters));
            }
        }
        Ok(())
    }
}

struct RuntimeSupervisor {
    device_ordinal: usize,
    control: Arc<RuntimeSupervisorControl>,
    join: Mutex<Option<thread::JoinHandle<()>>>,
    join_disposition: AtomicU8,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

/// Keeps only a weak pool pointer plus the runtime-owning supervisor alive for
/// process teardown. The registry does not retain query state or CUDA objects;
/// the supervisor is enough to request runtime shutdown and join every Tokio
/// worker if the final ordinary pool reference vanished on one of those
/// workers.
struct ProcessExitGpuExecutionPool {
    inner: Weak<GpuExecutionPoolInner>,
    supervisor: Arc<RuntimeSupervisor>,
}

const SUPERVISOR_NOT_JOINED: u8 = 0;
const SUPERVISOR_JOINED: u8 = 1;
const SUPERVISOR_JOIN_FAILED: u8 = 2;

#[derive(Debug, Default)]
struct RuntimeSupervisorControl {
    state: Mutex<RuntimeSupervisorState>,
    changed: Condvar,
    #[cfg(test)]
    panic_during_teardown: AtomicBool,
    #[cfg(test)]
    panic_after_exit: AtomicBool,
}

#[derive(Debug, Default)]
struct RuntimeSupervisorState {
    exit_requested: bool,
    exit: Option<RuntimeSupervisorExit>,
}

#[derive(Debug, Clone)]
enum RuntimeSupervisorExit {
    Clean,
    Panicked { panic_message: String },
    BuildFailed { message: String },
}

enum RuntimeSupervisorStartup {
    Ready(Handle),
    BuildFailed(String),
}

impl RuntimeSupervisor {
    fn start(
        device_ordinal: usize,
        worker_count: usize,
        pool_id: u64,
        next_worker_index: Arc<AtomicUsize>,
    ) -> Result<(Handle, Arc<Self>)> {
        let control = Arc::new(RuntimeSupervisorControl::default());
        let supervisor = Arc::new(Self {
            device_ordinal,
            control,
            join: Mutex::new(None),
            join_disposition: AtomicU8::new(SUPERVISOR_NOT_JOINED),
            process_exit_registration: Mutex::new(None),
        });
        let thread_control = Arc::clone(&supervisor.control);
        let worker_supervisor = Arc::clone(&supervisor);
        let (startup_sender, startup_receiver) = mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name(format!("nexus-gpu-{device_ordinal}-supervisor"))
            .spawn(move || {
                (|| {
                    let runtime = tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(worker_count)
                        .thread_name(format!("nexus-gpu-{device_ordinal}"))
                        .on_thread_start({
                            let next_worker_index = Arc::clone(&next_worker_index);
                            move || {
                                CURRENT_POOL_ID.with(|current| current.set(Some(pool_id)));
                                CURRENT_WORKER_INDEX.with(|current| {
                                    current.set(Some(next_worker_index.fetch_add(1, Ordering::Relaxed)));
                                });
                            }
                        })
                        .on_thread_stop(|| {
                            CURRENT_WORKER_INDEX.with(|current| current.set(None));
                            CURRENT_POOL_ID.with(|current| current.set(None));
                        })
                        .enable_all()
                        .build();
                    let runtime = match runtime {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            let message = error.to_string();
                            thread_control
                                .publish_exit(RuntimeSupervisorExit::BuildFailed { message: message.clone() });
                            let _ = startup_sender.send(RuntimeSupervisorStartup::BuildFailed(message));
                            return;
                        },
                    };
                    if startup_sender.send(RuntimeSupervisorStartup::Ready(runtime.handle().clone())).is_err() {
                        drop(runtime);
                        return;
                    }

                    let teardown = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        thread_control.wait_for_exit_request();
                        #[cfg(test)]
                        if thread_control.panic_during_teardown.load(Ordering::Acquire) {
                            panic!("injected GPU execution supervisor teardown panic");
                        }
                        drop(runtime);
                    }));
                    let exit = match teardown {
                        Ok(()) => RuntimeSupervisorExit::Clean,
                        Err(panic) => RuntimeSupervisorExit::Panicked { panic_message: panic_message(panic.as_ref()) },
                    };
                    thread_control.publish_exit(exit);

                    #[cfg(test)]
                    if thread_control.panic_after_exit.load(Ordering::Acquire) {
                        panic!("injected GPU execution supervisor join panic");
                    }
                })();
                worker_supervisor.unregister_process_exit();
            })
            .map_err(|error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "failed to start the dedicated GPU execution runtime supervisor",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("worker_count", worker_count)
                .with_fact("reason", "runtime_supervisor_spawn_failed")
                .with_source(error)
            })?;

        let handle = match startup_receiver.recv() {
            Ok(RuntimeSupervisorStartup::Ready(handle)) => handle,
            Ok(RuntimeSupervisorStartup::BuildFailed(message)) => {
                let join_failed = join.join().err().map(|panic| panic_message(panic.as_ref()));
                let mut error = QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "failed to initialize the dedicated GPU execution runtime",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("worker_count", worker_count)
                .with_fact("reason", "runtime_build_failed")
                .with_fact("runtime_error", message);
                if let Some(join_panic) = join_failed {
                    error = error.with_fact("supervisor_join_panic", join_panic);
                }
                return Err(error);
            },
            Err(error) => {
                let join_failed = join.join().err().map(|panic| panic_message(panic.as_ref()));
                let mut query_error = QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "GPU execution runtime supervisor exited before publishing its handle",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("worker_count", worker_count)
                .with_fact("reason", "runtime_supervisor_startup_disconnected")
                .with_source(error);
                if let Some(join_panic) = join_failed {
                    query_error = query_error.with_fact("supervisor_join_panic", join_panic);
                }
                return Err(query_error);
            },
        };

        *supervisor.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(join);
        Ok((handle, supervisor))
    }

    fn register_process_exit(self: &Arc<Self>, inner: Weak<GpuExecutionPoolInner>) {
        let shutdown: Arc<dyn ProcessExitShutdown> =
            Arc::new(ProcessExitGpuExecutionPool { inner, supervisor: Arc::clone(self) });
        let registration = register_cuda_process_exit_shutdown(shutdown);
        *self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = registration;
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    fn request_exit(&self) -> Result<()> {
        let mut state = self.control.state.lock().map_err(|_| supervisor_state_poisoned(self.device_ordinal))?;
        state.exit_requested = true;
        self.control.changed.notify_all();
        Ok(())
    }

    fn request_exit_from_drop(&self) {
        let mut state = self.control.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.exit_requested = true;
        self.control.changed.notify_all();
    }

    fn wait_for_exit_until(&self, deadline: Instant) -> Result<()> {
        let mut state =
            crate::runtime::deadline::mutex_lock_until(&self.control.state, deadline).map_err(|error| match error {
                crate::runtime::deadline::DeadlineMutexLockError::Poisoned => {
                    supervisor_state_poisoned(self.device_ordinal)
                },
                crate::runtime::deadline::DeadlineMutexLockError::TimedOut => {
                    supervisor_exit_timeout(self.device_ordinal)
                },
            })?;
        while state.exit.is_none() {
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return Err(supervisor_exit_timeout(self.device_ordinal));
            };
            let (next, timeout) = self
                .control
                .changed
                .wait_timeout(state, remaining)
                .map_err(|_| supervisor_state_poisoned(self.device_ordinal))?;
            state = next;
            if timeout.timed_out() && state.exit.is_none() {
                return Err(supervisor_exit_timeout(self.device_ordinal));
            }
        }
        match state.exit.as_ref().expect("exit checked above") {
            RuntimeSupervisorExit::Clean => Ok(()),
            RuntimeSupervisorExit::Panicked { panic_message } => {
                Err(supervisor_panic_error(self.device_ordinal, "runtime_teardown", panic_message))
            },
            RuntimeSupervisorExit::BuildFailed { message } => Err(QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution supervisor retained a runtime build failure",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "runtime_supervisor_build_failed")
            .with_fact("runtime_error", message)),
        }
    }

    fn join(&self) -> Result<()> {
        {
            let state = self.control.state.lock().map_err(|_| supervisor_state_poisoned(self.device_ordinal))?;
            if state.exit.is_none() {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    SHUTDOWN_OPERATION,
                    "GPU execution supervisor cannot be joined before it publishes exit",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "runtime_supervisor_join_before_exit"));
            }
        }

        let mut join = self.join.lock().map_err(|_| {
            QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution supervisor join mutex was poisoned",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "runtime_supervisor_join_poisoned")
        })?;
        if let Some(handle) = join.take() {
            if handle.thread().id() == thread::current().id() {
                *join = Some(handle);
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    SHUTDOWN_OPERATION,
                    "GPU execution supervisor cannot join itself",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "runtime_supervisor_self_join"));
            }
            match handle.join() {
                Ok(()) => self.join_disposition.store(SUPERVISOR_JOINED, Ordering::Release),
                Err(panic) => {
                    self.join_disposition.store(SUPERVISOR_JOIN_FAILED, Ordering::Release);
                    return Err(supervisor_panic_error(
                        self.device_ordinal,
                        "thread_join",
                        &panic_message(panic.as_ref()),
                    ));
                },
            }
        }
        let result = match self.join_disposition.load(Ordering::Acquire) {
            SUPERVISOR_JOINED => Ok(()),
            SUPERVISOR_JOIN_FAILED => Err(QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution supervisor previously failed to join",
            )
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "runtime_supervisor_join_failed")),
            unexpected => unreachable!("invalid GPU execution supervisor join disposition {unexpected}"),
        };
        self.unregister_process_exit();
        result
    }

    fn join_for_process_exit(&self) {
        self.request_exit_from_drop();
        let mut join = self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(handle) = join.take() else {
            return;
        };
        if handle.thread().id() == thread::current().id() {
            *join = Some(handle);
            return;
        }
        let joined = handle.join();
        self.join_disposition
            .store(if joined.is_ok() { SUPERVISOR_JOINED } else { SUPERVISOR_JOIN_FAILED }, Ordering::Release);
        drop(join);
        self.unregister_process_exit();
    }
}

impl ProcessExitShutdown for ProcessExitGpuExecutionPool {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::ExecutionPool
    }

    fn shutdown_for_process_exit(&self) {
        if let Some(inner) = self.inner.upgrade() {
            GpuExecutionPool { inner, resource_epoch: None }.close_and_join_for_process_exit();
        } else {
            self.supervisor.join_for_process_exit();
        }
    }
}

impl RuntimeSupervisorControl {
    fn wait_for_exit_request(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while !state.exit_requested {
            state = self.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn publish_exit(&self, exit: RuntimeSupervisorExit) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.exit = Some(exit);
        self.changed.notify_all();
    }
}

fn supervisor_exit_timeout(device_ordinal: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        SHUTDOWN_OPERATION,
        "timed out waiting for the GPU execution runtime supervisor to exit",
    )
    .with_fact("device_ordinal", device_ordinal)
    .with_fact("reason", "runtime_supervisor_exit_timeout")
    .with_cleanup_deadline_elapsed()
}

fn supervisor_state_poisoned(device_ordinal: usize) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, SHUTDOWN_OPERATION, "GPU execution supervisor state was poisoned")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("reason", "runtime_supervisor_state_poisoned")
}

fn supervisor_panic_error(device_ordinal: usize, stage: &'static str, panic_message: &str) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, SHUTDOWN_OPERATION, "GPU execution runtime supervisor panicked")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("supervisor_stage", stage)
        .with_fact("panic_message", panic_message)
        .with_fact("reason", "runtime_supervisor_panicked")
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    panic
        .downcast_ref::<&'static str>()
        .map(|message| (*message).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

struct JobObservation {
    observation: RuntimeObservationHandle,
    job_id: RuntimeJobId,
    job_kind: GpuExecutionJobKind,
    device_ordinal: u32,
    lifecycle: AtomicU8,
}

const JOB_LIFECYCLE_PENDING: u8 = 0;
const JOB_LIFECYCLE_CANCELLATION_RECORDING: u8 = 1;
const JOB_LIFECYCLE_CANCELLATION_REQUESTED: u8 = 2;
const JOB_LIFECYCLE_TERMINAL: u8 = 3;

impl JobObservation {
    fn enqueued(&self) {
        self.observation.record_topology(crate::observability::RuntimeObservationRecordClass::JobDefinition, || {
            RuntimeTopologyRecord::JobDefinition(JobDefinitionRow {
                plan_execution_id: None,
                job_id: self.job_id.clone(),
                job_kind: self.job_kind.as_str().to_string(),
                device_ordinal: self.device_ordinal,
            })
        });
        self.record(GpuJobLifecycleState::Enqueued, None);
    }

    fn started(&self, worker_index: Option<usize>) -> RuntimeWorkerId {
        let worker_id = RuntimeWorkerId::new(format!(
            "{}-worker-{}",
            self.job_id.as_str(),
            worker_index.map_or_else(|| "inline".to_string(), |index| index.to_string())
        ));
        self.observation.record_topology(crate::observability::RuntimeObservationRecordClass::WorkerDefinition, || {
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
    fn cancellation_requested(&self) -> bool {
        loop {
            match self.lifecycle.load(Ordering::Acquire) {
                JOB_LIFECYCLE_TERMINAL => return false,
                JOB_LIFECYCLE_CANCELLATION_REQUESTED => return true,
                JOB_LIFECYCLE_CANCELLATION_RECORDING => std::hint::spin_loop(),
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

    fn terminal(&self, lifecycle: GpuJobLifecycleState) {
        loop {
            let current = self.lifecycle.load(Ordering::Acquire);
            match current {
                JOB_LIFECYCLE_TERMINAL => return,
                JOB_LIFECYCLE_CANCELLATION_RECORDING => std::hint::spin_loop(),
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

    fn is_terminal(&self) -> bool {
        self.lifecycle.load(Ordering::Acquire) == JOB_LIFECYCLE_TERMINAL
    }

    fn has_cancellation_request(&self) -> bool {
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
struct PoolCounters {
    active: AtomicUsize,
    peak_active: AtomicUsize,
    queued: AtomicUsize,
    peak_queued: AtomicUsize,
    active_queries: AtomicUsize,
    peak_active_queries: AtomicUsize,
}

#[derive(Debug, Default)]
struct QueryCounters {
    active: AtomicUsize,
    peak: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobCounterState {
    Queued,
    Active,
}

struct JobCounterGuard {
    kind: GpuExecutionJobKind,
    counters: Arc<PoolCounters>,
    drain: Arc<PoolDrain>,
    query_tracker: GpuExecutionQueryTracker,
    state: JobCounterState,
    terminal: JobTerminalGuard,
    completion: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl JobCounterGuard {
    fn new(
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

    fn new_with_completion(
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

    fn activate(&mut self) -> RuntimeStreamLeaseScope {
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
            self.completion.take().and_then(|completion| std::panic::catch_unwind(AssertUnwindSafe(completion)).err());

        {
            // Synchronize the zero-counter transition with drain waiters.
            // Completion above releases the job's final resource anchors, so
            // a waiter that observes zero after this decrement may proceed.
            let _drain_state = self.drain.changed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
            std::panic::resume_unwind(panic);
        }
    }
}

struct JobTerminalGuard {
    observation: Arc<JobObservation>,
    cancellation_on_drop: bool,
}

impl JobTerminalGuard {
    fn spawned(observation: Arc<JobObservation>) -> Self {
        Self { observation, cancellation_on_drop: false }
    }

    fn inline(observation: Arc<JobObservation>) -> Self {
        Self { observation, cancellation_on_drop: true }
    }

    fn terminal(&mut self, lifecycle: GpuJobLifecycleState) {
        self.observation.terminal(lifecycle);
    }

    fn finish_drop(&mut self) {
        if self.observation.is_terminal() {
            return;
        }
        if std::thread::panicking() {
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

async fn run_counted_job<F, T>(
    mut guard: JobCounterGuard,
    _permit: OwnedSemaphorePermit,
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

async fn run_counted_job_scoped<B, F, T>(
    mut guard: JobCounterGuard,
    _permit: OwnedSemaphorePermit,
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

fn current_pool_id() -> Option<u64> {
    CURRENT_POOL_ID.with(Cell::get)
}

fn current_worker_index() -> Option<usize> {
    CURRENT_WORKER_INDEX.with(Cell::get)
}

struct AbortOnDropJoinHandle<T> {
    handle: JoinHandle<T>,
    observation: Arc<JobObservation>,
    completed: bool,
}

impl<T> AbortOnDropJoinHandle<T> {
    fn new(handle: JoinHandle<T>, observation: Arc<JobObservation>) -> Self {
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

fn validate_worker_count(device_ordinal: usize, worker_count: usize) -> Result<()> {
    if worker_count == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            BUILD_OPERATION,
            "GPU execution pool worker count must be greater than zero",
        )
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("worker_count", worker_count)
        .with_fact("reason", "zero"));
    }
    Ok(())
}

fn job_panic_error(device_ordinal: usize, kind: GpuExecutionJobKind) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, RUN_OPERATION, "GPU execution pool job panicked")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("job_kind", kind.as_str())
        .with_fact("reason", "job_panicked")
}

fn job_join_panic_error(device_ordinal: usize, kind: GpuExecutionJobKind, error: JoinError) -> QueryError {
    job_panic_error(device_ordinal, kind).with_source(error)
}

fn job_cancelled_error(device_ordinal: usize, kind: GpuExecutionJobKind, error: JoinError) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, RUN_OPERATION, "GPU execution pool job was cancelled")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("job_kind", kind.as_str())
        .with_fact("reason", "job_cancelled")
        .with_source(error)
}

fn increment_with_peak(active: &AtomicUsize, peak: &AtomicUsize) {
    let current = active.fetch_add(1, Ordering::AcqRel).saturating_add(1);
    peak.fetch_max(current, Ordering::AcqRel);
}

fn decrement(counter: &AtomicUsize) {
    let previous = counter.fetch_sub(1, Ordering::AcqRel);
    debug_assert!(previous > 0, "GPU execution pool counter underflow");
}

#[cfg(test)]
mod job_class_tests;
#[cfg(test)]
mod tests;
