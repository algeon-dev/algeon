//! Dedicated asynchronous executors for synchronous GPU work.
//!
//! GPU calls are commonly synchronous from Rust's point of view. Running them
//! on a serving runtime therefore pins serving workers until the GPU call
//! returns. A `GpuExecutionPool` gives each device a separately sized Tokio
//! runtime and keeps that blocking work away from the serving runtime.

use std::{
    cell::Cell,
    collections::HashMap,
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use futures::FutureExt;
use tokio::{
    runtime::{Handle, Runtime},
    sync::{OwnedSemaphorePermit, Semaphore},
    task::{JoinError, JoinHandle},
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    observability::{
        GpuJobLifecycleState, JobDefinitionRow, RuntimeGpuJobObservation, RuntimeJobId, RuntimeObservationHandle,
        RuntimeObservationPayload, RuntimeStreamLeaseScope, RuntimeTopologyRecord, RuntimeWorkerId,
        WorkerDefinitionRow,
    },
    runtime::context::QueryServiceResourceEpoch,
};

const BUILD_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.build";
const RUN_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.run";
const SHARED_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.shared";

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static CURRENT_POOL_ID: Cell<Option<u64>> = const { Cell::new(None) };
    static CURRENT_WORKER_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Classifies work submitted to a [`GpuExecutionPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuExecutionJobKind {
    /// CUDA and execution-context initialization.
    ContextSetup,
    /// Foreground query execution.
    QueryExecution,
    /// Resource synchronization and destruction detached from request serving.
    Cleanup,
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
}

/// Point-in-time counters for one device's GPU execution pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuExecutionPoolSnapshot {
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
}

/// Point-in-time counters shared by a group of GPU execution pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub struct GpuExecutionQueryTracker {
    counters: Arc<QueryCounters>,
}

impl GpuExecutionQueryTracker {
    /// Creates an empty aggregate query tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current and peak query concurrency across attached pools.
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

/// Fixed-width, per-device executor for work that can block on the GPU.
#[derive(Clone)]
pub struct GpuExecutionPool {
    inner: Arc<GpuExecutionPoolInner>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
}

impl GpuExecutionPool {
    /// Creates a pool with an independent query-concurrency tracker.
    pub fn new(device_ordinal: usize, worker_count: usize) -> Result<Self> {
        Self::new_with_tracker(device_ordinal, worker_count, GpuExecutionQueryTracker::new())
    }

    /// Creates a pool attached to a caller-provided aggregate query tracker.
    pub fn new_with_tracker(
        device_ordinal: usize,
        worker_count: usize,
        query_tracker: GpuExecutionQueryTracker,
    ) -> Result<Self> {
        validate_worker_count(device_ordinal, worker_count)?;

        let id = NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed);
        let next_worker_index = Arc::new(AtomicUsize::new(0));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_count)
            .thread_name(format!("nexus-gpu-{device_ordinal}"))
            .on_thread_start({
                let next_worker_index = Arc::clone(&next_worker_index);
                move || {
                    CURRENT_POOL_ID.with(|current| current.set(Some(id)));
                    CURRENT_WORKER_INDEX
                        .with(|current| current.set(Some(next_worker_index.fetch_add(1, Ordering::Relaxed))));
                }
            })
            .on_thread_stop(move || {
                CURRENT_WORKER_INDEX.with(|current| current.set(None));
                CURRENT_POOL_ID.with(|current| current.set(None));
            })
            .enable_all()
            .build()
            .map_err(|error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "failed to initialize the dedicated GPU execution runtime",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("worker_count", worker_count)
                .with_fact("reason", "runtime_build_failed")
                .with_source(error)
            })?;
        let handle = runtime.handle().clone();

        Ok(Self {
            inner: Arc::new(GpuExecutionPoolInner {
                id,
                device_ordinal,
                worker_count,
                handle,
                runtime: Some(runtime),
                concurrency: Arc::new(Semaphore::new(worker_count)),
                counters: Arc::new(PoolCounters::default()),
                query_tracker,
                next_job_id: AtomicU64::new(1),
            }),
            resource_epoch: None,
        })
    }

    /// Returns the process-global fallback pool for one CUDA device.
    ///
    /// A live fallback pool fixes the worker count for its device. Requesting a
    /// different width is rejected rather than silently constructing another
    /// executor for the same device. Once the last clone is dropped, a later
    /// call may install a new width.
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
        self.run_observed_scoped(kind, observation, move |_scope| future).await
    }

    /// Runs one observed job and passes its explicit job/worker correlation to
    /// work that may acquire a service-stream lease.
    pub async fn run_observed_scoped<B, F, T>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        build_future: B,
    ) -> Result<T>
    where
        B: FnOnce(RuntimeStreamLeaseScope) -> F + Send + 'static,
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.check_resource_epoch(RUN_OPERATION)?;
        let job = self.new_job_observation(kind, observation);
        if current_pool_id() == Some(self.inner.id) {
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

        let job_guard = JobCounterGuard::new(
            kind,
            Arc::clone(&self.inner.counters),
            self.inner.query_tracker.clone(),
            Arc::clone(&job),
        );
        let concurrency = Arc::clone(&self.inner.concurrency);
        let resource_epoch = self.resource_epoch.clone();
        let task = self.inner.handle.spawn(async move {
            let _resource_epoch = resource_epoch;
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            run_counted_job_scoped(job_guard, permit, build_future).await
        });

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
    pub fn spawn<F>(&self, kind: GpuExecutionJobKind, future: F) -> Result<()>
    where F: Future<Output = ()> + Send + 'static {
        self.spawn_observed(kind, RuntimeObservationHandle::disabled(), future)
    }

    /// Detaches one observed job onto the device pool.
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
        let job = self.new_job_observation(kind, observation);
        let job_guard =
            JobCounterGuard::new(kind, Arc::clone(&self.inner.counters), self.inner.query_tracker.clone(), job);
        let concurrency = Arc::clone(&self.inner.concurrency);
        let resource_epoch = self.resource_epoch.clone();
        self.inner.handle.spawn(async move {
            let _resource_epoch = resource_epoch;
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            let _ = run_counted_job(job_guard, permit, future).await;
        });
        Ok(())
    }

    /// Detaches one observed job and invokes `completion` after its terminal
    /// job event, including cancellation or panic cleanup.
    pub(crate) fn spawn_observed_with_completion<F, C>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        future: F,
        completion: C,
    ) where
        F: Future<Output = ()> + Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        let job = self.new_job_observation(kind, observation);
        let job_guard = JobCounterGuard::new_with_completion(
            kind,
            Arc::clone(&self.inner.counters),
            self.inner.query_tracker.clone(),
            job,
            Box::new(completion),
        );
        let concurrency = Arc::clone(&self.inner.concurrency);
        let resource_epoch = self.resource_epoch.clone();
        self.inner.handle.spawn(async move {
            let _resource_epoch = resource_epoch;
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            let _ = run_counted_job(job_guard, permit, future).await;
        });
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
            device_ordinal: self.inner.device_ordinal,
            worker_count: self.inner.worker_count,
            active_jobs,
            peak_active_jobs: counters.peak_active.load(Ordering::Acquire).max(active_jobs),
            queued_jobs,
            peak_queued_jobs: counters.peak_queued.load(Ordering::Acquire).max(queued_jobs),
            active_query_jobs,
            peak_active_query_jobs: counters.peak_active_queries.load(Ordering::Acquire).max(active_query_jobs),
        }
    }

    /// Returns the aggregate query tracker attached to this pool.
    pub fn query_tracker(&self) -> GpuExecutionQueryTracker {
        self.inner.query_tracker.clone()
    }

    pub(crate) fn with_resource_epoch(&self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        Self { inner: Arc::clone(&self.inner), resource_epoch: Some(resource_epoch) }
    }

    fn check_resource_epoch(&self, operation: &'static str) -> Result<()> {
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
    runtime: Option<Runtime>,
    concurrency: Arc<Semaphore>,
    counters: Arc<PoolCounters>,
    query_tracker: GpuExecutionQueryTracker,
    next_job_id: AtomicU64,
}

impl Drop for GpuExecutionPoolInner {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            // Dropping a runtime can block and panics inside an asynchronous
            // context. Final pool release is allowed from a request future, so
            // always use Tokio's explicitly non-blocking shutdown path.
            runtime.shutdown_background();
        }
    }
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
    query_tracker: GpuExecutionQueryTracker,
    state: JobCounterState,
    terminal: JobTerminalGuard,
    completion: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl JobCounterGuard {
    fn new(
        kind: GpuExecutionJobKind,
        counters: Arc<PoolCounters>,
        query_tracker: GpuExecutionQueryTracker,
        observation: Arc<JobObservation>,
    ) -> Self {
        increment_with_peak(&counters.queued, &counters.peak_queued);
        Self {
            kind,
            counters,
            query_tracker,
            state: JobCounterState::Queued,
            terminal: JobTerminalGuard::spawned(observation),
            completion: None,
        }
    }

    fn new_with_completion(
        kind: GpuExecutionJobKind,
        counters: Arc<PoolCounters>,
        query_tracker: GpuExecutionQueryTracker,
        observation: Arc<JobObservation>,
        completion: Box<dyn FnOnce() + Send + 'static>,
    ) -> Self {
        let mut guard = Self::new(kind, counters, query_tracker, observation);
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
        if let Some(completion) = self.completion.take() {
            completion();
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
mod tests {
    use std::{
        sync::{
            Condvar,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    use futures::task::{ArcWake, waker_ref};

    use super::*;

    #[derive(Default)]
    struct BlockingGate {
        released: Mutex<bool>,
        wake: Condvar,
    }

    impl BlockingGate {
        fn wait(&self) {
            let mut released = self.released.lock().unwrap();
            while !*released {
                released = self.wake.wait(released).unwrap();
            }
        }

        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.wake.notify_all();
        }
    }

    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    struct WakeFlag(AtomicBool);

    impl ArcWake for WakeFlag {
        fn wake_by_ref(flag: &Arc<Self>) {
            flag.0.store(true, Ordering::Release);
        }
    }

    async fn wait_until(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !predicate() {
            assert!(Instant::now() < deadline, "condition was not satisfied before the test deadline");
            tokio::task::yield_now().await;
        }
    }

    fn observed_job_states(owner: &crate::observability::RuntimeObservationOwner) -> Vec<GpuJobLifecycleState> {
        owner
            .live_snapshot()
            .expect("observation remains live")
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::GpuJob(observation) => Some(observation.lifecycle.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn observed_job_defines_worker_and_records_host_lifecycle() {
        let pool = GpuExecutionPool::new(0, 1).unwrap();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "gpu-job-success",
            None,
            crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
        );

        let value =
            pool.run_observed(GpuExecutionJobKind::QueryExecution, owner.handle(), async { 7_u8 }).await.unwrap();

        assert_eq!(value, 7);
        assert_eq!(
            observed_job_states(&owner),
            vec![GpuJobLifecycleState::Enqueued, GpuJobLifecycleState::Started, GpuJobLifecycleState::Finished]
        );
        let snapshot = owner.live_snapshot().unwrap();
        crate::observability::assert_live_topology_references_resolve(&snapshot);
        assert_eq!(snapshot.topology().filter(|row| matches!(row, RuntimeTopologyRecord::JobDefinition(_))).count(), 1);
        assert_eq!(
            snapshot.topology().filter(|row| matches!(row, RuntimeTopologyRecord::WorkerDefinition(_))).count(),
            1
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn observed_job_abort_records_one_cancel_terminal() {
        let pool = GpuExecutionPool::new(0, 1).unwrap();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "gpu-job-cancel",
            None,
            crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
        );
        let task = {
            let pool = pool.clone();
            let observation = owner.handle();
            tokio::spawn(async move {
                pool.run_observed(GpuExecutionJobKind::QueryExecution, observation, std::future::pending::<()>()).await
            })
        };
        wait_until(|| pool.snapshot().active_jobs == 1).await;

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        wait_until(|| pool.snapshot().active_jobs == 0).await;

        let states = observed_job_states(&owner);
        assert_eq!(
            states,
            vec![
                GpuJobLifecycleState::Enqueued,
                GpuJobLifecycleState::Started,
                GpuJobLifecycleState::CancellationRequested,
                GpuJobLifecycleState::Cancelled,
            ]
        );
        assert_eq!(
            states
                .iter()
                .filter(|state| matches!(state, GpuJobLifecycleState::Cancelled | GpuJobLifecycleState::Dropped))
                .count(),
            1
        );
    }

    #[test]
    fn dropping_woken_outer_after_child_finished_emits_no_post_terminal_cancel() {
        let pool = GpuExecutionPool::new(0, 1).unwrap();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "gpu-job-finished-before-outer-drop",
            None,
            crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
        );
        let gate = Arc::new(BlockingGate::default());
        let child_gate = Arc::clone(&gate);
        let (started_sender, started_receiver) = mpsc::sync_channel(0);
        let mut outer = Box::pin(pool.run_observed(GpuExecutionJobKind::QueryExecution, owner.handle(), async move {
            started_sender.send(()).unwrap();
            child_gate.wait();
            7_u8
        }));
        let wake_flag = Arc::new(WakeFlag(AtomicBool::new(false)));
        let waker = waker_ref(&wake_flag);
        let mut context = Context::from_waker(&waker);

        assert!(matches!(outer.as_mut().poll(&mut context), Poll::Pending));
        started_receiver.recv().unwrap();
        wake_flag.0.store(false, Ordering::Release);
        gate.release();

        let deadline = Instant::now() + Duration::from_secs(5);
        while !wake_flag.0.load(Ordering::Acquire)
            || !observed_job_states(&owner).iter().any(|state| matches!(state, GpuJobLifecycleState::Finished))
        {
            assert!(Instant::now() < deadline, "child did not finish and wake the outer future");
            thread::yield_now();
        }
        drop(outer);

        assert_eq!(
            observed_job_states(&owner),
            vec![GpuJobLifecycleState::Enqueued, GpuJobLifecycleState::Started, GpuJobLifecycleState::Finished]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropping_polled_reentrant_job_records_one_observed_cancellation() {
        let pool = GpuExecutionPool::new(0, 1).unwrap();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "gpu-inline-cancel",
            None,
            crate::observability::RuntimeObservationConfig::bounded(64, 64 * 1024),
        );
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task = {
            let outer_pool = pool.clone();
            let nested_pool = pool.clone();
            let outer_observation = owner.handle();
            let nested_observation = owner.handle();
            tokio::spawn(async move {
                outer_pool
                    .run_observed(GpuExecutionJobKind::QueryExecution, outer_observation, async move {
                        nested_pool
                            .run_observed(GpuExecutionJobKind::ContextSetup, nested_observation, async move {
                                let _ = started_tx.send(());
                                std::future::pending::<()>().await;
                            })
                            .await
                    })
                    .await
            })
        };
        started_rx.await.unwrap();

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        wait_until(|| pool.snapshot().active_jobs == 0).await;

        let states = owner
            .live_snapshot()
            .unwrap()
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::GpuJob(observation)
                    if observation.job_kind == GpuExecutionJobKind::ContextSetup.as_str() =>
                {
                    Some(observation.lifecycle.clone())
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            states,
            vec![
                GpuJobLifecycleState::Enqueued,
                GpuJobLifecycleState::Started,
                GpuJobLifecycleState::CancellationRequested,
                GpuJobLifecycleState::Cancelled,
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn panicking_reentrant_job_records_exactly_one_panic_terminal() {
        let pool = GpuExecutionPool::new(0, 1).unwrap();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "gpu-inline-panic",
            None,
            crate::observability::RuntimeObservationConfig::bounded(64, 64 * 1024),
        );
        let nested_pool = pool.clone();
        let nested_observation = owner.handle();
        let nested = pool
            .run(GpuExecutionJobKind::QueryExecution, async move {
                nested_pool
                    .run_observed(GpuExecutionJobKind::ContextSetup, nested_observation, async move {
                        panic!("inline job panic")
                    })
                    .await
            })
            .await
            .unwrap();
        assert_eq!(nested.unwrap_err().code(), QueryErrorCode::Internal);

        let terminals = owner
            .live_snapshot()
            .unwrap()
            .events()
            .iter()
            .filter(|event| {
                matches!(
                    &event.payload,
                    RuntimeObservationPayload::GpuJob(observation)
                        if observation.job_kind == GpuExecutionJobKind::ContextSetup.as_str()
                            && observation.lifecycle == GpuJobLifecycleState::Panicked
                )
            })
            .count();
        assert_eq!(terminals, 1);
    }

    #[test]
    fn zero_worker_count_is_structured() {
        let error = GpuExecutionPool::new(7, 0).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("device_ordinal"), Some("7"));
        assert_eq!(error.fact_value("worker_count"), Some("0"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fixed_width_tracks_peaks_without_starving_serving_runtime() {
        let pool = GpuExecutionPool::new(0, 2).unwrap();
        let gate = Arc::new(BlockingGate::default());
        let heartbeat = Arc::new(AtomicUsize::new(0));
        let stop_heartbeat = Arc::new(AtomicBool::new(false));

        let heartbeat_task = {
            let heartbeat = Arc::clone(&heartbeat);
            let stop_heartbeat = Arc::clone(&stop_heartbeat);
            tokio::spawn(async move {
                while !stop_heartbeat.load(Ordering::Acquire) {
                    heartbeat.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                }
            })
        };

        let mut jobs = Vec::new();
        for value in 0..4 {
            let pool = pool.clone();
            let gate = Arc::clone(&gate);
            jobs.push(tokio::spawn(async move {
                pool.run(GpuExecutionJobKind::QueryExecution, async move {
                    gate.wait();
                    value
                })
                .await
            }));
        }

        wait_until(|| {
            let snapshot = pool.snapshot();
            snapshot.active_jobs == 2 && snapshot.queued_jobs == 2 && snapshot.active_query_jobs == 2
        })
        .await;
        assert!(heartbeat.load(Ordering::Relaxed) > 0);
        let saturated = pool.snapshot();
        assert_eq!(saturated.active_query_jobs, 2);
        assert_eq!(saturated.peak_active_jobs, 2);
        assert_eq!(saturated.peak_active_query_jobs, 2);

        gate.release();
        for job in jobs {
            job.await.unwrap().unwrap();
        }
        stop_heartbeat.store(true, Ordering::Release);
        heartbeat_task.await.unwrap();

        let complete = pool.snapshot();
        assert_eq!(complete.active_jobs, 0);
        assert_eq!(complete.queued_jobs, 0);
        assert_eq!(complete.peak_active_jobs, 2);
        assert!(complete.peak_queued_jobs >= 2);
        assert_eq!(complete.peak_active_query_jobs, 2);
        assert_eq!(pool.query_tracker().snapshot().peak_active_query_jobs, 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn width_one_spawned_child_reentrant_run_stays_on_the_same_thread() {
        let pool = GpuExecutionPool::new(1, 1).unwrap();
        let nested_pool = pool.clone();
        let (outer_thread, inner_thread) = pool
            .run(GpuExecutionJobKind::QueryExecution, async move {
                let outer_thread = thread::current().id();
                let inner_thread = tokio::spawn(async move {
                    nested_pool.run(GpuExecutionJobKind::ContextSetup, async { thread::current().id() }).await
                })
                .await
                .unwrap()
                .unwrap();
                (outer_thread, inner_thread)
            })
            .await
            .unwrap();

        assert_eq!(outer_thread, inner_thread);
        assert_eq!(pool.snapshot().peak_active_jobs, 1);
        assert_eq!(pool.snapshot().peak_active_query_jobs, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropping_run_aborts_job_and_releases_counters_and_resources() {
        let pool = GpuExecutionPool::new(2, 1).unwrap();
        let dropped = Arc::new(AtomicBool::new(false));
        let task = {
            let pool = pool.clone();
            let dropped = Arc::clone(&dropped);
            tokio::spawn(async move {
                pool.run(GpuExecutionJobKind::QueryExecution, async move {
                    let _probe = DropProbe(dropped);
                    std::future::pending::<()>().await;
                })
                .await
            })
        };

        wait_until(|| pool.snapshot().active_jobs == 1).await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        wait_until(|| dropped.load(Ordering::Acquire) && pool.snapshot().active_jobs == 0).await;

        let snapshot = pool.snapshot();
        assert_eq!(snapshot.queued_jobs, 0);
        assert_eq!(snapshot.active_query_jobs, 0);
        assert_eq!(pool.query_tracker().snapshot().active_query_jobs, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn final_pool_drop_is_safe_inside_current_thread_runtime() {
        let pool = GpuExecutionPool::new(3, 1).unwrap();
        pool.run(GpuExecutionJobKind::Cleanup, async {}).await.unwrap();
        drop(pool);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn panicked_jobs_are_structured_permanent_internal_errors() {
        let pool = GpuExecutionPool::new(6, 1).unwrap();
        let error = pool.run(GpuExecutionJobKind::QueryExecution, async { panic!("test panic") }).await.unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.kind(), crate::error::QueryErrorKind::Internal);
        assert_eq!(error.status(), crate::error::QueryErrorStatus::Permanent);
        assert_eq!(error.fact_value("device_ordinal"), Some("6"));
        assert_eq!(error.fact_value("job_kind"), Some("query_execution"));
        assert_eq!(error.fact_value("reason"), Some("job_panicked"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shared_tracker_records_exact_cross_device_peak() {
        let tracker = GpuExecutionQueryTracker::new();
        let first = GpuExecutionPool::new_with_tracker(4, 1, tracker.clone()).unwrap();
        let second = GpuExecutionPool::new_with_tracker(5, 1, tracker.clone()).unwrap();
        let gate = Arc::new(BlockingGate::default());

        let first_job = {
            let first = first.clone();
            let gate = Arc::clone(&gate);
            tokio::spawn(
                async move { first.run(GpuExecutionJobKind::QueryExecution, async move { gate.wait() }).await },
            )
        };
        let second_job = {
            let second = second.clone();
            let gate = Arc::clone(&gate);
            tokio::spawn(
                async move { second.run(GpuExecutionJobKind::QueryExecution, async move { gate.wait() }).await },
            )
        };

        wait_until(|| tracker.snapshot().active_query_jobs == 2).await;
        assert_eq!(tracker.snapshot().peak_active_query_jobs, 2);
        gate.release();
        first_job.await.unwrap().unwrap();
        second_job.await.unwrap().unwrap();
        assert_eq!(tracker.snapshot().active_query_jobs, 0);
        assert_eq!(tracker.snapshot().peak_active_query_jobs, 2);
    }
}
