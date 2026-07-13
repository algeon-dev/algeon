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
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use futures::FutureExt;
use tokio::{
    runtime::{Handle, Runtime},
    sync::{OwnedSemaphorePermit, Semaphore},
    task::{JoinError, JoinHandle},
};

use crate::error::{QueryError, QueryErrorCode, Result};

const BUILD_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.build";
const RUN_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.run";
const SHARED_OPERATION: &str = "nexus_query_engine.gpu_execution_pool.shared";

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static CURRENT_POOL_ID: Cell<Option<u64>> = const { Cell::new(None) };
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
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_count)
            .thread_name(format!("nexus-gpu-{device_ordinal}"))
            .on_thread_start(move || CURRENT_POOL_ID.with(|current| current.set(Some(id))))
            .on_thread_stop(move || CURRENT_POOL_ID.with(|current| current.set(None)))
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
            }),
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
            return Ok(Self { inner });
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
        if current_pool_id() == Some(self.inner.id) {
            return AssertUnwindSafe(future)
                .catch_unwind()
                .await
                .map_err(|_panic| job_panic_error(self.inner.device_ordinal, kind));
        }

        let job_guard = JobCounterGuard::new(kind, Arc::clone(&self.inner.counters), self.inner.query_tracker.clone());
        let concurrency = Arc::clone(&self.inner.concurrency);
        let task = self.inner.handle.spawn(async move {
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            run_counted_job(job_guard, permit, future).await
        });

        match AbortOnDropJoinHandle::new(task).await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(_panic)) => Err(job_panic_error(self.inner.device_ordinal, kind)),
            Err(error) if error.is_cancelled() => Err(job_cancelled_error(self.inner.device_ordinal, kind, error)),
            Err(error) => Err(job_join_panic_error(self.inner.device_ordinal, kind, error)),
        }
    }

    /// Detaches one job onto the device pool.
    ///
    /// This is intended for cleanup that must not block a serving worker. The
    /// pool still accounts for the job, and its RAII counters are released if
    /// it completes, panics, or is cancelled during runtime shutdown.
    pub fn spawn<F>(&self, kind: GpuExecutionJobKind, future: F) -> Result<()>
    where F: Future<Output = ()> + Send + 'static {
        let job_guard = JobCounterGuard::new(kind, Arc::clone(&self.inner.counters), self.inner.query_tracker.clone());
        let concurrency = Arc::clone(&self.inner.concurrency);
        self.inner.handle.spawn(async move {
            let permit = concurrency.acquire_owned().await;
            let permit = permit.expect("GPU execution pool semaphore must remain open while jobs are live");
            let _ = run_counted_job(job_guard, permit, future).await;
        });
        Ok(())
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
}

impl JobCounterGuard {
    fn new(kind: GpuExecutionJobKind, counters: Arc<PoolCounters>, query_tracker: GpuExecutionQueryTracker) -> Self {
        increment_with_peak(&counters.queued, &counters.peak_queued);
        Self { kind, counters, query_tracker, state: JobCounterState::Queued }
    }

    fn activate(&mut self) {
        debug_assert_eq!(self.state, JobCounterState::Queued);
        decrement(&self.counters.queued);
        increment_with_peak(&self.counters.active, &self.counters.peak_active);
        if self.kind == GpuExecutionJobKind::QueryExecution {
            increment_with_peak(&self.counters.active_queries, &self.counters.peak_active_queries);
            self.query_tracker.start_query();
        }
        self.state = JobCounterState::Active;
    }
}

impl Drop for JobCounterGuard {
    fn drop(&mut self) {
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
    guard.activate();
    AssertUnwindSafe(future).catch_unwind().await.map_err(|_panic| ())
}

fn current_pool_id() -> Option<u64> {
    CURRENT_POOL_ID.with(Cell::get)
}

struct AbortOnDropJoinHandle<T> {
    handle: JoinHandle<T>,
}

impl<T> AbortOnDropJoinHandle<T> {
    fn new(handle: JoinHandle<T>) -> Self {
        Self { handle }
    }
}

impl<T> Future for AbortOnDropJoinHandle<T> {
    type Output = std::result::Result<T, JoinError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.handle).poll(context)
    }
}

impl<T> Drop for AbortOnDropJoinHandle<T> {
    fn drop(&mut self) {
        self.handle.abort();
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
        },
        thread,
        time::{Duration, Instant},
    };

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

    async fn wait_until(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !predicate() {
            assert!(Instant::now() < deadline, "condition was not satisfied before the test deadline");
            tokio::task::yield_now().await;
        }
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
