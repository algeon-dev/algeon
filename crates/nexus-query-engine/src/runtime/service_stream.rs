//! Service-level CUDA stream pooling for native worker tasks.

use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, OnceLock, mpsc},
    thread,
    time::Duration,
};

#[cfg(test)]
use super::DeviceFacts;
use super::{ExecutionCancellation, ExecutionContext};
use crate::error::{QueryError, QueryErrorCode, Result};

/// Fallback and upper bound for the default service stream pool size.
///
/// The pool creates streams lazily, so this is a concurrency cap rather than a
/// startup allocation count. [`default_service_stream_pool_size_for_device`]
/// uses this value as the fallback when device SM count detection is
/// unavailable and as the upper bound when detection succeeds.
pub const DEFAULT_SERVICE_STREAM_POOL_SIZE: usize = 64;
const STREAM_ACQUIRE_CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Returns the default service stream pool size for a CUDA device.
///
/// The default is `min(device_multiprocessor_count * 2, 64)`. If the
/// multiprocessor count cannot be detected, this returns
/// [`DEFAULT_SERVICE_STREAM_POOL_SIZE`]. The detected value is cached per
/// device ordinal for the process lifetime.
pub fn default_service_stream_pool_size_for_device(device: cudf::DeviceId) -> usize {
    static DEFAULT_SIZES: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    let sizes = DEFAULT_SIZES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut sizes = match sizes.lock() {
        Ok(sizes) => sizes,
        Err(_error) => {
            return default_service_stream_pool_size_from_sm_count(detect_device_multiprocessor_count(device));
        },
    };
    *sizes
        .entry(device.ordinal())
        .or_insert_with(|| default_service_stream_pool_size_from_sm_count(detect_device_multiprocessor_count(device)))
}

pub(super) fn default_service_stream_pool_size_from_sm_count(sm_count: Option<usize>) -> usize {
    sm_count
        .filter(|count| *count > 0)
        .and_then(|count| count.checked_mul(2))
        .map(|size| size.min(DEFAULT_SERVICE_STREAM_POOL_SIZE))
        .unwrap_or(DEFAULT_SERVICE_STREAM_POOL_SIZE)
}

#[cfg(test)]
pub(super) fn default_service_stream_pool_size_from_device_facts(facts: &DeviceFacts) -> usize {
    default_service_stream_pool_size_from_sm_count(facts.sm_count.and_then(nonzero_sm_count))
}

fn detect_device_multiprocessor_count(device: cudf::DeviceId) -> Option<usize> {
    device.multiprocessor_count().ok().and_then(nonzero_sm_count)
}

pub(super) fn nonzero_sm_count(count: u32) -> Option<usize> {
    (count > 0).then(|| usize::try_from(count).ok()).flatten()
}

#[derive(Debug)]
struct ServiceStreamPoolInner {
    device: cudf::DeviceId,
    pool_size: usize,
    state: Mutex<ServiceStreamPoolState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct ServiceStreamPoolState {
    idle: Vec<Arc<cudf::Runtime>>,
    created: usize,
}

/// Service-level CUDA stream pool for short-lived native worker tasks.
///
/// The pool owns a bounded set of same-device cuDF runtimes. Each runtime
/// carries one CUDA stream and the process-local device memory manager. Streams
/// are created lazily up to `pool_size`; dropping a [`StreamLease`] returns the
/// runtime to the pool for reuse. This avoids creating a CUDA stream for every
/// bounded source-read worker while preserving the existing reservation-aware
/// allocator contract.
#[derive(Debug, Clone)]
pub struct ServiceStreamPool {
    inner: Arc<ServiceStreamPoolInner>,
}

impl ServiceStreamPool {
    /// Create a stream pool for `device`.
    ///
    /// `pool_size` is the maximum number of simultaneously leased streams.
    /// Runtime/stream creation is lazy: construction records the limit but does
    /// not allocate CUDA streams immediately.
    pub fn new(device: cudf::DeviceId, pool_size: usize) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        Ok(Self {
            inner: Arc::new(ServiceStreamPoolInner {
                device,
                pool_size,
                state: Mutex::new(ServiceStreamPoolState::default()),
                available: Condvar::new(),
            }),
        })
    }

    /// Return the process-global default pool for `device` and `pool_size`.
    ///
    /// This is the fallback used by native execution when a service has not
    /// injected its own pool yet. Pools are keyed by `(device ordinal,
    /// pool_size)` so differently sized service tests do not silently share a
    /// capacity contract.
    pub fn for_device(device: cudf::DeviceId, pool_size: usize) -> Result<Self> {
        validate_service_stream_pool_size(pool_size)?;
        static POOLS: OnceLock<Mutex<HashMap<(usize, usize), ServiceStreamPool>>> = OnceLock::new();
        let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut pools = pools.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.service_stream_pool.registry",
                "service stream pool registry mutex was poisoned",
            )
        })?;
        let key = (device.ordinal(), pool_size);
        if let Some(pool) = pools.get(&key) {
            return Ok(pool.clone());
        }
        let pool = Self::new(device, pool_size)?;
        pools.insert(key, pool.clone());
        Ok(pool)
    }

    /// Return the process-global default-sized pool for `device`.
    pub fn default_for_device(device: cudf::DeviceId) -> Result<Self> {
        Self::for_device(device, default_service_stream_pool_size_for_device(device))
    }

    /// CUDA device served by this pool.
    pub fn device(&self) -> cudf::DeviceId {
        self.inner.device
    }

    /// Maximum number of simultaneously leased streams.
    pub fn pool_size(&self) -> usize {
        self.inner.pool_size
    }

    /// Acquire a CUDA stream/runtime lease, blocking when all pool slots are busy.
    pub fn acquire(&self) -> Result<StreamLease> {
        self.acquire_inner(None, "nexus_query_engine.service_stream_pool.acquire")
    }

    /// Acquire a CUDA stream/runtime lease while observing cooperative cancellation.
    pub fn acquire_with_cancellation(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<StreamLease> {
        self.acquire_inner(Some(cancellation), operation)
    }

    fn acquire_inner(
        &self,
        cancellation: Option<&ExecutionCancellation>,
        operation: &'static str,
    ) -> Result<StreamLease> {
        enum AcquireAction {
            Reuse(Arc<cudf::Runtime>),
            Create,
        }

        let action = loop {
            if let Some(cancellation) = cancellation {
                cancellation.check(operation)?;
            }
            let mut state = self.lock_state(operation)?;
            if let Some(runtime) = state.idle.pop() {
                break AcquireAction::Reuse(runtime);
            }
            if state.created < self.pool_size() {
                state.created += 1;
                break AcquireAction::Create;
            }
            if cancellation.is_some() {
                let (_guard, _timeout) =
                    self.inner.available.wait_timeout(state, STREAM_ACQUIRE_CANCELLATION_POLL_INTERVAL).map_err(
                        |_error| {
                            QueryError::new(
                                QueryErrorCode::Internal,
                                operation,
                                "service stream pool mutex was poisoned while waiting for a stream",
                            )
                        },
                    )?;
            } else {
                let _guard = self.inner.available.wait(state).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        operation,
                        "service stream pool mutex was poisoned while waiting for a stream",
                    )
                })?;
            }
        };

        let runtime = match action {
            AcquireAction::Reuse(runtime) => runtime,
            AcquireAction::Create => match self.build_runtime() {
                Ok(runtime) => Arc::new(runtime),
                Err(error) => {
                    self.discard_created_slot();
                    return Err(error);
                },
            },
        };
        Ok(StreamLease { pool: self.clone(), runtime: Some(runtime) })
    }

    fn build_runtime(&self) -> Result<cudf::Runtime> {
        cudf::Runtime::builder().with_device(self.device()).with_device_memory_manager().build().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.service_stream_pool.runtime",
                "failed to initialize service stream pool runtime",
                error,
            )
        })
    }

    fn lock_state(&self, operation: &'static str) -> Result<std::sync::MutexGuard<'_, ServiceStreamPoolState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "service stream pool mutex was poisoned")
        })
    }

    fn discard_created_slot(&self) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.created = state.created.saturating_sub(1);
                self.inner.available.notify_one();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to update poisoned service stream pool state");
            },
        }
    }

    fn release(&self, runtime: Arc<cudf::Runtime>) {
        match self.inner.state.lock() {
            Ok(mut state) => {
                state.idle.push(runtime);
                self.inner.available.notify_one();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to return runtime to poisoned service stream pool");
            },
        }
    }
}

/// RAII handle for a runtime borrowed from a [`ServiceStreamPool`].
///
/// Dropping the lease schedules runtime stream synchronization on the service
/// cleanup thread and returns the runtime to the pool only after that
/// synchronization completes. This keeps source-read output safe to consume on
/// another runtime before the stream is reused by an unrelated worker without
/// running CUDA synchronization on the dropping thread.
#[derive(Debug)]
pub struct StreamLease {
    pool: ServiceStreamPool,
    runtime: Option<Arc<cudf::Runtime>>,
}

impl StreamLease {
    pub(super) fn pool(&self) -> &ServiceStreamPool {
        &self.pool
    }

    /// Borrow the leased cuDF runtime.
    pub fn runtime(&self) -> &cudf::Runtime {
        self.runtime.as_ref().expect("stream lease runtime is present").as_ref()
    }

    /// Clone the shared runtime handle for a short-lived execution context.
    pub fn shared_runtime(&self) -> Arc<cudf::Runtime> {
        Arc::clone(self.runtime.as_ref().expect("stream lease runtime is present"))
    }
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        enqueue_stream_cleanup(self.pool.clone(), runtime);
    }
}

enum StreamCleanupJob {
    Runtime {
        pool: ServiceStreamPool,
        runtime: Arc<cudf::Runtime>,
    },
    #[cfg(test)]
    Probe {
        done: mpsc::Sender<thread::ThreadId>,
    },
}

fn enqueue_stream_cleanup(pool: ServiceStreamPool, runtime: Arc<cudf::Runtime>) {
    let job = StreamCleanupJob::Runtime { pool, runtime };
    match stream_cleanup_sender() {
        Some(sender) => {
            if let Err(error) = sender.send(job) {
                match error.0 {
                    StreamCleanupJob::Runtime { pool, runtime } => synchronize_and_release_stream(pool, runtime),
                    #[cfg(test)]
                    StreamCleanupJob::Probe { .. } => {},
                }
            }
        },
        None => match job {
            StreamCleanupJob::Runtime { pool, runtime } => synchronize_and_release_stream(pool, runtime),
            #[cfg(test)]
            StreamCleanupJob::Probe { .. } => {},
        },
    }
}

fn stream_cleanup_sender() -> Option<&'static mpsc::Sender<StreamCleanupJob>> {
    static CLEANUP: OnceLock<Option<mpsc::Sender<StreamCleanupJob>>> = OnceLock::new();
    CLEANUP
        .get_or_init(|| {
            let (sender, receiver) = mpsc::channel::<StreamCleanupJob>();
            match thread::Builder::new().name("nexus-service-stream-cleanup".to_string()).spawn(move || {
                for job in receiver {
                    match job {
                        StreamCleanupJob::Runtime { pool, runtime } => synchronize_and_release_stream(pool, runtime),
                        #[cfg(test)]
                        StreamCleanupJob::Probe { done } => {
                            let _ = done.send(thread::current().id());
                        },
                    }
                }
            }) {
                Ok(_handle) => Some(sender),
                Err(error) => {
                    eprintln!("[nexus-query-engine] warning: failed to spawn service stream cleanup thread: {error}");
                    None
                },
            }
        })
        .as_ref()
}

fn synchronize_and_release_stream(pool: ServiceStreamPool, runtime: Arc<cudf::Runtime>) {
    if let Err(error) = runtime.synchronize() {
        eprintln!("[nexus-query-engine] warning: failed to synchronize service stream before return: {error}");
    }
    pool.release(runtime);
}

#[cfg(test)]
fn stream_cleanup_thread_id_for_test() -> thread::ThreadId {
    let (sender, receiver) = mpsc::channel();
    stream_cleanup_sender()
        .expect("cleanup thread should be available in tests")
        .send(StreamCleanupJob::Probe { done: sender })
        .expect("cleanup thread accepts probe");
    receiver.recv().expect("cleanup thread returns probe")
}

pub(crate) struct SourceReadStreamLease {
    context: ExecutionContext,
    _stream_lease: StreamLease,
}

impl SourceReadStreamLease {
    pub(super) fn new(context: ExecutionContext, stream_lease: StreamLease) -> Self {
        Self { context, _stream_lease: stream_lease }
    }

    pub(crate) fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

pub(super) fn validate_service_stream_pool_size(pool_size: usize) -> Result<()> {
    if pool_size == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.service_stream_pool.validate",
            "service stream pool size must be greater than zero",
        )
        .with_fact("field", "service_stream_pool_size")
        .with_fact("reason", "zero"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_cleanup_jobs_run_on_background_thread() {
        assert_ne!(stream_cleanup_thread_id_for_test(), thread::current().id());
    }

    #[test]
    fn stream_acquire_observes_cancellation_while_pool_is_saturated() {
        let device = cudf::DeviceId::DEFAULT;
        let pool = ServiceStreamPool::new(device, 1).unwrap();
        let held = pool.acquire().unwrap();
        let cancellation = ExecutionCancellation::new();
        let waiter_cancellation = cancellation.clone();
        let waiter_pool = pool.clone();
        let (started_sender, started_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();

        let waiter = thread::spawn(move || {
            started_sender.send(()).unwrap();
            let result = waiter_pool.acquire_with_cancellation(
                &waiter_cancellation,
                "nexus_query_engine.service_stream_pool.test_cancelled_acquire",
            );
            result_sender.send(result.map(|_lease| ())).unwrap();
        });
        started_receiver.recv().unwrap();
        thread::sleep(Duration::from_millis(25));
        cancellation.cancel();
        let result = result_receiver.recv_timeout(Duration::from_secs(1));
        drop(held);
        waiter.join().unwrap();
        let error = result.expect("cancelled stream acquisition should finish").unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.operation(), "nexus_query_engine.service_stream_pool.test_cancelled_acquire");
    }
}
