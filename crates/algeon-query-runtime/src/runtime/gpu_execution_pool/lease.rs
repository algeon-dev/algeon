//! Lifecycle-bound lease and blocking-worker handle for the GPU execution pool.

use std::{
    future::Future,
    panic::AssertUnwindSafe,
    sync::{Arc, mpsc},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    GpuExecutionJobClass, GpuExecutionJobKind, GpuExecutionPool, JobObservation, RUN_OPERATION, job_panic_error,
};
use crate::observability::{RuntimeObservationHandle, RuntimeStreamLeaseScope};

/// Opaque lifecycle-bound capability for synchronous GPU work.
///
/// This lease deliberately exposes execution but not pool construction,
/// detached spawning, shutdown, counters, or raw resource-epoch access.
#[derive(Clone)]
pub struct GpuExecutorLease {
    pool: GpuExecutionPool,
    /// Class the lease's own query-execution job runs in.
    ///
    /// An attempt's root execution job is root-class work, one per attempt. A
    /// second native island of the same attempt executes on an auxiliary
    /// fragment context that already owns an attempt auxiliary permit, so its
    /// execution job is auxiliary work backed by that permit, under the same
    /// ownership rule the context-setup job and the execution lanes follow.
    /// Without this, both islands of one attempt would charge the root class,
    /// whose capacity is one job per admitted attempt.
    execution_class: GpuExecutionJobClass,
}

impl GpuExecutorLease {
    pub(crate) fn from_pool(pool: GpuExecutionPool) -> Self {
        Self { pool, execution_class: GpuExecutionJobClass::Lifecycle }
    }

    /// Builds a lease whose query-execution job runs as auxiliary work.
    ///
    /// The caller must already hold the attempt auxiliary permit that backs
    /// it, for at least the lifetime of the jobs this lease runs.
    pub(crate) fn from_pool_for_auxiliary_fragment(pool: GpuExecutionPool) -> Self {
        Self { pool, execution_class: GpuExecutionJobClass::Execution }
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
        let class = match kind {
            GpuExecutionJobKind::QueryExecution => self.execution_class,
            _ => kind.job_class(),
        };
        self.pool.clone().run_observed_scoped_in_class(kind, class, observation, build_future)
    }

    /// Submits one blocking worker onto the lease's dedicated device executor.
    ///
    /// This is the synchronous-work counterpart to [`run`](Self::run): the
    /// worker runs on a bounded blocking thread. It does not run on an async
    /// pool worker, and the same class admission applies. It fits this lease's remit of
    /// synchronous GPU work without exposing pool construction, detached
    /// spawning, shutdown, or counters.
    pub fn submit_blocking<F, T>(
        &self,
        kind: GpuExecutionJobKind,
        observation: RuntimeObservationHandle,
        work: F,
    ) -> Result<PoolBlockingWork<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.pool.submit_blocking(kind, observation, work)
    }

    /// Submits blocking work and calls `completion` after its class slot is
    /// reusable and its terminal outcome is available to join.
    #[doc(hidden)]
    pub fn submit_blocking_with_completion<F, T, C>(
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
        self.pool.submit_blocking_with_completion(kind, observation, work, completion)
    }
}

impl std::fmt::Debug for GpuExecutorLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("GpuExecutorLease").finish_non_exhaustive()
    }
}

/// Completion handle for one pool-carried blocking worker.
///
/// The waiter holds no pool permits: capacity is charged to the running
/// worker only, so a coordinator blocked in [`PoolBlockingWork::join`]
/// can never starve the worker it waits for.
pub struct PoolBlockingWork<T> {
    pub(super) receiver: mpsc::Receiver<BlockingWorkOutcome<T>>,
    pub(super) job: Arc<JobObservation>,
    pub(super) device_ordinal: usize,
    pub(super) kind: GpuExecutionJobKind,
}

impl<T> std::fmt::Debug for PoolBlockingWork<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PoolBlockingWork").finish_non_exhaustive()
    }
}

pub(super) enum BlockingWorkOutcome<T> {
    Completed(T),
    Panicked,
}

/// A submitted worker may be discarded by pool shutdown before its closure
/// starts. Its completion notification still has to wake a coordinator that
/// is waiting to join the submitted handle.
pub(super) struct BlockingCompletionNotify<C: FnOnce()> {
    completion: Option<C>,
}

impl<C: FnOnce()> BlockingCompletionNotify<C> {
    pub(super) fn new(completion: C) -> Self {
        Self { completion: Some(completion) }
    }

    pub(super) fn notify(&mut self) {
        if let Some(completion) = self.completion.take() {
            completion();
        }
    }
}

impl<C: FnOnce()> Drop for BlockingCompletionNotify<C> {
    fn drop(&mut self) {
        if self.completion.is_some() {
            // Shutdown can drop queued work during unwind; a notification panic
            // must not turn that cleanup into a second panic.
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| self.notify()));
        }
    }
}

impl<T> PoolBlockingWork<T> {
    /// Drains the worker and returns its exactly-once terminal outcome.
    ///
    /// Join drains the worker. A worker that already started is always awaited and
    /// its value returned, so the caller can release whatever that worker
    /// produced through its own accounting. A worker lost to pool shutdown or
    /// process exit reports typed cancellation and does not hang; a worker
    /// panic reports the pool's typed panic error. A resolved join implies
    /// the worker's class slot is already reusable.
    pub fn join(self) -> Result<T> {
        match self.receiver.recv() {
            Ok(BlockingWorkOutcome::Completed(value)) => Ok(value),
            Ok(BlockingWorkOutcome::Panicked) => Err(job_panic_error(self.device_ordinal, self.kind)),
            Err(_) => {
                self.job.cancellation_requested();
                Err(QueryError::new(
                    QueryErrorCode::Cancelled,
                    RUN_OPERATION,
                    "GPU execution pool blocking worker ended without delivering completion",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("job_kind", self.kind.as_str())
                .with_fact("reason", "gpu_execution_pool_blocking_worker_lost"))
            },
        }
    }
}
