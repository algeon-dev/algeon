//! Shared pool state, submission gate, and drain coordination.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::{runtime::Handle, sync::Semaphore, task::JoinHandle};

use super::{GpuExecutionQueryTracker, PoolCounters, RUN_OPERATION, SHUTDOWN_OPERATION, supervisor::RuntimeSupervisor};
use crate::runtime::deadline::{DeadlineMutexLockError, deadline_remaining, mutex_lock_until};

pub(super) struct GpuExecutionPoolInner {
    pub(super) id: u64,
    pub(super) device_ordinal: usize,
    pub(super) worker_count: usize,
    pub(super) handle: Handle,
    pub(super) supervisor: Arc<RuntimeSupervisor>,
    pub(super) submissions_closed: AtomicBool,
    pub(super) submission_gate: Mutex<()>,
    pub(super) concurrency: Arc<Semaphore>,
    pub(super) lifecycle_job_permits: Arc<Semaphore>,
    pub(super) execution_job_permits: Arc<Semaphore>,
    pub(super) release_cleanup_permits: Arc<Semaphore>,
    pub(super) lifecycle_job_capacity: usize,
    pub(super) execution_job_capacity: usize,
    pub(super) counters: Arc<PoolCounters>,
    pub(super) drain: Arc<PoolDrain>,
    pub(super) query_tracker: GpuExecutionQueryTracker,
    pub(super) next_job_id: AtomicU64,
}

impl Drop for GpuExecutionPoolInner {
    fn drop(&mut self) {
        self.submissions_closed.store(true, Ordering::Release);
        self.supervisor.request_exit_from_drop();
    }
}

impl GpuExecutionPoolInner {
    pub(super) fn spawn_task<F>(&self, future: F) -> Result<JoinHandle<F::Output>>
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

    pub(super) fn close_submissions(&self) -> Result<()> {
        let _submission = self.submission_gate.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, SHUTDOWN_OPERATION, "GPU execution submission gate was poisoned")
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "gpu_execution_pool_submission_gate_poisoned")
        })?;
        self.submissions_closed.store(true, Ordering::Release);
        Ok(())
    }

    pub(super) fn request_supervisor_exit(&self) -> Result<()> {
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

    pub(super) fn wait_for_drain_until(&self, deadline: Instant) -> Result<()> {
        self.drain.wait_until(deadline, self.device_ordinal, self.counters.as_ref())
    }
}

pub(super) fn pool_closed_error(device_ordinal: usize) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, RUN_OPERATION, "GPU execution pool is closing")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("reason", "gpu_execution_pool_closing")
}

pub(super) fn pool_drain_timeout(device_ordinal: usize, counters: &PoolCounters) -> QueryError {
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
pub(super) struct PoolDrain {
    pub(super) changed: Mutex<()>,
    pub(super) wake: Condvar,
}

impl PoolDrain {
    pub(super) fn wait_until(&self, deadline: Instant, device_ordinal: usize, counters: &PoolCounters) -> Result<()> {
        let mut state = mutex_lock_until(&self.changed, deadline).map_err(|error| match error {
            DeadlineMutexLockError::Poisoned => QueryError::new(
                QueryErrorCode::Internal,
                SHUTDOWN_OPERATION,
                "GPU execution pool drain mutex was poisoned",
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("reason", "gpu_execution_pool_drain_poisoned"),
            DeadlineMutexLockError::TimedOut => pool_drain_timeout(device_ordinal, counters),
        })?;
        while counters.active.load(Ordering::Acquire) != 0 || counters.queued.load(Ordering::Acquire) != 0 {
            let Some(remaining) = deadline_remaining(deadline) else {
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
