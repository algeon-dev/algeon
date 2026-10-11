//! Shared pool state, submission gate, and drain coordination.
//!
//! Every clone of a `GpuExecutionPool` shares one `GpuExecutionPoolInner`.
//! Shutdown correctness rests on two orderings this module enforces: no task
//! is spawned after `submissions_closed` is observed set, and the supervisor
//! is asked to drop the runtime only after submissions are closed. The first
//! is what makes the drain counters a complete census of outstanding work.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::{
    runtime::Handle,
    sync::{OwnedSemaphorePermit, Semaphore},
    task::JoinHandle,
};

use super::{
    GpuExecutionJobKind, GpuExecutionQueryTracker, PoolCounters, RUN_OPERATION, SHUTDOWN_OPERATION,
    supervisor::RuntimeSupervisor,
};
use crate::runtime::deadline::{DeadlineMutexLockError, deadline_remaining, mutex_lock_until};

/// State shared by all clones of one device pool.
///
/// Admission happens in two layers. A job-class permit
/// (`lifecycle_job_permits`, `execution_job_permits`, or
/// `release_cleanup_permits`) is taken before any task is spawned and bounds
/// how many accepted jobs can exist; it is the only layer that may fail. For
/// async jobs a worker seat (`concurrency`, plus `execution_workers` for jobs
/// outside the lifecycle reserve) is then taken by the spawned task before it
/// runs its body and bounds how many run at once; the seat only waits, and
/// the number of waiters is bounded by the class capacities. Blocking work
/// skips the seat layer and runs on the runtime's blocking threads.
pub(super) struct GpuExecutionPoolInner {
    pub(super) id: u64,
    pub(super) device_ordinal: usize,
    pub(super) worker_count: usize,
    /// Handle to the runtime owned by `supervisor`'s thread. Spawning through
    /// it after that runtime is dropped panics, which `submission_gate` and
    /// `submissions_closed` together prevent.
    pub(super) handle: Handle,
    pub(super) supervisor: Arc<RuntimeSupervisor>,
    /// Set once, never cleared. Readers that only load it may race a
    /// concurrent close; readers that must not spawn after a close hold
    /// `submission_gate` while they check it.
    pub(super) submissions_closed: AtomicBool,
    /// Serializes the `submissions_closed` check with the spawn that follows
    /// it, and with the store that closes submissions. Holding it makes
    /// "closed" a linearization point: every task spawned before the close
    /// is counted by the drain, and none is spawned after.
    pub(super) submission_gate: Mutex<()>,
    /// One seat per runtime worker. An async job occupies a seat for the
    /// whole time its synchronous GPU body runs, so `worker_count` bounds
    /// concurrent async job bodies. Blocking work submitted through
    /// `submit_blocking` takes no seat and is bounded by its class permit.
    pub(super) concurrency: Arc<Semaphore>,
    /// Seats outside the reserved lifecycle workers, taken first by the jobs
    /// that may not use the reserve.
    pub(super) execution_workers: Arc<Semaphore>,
    /// `A_d` permits: context setup and root query execution jobs.
    pub(super) lifecycle_job_permits: Arc<Semaphore>,
    /// `A_d * S_d` permits: lanes, source reads, and additional fragment
    /// setup issued underneath an admitted attempt.
    pub(super) execution_job_permits: Arc<Semaphore>,
    /// `A_d` permits reserved for the release cleanup of admitted attempts,
    /// separate from `lifecycle_job_permits` so a saturated lifecycle class
    /// cannot block a finished attempt from releasing its CUDA resources.
    pub(super) release_cleanup_permits: Arc<Semaphore>,
    pub(super) lifecycle_job_capacity: usize,
    pub(super) execution_job_capacity: usize,
    pub(super) counters: Arc<PoolCounters>,
    pub(super) drain: Arc<PoolDrain>,
    pub(super) query_tracker: GpuExecutionQueryTracker,
    pub(super) next_job_id: AtomicU64,
}

impl Drop for GpuExecutionPoolInner {
    /// Runs when the last pool clone is dropped. The flag store needs no
    /// gate because no clone remains that could submit. After an explicit
    /// close both calls are redundant and harmless; without one, this is the
    /// only path that asks the supervisor to drop the runtime.
    fn drop(&mut self) {
        self.submissions_closed.store(true, Ordering::Release);
        self.supervisor.request_exit_from_drop();
    }
}

impl GpuExecutionPoolInner {
    /// Spawns `future` onto the pool runtime unless submissions are closed.
    ///
    /// The returned task is not yet counted as queued; callers construct a
    /// `JobCounterGuard` before calling this so the drain census includes
    /// the task from the moment it can exist.
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

    pub(super) fn worker_seats(&self) -> WorkerSeats {
        WorkerSeats { workers: Arc::clone(&self.concurrency), execution: Arc::clone(&self.execution_workers) }
    }

    /// Rejects every later submission. Idempotent. Taking the gate ensures no
    /// submitter that already passed its `submissions_closed` check is still
    /// between that check and its spawn when this returns.
    pub(super) fn close_submissions(&self) -> Result<()> {
        let _submission = self.submission_gate.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, SHUTDOWN_OPERATION, "GPU execution submission gate was poisoned")
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "gpu_execution_pool_submission_gate_poisoned")
        })?;
        self.submissions_closed.store(true, Ordering::Release);
        Ok(())
    }

    /// Asks the supervisor to drop the runtime. Refused while submissions are
    /// open: dropping the runtime cancels unfinished tasks, and a pool that
    /// still accepts work would otherwise cancel jobs it had accepted.
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

/// Wakes shutdown waiters when the pool's queued and active counters change.
///
/// The counters themselves are atomics in `PoolCounters`. This mutex carries
/// no data; `JobCounterGuard::drop` holds it across its final decrement so a
/// waiter cannot check the counters, find them nonzero, and then sleep past
/// the notification for the decrement that would have made them zero.
#[derive(Debug, Default)]
pub(super) struct PoolDrain {
    pub(super) changed: Mutex<()>,
    pub(super) wake: Condvar,
}

impl PoolDrain {
    /// Blocks until both counters read zero or `deadline` passes.
    ///
    /// Zero counters mean every accepted job has released its guard. That is
    /// a complete census only after `close_submissions`, which is why the
    /// pool orders the two in `close_and_wait_until`. The timeout error
    /// reports the counters at the moment it fired and carries the
    /// cleanup-deadline marker.
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

/// The pool's worker semaphores, as a spawned job takes them.
pub(super) struct WorkerSeats {
    workers: Arc<Semaphore>,
    execution: Arc<Semaphore>,
}

/// The worker one running job holds.
pub(super) struct WorkerSeat {
    _execution: Option<OwnedSemaphorePermit>,
    _worker: OwnedSemaphorePermit,
}

impl WorkerSeats {
    /// Waits for a worker `kind` may run on. A job outside the lifecycle
    /// reserve takes an execution seat before a worker, so it can never hold
    /// the last worker the reserve keeps for lifecycle jobs.
    pub(super) async fn occupy(self, kind: GpuExecutionJobKind) -> WorkerSeat {
        const OPEN: &str = "GPU execution pool semaphore must remain open while jobs are live";
        let execution = if kind.runs_outside_lifecycle_reserve() {
            Some(self.execution.acquire_owned().await.expect(OPEN))
        } else {
            None
        };
        let worker = self.workers.acquire_owned().await.expect(OPEN);
        WorkerSeat { _execution: execution, _worker: worker }
    }
}
