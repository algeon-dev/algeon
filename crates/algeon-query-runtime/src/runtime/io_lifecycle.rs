//! Revocable process-I/O lifetime used by server generations.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

const IO_LIFECYCLE_CLOSED: usize = 1usize << (usize::BITS - 1);
const IO_LIFECYCLE_ACTIVE_MASK: usize = IO_LIFECYCLE_CLOSED - 1;

#[derive(Debug, Default)]
struct IoLifecycleInner {
    lifecycle: AtomicUsize,
    wait: Mutex<()>,
    changed: Condvar,
}

/// One revocable I/O generation shared by native execution contexts.
///
/// Acquisition and close update one atomic closed-bit/count word, so an
/// operation is either counted before close or rejected after close. Cloning
/// this handle does not itself count as active I/O.
#[derive(Clone, Debug, Default)]
pub struct IoLifecycle {
    inner: Arc<IoLifecycleInner>,
}

/// Counted permission to execute one file-backed native I/O operation.
#[derive(Debug)]
pub struct IoLifecycleLease {
    lifecycle: IoLifecycle,
}

impl IoLifecycle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn acquire(&self, operation: &'static str) -> Result<IoLifecycleLease> {
        let mut lifecycle = self.inner.lifecycle.load(Ordering::Acquire);
        loop {
            if lifecycle & IO_LIFECYCLE_CLOSED != 0 {
                return Err(io_lifecycle_closed(operation));
            }
            if lifecycle & IO_LIFECYCLE_ACTIVE_MASK == IO_LIFECYCLE_ACTIVE_MASK {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "server I/O generation lease count overflow",
                )
                .with_fact("reason", "io_lifecycle_lease_count_overflow"));
            }
            match self.inner.lifecycle.compare_exchange_weak(
                lifecycle,
                lifecycle + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(IoLifecycleLease { lifecycle: self.clone() }),
                Err(observed) => lifecycle = observed,
            }
        }
    }

    /// Close this generation without blocking. Repeated close calls are safe.
    pub fn close(&self) {
        self.inner.lifecycle.fetch_or(IO_LIFECYCLE_CLOSED, Ordering::AcqRel);
        self.inner.changed.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.inner.lifecycle.load(Ordering::Acquire) & IO_LIFECYCLE_CLOSED != 0
    }

    pub fn active_leases(&self) -> usize {
        self.inner.lifecycle.load(Ordering::Acquire) & IO_LIFECYCLE_ACTIVE_MASK
    }

    /// Close and wait until every lease acquired before close has dropped.
    pub fn close_and_wait_for_idle_until(&self, deadline: Instant, operation: &'static str) -> Result<()> {
        self.close();
        let mut wait = match super::deadline::mutex_lock_until(&self.inner.wait, deadline) {
            Ok(wait) => wait,
            Err(super::deadline::DeadlineMutexLockError::Poisoned) => return Err(io_lifecycle_lock_error(operation)),
            Err(super::deadline::DeadlineMutexLockError::TimedOut) => {
                return Err(io_lifecycle_timeout(operation, None));
            },
        };
        let mut waited_for_idle = false;
        loop {
            let active_leases = self.active_leases();
            if active_leases == 0 {
                if waited_for_idle && super::deadline::deadline_expired(deadline) {
                    return Err(io_lifecycle_timeout(operation, Some(0)));
                }
                return Ok(());
            }
            if super::deadline::deadline_expired(deadline) {
                return Err(io_lifecycle_timeout(operation, Some(active_leases)));
            }
            let Some(remaining) = super::deadline::deadline_remaining(deadline) else {
                return Err(io_lifecycle_timeout(operation, Some(active_leases)));
            };
            waited_for_idle = true;
            let (next, _timeout) = self.inner.changed.wait_timeout(wait, remaining).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "native I/O lifecycle condition variable was poisoned during shutdown",
                )
                .with_fact("reason", "io_lifecycle_wait_poisoned")
            })?;
            wait = next;
        }
    }

    fn release(&self) {
        let _wait = self.inner.wait.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = self.inner.lifecycle.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous & IO_LIFECYCLE_ACTIVE_MASK > 0, "native I/O lifecycle lease count underflow");
        let lifecycle = previous - 1;
        if lifecycle & IO_LIFECYCLE_CLOSED != 0 && lifecycle & IO_LIFECYCLE_ACTIVE_MASK == 0 {
            self.inner.changed.notify_all();
        }
    }
}

impl Drop for IoLifecycleLease {
    fn drop(&mut self) {
        self.lifecycle.release();
    }
}

fn io_lifecycle_closed(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "server I/O generation is closed")
        .with_fact("reason", "io_lifecycle_closed")
}

fn io_lifecycle_lock_error(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, operation, "native I/O lifecycle mutex was poisoned")
        .with_fact("reason", "io_lifecycle_mutex_poisoned")
}

fn io_lifecycle_timeout(operation: &'static str, active_leases: Option<usize>) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for server I/O generation to become idle",
    )
    .with_fact("reason", "io_lifecycle_shutdown_timeout");
    match active_leases {
        Some(active_leases) => error.with_fact("active_io_leases", active_leases),
        None => error,
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn close_rejects_new_io_and_waits_for_prior_lease() {
        let lifecycle = IoLifecycle::new();
        let lease = lifecycle.acquire("test.io.start").unwrap();
        lifecycle.close();
        let error = lifecycle.acquire("test.io.after_close").unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("io_lifecycle_closed"));
        let timeout = lifecycle.close_and_wait_for_idle_until(Instant::now(), "test.io.wait").unwrap_err();
        assert_eq!(timeout.fact_value("reason"), Some("io_lifecycle_shutdown_timeout"));
        assert_eq!(timeout.fact_value("active_io_leases"), Some("1"));
        drop(lease);
        lifecycle.close_and_wait_for_idle_until(Instant::now() + Duration::from_secs(1), "test.io.wait").unwrap();
    }

    #[test]
    fn contended_state_lock_cannot_turn_post_deadline_idle_into_success() {
        let lifecycle = IoLifecycle::new();
        let state = lifecycle.inner.wait.lock().unwrap();
        let deadline = Instant::now() + Duration::from_millis(20);
        let waiting_lifecycle = lifecycle.clone();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let wait = std::thread::spawn(move || {
            result_tx
                .send(waiting_lifecycle.close_and_wait_for_idle_until(deadline, "test.io.contended_shutdown"))
                .unwrap();
        });

        let result = result_rx.recv_timeout(Duration::from_millis(200));
        drop(state);
        wait.join().unwrap();

        let error = result
            .expect("I/O shutdown must return its mutex timeout before the contended lock is released")
            .expect_err("idle observed after the deadline must not be accepted");
        assert_eq!(error.fact_value("reason"), Some("io_lifecycle_shutdown_timeout"));
        assert_eq!(
            lifecycle.acquire("test.io.after_mutex_timeout").unwrap_err().fact_value("reason"),
            Some("io_lifecycle_closed"),
            "mutex timeout must leave Parquet I/O fail-closed"
        );
    }

    #[test]
    fn close_signal_is_nonblocking_and_fail_closed_during_state_lock_contention() {
        let lifecycle = IoLifecycle::new();
        let state = lifecycle.inner.wait.lock().unwrap();
        let closing_lifecycle = lifecycle.clone();
        let (closed_tx, closed_rx) = std::sync::mpsc::channel();
        let close = std::thread::spawn(move || {
            closing_lifecycle.close();
            closed_tx.send(()).unwrap();
        });

        closed_rx.recv_timeout(Duration::from_millis(50)).expect("close signal must not wait for the I/O state mutex");
        drop(state);
        close.join().unwrap();
        assert_eq!(
            lifecycle.acquire("test.io.after_nonblocking_close").unwrap_err().fact_value("reason"),
            Some("io_lifecycle_closed")
        );
    }
}
