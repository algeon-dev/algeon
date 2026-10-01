use std::{
    pin::pin,
    sync::{Arc, Mutex, MutexGuard, TryLockError},
    task::{Context, Poll, Wake, Waker},
    thread::{self, Thread},
    time::{Duration, Instant},
};

/// Failure to acquire a synchronous runtime-state mutex under an absolute
/// caller-owned deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeadlineMutexLockError {
    Poisoned,
    TimedOut,
}

/// Returns whether an absolute monotonic deadline has been reached.
pub(crate) fn deadline_expired(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

/// Returns positive time remaining before an absolute monotonic deadline.
pub(crate) fn deadline_remaining(deadline: Instant) -> Option<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    (!remaining.is_zero()).then_some(remaining)
}

/// Acquires a standard-library mutex without allowing lock contention to
/// create a second grace period beyond `deadline`.
///
/// One immediate acquisition attempt is made even at an expired deadline so
/// teardown callers can still close an uncontended gate. Callers must check
/// [`deadline_expired`] before accepting successful quiescence.
pub(crate) fn mutex_lock_until<T>(
    mutex: &Mutex<T>,
    deadline: Instant,
) -> std::result::Result<MutexGuard<'_, T>, DeadlineMutexLockError> {
    loop {
        match mutex.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(_)) => return Err(DeadlineMutexLockError::Poisoned),
            Err(TryLockError::WouldBlock) if deadline_expired(deadline) => {
                return Err(DeadlineMutexLockError::TimedOut);
            },
            Err(TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
}

/// Blocks the calling thread until `future` completes, or returns `None` once
/// `deadline` passes.
///
/// The future's wakers unpark this thread, so the wait ends when the awaited
/// event fires instead of on a poll interval. The future must not depend on
/// a Tokio timer; the deadline bounds the wait instead.
pub(crate) fn block_until<F: Future>(future: F, deadline: Option<Instant>) -> Option<F::Output> {
    let waker = Waker::from(Arc::new(ThreadUnparker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return Some(output);
        }
        match deadline {
            None => thread::park(),
            Some(deadline) => thread::park_timeout(deadline_remaining(deadline)?),
        }
    }
}

struct ThreadUnparker(Thread);

impl Wake for ThreadUnparker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use super::*;

    #[test]
    fn uncontended_expired_lock_is_available_for_best_effort_close() {
        let state = Mutex::new(false);
        let mut guard = mutex_lock_until(&state, Instant::now()).expect("uncontended close attempt");
        *guard = true;
        assert!(deadline_expired(Instant::now()));
    }

    #[test]
    fn contended_lock_does_not_wait_for_release_after_deadline() {
        let state = Arc::new(Mutex::new(()));
        let held = state.lock().unwrap();
        let deadline = Instant::now() + Duration::from_millis(20);
        let error = mutex_lock_until(&state, deadline).unwrap_err();
        assert_eq!(error, DeadlineMutexLockError::TimedOut);
        assert!(deadline_expired(deadline));
        drop(held);
    }
}
