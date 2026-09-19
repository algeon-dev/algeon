use std::{
    sync::{Mutex, MutexGuard, TryLockError},
    time::{Duration, Instant},
};

/// Failure to acquire a synchronous runtime-state mutex under an absolute
/// caller-owned deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum DeadlineMutexLockError {
    Poisoned,
    TimedOut,
}

/// Returns whether an absolute monotonic deadline has been reached.
#[doc(hidden)]
pub fn deadline_expired(deadline: Instant) -> bool {
    Instant::now() >= deadline
}

/// Returns positive time remaining before an absolute monotonic deadline.
#[doc(hidden)]
pub fn deadline_remaining(deadline: Instant) -> Option<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    (!remaining.is_zero()).then_some(remaining)
}

/// Acquires a standard-library mutex without allowing lock contention to
/// create a second grace period beyond `deadline`.
///
/// One immediate acquisition attempt is made even at an expired deadline so
/// teardown callers can still close an uncontended gate. Callers must check
/// [`deadline_expired`] before accepting successful quiescence.
#[doc(hidden)]
pub fn mutex_lock_until<T>(
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
