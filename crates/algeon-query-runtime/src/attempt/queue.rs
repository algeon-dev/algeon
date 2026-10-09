//! Bounded best-effort report delivery and exact backend counters.
#[cfg(test)]
use std::sync::MutexGuard;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::observability::QueryAttemptReport;

/// Process-lifetime counters for one backend's attempts. Each is monotonic
/// and exact. `attempts_begun` is incremented under the observation permit
/// that `begin_attempt` takes, and every begun attempt publishes one
/// report, so `attempts_begun - reports_ready` is the number of attempts
/// whose report is still pending.
#[derive(Debug, Default)]
pub struct AttemptBackendCounters {
    pub(crate) attempts_begun: AtomicU64,
    pub(crate) reports_ready: AtomicU64,
    pub(crate) completion_queue_pushed: AtomicU64,
    pub(crate) completion_queue_overflowed: AtomicU64,
}

/// Monotonic attempt and abandoned-report delivery counters for one backend.
/// A report becomes ready after cleanup and sealing, independently of whether
/// a caller retains a completion receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptBackendCountersSnapshot {
    /// Attempts created by this backend, including attempts rejected later.
    pub attempts_begun: u64,
    /// Attempts whose immutable terminal report has been published.
    pub reports_ready: u64,
    /// Reports routed into the abandoned-report queue with no active receiver.
    pub completion_queue_pushed: u64,
    /// Abandoned reports rejected because the configured queue was full.
    pub completion_queue_overflowed: u64,
}

impl AttemptBackendCounters {
    pub(crate) fn snapshot(&self) -> AttemptBackendCountersSnapshot {
        AttemptBackendCountersSnapshot {
            attempts_begun: self.attempts_begun.load(Ordering::Acquire),
            reports_ready: self.reports_ready.load(Ordering::Acquire),
            completion_queue_pushed: self.completion_queue_pushed.load(Ordering::Acquire),
            completion_queue_overflowed: self.completion_queue_overflowed.load(Ordering::Acquire),
        }
    }
}

/// Reports whose attempt had no completion receiver when it published.
///
/// Routing is decided under the attempt lock at publication; this queue only
/// stores. Capacity is fixed at construction, and a full queue rejects, which
/// the publisher counts as `completion_queue_overflowed`.
#[derive(Debug)]
pub struct AttemptCompletionQueue {
    capacity: usize,
    queue: Mutex<VecDeque<Arc<QueryAttemptReport>>>,
}

impl AttemptCompletionQueue {
    /// `None` for a zero capacity, so a disabled queue is an absent one.
    pub fn new(capacity: usize) -> Option<Self> {
        (capacity > 0).then(|| Self { capacity, queue: Mutex::new(VecDeque::new()) })
    }

    /// Returns `false` only when the queue is full; a contended lock waits.
    pub(crate) fn try_push(&self, report: Arc<QueryAttemptReport>) -> bool {
        // Blocking is bounded by one VecDeque push or drain. Waiting for the lock
        // keeps a contended push from counting as a capacity overflow.
        let mut queue = self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if queue.len() == self.capacity {
            false
        } else {
            queue.push_back(report);
            true
        }
    }
    pub(crate) fn drain(&self, max: usize) -> Vec<Arc<QueryAttemptReport>> {
        let mut queue = self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = max.min(queue.len());
        queue.drain(..n).collect()
    }

    /// Holds the queue mutex so tests can prove that a contended push waits for
    /// the lock and does not fail as if the queue were full.
    #[cfg(test)]
    pub(super) fn hold_lock_for_test(&self) -> MutexGuard<'_, VecDeque<Arc<QueryAttemptReport>>> {
        self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
