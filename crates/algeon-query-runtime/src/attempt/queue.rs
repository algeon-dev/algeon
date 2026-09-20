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

#[derive(Debug, Default)]
pub struct AttemptBackendCounters {
    pub attempts_begun: AtomicU64,
    pub reports_ready: AtomicU64,
    pub completion_queue_pushed: AtomicU64,
    pub completion_queue_overflowed: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptBackendCountersSnapshot {
    pub attempts_begun: u64,
    pub reports_ready: u64,
    pub completion_queue_pushed: u64,
    pub completion_queue_overflowed: u64,
}

impl AttemptBackendCounters {
    pub fn snapshot(&self) -> AttemptBackendCountersSnapshot {
        AttemptBackendCountersSnapshot {
            attempts_begun: self.attempts_begun.load(Ordering::Acquire),
            reports_ready: self.reports_ready.load(Ordering::Acquire),
            completion_queue_pushed: self.completion_queue_pushed.load(Ordering::Acquire),
            completion_queue_overflowed: self.completion_queue_overflowed.load(Ordering::Acquire),
        }
    }
}

#[derive(Debug)]
pub struct AttemptCompletionQueue {
    capacity: usize,
    queue: Mutex<VecDeque<Arc<QueryAttemptReport>>>,
}

impl AttemptCompletionQueue {
    pub fn new(capacity: usize) -> Option<Self> {
        (capacity > 0).then(|| Self { capacity, queue: Mutex::new(VecDeque::new()) })
    }
    #[doc(hidden)]
    pub fn try_push(&self, report: Arc<QueryAttemptReport>) -> bool {
        // Blocking is bounded by one VecDeque push or drain; a failed try_lock
        // used to miscount a contended push as a capacity overflow.
        let mut queue = self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if queue.len() == self.capacity {
            false
        } else {
            queue.push_back(report);
            true
        }
    }
    pub fn drain(&self, max: usize) -> Vec<Arc<QueryAttemptReport>> {
        let mut queue = self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = max.min(queue.len());
        queue.drain(..n).collect()
    }

    /// Holds the queue mutex so tests can prove a contended push waits for
    /// capacity instead of failing as if the queue were full.
    #[cfg(test)]
    pub(super) fn hold_lock_for_test(&self) -> MutexGuard<'_, VecDeque<Arc<QueryAttemptReport>>> {
        self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
