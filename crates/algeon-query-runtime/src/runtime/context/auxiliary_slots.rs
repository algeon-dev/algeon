//! Attempt-owned auxiliary stream slot accounting.

use std::{
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::ExecutionCancellation;

const ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Attempt-owned capacity for streams outside the admission envelope's root
/// binding.
///
/// The device ledger reserves `source_work_per_attempt` auxiliary slots for
/// every admitted attempt. This gate materializes that reservation locally so
/// one attempt cannot retain another attempt's share of the global auxiliary
/// stream pool.
#[derive(Debug)]
#[doc(hidden)]
pub struct AttemptAuxiliarySlots {
    device_ordinal: usize,
    capacity: usize,
    state: Mutex<AttemptAuxiliarySlotState>,
    available: Condvar,
    async_available: tokio::sync::Notify,
}

#[derive(Debug)]
struct AttemptAuxiliarySlotState {
    in_use: usize,
}

/// RAII ownership of one admitted attempt's auxiliary stream allowance.
#[derive(Debug)]
#[doc(hidden)]
pub struct AttemptAuxiliaryPermit {
    slots: Arc<AttemptAuxiliarySlots>,
}

impl AttemptAuxiliarySlots {
    #[doc(hidden)]
    pub fn new(device_ordinal: usize, capacity: usize) -> Arc<Self> {
        debug_assert!(capacity > 0, "admission profiles normalize auxiliary slot capacity to at least one");
        Arc::new(Self {
            device_ordinal,
            capacity: capacity.max(1),
            state: Mutex::new(AttemptAuxiliarySlotState { in_use: 0 }),
            available: Condvar::new(),
            async_available: tokio::sync::Notify::new(),
        })
    }

    pub(in crate::runtime) fn capacity(&self) -> usize {
        self.capacity
    }

    #[doc(hidden)]
    pub fn acquire_with_cancellation(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        self.acquire_inner(Some(cancellation), None, operation)
    }

    #[doc(hidden)]
    pub fn acquire_with_cancellation_deadline(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        self.acquire_inner(Some(cancellation), Some(deadline), operation)
    }

    /// Takes one free slot, or reports that none is free right now.
    ///
    /// A holder of an attempt auxiliary permit must never block on another
    /// permit of the same attempt, so work that already owns one takes what
    /// is left over instead of waiting for it.
    #[doc(hidden)]
    pub fn try_acquire(self: &Arc<Self>, operation: &'static str) -> Result<Option<AttemptAuxiliaryPermit>> {
        let mut state = self.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "attempt auxiliary-slot mutex was poisoned")
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "attempt_auxiliary_slots_poisoned")
        })?;
        if state.in_use < self.capacity {
            state.in_use += 1;
            return Ok(Some(AttemptAuxiliaryPermit { slots: Arc::clone(self) }));
        }
        Ok(None)
    }

    /// Asynchronously acquires one attempt-local auxiliary slot without
    /// parking the caller's Tokio worker while another fragment owns it.
    #[doc(hidden)]
    pub async fn acquire_with_cancellation_deadline_async(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        loop {
            cancellation.check(operation)?;
            // Register before observing availability so a permit return in
            // this interval cannot be lost.
            let available = self.async_available.notified();
            let in_use = {
                let mut state = self.state.lock().map_err(|_error| {
                    QueryError::new(QueryErrorCode::Internal, operation, "attempt auxiliary-slot mutex was poisoned")
                        .with_fact("device_ordinal", self.device_ordinal)
                        .with_fact("reason", "attempt_auxiliary_slots_poisoned")
                })?;
                if state.in_use < self.capacity {
                    state.in_use += 1;
                    return Ok(AttemptAuxiliaryPermit { slots: Arc::clone(self) });
                }
                state.in_use
            };
            if Instant::now() >= deadline {
                return Err(self.exhausted_error(operation, in_use, true));
            }
            let wait =
                deadline.saturating_duration_since(Instant::now()).min(ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL);
            let _ = tokio::time::timeout(wait, available).await;
        }
    }

    fn acquire_inner(
        self: &Arc<Self>,
        cancellation: Option<&ExecutionCancellation>,
        deadline: Option<Instant>,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        loop {
            if let Some(cancellation) = cancellation {
                cancellation.check(operation)?;
            }
            let mut state = self.state.lock().map_err(|_error| {
                QueryError::new(QueryErrorCode::Internal, operation, "attempt auxiliary-slot mutex was poisoned")
                    .with_fact("device_ordinal", self.device_ordinal)
                    .with_fact("reason", "attempt_auxiliary_slots_poisoned")
            })?;
            if state.in_use < self.capacity {
                state.in_use += 1;
                return Ok(AttemptAuxiliaryPermit { slots: Arc::clone(self) });
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(self.exhausted_error(operation, state.in_use, true));
            }
            let wait = deadline.map_or(ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL, |deadline| {
                deadline.saturating_duration_since(Instant::now()).min(ATTEMPT_AUXILIARY_SLOT_WAIT_POLL_INTERVAL)
            });
            let (_state, _timeout) = self.available.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "attempt auxiliary-slot mutex was poisoned while waiting for capacity",
                )
                .with_fact("device_ordinal", self.device_ordinal)
                .with_fact("reason", "attempt_auxiliary_slots_poisoned")
            })?;
        }
    }

    fn exhausted_error(&self, operation: &'static str, in_use: usize, deadline_elapsed: bool) -> QueryError {
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            operation,
            "admitted attempt auxiliary-stream capacity is exhausted",
        )
        .with_fact("device_ordinal", self.device_ordinal)
        .with_fact("attempt_auxiliary_capacity", self.capacity)
        .with_fact("attempt_auxiliary_in_use", in_use)
        .with_fact("deadline_elapsed", deadline_elapsed)
        .with_fact(
            "reason",
            if deadline_elapsed { "attempt_auxiliary_slot_wait_timeout" } else { "attempt_auxiliary_slots_exhausted" },
        )
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[cfg_attr(not(test), expect(dead_code, reason = "used by in-crate auxiliary-slot tests"))]
    fn in_use_for_test(&self) -> usize {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).in_use
    }
}

impl Drop for AttemptAuxiliaryPermit {
    fn drop(&mut self) {
        let mut state = self.slots.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.in_use > 0, "attempt auxiliary-slot permit underflow");
        state.in_use = state.in_use.saturating_sub(1);
        self.slots.available.notify_one();
        self.slots.async_available.notify_one();
    }
}

#[cfg(test)]
mod attempt_auxiliary_slots_tests {
    use super::*;

    #[test]
    fn execution_cancellation_preserves_deadline_across_clones_and_keeps_caller_cancel_first() {
        let deadline = Instant::now() - Duration::from_millis(1);
        let cancellation = ExecutionCancellation::with_deadline(Some(deadline));
        let clone = cancellation.clone();
        assert_eq!(clone.deadline(), Some(deadline));
        assert!(clone.is_expired());
        assert!(clone.is_cancelled(), "an elapsed deadline must stop native safe boundaries");
        let deadline_error = clone
            .check("query_engine.test.execution_cancellation_deadline")
            .expect_err("expired execution deadline must stop a safe boundary");
        assert_eq!(deadline_error.fact_value("reason"), Some("deadline_expired"));

        cancellation.cancel();
        let caller_error = clone
            .check("query_engine.test.execution_cancellation_caller_cancel")
            .expect_err("explicit caller cancellation remains terminal");
        assert_eq!(caller_error.fact_value("reason"), None, "caller cancellation takes precedence over expiry");
    }

    #[test]
    fn held_slot_rejects_cancelled_and_expired_waiters_then_releases() {
        let slots = AttemptAuxiliarySlots::new(7, 1);
        let active = ExecutionCancellation::new();
        let held = slots
            .acquire_with_cancellation(&active, "query_engine.test.attempt_auxiliary_slots")
            .expect("first auxiliary permit fits the attempt envelope");
        assert_eq!(slots.in_use_for_test(), 1);

        let cancelled = ExecutionCancellation::new();
        cancelled.cancel();
        let cancelled_error = slots
            .acquire_with_cancellation(&cancelled, "query_engine.test.attempt_auxiliary_slots")
            .expect_err("cancelled waiter must not exceed the attempt envelope");
        assert_eq!(cancelled_error.code(), QueryErrorCode::Cancelled);

        let deadline_error = slots
            .acquire_with_cancellation_deadline(&active, Instant::now(), "query_engine.test.attempt_auxiliary_slots")
            .expect_err("expired waiter must not exceed the attempt envelope");
        assert_eq!(deadline_error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(deadline_error.fact_value("reason"), Some("attempt_auxiliary_slot_wait_timeout"));
        assert_eq!(deadline_error.fact_value("attempt_auxiliary_capacity"), Some("1"));
        assert_eq!(deadline_error.fact_value("attempt_auxiliary_in_use"), Some("1"));
        assert_eq!(deadline_error.fact_value("deadline_elapsed"), Some("true"));

        drop(held);
        let released = slots
            .acquire_with_cancellation(&active, "query_engine.test.attempt_auxiliary_slots")
            .expect("dropping the permit returns this attempt's capacity");
        assert_eq!(slots.in_use_for_test(), 1);
        drop(released);
        assert_eq!(slots.in_use_for_test(), 0);
    }

    #[test]
    fn each_attempt_keeps_its_own_auxiliary_share() {
        let first_attempt = AttemptAuxiliarySlots::new(3, 1);
        let second_attempt = AttemptAuxiliarySlots::new(3, 1);
        let cancellation = ExecutionCancellation::new();
        let first = first_attempt
            .acquire_with_cancellation(&cancellation, "query_engine.test.attempt_auxiliary_slots")
            .expect("first attempt reserves its only slot");

        let first_exhausted = first_attempt
            .acquire_with_cancellation_deadline(
                &cancellation,
                Instant::now(),
                "query_engine.test.attempt_auxiliary_slots",
            )
            .expect_err("one attempt cannot borrow another attempt's auxiliary share");
        assert_eq!(first_exhausted.fact_value("reason"), Some("attempt_auxiliary_slot_wait_timeout"));

        let second = second_attempt
            .acquire_with_cancellation(&cancellation, "query_engine.test.attempt_auxiliary_slots")
            .expect("second attempt retains its independently admitted share");
        assert_eq!(first_attempt.in_use_for_test(), 1);
        assert_eq!(second_attempt.in_use_for_test(), 1);
        drop(first);
        drop(second);
    }
}
