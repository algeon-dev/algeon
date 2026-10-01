//! Attempt-owned auxiliary stream slot accounting.

use std::{sync::Arc, time::Instant};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError},
    time::sleep_until,
};

use super::ExecutionCancellation;
use crate::runtime::deadline::block_until;

/// Attempt-owned capacity for streams outside the admission envelope's root
/// binding.
///
/// The device ledger reserves `source_work_per_attempt` auxiliary slots for
/// every admitted attempt. This gate materializes that reservation locally so
/// one attempt cannot retain another attempt's share of the global auxiliary
/// stream pool.
#[derive(Debug)]
pub(crate) struct AttemptAuxiliarySlots {
    device_ordinal: usize,
    capacity: usize,
    permits: Arc<Semaphore>,
}

/// RAII ownership of one admitted attempt's auxiliary stream allowance.
#[derive(Debug)]
pub(crate) struct AttemptAuxiliaryPermit {
    _permit: OwnedSemaphorePermit,
}

impl AttemptAuxiliarySlots {
    pub(crate) fn new(device_ordinal: usize, capacity: usize) -> Arc<Self> {
        debug_assert!(capacity > 0, "admission profiles normalize auxiliary slot capacity to at least one");
        let capacity = capacity.max(1);
        Arc::new(Self { device_ordinal, capacity, permits: Arc::new(Semaphore::new(capacity)) })
    }

    pub(in crate::runtime) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn acquire_with_cancellation(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        self.acquire_blocking(cancellation, None, operation)
    }

    pub(crate) fn acquire_with_cancellation_deadline(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        self.acquire_blocking(cancellation, Some(deadline), operation)
    }

    /// Takes one free slot, or reports that none is free right now.
    ///
    /// A holder of an attempt auxiliary permit must never block on another
    /// permit of the same attempt, so work that already owns one takes what
    /// is left over instead of waiting for it.
    pub(crate) fn try_acquire(self: &Arc<Self>, operation: &'static str) -> Result<Option<AttemptAuxiliaryPermit>> {
        match Arc::clone(&self.permits).try_acquire_owned() {
            Ok(permit) => Ok(Some(AttemptAuxiliaryPermit { _permit: permit })),
            Err(TryAcquireError::NoPermits) => Ok(None),
            Err(TryAcquireError::Closed) => Err(self.closed_error(operation)),
        }
    }

    /// Asynchronously acquires one attempt-local auxiliary slot without
    /// parking the caller's Tokio worker while another fragment owns it.
    pub(crate) async fn acquire_with_cancellation_deadline_async(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        loop {
            cancellation.check(operation)?;
            if let Some(permit) = self.try_acquire(operation)? {
                return Ok(permit);
            }
            if Instant::now() >= deadline {
                return Err(self.exhausted_error(operation, true));
            }
            tokio::select! {
                biased;
                permit = Arc::clone(&self.permits).acquire_owned() => {
                    return permit
                        .map(|permit| AttemptAuxiliaryPermit { _permit: permit })
                        .map_err(|_closed| self.closed_error(operation));
                },
                () = cancellation.cancelled_or_expired() => {},
                () = sleep_until(deadline.into()) => {},
            }
        }
    }

    /// Blocks the calling thread until a slot frees, cancellation is
    /// requested, or the earlier of `deadline` and the cancellation deadline
    /// passes. Each of them wakes the thread directly.
    fn acquire_blocking(
        self: &Arc<Self>,
        cancellation: &ExecutionCancellation,
        deadline: Option<Instant>,
        operation: &'static str,
    ) -> Result<AttemptAuxiliaryPermit> {
        let wait_deadline = match (deadline, cancellation.deadline()) {
            (Some(deadline), Some(cancellation_deadline)) => Some(deadline.min(cancellation_deadline)),
            (deadline, cancellation_deadline) => deadline.or(cancellation_deadline),
        };
        loop {
            cancellation.check(operation)?;
            if let Some(permit) = self.try_acquire(operation)? {
                return Ok(permit);
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(self.exhausted_error(operation, true));
            }
            let acquired = block_until(
                async {
                    tokio::select! {
                        biased;
                        permit = Arc::clone(&self.permits).acquire_owned() => Some(permit),
                        () = cancellation.cancel_requested() => None,
                    }
                },
                wait_deadline,
            );
            if let Some(Some(permit)) = acquired {
                return permit
                    .map(|permit| AttemptAuxiliaryPermit { _permit: permit })
                    .map_err(|_closed| self.closed_error(operation));
            }
        }
    }

    fn in_use(&self) -> usize {
        self.capacity.saturating_sub(self.permits.available_permits())
    }

    fn exhausted_error(&self, operation: &'static str, deadline_elapsed: bool) -> QueryError {
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            operation,
            "admitted attempt auxiliary-stream capacity is exhausted",
        )
        .with_fact("device_ordinal", self.device_ordinal)
        .with_fact("attempt_auxiliary_capacity", self.capacity)
        .with_fact("attempt_auxiliary_in_use", self.in_use())
        .with_fact("deadline_elapsed", deadline_elapsed)
        .with_fact(
            "reason",
            if deadline_elapsed { "attempt_auxiliary_slot_wait_timeout" } else { "attempt_auxiliary_slots_exhausted" },
        )
    }

    /// The slots are never closed; a closed semaphore would mean the attempt's
    /// envelope accounting broke.
    fn closed_error(&self, operation: &'static str) -> QueryError {
        QueryError::new(QueryErrorCode::Internal, operation, "attempt auxiliary slots were closed")
            .with_fact("device_ordinal", self.device_ordinal)
            .with_fact("reason", "attempt_auxiliary_slots_closed")
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[cfg_attr(not(test), expect(dead_code, reason = "used by in-crate auxiliary-slot tests"))]
    fn in_use_for_test(&self) -> usize {
        self.in_use()
    }
}

#[cfg(test)]
mod attempt_auxiliary_slots_tests {
    use std::time::Duration;

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
