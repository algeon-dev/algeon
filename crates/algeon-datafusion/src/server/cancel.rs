//! Per-statement cancellation registry for the Flight SQL server.
//!
//! # Cancellation contract
//!
//! Each in-flight statement registers a Flight [`CancellationToken`] and its
//! attempt's [`AttemptCancellation`], the one token native execution observes.
//! The Flight token stops `DoGet`; the attempt token is checked cooperatively
//! at query-engine boundaries and does not preempt an in-flight libcudf/CUDA
//! call.
//!
//! `cancel()` cancels the **attempt first, then the Flight token**. The
//! ordering keeps the next native boundary from producing another batch after
//! the Flight stream starts tearing down.
//!
//! # Lifetime
//!
//! Entries are inserted after a statement ticket is consumed, before any
//! execution branch can return. `QueryExecutionGuard::drop` removes the entry
//! and, if the stream was dropped before a terminal result, cancels both the
//! attempt and the Flight token. Before a ticket is issued, shutdown still
//! reaches the attempt through the planning `shutdown_token` arm.

use dashmap::DashMap;
use query_runtime::attempt::AttemptCancellation;
use tokio_util::sync::CancellationToken;

/// Registry of in-flight statement cancellation handles, keyed by query id.
///
/// See the module-level docs for the cancellation contract.
#[derive(Debug, Default)]
pub struct QueryCancellation {
    tokens: DashMap<String, QueryCancellationEntry>,
}

/// Paired Flight stream token and attempt cancellation for one registered
/// statement. Both halves are flipped together by [`QueryCancellation::cancel`].
#[derive(Debug)]
struct QueryCancellationEntry {
    flight_token: CancellationToken,
    attempt_cancellation: Option<AttemptCancellation>,
}

impl QueryCancellation {
    /// Register `query_id` with the Flight stream token and the attempt
    /// cancellation observed by `query-engine`.
    pub fn insert(
        &self,
        query_id: String,
        flight_token: CancellationToken,
        attempt_cancellation: Option<AttemptCancellation>,
    ) {
        self.tokens.insert(query_id, QueryCancellationEntry { flight_token, attempt_cancellation });
    }

    /// Cancel the registered statement, if any.
    ///
    /// Returns `true` when `query_id` was registered. Cancellation is
    /// cooperative; an in-flight libcudf/CUDA call is not preempted.
    pub fn cancel(&self, query_id: &str) -> bool {
        self.tokens
            .get(query_id)
            .map(|entry| {
                if let Some(attempt_cancellation) = &entry.attempt_cancellation {
                    attempt_cancellation.cancel();
                }
                entry.flight_token.cancel();
            })
            .is_some()
    }

    pub fn remove(&self, query_id: &str) {
        self.tokens.remove(query_id);
    }

    /// Number of statements currently registered for cancellation. Used by
    /// graceful shutdown to report how many in-flight queries it is draining.
    /// Statements that have been planned but not yet inserted (before ticket
    /// consumption or during the narrow handoff into `execute_ticket`) are not
    /// counted; cancellation of those is handled by the shared shutdown token in
    /// [`crate::server::flight::DatafusionAlgeonFlightSqlService`].
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}
