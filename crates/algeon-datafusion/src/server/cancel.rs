//! Per-statement cancellation registry for the Flight SQL server.
//!
//! # Cancellation contract
//!
//! Each in-flight statement registers a Flight [`CancellationToken`] and a
//! native [`ExecutionCancellation`] flag. The token stops `DoGet`; the native
//! flag is checked cooperatively at query-engine boundaries and does not preempt
//! an in-flight libcudf/CUDA call.
//!
//! `cancel()` flips the **native flag first, then the Flight token**. The
//! ordering keeps the next native boundary from producing another batch after
//! the Flight stream starts tearing down.
//!
//! # Lifetime
//!
//! Entries are inserted after a statement ticket is consumed, before any
//! execution branch can return. `QueryExecutionGuard::drop` removes the entry
//! and, if the stream was dropped before a terminal result, cancels both the
//! native flag and Flight token. Before a ticket is issued, shutdown still
//! reaches the native flag through the planning `shutdown_token` arm.

use dashmap::DashMap;
use query_engine::{attempt::AttemptCancellation, runtime::ExecutionCancellation};
use tokio_util::sync::CancellationToken;

/// Registry of in-flight statement cancellation handles, keyed by query id.
///
/// See the module-level docs for the cancellation contract.
#[derive(Debug, Default)]
pub struct QueryCancellation {
    tokens: DashMap<String, QueryCancellationEntry>,
}

/// Paired Flight stream token and native execution cancellation flag for one
/// registered statement. Both halves are flipped together by [`QueryCancellation::cancel`].
#[derive(Debug)]
struct QueryCancellationEntry {
    flight_token: CancellationToken,
    native_cancellation: ExecutionCancellation,
    attempt_cancellation: Option<AttemptCancellation>,
}

impl QueryCancellation {
    /// Register `query_id` with the Flight stream token and the native
    /// cancellation flag observed by `query-engine`.
    pub fn insert(
        &self,
        query_id: String,
        flight_token: CancellationToken,
        native_cancellation: ExecutionCancellation,
        attempt_cancellation: Option<AttemptCancellation>,
    ) {
        self.tokens
            .insert(query_id, QueryCancellationEntry { flight_token, native_cancellation, attempt_cancellation });
    }

    /// Cancel the registered statement, if any.
    ///
    /// Returns `true` when `query_id` was registered. Cancellation is
    /// cooperative; an in-flight libcudf/CUDA call is not preempted.
    pub fn cancel(&self, query_id: &str) -> bool {
        self.tokens
            .get(query_id)
            .map(|entry| {
                entry.native_cancellation.cancel();
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
