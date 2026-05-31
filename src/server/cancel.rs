//! Per-statement cancellation registry for the Flight SQL server.
//!
//! # Cancellation contract
//!
//! Each in-flight statement registers two coupled handles:
//!
//! - A tonic [`CancellationToken`] that terminates the Flight `DoGet` stream
//!   and unblocks any `tokio::select!` arms guarding planning/execution
//!   futures. This is the only handle the gRPC layer observes directly.
//! - A native [`ExecutionCancellation`] flag (an `Arc<AtomicBool>`) observed
//!   *cooperatively* by [`nexus_query_engine`] at native execution boundaries
//!   (chunk loops, retries, source reads, compiled tasks, generic nodes,
//!   top-level execute). Setting it prevents the next boundary from starting
//!   but does **not** preempt a libcudf/CUDA call already in flight.
//!
//! `cancel()` flips the **native flag first, then the Flight token**. The
//! ordering matters: by the time the Flight stream wakes up from
//! `token.cancelled()` and starts to tear down, the native side has already
//! observed cancellation, so the next pipeline tick short-circuits with a
//! `QueryErrorCode::Cancelled` instead of producing another batch the client
//! will never read.
//!
//! # Lifetime
//!
//! Entries are inserted in `execute_statement_stream` after admission and
//! removed by `QueryExecutionGuard::drop` (which also aborts the
//! shutdown-token → native-cancellation bridge task). There is a small
//! pre-ticket window in `statement_flight_info` during which a freshly
//! constructed native cancellation flag is not yet reachable via
//! `do_action_cancel_query` — shutdown still cancels it via the parent
//! `shutdown_token` arm of the planning `select!`.

use dashmap::DashMap;
use nexus_query_engine::ExecutionCancellation;
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
}

impl QueryCancellation {
    /// Register `query_id` with the Flight stream token and the native
    /// cancellation flag observed by `nexus-query-engine`.
    pub fn insert(
        &self,
        query_id: String,
        flight_token: CancellationToken,
        native_cancellation: ExecutionCancellation,
    ) {
        self.tokens.insert(query_id, QueryCancellationEntry { flight_token, native_cancellation });
    }

    /// Cancel the registered statement, if any.
    ///
    /// Returns `true` if `query_id` was registered. The native cancellation
    /// flag is flipped first so a subsequent native boundary check
    /// short-circuits with [`nexus_query_engine::QueryErrorCode::Cancelled`];
    /// the Flight stream token is then flipped so `DoGet`'s
    /// `.take_until(token.cancelled_owned())` terminates the response stream.
    /// Cancellation is cooperative — a libcudf/CUDA call already in flight is
    /// not preempted.
    pub fn cancel(&self, query_id: &str) -> bool {
        self.tokens
            .get(query_id)
            .map(|entry| {
                entry.native_cancellation.cancel();
                entry.flight_token.cancel();
            })
            .is_some()
    }

    pub fn remove(&self, query_id: &str) {
        self.tokens.remove(query_id);
    }

    /// Number of statements currently registered for cancellation. Used by
    /// graceful shutdown to report how many in-flight queries it is draining.
    /// Statements that have been admitted but not yet inserted (a brief
    /// window inside `execute_ticket`) are not counted; cancellation of those
    /// is handled by the shared shutdown token in
    /// [`crate::server::flight::DatafusionCudfFlightSqlService`].
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}
