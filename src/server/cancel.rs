use dashmap::DashMap;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Default)]
pub struct QueryCancellation {
    tokens: DashMap<String, CancellationToken>,
}

impl QueryCancellation {
    pub fn insert(&self, query_id: String, token: CancellationToken) {
        self.tokens.insert(query_id, token);
    }

    pub fn cancel(&self, query_id: &str) -> bool {
        self.tokens.get(query_id).map(|token| token.cancel()).is_some()
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
