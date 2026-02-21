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
}
