use super::common::{VectorDistanceMetric, VectorRelationBinding};

/// Exact brute-force kNN parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorBruteForceKnnParams {
    /// Indexed dataset relation.
    pub dataset: VectorRelationBinding,
    /// Query-vector relation.
    pub queries: VectorRelationBinding,
    /// Number of neighbors emitted per query row.
    pub k: usize,
    /// Distance metric.
    pub metric: VectorDistanceMetric,
}

impl VectorBruteForceKnnParams {
    /// Construct exact brute-force kNN parameters.
    pub fn new(
        dataset: VectorRelationBinding,
        queries: VectorRelationBinding,
        k: usize,
        metric: VectorDistanceMetric,
    ) -> Self {
        Self { dataset, queries, k, metric }
    }
}
