use super::common::{VectorDistanceMetric, VectorRelationBinding};

/// Pairwise distance parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorPairwiseDistanceParams {
    /// First relation defining the result's x axis.
    pub x: VectorRelationBinding,
    /// Second relation defining the result's y axis.
    pub y: VectorRelationBinding,
    /// Metric evaluated for every x and y row pair.
    pub metric: VectorDistanceMetric,
}
