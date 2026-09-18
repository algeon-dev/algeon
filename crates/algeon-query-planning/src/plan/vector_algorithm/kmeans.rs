use std::fmt;

use super::{common::VectorRelationBinding, definition::CuvsOperation};

/// K-means centroid initialization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorKMeansInit {
    /// K-means++ initialization.
    KMeansPlusPlus,
    /// Random initialization.
    Random,
    /// Initial centroid vectors supplied by a relation binding.
    Relation,
}

impl VectorKMeansInit {
    /// Stable diagnostic label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::KMeansPlusPlus => "kmeans_plus_plus",
            Self::Random => "random",
            Self::Relation => "relation",
        }
    }
}

impl fmt::Display for VectorKMeansInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// K-means fit/predict parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorKMeansParams {
    /// Training and prediction input relation.
    pub input: VectorRelationBinding,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Maximum fit iterations.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Number of independent initializations.
    pub n_init: u32,
    /// Centroid initialization policy.
    pub init: VectorKMeansInit,
    /// Optional relation-backed initial centroids.
    pub initial_centroids: Option<VectorRelationBinding>,
    /// Distance metric used for cluster assignment.
    pub metric: VectorKMeansMetric,
}

impl VectorKMeansParams {
    /// Construct K-means parameters with stable defaults.
    pub fn new(input: VectorRelationBinding, n_clusters: usize) -> Self {
        let descriptor = CuvsOperation::KMeans.descriptor();
        Self {
            input,
            n_clusters,
            max_iterations: descriptor
                .option("max_iter")
                .and_then(|option| option.kind.default_u64())
                .expect("canonical KMeans max_iter default") as u32,
            tolerance: descriptor
                .option("tol")
                .and_then(|option| option.kind.default_f64())
                .expect("canonical KMeans tol default"),
            n_init: descriptor
                .option("n_init")
                .and_then(|option| option.kind.default_u64())
                .expect("canonical KMeans n_init default") as u32,
            init: match descriptor.option("init").and_then(|option| option.kind.default_str()) {
                Some("random") => VectorKMeansInit::Random,
                Some("kmeans++") => VectorKMeansInit::KMeansPlusPlus,
                _ => panic!("canonical KMeans init default"),
            },
            initial_centroids: None,
            metric: match descriptor.option("metric").and_then(|option| option.kind.default_str()) {
                Some("l2_sqrt_expanded") => VectorKMeansMetric::L2SqrtExpanded,
                Some("l2_expanded") => VectorKMeansMetric::L2Expanded,
                _ => panic!("canonical KMeans metric default"),
            },
        }
    }

    /// Use relation-backed initial centroids.
    pub fn with_initial_centroids(mut self, centroids: VectorRelationBinding) -> Self {
        self.init = VectorKMeansInit::Relation;
        self.initial_centroids = Some(centroids);
        self
    }
}

/// K-means admits only the native L2 family; this is intentionally not the
/// broader kNN distance enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorKMeansMetric {
    L2Expanded,
    L2SqrtExpanded,
}

impl VectorKMeansMetric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L2Expanded => "l2_expanded",
            Self::L2SqrtExpanded => "l2_sqrt_expanded",
        }
    }
}

impl fmt::Display for VectorKMeansMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
