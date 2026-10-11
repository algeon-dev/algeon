use super::{
    common::{VectorDistanceMetric, VectorRelationBinding},
    definition::CuvsOperation,
};

/// PQ codebook training scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorIvfPqCodebookKind {
    /// Trains one codebook for each product-quantization subspace.
    PerSubspace,
    /// Trains codebooks separately for inverted-list clusters.
    PerCluster,
}

/// Query-local IVF-PQ build and search parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorIvfPqParams {
    /// Dense rows indexed into inverted lists.
    pub dataset: VectorRelationBinding,
    /// Dense query rows matched against the dataset.
    pub queries: VectorRelationBinding,
    /// Number of neighbors returned per query row.
    pub k: usize,
    /// Distance or similarity used to rank neighbors.
    pub metric: VectorDistanceMetric,
    /// Number of inverted lists; must not exceed dataset row count.
    pub n_lists: u32,
    /// Lists searched per query; must not exceed `n_lists`.
    pub n_probes: u32,
    /// Maximum iterations for fitting list centers.
    pub kmeans_n_iters: u32,
    /// Fraction of dataset rows used to fit list centers, in `(0, 1]`.
    pub kmeans_trainset_fraction: f64,
    /// Compressed dimensions; zero selects them automatically. Nonzero `pq_dim * pq_bits` must be divisible by eight.
    pub pq_dim: u32,
    /// Bits in each product-quantization code.
    pub pq_bits: u32,
    /// Scope over which product-quantization codebooks are trained.
    pub codebook_kind: VectorIvfPqCodebookKind,
}

impl VectorIvfPqParams {
    /// Uses registry defaults for list training and search tuning.
    ///
    /// The supplied bindings, neighbor count, metric, and list count are retained
    /// without validation.
    pub fn new(
        dataset: VectorRelationBinding,
        queries: VectorRelationBinding,
        k: usize,
        metric: VectorDistanceMetric,
        n_lists: u32,
    ) -> Self {
        let descriptor = CuvsOperation::IvfPq.descriptor();
        let int = |name| {
            descriptor
                .option(name)
                .and_then(|option| option.kind.default_u64())
                .expect("canonical IVF-PQ integer default") as u32
        };
        Self {
            dataset,
            queries,
            k,
            metric,
            n_lists,
            n_probes: int("n_probes"),
            kmeans_n_iters: int("kmeans_n_iters"),
            kmeans_trainset_fraction: descriptor
                .option("kmeans_trainset_fraction")
                .and_then(|option| option.kind.default_f64())
                .expect("canonical IVF-PQ fraction default"),
            pq_dim: int("pq_dim"),
            pq_bits: int("pq_bits"),
            codebook_kind: VectorIvfPqCodebookKind::PerSubspace,
        }
    }
}
