use super::{
    common::{VectorDistanceMetric, VectorRelationBinding},
    definition::CuvsOperation,
};

/// Query-local IVF-Flat build and search parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorIvfFlatParams {
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
}

impl VectorIvfFlatParams {
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
        let descriptor = CuvsOperation::IvfFlat.descriptor();
        Self {
            dataset,
            queries,
            k,
            metric,
            n_lists,
            n_probes: descriptor
                .option("n_probes")
                .and_then(|option| option.kind.default_u64())
                .expect("IVF-Flat n_probes default") as u32,
            kmeans_n_iters: descriptor
                .option("kmeans_n_iters")
                .and_then(|option| option.kind.default_u64())
                .expect("IVF-Flat kmeans_n_iters default") as u32,
            kmeans_trainset_fraction: descriptor
                .option("kmeans_trainset_fraction")
                .and_then(|option| option.kind.default_f64())
                .expect("IVF-Flat training fraction default"),
        }
    }
}
