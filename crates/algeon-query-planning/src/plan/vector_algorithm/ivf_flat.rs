use super::{
    common::{VectorDistanceMetric, VectorRelationBinding},
    definition::CuvsOperation,
};

/// Query-local IVF-Flat build and search parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorIvfFlatParams {
    pub dataset: VectorRelationBinding,
    pub queries: VectorRelationBinding,
    pub k: usize,
    pub metric: VectorDistanceMetric,
    pub n_lists: u32,
    pub n_probes: u32,
    pub kmeans_n_iters: u32,
    pub kmeans_trainset_fraction: f64,
}

impl VectorIvfFlatParams {
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
