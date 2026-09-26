use super::{
    common::{VectorDistanceMetric, VectorRelationBinding},
    definition::CuvsOperation,
};

/// PQ codebook training scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorIvfPqCodebookKind {
    PerSubspace,
    PerCluster,
}

/// Query-local IVF-PQ build and search parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorIvfPqParams {
    pub dataset: VectorRelationBinding,
    pub queries: VectorRelationBinding,
    pub k: usize,
    pub metric: VectorDistanceMetric,
    pub n_lists: u32,
    pub n_probes: u32,
    pub kmeans_n_iters: u32,
    pub kmeans_trainset_fraction: f64,
    pub pq_dim: u32,
    pub pq_bits: u32,
    pub codebook_kind: VectorIvfPqCodebookKind,
}

impl VectorIvfPqParams {
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
