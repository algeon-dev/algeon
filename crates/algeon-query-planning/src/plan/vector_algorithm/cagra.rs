use super::{
    common::{VectorDistanceMetric, VectorRelationBinding},
    definition::CuvsOperation,
};

/// CAGRA graph construction algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorCagraBuildAlgo {
    IvfPq,
    NnDescent,
}

/// CAGRA search kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorCagraSearchAlgo {
    Auto,
    SingleCta,
    MultiCta,
    MultiKernel,
}

/// Query-local CAGRA graph and search parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorCagraParams {
    pub dataset: VectorRelationBinding,
    pub queries: VectorRelationBinding,
    pub k: usize,
    pub metric: VectorDistanceMetric,
    pub graph_degree: u32,
    pub intermediate_graph_degree: u32,
    pub build_algo: VectorCagraBuildAlgo,
    pub itopk_size: u32,
    pub search_width: u32,
    pub max_iterations: u32,
    pub search_algo: VectorCagraSearchAlgo,
}

impl VectorCagraParams {
    pub fn new(
        dataset: VectorRelationBinding,
        queries: VectorRelationBinding,
        k: usize,
        metric: VectorDistanceMetric,
    ) -> Self {
        let descriptor = CuvsOperation::Cagra.descriptor();
        let integer = |name| {
            descriptor.option(name).and_then(|option| option.kind.default_u64()).expect("CAGRA integer default") as u32
        };
        Self {
            dataset,
            queries,
            k,
            metric,
            graph_degree: integer("graph_degree"),
            intermediate_graph_degree: integer("intermediate_graph_degree"),
            build_algo: VectorCagraBuildAlgo::IvfPq,
            itopk_size: integer("itopk_size"),
            search_width: integer("search_width"),
            max_iterations: integer("max_iterations"),
            search_algo: VectorCagraSearchAlgo::Auto,
        }
    }
}
