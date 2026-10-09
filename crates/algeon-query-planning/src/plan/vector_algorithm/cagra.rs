use super::{
    common::{VectorDistanceMetric, VectorRelationBinding},
    definition::CuvsOperation,
};

/// CAGRA graph construction algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorCagraBuildAlgo {
    /// Constructs the graph using an IVF-PQ index.
    IvfPq,
    /// Constructs the graph using NN-descent; does not support cosine.
    NnDescent,
}

/// CAGRA search kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorCagraSearchAlgo {
    /// Allows cuVS to select the search kernel.
    Auto,
    /// Uses one cooperative thread array per query.
    SingleCta,
    /// Uses multiple cooperative thread arrays per query.
    MultiCta,
    /// Uses the multi-kernel search implementation.
    MultiKernel,
}

/// Query-local CAGRA graph and search parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorCagraParams {
    /// Dense rows indexed by the graph.
    pub dataset: VectorRelationBinding,
    /// Dense query rows matched against the dataset.
    pub queries: VectorRelationBinding,
    /// Number of neighbors returned per query row.
    pub k: usize,
    /// Distance or similarity used to rank neighbors.
    pub metric: VectorDistanceMetric,
    /// Output graph edges per vertex; must be below dataset row count.
    pub graph_degree: u32,
    /// Build graph edges per vertex; must exceed `graph_degree` and be below dataset rows.
    pub intermediate_graph_degree: u32,
    /// Algorithm used to construct the query-local graph.
    pub build_algo: VectorCagraBuildAlgo,
    /// Intermediate result capacity; must be at least `k`.
    pub itopk_size: u32,
    /// Starting graph nodes explored per search iteration.
    pub search_width: u32,
    /// Maximum search iteration count; zero selects it automatically.
    pub max_iterations: u32,
    /// Native search kernel selection.
    pub search_algo: VectorCagraSearchAlgo,
}

impl VectorCagraParams {
    /// Uses registry defaults for graph construction and search tuning.
    ///
    /// The supplied bindings, neighbor count, and metric are retained without validation.
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
