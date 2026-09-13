use arrow::datatypes::DataType;
use query_engine::plan::GraphAlgorithm as NativeGraphAlgorithm;

use super::types::CugraphGraphOptions;

pub const CUGRAPH_PAGERANK_FUNCTION: &str = "cugraph_pagerank";
pub const CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION: &str = "cugraph_personalized_pagerank";
pub const CUGRAPH_BFS_FUNCTION: &str = "cugraph_bfs";
pub const CUGRAPH_LOUVAIN_FUNCTION: &str = "cugraph_louvain";
pub const CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION: &str = "cugraph_eigenvector_centrality";
pub const CUGRAPH_KATZ_CENTRALITY_FUNCTION: &str = "cugraph_katz_centrality";
pub const CUGRAPH_HITS_FUNCTION: &str = "cugraph_hits";
pub const CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION: &str = "cugraph_betweenness_centrality";
pub const CUGRAPH_EDGE_BETWEENNESS_CENTRALITY_FUNCTION: &str = "cugraph_edge_betweenness_centrality";
pub const CUGRAPH_SSSP_FUNCTION: &str = "cugraph_sssp";
pub const CUGRAPH_DEGREES_ALL_FUNCTION: &str = "cugraph_degrees_all";
pub const CUGRAPH_IN_DEGREES_ALL_FUNCTION: &str = "cugraph_in_degrees_all";
pub const CUGRAPH_OUT_DEGREES_ALL_FUNCTION: &str = "cugraph_out_degrees_all";
pub const CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION: &str = "cugraph_weakly_connected_components";
pub const CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION: &str = "cugraph_strongly_connected_components";
pub const CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION: &str = "cugraph_triangle_count_all";
pub const CUGRAPH_CORE_NUMBER_FUNCTION: &str = "cugraph_core_number";
pub const CUGRAPH_K_CORE_FUNCTION: &str = "cugraph_k_core";
pub const CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION: &str = "cugraph_minimum_spanning_tree";
pub const CUGRAPH_LEIDEN_FUNCTION: &str = "cugraph_leiden";
pub const CUGRAPH_ECG_FUNCTION: &str = "cugraph_ecg";
pub const CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION: &str = "cugraph_spectral_modularity_maximization";
pub const CUGRAPH_FORCE_ATLAS2_FUNCTION: &str = "cugraph_force_atlas2";
pub const CUGRAPH_JACCARD_FUNCTION: &str = "cugraph_jaccard";
pub const CUGRAPH_SORENSEN_FUNCTION: &str = "cugraph_sorensen";
pub const CUGRAPH_OVERLAP_FUNCTION: &str = "cugraph_overlap";
pub const CUGRAPH_COSINE_FUNCTION: &str = "cugraph_cosine";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CugraphAlgorithm {
    PageRank,
    PersonalizedPageRank,
    Bfs,
    Louvain,
    EigenvectorCentrality,
    KatzCentrality,
    Hits,
    BetweennessCentrality,
    EdgeBetweennessCentrality,
    Sssp,
    DegreesAll,
    InDegreesAll,
    OutDegreesAll,
    WeaklyConnectedComponents,
    StronglyConnectedComponents,
    TriangleCountAll,
    CoreNumber,
    KCore,
    MinimumSpanningTree,
    Leiden,
    Ecg,
    SpectralModularityMaximization,
    ForceAtlas2,
    Jaccard,
    Sorensen,
    Overlap,
    Cosine,
}

impl CugraphAlgorithm {
    /// All algorithms exposed by the SQL surface, in registry order.
    pub fn all() -> impl ExactSizeIterator<Item = Self> + Clone {
        CUGRAPH_OPERATION_REGISTRY.iter().map(|entry| entry.algorithm)
    }

    pub fn function_name(self) -> &'static str {
        self.definition().sql_function_name
    }

    pub fn code(self) -> &'static str {
        self.native_algorithm().as_str()
    }

    /// Look up the SQL-side registry definition for this algorithm.
    pub fn definition(self) -> &'static CugraphOperationDefinition {
        CUGRAPH_OPERATION_REGISTRY
            .iter()
            .find(|entry| entry.algorithm == self)
            .expect("cuGraph SQL operation registry is missing an algorithm variant")
    }

    pub(crate) fn native_algorithm(self) -> NativeGraphAlgorithm {
        self.definition().native_algorithm
    }

    pub(crate) const fn is_similarity(self) -> bool {
        matches!(self, Self::Jaccard | Self::Sorensen | Self::Overlap | Self::Cosine)
    }

    pub(crate) fn effective_default_graph_options(self, defaults: &CugraphGraphOptions) -> CugraphGraphOptions {
        CugraphGraphOptions::from_native(
            &self.native_algorithm().definition().default_graph_options(&defaults.to_native()),
        )
    }
}

/// Registry-level metadata for a SQL-visible cuGraph operation.
///
/// One entry per [`CugraphAlgorithm`] variant. Mirrors
/// [`query_engine::plan::GRAPH_OPERATION_REGISTRY`]: each SQL UDTF resolves to
/// the same native algorithm advertised by the engine, and the SQL function
/// name follows the `cugraph_<native_name>` convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CugraphOperationDefinition {
    pub algorithm: CugraphAlgorithm,
    native_algorithm: NativeGraphAlgorithm,
    pub sql_function_name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CugraphLogicalNativeIdType {
    Int64,
}

impl CugraphLogicalNativeIdType {
    pub(crate) const fn data_type(self) -> DataType {
        match self {
            Self::Int64 => DataType::Int64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CugraphLogicalOutputMultiplicity {
    OnePerVertex,
    EdgeEndpointOccurrences,
    SideInputOccurrences,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CugraphLogicalOutputColumn {
    pub(crate) name: &'static str,
    pub(crate) nullable: bool,
    pub(crate) multiplicity: CugraphLogicalOutputMultiplicity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CugraphLogicalStringContract {
    pub(crate) native_id_type: CugraphLogicalNativeIdType,
    pub(crate) output_columns: &'static [CugraphLogicalOutputColumn],
}

const ONE_VERTEX_LOGICAL_OUTPUT: &[CugraphLogicalOutputColumn] = &[CugraphLogicalOutputColumn {
    name: "vertex",
    nullable: false,
    multiplicity: CugraphLogicalOutputMultiplicity::OnePerVertex,
}];

const BFS_LOGICAL_OUTPUTS: &[CugraphLogicalOutputColumn] = &[
    CugraphLogicalOutputColumn {
        name: "vertex",
        nullable: false,
        multiplicity: CugraphLogicalOutputMultiplicity::OnePerVertex,
    },
    CugraphLogicalOutputColumn {
        name: "predecessor",
        nullable: true,
        multiplicity: CugraphLogicalOutputMultiplicity::EdgeEndpointOccurrences,
    },
];

const EDGE_LOGICAL_OUTPUTS: &[CugraphLogicalOutputColumn] = &[
    CugraphLogicalOutputColumn {
        name: "src",
        nullable: false,
        multiplicity: CugraphLogicalOutputMultiplicity::EdgeEndpointOccurrences,
    },
    CugraphLogicalOutputColumn {
        name: "dst",
        nullable: false,
        multiplicity: CugraphLogicalOutputMultiplicity::EdgeEndpointOccurrences,
    },
];

const EDGE_BETWEENNESS_LOGICAL_OUTPUTS: &[CugraphLogicalOutputColumn] = &[
    CugraphLogicalOutputColumn {
        name: "source",
        nullable: false,
        multiplicity: CugraphLogicalOutputMultiplicity::EdgeEndpointOccurrences,
    },
    CugraphLogicalOutputColumn {
        name: "destination",
        nullable: false,
        multiplicity: CugraphLogicalOutputMultiplicity::EdgeEndpointOccurrences,
    },
];

const PAIR_LOGICAL_OUTPUTS: &[CugraphLogicalOutputColumn] = &[
    CugraphLogicalOutputColumn {
        name: "first",
        nullable: false,
        multiplicity: CugraphLogicalOutputMultiplicity::SideInputOccurrences,
    },
    CugraphLogicalOutputColumn {
        name: "second",
        nullable: false,
        multiplicity: CugraphLogicalOutputMultiplicity::SideInputOccurrences,
    },
];

impl CugraphOperationDefinition {
    /// Whether this SQL function supports logical string vertex IDs through
    /// the shared per-call codebook.
    pub const fn supports_logical_string_vertices(&self) -> bool {
        self.logical_string_contract().is_some()
    }

    pub(crate) const fn vertex_output_columns(&self) -> &'static [CugraphLogicalOutputColumn] {
        match self.algorithm {
            CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp => BFS_LOGICAL_OUTPUTS,
            CugraphAlgorithm::PageRank
            | CugraphAlgorithm::PersonalizedPageRank
            | CugraphAlgorithm::Louvain
            | CugraphAlgorithm::EigenvectorCentrality
            | CugraphAlgorithm::KatzCentrality
            | CugraphAlgorithm::Hits
            | CugraphAlgorithm::BetweennessCentrality
            | CugraphAlgorithm::DegreesAll
            | CugraphAlgorithm::InDegreesAll
            | CugraphAlgorithm::OutDegreesAll
            | CugraphAlgorithm::WeaklyConnectedComponents
            | CugraphAlgorithm::StronglyConnectedComponents
            | CugraphAlgorithm::TriangleCountAll
            | CugraphAlgorithm::CoreNumber
            | CugraphAlgorithm::Leiden
            | CugraphAlgorithm::Ecg
            | CugraphAlgorithm::SpectralModularityMaximization
            | CugraphAlgorithm::ForceAtlas2 => ONE_VERTEX_LOGICAL_OUTPUT,
            CugraphAlgorithm::KCore | CugraphAlgorithm::MinimumSpanningTree => EDGE_LOGICAL_OUTPUTS,
            CugraphAlgorithm::EdgeBetweennessCentrality => EDGE_BETWEENNESS_LOGICAL_OUTPUTS,
            CugraphAlgorithm::Jaccard
            | CugraphAlgorithm::Sorensen
            | CugraphAlgorithm::Overlap
            | CugraphAlgorithm::Cosine => PAIR_LOGICAL_OUTPUTS,
        }
    }

    pub(crate) const fn logical_string_contract(&self) -> Option<CugraphLogicalStringContract> {
        if matches!(
            self.algorithm,
            CugraphAlgorithm::MinimumSpanningTree | CugraphAlgorithm::SpectralModularityMaximization
        ) {
            return None;
        }
        let output_columns = self.vertex_output_columns();
        Some(CugraphLogicalStringContract { native_id_type: CugraphLogicalNativeIdType::Int64, output_columns })
    }
}

/// Source-of-truth list of SQL-visible cuGraph algorithms.
///
/// Order is preserved by [`CugraphAlgorithm::all`] and is the order in which
/// cuGraph table functions are registered.
pub const CUGRAPH_OPERATION_REGISTRY: &[CugraphOperationDefinition] = &[
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::PageRank,
        native_algorithm: NativeGraphAlgorithm::PageRank,
        sql_function_name: CUGRAPH_PAGERANK_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Bfs,
        native_algorithm: NativeGraphAlgorithm::Bfs,
        sql_function_name: CUGRAPH_BFS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Louvain,
        native_algorithm: NativeGraphAlgorithm::Louvain,
        sql_function_name: CUGRAPH_LOUVAIN_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::EigenvectorCentrality,
        native_algorithm: NativeGraphAlgorithm::EigenvectorCentrality,
        sql_function_name: CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::KatzCentrality,
        native_algorithm: NativeGraphAlgorithm::KatzCentrality,
        sql_function_name: CUGRAPH_KATZ_CENTRALITY_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Hits,
        native_algorithm: NativeGraphAlgorithm::Hits,
        sql_function_name: CUGRAPH_HITS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::BetweennessCentrality,
        native_algorithm: NativeGraphAlgorithm::BetweennessCentrality,
        sql_function_name: CUGRAPH_BETWEENNESS_CENTRALITY_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::EdgeBetweennessCentrality,
        native_algorithm: NativeGraphAlgorithm::EdgeBetweennessCentrality,
        sql_function_name: CUGRAPH_EDGE_BETWEENNESS_CENTRALITY_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Sssp,
        native_algorithm: NativeGraphAlgorithm::Sssp,
        sql_function_name: CUGRAPH_SSSP_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::DegreesAll,
        native_algorithm: NativeGraphAlgorithm::DegreesAll,
        sql_function_name: CUGRAPH_DEGREES_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::InDegreesAll,
        native_algorithm: NativeGraphAlgorithm::InDegreesAll,
        sql_function_name: CUGRAPH_IN_DEGREES_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::OutDegreesAll,
        native_algorithm: NativeGraphAlgorithm::OutDegreesAll,
        sql_function_name: CUGRAPH_OUT_DEGREES_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::WeaklyConnectedComponents,
        native_algorithm: NativeGraphAlgorithm::WeaklyConnectedComponents,
        sql_function_name: CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::StronglyConnectedComponents,
        native_algorithm: NativeGraphAlgorithm::StronglyConnectedComponents,
        sql_function_name: CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::TriangleCountAll,
        native_algorithm: NativeGraphAlgorithm::TriangleCountAll,
        sql_function_name: CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::CoreNumber,
        native_algorithm: NativeGraphAlgorithm::CoreNumber,
        sql_function_name: CUGRAPH_CORE_NUMBER_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::KCore,
        native_algorithm: NativeGraphAlgorithm::KCore,
        sql_function_name: CUGRAPH_K_CORE_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::MinimumSpanningTree,
        native_algorithm: NativeGraphAlgorithm::MinimumSpanningTree,
        sql_function_name: CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::PersonalizedPageRank,
        native_algorithm: NativeGraphAlgorithm::PersonalizedPageRank,
        sql_function_name: CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Leiden,
        native_algorithm: NativeGraphAlgorithm::Leiden,
        sql_function_name: CUGRAPH_LEIDEN_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Ecg,
        native_algorithm: NativeGraphAlgorithm::Ecg,
        sql_function_name: CUGRAPH_ECG_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::SpectralModularityMaximization,
        native_algorithm: NativeGraphAlgorithm::SpectralModularityMaximization,
        sql_function_name: CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::ForceAtlas2,
        native_algorithm: NativeGraphAlgorithm::ForceAtlas2,
        sql_function_name: CUGRAPH_FORCE_ATLAS2_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Jaccard,
        native_algorithm: NativeGraphAlgorithm::Jaccard,
        sql_function_name: CUGRAPH_JACCARD_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Sorensen,
        native_algorithm: NativeGraphAlgorithm::Sorensen,
        sql_function_name: CUGRAPH_SORENSEN_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Overlap,
        native_algorithm: NativeGraphAlgorithm::Overlap,
        sql_function_name: CUGRAPH_OVERLAP_FUNCTION,
    },
    CugraphOperationDefinition {
        algorithm: CugraphAlgorithm::Cosine,
        native_algorithm: NativeGraphAlgorithm::Cosine,
        sql_function_name: CUGRAPH_COSINE_FUNCTION,
    },
];
