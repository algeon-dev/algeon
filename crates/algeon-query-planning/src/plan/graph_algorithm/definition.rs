//! Graph algorithm definitions, registry metadata, and construction contracts.
use std::{fmt, sync::Arc};

use arrow::datatypes::DataType;

use super::{
    Field, GraphAlgorithmParams, GraphBetweennessCentralityParams, GraphBfsParams, GraphCoreNumberParams,
    GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams, GraphEigenvectorCentralityParams,
    GraphForceAtlas2Params, GraphHitsParams, GraphInDegreesAllParams, GraphKCoreParams, GraphKatzCentralityParams,
    GraphLayoutRequirement, GraphLeidenParams, GraphLouvainParams, GraphMstParams, GraphOutDegreesAllParams,
    GraphPageRankParams, GraphPersonalizedPageRankParams, GraphSccParams, GraphSimilarityParams,
    GraphSpectralClusteringParams, GraphSsspParams, GraphSsspSource, GraphTriangleCountAllParams, GraphWccParams,
    Schema, SchemaRef, stable_graph_result_schema,
};

/// Native graph algorithm supported by the cuGraph operator boundary.
///
/// Every variant must have a matching entry in [`GRAPH_OPERATION_REGISTRY`].
/// Ref: <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/>
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphAlgorithm {
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/centrality/#_CPPv416cugraph_pagerankPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tdd6size_t6bool_tPP27cugraph_centrality_result_tPP15cugraph_error_t>
    PageRank,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/centrality/#_CPPv429cugraph_personalized_pagerankPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_tdd6size_t6bool_tPP27cugraph_centrality_result_tPP15cugraph_error_t>
    PersonalizedPageRank,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/traversal/#_CPPv411cugraph_bfsPK25cugraph_resource_handle_tP15cugraph_graph_tP39cugraph_type_erased_device_array_view_t6bool_t6size_t6bool_t6bool_tPP22cugraph_paths_result_tPP15cugraph_error_t>
    Bfs,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/community/#_CPPv415cugraph_louvainPK25cugraph_resource_handle_tP15cugraph_graph_t6size_tdd6bool_tPP40cugraph_hierarchical_clustering_result_tPP15cugraph_error_t>
    Louvain,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/centrality/#_CPPv430cugraph_eigenvector_centralityPK25cugraph_resource_handle_tP15cugraph_graph_td6size_t6bool_tPP27cugraph_centrality_result_tPP15cugraph_error_t>
    EigenvectorCentrality,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/centrality/#_CPPv423cugraph_katz_centralityPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_tddd6size_t6bool_tPP27cugraph_centrality_result_tPP15cugraph_error_t>
    KatzCentrality,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/centrality/#_CPPv412cugraph_hitsPK25cugraph_resource_handle_tP15cugraph_graph_td6size_tPK39cugraph_type_erased_device_array_view_tPK39cugraph_type_erased_device_array_view_t6bool_t6bool_tPP21cugraph_hits_result_tPP15cugraph_error_t>
    Hits,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/centrality/#_CPPv430cugraph_betweenness_centralityPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_t6bool_t6bool_t6bool_tPP27cugraph_centrality_result_tPP15cugraph_error_t>
    BetweennessCentrality,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/centrality/#_CPPv435cugraph_edge_betweenness_centralityPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_t6bool_t6bool_tPP32cugraph_edge_centrality_result_tPP15cugraph_error_t>
    EdgeBetweennessCentrality,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/traversal/#_CPPv412cugraph_ssspPK25cugraph_resource_handle_tP15cugraph_graph_t6size_td6bool_t6bool_tPP22cugraph_paths_result_tPP15cugraph_error_t>
    Sssp,
    /// In-degree and out-degree for every vertex.
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/graph/#_CPPv415cugraph_degreesPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_t6bool_tPP24cugraph_degrees_result_tPP15cugraph_error_t>
    DegreesAll,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/graph/#_CPPv418cugraph_in_degreesPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_t6bool_tPP24cugraph_degrees_result_tPP15cugraph_error_t>
    InDegreesAll,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/graph/#_CPPv419cugraph_out_degreesPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_t6bool_tPP24cugraph_degrees_result_tPP15cugraph_error_t>
    OutDegreesAll,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/labeling/#_CPPv435cugraph_weakly_connected_componentsPK25cugraph_resource_handle_tP15cugraph_graph_t6bool_tPP25cugraph_labeling_result_tPP15cugraph_error_t>
    WeaklyConnectedComponents,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/labeling/#_CPPv437cugraph_strongly_connected_componentsPK25cugraph_resource_handle_tP15cugraph_graph_t6bool_tPP25cugraph_labeling_result_tPP15cugraph_error_t>
    StronglyConnectedComponents,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/community/#_CPPv422cugraph_triangle_countPK25cugraph_resource_handle_tP15cugraph_graph_tPK39cugraph_type_erased_device_array_view_t6bool_tPP31cugraph_triangle_count_result_tPP15cugraph_error_t>
    TriangleCountAll,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/core/#_CPPv419cugraph_core_numberPK25cugraph_resource_handle_tP15cugraph_graph_t28cugraph_k_core_degree_type_t6bool_tPP21cugraph_core_result_tPP15cugraph_error_t>
    CoreNumber,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/core/#_CPPv414cugraph_k_corePK25cugraph_resource_handle_tP15cugraph_graph_t6size_t28cugraph_k_core_degree_type_tPK21cugraph_core_result_t6bool_tPP23cugraph_k_core_result_tPP15cugraph_error_t>
    KCore,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/tree/#_CPPv429cugraph_minimum_spanning_treePK25cugraph_resource_handle_tP15cugraph_graph_t6bool_tPP33cugraph_induced_subgraph_result_tPP15cugraph_error_t>
    MinimumSpanningTree,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/community/#_CPPv414cugraph_leidenPK25cugraph_resource_handle_tP19cugraph_rng_state_tP15cugraph_graph_t6size_tdd6bool_tPP40cugraph_hierarchical_clustering_result_tPP15cugraph_error_t>
    Leiden,
    /// Ensemble clustering for graphs.
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/community/#_CPPv411cugraph_ecgPK25cugraph_resource_handle_tP19cugraph_rng_state_tP15cugraph_graph_td6size_t6size_tdd6bool_tPP40cugraph_hierarchical_clustering_result_tPP15cugraph_error_t>
    Ecg,
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/community/#_CPPv440cugraph_spectral_modularity_maximizationPK25cugraph_resource_handle_tP19cugraph_rng_state_tP15cugraph_graph_t6size_t6size_tdidi6bool_tPP27cugraph_clustering_result_tPP15cugraph_error_t>
    SpectralModularityMaximization,
    /// Force-directed graph layout.
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/layout/#_CPPv420cugraph_force_atlas2PK25cugraph_resource_handle_tP19cugraph_rng_state_tP15cugraph_graph_tiP39cugraph_type_erased_device_array_view_tP39cugraph_type_erased_device_array_view_tP39cugraph_type_erased_device_array_view_t6bool_t6bool_t6bool_tP39cugraph_type_erased_device_array_view_tP39cugraph_type_erased_device_array_view_tddd6bool_tdd6bool_tdP39cugraph_type_erased_device_array_view_tP39cugraph_type_erased_device_array_view_tP39cugraph_type_erased_device_array_view_tP39cugraph_type_erased_device_array_view_t6bool_t6bool_tPP23cugraph_layout_result_tPP15cugraph_error_t>
    ForceAtlas2,
    /// Explicit-pair Jaccard similarity coefficient.
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/similarity/#_CPPv428cugraph_jaccard_coefficientsPK25cugraph_resource_handle_tP15cugraph_graph_tPK22cugraph_vertex_pairs_t6bool_t6bool_tPP27cugraph_similarity_result_tPP15cugraph_error_t>
    Jaccard,
    /// Explicit-pair Sorensen similarity coefficient.
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/similarity/#_CPPv429cugraph_sorensen_coefficientsPK25cugraph_resource_handle_tP15cugraph_graph_tPK22cugraph_vertex_pairs_t6bool_t6bool_tPP27cugraph_similarity_result_tPP15cugraph_error_t>
    Sorensen,
    /// Explicit-pair overlap similarity coefficient.
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/similarity/#_CPPv428cugraph_overlap_coefficientsPK25cugraph_resource_handle_tP15cugraph_graph_tPK22cugraph_vertex_pairs_t6bool_t6bool_tPP27cugraph_similarity_result_tPP15cugraph_error_t>
    Overlap,
    /// Explicit-pair cosine similarity coefficient.
    /// <https://docs.nvidia.com/cugraph/latest/api_docs/cugraph_c/similarity/#_CPPv438cugraph_cosine_similarity_coefficientsPK25cugraph_resource_handle_tP15cugraph_graph_tPK22cugraph_vertex_pairs_t6bool_t6bool_tPP27cugraph_similarity_result_tPP15cugraph_error_t>
    Cosine,
}

impl GraphAlgorithm {
    /// Stable lowercase algorithm code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PageRank => "pagerank",
            Self::PersonalizedPageRank => "personalized_pagerank",
            Self::Bfs => "bfs",
            Self::Louvain => "louvain",
            Self::EigenvectorCentrality => "eigenvector_centrality",
            Self::KatzCentrality => "katz_centrality",
            Self::Hits => "hits",
            Self::BetweennessCentrality => "betweenness_centrality",
            Self::EdgeBetweennessCentrality => "edge_betweenness_centrality",
            Self::Sssp => "sssp",
            Self::DegreesAll => "degrees_all",
            Self::InDegreesAll => "in_degrees_all",
            Self::OutDegreesAll => "out_degrees_all",
            Self::WeaklyConnectedComponents => "weakly_connected_components",
            Self::StronglyConnectedComponents => "strongly_connected_components",
            Self::TriangleCountAll => "triangle_count_all",
            Self::CoreNumber => "core_number",
            Self::KCore => "k_core",
            Self::MinimumSpanningTree => "minimum_spanning_tree",
            Self::Leiden => "leiden",
            Self::Ecg => "ecg",
            Self::SpectralModularityMaximization => "spectral_modularity_maximization",
            Self::ForceAtlas2 => "force_atlas2",
            Self::Jaccard => "jaccard",
            Self::Sorensen => "sorensen",
            Self::Overlap => "overlap",
            Self::Cosine => "cosine",
        }
    }

    /// Whether this algorithm is one of the explicit-pair similarity metrics.
    pub const fn is_similarity(self) -> bool {
        matches!(self, Self::Jaccard | Self::Sorensen | Self::Overlap | Self::Cosine)
    }

    /// Look up the registry definition for this algorithm.
    ///
    /// Panics if the algorithm variant lacks a registry entry. The registry is
    /// the single source of truth for algorithm metadata; new
    /// [`GraphAlgorithm`] variants must add an entry to
    /// [`GRAPH_OPERATION_REGISTRY`].
    pub fn definition(self) -> &'static GraphOperationDefinition {
        GRAPH_OPERATION_REGISTRY
            .iter()
            .find(|def| def.algorithm == self)
            .expect("graph operation registry is missing an algorithm variant")
    }

    /// All algorithms exposed by the registry, in registry order.
    pub fn all() -> impl ExactSizeIterator<Item = Self> + Clone {
        GRAPH_OPERATION_REGISTRY.iter().map(|def| def.algorithm)
    }

    /// Whether this algorithm reads edge weights as a semantic input.
    ///
    /// When `false`, a bound `weight` column is irrelevant to the result and a
    /// null weight must not cause its edge to be dropped from the graph.
    pub fn consumes_weight(self) -> bool {
        self.definition().weight_policy().consumes_weight()
    }

    /// Whether this algorithm requires a source vertex argument.
    pub fn requires_source_vertex(self) -> bool {
        self.definition().source_vertex_requirement().requires_source_vertex()
    }

    /// Weight-column policy declared by the native graph descriptor.
    pub fn weight_policy(self) -> GraphWeightPolicy {
        self.definition().weight_policy()
    }

    /// Output-schema shape declared by the native graph descriptor.
    pub fn output_schema_kind(self) -> GraphOutputSchemaKind {
        self.definition().output_schema_kind()
    }

    /// Vertex-id dtype support declared by the native graph descriptor.
    pub fn vertex_id_type_support(self) -> GraphVertexIdTypeSupport {
        self.definition().vertex_id_type_support()
    }
}

impl fmt::Display for GraphAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Vertex-id dtype support for a native graph algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphVertexIdTypeSupport {
    /// The operation accepts Int32 and Int64 vertex id columns.
    Int32OrInt64,
    /// cuGraph C API operation whose edge type dispatch only supports
    /// Int32 vertex id columns.
    Int32OnlyLegacyEdgeType,
}

impl GraphVertexIdTypeSupport {
    /// Stable lowercase capability code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int32OrInt64 => "int32_or_int64",
            Self::Int32OnlyLegacyEdgeType => "int32_only_legacy_edge_type",
        }
    }

    /// SQL/diagnostic label for supported dtypes.
    pub const fn supported_dtypes_label(self) -> &'static str {
        match self {
            Self::Int32OrInt64 => "Int32|Int64",
            Self::Int32OnlyLegacyEdgeType => "Int32",
        }
    }

    /// Whether the Arrow dtype is supported for vertex id columns.
    pub fn accepts_arrow_data_type(self, data_type: &DataType) -> bool {
        match self {
            Self::Int32OrInt64 => matches!(data_type, DataType::Int32 | DataType::Int64),
            Self::Int32OnlyLegacyEdgeType => matches!(data_type, DataType::Int32),
        }
    }
}

impl fmt::Display for GraphVertexIdTypeSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Source-vertex contract for a graph algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphSourceVertexRequirement {
    /// The algorithm does not accept a source vertex.
    None,
    /// The algorithm requires exactly one source vertex.
    Required,
}

impl GraphSourceVertexRequirement {
    /// Stable lowercase source-vertex requirement code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Required => "required",
        }
    }

    /// Whether the algorithm requires a source vertex.
    pub const fn requires_source_vertex(self) -> bool {
        matches!(self, Self::Required)
    }
}

/// Weight-column policy for a graph algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphWeightPolicy {
    /// A bound weight column is accepted as graph-construction metadata but is
    /// ignored by the algorithm's current semantic result.
    Ignored,
    /// A bound weight column is optional and changes algorithm results when
    /// present.
    Optional,
    /// A bound weight column is required for semantically valid execution.
    Required,
}

impl GraphWeightPolicy {
    /// Stable lowercase policy code used by metadata projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ignored => "ignored",
            Self::Optional => "optional",
            Self::Required => "required",
        }
    }

    /// Whether this policy means the native executor consumes weights as an
    /// algorithm input when a weight column is bound.
    pub const fn consumes_weight(self) -> bool {
        matches!(self, Self::Optional | Self::Required)
    }

    /// Whether a weight column is mandatory for this algorithm.
    pub const fn requires_weight(self) -> bool {
        matches!(self, Self::Required)
    }
}

/// Native output-schema descriptor for a graph algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphOutputSchemaKind {
    /// The SQL-visible schema is the native stable base schema.
    Fixed,
    /// The native descriptor owns the base schema, while an adapter layer may
    /// expose call-specific modes derived from it.
    DefaultMode,
}

/// Engine-owned behavior when a graph operation receives no usable edges.
///
/// The executor applies this after null edge rows have been removed and before
/// graph admission or cuGraph construction. This keeps empty-input semantics
/// independent of libcugraph's incidental behavior and distinguishes a valid
/// empty result from an operation that requires an actual graph or side input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmptyGraphInputPolicy {
    /// Action for an empty edge list after input sanitization.
    pub edge_input: EmptyGraphEdgeInputAction,
    /// Contract for any relation-backed algorithm inputs.
    pub side_inputs: EmptyGraphSideInputPolicy,
}

impl EmptyGraphInputPolicy {
    const RETURN_EMPTY: Self =
        Self { edge_input: EmptyGraphEdgeInputAction::ReturnEmpty, side_inputs: EmptyGraphSideInputPolicy::None };
    const VALIDATE_SELECTORS_THEN_RETURN_EMPTY: Self = Self {
        edge_input: EmptyGraphEdgeInputAction::ValidateSelectorsThenReturnEmpty,
        side_inputs: EmptyGraphSideInputPolicy::None,
    };
    const REQUIRE_GRAPH: Self =
        Self { edge_input: EmptyGraphEdgeInputAction::Reject, side_inputs: EmptyGraphSideInputPolicy::None };
    const REQUIRE_GRAPH_AND_PERSONALIZATION: Self = Self {
        edge_input: EmptyGraphEdgeInputAction::Reject,
        side_inputs: EmptyGraphSideInputPolicy::RequirePersonalization,
    };
    const SIMILARITY: Self = Self {
        edge_input: EmptyGraphEdgeInputAction::SimilarityVertexPairs,
        side_inputs: EmptyGraphSideInputPolicy::RequireVertexPairs,
    };
    const BFS: Self = Self {
        edge_input: EmptyGraphEdgeInputAction::BfsEndpointPolicy,
        side_inputs: EmptyGraphSideInputPolicy::BfsPredicates,
    };
}

/// Action for a sanitized empty graph edge list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmptyGraphEdgeInputAction {
    /// The operation has a schema-correct zero-row result.
    ReturnEmpty,
    /// Validate explicit vertex selectors first, then return the zero-row
    /// result only when no selector requires a vertex.
    ValidateSelectorsThenReturnEmpty,
    /// The operation cannot run without at least one graph edge/vertex.
    Reject,
    /// BFS delegates empty-edge behavior to its explicit endpoint policy.
    BfsEndpointPolicy,
    /// Explicit-pair similarity returns a stable empty result for an empty
    /// pair relation and rejects non-empty pairs against an empty graph.
    SimilarityVertexPairs,
}

impl EmptyGraphEdgeInputAction {
    /// Stable policy label for diagnostics and contract tests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReturnEmpty => "return_empty",
            Self::ValidateSelectorsThenReturnEmpty => "validate_selectors_then_return_empty",
            Self::Reject => "reject",
            Self::BfsEndpointPolicy => "bfs_endpoint_policy",
            Self::SimilarityVertexPairs => "similarity_vertex_pairs",
        }
    }
}

/// Relation-backed input contract for an empty graph operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmptyGraphSideInputPolicy {
    /// The operation accepts no relation-backed side input.
    None,
    /// Personalized PageRank requires a non-empty personalization relation.
    RequirePersonalization,
    /// BFS has role-specific source and predicate semantics owned by its
    /// traversal executor.
    BfsPredicates,
    /// Explicit-pair similarity owns pair-cardinality and endpoint behavior.
    RequireVertexPairs,
}

impl EmptyGraphSideInputPolicy {
    /// Stable policy label for diagnostics and contract tests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::RequirePersonalization => "require_personalization",
            Self::BfsPredicates => "bfs_predicates",
            Self::RequireVertexPairs => "require_vertex_pairs",
        }
    }
}

impl GraphOutputSchemaKind {
    /// Stable lowercase schema-kind code used by metadata projections.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::DefaultMode => "default_mode",
        }
    }

    /// Whether this algorithm has adapter-visible mode schemas.
    pub const fn has_mode_schemas(self) -> bool {
        matches!(self, Self::DefaultMode)
    }
}

/// Registry-level metadata for a graph operation.
///
/// One entry per [`GraphAlgorithm`] variant. The registry is the place new
/// algorithms must declare their layout requirement, stable SQL name, and any
/// future metadata fields (parameter schema, weighted-edge support, etc.) so
/// the planner, executor, and SQL surface stay in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphOperationDefinition {
    /// Algorithm variant described by this registry row.
    pub algorithm: GraphAlgorithm,
    /// Stable identifier used in plan display, runtime metrics, and SQL
    /// function naming (`cugraph_<name>`).
    pub name: &'static str,
    /// cuGraph layout requirement for this operation.
    pub layout_requirement: GraphLayoutRequirement,
    /// Engine-owned empty-edge and side-input behavior.
    pub empty_input_policy: EmptyGraphInputPolicy,
}

impl GraphOperationDefinition {
    /// Result columns whose dtype is the graph's vertex-id domain.
    pub const fn vertex_output_columns(&self) -> &'static [&'static str] {
        match self.algorithm {
            GraphAlgorithm::Bfs => &["vertex", "predecessor", "source", "target"],
            GraphAlgorithm::Sssp => &["vertex", "predecessor"],
            GraphAlgorithm::EdgeBetweennessCentrality => &["source", "destination"],
            GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => &["src", "dst"],
            GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
                &["first", "second"]
            },
            GraphAlgorithm::PageRank
            | GraphAlgorithm::PersonalizedPageRank
            | GraphAlgorithm::Louvain
            | GraphAlgorithm::EigenvectorCentrality
            | GraphAlgorithm::KatzCentrality
            | GraphAlgorithm::Hits
            | GraphAlgorithm::BetweennessCentrality
            | GraphAlgorithm::DegreesAll
            | GraphAlgorithm::InDegreesAll
            | GraphAlgorithm::OutDegreesAll
            | GraphAlgorithm::WeaklyConnectedComponents
            | GraphAlgorithm::StronglyConnectedComponents
            | GraphAlgorithm::TriangleCountAll
            | GraphAlgorithm::CoreNumber
            | GraphAlgorithm::Leiden
            | GraphAlgorithm::Ecg
            | GraphAlgorithm::SpectralModularityMaximization
            | GraphAlgorithm::ForceAtlas2 => &["vertex"],
        }
    }

    /// Source-vertex requirement for this algorithm.
    pub const fn source_vertex_requirement(&self) -> GraphSourceVertexRequirement {
        match self.algorithm {
            GraphAlgorithm::Bfs | GraphAlgorithm::Sssp => GraphSourceVertexRequirement::Required,
            _ => GraphSourceVertexRequirement::None,
        }
    }

    /// Weight-column policy for this algorithm.
    pub const fn weight_policy(&self) -> GraphWeightPolicy {
        match self.algorithm {
            GraphAlgorithm::MinimumSpanningTree => GraphWeightPolicy::Required,
            GraphAlgorithm::PageRank
            | GraphAlgorithm::PersonalizedPageRank
            | GraphAlgorithm::Sssp
            | GraphAlgorithm::ForceAtlas2
            | GraphAlgorithm::Jaccard
            | GraphAlgorithm::Sorensen
            | GraphAlgorithm::Overlap
            | GraphAlgorithm::Cosine => GraphWeightPolicy::Optional,
            _ => GraphWeightPolicy::Ignored,
        }
    }

    /// Base output-schema descriptor for this algorithm.
    pub const fn output_schema_kind(&self) -> GraphOutputSchemaKind {
        match self.algorithm {
            GraphAlgorithm::Bfs => GraphOutputSchemaKind::DefaultMode,
            _ => GraphOutputSchemaKind::Fixed,
        }
    }

    /// Whether the stable result contains at most one row for each `vertex`.
    pub const fn output_vertex_is_unique(&self) -> bool {
        matches!(
            self.algorithm,
            GraphAlgorithm::PageRank
                | GraphAlgorithm::PersonalizedPageRank
                | GraphAlgorithm::Bfs
                | GraphAlgorithm::Louvain
                | GraphAlgorithm::EigenvectorCentrality
                | GraphAlgorithm::KatzCentrality
                | GraphAlgorithm::Hits
                | GraphAlgorithm::BetweennessCentrality
                | GraphAlgorithm::Sssp
                | GraphAlgorithm::DegreesAll
                | GraphAlgorithm::InDegreesAll
                | GraphAlgorithm::OutDegreesAll
                | GraphAlgorithm::WeaklyConnectedComponents
                | GraphAlgorithm::StronglyConnectedComponents
                | GraphAlgorithm::TriangleCountAll
                | GraphAlgorithm::CoreNumber
                | GraphAlgorithm::Leiden
                | GraphAlgorithm::Ecg
                | GraphAlgorithm::SpectralModularityMaximization
                | GraphAlgorithm::ForceAtlas2
        )
    }

    /// Required graph directedness, when the cuGraph operation only accepts
    /// one graph property.
    pub const fn required_graph_directedness(&self) -> Option<bool> {
        match self.algorithm {
            GraphAlgorithm::WeaklyConnectedComponents
            | GraphAlgorithm::MinimumSpanningTree
            | GraphAlgorithm::SpectralModularityMaximization
            | GraphAlgorithm::Jaccard
            | GraphAlgorithm::Sorensen
            | GraphAlgorithm::Overlap
            | GraphAlgorithm::Cosine => Some(false),
            GraphAlgorithm::StronglyConnectedComponents => Some(true),
            _ => None,
        }
    }

    /// Vertex-id dtype support for this algorithm.
    pub const fn vertex_id_type_support(&self) -> GraphVertexIdTypeSupport {
        match self.algorithm {
            GraphAlgorithm::MinimumSpanningTree | GraphAlgorithm::SpectralModularityMaximization => {
                GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType
            },
            _ => GraphVertexIdTypeSupport::Int32OrInt64,
        }
    }

    /// Engine-owned contract for empty graph inputs.
    pub const fn empty_input_policy(&self) -> EmptyGraphInputPolicy {
        self.empty_input_policy
    }

    /// Stable SQL-facing result schema for this algorithm.
    pub fn output_schema(&self) -> SchemaRef {
        stable_graph_result_schema(self.algorithm)
    }

    /// Native base result schema for this algorithm.
    pub fn base_result_schema(&self) -> SchemaRef {
        stable_graph_result_schema(self.algorithm)
    }

    /// Shared graph-construction defaults after algorithm-level overrides.
    pub fn default_graph_options(&self, defaults: &GraphConstructionOptions) -> GraphConstructionOptions {
        let mut options = defaults.clone();
        if matches!(
            self.algorithm,
            GraphAlgorithm::Louvain
                | GraphAlgorithm::WeaklyConnectedComponents
                | GraphAlgorithm::TriangleCountAll
                | GraphAlgorithm::CoreNumber
                | GraphAlgorithm::KCore
                | GraphAlgorithm::MinimumSpanningTree
                | GraphAlgorithm::Leiden
                | GraphAlgorithm::Ecg
                | GraphAlgorithm::SpectralModularityMaximization
                | GraphAlgorithm::ForceAtlas2
                | GraphAlgorithm::Jaccard
                | GraphAlgorithm::Sorensen
                | GraphAlgorithm::Overlap
                | GraphAlgorithm::Cosine
        ) {
            options.directed = false;
        }
        options
    }

    /// Default native parameter payload for metadata and planning projections.
    ///
    /// Source-vertex algorithms use `0` as a placeholder because the real
    /// source vertex remains a required SQL/parser input, not a defaulted
    /// option.
    pub fn default_params(&self) -> GraphAlgorithmParams {
        match self.algorithm {
            GraphAlgorithm::PageRank => GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
            GraphAlgorithm::PersonalizedPageRank => {
                GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams::default())
            },
            GraphAlgorithm::Bfs => GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
            GraphAlgorithm::Louvain => GraphAlgorithmParams::Louvain(GraphLouvainParams::default()),
            GraphAlgorithm::EigenvectorCentrality => {
                GraphAlgorithmParams::EigenvectorCentrality(GraphEigenvectorCentralityParams::default())
            },
            GraphAlgorithm::KatzCentrality => {
                GraphAlgorithmParams::KatzCentrality(GraphKatzCentralityParams::default())
            },
            GraphAlgorithm::Hits => GraphAlgorithmParams::Hits(GraphHitsParams::default()),
            GraphAlgorithm::BetweennessCentrality => {
                GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams::default())
            },
            GraphAlgorithm::EdgeBetweennessCentrality => {
                GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams::default())
            },
            GraphAlgorithm::Sssp => {
                GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex: GraphSsspSource::I64(0), cutoff: None })
            },
            GraphAlgorithm::DegreesAll => GraphAlgorithmParams::DegreesAll(GraphDegreesAllParams),
            GraphAlgorithm::InDegreesAll => GraphAlgorithmParams::InDegreesAll(GraphInDegreesAllParams),
            GraphAlgorithm::OutDegreesAll => GraphAlgorithmParams::OutDegreesAll(GraphOutDegreesAllParams),
            GraphAlgorithm::WeaklyConnectedComponents => {
                GraphAlgorithmParams::WeaklyConnectedComponents(GraphWccParams)
            },
            GraphAlgorithm::StronglyConnectedComponents => {
                GraphAlgorithmParams::StronglyConnectedComponents(GraphSccParams)
            },
            GraphAlgorithm::TriangleCountAll => GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
            GraphAlgorithm::CoreNumber => GraphAlgorithmParams::CoreNumber(GraphCoreNumberParams::default()),
            GraphAlgorithm::KCore => GraphAlgorithmParams::KCore(GraphKCoreParams::default()),
            GraphAlgorithm::MinimumSpanningTree => GraphAlgorithmParams::MinimumSpanningTree(GraphMstParams),
            GraphAlgorithm::Leiden => GraphAlgorithmParams::Leiden(GraphLeidenParams::default()),
            GraphAlgorithm::Ecg => GraphAlgorithmParams::Ecg(GraphEcgParams::default()),
            GraphAlgorithm::SpectralModularityMaximization => {
                GraphAlgorithmParams::SpectralModularityMaximization(GraphSpectralClusteringParams::default())
            },
            GraphAlgorithm::ForceAtlas2 => GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
            GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
                GraphAlgorithmParams::Similarity(GraphSimilarityParams::default())
            },
        }
    }

    /// Result schema for an algorithm parameter payload.
    ///
    /// This preserves native-owned parameter-sensitive schema semantics. SQL
    /// adapters may still apply a final user-facing overlay for modes that are
    /// not part of the native IR.
    pub fn result_schema_for_params(&self, params: &GraphAlgorithmParams) -> SchemaRef {
        assert!(params.supports_algorithm(self.algorithm), "graph algorithm and parameter variant must match");
        if let GraphAlgorithmParams::Bfs(params) = params
            && params.path_output.is_some()
        {
            return Arc::new(Schema::new(vec![
                Field::new("path_index", DataType::Int64, false),
                Field::new("source", DataType::Int64, false),
                Field::new("target", DataType::Int64, false),
                Field::new("vertex", DataType::Int64, false),
                Field::new("distance", DataType::Int64, false),
            ]));
        }
        let mut fields =
            self.base_result_schema().fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
        if let GraphAlgorithmParams::Bfs(params) = params
            && params.return_target_info
        {
            fields.push(Field::new("target_found", DataType::Boolean, false));
            fields.push(Field::new("target_distance", DataType::Int64, true));
        }
        Arc::new(Schema::new(fields))
    }
}

/// Source-of-truth list of supported cuGraph operations.
///
/// Order is preserved by [`GraphAlgorithm::all`] and is the order in which
/// the SQL surface iterates algorithms when registering UDTFs.
pub const GRAPH_OPERATION_REGISTRY: &[GraphOperationDefinition] = &[
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::PageRank,
        name: "pagerank",
        layout_requirement: GraphLayoutRequirement::Transposed,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Bfs,
        name: "bfs",
        layout_requirement: GraphLayoutRequirement::Forward,
        empty_input_policy: EmptyGraphInputPolicy::BFS,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Louvain,
        name: "louvain",
        layout_requirement: GraphLayoutRequirement::Forward,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::EigenvectorCentrality,
        name: "eigenvector_centrality",
        layout_requirement: GraphLayoutRequirement::Transposed,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::KatzCentrality,
        name: "katz_centrality",
        layout_requirement: GraphLayoutRequirement::Transposed,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Hits,
        name: "hits",
        layout_requirement: GraphLayoutRequirement::Transposed,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::BetweennessCentrality,
        name: "betweenness_centrality",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::VALIDATE_SELECTORS_THEN_RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::EdgeBetweennessCentrality,
        name: "edge_betweenness_centrality",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::VALIDATE_SELECTORS_THEN_RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Sssp,
        name: "sssp",
        layout_requirement: GraphLayoutRequirement::Forward,
        empty_input_policy: EmptyGraphInputPolicy::REQUIRE_GRAPH,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::DegreesAll,
        name: "degrees_all",
        layout_requirement: GraphLayoutRequirement::Transposed,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::InDegreesAll,
        name: "in_degrees_all",
        layout_requirement: GraphLayoutRequirement::Transposed,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::OutDegreesAll,
        name: "out_degrees_all",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::WeaklyConnectedComponents,
        name: "weakly_connected_components",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::StronglyConnectedComponents,
        name: "strongly_connected_components",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::TriangleCountAll,
        name: "triangle_count_all",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::CoreNumber,
        name: "core_number",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::KCore,
        name: "k_core",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::MinimumSpanningTree,
        name: "minimum_spanning_tree",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::PersonalizedPageRank,
        name: "personalized_pagerank",
        layout_requirement: GraphLayoutRequirement::Transposed,
        empty_input_policy: EmptyGraphInputPolicy::REQUIRE_GRAPH_AND_PERSONALIZATION,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Leiden,
        name: "leiden",
        layout_requirement: GraphLayoutRequirement::Forward,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Ecg,
        name: "ecg",
        layout_requirement: GraphLayoutRequirement::Forward,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::SpectralModularityMaximization,
        name: "spectral_modularity_maximization",
        layout_requirement: GraphLayoutRequirement::Forward,
        empty_input_policy: EmptyGraphInputPolicy::REQUIRE_GRAPH,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::ForceAtlas2,
        name: "force_atlas2",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::RETURN_EMPTY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Jaccard,
        name: "jaccard",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::SIMILARITY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Sorensen,
        name: "sorensen",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::SIMILARITY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Overlap,
        name: "overlap",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::SIMILARITY,
    },
    GraphOperationDefinition {
        algorithm: GraphAlgorithm::Cosine,
        name: "cosine",
        layout_requirement: GraphLayoutRequirement::Any,
        empty_input_policy: EmptyGraphInputPolicy::SIMILARITY,
    },
];

/// Edge-list column bindings consumed by a graph algorithm.
///
/// `weight` and `edge_id` are optional. When `weight` is present,
/// duplicate-edge normalization follows the selected
/// [`GraphConstructionPolicy`] and [`crate::plan::EdgeWeightCombiner`] so conflicting
/// weights are deterministic. When `edge_id` is present, the column is carried
/// through graph construction as the stable user-visible edge id used by
/// edge-id predicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgeColumnBindings {
    /// Source-vertex column name in the edge-list input.
    pub source: String,
    /// Destination-vertex column name in the edge-list input.
    pub destination: String,
    /// Optional edge-weight column name.
    pub weight: Option<String>,
    /// Optional stable edge-id column name.
    pub edge_id: Option<String>,
}

impl GraphEdgeColumnBindings {
    /// Create edge-list column bindings.
    pub fn new(source: impl Into<String>, destination: impl Into<String>, weight: Option<impl Into<String>>) -> Self {
        Self { source: source.into(), destination: destination.into(), weight: weight.map(Into::into), edge_id: None }
    }

    /// Attach a stable edge-id column binding.
    pub fn with_edge_id(mut self, edge_id: impl Into<String>) -> Self {
        self.edge_id = Some(edge_id.into());
        self
    }
}

/// Graph construction options shared by cuGraph algorithms.
///
/// Defaults match Python cuGraph: directed graph construction, vertex
/// renumbering enabled, and Python-compatible edge-list normalization. Changing
/// `directed` affects whether reverse edges are inserted by the
/// Python-compatible policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphConstructionOptions {
    /// Whether graph construction treats edges as directed.
    pub directed: bool,
    /// Whether cuGraph may renumber vertices internally.
    pub renumber: bool,
    /// Edge-list normalization policy used before graph construction.
    pub policy: GraphConstructionPolicy,
}

impl Default for GraphConstructionOptions {
    fn default() -> Self {
        Self { directed: true, renumber: true, policy: GraphConstructionPolicy::PythonCugraph }
    }
}

/// Graph construction policy understood by the native IR.
///
/// Mirrors `cugraph::cudf_interop::GraphConstructionPolicy`. See
/// `algeon_datafusion::CugraphGraphConstructionPolicy` for the SQL-level
/// semantics and trade-offs between the two variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphConstructionPolicy {
    /// Normalize edge lists to match Python cuGraph semantics.
    PythonCugraph,
    /// Pass edge lists directly to libcugraph-compatible construction.
    RawLibcugraph,
}

impl GraphConstructionPolicy {
    /// Stable lowercase construction-policy code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PythonCugraph => "python_cugraph",
            Self::RawLibcugraph => "raw_libcugraph",
        }
    }
}

impl fmt::Display for GraphConstructionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
