//! Native graph algorithm IR.
//!
//! This module defines stable graph algorithm names, graph construction
//! options, edge-list normalization, parameter defaults, and result schemas.
//! The IR can be built without the optional cuGraph feature; execution support
//! is selected later by the adapter/runtime.
use std::sync::Arc;

use arrow::datatypes::DataType as ArrowDataType;

use super::{Expr, Field, NodeId, PlanNode, ProjectionNode, QueryPlan, Schema, SchemaRef};

mod centrality;
mod community;
mod definition;
mod layout;
mod residency;
mod similarity;
mod structure;
mod traversal;

pub use centrality::{
    DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD, GraphBetweennessCentralityParams, GraphBetweennessSeeds,
    GraphEdgeBetweennessCentralityParams, GraphEigenvectorCentralityParams, GraphHitsParams, GraphKatzCentralityParams,
    GraphPageRankParams, GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
};
pub use community::{
    GraphEcgParams, GraphLeidenParams, GraphLouvainParams, GraphSccParams, GraphSpectralClusteringParams,
    GraphWccParams,
};
pub use definition::{
    EmptyGraphEdgeInputAction, EmptyGraphInputPolicy, EmptyGraphSideInputPolicy, GRAPH_OPERATION_REGISTRY,
    GraphAlgorithm, GraphConstructionOptions, GraphConstructionPolicy, GraphEdgeColumnBindings,
    GraphOperationDefinition, GraphOutputSchemaKind, GraphSourceVertexRequirement, GraphVertexIdTypeSupport,
    GraphWeightPolicy,
};
pub use layout::{
    GraphForceAtlas2InitialPositionsInput, GraphForceAtlas2Params, GraphForceAtlas2VertexAttributesInput,
    GraphLayoutRequirement,
};
pub use residency::{
    GraphResidencyDescriptor, GraphResidencyOrientation, GraphResidencyPlanAnalysis, GraphSourceFingerprint,
    GraphSourceFingerprintBypassReason, ResidentGraphKey, edge_plan_has_simple_directed_edge_topology,
    graph_residency_descriptor_for_edge_plan, graph_residency_descriptor_for_local_parquet_source,
};
pub use similarity::{GraphSimilarityParams, GraphVertexPairInput};
pub use structure::{
    EdgeDedupStrategy, EdgeNormalizeSpec, EdgeWeightCombiner, GraphCoreNumberParams, GraphDegreesAllParams,
    GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams, GraphMstParams, GraphOutDegreesAllParams,
    GraphTriangleCountAllParams,
};
pub use traversal::{
    GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPathOutput, GraphBfsPredicateInputs, GraphEdgePredicateInput,
    GraphPredicateInput, GraphSsspParams, GraphSsspSource,
};

/// Algorithm-specific parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphAlgorithmParams {
    PageRank(GraphPageRankParams),
    PersonalizedPageRank(GraphPersonalizedPageRankParams),
    Bfs(Box<GraphBfsParams>),
    Louvain(GraphLouvainParams),
    EigenvectorCentrality(GraphEigenvectorCentralityParams),
    KatzCentrality(GraphKatzCentralityParams),
    Hits(GraphHitsParams),
    BetweennessCentrality(GraphBetweennessCentralityParams),
    EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams),
    Sssp(GraphSsspParams),
    DegreesAll(GraphDegreesAllParams),
    InDegreesAll(GraphInDegreesAllParams),
    OutDegreesAll(GraphOutDegreesAllParams),
    WeaklyConnectedComponents(GraphWccParams),
    StronglyConnectedComponents(GraphSccParams),
    TriangleCountAll(GraphTriangleCountAllParams),
    CoreNumber(GraphCoreNumberParams),
    KCore(GraphKCoreParams),
    MinimumSpanningTree(GraphMstParams),
    Leiden(GraphLeidenParams),
    Ecg(GraphEcgParams),
    SpectralModularityMaximization(GraphSpectralClusteringParams),
    ForceAtlas2(GraphForceAtlas2Params),
    Similarity(GraphSimilarityParams),
}

impl GraphAlgorithmParams {
    /// Whether this payload is valid for an algorithm variant.
    ///
    /// Similarity deliberately has one shared payload for four metric
    /// identities. The metric stays in [`GraphAlgorithm`] so the pair side
    /// input and its validation cannot drift across copied parameter variants.
    pub const fn supports_algorithm(&self, algorithm: GraphAlgorithm) -> bool {
        match self {
            Self::PageRank(_) => matches!(algorithm, GraphAlgorithm::PageRank),
            Self::PersonalizedPageRank(_) => matches!(algorithm, GraphAlgorithm::PersonalizedPageRank),
            Self::Bfs(_) => matches!(algorithm, GraphAlgorithm::Bfs),
            Self::Louvain(_) => matches!(algorithm, GraphAlgorithm::Louvain),
            Self::EigenvectorCentrality(_) => matches!(algorithm, GraphAlgorithm::EigenvectorCentrality),
            Self::KatzCentrality(_) => matches!(algorithm, GraphAlgorithm::KatzCentrality),
            Self::Hits(_) => matches!(algorithm, GraphAlgorithm::Hits),
            Self::BetweennessCentrality(_) => matches!(algorithm, GraphAlgorithm::BetweennessCentrality),
            Self::EdgeBetweennessCentrality(_) => matches!(algorithm, GraphAlgorithm::EdgeBetweennessCentrality),
            Self::Sssp(_) => matches!(algorithm, GraphAlgorithm::Sssp),
            Self::DegreesAll(_) => matches!(algorithm, GraphAlgorithm::DegreesAll),
            Self::InDegreesAll(_) => matches!(algorithm, GraphAlgorithm::InDegreesAll),
            Self::OutDegreesAll(_) => matches!(algorithm, GraphAlgorithm::OutDegreesAll),
            Self::WeaklyConnectedComponents(_) => matches!(algorithm, GraphAlgorithm::WeaklyConnectedComponents),
            Self::StronglyConnectedComponents(_) => matches!(algorithm, GraphAlgorithm::StronglyConnectedComponents),
            Self::TriangleCountAll(_) => matches!(algorithm, GraphAlgorithm::TriangleCountAll),
            Self::CoreNumber(_) => matches!(algorithm, GraphAlgorithm::CoreNumber),
            Self::KCore(_) => matches!(algorithm, GraphAlgorithm::KCore),
            Self::MinimumSpanningTree(_) => matches!(algorithm, GraphAlgorithm::MinimumSpanningTree),
            Self::Leiden(_) => matches!(algorithm, GraphAlgorithm::Leiden),
            Self::Ecg(_) => matches!(algorithm, GraphAlgorithm::Ecg),
            Self::SpectralModularityMaximization(_) => {
                matches!(algorithm, GraphAlgorithm::SpectralModularityMaximization)
            },
            Self::ForceAtlas2(_) => matches!(algorithm, GraphAlgorithm::ForceAtlas2),
            Self::Similarity(_) => algorithm.is_similarity(),
        }
    }

    /// Visits every relation-backed side-input node in stable execution order.
    pub fn side_input_node_ids(&self) -> Box<dyn Iterator<Item = NodeId> + '_> {
        match self {
            Self::Bfs(params) => Box::new(params.side_input_node_ids()),
            Self::PersonalizedPageRank(params) => Box::new(params.side_input_node_ids()),
            Self::Similarity(params) => Box::new(params.side_input_node_ids()),
            Self::Sssp(params) => Box::new(params.side_input_node_ids()),
            Self::BetweennessCentrality(params) => Box::new(params.side_input_node_ids()),
            Self::EdgeBetweennessCentrality(params) => Box::new(params.side_input_node_ids()),
            Self::ForceAtlas2(params) => Box::new(params.side_input_node_ids()),
            _ => Box::new(std::iter::empty()),
        }
    }
}

/// Native graph algorithm specification.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphAlgorithmSpec {
    /// Algorithm to execute.
    pub algorithm: GraphAlgorithm,
    /// Edge-list column bindings.
    pub edge_columns: GraphEdgeColumnBindings,
    /// Shared graph construction options.
    pub graph_options: GraphConstructionOptions,
    /// Whether this graph algorithm consumes an explicit edge-normalize node
    /// inserted from Python-compatible construction semantics.
    pub edge_normalize_inserted: bool,
    /// Algorithm-specific parameter payload.
    pub params: GraphAlgorithmParams,
    /// Stable output schema expected from execution.
    pub output_schema: SchemaRef,
}

impl GraphAlgorithmSpec {
    /// Create a graph algorithm spec with an explicit output schema.
    pub fn new(
        algorithm: GraphAlgorithm,
        edge_columns: GraphEdgeColumnBindings,
        graph_options: GraphConstructionOptions,
        params: GraphAlgorithmParams,
        output_schema: SchemaRef,
    ) -> Self {
        assert!(params.supports_algorithm(algorithm), "graph algorithm and parameter variant must match");
        Self { algorithm, edge_columns, graph_options, edge_normalize_inserted: false, params, output_schema }
    }

    /// Mark this spec as consuming an explicit edge-normalize input.
    pub fn with_edge_normalize_inserted(mut self, inserted: bool) -> Self {
        self.edge_normalize_inserted = inserted;
        self
    }

    /// Create a graph algorithm spec using this crate's stable result schema.
    pub fn with_stable_schema(
        algorithm: GraphAlgorithm,
        edge_columns: GraphEdgeColumnBindings,
        graph_options: GraphConstructionOptions,
        params: GraphAlgorithmParams,
    ) -> Self {
        let output_schema = algorithm.definition().result_schema_for_params(&params);
        Self::new(algorithm, edge_columns, graph_options, params, output_schema)
    }

    /// Unique key emitted by this exact algorithm parameterization, when proven.
    pub fn unique_output_key_columns(&self) -> &'static [&'static str] {
        if matches!(&self.params, GraphAlgorithmParams::Bfs(params) if params.path_output.is_some()) {
            return &["path_index"];
        }
        if self.algorithm.definition().output_vertex_is_unique() { &["vertex"] } else { &[] }
    }
}

/// Stable graph identity available to transient coalescing and resident-cache
/// lookup, when known at planning time.
///
/// Keep this function in sync with graph execution dispatch. It intentionally
/// returns `None` for paths whose graph construction cannot form a stable
/// identity, such as edge-id graph construction and BFS include-edges
/// filtering.
pub fn planned_resident_graph_key(
    descriptor: &GraphResidencyDescriptor,
    spec: &GraphAlgorithmSpec,
) -> Option<ResidentGraphKey> {
    if spec.edge_columns.edge_id.is_some() {
        return None;
    }
    match (&spec.algorithm, &spec.params) {
        (GraphAlgorithm::Bfs, GraphAlgorithmParams::Bfs(params)) if params.predicates.include_edges.is_none() => {
            Some(descriptor.resident_key(spec, GraphResidencyOrientation::Forward))
        },
        (GraphAlgorithm::PageRank, GraphAlgorithmParams::PageRank(_)) => {
            Some(descriptor.resident_key(spec, GraphResidencyOrientation::Transposed))
        },
        (
            GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine,
            GraphAlgorithmParams::Similarity(_),
        ) => Some(descriptor.resident_key(spec, GraphResidencyOrientation::Forward)),
        _ => None,
    }
}

/// Native graph algorithm operation over one edge-list input.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphAlgorithmNode {
    /// Edge-list input node.
    pub input: NodeId,
    /// All semantic inputs in execution order: edge-list input first, followed
    /// by BFS source/predicate side-input nodes in stable role order.
    pub inputs: Vec<NodeId>,
    /// Graph algorithm specification.
    pub spec: GraphAlgorithmSpec,
}

impl GraphAlgorithmNode {
    /// Output schema declared by the graph algorithm spec.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.spec.output_schema
    }
}

/// Stable SQL-facing result schema for graph algorithms.
pub fn stable_graph_result_schema(algorithm: GraphAlgorithm) -> SchemaRef {
    let fields = match algorithm {
        GraphAlgorithm::PageRank
        | GraphAlgorithm::PersonalizedPageRank
        | GraphAlgorithm::EigenvectorCentrality
        | GraphAlgorithm::KatzCentrality
        | GraphAlgorithm::BetweennessCentrality => {
            // Centrality-style algorithms expose one score per vertex.
            vec![Field::new("vertex", ArrowDataType::Int64, false), Field::new("value", ArrowDataType::Float64, false)]
        },
        GraphAlgorithm::EdgeBetweennessCentrality => {
            vec![
                Field::new("source", ArrowDataType::Int64, false),
                Field::new("destination", ArrowDataType::Int64, false),
                Field::new("score", ArrowDataType::Float64, false),
            ]
        },
        GraphAlgorithm::Bfs => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            Field::new("distance", ArrowDataType::Int64, false),
            Field::new("predecessor", ArrowDataType::Int64, true),
        ],
        GraphAlgorithm::Louvain => {
            vec![
                Field::new("vertex", ArrowDataType::Int64, false),
                Field::new("partition", ArrowDataType::Int64, false),
            ]
        },
        GraphAlgorithm::Hits => {
            // HITS exposes both hub and authority scores per vertex;
            // upstream cuGraph names them `hub` / `authority`. The
            // adapter renames to `hub_score` / `authority_score` in
            // the stable SQL schema so the meaning is obvious to query
            // authors.
            vec![
                Field::new("vertex", ArrowDataType::Int64, false),
                Field::new("hub_score", ArrowDataType::Float64, false),
                Field::new("authority_score", ArrowDataType::Float64, false),
            ]
        },
        GraphAlgorithm::Sssp => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            // SSSP distances are weighted (Float64); BFS uses Int64 hop counts.
            Field::new("distance", ArrowDataType::Float64, false),
            // -1 sentinel means no predecessor (source vertex or unreachable);
            // nullable=true matches the BFS predecessor convention.
            Field::new("predecessor", ArrowDataType::Int64, true),
        ],
        GraphAlgorithm::DegreesAll => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            Field::new("in_degree", ArrowDataType::Int64, false),
            Field::new("out_degree", ArrowDataType::Int64, false),
        ],
        GraphAlgorithm::InDegreesAll => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            Field::new("in_degree", ArrowDataType::Int64, false),
        ],
        GraphAlgorithm::OutDegreesAll => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            Field::new("out_degree", ArrowDataType::Int64, false),
        ],
        // WCC and SCC both return a vertex-label assignment. Labels are
        // component identifiers and are not semantically ordered; parity
        // tests use partition-equivalence comparison.
        GraphAlgorithm::WeaklyConnectedComponents | GraphAlgorithm::StronglyConnectedComponents => {
            vec![Field::new("vertex", ArrowDataType::Int64, false), Field::new("label", ArrowDataType::Int64, false)]
        },
        GraphAlgorithm::TriangleCountAll => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            Field::new("triangle_count", ArrowDataType::Int64, false),
        ],
        GraphAlgorithm::CoreNumber => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            Field::new("core_number", ArrowDataType::Int64, false),
        ],
        // KCore and MST return an edge list; column names follow the
        // graph-SQL `src`/`dst` convention. The current output contract omits weight.
        GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => {
            vec![Field::new("src", ArrowDataType::Int64, false), Field::new("dst", ArrowDataType::Int64, false)]
        },
        // Leiden, ECG, and spectral modularity maximization return a
        // vertex-partition assignment. Labels are
        // partition identifiers; parity tests use partition-equivalence.
        GraphAlgorithm::Leiden | GraphAlgorithm::Ecg | GraphAlgorithm::SpectralModularityMaximization => {
            vec![
                Field::new("vertex", ArrowDataType::Int64, false),
                Field::new("partition", ArrowDataType::Int64, false),
            ]
        },
        GraphAlgorithm::ForceAtlas2 => vec![
            Field::new("vertex", ArrowDataType::Int64, false),
            Field::new("x", ArrowDataType::Float32, false),
            Field::new("y", ArrowDataType::Float32, false),
        ],
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => vec![
            Field::new("first", ArrowDataType::Int64, false),
            Field::new("second", ArrowDataType::Int64, false),
            Field::new("similarity", ArrowDataType::Float64, false),
        ],
    };
    Arc::new(Schema::new(fields))
}

/// Expected graph result schema after algorithm-specific output options.
///
/// Most graph algorithms use [`stable_graph_result_schema`] exactly. BFS can
/// opt into target metadata columns through [`GraphBfsParams::return_target_info`].
pub fn graph_result_schema_for_params(algorithm: GraphAlgorithm, params: &GraphAlgorithmParams) -> SchemaRef {
    algorithm.definition().result_schema_for_params(params)
}

/// Expected graph result schema in the input graph's numeric vertex-id domain.
pub fn graph_result_schema_for_vertex_type(
    algorithm: GraphAlgorithm,
    params: &GraphAlgorithmParams,
    vertex_type: &ArrowDataType,
) -> SchemaRef {
    assert!(
        matches!(vertex_type, ArrowDataType::Int32 | ArrowDataType::Int64),
        "native graph vertex type must be Int32 or Int64"
    );
    let vertex_columns = algorithm.definition().vertex_output_columns();
    let fields = graph_result_schema_for_params(algorithm, params)
        .fields()
        .iter()
        .map(|field| {
            if vertex_columns.contains(&field.name().as_str()) {
                field.as_ref().clone().with_data_type(vertex_type.clone())
            } else {
                field.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}
