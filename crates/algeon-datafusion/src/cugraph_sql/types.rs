use std::{num::NonZeroUsize, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::Result as DataFusionResult;
use query_planning::plan::{
    DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD, GraphAlgorithmParams as NativeGraphAlgorithmParams,
    GraphBetweennessSeeds as NativeGraphBetweennessSeeds, GraphConstructionOptions as NativeGraphConstructionOptions,
    GraphConstructionPolicy as NativeGraphConstructionPolicy, GraphKCoreDegreeType as NativeGraphKCoreDegreeType,
    GraphSsspSource as NativeGraphSsspSource,
};

use super::{plan, registry::CugraphAlgorithm};
use crate::error::{Error, ErrorCode, Result};

pub(crate) const CUGRAPH_EDGE_SOURCE_COLUMN: &str = "src";
pub(crate) const CUGRAPH_EDGE_DESTINATION_COLUMN: &str = "dst";
pub(crate) const CUGRAPH_EDGE_WEIGHT_COLUMN: &str = "weight";
pub(crate) const CUGRAPH_EDGE_ID_COLUMN: &str = "edge_id";

pub(crate) const CUGRAPH_VERTEX_COLUMN: &str = "vertex";
pub(crate) const CUGRAPH_VALUE_COLUMN: &str = "value";
pub(crate) const CUGRAPH_FIRST_VERTEX_COLUMN: &str = "first";
pub(crate) const CUGRAPH_SECOND_VERTEX_COLUMN: &str = "second";
pub(crate) const CUGRAPH_RADIUS_COLUMN: &str = "radius";
pub(crate) const CUGRAPH_MOBILITY_COLUMN: &str = "mobility";
pub(crate) const CUGRAPH_MASS_COLUMN: &str = "mass";
pub(crate) const CUGRAPH_X_COLUMN: &str = "x";
pub(crate) const CUGRAPH_Y_COLUMN: &str = "y";

/// Cumulative host-memory ceilings for one explicit CPU-to-GPU-to-CPU graph
/// island.
///
/// Input staging covers the edge source and every graph side input under one
/// shared limit. Shared input buffers count once across batches and partitions.
/// Output staging covers all eagerly materialized Arrow batches
/// retained until GPU ownership has been released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuIslandHostBoundaryConfig {
    max_input_staging_bytes: NonZeroUsize,
    max_output_staging_bytes: NonZeroUsize,
}

impl GpuIslandHostBoundaryConfig {
    /// Default cumulative input Arrow staging limit in bytes (1 GiB).
    pub const DEFAULT_MAX_INPUT_STAGING_BYTES: usize = 1024 * 1024 * 1024;
    /// Default cumulative output Arrow staging limit in bytes (1 GiB).
    pub const DEFAULT_MAX_OUTPUT_STAGING_BYTES: usize = 1024 * 1024 * 1024;

    /// Creates finite non-zero cumulative input and output staging limits.
    pub fn try_new(max_input_staging_bytes: usize, max_output_staging_bytes: usize) -> DataFusionResult<Self> {
        let max_input_staging_bytes = NonZeroUsize::new(max_input_staging_bytes)
            .ok_or_else(|| plan("cuGraph GPU-island input staging limit must be non-zero"))?;
        let max_output_staging_bytes = NonZeroUsize::new(max_output_staging_bytes)
            .ok_or_else(|| plan("cuGraph GPU-island output staging limit must be non-zero"))?;
        Ok(Self { max_input_staging_bytes, max_output_staging_bytes })
    }

    /// Returns the shared input staging ceiling in bytes for one graph call.
    pub fn max_input_staging_bytes(self) -> usize {
        self.max_input_staging_bytes.get()
    }

    /// Returns the cumulative materialized output ceiling in bytes for one graph call.
    pub fn max_output_staging_bytes(self) -> usize {
        self.max_output_staging_bytes.get()
    }
}

impl Default for GpuIslandHostBoundaryConfig {
    fn default() -> Self {
        Self {
            max_input_staging_bytes: NonZeroUsize::new(Self::DEFAULT_MAX_INPUT_STAGING_BYTES)
                .expect("default cuGraph input staging limit is non-zero"),
            max_output_staging_bytes: NonZeroUsize::new(Self::DEFAULT_MAX_OUTPUT_STAGING_BYTES)
                .expect("default cuGraph output staging limit is non-zero"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
/// Graph-construction defaults and host staging limits installed for cuGraph SQL calls.
pub struct CugraphSqlConfig {
    /// Graph construction defaults applied before named value arguments.
    pub default_graph_options: CugraphGraphOptions,
    /// Cumulative Arrow staging limits for graph inputs and materialized outputs.
    pub host_boundary_config: GpuIslandHostBoundaryConfig,
}

#[derive(Debug, Clone, PartialEq)]
/// Validated SQL call contract used to construct and execute one graph algorithm.
///
/// The parameter variant, input-column flags, and output schema must agree with
/// `algorithm`; SQL binding validates this combination before execution.
pub struct CugraphAlgorithmSpec {
    /// Algorithm whose input and output contracts this specification applies.
    pub algorithm: CugraphAlgorithm,
    /// Whether the edge child supplies the canonical `weight` column.
    pub has_weight: bool,
    /// Whether the edge child supplies the canonical `edge_id` column.
    pub has_edge_id: bool,
    /// Construction semantics applied to the input edge list.
    pub graph_options: CugraphGraphOptions,
    /// Parameters whose variant must match `algorithm`.
    pub params: CugraphAlgorithmParams,
    /// Arrow schema derived from validated inputs and result-shape options.
    pub output_schema: SchemaRef,
}

impl CugraphAlgorithmSpec {
    #[cfg(feature = "cugraph")]
    pub(crate) fn is_bfs_path_output(&self) -> bool {
        matches!(
            &self.params,
            CugraphAlgorithmParams::Bfs(params) if params.output.mode == CugraphBfsOutputMode::Path
        )
    }

    pub(crate) fn validate_logical_edge_id_capability(&self, logical_mapping: bool) -> Result<()> {
        if !logical_mapping {
            return Ok(());
        }
        let (include_edge_ids, include_edge_id_col) = match &self.params {
            CugraphAlgorithmParams::Bfs(params) => (
                params.predicates.include_edge_ids.is_some(),
                params.predicates.include_edges.as_ref().is_some_and(|input| input.has_edge_id),
            ),
            _ => (false, false),
        };
        if !self.has_edge_id && !include_edge_ids && !include_edge_id_col {
            return Ok(());
        }
        Err(Error::new(
            ErrorCode::RequiredGpuOperationUnsupported,
            "algeon_datafusion.cugraph_sql.logical_mapping.preflight",
            format!(
                "cuGraph {} logical vertex-id mapping does not support edge-id inputs",
                self.algorithm.function_name()
            ),
        )
        .with_fact("edge_id_col", self.has_edge_id)
        .with_fact("include_edge_ids", include_edge_ids)
        .with_fact("include_edge_id_col", include_edge_id_col))
    }

    pub(crate) fn predicate_source_specs(&self) -> Vec<Vec<PredicateSourceColumnRole>> {
        let mut sources = Vec::new();
        match &self.params {
            CugraphAlgorithmParams::Bfs(params) => {
                if params.source.relation().is_some() {
                    sources.push(vec![PredicateSourceColumnRole::Vertex]);
                }
                for _input in [
                    params.predicates.include_vertices.as_ref(),
                    params.predicates.exclude_vertices.as_ref(),
                    params.predicates.target_vertices.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    sources.push(vec![PredicateSourceColumnRole::Vertex]);
                }
                if params.predicates.include_edge_ids.is_some() {
                    sources.push(vec![PredicateSourceColumnRole::EdgeId]);
                }
                if let Some(input) = &params.predicates.include_edges {
                    let mut roles = vec![PredicateSourceColumnRole::Source, PredicateSourceColumnRole::Destination];
                    if input.has_edge_id {
                        roles.push(PredicateSourceColumnRole::EdgeId);
                    }
                    sources.push(roles);
                }
            },
            CugraphAlgorithmParams::PersonalizedPageRank(_) => {
                sources.push(vec![PredicateSourceColumnRole::Vertex, PredicateSourceColumnRole::FloatValue])
            },
            CugraphAlgorithmParams::Similarity(_) => {
                sources.push(vec![PredicateSourceColumnRole::FirstVertex, PredicateSourceColumnRole::SecondVertex])
            },
            CugraphAlgorithmParams::ForceAtlas2(params) => {
                if let Some(input) = &params.vertex_attributes {
                    let mut roles = vec![PredicateSourceColumnRole::Vertex];
                    if input.has_radius {
                        roles.push(PredicateSourceColumnRole::Radius);
                    }
                    if input.has_mobility {
                        roles.push(PredicateSourceColumnRole::Mobility);
                    }
                    if input.has_mass {
                        roles.push(PredicateSourceColumnRole::Mass);
                    }
                    sources.push(roles);
                }
                if params.initial_positions.is_some() {
                    sources.push(vec![
                        PredicateSourceColumnRole::Vertex,
                        PredicateSourceColumnRole::X,
                        PredicateSourceColumnRole::Y,
                    ]);
                }
            },
            _ => {},
        }
        sources
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredicateSourceColumnRole {
    Vertex,
    EdgeId,
    FloatValue,
    Source,
    Destination,
    FirstVertex,
    SecondVertex,
    Radius,
    Mobility,
    Mass,
    X,
    Y,
}

impl PredicateSourceColumnRole {
    pub(crate) const fn canonical_name(self) -> &'static str {
        match self {
            Self::Vertex => CUGRAPH_VERTEX_COLUMN,
            Self::EdgeId => CUGRAPH_EDGE_ID_COLUMN,
            Self::FloatValue => CUGRAPH_VALUE_COLUMN,
            Self::Source => CUGRAPH_EDGE_SOURCE_COLUMN,
            Self::Destination => CUGRAPH_EDGE_DESTINATION_COLUMN,
            Self::FirstVertex => CUGRAPH_FIRST_VERTEX_COLUMN,
            Self::SecondVertex => CUGRAPH_SECOND_VERTEX_COLUMN,
            Self::Radius => CUGRAPH_RADIUS_COLUMN,
            Self::Mobility => CUGRAPH_MOBILITY_COLUMN,
            Self::Mass => CUGRAPH_MASS_COLUMN,
            Self::X => CUGRAPH_X_COLUMN,
            Self::Y => CUGRAPH_Y_COLUMN,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Edge-list construction options applied before algorithm execution.
pub struct CugraphGraphOptions {
    /// Whether input edges describe a directed graph; defaults to `true`.
    pub directed: bool,
    /// Whether graph construction renumbers input vertex ids; defaults to `true`.
    pub renumber: bool,
    /// Duplicate and reverse-edge normalization policy; defaults to Python cuGraph semantics.
    pub construction_policy: CugraphGraphConstructionPolicy,
}

impl Default for CugraphGraphOptions {
    fn default() -> Self {
        Self { directed: true, renumber: true, construction_policy: CugraphGraphConstructionPolicy::PythonCugraph }
    }
}

impl CugraphGraphOptions {
    pub(crate) fn to_native(&self) -> NativeGraphConstructionOptions {
        NativeGraphConstructionOptions {
            directed: self.directed,
            renumber: self.renumber,
            policy: self.construction_policy.to_native(),
        }
    }

    pub(crate) fn from_native(options: &NativeGraphConstructionOptions) -> Self {
        Self {
            directed: options.directed,
            renumber: options.renumber,
            construction_policy: CugraphGraphConstructionPolicy::from_native(options.policy),
        }
    }
}

/// SQL-facing selector for the cuGraph edge-list ingress contract.
///
/// Mirrors `cugraph::cudf_interop::GraphConstructionPolicy` and is the public
/// `construction_policy` option exposed in the cuGraph table-function JSON.
///
/// - [`PythonCugraph`](Self::PythonCugraph): default Python cuGraph semantics;
///   normalizes duplicate/reverse edges before construction.
/// - [`RawLibcugraph`](Self::RawLibcugraph): pass already-normalized SQL edges
///   directly to libcugraph. Null edge rows are still filtered because libcugraph
///   has no null-edge semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphGraphConstructionPolicy {
    /// Normalize duplicate and reverse edges using Python cuGraph construction semantics.
    PythonCugraph,
    /// Pass normalized edges to libcugraph directly, after filtering null edge rows.
    RawLibcugraph,
}

impl CugraphGraphConstructionPolicy {
    /// Returns the stable SQL option value for this selector.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PythonCugraph => "python_cugraph",
            Self::RawLibcugraph => "raw_libcugraph",
        }
    }

    pub(crate) const fn to_native(self) -> NativeGraphConstructionPolicy {
        match self {
            Self::PythonCugraph => NativeGraphConstructionPolicy::PythonCugraph,
            Self::RawLibcugraph => NativeGraphConstructionPolicy::RawLibcugraph,
        }
    }

    pub(crate) const fn from_native(policy: NativeGraphConstructionPolicy) -> Self {
        match policy {
            NativeGraphConstructionPolicy::PythonCugraph => Self::PythonCugraph,
            NativeGraphConstructionPolicy::RawLibcugraph => Self::RawLibcugraph,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Algorithm-specific SQL parameters paired with [`CugraphAlgorithmSpec::algorithm`].
pub enum CugraphAlgorithmParams {
    /// Rank vertices by damped incoming-link influence.
    PageRank(CugraphPageRankParams),
    /// Rank vertices using a caller-supplied personalization relation.
    PersonalizedPageRank(CugraphPersonalizedPageRankParams),
    /// Traverse unweighted paths from one or more sources.
    Bfs(Box<CugraphBfsParams>),
    /// Detect communities by hierarchical modularity optimization.
    Louvain(CugraphLouvainParams),
    /// Compute vertex influence from neighboring centrality scores.
    EigenvectorCentrality(CugraphEigenvectorCentralityParams),
    /// Compute attenuated neighbor influence with a constant bias.
    KatzCentrality(CugraphKatzCentralityParams),
    /// Compute hub and authority scores.
    Hits(CugraphHitsParams),
    /// Score vertices by their contribution to shortest paths.
    BetweennessCentrality(CugraphBetweennessCentralityParams),
    /// Score edges by their contribution to shortest paths.
    EdgeBetweennessCentrality(CugraphEdgeBetweennessCentralityParams),
    /// Compute weighted shortest distances from one source.
    Sssp(CugraphSsspParams),
    /// Return incoming and outgoing vertex degrees.
    DegreesAll(CugraphDegreesAllParams),
    /// Return incoming vertex degrees.
    InDegreesAll(CugraphInDegreesAllParams),
    /// Return outgoing vertex degrees.
    OutDegreesAll(CugraphOutDegreesAllParams),
    /// Label components using connectivity without edge direction.
    WeaklyConnectedComponents(CugraphWccParams),
    /// Label components whose vertices are mutually reachable by directed paths.
    StronglyConnectedComponents(CugraphSccParams),
    /// Count triangles involving each vertex.
    TriangleCountAll(CugraphTriangleCountAllParams),
    /// Return each vertex's largest containing k-core.
    CoreNumber(CugraphCoreNumberParams),
    /// Extract the subgraph meeting the requested core-number bound.
    KCore(CugraphKCoreParams),
    /// Return minimum-weight spanning edges.
    MinimumSpanningTree(CugraphMstParams),
    /// Detect communities with modularity refinement.
    Leiden(CugraphLeidenParams),
    /// Detect communities using an ensemble of Louvain partitions.
    Ecg(CugraphEcgParams),
    /// Cluster vertices from a modularity spectral embedding.
    SpectralModularityMaximization(CugraphSpectralClusteringParams),
    /// Compute two-dimensional vertex layout coordinates.
    ForceAtlas2(CugraphForceAtlas2Params),
    /// Compute the selected coefficient for explicit vertex pairs.
    Similarity(CugraphSimilarityParams),
}

impl CugraphAlgorithmParams {
    pub(crate) fn from_native_defaults(algorithm: CugraphAlgorithm, source: Option<CugraphBfsSource>) -> Result<Self> {
        Ok(match algorithm.native_algorithm().definition().default_params() {
            NativeGraphAlgorithmParams::PageRank(params) => Self::PageRank(CugraphPageRankParams {
                alpha: params.alpha,
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            }),
            NativeGraphAlgorithmParams::PersonalizedPageRank(params) => {
                Self::PersonalizedPageRank(CugraphPersonalizedPageRankParams {
                    pagerank: CugraphPageRankParams {
                        alpha: params.pagerank.alpha,
                        epsilon: params.pagerank.epsilon,
                        max_iterations: params.pagerank.max_iterations,
                    },
                    personalization: CugraphPageRankPersonalizationTable,
                })
            },
            NativeGraphAlgorithmParams::Bfs(params) => Self::Bfs(Box::new(CugraphBfsParams {
                source: source.unwrap_or_else(|| CugraphBfsSource::scalar_i64(params.source_vertex)),
                depth_limit: params.depth_limit,
                predicates: CugraphBfsPredicateInputs::default(),
                output: CugraphBfsOutputOptions::default(),
            })),
            NativeGraphAlgorithmParams::Louvain(params) => Self::Louvain(CugraphLouvainParams {
                max_level: params.max_level,
                threshold: params.threshold,
                resolution: params.resolution,
            }),
            NativeGraphAlgorithmParams::EigenvectorCentrality(params) => {
                Self::EigenvectorCentrality(CugraphEigenvectorCentralityParams {
                    epsilon: params.epsilon,
                    max_iterations: params.max_iterations,
                })
            },
            NativeGraphAlgorithmParams::KatzCentrality(params) => Self::KatzCentrality(CugraphKatzCentralityParams {
                alpha: params.alpha,
                beta: params.beta,
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
            }),
            NativeGraphAlgorithmParams::Hits(params) => Self::Hits(CugraphHitsParams {
                epsilon: params.epsilon,
                max_iterations: params.max_iterations,
                normalize: params.normalize,
            }),
            NativeGraphAlgorithmParams::BetweennessCentrality(params) => {
                Self::BetweennessCentrality(CugraphBetweennessCentralityParams {
                    seeds: params.seeds.map(CugraphBetweennessSeeds::from_native),
                    k: params.k,
                    normalized: params.normalized,
                    include_endpoints: params.include_endpoints,
                    exact_vertex_threshold: params.exact_vertex_threshold,
                })
            },
            NativeGraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
                Self::EdgeBetweennessCentrality(CugraphEdgeBetweennessCentralityParams {
                    seeds: params.seeds.map(CugraphBetweennessSeeds::from_native),
                    k: params.k,
                    normalized: params.normalized,
                    exact_vertex_threshold: params.exact_vertex_threshold,
                })
            },
            NativeGraphAlgorithmParams::Sssp(params) => {
                let default_source = match params.source_vertex {
                    NativeGraphSsspSource::I64(value) => value,
                    NativeGraphSsspSource::Input(_) => {
                        return Err(plan("native SSSP defaults never contain relation-backed selectors"));
                    },
                };
                let source = source.unwrap_or_else(|| CugraphBfsSource::scalar_i64(default_source));
                let CugraphBfsSource::Scalar(source) = source else {
                    return Err(plan("source_vertices is supported only by cugraph_bfs"));
                };
                Self::Sssp(CugraphSsspParams { source_vertex: source.into(), cutoff: params.cutoff })
            },
            NativeGraphAlgorithmParams::DegreesAll(_) => Self::DegreesAll(CugraphDegreesAllParams),
            NativeGraphAlgorithmParams::InDegreesAll(_) => Self::InDegreesAll(CugraphInDegreesAllParams),
            NativeGraphAlgorithmParams::OutDegreesAll(_) => Self::OutDegreesAll(CugraphOutDegreesAllParams),
            NativeGraphAlgorithmParams::WeaklyConnectedComponents(_) => {
                Self::WeaklyConnectedComponents(CugraphWccParams)
            },
            NativeGraphAlgorithmParams::StronglyConnectedComponents(_) => {
                Self::StronglyConnectedComponents(CugraphSccParams)
            },
            NativeGraphAlgorithmParams::TriangleCountAll(_) => Self::TriangleCountAll(CugraphTriangleCountAllParams),
            NativeGraphAlgorithmParams::CoreNumber(params) => Self::CoreNumber(CugraphCoreNumberParams {
                degree_type: CugraphKCoreDegreeType::from_native(params.degree_type),
            }),
            NativeGraphAlgorithmParams::KCore(params) => Self::KCore(CugraphKCoreParams {
                k: params.k,
                degree_type: CugraphKCoreDegreeType::from_native(params.degree_type),
            }),
            NativeGraphAlgorithmParams::MinimumSpanningTree(_) => Self::MinimumSpanningTree(CugraphMstParams),
            NativeGraphAlgorithmParams::Leiden(params) => Self::Leiden(CugraphLeidenParams {
                max_level: params.max_level,
                resolution: params.resolution,
                seed: params.seed,
            }),
            NativeGraphAlgorithmParams::Ecg(params) => Self::Ecg(CugraphEcgParams {
                min_weight: params.min_weight,
                ensemble_size: params.ensemble_size,
                max_level: params.max_level,
                threshold: params.threshold,
                resolution: params.resolution,
                seed: params.seed,
            }),
            NativeGraphAlgorithmParams::SpectralModularityMaximization(params) => {
                Self::SpectralModularityMaximization(CugraphSpectralClusteringParams {
                    n_clusters: params.n_clusters,
                    n_eigenvectors: params.n_eigenvectors,
                    evs_tolerance: params.evs_tolerance,
                    seed: params.seed,
                })
            },
            NativeGraphAlgorithmParams::ForceAtlas2(params) => Self::ForceAtlas2(CugraphForceAtlas2Params {
                max_iter: params.max_iter,
                outbound_attraction_distribution: params.outbound_attraction_distribution,
                lin_log_mode: params.lin_log_mode,
                prevent_overlapping: params.prevent_overlapping,
                overlap_scaling_ratio: params.overlap_scaling_ratio,
                edge_weight_influence: params.edge_weight_influence,
                jitter_tolerance: params.jitter_tolerance,
                barnes_hut_optimize: params.barnes_hut_optimize,
                barnes_hut_theta: params.barnes_hut_theta,
                scaling_ratio: params.scaling_ratio,
                strong_gravity_mode: params.strong_gravity_mode,
                gravity: params.gravity,
                verbose: params.verbose,
                seed: params.seed,
                vertex_attributes: None,
                initial_positions: None,
            }),
            NativeGraphAlgorithmParams::Similarity(_) => Self::Similarity(CugraphSimilarityParams::default()),
        })
    }
}

/// Presence of the aligned vertex-pair relation consumed by a similarity
/// coefficient. Its child projection owns the canonical `first` and `second`
/// fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphVertexPairTable;

impl CugraphVertexPairTable {
    /// Constructs the relation-presence marker.
    ///
    /// The associated child plan supplies and validates the relation's data and schema.
    pub const fn new() -> Self {
        Self
    }
}

/// Parameters shared by the four explicit-pair similarity functions.
///
/// The metric remains on [`CugraphAlgorithm`] so all functions share the
/// same relation binding and cannot drift into copied SQL parameter types.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CugraphSimilarityParams {
    /// Presence of the relation supplying aligned `first` and `second` vertex pairs.
    pub vertex_pairs: CugraphVertexPairTable,
}

#[derive(Debug, Clone, PartialEq)]
/// Damping and convergence settings for PageRank.
pub struct CugraphPageRankParams {
    /// Damping factor in `[0, 1]`; defaults to `0.85`.
    pub alpha: f64,
    /// Positive convergence tolerance; defaults to `1e-5`.
    pub epsilon: f64,
    /// Positive iteration limit; defaults to `100`.
    pub max_iterations: u32,
}

impl Default for CugraphPageRankParams {
    fn default() -> Self {
        Self { alpha: 0.85, epsilon: 1e-5, max_iterations: 100 }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
/// PageRank settings and the presence of a personalization relation.
pub struct CugraphPersonalizedPageRankParams {
    /// Damping and convergence settings shared with ordinary PageRank.
    pub pagerank: CugraphPageRankParams,
    /// Presence of the child relation supplying personalization weights.
    pub personalization: CugraphPageRankPersonalizationTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Marks a planned personalization relation with canonical `vertex` and `value` columns.
pub struct CugraphPageRankPersonalizationTable;

impl CugraphPageRankPersonalizationTable {
    /// Constructs the relation-presence marker.
    ///
    /// The associated child plan supplies and validates the relation's data and schema.
    pub const fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Source, traversal predicates, and result-shape settings for breadth-first search.
pub struct CugraphBfsParams {
    /// Source vertex selector for the SQL BFS traversal.
    pub source: CugraphBfsSource,
    /// Optional non-negative BFS depth limit. `None` means unbounded depth.
    pub depth_limit: Option<i64>,
    /// Optional relation-backed predicate inputs.
    pub predicates: CugraphBfsPredicateInputs,
    /// SQL result-shape options. Defaults use the raw cuGraph distance/sentinel schema.
    pub output: CugraphBfsOutputOptions,
}

/// SQL-visible BFS source selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CugraphBfsSource {
    /// Scalar source vertex. Numeric scalar sources use the direct numeric
    /// path; string scalar sources are mapped through the logical-id
    /// codebook path.
    Scalar(CugraphBfsSourceVertex),
    /// Inline literal source list. This is lowered to a native source-vertices
    /// side input.
    LiteralList(Vec<CugraphBfsSourceVertex>),
    /// Relation-backed source vertices.
    Relation(CugraphGraphPredicateTable),
}

impl Default for CugraphBfsSource {
    fn default() -> Self {
        Self::scalar_i64(0)
    }
}

impl CugraphBfsSource {
    /// Selects one numeric source vertex without a relation-backed source input.
    pub fn scalar_i64(value: i64) -> Self {
        Self::Scalar(CugraphBfsSourceVertex::Int64(value))
    }

    /// Selects one logical string source vertex for mapping through the call's vertex codebook.
    pub fn scalar_utf8(value: impl Into<String>) -> Self {
        Self::Scalar(CugraphBfsSourceVertex::Utf8(value.into()))
    }

    /// Reports a string scalar or a literal list whose first value is a string.
    ///
    /// Relation-backed sources and empty lists return `false`; this does not validate
    /// a list's element types.
    pub fn is_utf8(&self) -> bool {
        match self {
            Self::Scalar(CugraphBfsSourceVertex::Utf8(_)) => true,
            Self::LiteralList(values) => matches!(values.first(), Some(CugraphBfsSourceVertex::Utf8(_))),
            Self::Scalar(CugraphBfsSourceVertex::Int64(_)) | Self::Relation(_) => false,
        }
    }

    /// Reports a numeric scalar or a literal list whose first value is numeric.
    ///
    /// Relation-backed sources and empty lists return `false`.
    pub fn is_literal_numeric_domain(&self) -> bool {
        match self {
            Self::Scalar(CugraphBfsSourceVertex::Int64(_)) => true,
            Self::LiteralList(values) => matches!(values.first(), Some(CugraphBfsSourceVertex::Int64(_))),
            Self::Scalar(CugraphBfsSourceVertex::Utf8(_)) | Self::Relation(_) => false,
        }
    }

    /// Returns the numeric scalar value, or `None` for a string, list, or relation selector.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Scalar(source) => source.as_i64(),
            Self::LiteralList(_) | Self::Relation(_) => None,
        }
    }

    /// Whether execution needs a source-vertices table; only a numeric scalar avoids that input.
    pub fn requires_source_vertices_input(&self) -> bool {
        !matches!(self, Self::Scalar(CugraphBfsSourceVertex::Int64(_)))
    }

    /// Returns whether logical planning already supplies a source-vertices child.
    ///
    /// Relations and numeric literal lists supply a child; string literals are
    /// materialized after logical-id mapping during execution.
    pub fn planned_source_vertices_input(&self) -> bool {
        match self {
            Self::Relation(_) => true,
            Self::LiteralList(values) => matches!(values.first(), Some(CugraphBfsSourceVertex::Int64(_))),
            Self::Scalar(_) => false,
        }
    }

    /// Returns `vertex` for a relation selector and the generated
    /// `__source_vertex` column name for a literal selector.
    pub fn source_vertices_column(&self) -> &str {
        match self {
            Self::Relation(_) => CUGRAPH_VERTEX_COLUMN,
            Self::Scalar(_) | Self::LiteralList(_) => "__source_vertex",
        }
    }

    /// Borrows the relation marker for a relation selector, or returns `None` for literals.
    pub fn relation(&self) -> Option<&CugraphGraphPredicateTable> {
        match self {
            Self::Relation(input) => Some(input),
            Self::Scalar(_) | Self::LiteralList(_) => None,
        }
    }

    /// Borrows one scalar as a one-element slice or the literal list.
    ///
    /// Returns `None` for a relation-backed selector.
    pub fn literal_values(&self) -> Option<&[CugraphBfsSourceVertex]> {
        match self {
            Self::Scalar(value) => Some(std::slice::from_ref(value)),
            Self::LiteralList(values) => Some(values.as_slice()),
            Self::Relation(_) => None,
        }
    }

    /// Returns the scalar/list source count, or `None` when a relation supplies the sources.
    pub fn literal_count(&self) -> Option<usize> {
        self.literal_values().map(<[_]>::len)
    }
}

/// SQL-visible BFS source vertex literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CugraphBfsSourceVertex {
    /// Numeric source vertex for the direct cuGraph vertex-id path.
    Int64(i64),
    /// Logical string source vertex. The SQL execution layer maps it to a
    /// deterministic numeric cuGraph vertex id for one graph call.
    Utf8(String),
}

/// SQL-visible scalar vertex selector used by algorithms whose native
/// execution accepts exactly one logical vertex value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CugraphVertexSelector {
    /// Numeric selector for the direct native vertex-id path.
    Int64(i64),
    /// Logical string selector mapped through the per-call codebook.
    Utf8(String),
}

impl CugraphVertexSelector {
    /// Returns whether this literal names a logical string vertex.
    pub const fn is_utf8(&self) -> bool {
        matches!(self, Self::Utf8(_))
    }

    /// Returns the numeric vertex value, or `None` for a logical string vertex.
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int64(value) => Some(*value),
            Self::Utf8(_) => None,
        }
    }
}

impl From<CugraphBfsSourceVertex> for CugraphVertexSelector {
    fn from(value: CugraphBfsSourceVertex) -> Self {
        match value {
            CugraphBfsSourceVertex::Int64(value) => Self::Int64(value),
            CugraphBfsSourceVertex::Utf8(value) => Self::Utf8(value),
        }
    }
}

impl Default for CugraphBfsSourceVertex {
    fn default() -> Self {
        Self::Int64(0)
    }
}

impl CugraphBfsSourceVertex {
    /// Returns whether this literal names a logical string vertex.
    pub const fn is_utf8(&self) -> bool {
        matches!(self, Self::Utf8(_))
    }

    /// Returns the numeric vertex value, or `None` for a logical string vertex.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int64(value) => Some(*value),
            Self::Utf8(_) => None,
        }
    }
}

/// SQL graph vertex-id domain after applying the algorithm-specific logical-id
/// policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CugraphVertexIdDomain {
    Int32,
    Int64,
    StringLogical,
}

impl CugraphVertexIdDomain {
    pub(crate) fn accepts(&self, data_type: &DataType) -> bool {
        match self {
            Self::Int32 => matches!(data_type, DataType::Int32),
            Self::Int64 => matches!(data_type, DataType::Int64),
            Self::StringLogical => is_string_vertex_dtype(data_type),
        }
    }

    pub(crate) const fn is_string_logical(&self) -> bool {
        matches!(self, Self::StringLogical)
    }

    pub(crate) const fn label(&self) -> &'static str {
        match self {
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::StringLogical => "Arrow string",
        }
    }

    pub(crate) fn output_data_type(&self) -> DataType {
        match self {
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::StringLogical => DataType::Utf8,
        }
    }

    /// Whether the per-call codebook must map this domain to the algorithm's
    /// native vertex ids: string domains always, numeric domains when the
    /// native graph type cannot index them.
    pub(crate) fn requires_mapping(&self, algorithm: CugraphAlgorithm) -> bool {
        !algorithm.native_algorithm().vertex_id_type_support().accepts_arrow_data_type(&self.output_data_type())
    }
}

pub(crate) fn vertex_id_domain_for_call(
    algorithm: CugraphAlgorithm,
    params: &CugraphAlgorithmParams,
    src_type: &DataType,
    dst_type: &DataType,
) -> Result<CugraphVertexIdDomain> {
    let selector_is_string = match params {
        CugraphAlgorithmParams::Bfs(params) => {
            params.source.literal_values().map(|values| values.first().is_some_and(CugraphBfsSourceVertex::is_utf8))
        },
        CugraphAlgorithmParams::Sssp(params) => Some(params.source_vertex.is_utf8()),
        CugraphAlgorithmParams::BetweennessCentrality(params) => {
            params.seeds.as_ref().and_then(|seeds| seeds.vertices.first()).map(CugraphVertexSelector::is_utf8)
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            params.seeds.as_ref().and_then(|seeds| seeds.vertices.first()).map(CugraphVertexSelector::is_utf8)
        },
        _ => None,
    };
    if selector_is_string == Some(true) {
        if !is_string_vertex_dtype(src_type) {
            return Err(unsupported_vertex_dtype_error(format!(
                "cuGraph {} string vertex selector requires a string source column, found {src_type:?}",
                algorithm.function_name()
            )));
        }
        if !is_string_vertex_dtype(dst_type) {
            return Err(unsupported_vertex_dtype_error(format!(
                "cuGraph {} string vertex selector requires a string destination column, found {dst_type:?}",
                algorithm.function_name()
            )));
        }
        return Ok(CugraphVertexIdDomain::StringLogical);
    }

    if selector_is_string == Some(false) && (is_string_vertex_dtype(src_type) || is_string_vertex_dtype(dst_type)) {
        return Err(vertex_dtype_mismatch_error(format!(
            "cuGraph {} numeric vertex selector requires Int32 or Int64 edge columns, found source={src_type:?}, destination={dst_type:?}",
            algorithm.function_name()
        )));
    }

    if is_string_vertex_dtype(src_type) || is_string_vertex_dtype(dst_type) {
        if !is_string_vertex_dtype(src_type) || !is_string_vertex_dtype(dst_type) {
            return Err(vertex_dtype_mismatch_error(format!(
                "cuGraph {} logical string vertex mapping requires string source and destination columns, found source={src_type:?}, destination={dst_type:?}",
                algorithm.function_name()
            )));
        }
        return Ok(CugraphVertexIdDomain::StringLogical);
    }

    if selector_is_string == Some(true) {
        return Err(vertex_dtype_mismatch_error(format!(
            "cuGraph {} string vertex selector requires string edge columns, found source={src_type:?}, destination={dst_type:?}",
            algorithm.function_name()
        )));
    }

    if !is_numeric_vertex_dtype(src_type) {
        return Err(unsupported_vertex_dtype_error(format!(
            "cuGraph {} source column has unsupported dtype {src_type:?}; supported vertex types: {}",
            algorithm.function_name(),
            supported_vertex_types().join(", ")
        )));
    }
    if !is_numeric_vertex_dtype(dst_type) {
        return Err(unsupported_vertex_dtype_error(format!(
            "cuGraph {} destination column has unsupported dtype {dst_type:?}; supported vertex types: {}",
            algorithm.function_name(),
            supported_vertex_types().join(", ")
        )));
    }
    if src_type != dst_type {
        return Err(vertex_dtype_mismatch_error(
            "cuGraph source and destination columns must have the same numeric vertex dtype",
        ));
    }
    match src_type {
        DataType::Int32 => Ok(CugraphVertexIdDomain::Int32),
        DataType::Int64 => Ok(CugraphVertexIdDomain::Int64),
        _ => unreachable!("numeric vertex dtype checked above"),
    }
}

/// Arrow dtypes of vertex columns that cuGraph indexes as integer vertex IDs.
pub(crate) const NUMERIC_VERTEX_DTYPES: [DataType; 2] = [DataType::Int32, DataType::Int64];

/// Arrow dtypes of vertex columns that a per-call codebook maps to native
/// vertex IDs.
pub(crate) const STRING_VERTEX_DTYPES: [DataType; 3] = [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View];

pub(crate) fn is_numeric_vertex_dtype(data_type: &DataType) -> bool {
    NUMERIC_VERTEX_DTYPES.contains(data_type)
}

/// Every vertex dtype a cuGraph call accepts, numeric first, as Arrow type
/// names.
pub(crate) fn supported_vertex_types() -> Vec<String> {
    NUMERIC_VERTEX_DTYPES.iter().chain(&STRING_VERTEX_DTYPES).map(|data_type| format!("{data_type:?}")).collect()
}

/// Validates the static schema contract for an explicit-pair similarity
/// relation after the caller resolves it from the catalog.
pub(crate) fn validate_similarity_pair_table_schema(
    _table: &CugraphVertexPairTable,
    schema: &SchemaRef,
    vertex_domain: &CugraphVertexIdDomain,
) -> Result<()> {
    let first = schema.field_with_name(CUGRAPH_FIRST_VERTEX_COLUMN).map_err(|_| {
        missing_column_error("cuGraph vertex-pair child is missing canonical first vertex column 'first'")
    })?;
    let second = schema.field_with_name(CUGRAPH_SECOND_VERTEX_COLUMN).map_err(|_| {
        missing_column_error("cuGraph vertex-pair child is missing canonical second vertex column 'second'")
    })?;

    let both_in_logical_string_domain = vertex_domain.is_string_logical()
        && vertex_domain.accepts(first.data_type())
        && vertex_domain.accepts(second.data_type());
    if first.data_type() != second.data_type() && !both_in_logical_string_domain {
        return Err(vertex_dtype_mismatch_error(format!(
            "cuGraph vertex-pair columns 'first' and 'second' must have the same dtype, found {:?} and {:?}",
            first.data_type(),
            second.data_type(),
        )));
    }
    if !vertex_domain.accepts(first.data_type()) {
        return Err(vertex_dtype_mismatch_error(format!(
            "cuGraph vertex-pair first vertex column 'first' must have dtype {}",
            vertex_domain.label(),
        )));
    }
    if !vertex_domain.accepts(second.data_type()) {
        return Err(vertex_dtype_mismatch_error(format!(
            "cuGraph vertex-pair second vertex column 'second' must have dtype {}",
            vertex_domain.label(),
        )));
    }
    Ok(())
}

fn unsupported_vertex_dtype_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedType, "algeon_datafusion.cugraph_sql.vertex_id_type", message)
        .with_fact("reason", "unsupported_vertex_dtype")
}

fn missing_column_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::FieldNotFound, "algeon_datafusion.cugraph_sql.vertex_pair_schema", message)
}

fn vertex_dtype_mismatch_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidExecutionPlan, "algeon_datafusion.cugraph_sql.vertex_id_type", message)
        .with_fact("reason", "vertex_dtype_mismatch")
}

pub(crate) fn is_string_vertex_dtype(data_type: &DataType) -> bool {
    STRING_VERTEX_DTYPES.contains(data_type)
}

/// SQL-visible BFS output options.
///
/// `mode = Raw` exposes the raw cuGraph distance/sentinel schema.
/// `mode = Normalized` adds `reachable` and converts
/// unreachable distance / absent predecessor sentinels to nulls. `mode = Path`
/// reconstructs one source-to-target path from the BFS predecessor output.
/// `return_target_info = true` is an explicit schema-widening request that
/// requires a target-vertex predicate and appends `target_found` plus nullable
/// `target_distance` to every output row for raw / normalized output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CugraphBfsOutputOptions {
    /// Result schema and sentinel-normalization mode.
    pub mode: CugraphBfsOutputMode,
    /// Whether target metadata is requested.
    pub return_target_info: bool,
}

/// BFS SQL result-shape selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CugraphBfsOutputMode {
    /// Preserve the raw `vertex`, `distance`, `predecessor` schema.
    #[default]
    Raw,
    /// Return `vertex`, nullable `distance`, nullable `predecessor`, and
    /// `reachable`.
    Normalized,
    /// Return one row per vertex on the reconstructed source-to-target path.
    Path,
}

impl CugraphBfsOutputMode {
    /// Stable JSON label used by `output_mode` and metadata UDTFs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Normalized => "normalized",
            Self::Path => "path",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Optional relation inputs restricting BFS traversal or selecting target vertices.
///
/// `None` leaves the corresponding predicate unspecified.
pub struct CugraphBfsPredicateInputs {
    /// Optional allowed-vertex relation; absence imposes no allowed-vertex predicate.
    pub include_vertices: Option<CugraphGraphPredicateTable>,
    /// Optional excluded-vertex relation; absence excludes no vertices by this predicate.
    pub exclude_vertices: Option<CugraphGraphPredicateTable>,
    /// Optional target-vertex relation used for target metadata or path output.
    pub target_vertices: Option<CugraphGraphPredicateTable>,
    /// Optional allowed edge-id relation, requiring an edge-id input contract.
    pub include_edge_ids: Option<CugraphGraphPredicateTable>,
    /// Optional allowed edge relation identified by endpoints and, when declared, edge ids.
    pub include_edges: Option<CugraphGraphEdgePredicateTable>,
}

impl CugraphBfsPredicateInputs {
    /// Returns whether any optional traversal predicate relation is present.
    pub fn has_inputs(&self) -> bool {
        self.include_vertices.is_some()
            || self.exclude_vertices.is_some()
            || self.target_vertices.is_some()
            || self.include_edge_ids.is_some()
            || self.include_edges.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Marks a planned one-column vertex or edge-id predicate relation.
pub struct CugraphGraphPredicateTable;

impl CugraphGraphPredicateTable {
    /// Constructs the relation-presence marker.
    ///
    /// The associated child plan supplies and validates the relation's data and schema.
    pub const fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Declares the canonical edge-predicate relation and whether it also supplies edge ids.
pub struct CugraphGraphEdgePredicateTable {
    /// Whether the predicate relation supplies `edge_id` in addition to `src` and `dst`.
    pub has_edge_id: bool,
}

impl CugraphGraphEdgePredicateTable {
    /// Declares whether the edge-predicate child has a canonical `edge_id` column.
    pub const fn new(has_edge_id: bool) -> Self {
        Self { has_edge_id }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Hierarchy depth and modularity settings for Louvain community detection.
pub struct CugraphLouvainParams {
    /// Maximum hierarchy depth / iteration count. Must be >= 1.
    pub max_level: u32,
    /// Convergence threshold for modularity gain at each level. Must be >= 0.
    pub threshold: f64,
    /// Resolution parameter in the modularity formula. Must be > 0.
    pub resolution: f64,
}

impl Default for CugraphLouvainParams {
    fn default() -> Self {
        Self { max_level: 100, threshold: 1.0e-7, resolution: 1.0 }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Convergence settings for eigenvector centrality.
pub struct CugraphEigenvectorCentralityParams {
    /// Positive convergence tolerance; defaults to `1e-6`.
    pub epsilon: f64,
    /// Positive iteration limit; defaults to `200`.
    pub max_iterations: u32,
}

impl Default for CugraphEigenvectorCentralityParams {
    fn default() -> Self {
        // Mirrors `query_planning::plan::GraphEigenvectorCentralityParams::default`.
        Self { epsilon: 1.0e-6, max_iterations: 200 }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Attenuation, bias, and convergence settings for Katz centrality.
pub struct CugraphKatzCentralityParams {
    /// Neighbor-score attenuation in `(0, 1]`; defaults to `0.01`.
    pub alpha: f64,
    /// Positive bias added to each vertex score; defaults to `1.0`.
    pub beta: f64,
    /// Positive convergence tolerance; defaults to `1e-6`.
    pub epsilon: f64,
    /// Positive iteration limit; defaults to `1000`.
    pub max_iterations: u32,
}

impl Default for CugraphKatzCentralityParams {
    fn default() -> Self {
        // Mirrors `query_planning::plan::GraphKatzCentralityParams::default`.
        Self { alpha: 0.01, beta: 1.0, epsilon: 1.0e-6, max_iterations: 1000 }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Convergence and output normalization for hub and authority scores.
pub struct CugraphHitsParams {
    /// Positive convergence tolerance; defaults to `1e-5`.
    pub epsilon: f64,
    /// Positive iteration limit; defaults to `100`.
    pub max_iterations: u32,
    /// Whether hub and authority scores are normalized; defaults to `false`.
    pub normalize: bool,
}

impl Default for CugraphHitsParams {
    fn default() -> Self {
        // Mirrors `query_planning::plan::GraphHitsParams::default`.
        Self { epsilon: 1.0e-5, max_iterations: 100, normalize: false }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Explicit logical vertex selectors used as betweenness traversal sources.
pub struct CugraphBetweennessSeeds {
    /// Traversal-source vertex ids, interpreted in the input graph's logical vertex domain.
    pub vertices: Vec<CugraphVertexSelector>,
}

impl CugraphBetweennessSeeds {
    pub(crate) fn from_native(seeds: NativeGraphBetweennessSeeds) -> Self {
        match seeds {
            NativeGraphBetweennessSeeds::I64(vertices) => {
                Self { vertices: vertices.into_iter().map(CugraphVertexSelector::Int64).collect() }
            },
            NativeGraphBetweennessSeeds::Input { .. } => {
                unreachable!("native registry defaults never contain relation-backed betweenness seeds")
            },
        }
    }

    /// Returns the number of explicit traversal-source vertices.
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Returns whether no explicit traversal-source vertices are listed.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Source selection and score normalization for vertex betweenness centrality.
pub struct CugraphBetweennessCentralityParams {
    /// Explicit traversal sources, or `None` for automatic source selection.
    pub seeds: Option<CugraphBetweennessSeeds>,
    /// Optional deterministic source count; execution selects the first `k` distinct
    /// graph vertices in stable order. Mutually exclusive with explicit `seeds`.
    pub k: Option<u64>,
    /// Whether returned betweenness scores use the algorithm's normalized scale.
    pub normalized: bool,
    /// Whether path endpoints contribute to vertex betweenness scores.
    pub include_endpoints: bool,
    /// Largest graph vertex count eligible for automatic exact traversal when neither `seeds` nor `k` is set.
    pub exact_vertex_threshold: u64,
}

impl Default for CugraphBetweennessCentralityParams {
    fn default() -> Self {
        Self {
            seeds: None,
            k: None,
            normalized: true,
            include_endpoints: false,
            exact_vertex_threshold: DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Source selection and score normalization for edge betweenness centrality.
pub struct CugraphEdgeBetweennessCentralityParams {
    /// Explicit traversal sources, or `None` for automatic source selection.
    pub seeds: Option<CugraphBetweennessSeeds>,
    /// Optional deterministic source count; execution selects the first `k` distinct
    /// graph vertices in stable order. Mutually exclusive with explicit `seeds`.
    pub k: Option<u64>,
    /// Whether returned betweenness scores use the algorithm's normalized scale.
    pub normalized: bool,
    /// Largest graph vertex count eligible for automatic exact traversal when neither `seeds` nor `k` is set.
    pub exact_vertex_threshold: u64,
}

impl Default for CugraphEdgeBetweennessCentralityParams {
    fn default() -> Self {
        Self {
            seeds: None,
            k: None,
            normalized: true,
            exact_vertex_threshold: DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Source vertex and optional weighted-distance cutoff for shortest paths.
pub struct CugraphSsspParams {
    /// One logical source vertex from which weighted shortest distances are computed.
    pub source_vertex: CugraphVertexSelector,
    /// Maximum edge-weight sum to consider. `None` means no cutoff; must be >= 0 when set.
    pub cutoff: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Parameter marker for returning both incoming and outgoing degree counts.
pub struct CugraphDegreesAllParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Parameter marker for returning incoming degree counts.
pub struct CugraphInDegreesAllParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Parameter marker for returning outgoing degree counts.
pub struct CugraphOutDegreesAllParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Parameter marker for weakly connected component labels.
pub struct CugraphWccParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Parameter marker for strongly connected component labels.
pub struct CugraphSccParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Parameter marker for triangle counts for each graph vertex.
pub struct CugraphTriangleCountAllParams;

/// Degree direction for core-number and k-core SQL parameters.
///
/// Mirrors `query_planning::plan::GraphKCoreDegreeType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CugraphKCoreDegreeType {
    /// Use incoming degree.
    In,
    /// Use outgoing degree.
    Out,
    #[default]
    /// Use combined incoming and outgoing degree; the default.
    InOut,
}

impl CugraphKCoreDegreeType {
    /// Returns the stable SQL option value for this degree direction.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::InOut => "in_out",
        }
    }

    pub(crate) const fn from_native(degree_type: NativeGraphKCoreDegreeType) -> Self {
        match degree_type {
            NativeGraphKCoreDegreeType::In => Self::In,
            NativeGraphKCoreDegreeType::Out => Self::Out,
            NativeGraphKCoreDegreeType::InOut => Self::InOut,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Degree direction used to compute vertex core numbers.
pub struct CugraphCoreNumberParams {
    /// Degree direction defining the core-number calculation.
    pub degree_type: CugraphKCoreDegreeType,
}

impl Default for CugraphCoreNumberParams {
    fn default() -> Self {
        Self { degree_type: CugraphKCoreDegreeType::InOut }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Minimum core number and degree direction for extracting a k-core.
pub struct CugraphKCoreParams {
    /// Minimum core number retained in the extracted subgraph; defaults to `2`.
    pub k: u32,
    /// Degree direction defining core membership.
    pub degree_type: CugraphKCoreDegreeType,
}

impl Default for CugraphKCoreParams {
    fn default() -> Self {
        // Mirrors `query_planning::plan::GraphKCoreParams::default`.
        Self { k: 2, degree_type: CugraphKCoreDegreeType::InOut }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Parameter marker for the minimum spanning tree operation.
pub struct CugraphMstParams;

/// Parameters for Leiden community detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphLeidenParams {
    /// Maximum hierarchy depth / iteration count. Must be >= 1.
    pub max_level: u32,
    /// Resolution parameter (`gamma`) in the modularity formula. Must be > 0.
    pub resolution: f64,
    /// RNG seed for the random choices in the refinement step. The same seed
    /// does not guarantee identical communities across runs.
    pub seed: u64,
}

impl Default for CugraphLeidenParams {
    fn default() -> Self {
        // Matches query_planning::plan::GraphLeidenParams::default().
        Self { max_level: 100, resolution: 1.0, seed: 0 }
    }
}

/// Parameters for ECG community detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphEcgParams {
    /// Minimum edge weight in the final Louvain graph. Must be >= 0.
    pub min_weight: f64,
    /// Number of ensemble Louvain iterations. Must be >= 1.
    pub ensemble_size: u32,
    /// Maximum hierarchy depth for final Louvain. Must be >= 1.
    pub max_level: u32,
    /// Convergence threshold for final Louvain. Must be in (0, 1].
    pub threshold: f64,
    /// Resolution parameter in the modularity formula. Must be in (0, 1].
    pub resolution: f64,
    /// RNG seed for the random initial community labels of each Louvain pass.
    /// The same seed does not guarantee identical communities across runs.
    pub seed: u64,
}

impl Default for CugraphEcgParams {
    fn default() -> Self {
        // Matches query_planning::plan::GraphEcgParams::default().
        Self { min_weight: 0.001, ensemble_size: 10, max_level: 100, threshold: 1.0e-7, resolution: 1.0, seed: 0 }
    }
}

/// Parameters for spectral modularity maximization.
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphSpectralClusteringParams {
    /// Desired number of clusters. Must be in `[2, i32::MAX]`.
    pub n_clusters: u32,
    /// Number of eigenvectors to use. Must be in `[2, i32::MAX]` and must not
    /// exceed `n_clusters`. The first eigenvector is constant, so one
    /// eigenvector would give k-means a constant embedding.
    pub n_eigenvectors: u32,
    /// Eigenvalue-solver tolerance. Must be in `[0, 1)`.
    pub evs_tolerance: f64,
    /// RNG seed. Default 0.
    ///
    /// The seed only fixes the random initialization (Lanczos start vector
    /// and k-means centroid seeding). It does not make the partition
    /// bit-identical across runs: GPU parallel reductions inside k-means
    /// accumulate in scheduling order, so same-seed replays may land in a
    /// different but equally valid community assignment.
    pub seed: u64,
}

impl Default for CugraphSpectralClusteringParams {
    fn default() -> Self {
        // Matches query_planning::plan::GraphSpectralClusteringParams::default().
        Self { n_clusters: 2, n_eigenvectors: 2, evs_tolerance: 0.01, seed: 0 }
    }
}

/// Parameters for ForceAtlas2 graph layout.
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphForceAtlas2Params {
    /// Maximum number of iterations. Must be in `[1, i32::MAX]`.
    pub max_iter: u32,
    /// Distribute attraction along outbound edges.
    pub outbound_attraction_distribution: bool,
    /// Use the lin-log attraction model.
    pub lin_log_mode: bool,
    /// Prevent nodes from overlapping.
    ///
    /// Currently rejected by the SQL parser because ForceAtlas2 overlap
    /// prevention requires per-vertex radius side input that this SQL surface
    /// does not expose yet.
    pub prevent_overlapping: bool,
    /// Repulsion scaling for overlapping nodes. Must be > 0.
    pub overlap_scaling_ratio: f64,
    /// Influence of edge weights. Must be >= 0.
    pub edge_weight_influence: f64,
    /// Swinging tolerance. Must be >= 0.
    pub jitter_tolerance: f64,
    /// Use the Barnes-Hut approximation.
    pub barnes_hut_optimize: bool,
    /// Barnes-Hut speed/accuracy trade-off. Must be in [0, 1].
    pub barnes_hut_theta: f64,
    /// Repulsion scaling ratio. Must be > 0.
    pub scaling_ratio: f64,
    /// Use stronger gravity for distant nodes.
    pub strong_gravity_mode: bool,
    /// Gravity strength. Must be >= 0.
    pub gravity: f64,
    /// Whether cuGraph should print convergence info.
    pub verbose: bool,
    /// RNG seed for the random initial positions.
    ///
    /// The seed does not make runs reproducible: the parallel force
    /// iteration accumulates in scheduling-dependent order, so reruns may
    /// place individual vertices far apart. Compare layouts structurally,
    /// not coordinate-by-coordinate.
    pub seed: u64,
    /// Optional relation containing per-vertex layout attributes.
    pub vertex_attributes: Option<CugraphForceAtlas2VertexAttributesTable>,
    /// Optional relation containing complete warm-start positions.
    pub initial_positions: Option<CugraphForceAtlas2InitialPositionsTable>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Declares optional columns in the planned vertex-attributes relation.
pub struct CugraphForceAtlas2VertexAttributesTable {
    /// Whether a `radius` column is present.
    pub has_radius: bool,
    /// Whether a `mobility` column is present.
    pub has_mobility: bool,
    /// Whether a `mass` column is present.
    pub has_mass: bool,
}

impl CugraphForceAtlas2VertexAttributesTable {
    /// Declares which optional attribute columns are present in the relation child.
    pub const fn new(has_radius: bool, has_mobility: bool, has_mass: bool) -> Self {
        Self { has_radius, has_mobility, has_mass }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Marks a planned relation supplying canonical `vertex`, `x`, and `y` starting positions.
pub struct CugraphForceAtlas2InitialPositionsTable;

impl Default for CugraphForceAtlas2InitialPositionsTable {
    fn default() -> Self {
        Self
    }
}

impl CugraphForceAtlas2InitialPositionsTable {
    /// Constructs the relation-presence marker.
    ///
    /// The associated child plan supplies and validates the relation's data and schema.
    pub const fn new() -> Self {
        Self
    }
}

impl Default for CugraphForceAtlas2Params {
    fn default() -> Self {
        // Matches query_planning::plan::GraphForceAtlas2Params::default().
        Self {
            max_iter: 500,
            outbound_attraction_distribution: false,
            lin_log_mode: false,
            prevent_overlapping: false,
            overlap_scaling_ratio: 2.0,
            edge_weight_influence: 1.0,
            jitter_tolerance: 1.0,
            barnes_hut_optimize: false,
            barnes_hut_theta: 0.5,
            scaling_ratio: 2.0,
            strong_gravity_mode: false,
            gravity: 1.0,
            verbose: false,
            seed: 0,
            vertex_attributes: None,
            initial_positions: None,
        }
    }
}

/// Returns the algorithm registry's base result schema.
///
/// Use [`result_schema_for_params`] for BFS options that change the result shape.
pub fn stable_result_schema(algorithm: CugraphAlgorithm) -> SchemaRef {
    algorithm.native_algorithm().definition().base_result_schema()
}

/// SQL output schema for one BFS call after applying output options.
pub fn bfs_result_schema(params: &CugraphBfsParams) -> SchemaRef {
    let vertex_type = if params.source.is_utf8() { DataType::Utf8 } else { DataType::Int64 };
    bfs_result_schema_with_vertex_type(params, vertex_type)
}

pub(crate) fn bfs_result_schema_for_vertex_domain(
    params: &CugraphBfsParams,
    domain: &CugraphVertexIdDomain,
) -> SchemaRef {
    bfs_result_schema_with_vertex_type(params, domain.output_data_type())
}

fn bfs_result_schema_with_vertex_type(params: &CugraphBfsParams, vertex_type: DataType) -> SchemaRef {
    let predecessor_type = vertex_type.clone();
    let mut fields = match params.output.mode {
        CugraphBfsOutputMode::Raw => vec![
            Field::new("vertex", vertex_type, false),
            Field::new("distance", DataType::Int64, false),
            Field::new("predecessor", predecessor_type, true),
        ],
        CugraphBfsOutputMode::Normalized => vec![
            Field::new("vertex", vertex_type, false),
            Field::new("distance", DataType::Int64, true),
            Field::new("predecessor", predecessor_type, true),
            Field::new("reachable", DataType::Boolean, false),
        ],
        CugraphBfsOutputMode::Path => vec![
            Field::new("path_index", DataType::Int64, false),
            Field::new("source", vertex_type.clone(), false),
            Field::new("target", vertex_type.clone(), false),
            Field::new("vertex", vertex_type, false),
            Field::new("distance", DataType::Int64, false),
        ],
    };
    if params.output.mode != CugraphBfsOutputMode::Path && params.output.return_target_info {
        fields.push(Field::new("target_found", DataType::Boolean, false));
        fields.push(Field::new("target_distance", DataType::Int64, true));
    }
    Arc::new(Schema::new(fields))
}

pub(crate) fn result_schema_for_vertex_domain(
    algorithm: CugraphAlgorithm,
    params: &CugraphAlgorithmParams,
    domain: Option<&CugraphVertexIdDomain>,
) -> SchemaRef {
    let Some(domain) = domain else {
        return result_schema_for_params(algorithm, params);
    };

    match (algorithm, params) {
        (CugraphAlgorithm::Bfs, CugraphAlgorithmParams::Bfs(params)) => {
            bfs_result_schema_for_vertex_domain(params, domain)
        },
        _ => {
            let output_columns = algorithm.definition().vertex_output_columns();
            let output_type = domain.output_data_type();
            Arc::new(Schema::new(
                result_schema_for_params(algorithm, params)
                    .fields()
                    .iter()
                    .map(|field| {
                        output_columns.iter().find(|column| column.name == field.name()).map_or_else(
                            || field.as_ref().clone(),
                            |column| Field::new(column.name, output_type.clone(), column.nullable),
                        )
                    })
                    .collect::<Vec<_>>(),
            ))
        },
    }
}

/// SQL output schema for one parsed cuGraph algorithm call.
pub fn result_schema_for_params(algorithm: CugraphAlgorithm, params: &CugraphAlgorithmParams) -> SchemaRef {
    match (algorithm, params) {
        (CugraphAlgorithm::Bfs, CugraphAlgorithmParams::Bfs(params)) => bfs_result_schema(params),
        _ => stable_result_schema(algorithm),
    }
}

#[cfg(test)]
mod tests {
    use query_planning::plan::stable_graph_result_schema;

    use super::stable_result_schema;
    use crate::cugraph_sql::registry::CUGRAPH_OPERATION_REGISTRY;

    #[test]
    fn registry_algorithm_sql_output_schemas_agree_with_native_engine() {
        for entry in CUGRAPH_OPERATION_REGISTRY {
            let sql_schema = stable_result_schema(entry.algorithm);
            let native_schema = stable_graph_result_schema(entry.algorithm.native_algorithm());
            assert_eq!(
                sql_schema.as_ref(),
                native_schema.as_ref(),
                "SQL-facing output schema for {} must mirror the native engine schema",
                entry.sql_function_name,
            );
        }
    }
}
