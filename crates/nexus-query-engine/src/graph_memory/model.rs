use std::collections::HashMap;

use arrow::datatypes::{DataType, SchemaRef};

use super::{
    BETWEENNESS_MAX_CONCURRENT_SOURCES, BETWEENNESS_PUSH_TUPLE_DEVICE_MEMORY_DENOMINATOR,
    BETWEENNESS_SOURCE_ORIGIN_WIDTH_BYTES, GRAPH_MEMORY_BOUNDARY, GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES,
    GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES, GraphCalibrationSource, GraphConstructionWorkspaceBounds,
    GraphMemoryAdmissionEvidence, GraphMemoryCalibrationOverride, GraphMemoryClass, GraphMemoryEstimate,
    GraphMemoryOwnerComponents, GraphMemoryRetainedOutputContribution, GraphPairCardinalityAdmissionMode,
    GraphPairCardinalitySource, GraphPlanMemoryEstimate, GraphReservationReport, GraphSeedMode, GraphStatistics,
    GraphVertexEstimate, GraphVertexEstimateSource, TRIANGLE_COUNT_CHUNK_SAFETY_DENOMINATOR,
    TRIANGLE_COUNT_CHUNK_SAFETY_NUMERATOR, TRIANGLE_COUNT_DFG_REBUILD_VERTEX_COLUMNS,
    TRIANGLE_COUNT_NO_STATS_EDGE_VERTEX_COLUMNS, TRIANGLE_COUNT_PAIR_LIST_VERTEX_COLUMNS,
    TRIANGLE_COUNT_PEAK_INTERSECTION_VERTEX_COLUMNS, TRIANGLE_COUNT_SORT_STAGING_VERTEX_COLUMNS,
    TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::{MemoryReservationRequest, MemorySpaceId, ReservationKind},
    metrics::PlanNodeId,
    plan::{
        EdgeNormalizeSpec, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionPolicy,
        GraphEdgeColumnBindings, JoinSide, JoinType, NodeId, PlanNode, QueryPlan, estimated_output_rows,
    },
};

/// Plan/runtime inputs used by the approximate graph memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphMemoryModelInput {
    /// Raw edge rows before any explicit edge-normalize operator, when known.
    pub raw_input_edges: Option<u64>,
    /// Edge-row count after graph edge filtering/normalization.
    pub input_edges: u64,
    /// Width in bytes of the vertex id type.
    pub vertex_width_bytes: u64,
    /// Width in bytes of the optional weight column consumed by the algorithm.
    pub weight_width_bytes: Option<u64>,
    /// Whether the graph has a stable edge-id column.
    pub has_edge_id: bool,
    /// Total bytes reported by the CUDA device. This lets seed-dependent
    /// algorithms mirror cuGraph's physical-memory source batching heuristic.
    pub device_total_memory_bytes: Option<u64>,
    /// Better vertex-count signal when available.
    pub vertex_estimate: Option<GraphVertexEstimate>,
    /// Whether an edge-normalize operator was inserted before graph execution.
    pub edge_normalize_inserted: bool,
    /// Upper bounds for raw libcugraph graph-construction phases, supplied by
    /// the owning cuGraph preflight contract. `None` makes bounded admission
    /// reject rather than invent an allowance.
    pub construction_workspace_bounds: Option<GraphConstructionWorkspaceBounds>,
    /// Runtime topology evidence used by data-dependent pricing, when available.
    pub statistics: Option<GraphStatistics>,
    /// Exact or hard-bounded candidate-pair rows for a similarity operation.
    pub pair_rows: Option<u64>,
    /// Width in bytes of each vertex column in the candidate-pair relation.
    pub pair_vertex_width_bytes: Option<u64>,
    /// Provenance for [`Self::pair_rows`].
    pub pair_cardinality_source: Option<GraphPairCardinalitySource>,
}

impl GraphMemoryModelInput {
    /// Conservative upper bound on vertex count: every edge contributes two
    /// distinct endpoints.
    pub fn vertex_upper_bound(self) -> Result<u64> {
        self.input_edges
            .checked_mul(2)
            .ok_or_else(|| graph_memory_arithmetic_error("vertex_upper_bound", "multiplication"))
    }

    /// Raw edge count when known, otherwise the effective graph-stage count.
    pub fn raw_edge_count(self) -> u64 {
        self.raw_input_edges.unwrap_or(self.input_edges)
    }

    /// Attach raw pre-normalization edge-row evidence.
    pub fn with_raw_input_edges(mut self, raw_input_edges: u64) -> Self {
        self.raw_input_edges = Some(raw_input_edges);
        self
    }

    /// Attach refined vertex-count evidence.
    pub fn with_vertex_estimate(mut self, estimate: GraphVertexEstimate) -> Self {
        self.vertex_estimate = Some(estimate);
        self
    }

    /// Attach runtime graph statistics used by data-dependent pricing.
    pub fn with_statistics(mut self, statistics: GraphStatistics) -> Self {
        self.statistics = Some(statistics);
        self
    }

    /// Attach physical device-memory evidence for algorithms whose cuGraph
    /// implementation chooses resident batch width from total GPU memory.
    pub fn with_device_total_memory_bytes(mut self, total_bytes: u64) -> Self {
        self.device_total_memory_bytes = Some(total_bytes);
        self
    }

    /// Mark the input as having passed through explicit edge normalization.
    pub fn with_edge_normalize_inserted(mut self, inserted: bool) -> Self {
        self.edge_normalize_inserted = inserted;
        self
    }

    /// Attach the owning libcugraph graph-construction phase bounds.
    pub fn with_construction_workspace_bounds(mut self, bounds: GraphConstructionWorkspaceBounds) -> Self {
        self.construction_workspace_bounds = Some(bounds);
        self
    }

    /// Attach explicit-pair cardinality and dtype evidence for a similarity
    /// operation.
    pub fn with_similarity_pair_input(
        mut self,
        pair_rows: u64,
        pair_vertex_width_bytes: u64,
        pair_cardinality_source: GraphPairCardinalitySource,
    ) -> Self {
        self.pair_rows = Some(pair_rows);
        self.pair_vertex_width_bytes = Some(pair_vertex_width_bytes);
        self.pair_cardinality_source = Some(pair_cardinality_source);
        self
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphMemoryModel;

impl GraphMemoryModel {
    /// Estimate the graph fragment memory for an algorithm and graph input.
    pub fn estimate(&self, spec: &GraphAlgorithmSpec, input: GraphMemoryModelInput) -> Result<GraphMemoryEstimate> {
        self.estimate_with_pair_cardinality_admission_mode(
            spec,
            input,
            GraphPairCardinalityAdmissionMode::Unbounded,
            true,
        )
    }

    /// Estimate a graph fragment for bounded admission.
    ///
    /// Similarity requires plan-exact or hard-upper-bound pair cardinality
    /// evidence here.  A count obtained by executing the pair relation is
    /// deliberately rejected, because it is post-admission runtime evidence
    /// and cannot be retroactively promoted to an admission proof.
    pub(crate) fn estimate_for_bounded(
        &self,
        spec: &GraphAlgorithmSpec,
        input: GraphMemoryModelInput,
    ) -> Result<GraphMemoryEstimate> {
        require_bounded_similarity_pair_cardinality_evidence(spec, input)?;
        require_graph_construction_workspace_preflight(spec, input)?;
        self.estimate_with_pair_cardinality_admission_mode(
            spec,
            input,
            GraphPairCardinalityAdmissionMode::Bounded,
            false,
        )
    }

    fn estimate_with_pair_cardinality_admission_mode(
        &self,
        spec: &GraphAlgorithmSpec,
        input: GraphMemoryModelInput,
        pair_cardinality_admission_mode: GraphPairCardinalityAdmissionMode,
        diagnostic_projection: bool,
    ) -> Result<GraphMemoryEstimate> {
        if input.vertex_width_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.estimate",
                "graph memory model requires a positive vertex dtype width",
            )
            .with_fact("vertex_width_bytes", input.vertex_width_bytes));
        }

        let class = graph_memory_class(spec.algorithm);
        let (pair_rows, pair_vertex_width_bytes, pair_cardinality_source) =
            similarity_pair_memory_evidence(spec, input)?;
        let endpoint_bound = model_product(input.input_edges, 2, "vertex_upper_bound")?;
        let _ = project_model_bytes(endpoint_bound, "vertex_upper_bound", diagnostic_projection)?;
        let (vertex_estimate_model, vertex_estimate_source) = input
            .vertex_estimate
            .map(|estimate| (ModelBytes::from_u64(estimate.count).min_model(endpoint_bound), estimate.source))
            .unwrap_or((endpoint_bound, GraphVertexEstimateSource::EdgeEndpointUpperBound));
        let vertex_estimate = project_model_bytes(vertex_estimate_model, "vertex_estimate", diagnostic_projection)?;
        let derived = derive_graph_terms(spec, input, vertex_estimate, pair_rows, pair_vertex_width_bytes)?;
        let execution_components = derived.execution_workspace.add(derived.output, "execution_output_workspace")?;
        if execution_components != derived.execution {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.arithmetic",
                "graph execution owner components do not match the inclusive execution estimate",
            )
            .with_fact("reason", "graph_memory_owner_component_mismatch"));
        }
        let construction_components =
            derived.graph_persistent.add(derived.construction_transient, "construction_persistent_transient")?;
        if construction_components != derived.construction {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.arithmetic",
                "graph construction owner components do not match the inclusive construction estimate",
            )
            .with_fact("reason", "graph_memory_owner_component_mismatch"));
        }
        let owner_components = GraphMemoryOwnerComponents::from_derived(derived, diagnostic_projection)?;
        let output_row_count_upper_bound = derived.output_rows;
        let retained_output_bytes = GraphMemoryRetainedOutputContribution::new(
            project_model_bytes(derived.output, "output_bytes", diagnostic_projection)?,
            output_row_count_upper_bound,
        );
        let seed_input = graph_seed_memory_input(&spec.params, vertex_estimate);
        let seed_count = seed_input.map(|seed_input| seed_input.requested_count);
        let seed_mode = seed_input.map(|seed_input| seed_input.mode);
        let calibration_source = graph_calibration_source(input);

        Ok(GraphMemoryEstimate {
            algorithm: spec.algorithm,
            class,
            raw_input_edges: input.raw_input_edges,
            input_edges: input.input_edges,
            vertex_upper_bound: project_model_bytes(endpoint_bound, "vertex_upper_bound", diagnostic_projection)?,
            vertex_estimate,
            vertex_estimate_source,
            vertex_width_bytes: input.vertex_width_bytes,
            weight_width_bytes: input.weight_width_bytes,
            pair_rows,
            pair_vertex_width_bytes,
            pair_cardinality_source,
            pair_cardinality_admission_mode: spec.algorithm.is_similarity().then_some(pair_cardinality_admission_mode),
            has_edge_id: input.has_edge_id,
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            construction_policy: spec.graph_options.policy,
            edge_normalize_inserted: input.edge_normalize_inserted,
            calibration_source,
            seed_count,
            seed_mode,
            data_dependent: class.is_data_dependent(),
            construction_bytes: project_model_bytes(derived.construction, "construction_bytes", diagnostic_projection)?,
            execution_bytes: project_model_bytes(derived.execution, "execution_bytes", diagnostic_projection)?,
            retained_output_bytes,
            requested_bytes: project_model_bytes(derived.requested, "requested_bytes", diagnostic_projection)?,
            owner_components,
        })
    }

    /// Build the reservation request used by graph-tier admission.
    pub fn reservation_request(
        &self,
        node_id: impl ToString,
        report: &GraphReservationReport,
        memory_space: MemorySpaceId,
    ) -> Option<MemoryReservationRequest> {
        let budget_bytes = report.budget_bytes?;
        Some(
            MemoryReservationRequest::new(ReservationKind::GraphAlgorithm, report.requested_bytes, budget_bytes)
                .with_memory_space(memory_space)
                .with_memory_boundary(GRAPH_MEMORY_BOUNDARY)
                .with_node_id(PlanNodeId::new(node_id.to_string()))
                .with_stage("graph_memory_admission"),
        )
    }
}

/// Return the memory class for a graph algorithm.
pub const fn graph_memory_class(algorithm: GraphAlgorithm) -> GraphMemoryClass {
    match algorithm {
        GraphAlgorithm::PageRank
        | GraphAlgorithm::PersonalizedPageRank
        | GraphAlgorithm::EigenvectorCentrality
        | GraphAlgorithm::KatzCentrality
        | GraphAlgorithm::Hits
        | GraphAlgorithm::Bfs
        | GraphAlgorithm::Sssp
        | GraphAlgorithm::DegreesAll
        | GraphAlgorithm::InDegreesAll
        | GraphAlgorithm::OutDegreesAll
        | GraphAlgorithm::WeaklyConnectedComponents
        | GraphAlgorithm::CoreNumber
        | GraphAlgorithm::MinimumSpanningTree => GraphMemoryClass::OutputSized,
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            GraphMemoryClass::PairOutputSized
        },
        GraphAlgorithm::Louvain
        | GraphAlgorithm::Leiden
        | GraphAlgorithm::Ecg
        | GraphAlgorithm::KCore
        | GraphAlgorithm::StronglyConnectedComponents => GraphMemoryClass::MultiArray,
        GraphAlgorithm::TriangleCountAll
        | GraphAlgorithm::SpectralModularityMaximization
        | GraphAlgorithm::ForceAtlas2 => GraphMemoryClass::WorkspaceHeavyDataDependent,
        GraphAlgorithm::BetweennessCentrality | GraphAlgorithm::EdgeBetweennessCentrality => {
            GraphMemoryClass::WorkspaceHeavySeedDependent
        },
    }
}

/// Whether bounded graph execution must reserve an algorithm workspace tier.
///
/// Algorithms stay on the graph-workspace path until their cuGraph execution is
/// audited as table/view-only over already-accounted resident inputs.
pub const fn graph_memory_reservation_required(algorithm: GraphAlgorithm) -> bool {
    match algorithm {
        GraphAlgorithm::Bfs => false,
        GraphAlgorithm::PageRank
        | GraphAlgorithm::PersonalizedPageRank
        | GraphAlgorithm::Louvain
        | GraphAlgorithm::EigenvectorCentrality
        | GraphAlgorithm::KatzCentrality
        | GraphAlgorithm::Hits
        | GraphAlgorithm::BetweennessCentrality
        | GraphAlgorithm::EdgeBetweennessCentrality
        | GraphAlgorithm::Sssp
        | GraphAlgorithm::DegreesAll
        | GraphAlgorithm::InDegreesAll
        | GraphAlgorithm::OutDegreesAll
        | GraphAlgorithm::WeaklyConnectedComponents
        | GraphAlgorithm::StronglyConnectedComponents
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
        | GraphAlgorithm::Cosine => true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphSeedMemoryInput {
    requested_count: u64,
    mode: GraphSeedMode,
    explicit_seed_list: bool,
    requires_runtime_seed_selection: bool,
}

fn graph_seed_memory_input(params: &GraphAlgorithmParams, vertex_estimate: u64) -> Option<GraphSeedMemoryInput> {
    match params {
        GraphAlgorithmParams::BetweennessCentrality(params) => {
            Some(betweenness_seed_memory_input(params.seeds.as_ref(), params.k, vertex_estimate))
        },
        GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            Some(betweenness_seed_memory_input(params.seeds.as_ref(), params.k, vertex_estimate))
        },
        _ => None,
    }
}

fn betweenness_seed_memory_input(
    seeds: Option<&crate::plan::GraphBetweennessSeeds>,
    k: Option<u64>,
    vertex_estimate: u64,
) -> GraphSeedMemoryInput {
    match seeds {
        Some(seeds) => GraphSeedMemoryInput {
            requested_count: seeds.row_count_upper_bound(),
            mode: GraphSeedMode::ExplicitSeedList,
            explicit_seed_list: true,
            requires_runtime_seed_selection: false,
        },
        None => GraphSeedMemoryInput {
            requested_count: k.unwrap_or(vertex_estimate),
            mode: if k.is_some() { GraphSeedMode::RuntimeKSelection } else { GraphSeedMode::AllVertices },
            explicit_seed_list: k.is_some(),
            requires_runtime_seed_selection: k.is_some(),
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelBytes(pub(crate) u128);

impl ModelBytes {
    fn from_u64(value: u64) -> Self {
        Self(u128::from(value))
    }

    fn add(self, other: Self, term: &'static str) -> Result<Self> {
        self.0.checked_add(other.0).map(Self).ok_or_else(|| graph_memory_arithmetic_error(term, "addition"))
    }

    fn mul(self, other: Self, term: &'static str) -> Result<Self> {
        self.0.checked_mul(other.0).map(Self).ok_or_else(|| graph_memory_arithmetic_error(term, "multiplication"))
    }

    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    fn min_model(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    pub(crate) fn diagnostic_u64(self) -> u64 {
        self.0.min(u128::from(u64::MAX)) as u64
    }

    pub(crate) fn bounded_u64(self, term: &'static str) -> Result<u64> {
        if self.0 >= u128::from(u64::MAX) {
            return Err(graph_memory_arithmetic_error(term, "saturated"));
        }
        u64::try_from(self.0).map_err(|_| graph_memory_arithmetic_error(term, "u128_to_u64"))
    }
}

pub(super) fn graph_memory_arithmetic_error(term: &'static str, operation: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.arithmetic",
        "graph memory bounded arithmetic is not finite",
    )
    .with_fact("reason", "graph_memory_arithmetic_overflow")
    .with_fact("term", term)
    .with_fact("operation", operation)
}

pub(super) fn project_model_bytes(value: ModelBytes, term: &'static str, diagnostic: bool) -> Result<u64> {
    if diagnostic { Ok(value.diagnostic_u64()) } else { value.bounded_u64(term) }
}

pub(super) fn graph_live_reservation_floor_top_up_bytes(cold_model_requested_bytes: u64) -> Result<Option<u64>> {
    if cold_model_requested_bytes == u64::MAX {
        return Err(graph_memory_arithmetic_error("graph_live_reservation_floor_top_up", "saturated"));
    }
    if cold_model_requested_bytes >= GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES {
        return Ok(None);
    }
    GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES
        .checked_sub(cold_model_requested_bytes)
        .map(Some)
        .ok_or_else(|| graph_memory_arithmetic_error("graph_live_reservation_floor_top_up", "subtraction"))
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GraphDerivedTerms {
    pub(super) construction: ModelBytes,
    pub(super) graph_persistent: ModelBytes,
    pub(super) construction_transient: ModelBytes,
    pub(super) output: ModelBytes,
    pub(super) execution_workspace: ModelBytes,
    pub(super) execution: ModelBytes,
    pub(super) requested: ModelBytes,
    pub(super) output_rows: u64,
}

fn checked_add_all(values: impl IntoIterator<Item = (ModelBytes, &'static str)>) -> Result<ModelBytes> {
    values.into_iter().try_fold(ModelBytes(0), |total, (value, term)| total.add(value, term))
}

fn model_product(left: u64, right: u64, term: &'static str) -> Result<ModelBytes> {
    ModelBytes::from_u64(left).mul(ModelBytes::from_u64(right), term)
}

fn construction_multiplier(policy: GraphConstructionPolicy, directed: bool) -> u64 {
    match (policy, directed) {
        (GraphConstructionPolicy::PythonCugraph, false) => 3,
        (GraphConstructionPolicy::PythonCugraph, true) => 2,
        (GraphConstructionPolicy::RawLibcugraph, _) => 1,
    }
}

fn derive_graph_terms(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    pair_rows: Option<u64>,
    pair_vertex_width_bytes: Option<u64>,
) -> Result<GraphDerivedTerms> {
    let edge_row_width = semantic_edge_row_width_model(input)?;
    let normalized_edge = ModelBytes::from_u64(input.input_edges).mul(edge_row_width, "normalized_edge_bytes")?;
    let construction_base = normalized_edge.mul(
        ModelBytes::from_u64(construction_multiplier(spec.graph_options.policy, spec.graph_options.directed)),
        "construction_base",
    )?;
    let normalization = if input.edge_normalize_inserted { normalized_edge } else { ModelBytes(0) };
    let retained_normalization_source = normalization;
    let live_graph_inputs = normalization.add(retained_normalization_source, "live_graph_inputs")?;
    let construction_transient = if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph {
        let unnormalized_input_copy = if input.edge_normalize_inserted { ModelBytes(0) } else { normalized_edge };
        let renumber_workspace = ModelBytes::from_u64(graph_construction_renumber_workspace_bound_bytes(spec, input));
        let compression_workspace =
            ModelBytes::from_u64(graph_construction_compression_workspace_bound_bytes(spec, input));
        let property_output_staging = raw_libcugraph_property_output_staging(input)?;
        let renumber_phase = if spec.graph_options.renumber {
            checked_add_all([
                (live_graph_inputs, "renumber_live_graph_inputs"),
                (unnormalized_input_copy, "renumber_unnormalized_input_copy"),
                (renumber_workspace, "renumber_workspace"),
            ])?
        } else {
            live_graph_inputs.add(unnormalized_input_copy, "construction_live_graph_inputs")?
        };
        let retained_number_map = if spec.graph_options.renumber {
            model_product(vertex_estimate, input.vertex_width_bytes, "retained_number_map")?
        } else {
            ModelBytes(0)
        };
        let compression_phase = checked_add_all([
            (live_graph_inputs, "compression_live_graph_inputs"),
            (unnormalized_input_copy, "compression_unnormalized_input_copy"),
            (retained_number_map, "retained_number_map"),
            (property_output_staging, "property_output_staging"),
            (compression_workspace, "compression_workspace"),
        ])?;
        renumber_phase.max(compression_phase)
    } else {
        let renumber_sort_staging = model_product(
            input.input_edges,
            input
                .vertex_width_bytes
                .checked_mul(2)
                .ok_or_else(|| graph_memory_arithmetic_error("construction_sort_staging_width", "multiplication"))?,
            "renumber_sort_staging",
        )?;
        let renumber_workspace = if spec.graph_options.renumber {
            model_product(
                vertex_estimate,
                input
                    .vertex_width_bytes
                    .checked_mul(3)
                    .ok_or_else(|| graph_memory_arithmetic_error("renumber_staging_width", "multiplication"))?,
                "renumber_staging",
            )?
        } else {
            ModelBytes(0)
        };
        checked_add_all([
            (live_graph_inputs, "live_graph_inputs"),
            (renumber_sort_staging, "renumber_sort_staging"),
            (renumber_workspace, "renumber_staging"),
        ])?
    };
    let construction = construction_base.add(construction_transient, "construction")?;
    let output = output_bytes_model(spec, input, vertex_estimate, pair_rows)?;
    let seed_input = graph_seed_memory_input(&spec.params, vertex_estimate);
    let mut workspace = workspace_bytes_model(
        spec.algorithm,
        &spec.params,
        graph_memory_class(spec.algorithm),
        input,
        vertex_estimate,
        seed_input,
        pair_rows,
        pair_vertex_width_bytes,
    )?;
    if let GraphAlgorithmParams::Bfs(params) = &spec.params
        && let Some(path_output) = params.path_output
    {
        workspace = workspace.add(
            bfs_path_extraction_peak_model(input, vertex_estimate, path_output.max_output_rows)?,
            "bfs_path_extraction_peak",
        )?;
    }
    let execution = output.add(workspace, "execution_output_workspace")?;
    let requested = match spec.algorithm {
        GraphAlgorithm::ForceAtlas2 => {
            force_atlas2_phase_peak_model(input, vertex_estimate, &spec.params, construction, output)?
        },
        GraphAlgorithm::PageRank => construction
            .max(
                live_graph_inputs
                    .add(construction_base, "persistent_graph_inputs")?
                    .add(execution, "persistent_graph_execution")?,
            )
            .max(ModelBytes(1)),
        GraphAlgorithm::Bfs if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph => construction
            .max(
                live_graph_inputs
                    .add(construction_base, "persistent_graph_inputs")?
                    .add(execution, "persistent_graph_execution")?,
            )
            .max(ModelBytes(1)),
        _ => construction.add(execution, "requested_construction_execution")?.max(ModelBytes(1)),
    };
    Ok(GraphDerivedTerms {
        construction,
        graph_persistent: construction_base,
        construction_transient,
        output,
        execution_workspace: workspace,
        execution,
        requested,
        output_rows: output_row_count_upper_bound_for_algorithm(spec, input, vertex_estimate, pair_rows),
    })
}

fn semantic_edge_row_width_model(input: GraphMemoryModelInput) -> Result<ModelBytes> {
    checked_add_all([
        (
            ModelBytes::from_u64(input.vertex_width_bytes).mul(ModelBytes::from_u64(2), "edge_endpoint_width")?,
            "edge_endpoint_width",
        ),
        (ModelBytes::from_u64(input.weight_width_bytes.unwrap_or(0)), "edge_weight_width"),
        (
            if input.has_edge_id { ModelBytes::from_u64(input.vertex_width_bytes) } else { ModelBytes(0) },
            "edge_id_width",
        ),
    ])
}

pub(crate) fn raw_libcugraph_property_output_staging(input: GraphMemoryModelInput) -> Result<ModelBytes> {
    let property_width = match (input.weight_width_bytes, input.has_edge_id) {
        (None, false) => 0,
        (Some(width), false) => width,
        (None, true) => input.vertex_width_bytes,
        (Some(_), true) => input.vertex_width_bytes,
    };
    model_product(input.input_edges, property_width, "property_output_staging")
}

fn output_bytes_model(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    pair_rows: Option<u64>,
) -> Result<ModelBytes> {
    if let GraphAlgorithmParams::Bfs(params) = &spec.params
        && let Some(path_output) = params.path_output
    {
        return model_product(path_output.max_output_rows, 40, "bfs_path_output_bytes");
    }
    let (rows, width) = match spec.algorithm {
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            (pair_rows.unwrap_or(0), 24)
        },
        GraphAlgorithm::EdgeBetweennessCentrality => {
            (input.input_edges, edge_betweenness_output_row_width_model(input)?)
        },
        GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => (input.input_edges, 16),
        GraphAlgorithm::Hits | GraphAlgorithm::DegreesAll | GraphAlgorithm::Bfs | GraphAlgorithm::Sssp => {
            (vertex_estimate, 24)
        },
        GraphAlgorithm::ForceAtlas2 => (vertex_estimate, 16),
        _ => (vertex_estimate, 16),
    };
    model_product(rows, width, "output_bytes")
}

#[cfg(feature = "cugraph")]
fn bfs_path_extraction_peak_model(
    input: GraphMemoryModelInput,
    predecessor_rows: u64,
    max_output_rows: u64,
) -> Result<ModelBytes> {
    let vertex_type = match input.vertex_width_bytes {
        4 => cugraph::types::DataType::Int32,
        8 => cugraph::types::DataType::Int64,
        width => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_memory.bfs_path_preflight",
                "BFS path extraction requires Int32 or Int64 vertices",
            )
            .with_fact("vertex_width_bytes", width));
        },
    };
    let predecessor_rows = usize::try_from(predecessor_rows)
        .map_err(|_| graph_memory_arithmetic_error("bfs_path_predecessor_rows", "conversion"))?;
    let max_output_rows = usize::try_from(max_output_rows)
        .map_err(|_| graph_memory_arithmetic_error("bfs_path_output_rows", "conversion"))?;
    let preflight = cugraph::traversal::extract_paths_preflight(
        &cugraph::traversal::ExtractPathsPreflightInput::new(vertex_type, 1, predecessor_rows, 1, max_output_rows)
            .map_err(|error| {
                QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "nexus_query_engine.graph_memory.bfs_path_preflight",
                    "native BFS path allocation preflight rejected the bounded input",
                )
                .with_fact("reason", "bfs_path_owner_preflight_unavailable")
                .with_source(error)
            })?,
    )
    .map_err(|error| {
        QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.graph_memory.bfs_path_preflight",
            "native BFS path allocation preflight failed",
        )
        .with_fact("reason", "bfs_path_owner_preflight_unavailable")
        .with_source(error)
    })?;
    Ok(ModelBytes::from_u64(preflight.peak_bytes()))
}

#[cfg(not(feature = "cugraph"))]
fn bfs_path_extraction_peak_model(
    _input: GraphMemoryModelInput,
    _predecessor_rows: u64,
    _max_output_rows: u64,
) -> Result<ModelBytes> {
    Err(QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.graph_memory.bfs_path_preflight",
        "native BFS path allocation preflight is unavailable without cuGraph",
    )
    .with_fact("reason", "bfs_path_owner_preflight_unavailable"))
}

fn edge_betweenness_output_row_width_model(input: GraphMemoryModelInput) -> Result<u64> {
    let vertex = input.vertex_width_bytes.max(8);
    vertex
        .checked_mul(2)
        .and_then(|value| value.checked_add(8))
        .and_then(|value| if input.has_edge_id { value.checked_add(vertex) } else { Some(value) })
        .ok_or_else(|| graph_memory_arithmetic_error("edge_betweenness_output_row_width", "addition"))
}

fn workspace_bytes_model(
    algorithm: GraphAlgorithm,
    params: &GraphAlgorithmParams,
    class: GraphMemoryClass,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
    pair_rows: Option<u64>,
    pair_vertex_width_bytes: Option<u64>,
) -> Result<ModelBytes> {
    let vertex_array = model_product(vertex_estimate, input.vertex_width_bytes.max(8), "workspace_vertex_array")?;
    let edge_index = model_product(
        input.input_edges,
        input
            .vertex_width_bytes
            .checked_mul(3)
            .ok_or_else(|| graph_memory_arithmetic_error("workspace_edge_index_width", "multiplication"))?,
        "workspace_edge_index",
    )?;
    let output_sized = vertex_array
        .mul(ModelBytes::from_u64(3), "workspace_output_arrays")?
        .add(edge_index, "workspace_output_sized")?;
    match class {
        GraphMemoryClass::OutputSized => Ok(output_sized),
        GraphMemoryClass::MultiArray => output_sized.mul(ModelBytes::from_u64(4), "multi_array_workspace"),
        GraphMemoryClass::WorkspaceHeavyDataDependent => match algorithm {
            GraphAlgorithm::TriangleCountAll => triangle_count_workspace_model(input, vertex_estimate),
            GraphAlgorithm::ForceAtlas2 => force_atlas2_workspace_model(input, vertex_estimate, params, output_sized),
            GraphAlgorithm::SpectralModularityMaximization => {
                legacy_workspace_heavy_model(vertex_estimate, output_sized)
            },
            _ => Ok(output_sized),
        },
        GraphMemoryClass::WorkspaceHeavySeedDependent => match algorithm {
            GraphAlgorithm::BetweennessCentrality => output_sized.add(
                vertex_betweenness_workspace_model(input, vertex_estimate, seed_input)?,
                "vertex_betweenness_workspace",
            ),
            GraphAlgorithm::EdgeBetweennessCentrality => output_sized.add(
                edge_betweenness_workspace_model(input, vertex_estimate, seed_input)?,
                "edge_betweenness_workspace",
            ),
            _ => Ok(output_sized),
        },
        GraphMemoryClass::PairOutputSized => similarity_workspace_model(
            pair_rows.expect("similarity memory evidence validated before workspace pricing"),
            pair_vertex_width_bytes.expect("similarity memory evidence validated before workspace pricing"),
            input.weight_width_bytes.is_some(),
        ),
    }
}

fn legacy_workspace_heavy_model(vertex_estimate: u64, output_sized: ModelBytes) -> Result<ModelBytes> {
    let probe = model_product(vertex_estimate, vertex_estimate.min(1024), "spectral_probe")?
        .mul(ModelBytes::from_u64(4), "spectral_probe_scale")?;
    output_sized.mul(ModelBytes::from_u64(8), "spectral_output_workspace")?.add(probe, "spectral_workspace")
}

fn similarity_workspace_model(rows: u64, width: u64, weighted: bool) -> Result<ModelBytes> {
    let pair_width =
        width.checked_mul(2).ok_or_else(|| graph_memory_arithmetic_error("similarity_pair_width", "multiplication"))?;
    let pair_columns = model_product(rows, pair_width, "similarity_pair_columns")?;
    let scores = model_product(rows, SIMILARITY_NATIVE_SCORE_WIDTH_BYTES, "similarity_scores")?;
    let kernel = ModelBytes::from_u64(SIMILARITY_WORKSPACE_FIXED_BYTES).add(
        model_product(rows, similarity_workspace_per_pair_bytes(width, weighted), "similarity_kernel")?,
        "similarity_kernel_total",
    )?;
    checked_add_all([
        (pair_columns, "similarity_pair_input"),
        (pair_columns, "similarity_native_pairs"),
        (scores, "similarity_scores"),
        (kernel, "similarity_kernel"),
    ])
}

fn force_atlas2_workspace_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: ModelBytes,
) -> Result<ModelBytes> {
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, output_sized)?;
    checked_add_all([
        (phase.persistent_csr_and_decompressed_coo_bytes, "force_atlas2_persistent_graph"),
        (phase.barnes_hut_workspace_bytes, "force_atlas2_barnes_hut"),
        (phase.legacy_coo_sort_workspace_bytes, "force_atlas2_coo_sort"),
    ])
}

#[derive(Debug, Clone, Copy)]
struct ForceAtlas2WorkspacePhaseModel {
    persistent_csr_and_decompressed_coo_bytes: ModelBytes,
    persistent_number_map_and_positions_bytes: ModelBytes,
    legacy_coo_sort_workspace_bytes: ModelBytes,
    barnes_hut_workspace_bytes: ModelBytes,
}

fn force_atlas2_workspace_phase_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: ModelBytes,
) -> Result<ForceAtlas2WorkspacePhaseModel> {
    let barnes_hut = match params {
        GraphAlgorithmParams::ForceAtlas2(params) if params.barnes_hut_optimize => checked_add_all([
            (model_product(vertex_estimate, 120, "force_atlas2_barnes_hut")?, "force_atlas2_barnes_hut"),
            (
                model_product(
                    vertex_estimate,
                    input
                        .vertex_width_bytes
                        .checked_mul(2)
                        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_degree_width", "multiplication"))?,
                    "force_atlas2_degree",
                )?,
                "force_atlas2_degree",
            ),
        ])?,
        GraphAlgorithmParams::ForceAtlas2(_) => model_product(vertex_estimate, 44, "force_atlas2_exact")?,
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.arithmetic",
                "ForceAtlas2 memory model received incompatible parameters",
            ));
        },
    };
    let position_width = input
        .vertex_width_bytes
        .checked_add(8)
        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_position_width", "addition"))?;
    let coo_record_width = input
        .vertex_width_bytes
        .checked_mul(2)
        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_coo_width", "multiplication"))?
        .checked_add(input.weight_width_bytes.unwrap_or(4))
        .and_then(|width| width.checked_add(1))
        .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_coo_record", "addition"))?;
    Ok(ForceAtlas2WorkspacePhaseModel {
        persistent_csr_and_decompressed_coo_bytes: output_sized,
        persistent_number_map_and_positions_bytes: model_product(
            vertex_estimate,
            position_width,
            "force_atlas2_positions",
        )?,
        legacy_coo_sort_workspace_bytes: model_product(input.input_edges, coo_record_width, "force_atlas2_coo_sort")?,
        barnes_hut_workspace_bytes: barnes_hut,
    })
}

fn force_atlas2_phase_peak_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    construction: ModelBytes,
    result_conversion: ModelBytes,
) -> Result<ModelBytes> {
    let vertex_array = model_product(vertex_estimate, input.vertex_width_bytes.max(8), "force_atlas2_vertex_array")?;
    let edge_index = model_product(
        input.input_edges,
        input
            .vertex_width_bytes
            .checked_mul(3)
            .ok_or_else(|| graph_memory_arithmetic_error("force_atlas2_edge_index_width", "multiplication"))?,
        "force_atlas2_edge_index",
    )?;
    let output_sized = vertex_array
        .mul(ModelBytes::from_u64(3), "force_atlas2_output_arrays")?
        .add(edge_index, "force_atlas2_output_sized")?;
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, output_sized)?;
    let property_payload =
        model_product(input.input_edges, input.weight_width_bytes.unwrap_or(4), "force_atlas2_property_payload")?;
    let base = checked_add_all([
        (construction, "force_atlas2_construction"),
        (property_payload, "force_atlas2_property_payload"),
        (phase.persistent_csr_and_decompressed_coo_bytes, "force_atlas2_persistent_graph"),
        (phase.persistent_number_map_and_positions_bytes, "force_atlas2_positions"),
    ])?;
    let sort_peak = base.add(phase.legacy_coo_sort_workspace_bytes, "force_atlas2_sort_peak")?;
    let barnes_peak = base.add(phase.barnes_hut_workspace_bytes, "force_atlas2_barnes_peak")?;
    let result_peak = base.add(result_conversion, "force_atlas2_result_peak")?;
    Ok(sort_peak.max(barnes_peak).max(result_peak).max(ModelBytes(1)))
}

#[cfg(test)]
pub(crate) fn force_atlas2_workspace_bytes(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: u64,
) -> u64 {
    force_atlas2_workspace_model(input, vertex_estimate, params, ModelBytes::from_u64(output_sized))
        .expect("test ForceAtlas2 input is finite")
        .diagnostic_u64()
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ForceAtlas2WorkspacePhaseBytes {
    pub(crate) persistent_csr_and_decompressed_coo_bytes: u64,
    pub(crate) persistent_number_map_and_positions_bytes: u64,
    pub(crate) legacy_coo_sort_workspace_bytes: u64,
    pub(crate) barnes_hut_workspace_bytes: u64,
}

#[cfg(test)]
pub(crate) fn force_atlas2_workspace_phase_bytes(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    output_sized: u64,
) -> ForceAtlas2WorkspacePhaseBytes {
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, ModelBytes::from_u64(output_sized))
        .expect("test ForceAtlas2 input is finite");
    ForceAtlas2WorkspacePhaseBytes {
        persistent_csr_and_decompressed_coo_bytes: phase.persistent_csr_and_decompressed_coo_bytes.diagnostic_u64(),
        persistent_number_map_and_positions_bytes: phase.persistent_number_map_and_positions_bytes.diagnostic_u64(),
        legacy_coo_sort_workspace_bytes: phase.legacy_coo_sort_workspace_bytes.diagnostic_u64(),
        barnes_hut_workspace_bytes: phase.barnes_hut_workspace_bytes.diagnostic_u64(),
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ForceAtlas2PhasePeaks {
    pub(crate) persistent_base_bytes: u64,
    pub(crate) legacy_coo_sort_before_degree_bytes: u64,
    pub(crate) barnes_hut_with_degree_bytes: u64,
    pub(crate) post_barnes_result_conversion_bytes: u64,
}

#[cfg(test)]
impl ForceAtlas2PhasePeaks {
    pub(crate) fn peak_bytes(self) -> u64 {
        self.legacy_coo_sort_before_degree_bytes
            .max(self.barnes_hut_with_degree_bytes)
            .max(self.post_barnes_result_conversion_bytes)
            .max(1)
    }
}

#[cfg(test)]
pub(crate) fn force_atlas2_phase_peaks(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    params: &GraphAlgorithmParams,
    construction_bytes: u64,
    result_conversion_bytes: u64,
) -> ForceAtlas2PhasePeaks {
    let vertex_array =
        model_product(vertex_estimate, input.vertex_width_bytes.max(8), "test_force_atlas2_vertex_array")
            .expect("test ForceAtlas2 input is finite");
    let edge_index = model_product(
        input.input_edges,
        input.vertex_width_bytes.checked_mul(3).expect("test ForceAtlas2 width is finite"),
        "test_force_atlas2_edge_index",
    )
    .expect("test ForceAtlas2 input is finite");
    let output_sized = vertex_array
        .mul(ModelBytes::from_u64(3), "test_force_atlas2_output_arrays")
        .and_then(|value| value.add(edge_index, "test_force_atlas2_output_sized"))
        .expect("test ForceAtlas2 input is finite");
    let phase = force_atlas2_workspace_phase_model(input, vertex_estimate, params, output_sized)
        .expect("test ForceAtlas2 input is finite");
    let base = checked_add_all([
        (ModelBytes::from_u64(construction_bytes), "test_force_atlas2_construction"),
        (
            model_product(input.input_edges, input.weight_width_bytes.unwrap_or(4), "test_force_atlas2_property")
                .expect("test ForceAtlas2 input is finite"),
            "test_force_atlas2_property",
        ),
        (phase.persistent_csr_and_decompressed_coo_bytes, "test_force_atlas2_graph"),
        (phase.persistent_number_map_and_positions_bytes, "test_force_atlas2_positions"),
    ])
    .expect("test ForceAtlas2 input is finite");
    ForceAtlas2PhasePeaks {
        persistent_base_bytes: base.diagnostic_u64(),
        legacy_coo_sort_before_degree_bytes: base
            .add(phase.legacy_coo_sort_workspace_bytes, "test_force_atlas2_sort")
            .expect("test ForceAtlas2 input is finite")
            .diagnostic_u64(),
        barnes_hut_with_degree_bytes: base
            .add(phase.barnes_hut_workspace_bytes, "test_force_atlas2_barnes")
            .expect("test ForceAtlas2 input is finite")
            .diagnostic_u64(),
        post_barnes_result_conversion_bytes: base
            .add(ModelBytes::from_u64(result_conversion_bytes), "test_force_atlas2_result")
            .expect("test ForceAtlas2 input is finite")
            .diagnostic_u64(),
    }
}

fn triangle_count_workspace_model(input: GraphMemoryModelInput, vertex_estimate: u64) -> Result<ModelBytes> {
    let width = input.vertex_width_bytes.max(1);
    if let Some(statistics) = input.statistics
        && let (Some(oriented_edges), Some(peak_min_degree)) =
            (statistics.oriented_edge_count, statistics.peak_chunk_min_degree_sum)
    {
        return checked_add_all([
            (
                model_product(oriented_edges, width, "triangle_dfg_edges")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_DFG_REBUILD_VERTEX_COLUMNS), "triangle_dfg_columns")?,
                "triangle_dfg",
            ),
            (
                model_product(oriented_edges, width, "triangle_pair_edges")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_PAIR_LIST_VERTEX_COLUMNS), "triangle_pair_columns")?,
                "triangle_pairs",
            ),
            (
                model_product(peak_min_degree, width, "triangle_intersection")?.mul(
                    ModelBytes::from_u64(TRIANGLE_COUNT_PEAK_INTERSECTION_VERTEX_COLUMNS),
                    "triangle_intersection_buffers",
                )?,
                "triangle_intersection",
            ),
            (
                model_product(statistics.exact_vertices, width.max(8), "triangle_vertex_scratch")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS), "triangle_vertex_columns")?,
                "triangle_vertex_scratch",
            ),
            (
                model_product(input.input_edges, width, "triangle_sort")?
                    .mul(ModelBytes::from_u64(TRIANGLE_COUNT_SORT_STAGING_VERTEX_COLUMNS), "triangle_sort_columns")?,
                "triangle_sort",
            ),
        ]);
    }
    checked_add_all([
        (
            model_product(input.input_edges, width, "triangle_no_stats_edges")?.mul(
                ModelBytes::from_u64(TRIANGLE_COUNT_NO_STATS_EDGE_VERTEX_COLUMNS),
                "triangle_no_stats_edge_columns",
            )?,
            "triangle_no_stats_edges",
        ),
        (
            model_product(vertex_estimate, width.max(8), "triangle_no_stats_vertices")?
                .mul(ModelBytes::from_u64(TRIANGLE_COUNT_VERTEX_SCRATCH_COLUMNS), "triangle_no_stats_vertex_columns")?,
            "triangle_no_stats_vertices",
        ),
    ])
}

fn vertex_betweenness_workspace_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> Result<ModelBytes> {
    let requested_sources = seed_input.map_or(vertex_estimate, |seed| seed.requested_count);
    let concurrent_sources =
        vertex_betweenness_concurrent_source_count_model(input, vertex_estimate, requested_sources)?;
    checked_add_all([
        (
            model_product(concurrent_sources, vertex_estimate, "betweenness_source_count")?
                .mul(vertex_betweenness_state_width_model(input)?, "betweenness_source_state")?,
            "betweenness_source_state",
        ),
        (betweenness_push_batch_model(input, concurrent_sources)?, "betweenness_push_batch"),
        (explicit_seed_model(input, seed_input)?, "explicit_seed_list"),
        (runtime_seed_model(input, vertex_estimate, seed_input)?, "runtime_seed_selection"),
    ])
}

fn edge_betweenness_workspace_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> Result<ModelBytes> {
    checked_add_all([
        (
            model_product(vertex_estimate, edge_betweenness_state_width_model(input)?, "edge_betweenness_state")?,
            "edge_betweenness_state",
        ),
        (betweenness_push_batch_model(input, 1)?, "betweenness_push_batch"),
        (explicit_seed_model(input, seed_input)?, "explicit_seed_list"),
        (runtime_seed_model(input, vertex_estimate, seed_input)?, "runtime_seed_selection"),
    ])
}

fn betweenness_push_batch_model(input: GraphMemoryModelInput, resident_source_copies: u64) -> Result<ModelBytes> {
    let vertex_width = input.vertex_width_bytes.max(8);
    let tuple_width = vertex_width
        .checked_add(BETWEENNESS_SOURCE_ORIGIN_WIDTH_BYTES)
        .and_then(|value| value.checked_add(vertex_width))
        .ok_or_else(|| graph_memory_arithmetic_error("betweenness_tuple_width", "addition"))?;
    let pushes = input.statistics.and_then(|statistics| statistics.sum_out_degree).map_or_else(
        || {
            input
                .input_edges
                .checked_mul(resident_source_copies.max(1))
                .ok_or_else(|| graph_memory_arithmetic_error("betweenness_pushes", "multiplication"))
        },
        Ok,
    )?;
    let resident_pushes = input.device_total_memory_bytes.map_or(pushes, |total| {
        let cap = total / BETWEENNESS_PUSH_TUPLE_DEVICE_MEMORY_DENOMINATOR / tuple_width;
        pushes.min(cap.max(1))
    });
    model_product(resident_pushes, tuple_width, "betweenness_push_batch")
}

fn explicit_seed_model(input: GraphMemoryModelInput, seed_input: Option<GraphSeedMemoryInput>) -> Result<ModelBytes> {
    match seed_input {
        Some(seed) if seed.explicit_seed_list => {
            model_product(seed.requested_count, input.vertex_width_bytes, "explicit_seed_list")
        },
        _ => Ok(ModelBytes(0)),
    }
}

fn runtime_seed_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    seed_input: Option<GraphSeedMemoryInput>,
) -> Result<ModelBytes> {
    let Some(seed) = seed_input else {
        return Ok(ModelBytes(0));
    };
    if !seed.requires_runtime_seed_selection {
        return Ok(ModelBytes(0));
    }
    let vertex_width = input.vertex_width_bytes.max(8);
    let endpoint_copy = model_product(input.input_edges, vertex_width, "runtime_seed_endpoint")?
        .mul(ModelBytes::from_u64(4), "runtime_seed_endpoint_copies")?;
    let distinct_sort = model_product(vertex_estimate, vertex_width, "runtime_seed_distinct")?
        .mul(ModelBytes::from_u64(3), "runtime_seed_distinct_buffers")?;
    endpoint_copy.add(distinct_sort, "runtime_seed_selection")
}

fn vertex_betweenness_concurrent_source_count_model(
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    requested_sources: u64,
) -> Result<u64> {
    let mut sources = requested_sources.clamp(1, BETWEENNESS_MAX_CONCURRENT_SOURCES);
    let Some(total_memory_bytes) = input.device_total_memory_bytes else {
        return Ok(sources);
    };
    if sources <= 1 || vertex_estimate == 0 {
        return Ok(sources);
    }
    let per_source = model_product(
        vertex_estimate,
        input
            .vertex_width_bytes
            .checked_add(input.vertex_width_bytes)
            .ok_or_else(|| graph_memory_arithmetic_error("betweenness_bfs_result_width", "addition"))?,
        "betweenness_bfs_result_bytes",
    )?;
    if per_source.0 == 0 {
        return Ok(sources);
    }
    let physical_batch_cap = u128::from(total_memory_bytes) / 4 / per_source.0;
    sources = sources.min(u64::try_from(physical_batch_cap).unwrap_or(u64::MAX)).max(1);
    Ok(sources)
}

fn vertex_betweenness_state_width_model(input: GraphMemoryModelInput) -> Result<ModelBytes> {
    checked_add_all([
        (ModelBytes::from_u64(input.vertex_width_bytes.max(8)), "betweenness_vertex_width"),
        (ModelBytes::from_u64(input.vertex_width_bytes.max(8)), "betweenness_edge_width"),
        (ModelBytes::from_u64(input.weight_width_bytes.unwrap_or(8).max(8)), "betweenness_weight_width"),
        (model_product(input.vertex_width_bytes.max(8), 2, "betweenness_vertex_state")?, "betweenness_vertex_state"),
        (ModelBytes::from_u64(4), "betweenness_origin_state"),
    ])
}

fn edge_betweenness_state_width_model(input: GraphMemoryModelInput) -> Result<u64> {
    input
        .vertex_width_bytes
        .max(8)
        .checked_add(input.vertex_width_bytes.max(8))
        .and_then(|value| value.checked_add(input.weight_width_bytes.unwrap_or(8).max(8)))
        .ok_or_else(|| graph_memory_arithmetic_error("edge_betweenness_state_width", "addition"))
}

/// Returns the model-owned row bound represented by a graph result.
///
/// Every supported graph algorithm emits one result row per vertex/edge/pair
/// (depending on its output contract).  Keeping this cardinality beside the
/// retained byte contribution lets an engine-owned consumer proof derive a
/// downstream output bound without asking an adapter to guess bytes or rows.
fn output_row_count_upper_bound_for_algorithm(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
    vertex_estimate: u64,
    pair_rows: Option<u64>,
) -> u64 {
    if let GraphAlgorithmParams::Bfs(params) = &spec.params
        && let Some(path_output) = params.path_output
    {
        return path_output.max_output_rows;
    }
    match spec.algorithm {
        GraphAlgorithm::Jaccard | GraphAlgorithm::Sorensen | GraphAlgorithm::Overlap | GraphAlgorithm::Cosine => {
            pair_rows.unwrap_or(0)
        },
        GraphAlgorithm::EdgeBetweennessCentrality | GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => {
            input.input_edges
        },
        GraphAlgorithm::Hits | GraphAlgorithm::DegreesAll | GraphAlgorithm::Bfs | GraphAlgorithm::Sssp => {
            vertex_estimate
        },
        _ => vertex_estimate,
    }
}

#[cfg(any(feature = "cugraph", test))]
pub(crate) fn semantic_edge_row_width_bytes(
    vertex_width_bytes: u64,
    weight_width_bytes: Option<u64>,
    has_edge_id: bool,
) -> u64 {
    semantic_edge_row_width_model(GraphMemoryModelInput {
        vertex_width_bytes,
        weight_width_bytes,
        has_edge_id,
        ..graph_memory_model_input(
            0,
            vertex_width_bytes,
            &GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            false,
            None,
        )
    })
    .map(ModelBytes::diagnostic_u64)
    .unwrap_or(u64::MAX)
}

fn graph_construction_renumber_workspace_bound_bytes(spec: &GraphAlgorithmSpec, input: GraphMemoryModelInput) -> u64 {
    if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph {
        input.construction_workspace_bounds.map_or(0, GraphConstructionWorkspaceBounds::renumber_workspace_bytes)
    } else {
        0
    }
}

fn graph_construction_compression_workspace_bound_bytes(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> u64 {
    if spec.graph_options.policy == GraphConstructionPolicy::RawLibcugraph {
        input.construction_workspace_bounds.map_or(0, GraphConstructionWorkspaceBounds::compression_workspace_bytes)
    } else {
        0
    }
}

fn require_graph_construction_workspace_preflight(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Result<()> {
    if spec.graph_options.policy != GraphConstructionPolicy::RawLibcugraph
        || input.construction_workspace_bounds.is_some()
    {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.estimate_for_bounded",
        "raw cuGraph construction lacks pre-execution workspace bounds",
    )
    .with_fact("reason", "graph_construction_workspace_preflight_unavailable")
    .with_fact("algorithm", spec.algorithm.as_str())
    .with_fact("construction_policy", spec.graph_options.policy.as_str())
    .with_fact("edge_normalize_inserted", input.edge_normalize_inserted))
}

/// Fixed cuGraph allocator floor retained during pairwise similarity
/// coefficient dispatch. The linear term is calibrated separately for the
/// selected vertex width and weighted path: weighted coefficients retain
/// additional per-pair temporary state in libcugraph. Input overlap and
/// `VertexPairs` are accounted for in [`similarity_workspace_bytes`] as
/// separate live buffers.
///
/// The 2026-07-25 `cugraph-similarity-calibrate` 100k-pair observations pin
/// these terms for both Int32/Int64 and weighted/unweighted Jaccard across
/// sparse-uniform and hub-skewed graph topologies. The sparse topology is the
/// observed high-water case; the hub case remains admitted without projecting
/// its lower peak into the model. The same libcugraph pairwise workspace is
/// used by all four exposed metrics. See the calibration-band contract test
/// for the captured peaks and commands.
const SIMILARITY_WORKSPACE_FIXED_BYTES: u64 = 120 * 1024;
const SIMILARITY_NATIVE_SCORE_WIDTH_BYTES: u64 = 8;

fn similarity_workspace_per_pair_bytes(pair_vertex_width_bytes: u64, weighted: bool) -> u64 {
    if weighted {
        // The supported vertex domains are exact-width Int32 and Int64. The
        // isolated calibration matrix observes 66 B and 85 B respectively;
        // a single linear term would either under-admit Int64 or add an
        // unexplained broad multiplier to Int32.
        match pair_vertex_width_bytes {
            4 => 66,
            8 => 85,
            _ => unreachable!("similarity validation admits only Int32 or Int64 vertex domains"),
        }
    } else {
        // As above, price the two supported widths from the isolated matrix
        // rather than using a broad common slope: Int32 => 8 B, Int64 => 28 B.
        match pair_vertex_width_bytes {
            4 => 8,
            8 => 28,
            _ => unreachable!("similarity validation admits only Int32 or Int64 vertex domains"),
        }
    }
}

fn similarity_pair_memory_evidence(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Result<(Option<u64>, Option<u64>, Option<GraphPairCardinalitySource>)> {
    if !spec.algorithm.is_similarity() {
        return Ok((None, None, None));
    }
    let (Some(pair_rows), Some(pair_vertex_width_bytes), Some(pair_cardinality_source)) =
        (input.pair_rows, input.pair_vertex_width_bytes, input.pair_cardinality_source)
    else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.similarity",
            "explicit-pair similarity requires pair-cardinality evidence before memory admission",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("reason", "missing_pair_cardinality_evidence")
        .with_fact("pair_rows", input.pair_rows.unwrap_or(0))
        .with_fact("pair_vertex_width_bytes", input.pair_vertex_width_bytes.unwrap_or(0))
        .with_fact(
            "pair_cardinality_source",
            input.pair_cardinality_source.map_or("none", GraphPairCardinalitySource::as_str),
        ));
    };
    if pair_vertex_width_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.similarity",
            "explicit-pair similarity requires a positive pair vertex dtype width",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("reason", "invalid_pair_vertex_width")
        .with_fact("pair_vertex_width_bytes", pair_vertex_width_bytes));
    }
    if pair_vertex_width_bytes != input.vertex_width_bytes {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_memory.similarity",
            "explicit-pair vertex width must match the graph vertex width",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("reason", "pair_graph_vertex_width_mismatch")
        .with_fact("pair_vertex_width_bytes", pair_vertex_width_bytes)
        .with_fact("vertex_width_bytes", input.vertex_width_bytes));
    }
    Ok((Some(pair_rows), Some(pair_vertex_width_bytes), Some(pair_cardinality_source)))
}

fn require_bounded_similarity_pair_cardinality_evidence(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Result<()> {
    if !spec.algorithm.is_similarity() {
        return Ok(());
    }

    let Some(source) = input.pair_cardinality_source else {
        return Err(bounded_similarity_pair_cardinality_error(spec.algorithm, "missing_pair_cardinality_evidence")
            .with_fact("pair_rows", input.pair_rows.unwrap_or(0))
            .with_fact("pair_vertex_width_bytes", input.pair_vertex_width_bytes.unwrap_or(0))
            .with_fact("pair_cardinality_source", "none"));
    };
    if !source.is_pre_admission_bound() {
        return Err(bounded_similarity_pair_cardinality_error(
            spec.algorithm,
            "bounded_pair_cardinality_evidence_not_plan_bound",
        )
        .with_fact("pair_rows", input.pair_rows.unwrap_or(0))
        .with_fact("pair_vertex_width_bytes", input.pair_vertex_width_bytes.unwrap_or(0))
        .with_fact("pair_cardinality_source", source.as_str()));
    }
    Ok(())
}

fn bounded_similarity_pair_cardinality_error(algorithm: GraphAlgorithm, reason: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.bounded_similarity_admission",
        "bounded similarity admission requires plan-exact pair cardinality or a hard pair-row upper bound",
    )
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("admission_mode", GraphPairCardinalityAdmissionMode::Bounded.as_str())
    .with_fact("reason", reason)
}

/// Derive model inputs from runtime-observed graph edge metadata.
pub fn graph_memory_model_input(
    input_edges: u64,
    vertex_width_bytes: u64,
    edge_columns: &GraphEdgeColumnBindings,
    consumes_weight: bool,
    weight_width_bytes: Option<u64>,
) -> GraphMemoryModelInput {
    GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges,
        vertex_width_bytes,
        weight_width_bytes: consumes_weight.then_some(weight_width_bytes).flatten(),
        has_edge_id: edge_columns.edge_id.is_some(),
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
        construction_workspace_bounds: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    }
}

/// Estimate graph memory asks for bounded admission when plan/config
/// facts are known.
///
/// Missing ordinary edge-row evidence leaves a graph node unpriced. In
/// contrast, a similarity node with known edge rows but no plan-exact or hard
/// pair-row bound is an admission failure: materializing the pair source first
/// would create post-admission evidence, not a pre-admission proof.
pub fn graph_memory_estimates_for_plan(
    plan: &QueryPlan,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<Vec<GraphPlanMemoryEstimate>> {
    graph_memory_estimates_for_plan_with_resident_rows(
        plan,
        calibration_override,
        device_total_memory_bytes,
        &HashMap::new(),
    )
}

fn graph_memory_estimates_for_plan_with_resident_rows(
    plan: &QueryPlan,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
    resident_rows: &HashMap<String, u64>,
) -> Result<Vec<GraphPlanMemoryEstimate>> {
    let mut estimates = Vec::new();
    for node in plan.nodes() {
        let PlanNode::GraphAlgorithm(graph) = &node.kind else {
            continue;
        };
        let Some(mut input) = graph_memory_model_input_for_plan_node_with_resident_rows(
            plan,
            graph,
            calibration_override,
            resident_rows,
        )?
        else {
            continue;
        };
        input = calibration_override.apply_to_input(input);
        if let Some(device_total_memory_bytes) = device_total_memory_bytes {
            input = input.with_device_total_memory_bytes(device_total_memory_bytes);
        }
        #[cfg(feature = "cugraph")]
        {
            input = with_graph_construction_workspace_bounds_from_owner(&graph.spec, input);
        }
        let estimate = GraphMemoryModel.estimate_for_bounded(&graph.spec, input)?;
        estimates.push(GraphPlanMemoryEstimate { node_id: node.id, estimate });
    }
    Ok(estimates)
}

/// Derive one opaque graph admission evidence token for an exact lowered graph
/// execution plan. The plan must contain exactly one graph algorithm node;
/// ambiguous multi-graph plans cannot be attached to one capability proof.
pub fn graph_memory_bounded_admission_evidence_for_plan(
    plan: &QueryPlan,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<GraphMemoryAdmissionEvidence> {
    let mut estimates = graph_memory_estimates_for_plan(plan, calibration_override, device_total_memory_bytes)?;
    let Some(estimate) = (estimates.len() == 1).then(|| estimates.pop()).flatten() else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.admission_evidence",
            "bounded graph admission evidence requires exactly one graph node in the execution plan",
        )
        .with_fact("reason", if estimates.is_empty() { "missing_graph_scope" } else { "ambiguous_graph_scope" })
        .with_fact("graph_node_count", estimates.len()));
    };
    GraphMemoryAdmissionEvidence::from_bounded_plan_estimate(plan, estimate.node_id, &estimate.estimate)
}

pub(crate) fn graph_memory_bounded_admission_evidence_for_plan_with_resident_rows(
    plan: &QueryPlan,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
    resident_rows: &HashMap<String, u64>,
) -> Result<GraphMemoryAdmissionEvidence> {
    let mut estimates = graph_memory_estimates_for_plan_with_resident_rows(
        plan,
        calibration_override,
        device_total_memory_bytes,
        resident_rows,
    )?;
    let graph_node_count = estimates.len();
    let Some(estimate) = (graph_node_count == 1).then(|| estimates.pop()).flatten() else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.resident_admission_evidence",
            "resident graph-phase admission evidence requires exactly one graph node",
        )
        .with_fact("reason", if graph_node_count == 0 { "missing_graph_scope" } else { "ambiguous_graph_scope" })
        .with_fact("graph_node_count", graph_node_count));
    };
    GraphMemoryAdmissionEvidence::from_bounded_plan_estimate(plan, estimate.node_id, &estimate.estimate)
}

/// Estimate graph memory for a graph edge list when planning has direct edge-count evidence.
pub fn graph_memory_estimate_for_edge_list(
    spec: &GraphAlgorithmSpec,
    input_schema: &SchemaRef,
    input_edges: u64,
    raw_input_edges: Option<u64>,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<GraphMemoryEstimate> {
    graph_memory_estimate_for_edge_list_with_vertex_estimate(
        spec,
        input_schema,
        input_edges,
        raw_input_edges,
        None,
        calibration_override,
        device_total_memory_bytes,
    )
}

/// Computes graph memory for one edge-list input, optionally carrying a
/// source-rowset-bound graph vertex estimate.
///
/// The estimate is an input to the engine-owned admission model, not an
/// externally supplied calibration. Frontends may provide it only from source
/// metadata bound to the exact edge rowset they lower.
pub fn graph_memory_estimate_for_edge_list_with_vertex_estimate(
    spec: &GraphAlgorithmSpec,
    input_schema: &SchemaRef,
    input_edges: u64,
    raw_input_edges: Option<u64>,
    vertex_estimate: Option<GraphVertexEstimate>,
    calibration_override: GraphMemoryCalibrationOverride,
    device_total_memory_bytes: Option<u64>,
) -> Result<GraphMemoryEstimate> {
    let mut input = graph_memory_model_input_for_edge_list(spec, input_schema, input_edges, raw_input_edges)?;
    if let Some(vertex_estimate) = vertex_estimate {
        input = input.with_vertex_estimate(vertex_estimate);
    }
    if let Some(device_total_memory_bytes) = device_total_memory_bytes {
        input = input.with_device_total_memory_bytes(device_total_memory_bytes);
    }
    GraphMemoryModel.estimate(spec, calibration_override.apply_to_input(input))
}

/// Estimate the row count produced by a native edge-list plan, when the plan
/// carries deterministic planning evidence.
pub fn graph_edge_list_output_rows_for_plan(plan: &QueryPlan) -> Option<u64> {
    estimated_output_rows(plan, plan.root_id())
}

/// Returns a mathematically hard upper bound on the root row count when the
/// native plan can prove one without executing any source.
///
/// This is intentionally narrower than [`graph_edge_list_output_rows_for_plan`]:
/// admission may use it for resident side inputs, while descriptive
/// planning estimates remain unsuitable as an admission proof.
pub fn query_plan_hard_row_upper_bound(plan: &QueryPlan) -> Option<u64> {
    query_plan_node_hard_row_upper_bound(plan, plan.root_id())
}

pub(crate) fn query_plan_node_hard_row_upper_bound(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    hard_row_upper_bound_for_plan_node(plan, node_id).map(|(rows, _)| rows)
}

pub(crate) fn query_plan_node_hard_row_upper_bound_with_resident_rows(
    plan: &QueryPlan,
    node_id: NodeId,
    resident_rows: &HashMap<String, u64>,
) -> Option<u64> {
    hard_row_upper_bound_with_resident_rows(plan, node_id, resident_rows).map(|(rows, _)| rows)
}

/// Returns a source-owned graph vertex upper bound when the entire native plan
/// preserves the edge-file rowset that carried the footer metadata.
///
/// Unlike [`query_plan_hard_row_upper_bound`], this intentionally refuses
/// filters, limits, and other row-changing nodes. A vertex bound is tied to an
/// exact source rowset; forwarding it through a transformed relation would
/// turn a source fact into detached query metadata.
pub fn query_plan_graph_vertex_upper_bound(plan: &QueryPlan) -> Option<u64> {
    graph_vertex_upper_bound_for_plan_node(plan, plan.root_id())
}

#[cfg(test)]
pub(crate) fn graph_memory_model_input_for_plan_node(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
    calibration_override: GraphMemoryCalibrationOverride,
) -> Result<Option<GraphMemoryModelInput>> {
    graph_memory_model_input_for_plan_node_with_resident_rows(plan, graph, calibration_override, &HashMap::new())
}

fn graph_memory_model_input_for_plan_node_with_resident_rows(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
    calibration_override: GraphMemoryCalibrationOverride,
    resident_rows: &HashMap<String, u64>,
) -> Result<Option<GraphMemoryModelInput>> {
    let Some(edge_counts) =
        graph_plan_edge_counts(plan, graph.input, calibration_override.raw_input_edges, resident_rows)?
    else {
        return Ok(None);
    };
    let input_schema =
        plan.node(graph.input).ok_or_else(|| missing_graph_input_error(graph.input))?.output_schema().clone();
    let mut input = graph_memory_model_input_for_edge_list(
        &graph.spec,
        &input_schema,
        edge_counts.input_edges,
        edge_counts.raw_input_edges,
    )?;
    if let Some(vertex_upper_bound) = edge_counts.graph_vertex_upper_bound {
        input = input.with_vertex_estimate(GraphVertexEstimate::graph_metadata(vertex_upper_bound));
    }
    if let Some((pair_rows, pair_cardinality_source)) = bounded_similarity_pair_cardinality_for_plan(plan, graph)? {
        let crate::plan::GraphAlgorithmParams::Similarity(params) = &graph.spec.params else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.plan_estimate",
                "similarity algorithm node carried non-similarity parameters",
            )
            .with_fact("algorithm", graph.spec.algorithm.as_str()));
        };
        let pair_schema = plan
            .node(params.vertex_pairs.input)
            .ok_or_else(|| missing_graph_input_error(params.vertex_pairs.input))?
            .output_schema();
        let first_width = graph_vertex_field_width(
            pair_schema,
            &params.vertex_pairs.first_column,
            "pair_first",
            graph.spec.algorithm,
        )?;
        let second_width = graph_vertex_field_width(
            pair_schema,
            &params.vertex_pairs.second_column,
            "pair_second",
            graph.spec.algorithm,
        )?;
        if first_width != second_width || first_width != input.vertex_width_bytes {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_memory.plan_estimate",
                "similarity pair columns must match each other and the graph vertex width",
            )
            .with_fact("algorithm", graph.spec.algorithm.as_str())
            .with_fact("first_vertex_width_bytes", first_width)
            .with_fact("second_vertex_width_bytes", second_width)
            .with_fact("graph_vertex_width_bytes", input.vertex_width_bytes));
        }
        input = input.with_similarity_pair_input(pair_rows, first_width, pair_cardinality_source);
    }
    Ok(Some(input))
}

/// Queries the owning cuGraph host-only preflight for raw graph construction.
///
/// This deliberately maps every unavailable input or owner-query failure to
/// `None`: bounded admission then emits its existing structured preflight
/// rejection rather than converting an owner failure into a guessed byte
/// allowance. The graph definition supplies the actual transposed-layout
/// requirement used by the native constructor.
#[cfg(feature = "cugraph")]
pub(crate) fn with_graph_construction_workspace_bounds_from_owner(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> GraphMemoryModelInput {
    graph_construction_workspace_bounds_from_owner(spec, input)
        .map(|bounds| input.with_construction_workspace_bounds(bounds))
        .unwrap_or(input)
}

#[cfg(feature = "cugraph")]
pub(crate) fn graph_construction_workspace_bounds_from_owner(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
) -> Option<GraphConstructionWorkspaceBounds> {
    graph_construction_workspace_bounds_from_owner_query(spec, input, |preflight| {
        let result = cugraph::cudf_interop::graph_construction_workspace_preflight(preflight).ok()?;
        Some(GraphConstructionWorkspaceBounds::new(
            result.renumber_workspace_bytes(),
            result.compression_workspace_bytes(),
        ))
    })
}

/// Builds the exact owner preflight input, then projects a failed owner query
/// to absence so the bounded model retains the stable fail-closed rejection.
///
/// The query callback is deliberately narrow: it lets the no-bound outcome be
/// unit-tested without introducing an engine-side substitute for the owner
/// contract.
#[cfg(feature = "cugraph")]
pub(crate) fn graph_construction_workspace_bounds_from_owner_query(
    spec: &GraphAlgorithmSpec,
    input: GraphMemoryModelInput,
    query: impl FnOnce(&cugraph::cudf_interop::GraphConstructionPreflightInput) -> Option<GraphConstructionWorkspaceBounds>,
) -> Option<GraphConstructionWorkspaceBounds> {
    if spec.graph_options.policy != GraphConstructionPolicy::RawLibcugraph {
        return None;
    }

    let vertex_type = match input.vertex_width_bytes {
        4 => cugraph::types::DataType::Int32,
        8 => cugraph::types::DataType::Int64,
        _ => return None,
    };
    let weight_type = match input.weight_width_bytes {
        None => None,
        Some(4) => Some(cugraph::types::DataType::Float32),
        Some(8) => Some(cugraph::types::DataType::Float64),
        Some(_) => return None,
    };
    let store_transposed =
        matches!(spec.algorithm.definition().layout_requirement, crate::plan::GraphLayoutRequirement::Transposed);
    let vertex_upper_bound = input.vertex_upper_bound().ok()?;
    let num_vertices =
        input.vertex_estimate.map_or(vertex_upper_bound, |estimate| estimate.count.min(vertex_upper_bound));
    let preflight = cugraph::cudf_interop::GraphConstructionPreflightInput::new(
        vertex_type,
        vertex_type,
        weight_type,
        input.has_edge_id,
        store_transposed,
        spec.graph_options.renumber,
        usize::try_from(input.input_edges).ok()?,
        usize::try_from(num_vertices).ok()?,
    )
    .ok()?;
    query(&preflight)
}

/// Returns only the pair-cardinality evidence that can prove bounded
/// similarity admission before the candidate relation is materialized.
///
/// This intentionally excludes runtime counts.  Callers that execute a
/// bounded graph node must invoke it before evaluating graph side
/// inputs, then price the graph reservation with the returned exact count or
/// upper bound.
pub(crate) fn bounded_similarity_pair_cardinality_for_plan(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
) -> Result<Option<(u64, GraphPairCardinalitySource)>> {
    if !graph.spec.algorithm.is_similarity() {
        return Ok(None);
    }
    let crate::plan::GraphAlgorithmParams::Similarity(params) = &graph.spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_memory.bounded_similarity_admission",
            "similarity algorithm node carried non-similarity parameters",
        )
        .with_fact("algorithm", graph.spec.algorithm.as_str()));
    };
    hard_row_upper_bound_for_plan_node(plan, params.vertex_pairs.input).map(Some).ok_or_else(|| {
        bounded_similarity_pair_cardinality_error(graph.spec.algorithm, "missing_pair_cardinality_evidence")
    })
}

/// Derive only mathematically hard cardinality evidence.  This deliberately
/// does not reuse [`estimated_output_rows`], whose contract is descriptive and
/// unsuitable for bounded admission.
pub(crate) fn hard_row_upper_bound_for_plan_node(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Option<(u64, GraphPairCardinalitySource)> {
    hard_row_upper_bound_with_resident_rows(plan, node_id, &HashMap::new())
}

fn hard_row_upper_bound_with_resident_rows(
    plan: &QueryPlan,
    node_id: NodeId,
    resident_rows: &HashMap<String, u64>,
) -> Option<(u64, GraphPairCardinalitySource)> {
    let node = plan.node(node_id)?;
    let input_bound = |input| hard_row_upper_bound_with_resident_rows(plan, input, resident_rows);
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LiteralBatch(source) => {
                Some((source.row_count() as u64, GraphPairCardinalitySource::PlanExact))
            },
            crate::source::SourceNode::LocalParquet(source) => {
                let rows = source.exact_selected_row_count()?;
                let rows = source.scan_limit.map_or(rows, |limit| rows.min(limit as u64));
                Some((
                    source.max_rows.map_or(rows, |max_rows| rows.min(max_rows as u64)),
                    GraphPairCardinalitySource::PlanHardUpperBound,
                ))
            },
            crate::source::SourceNode::GpuResident(source) => resident_rows
                .get(&source.input_name)
                .copied()
                .or(source.row_count_upper_bound)
                .map(|rows| (rows, GraphPairCardinalitySource::PlanHardUpperBound)),
        },
        PlanNode::Projection(projection) => input_bound(projection.input),
        PlanNode::Filter(filter) => {
            input_bound(filter.input).map(|(rows, _)| (rows, GraphPairCardinalitySource::PlanHardUpperBound))
        },
        PlanNode::Window(window) => input_bound(window.input),
        PlanNode::Sort(sort) => input_bound(sort.input),
        PlanNode::Limit(limit) => match limit.spec.fetch {
            Some(fetch) => Some((fetch as u64, GraphPairCardinalitySource::PlanHardUpperBound)),
            None => input_bound(limit.input).map(|(rows, _)| {
                (rows.saturating_sub(limit.spec.skip as u64), GraphPairCardinalitySource::PlanHardUpperBound)
            }),
        },
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .try_fold(0_u64, |total, input| input_bound(*input)?.0.checked_add(total))
            .map(|rows| (rows, GraphPairCardinalitySource::PlanHardUpperBound)),
        PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => {
            Some((1, GraphPairCardinalitySource::PlanExact))
        },
        PlanNode::Aggregate(aggregate) => {
            input_bound(aggregate.input).map(|(rows, _)| (rows, GraphPairCardinalitySource::PlanHardUpperBound))
        },
        PlanNode::Join(join) => {
            let left_rows = input_bound(join.inputs[0])?.0;
            let rows = match join.spec.join_type {
                JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => Some(left_rows),
                JoinType::Inner
                    if !join.spec.keys.is_empty()
                        && join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) =>
                {
                    Some(left_rows)
                },
                JoinType::Inner
                    if !join.spec.keys.is_empty()
                        && join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Right) =>
                {
                    Some(input_bound(join.inputs[1])?.0)
                },
                JoinType::Inner | JoinType::Cross => left_rows.checked_mul(input_bound(join.inputs[1])?.0),
                JoinType::Left
                    if !join.spec.keys.is_empty()
                        && join.spec.has_at_most_one_match_per_bounded_row(JoinSide::Left) =>
                {
                    Some(left_rows)
                },
                JoinType::Left => left_rows.checked_mul(input_bound(join.inputs[1])?.0.max(1)),
                JoinType::Full => {
                    let right_rows = input_bound(join.inputs[1])?.0;
                    left_rows.checked_mul(right_rows)?.checked_add(left_rows)?.checked_add(right_rows)
                },
            }?;
            Some((rows, GraphPairCardinalitySource::PlanHardUpperBound))
        },
        PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) | PlanNode::VectorAlgorithm(_) => None,
    }
}

fn graph_vertex_upper_bound_for_plan_node(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => source.graph_vertex_upper_bound(),
            crate::source::SourceNode::GpuResident(source) => source.graph_vertex_upper_bound,
            crate::source::SourceNode::LiteralBatch(_) => None,
        },
        PlanNode::Projection(projection) => graph_vertex_upper_bound_for_plan_node(plan, projection.input),
        PlanNode::Window(window) => graph_vertex_upper_bound_for_plan_node(plan, window.input),
        PlanNode::Sort(sort) => graph_vertex_upper_bound_for_plan_node(plan, sort.input),
        PlanNode::EdgeNormalize(edge_normalize) => graph_vertex_upper_bound_for_plan_node(plan, edge_normalize.input),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .try_fold(0_u64, |total, input| total.checked_add(graph_vertex_upper_bound_for_plan_node(plan, *input)?)),
        PlanNode::Filter(_)
        | PlanNode::Limit(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Join(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => None,
    }
}

fn graph_memory_model_input_for_edge_list(
    spec: &GraphAlgorithmSpec,
    input_schema: &SchemaRef,
    input_edges: u64,
    raw_input_edges: Option<u64>,
) -> Result<GraphMemoryModelInput> {
    let source_width = graph_vertex_field_width(input_schema, &spec.edge_columns.source, "source", spec.algorithm)?;
    let destination_width =
        graph_vertex_field_width(input_schema, &spec.edge_columns.destination, "destination", spec.algorithm)?;
    if source_width != destination_width {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_memory.plan_estimate",
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_width_bytes", source_width)
        .with_fact("dst_width_bytes", destination_width));
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        let edge_id_width = graph_vertex_field_width(input_schema, edge_id, "edge_id", spec.algorithm)?;
        if edge_id_width != source_width {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_memory.plan_estimate",
                "cuGraph edge-id column dtype must match the graph vertex dtype",
            )
            .with_fact("edge_id_col", edge_id)
            .with_fact("edge_id_width_bytes", edge_id_width)
            .with_fact("vertex_width_bytes", source_width));
        }
    }

    let weight_width_bytes = if spec.algorithm.consumes_weight() {
        spec.edge_columns
            .weight
            .as_ref()
            .map(|weight| graph_weight_field_width(input_schema, weight, spec.algorithm))
            .transpose()?
    } else {
        None
    };
    let mut input = graph_memory_model_input(
        input_edges,
        source_width,
        &spec.edge_columns,
        spec.algorithm.consumes_weight(),
        weight_width_bytes,
    )
    .with_edge_normalize_inserted(spec.edge_normalize_inserted);
    if let Some(raw_input_edges) = raw_input_edges {
        input = input.with_raw_input_edges(raw_input_edges);
    }
    Ok(input)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphPlanEdgeCounts {
    raw_input_edges: Option<u64>,
    input_edges: u64,
    graph_vertex_upper_bound: Option<u64>,
}

fn graph_plan_edge_counts(
    plan: &QueryPlan,
    graph_input: NodeId,
    configured_raw_input_edges: Option<u64>,
    resident_rows: &HashMap<String, u64>,
) -> Result<Option<GraphPlanEdgeCounts>> {
    let Some(node) = plan.node(graph_input) else {
        return Ok(None);
    };
    if let PlanNode::EdgeNormalize(edge_normalize) = &node.kind {
        let Some(raw_input_edges) = hard_row_upper_bound_with_resident_rows(plan, edge_normalize.input, resident_rows)
            .map(|(rows, _)| rows)
            .or(configured_raw_input_edges)
        else {
            return Ok(None);
        };
        return Ok(Some(GraphPlanEdgeCounts {
            raw_input_edges: Some(raw_input_edges),
            input_edges: edge_normalize_output_row_upper_bound(raw_input_edges, &edge_normalize.spec)?,
            graph_vertex_upper_bound: graph_vertex_upper_bound_for_plan_node(plan, edge_normalize.input),
        }));
    }
    let Some(input_edges) = hard_row_upper_bound_with_resident_rows(plan, graph_input, resident_rows)
        .map(|(rows, _)| rows)
        .or(configured_raw_input_edges)
    else {
        return Ok(None);
    };
    Ok(Some(GraphPlanEdgeCounts {
        raw_input_edges: configured_raw_input_edges,
        input_edges,
        graph_vertex_upper_bound: graph_vertex_upper_bound_for_plan_node(plan, graph_input),
    }))
}

/// Size the exact-vertex probe reservation.
///
/// The probe copies both endpoint columns, concatenates them, distincts the
/// result, and sorts vertices for deterministic seed selection. Under the
/// graph stream lease this includes output buffers and cudf temporary
/// workspaces, so the multiplier is intentionally higher than the old
/// output-only budget-tracking estimate.
pub fn graph_vertex_estimate_probe_bytes(input: GraphMemoryModelInput, remaining_bytes: u64) -> u64 {
    input
        .input_edges
        .saturating_mul(input.vertex_width_bytes.saturating_mul(16).saturating_add(16))
        .max(GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES)
        .min(remaining_bytes)
        .max(1)
}

/// Size the bounded runtime statistics probe.
///
/// Algorithms that only need exact endpoint cardinality use the narrower
/// exact-vertex probe. Algorithms whose admission depends on additional
/// topology evidence get the full transient graph-statistics window; the probe
/// reservation is released before the graph-stage reservation, so this does not
/// inflate the final graph ask.
pub fn graph_runtime_statistics_probe_bytes(
    algorithm: GraphAlgorithm,
    input: GraphMemoryModelInput,
    remaining_bytes: u64,
) -> u64 {
    if graph_runtime_statistics_requires_algorithm_evidence(algorithm) {
        remaining_bytes.max(1)
    } else {
        graph_vertex_estimate_probe_bytes(input, remaining_bytes)
    }
}

const fn graph_runtime_statistics_requires_algorithm_evidence(algorithm: GraphAlgorithm) -> bool {
    matches!(
        algorithm,
        GraphAlgorithm::TriangleCountAll
            | GraphAlgorithm::BetweennessCentrality
            | GraphAlgorithm::EdgeBetweennessCentrality
    )
}

/// Derive the triangle-count peak chunk min-degree pricing term.
///
/// cuGraph chunks DFG endpoint pairs by dense destination id, which is not
/// directly observable from sparse external vertex ids. Use the average chunk
/// load implied by the measured full min-degree sum plus a fixed safety margin;
/// the graph stream lease turns residual under-margin into a contained,
/// retryable allocation failure instead of shared-pool pressure.
pub fn triangle_count_peak_chunk_min_degree_pricing_term(
    sum_min_degree_oriented: u64,
    oriented_edge_count: u64,
    device_multiprocessor_count: Option<u32>,
) -> Result<u64> {
    if sum_min_degree_oriented == 0 || oriented_edge_count == 0 {
        return Ok(0);
    }
    let sm_count = u64::from(device_multiprocessor_count.unwrap_or(1)).max(1);
    let max_chunk_size = sm_count
        .checked_mul(1 << 15)
        .ok_or_else(|| graph_memory_arithmetic_error("triangle_chunk_size", "multiplication"))?;
    let chunk_count = ceil_div_u64(oriented_edge_count, max_chunk_size)?.max(1);
    let average_chunk = ceil_div_u64(sum_min_degree_oriented, chunk_count)?;
    Ok(ceil_checked_ratio(
        average_chunk,
        TRIANGLE_COUNT_CHUNK_SAFETY_NUMERATOR,
        TRIANGLE_COUNT_CHUNK_SAFETY_DENOMINATOR,
        "triangle_peak_chunk_min_degree",
    )?
    .min(sum_min_degree_oriented))
}

fn graph_calibration_source(input: GraphMemoryModelInput) -> GraphCalibrationSource {
    if input.raw_input_edges.is_some() { GraphCalibrationSource::RawInputEdges } else { GraphCalibrationSource::None }
}

fn ceil_div_u64(value: u64, denominator: u64) -> Result<u64> {
    if denominator == 0 {
        return Err(graph_memory_arithmetic_error("triangle_chunk_count", "division"));
    }
    let numerator = u128::from(value);
    let denominator = u128::from(denominator);
    u64::try_from(numerator.div_ceil(denominator))
        .map_err(|_| graph_memory_arithmetic_error("triangle_ceil_division", "u128_to_u64"))
}

fn ceil_checked_ratio(value: u64, numerator: u64, denominator: u64, term: &'static str) -> Result<u64> {
    if denominator == 0 {
        return Err(graph_memory_arithmetic_error(term, "division"));
    }
    let product = u128::from(value)
        .checked_mul(u128::from(numerator))
        .ok_or_else(|| graph_memory_arithmetic_error(term, "multiplication"))?;
    let rounded = product
        .checked_add(u128::from(denominator - 1))
        .ok_or_else(|| graph_memory_arithmetic_error(term, "addition"))?
        / u128::from(denominator);
    u64::try_from(rounded).map_err(|_| graph_memory_arithmetic_error(term, "u128_to_u64"))
}

fn edge_normalize_output_row_upper_bound(raw_input_edges: u64, spec: &EdgeNormalizeSpec) -> Result<u64> {
    raw_input_edges
        .checked_mul(if spec.add_reverse { 2 } else { 1 })
        .ok_or_else(|| graph_memory_arithmetic_error("edge_normalize_output_rows", "multiplication"))
}

fn graph_vertex_field_width(
    schema: &SchemaRef,
    column: &str,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<u64> {
    let field = schema.field_with_name(column).map_err(|_error| graph_field_error(column, role, algorithm))?;
    match field.data_type() {
        DataType::Int32 => Ok(4),
        DataType::Int64 => Ok(8),
        other => Err(graph_field_type_error(column, role, algorithm, other)),
    }
}

fn graph_weight_field_width(schema: &SchemaRef, column: &str, algorithm: GraphAlgorithm) -> Result<u64> {
    let field = schema.field_with_name(column).map_err(|_error| graph_field_error(column, "weight", algorithm))?;
    match field.data_type() {
        DataType::Float32 => Ok(4),
        DataType::Float64 => Ok(8),
        other => Err(graph_field_type_error(column, "weight", algorithm, other)),
    }
}

fn graph_field_error(column: &str, role: &'static str, algorithm: GraphAlgorithm) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.plan_estimate",
        "graph memory plan estimate could not resolve an edge-list column",
    )
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("column", column)
    .with_fact("role", role)
}

fn graph_field_type_error(
    column: &str,
    role: &'static str,
    algorithm: GraphAlgorithm,
    data_type: &DataType,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedType,
        "nexus_query_engine.graph_memory.plan_estimate",
        "graph memory plan estimate only supports fixed-width cuGraph edge types",
    )
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("column", column)
    .with_fact("role", role)
    .with_fact("arrow_type", format!("{data_type:?}"))
}

pub(super) fn missing_graph_input_error(input: NodeId) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_memory.plan_estimate",
        "graph memory plan estimate referenced a missing graph input node",
    )
    .with_fact("input", input.to_string())
}
