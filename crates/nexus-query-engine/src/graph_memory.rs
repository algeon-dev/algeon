//! Approximate cuGraph memory pricing outside the strict bounded gate.
//!
//! These estimates are intentionally not strict-bounded contracts. They price a
//! graph fragment before entering cuGraph so bounded service admission can fail
//! locally with structured facts instead of relying on a raw RMM OOM.

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionPolicy, GraphEdgeColumnBindings,
    },
    runtime::{MemoryReservationRequest, MemorySpaceId, ReservationKind},
};

/// Stable graph memory-tier boundary used in reports and reservation facts.
pub const GRAPH_MEMORY_BOUNDARY: &str = "graph_memory";

/// Stable local rejection reason for graph estimate denial.
pub const GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON: &str = "graph_memory_estimate_exceeds_budget";

/// Minimum live reservation used when a graph fragment is hard-limited by RMM.
///
/// Tiny model estimates are useful diagnostics, but cuGraph still needs a
/// practical allocator envelope for graph construction and result conversion.
/// Bounded admission reserves this floor so the RMM cap never exceeds the
/// live graph reservation.
pub const GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES: u64 = 1024 * 1024;

/// Conservative class assigned to a cuGraph algorithm for approximate pricing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphMemoryClass {
    /// Output-sized algorithms with `O(V + E)` storage.
    OutputSized,
    /// Multi-array algorithms with extra per-level or ensemble state.
    MultiArray,
    /// Workspace-heavy algorithms with data-dependent allocation shape.
    WorkspaceHeavyDataDependent,
    /// Workspace-heavy algorithms whose memory is primarily seed-count driven.
    WorkspaceHeavySeedDependent,
}

impl GraphMemoryClass {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutputSized => "output_sized",
            Self::MultiArray => "multi_array",
            Self::WorkspaceHeavyDataDependent => "workspace_heavy_data_dependent",
            Self::WorkspaceHeavySeedDependent => "workspace_heavy_seed_dependent",
        }
    }

    const fn is_data_dependent(self) -> bool {
        matches!(self, Self::WorkspaceHeavyDataDependent | Self::WorkspaceHeavySeedDependent)
    }
}

impl std::fmt::Display for GraphMemoryClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome of graph memory-tier admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphReservationOutcome {
    /// No bounded budget was active, so the estimate was diagnostic-only.
    NotBounded,
    /// Estimate fit the current device budget.
    Admitted,
    /// Estimate exceeded the current device budget.
    Denied,
}

impl GraphReservationOutcome {
    /// Stable label used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotBounded => "not_bounded",
            Self::Admitted => "admitted",
            Self::Denied => "denied",
        }
    }
}

impl std::fmt::Display for GraphReservationOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Plan/runtime inputs used by the approximate graph memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphMemoryModelInput {
    /// Edge-row count after graph edge filtering/normalization.
    pub input_edges: u64,
    /// Width in bytes of the vertex id type.
    pub vertex_width_bytes: u64,
    /// Width in bytes of the optional weight column consumed by the algorithm.
    pub weight_width_bytes: Option<u64>,
    /// Whether the graph has a stable edge-id column.
    pub has_edge_id: bool,
}

impl GraphMemoryModelInput {
    /// Conservative upper bound on vertex count: every edge contributes two
    /// distinct endpoints.
    pub fn vertex_upper_bound(self) -> u64 {
        self.input_edges.saturating_mul(2)
    }
}

/// Approximate graph memory estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphMemoryEstimate {
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub seed_count: Option<u64>,
    pub data_dependent: bool,
    pub requested_bytes: u64,
}

/// Public graph memory-tier report row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphReservationReport {
    pub node_id: String,
    pub algorithm: GraphAlgorithm,
    pub class: GraphMemoryClass,
    pub outcome: GraphReservationOutcome,
    pub input_edges: u64,
    pub vertex_upper_bound: u64,
    pub vertex_width_bytes: u64,
    pub weight_width_bytes: Option<u64>,
    pub has_edge_id: bool,
    pub directed: bool,
    pub renumber: bool,
    pub construction_policy: GraphConstructionPolicy,
    pub seed_count: Option<u64>,
    pub data_dependent: bool,
    pub requested_bytes: u64,
    pub budget_bytes: Option<u64>,
    pub reason: Option<&'static str>,
}

impl GraphReservationReport {
    /// Builds a report from an estimate and optional bounded budget.
    pub fn from_estimate(node_id: impl ToString, estimate: GraphMemoryEstimate, budget_bytes: Option<u64>) -> Self {
        let requested_bytes = if budget_bytes.is_some() {
            estimate.requested_bytes.max(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)
        } else {
            estimate.requested_bytes
        };
        let denied = budget_bytes.is_some_and(|budget| requested_bytes > budget);
        let outcome = match (budget_bytes, denied) {
            (None, _) => GraphReservationOutcome::NotBounded,
            (Some(_), true) => GraphReservationOutcome::Denied,
            (Some(_), false) => GraphReservationOutcome::Admitted,
        };
        Self {
            node_id: node_id.to_string(),
            algorithm: estimate.algorithm,
            class: estimate.class,
            outcome,
            input_edges: estimate.input_edges,
            vertex_upper_bound: estimate.vertex_upper_bound,
            vertex_width_bytes: estimate.vertex_width_bytes,
            weight_width_bytes: estimate.weight_width_bytes,
            has_edge_id: estimate.has_edge_id,
            directed: estimate.directed,
            renumber: estimate.renumber,
            construction_policy: estimate.construction_policy,
            seed_count: estimate.seed_count,
            data_dependent: estimate.data_dependent,
            requested_bytes,
            budget_bytes,
            reason: denied.then_some(GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON),
        }
    }

    /// Builds the structured local admission error for a denied graph fragment.
    pub fn denied_error(&self, request: &MemoryReservationRequest, granted_bytes: u64) -> QueryError {
        let mut error = request
            .denied_error(granted_bytes)
            .with_fact("reason", GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON)
            .with_fact("algorithm", self.algorithm.as_str())
            .with_fact("graph_memory_class", self.class.as_str())
            .with_fact("input_edges", self.input_edges)
            .with_fact("vertex_upper_bound", self.vertex_upper_bound)
            .with_fact("vertex_width_bytes", self.vertex_width_bytes)
            .with_fact("directed", self.directed)
            .with_fact("renumber", self.renumber)
            .with_fact("construction_policy", self.construction_policy.as_str())
            .with_fact("data_dependent", self.data_dependent);
        if let Some(seed_count) = self.seed_count {
            error = error.with_fact("seed_count", seed_count);
        }
        error
    }
}

/// Approximate model for the graph memory tier.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphMemoryModel;

impl GraphMemoryModel {
    /// Estimate the graph fragment memory for an algorithm and graph input.
    pub fn estimate(&self, spec: &GraphAlgorithmSpec, input: GraphMemoryModelInput) -> Result<GraphMemoryEstimate> {
        if input.vertex_width_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.estimate",
                "graph memory model requires a positive vertex dtype width",
            )
            .with_fact("vertex_width_bytes", input.vertex_width_bytes));
        }

        let class = graph_memory_class(spec.algorithm);
        let vertex_upper_bound = input.vertex_upper_bound();
        let edge_row_width = input
            .vertex_width_bytes
            .saturating_mul(2)
            .saturating_add(input.weight_width_bytes.unwrap_or(0))
            .saturating_add(if input.has_edge_id { input.vertex_width_bytes } else { 0 });
        let construction_multiplier = construction_multiplier(spec.graph_options.policy, spec.graph_options.directed);
        let construction_bytes = saturating_product(input.input_edges, edge_row_width)
            .saturating_mul(construction_multiplier)
            .saturating_add(if spec.graph_options.renumber {
                saturating_product(vertex_upper_bound, input.vertex_width_bytes.saturating_mul(3))
            } else {
                0
            });
        let output_bytes = output_bytes_for_algorithm(spec.algorithm, vertex_upper_bound, input.input_edges);
        let seed_count = graph_seed_count(&spec.params, vertex_upper_bound);
        let workspace_bytes = workspace_bytes_for_class(class, input, vertex_upper_bound, seed_count);
        let requested_bytes = construction_bytes.saturating_add(output_bytes).saturating_add(workspace_bytes).max(1);

        Ok(GraphMemoryEstimate {
            algorithm: spec.algorithm,
            class,
            input_edges: input.input_edges,
            vertex_upper_bound,
            vertex_width_bytes: input.vertex_width_bytes,
            weight_width_bytes: input.weight_width_bytes,
            has_edge_id: input.has_edge_id,
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            construction_policy: spec.graph_options.policy,
            seed_count,
            data_dependent: class.is_data_dependent(),
            requested_bytes,
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
                .with_node_id(node_id)
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
        GraphAlgorithm::Louvain
        | GraphAlgorithm::Leiden
        | GraphAlgorithm::Ecg
        | GraphAlgorithm::KCore
        | GraphAlgorithm::StronglyConnectedComponents => GraphMemoryClass::MultiArray,
        GraphAlgorithm::TriangleCountAll
        | GraphAlgorithm::SpectralModularityMaximization
        | GraphAlgorithm::BalancedCutClustering
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
        | GraphAlgorithm::BalancedCutClustering
        | GraphAlgorithm::ForceAtlas2 => true,
    }
}

fn graph_seed_count(params: &GraphAlgorithmParams, vertex_upper_bound: u64) -> Option<u64> {
    match params {
        GraphAlgorithmParams::BetweennessCentrality(params) => Some(
            params.seeds.as_ref().map_or_else(|| params.k.unwrap_or(vertex_upper_bound), |seeds| seeds.len() as u64),
        ),
        GraphAlgorithmParams::EdgeBetweennessCentrality(params) => Some(
            params.seeds.as_ref().map_or_else(|| params.k.unwrap_or(vertex_upper_bound), |seeds| seeds.len() as u64),
        ),
        _ => None,
    }
}

fn construction_multiplier(policy: GraphConstructionPolicy, directed: bool) -> u64 {
    match (policy, directed) {
        (GraphConstructionPolicy::PythonCugraph, false) => 3,
        (GraphConstructionPolicy::PythonCugraph, true) => 2,
        (GraphConstructionPolicy::RawLibcugraph, _) => 1,
    }
}

fn output_bytes_for_algorithm(algorithm: GraphAlgorithm, vertex_upper_bound: u64, input_edges: u64) -> u64 {
    match algorithm {
        GraphAlgorithm::KCore | GraphAlgorithm::MinimumSpanningTree => input_edges.saturating_mul(16),
        GraphAlgorithm::Hits | GraphAlgorithm::DegreesAll => vertex_upper_bound.saturating_mul(24),
        GraphAlgorithm::Bfs | GraphAlgorithm::Sssp => vertex_upper_bound.saturating_mul(24),
        GraphAlgorithm::ForceAtlas2 => vertex_upper_bound.saturating_mul(16),
        _ => vertex_upper_bound.saturating_mul(16),
    }
}

fn workspace_bytes_for_class(
    class: GraphMemoryClass,
    input: GraphMemoryModelInput,
    vertex_upper_bound: u64,
    seed_count: Option<u64>,
) -> u64 {
    let vertex_array = vertex_upper_bound.saturating_mul(input.vertex_width_bytes.max(8));
    let edge_index = input.input_edges.saturating_mul(input.vertex_width_bytes.saturating_mul(2));
    let output_sized = vertex_array.saturating_mul(3).saturating_add(edge_index);
    match class {
        GraphMemoryClass::OutputSized => output_sized,
        GraphMemoryClass::MultiArray => output_sized.saturating_mul(4),
        GraphMemoryClass::WorkspaceHeavyDataDependent => {
            let superlinear_probe = vertex_upper_bound.saturating_mul(vertex_upper_bound.min(1024)).saturating_mul(4);
            output_sized.saturating_mul(8).saturating_add(superlinear_probe)
        },
        GraphMemoryClass::WorkspaceHeavySeedDependent => {
            let seeds = seed_count.unwrap_or(vertex_upper_bound);
            output_sized.saturating_add(seeds.saturating_mul(vertex_upper_bound).saturating_mul(4))
        },
    }
}

fn saturating_product(left: u64, right: u64) -> u64 {
    left.saturating_mul(right)
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
        input_edges,
        vertex_width_bytes,
        weight_width_bytes: consumes_weight.then_some(weight_width_bytes).flatten(),
        has_edge_id: edge_columns.edge_id.is_some(),
    }
}
