//! Graph layout requirements and algorithm parameter contracts.
use std::fmt;

use super::NodeId;

/// Required cuGraph graph-layout state for an algorithm.
///
/// Mirrors the `Graph<Forward>` / `Graph<Transposed>` type-state in the local `cugraph` crate.
/// [`Any`](Self::Any) indicates that the algorithm runs under either layout;
/// the executor still picks one concrete layout at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphLayoutRequirement {
    /// Algorithm requires a forward graph layout.
    Forward,
    /// Algorithm requires a transposed graph layout.
    Transposed,
    /// Algorithm can run on either layout.
    Any,
}

impl GraphLayoutRequirement {
    /// Stable lowercase layout code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Transposed => "transposed",
            Self::Any => "any",
        }
    }
}

impl fmt::Display for GraphLayoutRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parameters for ForceAtlas2 graph layout.
///
/// Mirrors `cugraph::layout::ForceAtlas2Params` without the `cugraph` crate's
/// newtype wrappers and adds a deterministic RNG seed used to create the cuGraph
/// `RngState` for each execution.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphForceAtlas2Params {
    /// Maximum number of iterations. Must be in `[1, i32::MAX]`.
    pub max_iter: u32,
    /// Distribute attraction along outbound edges.
    pub outbound_attraction_distribution: bool,
    /// Use the lin-log attraction model.
    pub lin_log_mode: bool,
    /// Prevent nodes from overlapping.
    ///
    /// Currently rejected by the native executor because ForceAtlas2 overlap
    /// prevention requires per-vertex radius side input that the native graph
    /// plan does not expose yet.
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
    /// RNG seed for deterministic layout runs.
    pub seed: u64,
    /// Optional per-vertex radius, mobility, and mass relation.
    pub vertex_attributes: Option<GraphForceAtlas2VertexAttributesInput>,
    /// Optional complete warm-start position relation.
    pub initial_positions: Option<GraphForceAtlas2InitialPositionsInput>,
}

impl GraphForceAtlas2Params {
    /// Visits every side-input node id in stable role order.
    pub fn side_input_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.vertex_attributes
            .iter()
            .map(|input| input.input)
            .chain(self.initial_positions.iter().map(|input| input.input))
    }
}

/// Optional per-vertex ForceAtlas2 attributes produced by another plan node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphForceAtlas2VertexAttributesInput {
    /// Plan node producing the attribute relation.
    pub input: NodeId,
    /// Vertex identifier column.
    pub vertex: String,
    /// Optional non-null Float32 radius column.
    pub radius: Option<String>,
    /// Optional non-null Float32 mobility column.
    pub mobility: Option<String>,
    /// Optional non-null Float32 mass column.
    pub mass: Option<String>,
}

impl GraphForceAtlas2VertexAttributesInput {
    /// Creates a vertex-attribute binding.
    pub fn new(
        input: NodeId,
        vertex: impl Into<String>,
        radius: Option<impl Into<String>>,
        mobility: Option<impl Into<String>>,
        mass: Option<impl Into<String>>,
    ) -> Self {
        Self {
            input,
            vertex: vertex.into(),
            radius: radius.map(Into::into),
            mobility: mobility.map(Into::into),
            mass: mass.map(Into::into),
        }
    }
}

/// Complete per-vertex ForceAtlas2 warm-start positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphForceAtlas2InitialPositionsInput {
    /// Plan node producing the position relation.
    pub input: NodeId,
    /// Vertex identifier column.
    pub vertex: String,
    /// Non-null Float32 X-coordinate column.
    pub x: String,
    /// Non-null Float32 Y-coordinate column.
    pub y: String,
}

impl GraphForceAtlas2InitialPositionsInput {
    /// Creates an initial-position binding.
    pub fn new(input: NodeId, vertex: impl Into<String>, x: impl Into<String>, y: impl Into<String>) -> Self {
        Self { input, vertex: vertex.into(), x: x.into(), y: y.into() }
    }
}

impl Default for GraphForceAtlas2Params {
    fn default() -> Self {
        // Matches cugraph::layout::ForceAtlas2Params::default(), plus a
        // deterministic seed for the required cuGraph RNG state.
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
