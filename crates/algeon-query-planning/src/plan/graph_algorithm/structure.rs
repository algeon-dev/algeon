//! Graph structure, edge normalization, and structural algorithm contracts.
use std::fmt;

use super::{GraphConstructionOptions, GraphEdgeColumnBindings};

/// Edge-list normalization inserted before raw cuGraph construction.
///
/// The Python-compatible policy drops duplicate directed edges and inserts
/// reverse edges for undirected graphs before cuGraph construction. Weighted
/// duplicate conflicts use [`EdgeWeightCombiner`] to define the retained value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeNormalizeSpec {
    /// Edge-list columns used by normalization.
    pub edge_columns: GraphEdgeColumnBindings,
    /// Whether reverse edges are inserted before deduplication.
    pub add_reverse: bool,
    /// Duplicate-edge retention policy.
    pub dedup: EdgeDedupStrategy,
    /// Duplicate-weight conflict policy.
    pub weight_combiner: EdgeWeightCombiner,
}

impl EdgeNormalizeSpec {
    /// Build the Python-compatible edge normalization spec.
    pub fn python_cugraph(edge_columns: GraphEdgeColumnBindings, graph_options: &GraphConstructionOptions) -> Self {
        let has_weight = edge_columns.weight.is_some();
        let (dedup, weight_combiner) = match (graph_options.directed, has_weight) {
            (_, false) => (EdgeDedupStrategy::KeepLast, EdgeWeightCombiner::None),
            (true, true) => (EdgeDedupStrategy::KeepLast, EdgeWeightCombiner::KeepLastInputOrder),
            (false, true) => (EdgeDedupStrategy::KeepFirst, EdgeWeightCombiner::MinWeight),
        };
        Self { edge_columns, add_reverse: !graph_options.directed, dedup, weight_combiner }
    }
}

/// Duplicate-edge handling applied by edge-list normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeDedupStrategy {
    /// Keep duplicate edges.
    None,
    /// Keep the first duplicate edge in input order.
    KeepFirst,
    /// Keep the last duplicate edge in input order.
    KeepLast,
}

impl EdgeDedupStrategy {
    /// Stable lowercase duplicate-edge policy code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::KeepFirst => "keep_first",
            Self::KeepLast => "keep_last",
        }
    }
}

impl fmt::Display for EdgeDedupStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Weighted duplicate-edge conflict policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeWeightCombiner {
    /// No weight combination is required.
    None,
    /// Keep the minimum weight among duplicate edges.
    MinWeight,
    /// Keep the weight from the last duplicate in input order.
    KeepLastInputOrder,
}

impl EdgeWeightCombiner {
    /// Stable lowercase weight-combiner code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::MinWeight => "min_weight",
            Self::KeepLastInputOrder => "keep_last_input_order",
        }
    }
}

impl fmt::Display for EdgeWeightCombiner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Degree direction for core-number and k-core computations.
///
/// Mirrors `cugraph::structure::KCoreDegreeType` without depending on the
/// cugraph feature so the native IR can be constructed without the cuGraph
/// feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GraphKCoreDegreeType {
    /// Use incoming edges.
    In,
    /// Use outgoing edges.
    Out,
    #[default]
    /// Use both incoming and outgoing edges.
    InOut,
}

impl GraphKCoreDegreeType {
    /// Stable lowercase degree-type code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::InOut => "in_out",
        }
    }
}

impl fmt::Display for GraphKCoreDegreeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Zero-argument all-vertex degree computation.
///
/// No algorithm-specific options beyond the shared graph construction options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphDegreesAllParams;

/// Zero-argument all-vertex in-degree computation.
///
/// Direction is fixed by the algorithm; shared graph construction options
/// still control directedness, renumbering, and edge normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphInDegreesAllParams;

/// Zero-argument all-vertex out-degree computation.
///
/// Direction is fixed by the algorithm; shared graph construction options
/// still control directedness, renumbering, and edge normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphOutDegreesAllParams;

/// Zero-argument all-vertex triangle count.
///
/// Output schema is fixed by [`crate::plan::stable_graph_result_schema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphTriangleCountAllParams;

/// Parameters for core-number computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphCoreNumberParams {
    /// Which edge direction to use when computing core numbers.
    pub degree_type: GraphKCoreDegreeType,
}

impl Default for GraphCoreNumberParams {
    fn default() -> Self {
        Self { degree_type: GraphKCoreDegreeType::InOut }
    }
}

/// Parameters for k-core subgraph extraction.
///
/// Mirrors `cugraph::structure::KCoreParams` without the `cugraph` crate's
/// newtype wrappers so the native IR can be constructed without the cuGraph feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphKCoreParams {
    /// K value for the extracted core. Must be >= 1.
    pub k: u32,
    /// Degree direction to use when computing core numbers internally.
    pub degree_type: GraphKCoreDegreeType,
}

impl Default for GraphKCoreParams {
    fn default() -> Self {
        Self { k: 2, degree_type: GraphKCoreDegreeType::InOut }
    }
}

/// Parameters for minimum spanning tree extraction.
///
/// No algorithm-specific options beyond the shared graph construction options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphMstParams;
