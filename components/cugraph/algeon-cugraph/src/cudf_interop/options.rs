use crate::types::GraphProperties;

/// Options for building a graph from table-shaped cudf data using column indices.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::TableGraphOptions;
///
/// // Minimal: source and destination column indices
/// let opts = TableGraphOptions::new(0, 1);
///
/// // With weight column and renumbering
/// let opts = TableGraphOptions::new(0, 1).weight_col(2).renumber();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum GraphConstructionPolicy {
    /// Preserve the raw libcugraph graph-construction contract.
    ///
    /// Applies lower-level [`GraphBuilder`](crate::GraphBuilder) configuration
    /// directly without pre-normalizing ingress edge tables.
    RawLibcugraph,
    /// Match Python `cugraph.Graph.from_cudf_edgelist(...)` semantics for
    /// high-level cudf ingress workloads.
    ///
    /// This policy normalizes ingress edge data before raw graph
    /// construction:
    /// - directed simple graphs: drop duplicate directed edges and keep the
    ///   minimum edge value when weights exist
    /// - directed multigraphs: preserve multiplicity
    /// - undirected simple graphs: add reverse edges, then drop duplicate
    ///   directed edges and keep the minimum edge value when weights exist
    /// - undirected multigraphs: add reverse edges and preserve multiplicity
    #[default]
    PythonCugraph,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct TableGraphOptions {
    /// Zero-based column index for source vertex IDs.
    pub src_col: usize,
    /// Zero-based column index for destination vertex IDs.
    pub dst_col: usize,
    /// Optional zero-based column index for edge weights.
    pub weight_col: Option<usize>,
    /// Optional zero-based column index for stable edge IDs.
    pub edge_id_col: Option<usize>,
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices into a contiguous zero-based index range.
    pub renumber: bool,
    /// High-level graph-construction semantics to apply before crossing into
    /// libcugraph.
    pub construction_policy: GraphConstructionPolicy,
}

impl TableGraphOptions {
    /// Create options with source and destination column indices.
    ///
    /// Defaults: no weight column, directed graph, renumber disabled, and
    /// Python-compatible construction semantics.
    pub fn new(src_col: usize, dst_col: usize) -> Self {
        Self {
            src_col,
            dst_col,
            weight_col: None,
            edge_id_col: None,
            properties: GraphProperties::directed(),
            renumber: false,
            construction_policy: GraphConstructionPolicy::PythonCugraph,
        }
    }

    /// Set the weight column index.
    pub fn weight_col(mut self, col: usize) -> Self {
        self.weight_col = Some(col);
        self
    }

    /// Set the stable edge-ID column index.
    pub fn edge_id_col(mut self, col: usize) -> Self {
        self.edge_id_col = Some(col);
        self
    }

    /// Set graph properties.
    pub fn properties(mut self, properties: GraphProperties) -> Self {
        self.properties = properties;
        self
    }

    /// Enable vertex renumbering.
    pub fn renumber(mut self) -> Self {
        self.renumber = true;
        self
    }

    /// Set graph-construction semantics for this high-level ingress path.
    pub fn construction_policy(mut self, policy: GraphConstructionPolicy) -> Self {
        self.construction_policy = policy;
        self
    }
}

/// Column selection for explicit vertex-pair input from a prepared cudf table
/// export.
///
/// Pairwise similarity preserves row alignment: `first_col` and `second_col`
/// are read as two equally sized, same-typed vertex columns.  The selected
/// columns must be `Int32` or `Int64` and match the graph vertex domain.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::TableVertexPairOptions;
///
/// let options = TableVertexPairOptions::new(0, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct TableVertexPairOptions {
    /// Zero-based column index for the first vertex of each pair.
    pub first_col: usize,
    /// Zero-based column index for the second vertex of each pair.
    pub second_col: usize,
}

impl TableVertexPairOptions {
    /// Select the first and second vertex columns by zero-based index.
    pub const fn new(first_col: usize, second_col: usize) -> Self {
        Self { first_col, second_col }
    }
}

/// Options for building a graph from separate prepared cudf column exports.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::PreparedGraphOptions;
///
/// let opts = PreparedGraphOptions::default().renumber();
/// ```
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct PreparedGraphOptions {
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices.
    pub renumber: bool,
    /// High-level graph-construction semantics to apply before crossing into
    /// libcugraph.
    pub construction_policy: GraphConstructionPolicy,
}

impl PreparedGraphOptions {
    /// Set graph properties.
    pub fn properties(mut self, properties: GraphProperties) -> Self {
        self.properties = properties;
        self
    }

    /// Enable vertex renumbering.
    pub fn renumber(mut self) -> Self {
        self.renumber = true;
        self
    }

    /// Set graph-construction semantics for this high-level ingress path.
    pub fn construction_policy(mut self, policy: GraphConstructionPolicy) -> Self {
        self.construction_policy = policy;
        self
    }
}
