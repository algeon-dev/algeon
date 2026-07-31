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
    /// Use this when you need the high-level cudf interop entry points to
    /// behave exactly like the lower-level [`GraphBuilder`](crate::GraphBuilder) flags you specify.
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
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices (recommended when vertex IDs are not
    /// sequential from 0).
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

/// Options for building a graph from a cudf [`DataFrame`] using column names.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::DataFrameGraphOptions;
///
/// let opts = DataFrameGraphOptions::new("src", "dst")
///     .weight_col("weight")
///     .renumber();
/// ```
///
/// [`DataFrame`]: cudf::DataFrame
#[derive(Debug, Clone)]
#[must_use]
pub struct DataFrameGraphOptions<'a> {
    /// Column name for source vertex IDs.
    pub src_col: &'a str,
    /// Column name for destination vertex IDs.
    pub dst_col: &'a str,
    /// Optional column name for edge weights.
    pub weight_col: Option<&'a str>,
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices.
    pub renumber: bool,
    /// High-level graph-construction semantics to apply before crossing into libcugraph.
    pub construction_policy: GraphConstructionPolicy,
}

impl<'a> DataFrameGraphOptions<'a> {
    /// Create options with source and destination column names.
    ///
    /// Defaults: no weight column, directed graph, renumber disabled, and
    /// Python-compatible construction semantics.
    pub fn new(src_col: &'a str, dst_col: &'a str) -> Self {
        Self {
            src_col,
            dst_col,
            weight_col: None,
            properties: GraphProperties::directed(),
            renumber: false,
            construction_policy: GraphConstructionPolicy::PythonCugraph,
        }
    }

    /// Set the weight column name.
    pub fn weight_col(mut self, col: &'a str) -> Self {
        self.weight_col = Some(col);
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

/// Options for building a graph from cudf [`Series`] directly.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::SeriesGraphOptions;
///
/// let opts = SeriesGraphOptions::default().renumber();
/// ```
///
/// [`Series`]: cudf::Series
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct SeriesGraphOptions {
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices.
    pub renumber: bool,
    /// High-level graph-construction semantics to apply before crossing into
    /// libcugraph.
    pub construction_policy: GraphConstructionPolicy,
}

impl SeriesGraphOptions {
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

/// Options for loading a graph from a Parquet file.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::ParquetGraphOptions;
///
/// let opts = ParquetGraphOptions::new("edges.parquet", 0, 1)
///     .weight_col(2)
///     .renumber();
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct ParquetGraphOptions<'a> {
    /// File-system path to the Parquet file.
    pub path: &'a str,
    /// Zero-based column index for source vertex IDs.
    pub src_col: usize,
    /// Zero-based column index for destination vertex IDs.
    pub dst_col: usize,
    /// Optional zero-based column index for edge weights.
    pub weight_col: Option<usize>,
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices.
    pub renumber: bool,
    /// High-level graph-construction semantics to apply before crossing into
    /// libcugraph.
    pub construction_policy: GraphConstructionPolicy,
}

impl<'a> ParquetGraphOptions<'a> {
    /// Create options with path and source/destination column indices.
    ///
    /// Defaults: no weight column, directed graph, renumber disabled, and
    /// Python-compatible construction semantics.
    pub fn new(path: &'a str, src_col: usize, dst_col: usize) -> Self {
        Self {
            path,
            src_col,
            dst_col,
            weight_col: None,
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
