//! Interop between cudf and cugraph types.
//!
//! This module is only available when the `cudf-interop` feature is enabled.
//!
//! # Memory ownership model
//!
//! Functions in this module differ in whether they copy data or create
//! zero-copy views. The distinction matters for lifetime management:
//!
//! ## Zero-copy view paths
//!
//! - [`DeviceSliceExt::as_device_array_view()`] — wraps a cudf
//!   [`DeviceSlice`] as a cugraph `RawDeviceArrayView` without copying.
//!   The view borrows the slice; the slice must outlive the view.
//! - [`graph_from_table_view()`] — creates views into the cudf table's
//!   column buffers during construction. The table must outlive the
//!   intermediate views. Once graph construction completes, the [`Graph`]
//!   owns its internal CSR/CSC representation independently.
//!
//! ## Copy-based paths
//!
//! - [`RawDeviceArrayViewExt::to_cudf_column()`] — performs a
//!   device-to-device copy. The returned column owns its memory
//!   independently of the source view.
//! - [`graph_from_parquet()`] — reads Parquet from disk into GPU memory
//!   (I/O + copy). Returns a [`LoadedGraph`] that owns both the [`Graph`]
//!   and the backing table.
//! - [`graph_from_dataframe()`] — copies the DataFrame to a table, then
//!   delegates to [`graph_from_table_view()`].
//!
//! ## Owned result type
//!
//! [`LoadedGraph`] bundles a [`Graph`] with the source table loaded from
//! an external source (e.g. Parquet). The graph's internal representation
//! is independent of the table once constructed, but keeping the table
//! allows callers to reuse it for additional queries.
//!
//! ## Summary
//!
//! | Function | Copy? | Lifetime constraint |
//! |----------|-------|---------------------|
//! | [`DeviceSliceExt::as_device_array_view`] | No | View borrows slice |
//! | [`RawDeviceArrayViewExt::to_cudf_column`] | Yes | Column is independent |
//! | [`graph_from_table_view`] | Yes (CSR) | Table must outlive views during build |
//! | [`graph_from_dataframe`] | Yes | Copies table internally |
//! | [`graph_from_parquet`] | Yes | [`LoadedGraph`] owns both |
//!
//! [`DeviceSlice`]: cudf::DeviceSlice

use cudf::{column::DeviceSlice, io::parquet, types::TypeId};

use crate::{
    array::RawDeviceArrayView,
    error::{Error, ErrorKind, Result},
    graph::{Graph, GraphBuilder, Transposition},
    handle::ResourceHandle,
    types::{DataType, GraphProperties},
};

/// Convert a cudf [`TypeId`] to a cugraph [`DataType`].
///
/// Returns `None` for cudf types that have no direct cugraph equivalent
/// (e.g. timestamps, strings, structs).
pub fn cudf_type_to_cugraph(type_id: TypeId) -> Option<DataType> {
    match type_id {
        TypeId::Int8 => Some(DataType::Int8),
        TypeId::Int16 => Some(DataType::Int16),
        TypeId::Int32 => Some(DataType::Int32),
        TypeId::Int64 => Some(DataType::Int64),
        TypeId::UInt8 => Some(DataType::Uint8),
        TypeId::UInt16 => Some(DataType::Uint16),
        TypeId::UInt32 => Some(DataType::Uint32),
        TypeId::UInt64 => Some(DataType::Uint64),
        TypeId::Float32 => Some(DataType::Float32),
        TypeId::Float64 => Some(DataType::Float64),
        TypeId::Bool8 => Some(DataType::Bool),
        _ => None,
    }
}

/// Convert a cugraph [`DataType`] to a cudf [`TypeId`].
///
/// Returns `None` for cugraph types that have no direct cudf equivalent
/// (i.e. [`DataType::SizeT`], which is an internal cugraph sizing type).
pub fn cugraph_type_to_cudf(dtype: DataType) -> Option<TypeId> {
    match dtype {
        DataType::Int8 => Some(TypeId::Int8),
        DataType::Int16 => Some(TypeId::Int16),
        DataType::Int32 => Some(TypeId::Int32),
        DataType::Int64 => Some(TypeId::Int64),
        DataType::Uint8 => Some(TypeId::UInt8),
        DataType::Uint16 => Some(TypeId::UInt16),
        DataType::Uint32 => Some(TypeId::UInt32),
        DataType::Uint64 => Some(TypeId::UInt64),
        DataType::Float32 => Some(TypeId::Float32),
        DataType::Float64 => Some(TypeId::Float64),
        DataType::Bool => Some(TypeId::Bool8),
        DataType::SizeT => None,
    }
}

/// Extension trait for creating a [`RawDeviceArrayView`] from a cudf [`DeviceSlice`].
#[allow(dead_code)] // available for future internal use
pub(crate) trait DeviceSliceExt {
    /// Create a zero-copy [`RawDeviceArrayView`] from this [`DeviceSlice`].
    fn as_device_array_view(&self) -> Result<RawDeviceArrayView<'_>>;
}

impl DeviceSliceExt for DeviceSlice<'_> {
    fn as_device_array_view(&self) -> Result<RawDeviceArrayView<'_>> {
        let type_id = self.type_id();
        let cugraph_dtype = cudf_type_to_cugraph(type_id).ok_or_else(|| {
            Error::new(ErrorKind::UnsupportedType, "DeviceSliceExt::as_device_array_view")
                .with_message(format!("cudf TypeId {:?} has no cugraph DataType equivalent", type_id))
        })?;

        Ok(unsafe { RawDeviceArrayView::from_raw(self.as_ptr(), self.len(), cugraph_dtype) })
    }
}

/// Extension trait for converting a cugraph [`RawDeviceArrayView`] into a cudf [`Column`].
///
/// [`Column`]: cudf::Column
pub(crate) trait RawDeviceArrayViewExt {
    /// Convert this cugraph device array into a cudf [`Column`].
    ///
    /// Performs a device-to-device copy. The returned column owns its memory
    /// independently of this view, so the view (and its parent result object)
    /// can be dropped afterwards.
    ///
    /// Returns an error if the view's [`DataType`] has no cudf equivalent
    /// (i.e. [`DataType::SizeT`]).
    ///
    /// [`Column`]: cudf::Column
    fn to_cudf_column(&self) -> Result<cudf::Column>;
}

impl RawDeviceArrayViewExt for RawDeviceArrayView<'_> {
    fn to_cudf_column(&self) -> Result<cudf::Column> {
        let dtype = self.dtype()?;
        let cudf_type = cugraph_type_to_cudf(dtype).ok_or_else(|| {
            Error::new(ErrorKind::UnsupportedType, "RawDeviceArrayViewExt::to_cudf_column")
                .with_message(format!("cugraph DataType {:?} has no cudf TypeId equivalent", dtype))
        })?;

        // SAFETY: `self.device_ptr()` points to valid GPU memory containing
        // `self.len()` elements of `dtype`, which we have verified maps to
        // `cudf_type`. `Column::from_device_ptr` performs a device-to-device
        // copy, so the returned column is independent of this view.
        let col = unsafe {
            cudf::Column::from_device_ptr(self.device_ptr() as *const std::ffi::c_void, self.len(), cudf_type)
        }
        .map_err(|e| {
            Error::new(ErrorKind::Unexpected, "RawDeviceArrayViewExt::to_cudf_column")
                .with_message(format!("cudf Column::from_device_ptr failed: {}", e))
        })?;

        Ok(col)
    }
}

/// Options for building a graph from a cudf [`TableView`] using column indices.
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
#[derive(Debug, Clone)]
pub struct TableGraphOptions {
    /// Zero-based column index for source vertex IDs.
    pub src_col: usize,
    /// Zero-based column index for destination vertex IDs.
    pub dst_col: usize,
    /// Optional zero-based column index for edge weights.
    pub wgt_col: Option<usize>,
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices (recommended when vertex IDs are not
    /// sequential from 0).
    pub renumber: bool,
}

impl TableGraphOptions {
    /// Create options with source and destination column indices.
    ///
    /// Defaults: no weight column, directed graph, renumber disabled.
    pub fn new(src_col: usize, dst_col: usize) -> Self {
        Self { src_col, dst_col, wgt_col: None, properties: GraphProperties::directed(), renumber: false }
    }

    /// Set the weight column index.
    pub fn weight_col(mut self, col: usize) -> Self {
        self.wgt_col = Some(col);
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
pub struct DataFrameGraphOptions<'a> {
    /// Column name for source vertex IDs.
    pub src_col: &'a str,
    /// Column name for destination vertex IDs.
    pub dst_col: &'a str,
    /// Optional column name for edge weights.
    pub wgt_col: Option<&'a str>,
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices.
    pub renumber: bool,
}

impl<'a> DataFrameGraphOptions<'a> {
    /// Create options with source and destination column names.
    ///
    /// Defaults: no weight column, directed graph, renumber disabled.
    pub fn new(src_col: &'a str, dst_col: &'a str) -> Self {
        Self { src_col, dst_col, wgt_col: None, properties: GraphProperties::directed(), renumber: false }
    }

    /// Set the weight column name.
    pub fn weight_col(mut self, col: &'a str) -> Self {
        self.wgt_col = Some(col);
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
pub struct ParquetGraphOptions<'a> {
    /// File-system path to the Parquet file.
    pub path: &'a str,
    /// Zero-based column index for source vertex IDs.
    pub src_col: usize,
    /// Zero-based column index for destination vertex IDs.
    pub dst_col: usize,
    /// Optional zero-based column index for edge weights.
    pub wgt_col: Option<usize>,
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices.
    pub renumber: bool,
}

impl<'a> ParquetGraphOptions<'a> {
    /// Create options with path and source/destination column indices.
    ///
    /// Defaults: no weight column, directed graph, renumber disabled.
    pub fn new(path: &'a str, src_col: usize, dst_col: usize) -> Self {
        Self { path, src_col, dst_col, wgt_col: None, properties: GraphProperties::directed(), renumber: false }
    }

    /// Set the weight column index.
    pub fn weight_col(mut self, col: usize) -> Self {
        self.wgt_col = Some(col);
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
}

/// Build a cuGraph [`Graph`] from a cudf [`TableView`].
///
/// This is the core function that extracts column views from a table, validates
/// them, and builds a graph. Both [`graph_from_parquet`] and
/// [`graph_from_dataframe`] delegate to this function.
///
/// **Copy semantics**: cuGraph reads the edge data from the table's columns
/// and builds its own internal CSR/CSC representation. The returned [`Graph`]
/// is independent of the input table — the table can be dropped immediately
/// after this function returns.
///
/// The type parameter `T` determines the transposition state of the
/// resulting graph:
///
/// ```ignore
/// use cugraph::{Forward, Transposed, cudf_interop::{graph_from_table_view, TableGraphOptions}};
///
/// // For BFS (requires forward):
/// let opts = TableGraphOptions::new(0, 1).renumber();
/// let graph = graph_from_table_view::<Forward>(&handle, &view, &opts)?;
///
/// // For PageRank (requires transposed):
/// let graph = graph_from_table_view::<Transposed>(&handle, &view, &opts)?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - any column index is out of bounds
/// - selected columns contain null values
/// - column types are incompatible with cugraph
/// - graph construction fails
///
/// [`TableView`]: cudf::TableView
pub fn graph_from_table_view<T: Transposition>(
    handle: &ResourceHandle,
    view: &cudf::TableView<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let src_col = options.src_col;
    let dst_col = options.dst_col;
    let wgt_col = options.wgt_col;
    let props = options.properties;
    let renumber = options.renumber;
    let num_cols = view.num_columns().map_err(|e| {
        Error::new(ErrorKind::Unexpected, "graph_from_table_view")
            .with_message(format!("failed to get column count: {}", e))
    })? as usize;

    // ── Validate column indices ─────────────────────────────────────
    let validate_col_index = |idx: usize, name: &str| -> Result<()> {
        if idx >= num_cols {
            return Err(Error::new(ErrorKind::InvalidInput, "graph_from_table_view")
                .with_message(format!(
                    "{} column index {} is out of bounds (table has {} columns)",
                    name, idx, num_cols
                ))
                .with_context("column_name", name.to_string())
                .with_context("column_index", idx.to_string())
                .with_context("num_columns", num_cols.to_string()));
        }
        Ok(())
    };

    validate_col_index(src_col, "src")?;
    validate_col_index(dst_col, "dst")?;
    if let Some(w) = wgt_col {
        validate_col_index(w, "wgt")?;
    }

    // ── Helper: extract a cugraph RawDeviceArrayView from a table column
    // Rejects nullable columns (graph edges/weights must be fully defined)
    // and accounts for cudf column offsets when computing the device ptr.
    let col_to_cugraph_view = |idx: usize, name: &str| -> Result<RawDeviceArrayView<'_>> {
        let col = view.column(idx as i32).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("failed to get {} column: {}", name, e))
                .with_context("column_index", idx.to_string())
        })?;

        let dt = col.data_type().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("failed to get {} column data type: {}", name, e))
                .with_context("column_index", idx.to_string())
        })?;

        let cudf_type_id = dt.id();
        let cugraph_dtype = cudf_type_to_cugraph(cudf_type_id).ok_or_else(|| {
            Error::new(ErrorKind::UnsupportedType, "graph_from_table_view")
                .with_message(format!(
                    "{} column has cudf type {:?} which has no cugraph equivalent",
                    name, cudf_type_id
                ))
                .with_context("column_index", idx.to_string())
        })?;

        let null_count = col.null_count().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("failed to get {} column null count: {}", name, e))
                .with_context("column_index", idx.to_string())
        })?;

        if null_count != 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "graph_from_table_view")
                .with_message(format!(
                    "{} column contains {} null value(s); nulls are not supported for edge lists",
                    name, null_count
                ))
                .with_context("column_name", name.to_string())
                .with_context("column_index", idx.to_string())
                .with_context("null_count", null_count.to_string()));
        }

        let base_ptr = col.device_data_ptr().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("failed to get {} column device pointer: {}", name, e))
                .with_context("column_index", idx.to_string())
        })? as *const u8;

        let len = col.len().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("failed to get {} column length: {}", name, e))
                .with_context("column_index", idx.to_string())
        })?;

        let offset = col.offset().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("failed to get {} column offset: {}", name, e))
                .with_context("column_index", idx.to_string())
        })?;

        if offset < 0 {
            return Err(Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("{} column returned a negative offset {}", name, offset))
                .with_context("column_index", idx.to_string()));
        }
        let offset = offset as usize;

        let byte_offset = offset.checked_mul(cugraph_dtype.size_of()).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, "graph_from_table_view")
                .with_message(format!("{} column offset is too large", name))
                .with_context("column_index", idx.to_string())
                .with_context("offset", offset.to_string())
                .with_context("dtype_size", cugraph_dtype.size_of().to_string())
        })?;

        let ptr = if base_ptr.is_null() {
            base_ptr as *mut std::ffi::c_void
        } else {
            // SAFETY: `base_ptr` is the start of the cudf device buffer for this
            // column, and `byte_offset` is derived from the column's logical offset
            // and element size.
            unsafe { base_ptr.add(byte_offset) as *mut std::ffi::c_void }
        };

        // SAFETY: `ptr` points at the active (offset-adjusted) region of
        // the cudf column's device buffer for `len` elements of `cugraph_dtype`.
        // The table outlives all intermediate views in this scope.
        Ok(unsafe { RawDeviceArrayView::from_raw(ptr, len, cugraph_dtype) })
    };

    let src_cg = col_to_cugraph_view(src_col, "src")?;
    let dst_cg = col_to_cugraph_view(dst_col, "dst")?;
    let wgt_cg = wgt_col.map(|w| col_to_cugraph_view(w, "wgt")).transpose()?;

    // ── Build the graph ─────────────────────────────────────────────
    let mut builder = GraphBuilder::from_raw_views(props, &src_cg, &dst_cg).renumber_flag(renumber);
    if let Some(ref w) = wgt_cg {
        builder = builder.weights_raw(w);
    }
    builder.build_inner(handle, T::store_transposed())
}

/// Build a cuGraph [`Graph`] from a cudf [`DataFrame`] using column names.
///
/// Resolves column names to indices, copies the DataFrame into a [`Table`],
/// and delegates to [`graph_from_table_view`].
///
/// **Copy semantics**: this performs a device-to-device copy of the
/// DataFrame's columns into a table, then cuGraph builds its internal
/// representation from that table. The returned graph is independent of
/// the input DataFrame.
///
/// The type parameter `T` determines the transposition state of the
/// resulting graph. See [`graph_from_table_view`] for details.
///
/// # Errors
///
/// Returns an error if any column name is not found in the DataFrame, or if
/// [`graph_from_table_view`] fails.
///
/// [`DataFrame`]: cudf::DataFrame
/// [`Table`]: cudf::Table
pub fn graph_from_dataframe<T: Transposition>(
    handle: &ResourceHandle,
    df: &cudf::DataFrame,
    options: &DataFrameGraphOptions<'_>,
) -> Result<Graph<T>> {
    let resolve = |name: &str, role: &str| -> Result<usize> {
        df.column_index(name).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, "graph_from_dataframe")
                .with_message(format!("{} column \"{}\" not found in DataFrame", role, name))
                .with_context("column_name", name.to_string())
        })
    };

    let src_idx = resolve(options.src_col, "src")?;
    let dst_idx = resolve(options.dst_col, "dst")?;
    let wgt_idx = options.wgt_col.map(|name| resolve(name, "wgt")).transpose()?;

    let table = df.to_table_copy().map_err(|e| {
        Error::new(ErrorKind::Unexpected, "graph_from_dataframe")
            .with_message(format!("failed to build table from DataFrame: {}", e))
    })?;
    let view = table.view().map_err(|e| {
        Error::new(ErrorKind::Unexpected, "graph_from_dataframe")
            .with_message(format!("failed to get table view: {}", e))
    })?;

    let table_opts = TableGraphOptions {
        src_col: src_idx,
        dst_col: dst_idx,
        wgt_col: wgt_idx,
        properties: options.properties,
        renumber: options.renumber,
    };
    graph_from_table_view::<T>(handle, &view, &table_opts)
}

/// A graph loaded from an external data source along with its backing table.
///
/// The [`table`](LoadedGraph::table) field holds the source edge-list data in
/// GPU memory. The [`graph`](LoadedGraph::graph) field holds the constructed
/// graph with its own internal CSR/CSC representation.
///
/// **Ownership**: the graph's internal representation is independent of the
/// table — it was built by copying data from the table's columns during
/// construction. The table is retained so callers can reuse the loaded
/// edge-list data for additional queries or inspection without re-reading
/// the source file.
pub struct LoadedGraph<T: Transposition> {
    /// The constructed graph.
    pub graph: Graph<T>,
    /// The source edge-list table in GPU memory.
    pub table: cudf::Table,
}

impl<T: Transposition> std::fmt::Debug for LoadedGraph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedGraph").field("graph", &self.graph).finish_non_exhaustive()
    }
}

/// Read a Parquet edge-list and build a cuGraph [`Graph`] on the GPU.
///
/// This convenience helper eliminates the ~20 lines of boilerplate that every
/// Parquet-to-Graph workflow repeats: read table → extract column views →
/// create DeviceSlices → convert to cugraph views → build Graph.
///
/// **Copy semantics**: this function performs two copies:
/// 1. **Disk → GPU**: the Parquet file is read into a cudf [`Table`] in GPU
///    memory (managed by cudf's Parquet reader).
/// 2. **Table → Graph**: cuGraph reads the table's column data and builds its
///    own internal CSR/CSC representation.
///
/// The returned [`LoadedGraph`] bundles both the graph and the table. The
/// graph is independent of the table — the table is retained so callers can
/// reuse the loaded edge-list data without re-reading the Parquet file.
///
/// The type parameter `T` determines the transposition state of the
/// resulting graph. See [`graph_from_table_view`] for details.
///
/// [`Table`]: cudf::Table
///
/// # Errors
///
/// Returns an error if:
/// - the Parquet file cannot be read
/// - any column index is out of bounds
/// - selected columns contain null values
/// - column types are incompatible with cugraph
/// - graph construction fails
///
/// # Example
///
/// ```no_run
/// use cugraph::{
///     Forward, ResourceHandle,
///     cudf_interop::{ParquetGraphOptions, graph_from_parquet},
/// };
///
/// let handle = ResourceHandle::new().unwrap();
///
/// let opts = ParquetGraphOptions::new("edges.parquet", 0, 1)
///     .weight_col(2)
///     .renumber();
///
/// let loaded = graph_from_parquet::<Forward>(&handle, &opts).unwrap();
/// let mut graph = loaded.graph;
/// let _table = loaded.table;
/// ```
pub fn graph_from_parquet<T: Transposition>(
    handle: &ResourceHandle,
    options: &ParquetGraphOptions<'_>,
) -> Result<LoadedGraph<T>> {
    // ── Read the Parquet file into GPU memory ───────────────────────────
    let reader_opts = parquet::ParquetReaderOptions::new(options.path);

    let result = parquet::read_parquet_with_metadata(reader_opts).map_err(|e| {
        Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
            .with_message(format!("failed to read Parquet file: {}", e))
            .with_context("path", options.path.to_string())
    })?;
    let table = result.table;

    // Build the graph inside an inner scope so that the TableView and all
    // intermediate RawDeviceArrayViews are dropped before we return `table`.
    let graph = {
        let view = table.view().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                .with_message(format!("failed to get table view: {}", e))
        })?;

        let table_opts = TableGraphOptions {
            src_col: options.src_col,
            dst_col: options.dst_col,
            wgt_col: options.wgt_col,
            properties: options.properties,
            renumber: options.renumber,
        };
        graph_from_table_view::<T>(handle, &view, &table_opts)?
    };

    Ok(LoadedGraph { graph, table })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All cugraph DataType variants that have a cudf equivalent (everything except SizeT).
    const MAPPED_DATA_TYPES: &[DataType] = &[
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Uint8,
        DataType::Uint16,
        DataType::Uint32,
        DataType::Uint64,
        DataType::Float32,
        DataType::Float64,
        DataType::Bool,
    ];

    /// All cudf TypeId variants that have a cugraph equivalent.
    const MAPPED_TYPE_IDS: &[TypeId] = &[
        TypeId::Int8,
        TypeId::Int16,
        TypeId::Int32,
        TypeId::Int64,
        TypeId::UInt8,
        TypeId::UInt16,
        TypeId::UInt32,
        TypeId::UInt64,
        TypeId::Float32,
        TypeId::Float64,
        TypeId::Bool8,
    ];

    #[test]
    fn cugraph_to_cudf_roundtrip() {
        // For every mapped cugraph DataType, converting to cudf and back yields the original.
        for &dt in MAPPED_DATA_TYPES {
            let cudf_tid =
                cugraph_type_to_cudf(dt).unwrap_or_else(|| panic!("cugraph_type_to_cudf returned None for {:?}", dt));
            let back = cudf_type_to_cugraph(cudf_tid)
                .unwrap_or_else(|| panic!("cudf_type_to_cugraph returned None for {:?}", cudf_tid));
            assert_eq!(dt, back, "round-trip failed: {:?} → {:?} → {:?}", dt, cudf_tid, back);
        }
    }

    #[test]
    fn cudf_to_cugraph_roundtrip() {
        // For every mapped cudf TypeId, converting to cugraph and back yields the original.
        for &tid in MAPPED_TYPE_IDS {
            let cugraph_dt =
                cudf_type_to_cugraph(tid).unwrap_or_else(|| panic!("cudf_type_to_cugraph returned None for {:?}", tid));
            let back = cugraph_type_to_cudf(cugraph_dt)
                .unwrap_or_else(|| panic!("cugraph_type_to_cudf returned None for {:?}", cugraph_dt));
            assert_eq!(tid, back, "round-trip failed: {:?} → {:?} → {:?}", tid, cugraph_dt, back);
        }
    }

    #[test]
    fn size_t_has_no_cudf_equivalent() {
        assert_eq!(cugraph_type_to_cudf(DataType::SizeT), None);
    }

    #[test]
    fn unsupported_cudf_types_return_none() {
        // Non-numeric cudf types should not map to cugraph.
        let unsupported = [
            TypeId::Empty,
            TypeId::TimestampDays,
            TypeId::TimestampSeconds,
            TypeId::TimestampMilliseconds,
            TypeId::TimestampMicroseconds,
            TypeId::TimestampNanoseconds,
            TypeId::DurationDays,
            TypeId::DurationSeconds,
            TypeId::DurationMilliseconds,
            TypeId::DurationMicroseconds,
            TypeId::DurationNanoseconds,
            TypeId::Dictionary32,
            TypeId::String,
            TypeId::List,
            TypeId::Decimal32,
            TypeId::Decimal64,
            TypeId::Decimal128,
            TypeId::Struct,
        ];
        for tid in unsupported {
            assert_eq!(cudf_type_to_cugraph(tid), None, "expected None for unsupported cudf TypeId {:?}", tid);
        }
    }
}
