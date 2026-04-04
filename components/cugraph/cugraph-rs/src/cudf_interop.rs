//! Interop between cudf and cugraph types.
//!
//! # Memory ownership model
//!
//! Functions in this module differ in whether they copy data or create
//! zero-copy views. The distinction matters for lifetime management:
//!
//! ## Zero-copy view paths
//!
//! - [`device_array_view_from_slice()`] — wraps a cudf column's GPU memory as a
//!   typed cugraph array view without copying, with a runtime type check.
//! - [`graph_from_series()`] — reads Series column pointers directly during
//!   graph construction. No intermediate table copy.
//! - [`graph_from_dataframe()`] — reads DataFrame column pointers directly
//!   during graph construction. No intermediate table copy.
//! - [`graph_from_table_view()`] — creates views into the cudf table's
//!   column buffers during construction.
//!
//! ## Copy-based paths
//!
//! - [`IntoSeries`] trait — converts cugraph result arrays to cudf Series.
//!   Performs a device-to-device copy (unavoidable: cugraph owns result memory).
//! - [`graph_from_parquet()`] — reads Parquet from disk into GPU memory
//!   (I/O + copy). Returns a [`LoadedGraph`] that owns both the [`Graph`]
//!   and the backing table.
//!
//! ## Summary
//!
//! | Function | Copy? | Lifetime constraint |
//! |----------|-------|---------------------|
//! | [`device_array_view_from_slice`] | No | View borrows slice |
//! | [`graph_from_series`] | No (input) | Series must outlive build |
//! | [`graph_from_dataframe`] | No (input) | DataFrame must outlive build |
//! | [`graph_from_table_view`] | No (input) | Table must outlive build |
//! | [`IntoSeries::to_series`] | Yes (D2D) | Series is independent |
//! | [`graph_from_parquet`] | Yes | [`LoadedGraph`] owns both |
//!
//! [`DeviceSlice`]: cudf::DeviceSlice

use std::ffi::c_void;

use cudf::{TypeId, core::DeviceSlice, io::parquet};

use crate::{
    array::RawDeviceArrayView,
    centrality::{BetweennessCentralityResult, PageRankResult},
    error::{Error, ErrorKind, Result},
    graph::{Graph, GraphBuilder, Transposition},
    handle::ResourceHandle,
    layout::LayoutResult,
    structure::{
        degrees::{DegreeValues, DegreesResult, InDegreesResult, OutDegreesResult},
        typed_results::{Distances, ScoreValues, VertexIds},
    },
    traversal::bfs::BfsResult,
    types::{CugraphElement, DataType, GraphProperties},
};

// ═══════════════════════════════════════════════════════════════════════════
// Type conversions
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// Checked DeviceSlice → DeviceArrayView — zero-copy bridge
// ═══════════════════════════════════════════════════════════════════════════

/// Create a typed [`DeviceArrayView`] from a runtime-typed cudf [`DeviceSlice`],
/// verifying that the slice's element type matches `T` at runtime.
///
/// This is the safe zero-copy bridge between cudf and cugraph. The returned
/// view borrows the slice's GPU memory without copying.
///
/// # Errors
///
/// Returns [`ErrorKind::UnsupportedType`] if the slice's cudf type has no
/// cugraph equivalent, or if the mapped cugraph type does not match `T`.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::device_array_view_from_slice;
///
/// let runtime = cudf::Runtime::builder().build().unwrap();
/// let series =
///     cudf::Series::new_with_runtime(&runtime, "v", &[0i32, 1, 2]).unwrap();
/// let col = series.as_column_view().unwrap();
/// let slice = col.device_slice::<i32>().unwrap();
///
/// let view = device_array_view_from_slice::<i32>(&slice).unwrap();
/// assert_eq!(view.len(), 3);
/// ```
pub fn device_array_view_from_slice<'a, T: CugraphElement>(
    slice: &'a DeviceSlice<'a>,
) -> Result<crate::array::DeviceArrayView<'a, T>> {
    let cudf_tid = slice.type_id();
    let cugraph_dtype = cudf_type_to_cugraph(cudf_tid).ok_or_else(|| {
        Error::new(ErrorKind::UnsupportedType, "device_array_view_from_slice")
            .with_message(format!("cudf TypeId {:?} has no cugraph equivalent", cudf_tid))
    })?;

    let expected = T::data_type();
    if cugraph_dtype != expected {
        return Err(Error::new(ErrorKind::UnsupportedType, "device_array_view_from_slice")
            .with_message(format!(
                "dtype mismatch: slice contains {:?} but requested type maps to {:?}",
                cugraph_dtype, expected
            ))
            .with_context("slice_dtype", format!("{:?}", cugraph_dtype))
            .with_context("requested_dtype", format!("{:?}", expected)));
    }

    // SAFETY: we have verified that the slice's runtime dtype matches T.
    // DeviceSlice guarantees a valid device pointer for its lifetime.
    let raw = unsafe { crate::array::RawDeviceArrayView::from_raw(slice.as_ptr(), slice.len(), expected) };
    Ok(crate::array::DeviceArrayView::from_raw_unchecked(raw))
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal extension traits (pub(crate))
// ═══════════════════════════════════════════════════════════════════════════

/// Extension trait for converting a cugraph [`RawDeviceArrayView`] into a cudf [`Column`].
///
/// [`Column`]: cudf::Column
pub(crate) trait RawDeviceArrayViewExt {
    /// Convert this cugraph device array into a cudf [`Column`].
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
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
        let col = unsafe { cudf::Column::from_device_ptr(self.device_ptr() as *const c_void, self.len(), cudf_type) }
            .map_err(|e| {
            Error::new(ErrorKind::Unexpected, "RawDeviceArrayViewExt::to_cudf_column")
                .with_message(format!("cudf Column::from_device_ptr failed: {}", e))
        })?;

        Ok(col)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IntoSeries — public extension trait for result → cudf Series conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Extension trait for converting cugraph GPU arrays to cudf [`Series`].
///
/// Device-to-device copy: unavoidable because the cugraph C API owns result
/// memory and frees it on drop. Matches Python cudf/cugraph behavior.
///
/// [`Series`]: cudf::Series
pub trait IntoSeries {
    /// Convert this GPU array to a named cudf [`Series`].
    ///
    /// [`Series`]: cudf::Series
    fn to_series(&self, name: &str) -> Result<cudf::Series>;
}

impl IntoSeries for VertexIds<'_> {
    fn to_series(&self, name: &str) -> Result<cudf::Series> {
        let raw = match self {
            Self::I32(v) => &v.inner,
            Self::I64(v) => &v.inner,
        };
        let col = raw.to_cudf_column()?;
        cudf::Series::from_column(col).map(|s| s.rename(name)).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "VertexIds::to_series")
                .with_message(format!("failed to create Series: {}", e))
        })
    }
}

impl IntoSeries for ScoreValues<'_> {
    fn to_series(&self, name: &str) -> Result<cudf::Series> {
        let raw = match self {
            Self::F32(v) => &v.inner,
            Self::F64(v) => &v.inner,
        };
        let col = raw.to_cudf_column()?;
        cudf::Series::from_column(col).map(|s| s.rename(name)).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "ScoreValues::to_series")
                .with_message(format!("failed to create Series: {}", e))
        })
    }
}

impl IntoSeries for Distances<'_> {
    fn to_series(&self, name: &str) -> Result<cudf::Series> {
        let raw = match self {
            Self::I32(v) => &v.inner,
            Self::I64(v) => &v.inner,
            Self::F32(v) => &v.inner,
            Self::F64(v) => &v.inner,
        };
        let col = raw.to_cudf_column()?;
        cudf::Series::from_column(col).map(|s| s.rename(name)).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "Distances::to_series")
                .with_message(format!("failed to create Series: {}", e))
        })
    }
}

impl IntoSeries for DegreeValues<'_> {
    fn to_series(&self, name: &str) -> Result<cudf::Series> {
        let raw = match self {
            Self::I32(v) => &v.inner,
            Self::I64(v) => &v.inner,
        };
        let col = raw.to_cudf_column()?;
        cudf::Series::from_column(col).map(|s| s.rename(name)).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "DegreeValues::to_series")
                .with_message(format!("failed to create Series: {}", e))
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TryFrom impls — result structs → cudf::DataFrame
// ═══════════════════════════════════════════════════════════════════════════

impl TryFrom<&PageRankResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &PageRankResult) -> Result<Self> {
        let vert = result.vertex_ids()?.to_series("vertex")?;
        let vals = result.scores()?.to_series("value")?;
        cudf::DataFrame::from_series([("vertex", vert), ("value", vals)]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&PageRankResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl TryFrom<&BetweennessCentralityResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &BetweennessCentralityResult) -> Result<Self> {
        let vert = result.vertex_ids()?.to_series("vertex")?;
        let vals = result.scores()?.to_series("value")?;
        cudf::DataFrame::from_series([("vertex", vert), ("value", vals)]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&BetweennessCentralityResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl TryFrom<&BfsResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &BfsResult) -> Result<Self> {
        let vertices_raw = result.vertices_raw()?;
        let distances_raw = result.distances_raw()?;
        let expected_len = vertices_raw.len();

        let vert_col = vertices_raw.to_cudf_column()?;
        let dist_col = distances_raw.to_cudf_column()?;

        let (names, columns) = match result.predecessors_raw() {
            Some(preds) if preds.len() == expected_len => {
                let pred_col = preds.to_cudf_column()?;
                (vec!["vertex".into(), "distance".into(), "predecessor".into()], vec![vert_col, dist_col, pred_col])
            },
            Some(preds) => {
                return Err(Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>").with_message(format!(
                    "predecessor length {} does not match vertex length {}",
                    preds.len(),
                    expected_len
                )));
            },
            None => (vec!["vertex".into(), "distance".into()], vec![vert_col, dist_col]),
        };

        cudf::DataFrame::new(names, columns).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl TryFrom<&LayoutResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &LayoutResult) -> Result<Self> {
        let vert_col = result.vertices_raw()?.to_cudf_column()?;
        let x_col = result.x_raw()?.to_cudf_column()?;
        let y_col = result.y_raw()?.to_cudf_column()?;

        cudf::DataFrame::new(vec!["vertex".into(), "x".into(), "y".into()], vec![vert_col, x_col, y_col]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&LayoutResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl TryFrom<&DegreesResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &DegreesResult) -> Result<Self> {
        let vert = result.vertex_ids()?.to_series("vertex")?;
        let in_deg = result.in_degrees()?.to_series("in_degree")?;
        let out_deg = result.out_degrees()?.to_series("out_degree")?;
        cudf::DataFrame::from_series([("vertex", vert), ("in_degree", in_deg), ("out_degree", out_deg)]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&DegreesResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl TryFrom<&InDegreesResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &InDegreesResult) -> Result<Self> {
        let vert = result.vertex_ids()?.to_series("vertex")?;
        let in_deg = result.in_degrees()?.to_series("in_degree")?;
        cudf::DataFrame::from_series([("vertex", vert), ("in_degree", in_deg)]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&InDegreesResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl TryFrom<&OutDegreesResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &OutDegreesResult) -> Result<Self> {
        let vert = result.vertex_ids()?.to_series("vertex")?;
        let out_deg = result.out_degrees()?.to_series("out_degree")?;
        cudf::DataFrame::from_series([("vertex", vert), ("out_degree", out_deg)]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&OutDegreesResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction option types
// ═══════════════════════════════════════════════════════════════════════════

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
pub struct SeriesGraphOptions {
    /// Graph properties (directed/undirected, multigraph).
    pub properties: GraphProperties,
    /// Whether to renumber vertices.
    pub renumber: bool,
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

// ═══════════════════════════════════════════════════════════════════════════
// Helper: extract a RawDeviceArrayView from a cudf ColumnView
// ═══════════════════════════════════════════════════════════════════════════

/// Extract a cugraph [`RawDeviceArrayView`] from a cudf [`ColumnView`],
/// validating nulls, dtype, and offset.
///
/// [`ColumnView`]: cudf::ColumnView
fn column_view_to_raw<'a>(
    col: &cudf::ColumnView<'a>,
    name: &str,
    operation: &'static str,
) -> Result<RawDeviceArrayView<'a>> {
    let dt = col.data_type().map_err(|e| {
        Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("failed to get {} column data type: {}", name, e))
    })?;

    let cudf_type_id = dt.id();
    let cugraph_dtype = cudf_type_to_cugraph(cudf_type_id).ok_or_else(|| {
        Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("{} column has cudf type {:?} which has no cugraph equivalent", name, cudf_type_id))
    })?;

    let null_count = col.null_count().map_err(|e| {
        Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("failed to get {} column null count: {}", name, e))
    })?;

    if null_count != 0 {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!(
                "{} column contains {} null value(s); nulls are not supported for edge lists",
                name, null_count
            ))
            .with_context("column_name", name.to_string())
            .with_context("null_count", null_count.to_string()));
    }

    let base_ptr = col.device_data_ptr().map_err(|e| {
        Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("failed to get {} column device pointer: {}", name, e))
    })? as *const u8;

    let len = col.len().map_err(|e| {
        Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("failed to get {} column length: {}", name, e))
    })?;

    let offset = col.offset().map_err(|e| {
        Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("failed to get {} column offset: {}", name, e))
    })?;

    if offset < 0 {
        return Err(Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("{} column returned a negative offset {}", name, offset)));
    }
    let offset = offset as usize;

    let byte_offset = offset.checked_mul(cugraph_dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!("{} column offset is too large", name))
            .with_context("offset", offset.to_string())
            .with_context("dtype_size", cugraph_dtype.size_of().to_string())
    })?;

    let ptr = if base_ptr.is_null() {
        base_ptr as *mut c_void
    } else {
        // SAFETY: `base_ptr` is the start of the cudf device buffer for this
        // column, and `byte_offset` is derived from the column's logical offset
        // and element size.
        unsafe { base_ptr.add(byte_offset) as *mut c_void }
    };

    // SAFETY: `ptr` points at the active (offset-adjusted) region of
    // the cudf column's device buffer for `len` elements of `cugraph_dtype`.
    Ok(unsafe { RawDeviceArrayView::from_raw(ptr, len, cugraph_dtype) })
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction functions
// ═══════════════════════════════════════════════════════════════════════════

/// Build a cuGraph [`Graph`] from a cudf [`TableView`].
///
/// This is the core function that extracts column views from a table, validates
/// them, and builds a graph. [`graph_from_parquet`] delegates to this function.
///
/// **Zero-copy input**: cuGraph reads the edge data directly from the table's
/// column device pointers and builds its own internal CSR/CSC representation.
/// The returned [`Graph`] is independent of the input table.
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
    let col_to_cugraph_view = |idx: usize, name: &str| -> Result<RawDeviceArrayView<'_>> {
        let col = view.column(idx as i32).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_table_view")
                .with_message(format!("failed to get {} column: {}", name, e))
                .with_context("column_index", idx.to_string())
        })?;
        column_view_to_raw(&col, name, "graph_from_table_view")
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

/// Map a cudf error from `series_view()` to a cugraph error with the correct
/// kind and retry classification.
///
/// `series_view()` can fail for multiple reasons: column not found (caller
/// error), FFI/GPU failures (runtime error). Mapping all failures to
/// `InvalidInput` would misclassify transient GPU errors as permanent.
fn map_series_view_error(e: cudf::Error, col_label: &str, operation: &'static str) -> Error {
    match e.kind() {
        cudf::ErrorKind::InvalidOperation => {
            // Column not found — caller input error, permanent.
            Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("{} column: {}", col_label, e))
                .with_context("column_label", col_label.to_string())
                .set_source(e)
        },
        cudf::ErrorKind::OutOfMemory => Error::new(ErrorKind::OutOfMemory, operation)
            .with_message(format!("{} column: {}", col_label, e))
            .with_context("column_label", col_label.to_string())
            .set_temporary()
            .set_source(e),
        cudf::ErrorKind::CudaError => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("{} column: {}", col_label, e))
            .with_context("column_label", col_label.to_string())
            .set_temporary()
            .set_source(e),
        cudf::ErrorKind::FatalCudaError => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("{} column: {}", col_label, e))
            .with_context("column_label", col_label.to_string())
            .set_persistent()
            .set_source(e),
        _ => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("{} column: {}", col_label, e))
            .with_context("column_label", col_label.to_string())
            .set_source(e),
    }
}

/// Build a cuGraph [`Graph`] from a cudf [`DataFrame`] using column names.
///
/// **Zero-copy input**: resolves column names, obtains column views from the
/// DataFrame, and passes device pointers directly to cuGraph. No intermediate
/// table copy is performed.
///
/// # Errors
///
/// Returns an error if any column name is not found in the DataFrame, or if
/// graph construction fails.
///
/// [`DataFrame`]: cudf::DataFrame
pub fn graph_from_dataframe<T: Transposition>(
    handle: &ResourceHandle,
    df: &cudf::DataFrame,
    options: &DataFrameGraphOptions<'_>,
) -> Result<Graph<T>> {
    let operation = "graph_from_dataframe";

    // Resolve column names to SeriesViews (zero-copy borrows from DataFrame)
    let src_sv = df.series_view(options.src_col).map_err(|e| map_series_view_error(e, "src", operation))?;
    let dst_sv = df.series_view(options.dst_col).map_err(|e| map_series_view_error(e, "dst", operation))?;
    let wgt_sv = options
        .wgt_col
        .map(|name| df.series_view(name).map_err(|e| map_series_view_error(e, "wgt", operation)))
        .transpose()?;

    // SeriesView derefs to ColumnView — extract raw device array views
    let src_cg = column_view_to_raw(&src_sv, "src", operation)?;
    let dst_cg = column_view_to_raw(&dst_sv, "dst", operation)?;
    let wgt_cg = wgt_sv.as_ref().map(|sv| column_view_to_raw(sv, "wgt", operation)).transpose()?;

    let mut builder =
        GraphBuilder::from_raw_views(options.properties, &src_cg, &dst_cg).renumber_flag(options.renumber);
    if let Some(ref w) = wgt_cg {
        builder = builder.weights_raw(w);
    }
    builder.build_inner(handle, T::store_transposed())
}

/// Build a cuGraph [`Graph`] from cudf [`Series`] (src vertices, dst vertices,
/// optional weights).
///
/// **Zero-copy input**: the Series' GPU memory is read directly during graph
/// construction. The Graph's internal CSR/CSC is independent after construction
/// — the Series can be dropped afterward.
///
/// # Example
///
/// ```no_run
/// use cugraph::{
///     Forward, ResourceHandle,
///     cudf_interop::{SeriesGraphOptions, graph_from_series},
/// };
///
/// let runtime = cudf::Runtime::builder().build().unwrap();
/// let handle =
///     ResourceHandle::new_on_device(runtime.device().ordinal() as i32)
///         .unwrap();
/// let src =
///     cudf::Series::new_with_runtime(&runtime, "src", &[0i32, 1, 2]).unwrap();
/// let dst =
///     cudf::Series::new_with_runtime(&runtime, "dst", &[1i32, 2, 0]).unwrap();
///
/// let opts = SeriesGraphOptions::default().renumber();
/// let graph =
///     graph_from_series::<Forward>(&handle, &src, &dst, None, &opts).unwrap();
/// ```
///
/// [`Series`]: cudf::Series
pub fn graph_from_series<T: Transposition>(
    handle: &ResourceHandle,
    src: &cudf::Series,
    dst: &cudf::Series,
    weights: Option<&cudf::Series>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_series";

    // ── Provenance validation ────────────────────────────────────────
    if !src.same_provenance(dst) {
        return Err(Error::new(ErrorKind::InvalidInput, operation).with_message(
            "src and dst series have different retained provenance \
             (CUDA stream / memory resource); combine them in a single \
             cudf Runtime or DataFrame before building a graph",
        ));
    }
    if let Some(wgt) = weights {
        if !src.same_provenance(wgt) {
            return Err(Error::new(ErrorKind::InvalidInput, operation).with_message(
                "weights series has different retained provenance from \
                 src/dst (CUDA stream / memory resource); combine them in \
                 a single cudf Runtime or DataFrame before building a graph",
            ));
        }
    }

    let src_cv = src.as_column_view().map_err(|e| {
        Error::new(ErrorKind::Unexpected, operation).with_message(format!("failed to get src column view: {}", e))
    })?;
    let dst_cv = dst.as_column_view().map_err(|e| {
        Error::new(ErrorKind::Unexpected, operation).with_message(format!("failed to get dst column view: {}", e))
    })?;

    let src_cg = column_view_to_raw(&src_cv, "src", operation)?;
    let dst_cg = column_view_to_raw(&dst_cv, "dst", operation)?;

    let wgt_cv = weights
        .map(|w| {
            w.as_column_view().map_err(|e| {
                Error::new(ErrorKind::Unexpected, operation)
                    .with_message(format!("failed to get wgt column view: {}", e))
            })
        })
        .transpose()?;

    let wgt_cg = wgt_cv.as_ref().map(|cv| column_view_to_raw(cv, "wgt", operation)).transpose()?;

    let mut builder =
        GraphBuilder::from_raw_views(options.properties, &src_cg, &dst_cg).renumber_flag(options.renumber);
    if let Some(ref w) = wgt_cg {
        builder = builder.weights_raw(w);
    }
    builder.build_inner(handle, T::store_transposed())
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
pub fn graph_from_parquet<T: Transposition>(
    handle: &ResourceHandle,
    options: &ParquetGraphOptions<'_>,
) -> Result<LoadedGraph<T>> {
    let reader_opts = parquet::ParquetReaderOptions::new(options.path);

    let result = parquet::read_parquet_with_metadata(reader_opts).map_err(|e| {
        Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
            .with_message(format!("failed to read Parquet file: {}", e))
            .with_context("path", options.path.to_string())
    })?;
    let table = result.table;

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

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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

    // ── map_series_view_error tests ─────────────────────────────────────

    #[test]
    fn map_series_view_error_column_not_found() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::InvalidOperation, "column 'foo' not found");
        let err = map_series_view_error(cudf_err, "src", "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.is_permanent(), "column-not-found should be permanent");
        assert!(err.message().contains("src column"), "message: {}", err.message());
    }

    #[test]
    fn map_series_view_error_cuda_error() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::CudaError, "device launch failed");
        let err = map_series_view_error(cudf_err, "dst", "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "CUDA error should be temporary");
    }

    #[test]
    fn map_series_view_error_out_of_memory() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::OutOfMemory, "alloc failed");
        let err = map_series_view_error(cudf_err, "wgt", "test_op");
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.is_temporary(), "OOM should be temporary");
    }

    #[test]
    fn map_series_view_error_unknown() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::Unknown, "something else");
        let err = map_series_view_error(cudf_err, "src", "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_permanent(), "unknown should be permanent (default)");
    }

    #[test]
    fn unsupported_cudf_types_return_none() {
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
