//! Interop between cudf and cugraph types.
//!
//! # Memory ownership model
//!
//! Functions in this module differ in whether they copy data or create
//! zero-copy views. The distinction matters for lifetime management:
//!
//! ## Primary shared-contract paths
//!
//! - [`graph_from_prepared_columns()`] — consumes single-GPU prepared-device
//!   exports via explicit read-lease activation.
//! - [`graph_from_prepared_table_export()`] — same as above for a prepared
//!   table export plus explicit column selection.
//!
//! ## Compatibility zero-copy view paths
//!
//! - [`device_array_view_from_slice()`] — wraps a cudf column's GPU memory as a
//!   typed cugraph array view without copying, with a runtime type check.
//! - [`graph_from_borrowed_columns()`] — consumes the shared RAPIDS borrowed
//!   ABI directly.
//! - [`graph_from_series()`], [`graph_from_dataframe()`], and
//!   [`graph_from_table_view()`] — convenience producers that normalize cudf
//!   objects into the borrowed ABI at the API boundary.
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
//! | [`graph_from_prepared_columns`] | No (input) | Producer export stays alive until explicit lease completion |
//! | [`graph_from_prepared_table_export`] | No (input) | Producer export stays alive until explicit lease completion |
//! | [`graph_from_borrowed_columns`] | No (input) | Borrowed descriptors must outlive build |
//! | [`graph_from_series`] | No (input) | Series must outlive build |
//! | [`graph_from_dataframe`] | No (input) | DataFrame must outlive build |
//! | [`graph_from_table_view`] | No (input) | Table must outlive build |
//! | [`IntoSeries::to_series`] | Yes (D2D) | Series is independent |
//! | [`graph_from_parquet`] | Yes | [`LoadedGraph`] owns both |
//!
//! # Single-GPU contract position
//!
//! For new single-GPU consumer integrations, prefer the prepared-device export
//! path because it keeps read-start, completion, and producer liveness explicit
//! at the shared contract layer.
//!
//! The borrowed graph-construction entry points remain supported as
//! compatibility zero-copy paths, especially for callers that already operate
//! on explicit borrowed descriptors and only need synchronous host-side fencing.
//!
//! # Borrowed compatibility policy
//!
//! Borrowed graph construction is explicit-only and same-device-only:
//!
//! - cudf producers must expose explicit execution provenance
//! - ambient/default-only cudf objects are rejected
//! - all borrowed inputs must live on the same CUDA device
//! - that device must match the [`ResourceHandle`] device
//! - mixed producer streams on one device are allowed, but they are fenced
//!   conservatively on the host before cuGraph reads borrowed memory
//!
//! [`DeviceSlice`]: cudf::DeviceSlice

use std::{collections::BTreeSet, ffi::c_void};

use cudf::{
    TypeId as CudfTypeId,
    core::{Column, DeviceSlice, Table, TableView},
    io::parquet,
};
use rapids_interop::{
    BorrowedColumnView, BorrowedTableView, ColumnLayout, CompletionAuthority, InteropError, InteropErrorKind,
    LeaseCompletionResult, NullMaskKind, PhysicalValidity, Placement, PreparedColumnExport, PreparedTableExport,
    ReadStartAuthority, TypeId as InteropTypeId,
};

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
pub fn cudf_type_to_cugraph(type_id: CudfTypeId) -> Option<DataType> {
    match type_id {
        CudfTypeId::Int8 => Some(DataType::Int8),
        CudfTypeId::Int16 => Some(DataType::Int16),
        CudfTypeId::Int32 => Some(DataType::Int32),
        CudfTypeId::Int64 => Some(DataType::Int64),
        CudfTypeId::UInt8 => Some(DataType::Uint8),
        CudfTypeId::UInt16 => Some(DataType::Uint16),
        CudfTypeId::UInt32 => Some(DataType::Uint32),
        CudfTypeId::UInt64 => Some(DataType::Uint64),
        CudfTypeId::Float32 => Some(DataType::Float32),
        CudfTypeId::Float64 => Some(DataType::Float64),
        CudfTypeId::Bool8 => Some(DataType::Bool),
        _ => None,
    }
}

fn borrowed_type_to_cugraph(type_id: InteropTypeId) -> Option<DataType> {
    match type_id {
        InteropTypeId::Int8 => Some(DataType::Int8),
        InteropTypeId::Int16 => Some(DataType::Int16),
        InteropTypeId::Int32 => Some(DataType::Int32),
        InteropTypeId::Int64 => Some(DataType::Int64),
        InteropTypeId::UInt8 => Some(DataType::Uint8),
        InteropTypeId::UInt16 => Some(DataType::Uint16),
        InteropTypeId::UInt32 => Some(DataType::Uint32),
        InteropTypeId::UInt64 => Some(DataType::Uint64),
        InteropTypeId::Float32 => Some(DataType::Float32),
        InteropTypeId::Float64 => Some(DataType::Float64),
        InteropTypeId::Bool8 => Some(DataType::Bool),
        _ => None,
    }
}

/// Convert a cugraph [`DataType`] to a cudf [`TypeId`].
///
/// Returns `None` for cugraph types that have no direct cudf equivalent
/// (i.e. [`DataType::SizeT`], which is an internal cugraph sizing type).
pub fn cugraph_type_to_cudf(dtype: DataType) -> Option<CudfTypeId> {
    match dtype {
        DataType::Int8 => Some(CudfTypeId::Int8),
        DataType::Int16 => Some(CudfTypeId::Int16),
        DataType::Int32 => Some(CudfTypeId::Int32),
        DataType::Int64 => Some(CudfTypeId::Int64),
        DataType::Uint8 => Some(CudfTypeId::UInt8),
        DataType::Uint16 => Some(CudfTypeId::UInt16),
        DataType::Uint32 => Some(CudfTypeId::UInt32),
        DataType::Uint64 => Some(CudfTypeId::UInt64),
        DataType::Float32 => Some(CudfTypeId::Float32),
        DataType::Float64 => Some(CudfTypeId::Float64),
        DataType::Bool => Some(CudfTypeId::Bool8),
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
    let raw =
        unsafe { crate::array::RawDeviceArrayView::from_raw(slice.as_ptr() as *mut c_void, slice.len(), expected) };
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
    fn to_cudf_column(&self) -> Result<Column>;
}

impl RawDeviceArrayViewExt for RawDeviceArrayView<'_> {
    fn to_cudf_column(&self) -> Result<Column> {
        let dtype = self.dtype()?;
        let cudf_type = cugraph_type_to_cudf(dtype).ok_or_else(|| {
            Error::new(ErrorKind::UnsupportedType, "RawDeviceArrayViewExt::to_cudf_column")
                .with_message(format!("cugraph DataType {:?} has no cudf TypeId equivalent", dtype))
        })?;

        // SAFETY: `self.device_ptr()` points to valid GPU memory containing
        // `self.len()` elements of `dtype`, which we have verified maps to
        // `cudf_type`. `Column::from_device_ptr` performs a device-to-device
        // copy, so the returned column is independent of this view.
        let col = unsafe { Column::from_device_ptr(self.device_ptr() as *const c_void, self.len(), cudf_type) }
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
        cudf::DataFrame::from_series([vert, vals]).map_err(|e| {
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
        cudf::DataFrame::from_series([vert, vals]).map_err(|e| {
            Error::new(ErrorKind::Unexpected, "TryFrom<&BetweennessCentralityResult>")
                .with_message(format!("failed to build DataFrame: {}", e))
        })
    }
}

impl TryFrom<&BfsResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &BfsResult) -> Result<Self> {
        let vertices = result.vertices_raw()?;
        let distances = result.distances_raw()?;
        let expected_len = vertices.len();

        let vertex_series = cudf::Series::from_column(vertices.to_cudf_column()?)
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>")
                    .with_message(format!("failed to create vertex Series: {}", e))
            })?
            .rename("vertex");
        let distance_series = cudf::Series::from_column(distances.to_cudf_column()?)
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>")
                    .with_message(format!("failed to create distance Series: {}", e))
            })?
            .rename("distance");

        match result.predecessors_raw() {
            Some(preds) if preds.len() == expected_len => {
                let predecessor_series = cudf::Series::from_column(preds.to_cudf_column()?)
                    .map_err(|e| {
                        Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>")
                            .with_message(format!("failed to create predecessor Series: {}", e))
                    })?
                    .rename("predecessor");
                cudf::DataFrame::from_series([vertex_series, distance_series, predecessor_series]).map_err(|e| {
                    Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>")
                        .with_message(format!("failed to build DataFrame: {}", e))
                })
            },
            Some(preds) => {
                return Err(Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>").with_message(format!(
                    "predecessor length {} does not match vertex length {}",
                    preds.len(),
                    expected_len
                )));
            },
            None => cudf::DataFrame::from_series([vertex_series, distance_series]).map_err(|e| {
                Error::new(ErrorKind::Unexpected, "TryFrom<&BfsResult>")
                    .with_message(format!("failed to build DataFrame: {}", e))
            }),
        }
    }
}

impl TryFrom<&LayoutResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &LayoutResult) -> Result<Self> {
        let vertex_series = cudf::Series::from_column(result.vertices_raw()?.to_cudf_column()?)
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "TryFrom<&LayoutResult>")
                    .with_message(format!("failed to create vertex Series: {}", e))
            })?
            .rename("vertex");
        let x_series = cudf::Series::from_column(result.x_raw()?.to_cudf_column()?)
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "TryFrom<&LayoutResult>")
                    .with_message(format!("failed to create x Series: {}", e))
            })?
            .rename("x");
        let y_series = cudf::Series::from_column(result.y_raw()?.to_cudf_column()?)
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "TryFrom<&LayoutResult>")
                    .with_message(format!("failed to create y Series: {}", e))
            })?
            .rename("y");

        cudf::DataFrame::from_series([vertex_series, x_series, y_series]).map_err(|e| {
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
        cudf::DataFrame::from_series([vert, in_deg, out_deg]).map_err(|e| {
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
        cudf::DataFrame::from_series([vert, in_deg]).map_err(|e| {
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
        cudf::DataFrame::from_series([vert, out_deg]).map_err(|e| {
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
// Borrowed interop normalization
// ═══════════════════════════════════════════════════════════════════════════

struct BorrowedEdgeList<'a> {
    src: BorrowedColumnView<'a>,
    dst: BorrowedColumnView<'a>,
    weights: Option<BorrowedColumnView<'a>>,
}

#[derive(Clone, Copy)]
struct PreparedEdgeColumn<'a> {
    column: &'a rapids_interop::SemanticColumn,
    bindings: &'a [rapids_interop::BufferBinding<'a>],
}

struct PreparedEdgeList<'a> {
    src: PreparedEdgeColumn<'a>,
    dst: PreparedEdgeColumn<'a>,
    weights: Option<PreparedEdgeColumn<'a>>,
}

fn map_interop_error(error: InteropError, operation: &'static str) -> Error {
    let mut mapped = match error.kind() {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => Error::new(ErrorKind::UnsupportedType, operation),
        InteropErrorKind::ExecutionBindingUnavailable | InteropErrorKind::ReadinessUnavailable => {
            Error::new(ErrorKind::Unexpected, operation).set_temporary()
        },
        InteropErrorKind::AmbientInteropUnsupported
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::NullMaskUnsupported
        | InteropErrorKind::InvalidView
        | InteropErrorKind::LeaseAlreadyActive => Error::new(ErrorKind::InvalidInput, operation),
    }
    .with_message(error.to_string())
    .set_source(error.clone());

    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }

    mapped
}

#[cfg(test)]
fn map_series_view_error(error: cudf::Error, role: &str, operation: &'static str) -> Error {
    let kind = match error.kind() {
        cudf::ErrorKind::OutOfMemory => ErrorKind::OutOfMemory,
        cudf::ErrorKind::CudaError | cudf::ErrorKind::FatalCudaError | cudf::ErrorKind::Unknown => {
            ErrorKind::Unexpected
        },
        _ => ErrorKind::InvalidInput,
    };

    let mut mapped =
        Error::new(kind, operation).with_message(format!("{role} column view failed: {error}")).set_source(error);

    match kind {
        ErrorKind::OutOfMemory | ErrorKind::Unexpected => {
            mapped = mapped.set_temporary();
        },
        _ => {},
    }

    mapped.with_context("column_role", role.to_string())
}

fn check_cudf_status(status: cudf_sys::cudf_status_t, operation: &'static str) -> Result<()> {
    if status == cudf_sys::CUDF_SUCCESS {
        return Ok(());
    }

    let error = match status {
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => Error::new(ErrorKind::OutOfMemory, operation)
            .with_message(format!("cudf helper failed with status {}", status))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {}", status))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_FATAL_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {}", status))
            .set_persistent(),
        _ => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {}", status)),
    };
    Err(error)
}

struct DeviceGuard {
    previous: i32,
    restore: bool,
}

impl DeviceGuard {
    fn switch_to(target: i32, operation: &'static str) -> Result<Self> {
        let mut current = 0;
        check_cudf_status(unsafe { cudf_sys::cudf_cuda_get_device(&mut current) }, operation)?;

        if current != target {
            check_cudf_status(unsafe { cudf_sys::cudf_cuda_set_device(target) }, operation)?;
        }

        Ok(Self { previous: current, restore: current != target })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.restore {
            let _ = unsafe { cudf_sys::cudf_cuda_set_device(self.previous) };
        }
    }
}

fn borrowed_column_to_raw<'a>(
    column: &BorrowedColumnView<'a>,
    name: &str,
    operation: &'static str,
) -> Result<RawDeviceArrayView<'a>> {
    let cugraph_dtype = borrowed_type_to_cugraph(column.type_id()).ok_or_else(|| {
        Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!(
                "{} column has interop type {:?} which has no cugraph equivalent",
                name,
                column.type_id()
            ))
            .with_context("column_name", name.to_string())
    })?;

    if let NullMaskKind::Present { null_count, .. } = column.null_mask().kind() {
        if *null_count != 0 {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} column contains {} null value(s); nulls are not supported for edge lists",
                    name, null_count
                ))
                .with_context("column_name", name.to_string())
                .with_context("null_count", null_count.to_string()));
        }
    }

    let byte_offset = column.offset().checked_mul(cugraph_dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!("{} column offset is too large", name))
            .with_context("column_name", name.to_string())
            .with_context("offset", column.offset().to_string())
            .with_context("dtype_size", cugraph_dtype.size_of().to_string())
    })?;

    let base_ptr = column.data_ptr() as *const u8;
    let ptr = if base_ptr.is_null() {
        base_ptr as *mut c_void
    } else {
        // SAFETY: the borrowed descriptor has already validated the active range.
        unsafe { base_ptr.add(byte_offset) as *mut c_void }
    };

    // SAFETY: `ptr` points at the active region of the producer-owned device buffer.
    Ok(unsafe { RawDeviceArrayView::from_raw(ptr, column.len(), cugraph_dtype) })
}

fn prepared_profile_supported(profile: rapids_interop::ExportProfile, operation: &'static str) -> Result<()> {
    if profile == rapids_interop::ExportProfile::PreparedDeviceExport {
        return Ok(());
    }

    Err(Error::new(ErrorKind::UnsupportedType, operation)
        .with_message(format!("{operation} requires prepared device export inputs, got {:?}", profile)))
}

fn prepared_column_device<'a>(column: PreparedEdgeColumn<'a>, name: &str, operation: &'static str) -> Result<i32> {
    let values = match column.column.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => {
            return Err(Error::new(ErrorKind::UnsupportedType, operation)
                .with_message(format!(
                    "{} column uses layout {:?}; graph inputs currently require fixed-width prepared exports",
                    name, layout
                ))
                .with_context("column_name", name.to_string()));
        },
    };

    let binding = column.bindings.iter().find(|binding| binding.id() == values).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!(
                "{} column prepared export is missing the values buffer binding referenced by its semantic layout",
                name
            ))
            .with_context("column_name", name.to_string())
            .with_context("buffer_id", values.get().to_string())
    })?;

    match binding.placement() {
        Placement::CudaDevice { device } => Ok(*device),
        Placement::Host => Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!(
                "{} column prepared export is host-materialized; graph construction requires device-resident prepared export inputs",
                name
            ))
            .with_context("column_name", name.to_string())),
    }
}

fn prepared_column_to_raw<'a>(
    column: PreparedEdgeColumn<'a>,
    name: &str,
    operation: &'static str,
) -> Result<RawDeviceArrayView<'a>> {
    let cugraph_dtype = borrowed_type_to_cugraph(column.column.data_type().id()).ok_or_else(|| {
        Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!(
                "{} column has interop type {:?} which has no cugraph equivalent",
                name,
                column.column.data_type().id()
            ))
            .with_context("column_name", name.to_string())
    })?;

    match column.column.validity() {
        PhysicalValidity::NoValidityBuffer => {},
        PhysicalValidity::ValidityMask { null_count: Some(0), .. } => {},
        PhysicalValidity::ValidityMask { null_count: Some(null_count), .. } => {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} column contains {} null value(s); nulls are not supported for edge lists",
                    name, null_count
                ))
                .with_context("column_name", name.to_string())
                .with_context("null_count", null_count.to_string()));
        },
        PhysicalValidity::ValidityMask { null_count: None, .. } => {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} column prepared export does not make null count explicit; graph inputs must prove a null-free active range",
                    name
                ))
                .with_context("column_name", name.to_string()));
        },
    }

    let values = match column.column.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => {
            return Err(Error::new(ErrorKind::UnsupportedType, operation)
                .with_message(format!(
                    "{} column uses layout {:?}; graph inputs currently require fixed-width prepared exports",
                    name, layout
                ))
                .with_context("column_name", name.to_string()));
        },
    };

    let binding = column.bindings.iter().find(|binding| binding.id() == values).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!(
                "{} column prepared export is missing the values buffer binding referenced by its semantic layout",
                name
            ))
            .with_context("column_name", name.to_string())
            .with_context("buffer_id", values.get().to_string())
    })?;

    let Placement::CudaDevice { .. } = binding.placement() else {
        return Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!(
                "{} column prepared export is host-materialized; graph construction requires device-resident prepared export inputs",
                name
            ))
            .with_context("column_name", name.to_string()));
    };

    let byte_offset = column.column.offset().checked_mul(cugraph_dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!("{} column offset is too large", name))
            .with_context("column_name", name.to_string())
            .with_context("offset", column.column.offset().to_string())
            .with_context("dtype_size", cugraph_dtype.size_of().to_string())
    })?;
    let active_len = column.column.len().checked_mul(cugraph_dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!("{} column active byte range is too large", name))
            .with_context("column_name", name.to_string())
            .with_context("len", column.column.len().to_string())
            .with_context("dtype_size", cugraph_dtype.size_of().to_string())
    })?;
    let required_end = byte_offset.checked_add(active_len).ok_or_else(|| {
        Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!("{} column active byte range overflowed", name))
            .with_context("column_name", name.to_string())
    })?;

    if binding.bytes().byte_len() < required_end {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!(
                "{} column prepared export binding is smaller than the active fixed-width range",
                name
            ))
            .with_context("column_name", name.to_string())
            .with_context("binding_byte_len", binding.bytes().byte_len().to_string())
            .with_context("required_end", required_end.to_string()));
    }

    let base_ptr = binding.bytes().base_ptr() as *const u8;
    let ptr = if active_len == 0 {
        base_ptr as *mut c_void
    } else {
        // SAFETY: prepared-export validation plus the range check above ensure
        // that the active fixed-width range is inside the bound bytes region.
        unsafe { base_ptr.add(byte_offset) as *mut c_void }
    };

    // SAFETY: `ptr` points at the active fixed-width region of the prepared export.
    Ok(unsafe { RawDeviceArrayView::from_raw(ptr, column.column.len(), cugraph_dtype) })
}

fn borrowed_table_to_edge_list<'a>(
    table: &'a BorrowedTableView<'a>,
    options: &TableGraphOptions,
    operation: &'static str,
) -> Result<BorrowedEdgeList<'a>> {
    let num_cols = table.num_columns();

    let validate_col_index = |idx: usize, name: &str| -> Result<()> {
        if idx >= num_cols {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
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

    validate_col_index(options.src_col, "src")?;
    validate_col_index(options.dst_col, "dst")?;
    if let Some(wgt_col) = options.wgt_col {
        validate_col_index(wgt_col, "wgt")?;
    }

    let columns = table.columns();
    Ok(BorrowedEdgeList {
        src: columns[options.src_col].clone(),
        dst: columns[options.dst_col].clone(),
        weights: options.wgt_col.map(|index| columns[index].clone()),
    })
}

fn validate_edge_list_policy(
    handle: &ResourceHandle,
    edge_list: &BorrowedEdgeList<'_>,
    operation: &'static str,
) -> Result<()> {
    let producer_device = edge_list.src.execution().device();
    let mut columns = vec![("dst", &edge_list.dst)];
    if let Some(weights) = edge_list.weights.as_ref() {
        columns.push(("wgt", weights));
    }

    for (name, column) in columns {
        let device = column.execution().device();
        if device != producer_device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("borrowed graph inputs must all reside on the same CUDA device")
                .with_context("producer_device", producer_device.to_string())
                .with_context("column_name", name.to_string())
                .with_context("column_device", device.to_string()));
        }
    }

    if producer_device != handle.device() {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("borrowed graph inputs must match the consumer handle device")
            .with_context("producer_device", producer_device.to_string())
            .with_context("handle_device", handle.device().to_string()));
    }

    Ok(())
}

fn validate_prepared_edge_list_policy(
    handle: &ResourceHandle,
    edge_list: &PreparedEdgeList<'_>,
    operation: &'static str,
) -> Result<()> {
    let producer_device = prepared_column_device(edge_list.src, "src", operation)?;

    for (name, column) in [("dst", edge_list.dst), ("wgt", edge_list.weights.unwrap_or(edge_list.src))] {
        if name == "wgt" && edge_list.weights.is_none() {
            continue;
        }

        let device = prepared_column_device(column, name, operation)?;
        if device != producer_device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("prepared graph inputs must all reside on the same CUDA device")
                .with_context("producer_device", producer_device.to_string())
                .with_context("column_name", name.to_string())
                .with_context("column_device", device.to_string()));
        }
    }

    if producer_device != handle.device() {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("prepared graph inputs must match the consumer handle device")
            .with_context("producer_device", producer_device.to_string())
            .with_context("handle_device", handle.device().to_string()));
    }

    Ok(())
}

fn fence_producer_streams(edge_list: &BorrowedEdgeList<'_>, operation: &'static str) -> Result<()> {
    let _device = DeviceGuard::switch_to(edge_list.src.execution().device(), operation)?;
    let mut seen = BTreeSet::new();

    for stream in [
        edge_list.src.execution().cuda_stream().as_raw(),
        edge_list.dst.execution().cuda_stream().as_raw(),
        edge_list
            .weights
            .as_ref()
            .map(|column| column.execution().cuda_stream().as_raw())
            .unwrap_or(std::ptr::null_mut()),
    ] {
        if stream.is_null() || !seen.insert(stream as usize) {
            continue;
        }
        check_cudf_status(unsafe { cudf_sys::cudf_raw_cuda_stream_synchronize(stream) }, operation)?;
    }

    Ok(())
}

fn build_graph_from_borrowed_edge_list<'a, T: Transposition>(
    handle: &ResourceHandle,
    edge_list: &BorrowedEdgeList<'a>,
    properties: GraphProperties,
    renumber: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    validate_edge_list_policy(handle, edge_list, operation)?;
    let src = borrowed_column_to_raw(&edge_list.src, "src", operation)?;
    let dst = borrowed_column_to_raw(&edge_list.dst, "dst", operation)?;
    let wgt = edge_list.weights.as_ref().map(|column| borrowed_column_to_raw(column, "wgt", operation)).transpose()?;
    fence_producer_streams(edge_list, operation)?;

    let mut builder = GraphBuilder::from_raw_views(properties, &src, &dst).renumber_flag(renumber);
    if let Some(ref weights) = wgt {
        builder = builder.weights_raw(weights);
    }
    builder.build_inner(handle, T::store_transposed())
}

fn complete_prepared_lease(
    result: std::result::Result<LeaseCompletionResult, InteropError>,
    operation: &'static str,
) -> Result<()> {
    match result.map_err(|error| map_interop_error(error, operation))? {
        LeaseCompletionResult::FallbackSynchronized => Ok(()),
        LeaseCompletionResult::Completed(witness) => {
            witness.synchronize().map_err(|error| map_interop_error(error, operation))
        },
    }
}

fn finish_prepared_graph_result<T: Transposition>(
    graph_result: Result<Graph<T>>,
    completions: impl IntoIterator<Item = Result<()>>,
) -> Result<Graph<T>> {
    let mut completion_error = None;
    for completion in completions {
        if let Err(error) = completion {
            if completion_error.is_none() {
                completion_error = Some(error);
            }
        }
    }

    match graph_result {
        Ok(graph) => {
            if let Some(error) = completion_error {
                Err(error)
            } else {
                Ok(graph)
            }
        },
        Err(error) => Err(error),
    }
}

fn build_graph_from_prepared_edge_list<'a, T: Transposition>(
    handle: &ResourceHandle,
    edge_list: &PreparedEdgeList<'a>,
    properties: GraphProperties,
    renumber: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    validate_prepared_edge_list_policy(handle, edge_list, operation)?;

    let src = prepared_column_to_raw(edge_list.src, "src", operation)?;
    let dst = prepared_column_to_raw(edge_list.dst, "dst", operation)?;
    let wgt = edge_list.weights.map(|column| prepared_column_to_raw(column, "wgt", operation)).transpose()?;

    let mut builder = GraphBuilder::from_raw_views(properties, &src, &dst).renumber_flag(renumber);
    if let Some(ref weights) = wgt {
        builder = builder.weights_raw(weights);
    }
    builder.build_inner(handle, T::store_transposed())
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction functions
// ═══════════════════════════════════════════════════════════════════════════

/// Build a cuGraph [`Graph`] directly from borrowed RAPIDS column descriptors.
pub fn graph_from_borrowed_columns<T: Transposition>(
    handle: &ResourceHandle,
    src: &BorrowedColumnView<'_>,
    dst: &BorrowedColumnView<'_>,
    weights: Option<&BorrowedColumnView<'_>>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    let edge_list = BorrowedEdgeList { src: src.clone(), dst: dst.clone(), weights: weights.cloned() };
    build_graph_from_borrowed_edge_list(
        handle,
        &edge_list,
        options.properties,
        options.renumber,
        "graph_from_borrowed_columns",
    )
}

/// Build a cuGraph [`Graph`] directly from a borrowed RAPIDS table descriptor.
pub fn graph_from_borrowed_table_view<T: Transposition>(
    handle: &ResourceHandle,
    view: &BorrowedTableView<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_borrowed_table_view";
    let edge_list = borrowed_table_to_edge_list(view, options, operation)?;
    build_graph_from_borrowed_edge_list(handle, &edge_list, options.properties, options.renumber, operation)
}

/// Build a cuGraph [`Graph`] directly from prepared-device RAPIDS column exports.
///
/// This is the primary single-GPU shared-contract path. The function activates
/// one read lease per input export with conservative synchronous-by-return
/// authorities, builds the graph, then completes the leases before returning.
pub fn graph_from_prepared_columns<T: Transposition>(
    handle: &ResourceHandle,
    src: &PreparedColumnExport<'_>,
    dst: &PreparedColumnExport<'_>,
    weights: Option<&PreparedColumnExport<'_>>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_prepared_columns";
    prepared_profile_supported(src.diagnostics().profile(), operation)?;
    prepared_profile_supported(dst.diagnostics().profile(), operation)?;
    if let Some(weights) = weights {
        prepared_profile_supported(weights.diagnostics().profile(), operation)?;
    }

    let read_start = ReadStartAuthority::SynchronousByReturn;
    let completion = CompletionAuthority::SynchronousByReturn;
    let mut src_lease =
        Some(src.try_activate_read_lease(read_start, completion).map_err(|error| map_interop_error(error, operation))?);
    let mut dst_lease = None;
    let mut wgt_lease = None;

    let graph_result = (|| {
        dst_lease = Some(
            dst.try_activate_read_lease(read_start, completion).map_err(|error| map_interop_error(error, operation))?,
        );
        wgt_lease = weights
            .map(|weights| {
                weights
                    .try_activate_read_lease(read_start, completion)
                    .map_err(|error| map_interop_error(error, operation))
            })
            .transpose()?;

        let edge_list = PreparedEdgeList {
            src: PreparedEdgeColumn {
                column: src_lease.as_ref().expect("src lease should be active").export().column(),
                bindings: src_lease.as_ref().expect("src lease should be active").export().bindings(),
            },
            dst: PreparedEdgeColumn {
                column: dst_lease.as_ref().expect("dst lease should be active").export().column(),
                bindings: dst_lease.as_ref().expect("dst lease should be active").export().bindings(),
            },
            weights: wgt_lease.as_ref().map(|lease| PreparedEdgeColumn {
                column: lease.export().column(),
                bindings: lease.export().bindings(),
            }),
        };

        build_graph_from_prepared_edge_list(handle, &edge_list, options.properties, options.renumber, operation)
    })();

    let src_complete =
        src_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation)).unwrap_or(Ok(()));
    let dst_complete =
        dst_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation)).unwrap_or(Ok(()));
    let wgt_complete =
        wgt_lease.take().map(|lease| complete_prepared_lease(lease.complete(), operation)).unwrap_or(Ok(()));

    finish_prepared_graph_result(graph_result, [src_complete, dst_complete, wgt_complete])
}

fn validate_prepared_table_column_selection(
    num_cols: usize,
    options: &TableGraphOptions,
    operation: &'static str,
) -> Result<()> {
    let validate_col_index = |idx: usize, name: &str| -> Result<()> {
        if idx >= num_cols {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} column index {} is out of bounds (prepared table export has {} columns)",
                    name, idx, num_cols
                ))
                .with_context("column_name", name.to_string())
                .with_context("column_index", idx.to_string())
                .with_context("num_columns", num_cols.to_string()));
        }
        Ok(())
    };

    validate_col_index(options.src_col, "src")?;
    validate_col_index(options.dst_col, "dst")?;
    if let Some(wgt_col) = options.wgt_col {
        validate_col_index(wgt_col, "wgt")?;
    }

    Ok(())
}

/// Build a cuGraph [`Graph`] directly from a prepared-device RAPIDS table export.
pub fn graph_from_prepared_table_export<T: Transposition>(
    handle: &ResourceHandle,
    export: &PreparedTableExport<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_prepared_table_export";
    prepared_profile_supported(export.diagnostics().profile(), operation)?;
    validate_prepared_table_column_selection(export.columns().len(), options, operation)?;

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .map_err(|error| map_interop_error(error, operation))?;

    let columns = lease.export().columns();
    let bindings = lease.export().bindings();
    let edge_list = PreparedEdgeList {
        src: PreparedEdgeColumn { column: &columns[options.src_col], bindings },
        dst: PreparedEdgeColumn { column: &columns[options.dst_col], bindings },
        weights: options.wgt_col.map(|index| PreparedEdgeColumn { column: &columns[index], bindings }),
    };

    let graph_result =
        build_graph_from_prepared_edge_list(handle, &edge_list, options.properties, options.renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}

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
    view: &TableView<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_table_view";
    let borrowed = view.try_borrow_table_view().map_err(|error| map_interop_error(error, operation))?;
    let edge_list = borrowed_table_to_edge_list(&borrowed, options, operation)?;
    build_graph_from_borrowed_edge_list(handle, &edge_list, options.properties, options.renumber, operation)
}

/// Build a cuGraph [`Graph`] from a cudf [`DataFrame`] using column names.
///
/// **Prepared-export-first input**: resolves column names, prepares a
/// producer-rooted device table export, then activates read access explicitly
/// during graph construction. No intermediate table copy is performed.
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
    let names = df.column_names();
    let resolve_index = |name: &str, role: &str| -> Result<usize> {
        names.iter().position(|candidate| *candidate == name).ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("{} column '{}' not found in DataFrame schema", role, name))
                .with_context("column_label", role.to_string())
                .with_context("column_name", name.to_string())
        })
    };

    let table_options = TableGraphOptions {
        src_col: resolve_index(options.src_col, "src")?,
        dst_col: resolve_index(options.dst_col, "dst")?,
        wgt_col: options.wgt_col.map(|name| resolve_index(name, "wgt")).transpose()?,
        properties: options.properties,
        renumber: options.renumber,
    };

    let export = df.try_prepare_device_table_export().map_err(|error| map_interop_error(error, operation))?;
    graph_from_prepared_table_export(handle, &export, &table_options)
}

/// Build a cuGraph [`Graph`] from cudf [`Series`] (src vertices, dst vertices,
/// optional weights).
///
/// **Prepared-export-first input**: each Series is projected into a
/// producer-rooted device export and consumed through explicit read-lease
/// activation during graph construction. The Graph's internal CSR/CSC is
/// independent after construction.
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
    let src_export = src.try_prepare_device_column_export().map_err(|error| map_interop_error(error, operation))?;
    let dst_export = dst.try_prepare_device_column_export().map_err(|error| map_interop_error(error, operation))?;
    let wgt_export = weights
        .map(|column| column.try_prepare_device_column_export().map_err(|error| map_interop_error(error, operation)))
        .transpose()?;

    graph_from_prepared_columns(handle, &src_export, &dst_export, wgt_export.as_ref(), options)
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
    pub table: Table,
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
    let runtime = cudf::Runtime::builder()
        .with_device(cudf::DeviceId::new(handle.device() as usize).map_err(|error| {
            Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
                .with_message(format!("invalid handle device {}: {}", handle.device(), error))
                .set_source(error)
        })?)
        .build()
        .map_err(|error| {
            Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                .with_message(format!("failed to create cudf runtime for device {}: {}", handle.device(), error))
                .set_source(error)
        })?;

    let reader_opts = parquet::ParquetReadOptions::builder().filepath(options.path).with_runtime(&runtime).build();

    let result = parquet::read_parquet(&reader_opts).map_err(|e| {
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
    const MAPPED_TYPE_IDS: &[CudfTypeId] = &[
        CudfTypeId::Int8,
        CudfTypeId::Int16,
        CudfTypeId::Int32,
        CudfTypeId::Int64,
        CudfTypeId::UInt8,
        CudfTypeId::UInt16,
        CudfTypeId::UInt32,
        CudfTypeId::UInt64,
        CudfTypeId::Float32,
        CudfTypeId::Float64,
        CudfTypeId::Bool8,
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
            CudfTypeId::Empty,
            CudfTypeId::TimestampDays,
            CudfTypeId::TimestampSeconds,
            CudfTypeId::TimestampMilliseconds,
            CudfTypeId::TimestampMicroseconds,
            CudfTypeId::TimestampNanoseconds,
            CudfTypeId::DurationDays,
            CudfTypeId::DurationSeconds,
            CudfTypeId::DurationMilliseconds,
            CudfTypeId::DurationMicroseconds,
            CudfTypeId::DurationNanoseconds,
            CudfTypeId::Dictionary32,
            CudfTypeId::String,
            CudfTypeId::List,
            CudfTypeId::Decimal32,
            CudfTypeId::Decimal64,
            CudfTypeId::Decimal128,
            CudfTypeId::Struct,
        ];
        for tid in unsupported {
            assert_eq!(cudf_type_to_cugraph(tid), None, "expected None for unsupported cudf TypeId {:?}", tid);
        }
    }
}
