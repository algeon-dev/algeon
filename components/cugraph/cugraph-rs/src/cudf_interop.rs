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
//! - [`graph_from_table_view()`] — borrowed-compatibility construction from a
//!   cudf table view.
//! - [`graph_from_series()`] and [`graph_from_dataframe()`] — convenience
//!   producers that prepare device exports at the API boundary, then delegate
//!   into the prepared-export graph construction path.
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
//! [`DeviceSlice`]: cudf::core::DeviceSlice

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
    cuda::{DeviceGuard, check_cudf_status},
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

macro_rules! map_type_id_to_cugraph {
    ($type_id:expr, $type:path) => {
        match $type_id {
            <$type>::Int8 => Some(DataType::Int8),
            <$type>::Int16 => Some(DataType::Int16),
            <$type>::Int32 => Some(DataType::Int32),
            <$type>::Int64 => Some(DataType::Int64),
            <$type>::UInt8 => Some(DataType::Uint8),
            <$type>::UInt16 => Some(DataType::Uint16),
            <$type>::UInt32 => Some(DataType::Uint32),
            <$type>::UInt64 => Some(DataType::Uint64),
            <$type>::Float32 => Some(DataType::Float32),
            <$type>::Float64 => Some(DataType::Float64),
            <$type>::Bool8 => Some(DataType::Bool),
            _ => None,
        }
    };
}

/// Convert a cudf [`cudf::TypeId`] to a cugraph [`DataType`].
///
/// Returns `None` for cudf types that have no direct cugraph equivalent
/// (e.g. timestamps, strings, structs).
pub fn cudf_type_to_cugraph(type_id: CudfTypeId) -> Option<DataType> {
    map_type_id_to_cugraph!(type_id, CudfTypeId)
}

fn borrowed_type_to_cugraph(type_id: InteropTypeId) -> Option<DataType> {
    map_type_id_to_cugraph!(type_id, InteropTypeId)
}

/// Convert a cugraph [`DataType`] to a cudf [`cudf::TypeId`].
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

/// Create a typed [`crate::DeviceArrayView`] from a runtime-typed cudf [`DeviceSlice`],
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
/// let runtime = cudf::Runtime::new().unwrap();
/// let series = runtime.series("v", &[0i32, 1, 2]).unwrap();
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
    let raw = unsafe {
        crate::array::RawDeviceArrayView::try_from_raw(
            slice.as_ptr() as *mut c_void,
            slice.len(),
            expected,
            "device_array_view_from_slice",
        )?
    };
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
            .map_err(|error| {
                map_cudf_error_with(
                    error,
                    "RawDeviceArrayViewExt::to_cudf_column",
                    "failed to materialize cudf Column from cugraph device array",
                    [("len", self.len().to_string()), ("dtype", format!("{:?}", dtype))],
                )
            })?;

        Ok(col)
    }
}

fn named_series_from_column(column: Column, name: &str, operation: &'static str) -> Result<cudf::Series> {
    cudf::Series::from_column(column).map(|series| series.rename(name)).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            format!("failed to create cudf Series '{name}' from cugraph result column"),
            [("series_name", name.to_string())],
        )
    })
}

fn named_series_from_column_result(
    column: Result<Column>,
    name: &str,
    operation: &'static str,
) -> Result<cudf::Series> {
    named_series_from_column(column?, name, operation)
}

fn named_series_from_raw_view(
    view: &RawDeviceArrayView<'_>,
    name: &str,
    operation: &'static str,
) -> Result<cudf::Series> {
    named_series_from_column_result(view.to_cudf_column(), name, operation)
}

fn dataframe_from_named_series<const N: usize>(
    series: [cudf::Series; N],
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    cudf::DataFrame::from_series(series).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to build cudf DataFrame from cugraph result columns",
            [("series_count", N.to_string())],
        )
    })
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
        named_series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(),
                Self::I64(v) => v.raw_view().to_cudf_column(),
            },
            name,
            "VertexIds::to_series",
        )
    }
}

impl IntoSeries for ScoreValues<'_> {
    fn to_series(&self, name: &str) -> Result<cudf::Series> {
        named_series_from_column_result(
            match self {
                Self::F32(v) => v.raw_view().to_cudf_column(),
                Self::F64(v) => v.raw_view().to_cudf_column(),
            },
            name,
            "ScoreValues::to_series",
        )
    }
}

impl IntoSeries for Distances<'_> {
    fn to_series(&self, name: &str) -> Result<cudf::Series> {
        named_series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(),
                Self::I64(v) => v.raw_view().to_cudf_column(),
                Self::F32(v) => v.raw_view().to_cudf_column(),
                Self::F64(v) => v.raw_view().to_cudf_column(),
            },
            name,
            "Distances::to_series",
        )
    }
}

impl IntoSeries for DegreeValues<'_> {
    fn to_series(&self, name: &str) -> Result<cudf::Series> {
        named_series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(),
                Self::I64(v) => v.raw_view().to_cudf_column(),
            },
            name,
            "DegreeValues::to_series",
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TryFrom impls — result structs → cudf::DataFrame
// ═══════════════════════════════════════════════════════════════════════════

impl TryFrom<&PageRankResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &PageRankResult) -> Result<Self> {
        let operation = "TryFrom<&PageRankResult>";
        dataframe_from_named_series(
            [result.vertex_ids()?.to_series("vertex")?, result.scores()?.to_series("value")?],
            operation,
        )
    }
}

impl TryFrom<&BetweennessCentralityResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &BetweennessCentralityResult) -> Result<Self> {
        let operation = "TryFrom<&BetweennessCentralityResult>";
        dataframe_from_named_series(
            [result.vertex_ids()?.to_series("vertex")?, result.scores()?.to_series("value")?],
            operation,
        )
    }
}

impl TryFrom<&BfsResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &BfsResult) -> Result<Self> {
        let operation = "TryFrom<&BfsResult>";
        let vertices = result.vertices_raw()?;
        let distances = result.distances_raw()?;
        let expected_len = vertices.len();

        let vertex_series = named_series_from_raw_view(&vertices, "vertex", operation)?;
        let distance_series = named_series_from_raw_view(&distances, "distance", operation)?;

        match result.predecessors_raw() {
            Some(preds) if preds.len() == expected_len => {
                let predecessor_series = named_series_from_raw_view(&preds, "predecessor", operation)?;
                dataframe_from_named_series([vertex_series, distance_series, predecessor_series], operation)
            },
            Some(preds) => Err(Error::new(ErrorKind::Unexpected, operation).with_message(format!(
                "predecessor length {} does not match vertex length {}",
                preds.len(),
                expected_len
            ))),
            None => dataframe_from_named_series([vertex_series, distance_series], operation),
        }
    }
}

impl TryFrom<&LayoutResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &LayoutResult) -> Result<Self> {
        let operation = "TryFrom<&LayoutResult>";
        let vertex_series = named_series_from_raw_view(&result.vertices_raw()?, "vertex", operation)?;
        let x_series = named_series_from_raw_view(&result.x_raw()?, "x", operation)?;
        let y_series = named_series_from_raw_view(&result.y_raw()?, "y", operation)?;

        dataframe_from_named_series([vertex_series, x_series, y_series], operation)
    }
}

impl TryFrom<&DegreesResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &DegreesResult) -> Result<Self> {
        let operation = "TryFrom<&DegreesResult>";
        dataframe_from_named_series(
            [
                result.vertex_ids()?.to_series("vertex")?,
                result.in_degrees()?.to_series("in_degree")?,
                result.out_degrees()?.to_series("out_degree")?,
            ],
            operation,
        )
    }
}

impl TryFrom<&InDegreesResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &InDegreesResult) -> Result<Self> {
        let operation = "TryFrom<&InDegreesResult>";
        dataframe_from_named_series(
            [result.vertex_ids()?.to_series("vertex")?, result.in_degrees()?.to_series("in_degree")?],
            operation,
        )
    }
}

impl TryFrom<&OutDegreesResult> for cudf::DataFrame {
    type Error = Error;

    fn try_from(result: &OutDegreesResult) -> Result<Self> {
        let operation = "TryFrom<&OutDegreesResult>";
        dataframe_from_named_series(
            [result.vertex_ids()?.to_series("vertex")?, result.out_degrees()?.to_series("out_degree")?],
            operation,
        )
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
    pub weight_col: Option<usize>,
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
        Self { src_col, dst_col, weight_col: None, properties: GraphProperties::directed(), renumber: false }
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
    pub weight_col: Option<&'a str>,
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
        Self { src_col, dst_col, weight_col: None, properties: GraphProperties::directed(), renumber: false }
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
    pub weight_col: Option<usize>,
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
        Self { path, src_col, dst_col, weight_col: None, properties: GraphProperties::directed(), renumber: false }
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
        InteropErrorKind::NotFound
        | InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::AmbientInteropUnsupported
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => Error::new(ErrorKind::InvalidState, operation),
        InteropErrorKind::MixedDeviceUnsupported | InteropErrorKind::NullMaskUnsupported => {
            Error::new(ErrorKind::InvalidInput, operation)
        },
    }
    .with_message(error.to_string())
    .set_source(error.clone());

    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }

    mapped
}

fn map_cudf_error_kind(kind: cudf::ErrorKind) -> ErrorKind {
    match kind {
        cudf::ErrorKind::OutOfMemory => ErrorKind::OutOfMemory,
        cudf::ErrorKind::IoError | cudf::ErrorKind::NotFound => ErrorKind::IoError,
        cudf::ErrorKind::TypeError | cudf::ErrorKind::NotSupported => ErrorKind::UnsupportedType,
        cudf::ErrorKind::InvalidOperation => ErrorKind::InvalidState,
        cudf::ErrorKind::InvalidArgument | cudf::ErrorKind::NullError | cudf::ErrorKind::IndexError => {
            ErrorKind::InvalidInput
        },
        cudf::ErrorKind::CudaError | cudf::ErrorKind::FatalCudaError | cudf::ErrorKind::Unknown => {
            ErrorKind::Unexpected
        },
    }
}

#[cfg(test)]
fn map_cudf_error(error: cudf::Error, operation: &'static str) -> Error {
    map_cudf_error_with(error, operation, "cudf operation failed", std::iter::empty())
}

fn map_cudf_error_with<I>(
    error: cudf::Error,
    operation: &'static str,
    message: impl Into<String>,
    context: I,
) -> Error
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    let source_kind = error.kind();
    let source_context = error.context().iter().map(|(key, value)| (*key, value.clone())).collect::<Vec<_>>();
    let mut mapped =
        Error::new(map_cudf_error_kind(source_kind), operation).with_message(message.into()).set_source(error);

    match source_kind {
        cudf::ErrorKind::OutOfMemory | cudf::ErrorKind::CudaError => mapped = mapped.set_temporary(),
        cudf::ErrorKind::FatalCudaError => mapped = mapped.set_persistent(),
        _ => {},
    }

    for (key, value) in source_context {
        mapped = mapped.with_context(key, value);
    }

    for (key, value) in context {
        mapped = mapped.with_context(key, value);
    }

    mapped
}

fn edge_list_dtype(type_id: InteropTypeId, name: &str, operation: &'static str) -> Result<DataType> {
    borrowed_type_to_cugraph(type_id).ok_or_else(|| {
        Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("{} column has interop type {:?} which has no cugraph equivalent", name, type_id))
            .with_context("column_name", name.to_string())
    })
}

fn checked_column_offset_bytes(offset: usize, dtype: DataType, name: &str, operation: &'static str) -> Result<usize> {
    offset.checked_mul(dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!("{} column offset is too large", name))
            .with_context("column_name", name.to_string())
            .with_context("offset", offset.to_string())
            .with_context("dtype_size", dtype.size_of().to_string())
    })
}

fn checked_active_byte_len(len: usize, dtype: DataType, name: &str, operation: &'static str) -> Result<usize> {
    len.checked_mul(dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!("{} column active byte range is too large", name))
            .with_context("column_name", name.to_string())
            .with_context("len", len.to_string())
            .with_context("dtype_size", dtype.size_of().to_string())
    })
}

/// Return the device pointer for the active fixed-width range of a borrowed
/// cudf column buffer.
///
/// Callers must validate the column's active byte range before calling this
/// helper. In particular, `byte_offset` must already be known to be in-bounds
/// for `base_ptr` whenever `active_len > 0`. Null pointers and empty active
/// ranges are passed through unchanged to preserve the C API's empty-column
/// convention.
fn offset_device_ptr(base_ptr: *const u8, byte_offset: usize, active_len: usize) -> *mut c_void {
    if base_ptr.is_null() || active_len == 0 {
        base_ptr as *mut c_void
    } else {
        // SAFETY: callers validate that the active fixed-width range is in-bounds.
        unsafe { base_ptr.add(byte_offset) as *mut c_void }
    }
}

fn validate_edge_list_devices<'a>(
    handle: &ResourceHandle,
    producer_device: i32,
    columns: impl IntoIterator<Item = (&'a str, i32)>,
    input_kind: &str,
    operation: &'static str,
) -> Result<()> {
    for (name, device) in columns {
        if device != producer_device {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message(format!("{input_kind} graph inputs must all reside on the same CUDA device"))
                .with_context("producer_device", producer_device.to_string())
                .with_context("column_name", name.to_string())
                .with_context("column_device", device.to_string()));
        }
    }

    if producer_device != handle.device() {
        return Err(Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!("{input_kind} graph inputs must match the consumer handle device"))
            .with_context("producer_device", producer_device.to_string())
            .with_context("handle_device", handle.device().to_string()));
    }

    Ok(())
}

fn borrowed_column_to_raw<'a>(
    column: &BorrowedColumnView<'a>,
    name: &str,
    operation: &'static str,
) -> Result<RawDeviceArrayView<'a>> {
    let cugraph_dtype = edge_list_dtype(column.type_id(), name, operation)?;

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

    let byte_offset = checked_column_offset_bytes(column.offset(), cugraph_dtype, name, operation)?;
    let active_len = checked_active_byte_len(column.len(), cugraph_dtype, name, operation)?;
    let ptr = offset_device_ptr(column.data_ptr() as *const u8, byte_offset, active_len);

    // SAFETY: `ptr` points at the active region of the producer-owned device buffer.
    Ok(unsafe { RawDeviceArrayView::try_from_raw(ptr, column.len(), cugraph_dtype, operation)? })
}

fn prepared_profile_supported(profile: rapids_interop::ExportProfile, operation: &'static str) -> Result<()> {
    if profile == rapids_interop::ExportProfile::PreparedDeviceExport {
        return Ok(());
    }

    Err(Error::new(ErrorKind::UnsupportedType, operation)
        .with_message(format!("{operation} requires prepared device export inputs, got {:?}", profile)))
}

fn prepared_fixed_width_values_binding<'a>(
    column: PreparedEdgeColumn<'a>,
    name: &str,
    operation: &'static str,
) -> Result<&'a rapids_interop::BufferBinding<'a>> {
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
        Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!(
                "{} column prepared export is missing the values buffer binding referenced by its semantic layout",
                name
            ))
            .with_context("column_name", name.to_string())
            .with_context("buffer_id", values.get().to_string())
    })?;

    Ok(binding)
}

fn prepared_device_binding<'a>(
    column: PreparedEdgeColumn<'a>,
    name: &str,
    operation: &'static str,
) -> Result<&'a rapids_interop::BufferBinding<'a>> {
    let binding = prepared_fixed_width_values_binding(column, name, operation)?;

    let Placement::CudaDevice { .. } = binding.placement() else {
        return Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!(
                "{} column prepared export is host-materialized; graph construction requires device-resident prepared export inputs",
                name
            ))
            .with_context("column_name", name.to_string()));
    };

    Ok(binding)
}

fn prepared_column_device<'a>(column: PreparedEdgeColumn<'a>, name: &str, operation: &'static str) -> Result<i32> {
    let binding = prepared_device_binding(column, name, operation)?;
    let Placement::CudaDevice { device } = binding.placement() else {
        unreachable!("prepared_device_binding must return a device-resident binding")
    };
    Ok(*device)
}

fn prepared_column_to_raw<'a>(
    column: PreparedEdgeColumn<'a>,
    name: &str,
    operation: &'static str,
) -> Result<RawDeviceArrayView<'a>> {
    let cugraph_dtype = edge_list_dtype(column.column.data_type().id(), name, operation)?;

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
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message(format!(
                    "{} column prepared export does not make null count explicit; graph inputs must prove a null-free active range",
                    name
                ))
                .with_context("column_name", name.to_string()));
        },
    }

    let binding = prepared_device_binding(column, name, operation)?;
    let byte_offset = checked_column_offset_bytes(column.column.offset(), cugraph_dtype, name, operation)?;
    let active_len = checked_active_byte_len(column.column.len(), cugraph_dtype, name, operation)?;
    let required_end = byte_offset.checked_add(active_len).ok_or_else(|| {
        Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!("{} column active byte range overflowed", name))
            .with_context("column_name", name.to_string())
    })?;

    if binding.bytes().byte_len() < required_end {
        return Err(Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!(
                "{} column prepared export binding is smaller than the active fixed-width range",
                name
            ))
            .with_context("column_name", name.to_string())
            .with_context("binding_byte_len", binding.bytes().byte_len().to_string())
            .with_context("required_end", required_end.to_string()));
    }

    let ptr = offset_device_ptr(binding.bytes().base_ptr() as *const u8, byte_offset, active_len);

    // SAFETY: `ptr` points at the active fixed-width region of the prepared export.
    Ok(unsafe { RawDeviceArrayView::try_from_raw(ptr, column.column.len(), cugraph_dtype, operation)? })
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
    if let Some(weight_col) = options.weight_col {
        validate_col_index(weight_col, "weight")?;
    }

    let columns = table.columns();
    Ok(BorrowedEdgeList {
        src: columns[options.src_col].clone(),
        dst: columns[options.dst_col].clone(),
        weights: options.weight_col.map(|index| columns[index].clone()),
    })
}

fn validate_edge_list_policy(
    handle: &ResourceHandle,
    edge_list: &BorrowedEdgeList<'_>,
    operation: &'static str,
) -> Result<()> {
    let producer_device = edge_list.src.execution().device();
    let mut columns = vec![("dst", edge_list.dst.execution().device())];
    if let Some(weights) = edge_list.weights.as_ref() {
        columns.push(("wgt", weights.execution().device()));
    }

    validate_edge_list_devices(handle, producer_device, columns, "borrowed", operation)
}

fn validate_prepared_edge_list_policy(
    handle: &ResourceHandle,
    edge_list: &PreparedEdgeList<'_>,
    operation: &'static str,
) -> Result<()> {
    let producer_device = prepared_column_device(edge_list.src, "src", operation)?;
    let mut columns = vec![("dst", prepared_column_device(edge_list.dst, "dst", operation)?)];
    if let Some(weights) = edge_list.weights {
        columns.push(("wgt", prepared_column_device(weights, "wgt", operation)?));
    }

    validate_edge_list_devices(handle, producer_device, columns, "prepared", operation)
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

    let mut builder = GraphBuilder::from_raw_views(properties, &src, &dst)?.renumber_flag(renumber);
    if let Some(ref weights) = wgt {
        builder = builder.weights_raw(weights)?;
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

    let mut builder = GraphBuilder::from_raw_views(properties, &src, &dst)?.renumber_flag(renumber);
    if let Some(ref weights) = wgt {
        builder = builder.weights_raw(weights)?;
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
    if let Some(weight_col) = options.weight_col {
        validate_col_index(weight_col, "weight")?;
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
        weights: options.weight_col.map(|index| PreparedEdgeColumn { column: &columns[index], bindings }),
    };

    let graph_result =
        build_graph_from_prepared_edge_list(handle, &edge_list, options.properties, options.renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}

/// Build a cuGraph [`Graph`] from a cudf [`TableView`].
///
/// **Borrowed compatibility path**: this entry point projects the cudf table
/// view into the shared RAPIDS borrowed ABI and performs synchronous host-side
/// fencing before cuGraph reads the borrowed device buffers.
///
/// For new single-GPU integrations, prefer
/// [`graph_from_prepared_table_export()`], which keeps producer read-start,
/// completion, and liveness explicit at the shared-contract layer. This
/// function remains the supported compatibility path for callers that already
/// operate on borrowed descriptors or only have a cudf [`TableView`] available.
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
/// [`TableView`]: cudf::core::TableView
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
        weight_col: options.weight_col.map(|name| resolve_index(name, "weight")).transpose()?,
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
/// let runtime = cudf::Runtime::new().unwrap();
/// let handle = ResourceHandle::for_runtime_device(&runtime).unwrap();
/// let src = runtime.series("src", &[0i32, 1, 2]).unwrap();
/// let dst = runtime.series("dst", &[1i32, 2, 0]).unwrap();
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
/// This is the file-I/O entry point for graph construction. It creates an
/// internal cudf [`cudf::Runtime`] pinned to `handle.device()`, reads the Parquet file
/// into a cudf [`Table`] on that device, prepares a producer-rooted table
/// export, then delegates graph construction to
/// [`graph_from_prepared_table_export()`].
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
/// [`Table`]: cudf::core::Table
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
    let operation = "graph_from_parquet";
    let runtime = cudf::Runtime::builder()
        .with_device(cudf::DeviceId::new(handle.device() as usize).map_err(|error| {
            Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("invalid handle device {}: {}", handle.device(), error))
                .set_source(error)
        })?)
        .build()
        .map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                format!("failed to create cudf runtime for device {}", handle.device()),
                [("device", handle.device().to_string())],
            )
        })?;

    let reader_opts = parquet::ParquetTableReadOptions::builder().filepath(options.path).with_runtime(&runtime).build();

    let result = parquet::read_parquet_table(&reader_opts).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to read Parquet file into cudf table",
            [("path", options.path.to_string())],
        )
    })?;
    let table = result.table;

    let graph = {
        let export = table.try_prepare_device_table_export().map_err(|error| map_interop_error(error, operation))?;

        let table_opts = TableGraphOptions {
            src_col: options.src_col,
            dst_col: options.dst_col,
            weight_col: options.weight_col,
            properties: options.properties,
            renumber: options.renumber,
        };
        graph_from_prepared_table_export::<T>(handle, &export, &table_opts)?
    };

    Ok(LoadedGraph { graph, table })
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use rapids_interop::SemanticColumn;

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
    fn borrowed_type_to_cugraph_matches_cudf_mapping() {
        let borrowed = [
            (InteropTypeId::Int8, CudfTypeId::Int8),
            (InteropTypeId::Int16, CudfTypeId::Int16),
            (InteropTypeId::Int32, CudfTypeId::Int32),
            (InteropTypeId::Int64, CudfTypeId::Int64),
            (InteropTypeId::UInt8, CudfTypeId::UInt8),
            (InteropTypeId::UInt16, CudfTypeId::UInt16),
            (InteropTypeId::UInt32, CudfTypeId::UInt32),
            (InteropTypeId::UInt64, CudfTypeId::UInt64),
            (InteropTypeId::Float32, CudfTypeId::Float32),
            (InteropTypeId::Float64, CudfTypeId::Float64),
            (InteropTypeId::Bool8, CudfTypeId::Bool8),
        ];

        for (borrowed_tid, cudf_tid) in borrowed {
            assert_eq!(
                borrowed_type_to_cugraph(borrowed_tid),
                cudf_type_to_cugraph(cudf_tid),
                "borrowed and cudf mappings diverged for {:?} / {:?}",
                borrowed_tid,
                cudf_tid
            );
        }
    }

    #[test]
    fn size_t_has_no_cudf_equivalent() {
        assert_eq!(cugraph_type_to_cudf(DataType::SizeT), None);
    }

    // ── map_cudf_error tests ────────────────────────────────────────────

    #[test]
    fn map_cudf_error_invalid_argument() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::InvalidArgument, "column 'foo' not found");
        let err =
            map_cudf_error_with(cudf_err, "test_op", "input adapter failed", [("column_role", "src".to_string())]);
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.is_permanent(), "invalid argument should be permanent");
        assert_eq!(err.message(), "input adapter failed");
    }

    #[test]
    fn map_cudf_error_invalid_operation_is_invalid_state() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::InvalidOperation, "runtime must not use ambient provenance");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "invalid operation should be permanent");
    }

    #[test]
    fn map_cudf_error_cuda_error() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::CudaError, "device launch failed");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "CUDA error should be temporary");
    }

    #[test]
    fn map_cudf_error_out_of_memory() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::OutOfMemory, "alloc failed");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.is_temporary(), "OOM should be temporary");
    }

    #[test]
    fn map_cudf_error_io_error() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::IoError, "file not found");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::IoError);
        assert!(err.is_permanent(), "I/O errors should be permanent by default");
    }

    #[test]
    fn map_cudf_error_not_found_is_io_error() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::NotFound, "file not found");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::IoError);
        assert!(err.is_permanent(), "not found should be treated as permanent I/O by default");
    }

    #[test]
    fn map_cudf_error_unknown() {
        let cudf_err = cudf::Error::new(cudf::ErrorKind::Unknown, "something else");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_permanent(), "unknown should be permanent (default)");
    }

    #[test]
    fn map_interop_error_readiness_mismatch_is_invalid_state() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessMismatch, "producer readiness does not match");
        let err = map_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "state mismatches should be permanent");
    }

    #[test]
    fn map_interop_error_invalid_view_is_invalid_state() {
        let interop_err =
            InteropError::new(InteropErrorKind::InvalidView, "prepared export is internally inconsistent");
        let err = map_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "invalid exported views should be permanent state failures");
    }

    #[test]
    fn map_interop_error_not_found_is_invalid_state() {
        let interop_err =
            InteropError::new(InteropErrorKind::NotFound, "prepared export referenced a missing buffer binding");
        let err = map_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "missing interop bindings should be permanent state failures");
    }

    #[test]
    fn map_interop_error_ambient_interop_unsupported_is_invalid_state() {
        let interop_err = InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "producer must provide explicit execution provenance",
        );
        let err = map_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "ambient provenance rejection should be a permanent state failure");
    }

    #[test]
    fn map_interop_error_readiness_unavailable_is_temporary_unexpected() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to query producer event");
        let err = map_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "readiness availability failures remain retryable");
    }

    #[test]
    fn validate_edge_list_devices_mixed_devices_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let err =
            validate_edge_list_devices(&handle, handle.device(), [("dst", handle.device() + 1)], "borrowed", "test_op")
                .expect_err("mixed-device inputs should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn validate_edge_list_devices_handle_mismatch_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let err = validate_edge_list_devices(
            &handle,
            handle.device() + 1,
            std::iter::empty::<(&str, i32)>(),
            "prepared",
            "test_op",
        )
        .expect_err("producer and consumer device mismatch should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn prepared_fixed_width_values_binding_missing_values_binding_is_invalid_state() {
        let column = SemanticColumn::new(
            rapids_interop::ColumnId::new(1),
            rapids_interop::DataType::new(InteropTypeId::Int32),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            rapids_interop::SchemaNullability::NonNullable,
            ColumnLayout::FixedWidth { values: rapids_interop::BufferId::new(42) },
        );
        let prepared = PreparedEdgeColumn { column: &column, bindings: &[] };

        let err = prepared_fixed_width_values_binding(prepared, "src", "test_op")
            .expect_err("missing prepared values binding should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn prepared_column_without_explicit_null_count_is_invalid_state() {
        let bytes = rapids_interop::BoundBytes::from_raw_parts(std::ptr::dangling_mut::<c_void>(), 12).expect("bytes");
        let binding = rapids_interop::BufferBinding::new(
            rapids_interop::BufferId::new(7),
            bytes,
            Placement::CudaDevice { device: 0 },
        );
        let column = SemanticColumn::new(
            rapids_interop::ColumnId::new(1),
            rapids_interop::DataType::new(InteropTypeId::Int32),
            3,
            0,
            PhysicalValidity::ValidityMask { mask: rapids_interop::BufferId::new(99), null_count: None },
            rapids_interop::SchemaNullability::Nullable,
            ColumnLayout::FixedWidth { values: rapids_interop::BufferId::new(7) },
        );
        let bindings = [binding];
        let prepared = PreparedEdgeColumn { column: &column, bindings: &bindings };

        let err = prepared_column_to_raw(prepared, "src", "test_op")
            .expect_err("prepared exports must make null count explicit");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn prepared_column_binding_shorter_than_active_range_is_invalid_state() {
        let bytes = rapids_interop::BoundBytes::from_raw_parts(std::ptr::dangling_mut::<c_void>(), 8).expect("bytes");
        let binding = rapids_interop::BufferBinding::new(
            rapids_interop::BufferId::new(7),
            bytes,
            Placement::CudaDevice { device: 0 },
        );
        let column = SemanticColumn::new(
            rapids_interop::ColumnId::new(1),
            rapids_interop::DataType::new(InteropTypeId::Int32),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            rapids_interop::SchemaNullability::NonNullable,
            ColumnLayout::FixedWidth { values: rapids_interop::BufferId::new(7) },
        );
        let bindings = [binding];
        let prepared = PreparedEdgeColumn { column: &column, bindings: &bindings };

        let err = prepared_column_to_raw(prepared, "src", "test_op")
            .expect_err("prepared bindings smaller than the active range should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
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

    #[test]
    fn unsupported_borrowed_types_return_none() {
        let unsupported = [
            InteropTypeId::Empty,
            InteropTypeId::TimestampDays,
            InteropTypeId::TimestampSeconds,
            InteropTypeId::TimestampMilliseconds,
            InteropTypeId::TimestampMicroseconds,
            InteropTypeId::TimestampNanoseconds,
            InteropTypeId::DurationDays,
            InteropTypeId::DurationSeconds,
            InteropTypeId::DurationMilliseconds,
            InteropTypeId::DurationMicroseconds,
            InteropTypeId::DurationNanoseconds,
            InteropTypeId::Dictionary32,
            InteropTypeId::String,
            InteropTypeId::List,
            InteropTypeId::Decimal32,
            InteropTypeId::Decimal64,
            InteropTypeId::Decimal128,
            InteropTypeId::Struct,
        ];

        for tid in unsupported {
            assert_eq!(borrowed_type_to_cugraph(tid), None, "expected None for unsupported interop TypeId {:?}", tid);
        }
    }
}
