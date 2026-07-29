//! Interop between cudf and cugraph types.
//!
//! # Memory ownership model
//!
//! Functions in this module differ in whether they copy data or create
//! zero-copy views. The distinction matters for lifetime management:
//!
//! `GraphConstructionPolicy::PythonCugraph` adds one extra wrinkle: some
//! ingress paths normalize edge data before raw graph construction. For
//! borrowed and prepared compatibility inputs, that normalization may import
//! the selected edge columns into owned cudf objects on the consumer device
//! before building the graph.
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
//! - [`graph_from_borrowed_table_view()`] — borrowed-compatibility
//!   construction from an explicit RAPIDS borrowed table descriptor.
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
//! | [`graph_from_prepared_columns`] | Depends on policy | Producer export stays alive until explicit lease completion |
//! | [`graph_from_prepared_table_export`] | Depends on policy | Producer export stays alive until explicit lease completion |
//! | [`graph_from_borrowed_columns`] | Depends on policy | Borrowed descriptors must outlive build |
//! | [`graph_from_series`] | No (input) | Series must outlive build |
//! | [`graph_from_dataframe`] | No (input) | DataFrame must outlive build |
//! | [`graph_from_borrowed_table_view`] | Depends on policy | Borrowed descriptors must outlive build |
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
//! on explicit borrowed descriptors. Runtime-bound borrowed paths use
//! device-side ordering; non-runtime-bound compatibility paths may still rely
//! on host fencing.
//!
//! # Borrowed compatibility policy
//!
//! Borrowed graph construction is explicit-only and same-device-only:
//!
//! - cudf producers must expose explicit execution provenance
//! - ambient/default-only cudf objects are rejected
//! - all borrowed inputs must live on the same CUDA device
//! - that device must match the [`ResourceHandle`] device
//! - on runtime-bound handles, same-device different-stream producers insert
//!   device-side ordering into the runtime stream before cuGraph reads
//!   borrowed memory
//! - on non-runtime-bound handles, compatibility borrowed imports may still
//!   fence producer streams on the host before cuGraph reads borrowed memory
//!
//! [`DeviceSlice`]: cudf::core::DeviceSlice

use std::{any::Any, collections::BTreeSet, error::Error as StdError, ffi::c_void, sync::Arc};

use cudf::{
    DuplicateKeepOption, Runtime, SortOptions, TypeId as CudfTypeId,
    core::{Column, DeviceSlice, Table, table::TableView},
    frame::dataframe::DropDuplicatesOptions,
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
    community::{LouvainResult, TriangleCountResult},
    cuda::{DeviceGuard, check_cudf_status},
    error::{Error, ErrorKind, ErrorStatus, Result},
    graph::{DeferredInputCompletion, Graph, GraphBuilder, Transposition},
    handle::EnteredContext,
    labeling::{StronglyConnectedComponentsResult, WeaklyConnectedComponentsResult},
    layout::LayoutResult,
    structure::{
        CoreNumberResult,
        degrees::{DegreeValues, DegreesResult, InDegreesResult, OutDegreesResult},
        typed_results::{Distances, IntegerValues, ScoreValues, VertexIds},
    },
    traversal::{bfs::BfsResult, sssp::SsspResult},
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

fn validate_runtime_result_device(
    runtime: &Runtime,
    source_device: Option<i32>,
    operation: &'static str,
    message: &'static str,
) -> Result<()> {
    let Some(source_device) = source_device else {
        return Ok(());
    };

    let runtime_device = runtime.device().ordinal() as i32;
    if runtime_device == source_device {
        return Ok(());
    }

    Err(Error::new(ErrorKind::InvalidInput, operation)
        .with_message(message)
        .with_context("runtime_device", runtime_device.to_string())
        .with_context("result_device", source_device.to_string()))
}

const NORMALIZED_SRC_COL: &str = "src";
const NORMALIZED_DST_COL: &str = "dst";
const NORMALIZED_WEIGHT_COL: &str = "weight";

fn runtime_for_handle(handle: &EnteredContext<'_>, operation: &'static str) -> Result<Runtime> {
    Runtime::builder()
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
        })
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
/// let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
/// let series = runtime.series(&[0i32, 1, 2]).unwrap();
/// let col = series.as_column();
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
    fn to_cudf_column(&self, runtime: &Runtime) -> Result<Column>;
}

impl RawDeviceArrayViewExt for RawDeviceArrayView<'_> {
    fn to_cudf_column(&self, runtime: &Runtime) -> Result<Column> {
        let dtype = self.dtype()?;
        let cudf_type = cugraph_type_to_cudf(dtype).ok_or_else(|| {
            Error::new(ErrorKind::UnsupportedType, "RawDeviceArrayViewExt::to_cudf_column")
                .with_message(format!("cugraph DataType {:?} has no cudf TypeId equivalent", dtype))
        })?;

        validate_runtime_result_device(
            runtime,
            self.device(),
            "RawDeviceArrayViewExt::to_cudf_column",
            "runtime device must match the cugraph result device",
        )?;

        // SAFETY: `self.device_ptr()` points to valid GPU memory containing
        // `self.len()` elements of `dtype`, which we have verified maps to
        // `cudf_type`. The explicit runtime owns the destination allocation
        // and execution domain for the device-to-device copy.
        let col = unsafe { runtime.column_from_device_ptr(self.device_ptr() as *const c_void, self.len(), cudf_type) }
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

fn series_from_column(column: Column, operation: &'static str) -> Result<cudf::Series> {
    cudf::Series::from_column(column).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to create cudf Series from cugraph result column",
            std::iter::empty(),
        )
    })
}

fn series_from_column_result(column: Result<Column>, operation: &'static str) -> Result<cudf::Series> {
    series_from_column(column?, operation)
}

fn dataframe_from_named_series<I, S>(series: I, operation: &'static str) -> Result<cudf::DataFrame>
where
    I: IntoIterator<Item = (S, cudf::Series)>,
    S: Into<String>,
{
    let series = series.into_iter();
    let (series_count, _) = series.size_hint();
    cudf::DataFrame::from_series(series).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to build cudf DataFrame from cugraph result columns",
            [("series_count", series_count.to_string())],
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
    /// Convert this GPU array to an unnamed cudf [`Series`].
    ///
    /// [`Series`]: cudf::Series
    fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series>;
}

impl IntoSeries for VertexIds<'_> {
    fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "VertexIds::to_series",
        )
    }
}

impl IntoSeries for ScoreValues<'_> {
    fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::F32(v) => v.raw_view().to_cudf_column(runtime),
                Self::F64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "ScoreValues::to_series",
        )
    }
}

impl IntoSeries for Distances<'_> {
    fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
                Self::F32(v) => v.raw_view().to_cudf_column(runtime),
                Self::F64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "Distances::to_series",
        )
    }
}

impl IntoSeries for DegreeValues<'_> {
    fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "DegreeValues::to_series",
        )
    }
}

impl IntoSeries for IntegerValues<'_> {
    fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "IntegerValues::to_series",
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TryFrom impls — result structs → cudf::DataFrame
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn page_rank_result_to_dataframe(result: &PageRankResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "page_rank_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("value", result.scores()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn betweenness_result_to_dataframe(
    result: &BetweennessCentralityResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    let operation = "betweenness_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("value", result.scores()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn louvain_result_to_dataframe(result: &LouvainResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "louvain_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("partition", result.clusters()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn bfs_result_to_dataframe(result: &BfsResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "bfs_result_to_dataframe";
    let vertex_series = result.vertex_ids()?.to_series(runtime)?;
    let distance_series = result.distances()?.to_series(runtime)?;

    match result.predecessors()? {
        Some(predecessors) => {
            let predecessor_series = predecessors.to_series(runtime)?;
            dataframe_from_named_series(
                [("vertex", vertex_series), ("distance", distance_series), ("predecessor", predecessor_series)],
                operation,
            )
        },
        None => dataframe_from_named_series([("vertex", vertex_series), ("distance", distance_series)], operation),
    }
}

pub(crate) fn sssp_result_to_dataframe(result: &SsspResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "sssp_result_to_dataframe";
    let vertex_series = result.vertex_ids()?.to_series(runtime)?;
    let distance_series = result.distances()?.to_series(runtime)?;

    match result.predecessors()? {
        Some(predecessors) => {
            let predecessor_series = predecessors.to_series(runtime)?;
            dataframe_from_named_series(
                [("vertex", vertex_series), ("distance", distance_series), ("predecessor", predecessor_series)],
                operation,
            )
        },
        None => dataframe_from_named_series([("vertex", vertex_series), ("distance", distance_series)], operation),
    }
}

pub(crate) fn layout_result_to_dataframe(result: &LayoutResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "layout_result_to_dataframe";
    let vertex_series = result.vertex_ids()?.to_series(runtime)?;
    let x_series = result.x()?.to_series(runtime)?;
    let y_series = result.y()?.to_series(runtime)?;

    dataframe_from_named_series([("vertex", vertex_series), ("x", x_series), ("y", y_series)], operation)
}

pub(crate) fn degrees_result_to_dataframe(result: &DegreesResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "degrees_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("in_degree", result.in_degrees()?.to_series(runtime)?),
            ("out_degree", result.out_degrees()?.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn in_degrees_result_to_dataframe(result: &InDegreesResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "in_degrees_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("in_degree", result.in_degrees()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn out_degrees_result_to_dataframe(result: &OutDegreesResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "out_degrees_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("out_degree", result.out_degrees()?.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn weakly_connected_components_result_to_dataframe(
    result: &WeaklyConnectedComponentsResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    let operation = "weakly_connected_components_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("label", result.labels()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn strongly_connected_components_result_to_dataframe(
    result: &StronglyConnectedComponentsResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    let operation = "strongly_connected_components_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("label", result.labels()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn core_number_result_to_dataframe(result: &CoreNumberResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "core_number_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("core_number", result.core_numbers()?.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn triangle_count_result_to_dataframe(
    result: &TriangleCountResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    let operation = "triangle_count_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("triangle_count", result.counts()?.to_series(runtime)?),
        ],
        operation,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction option types
// ═══════════════════════════════════════════════════════════════════════════

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
    /// behave exactly like the lower-level [`GraphBuilder`] flags you specify.
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

struct RetainedPreparedCompletion {
    owner: Arc<dyn Any + Send + Sync>,
    witness: rapids_interop::CompletionWitness,
}

/// Testing-only borrowed-input ordering plan.
///
/// This captures the intended runtime-bound ordering contract before the
/// native runtime-bound cuGraph handle lands. It is intentionally narrower
/// than the eventual execution implementation and exists so Phase 1 can lock
/// the contract in executable tests without pretending the old handle is
/// already bound to the runtime stream.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowedInputOrderingPlan {
    CompatibilityHostFence { unique_streams: usize },
    RuntimeNoHostFence,
    RuntimeDeviceSideDependencies { dependency_count: usize },
}

fn borrowed_unique_stream_count(edge_list: &BorrowedEdgeList<'_>) -> usize {
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
        if stream.is_null() {
            continue;
        }
        seen.insert(stream as usize);
    }

    seen.len()
}

fn borrowed_input_ordering_plan(
    handle: &EnteredContext<'_>,
    edge_list: &BorrowedEdgeList<'_>,
    operation: &'static str,
) -> Result<BorrowedInputOrderingPlan> {
    validate_edge_list_policy(handle, edge_list, operation)?;

    let Some(runtime) = handle.runtime() else {
        return Ok(BorrowedInputOrderingPlan::CompatibilityHostFence {
            unique_streams: borrowed_unique_stream_count(edge_list),
        });
    };

    let consumer_stream = cudf::interop::cuda_stream_handle(runtime.stream())
        .map_err(|error| map_runtime_ordering_interop_error(error, operation))?;
    let mut seen = BTreeSet::new();
    let mut dependency_count = 0usize;

    let mut executions = vec![edge_list.src.execution(), edge_list.dst.execution()];
    if let Some(weights) = edge_list.weights.as_ref() {
        executions.push(weights.execution());
    }

    for execution in executions {
        if execution.device() != runtime.device().ordinal() as i32 {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("borrowed graph inputs must match the runtime-bound handle CUDA device")
                .with_context("producer_device", execution.device().to_string())
                .with_context("runtime_device", runtime.device().ordinal().to_string()));
        }

        let producer_stream = execution.cuda_stream().as_raw();
        if producer_stream.is_null() || !seen.insert(producer_stream as usize) {
            continue;
        }
        if producer_stream != consumer_stream.as_raw() {
            dependency_count += 1;
        }
    }

    if dependency_count == 0 {
        Ok(BorrowedInputOrderingPlan::RuntimeNoHostFence)
    } else {
        Ok(BorrowedInputOrderingPlan::RuntimeDeviceSideDependencies { dependency_count })
    }
}

fn order_runtime_after_execution_bindings<'a>(
    runtime: &Runtime,
    executions: impl IntoIterator<Item = rapids_interop::ExecutionBinding<'a>>,
    operation: &'static str,
) -> Result<Vec<cudf::runtime::CudaEvent>> {
    let mut seen = BTreeSet::new();
    let mut barriers = Vec::new();

    for execution in executions {
        let stream_raw = execution.cuda_stream().as_raw();
        if stream_raw.is_null() || !seen.insert(stream_raw as usize) {
            continue;
        }

        match runtime
            .order_after_execution_binding(execution, operation)
            .map_err(|error| map_runtime_ordering_interop_error(error, operation))?
        {
            cudf::StreamDependencyOutcome::NoOp => {},
            cudf::StreamDependencyOutcome::Barrier(barrier) => barriers.push(barrier),
        }
    }

    Ok(barriers)
}

fn order_runtime_after_borrowed_edge_list(
    runtime: &Runtime,
    edge_list: &BorrowedEdgeList<'_>,
    operation: &'static str,
) -> Result<Vec<cudf::runtime::CudaEvent>> {
    let mut executions = vec![edge_list.src.execution(), edge_list.dst.execution()];
    if let Some(weights) = edge_list.weights.as_ref() {
        executions.push(weights.execution());
    }

    order_runtime_after_execution_bindings(runtime, executions, operation)
}

#[doc(hidden)]
pub mod testing {
    use rapids_interop::BorrowedColumnView;

    use super::{BorrowedEdgeList, BorrowedInputOrderingPlan, borrowed_input_ordering_plan};
    use crate::{error::Result, handle::EnteredContext};

    /// Compute the Phase 1 borrowed-input ordering plan for explicit borrowed
    /// columns.
    pub fn borrowed_columns_ordering_plan(
        handle: &EnteredContext<'_>,
        src: &BorrowedColumnView<'_>,
        dst: &BorrowedColumnView<'_>,
        weights: Option<&BorrowedColumnView<'_>>,
    ) -> Result<BorrowedInputOrderingPlan> {
        let edge_list = BorrowedEdgeList { src: src.clone(), dst: dst.clone(), weights: weights.cloned() };
        borrowed_input_ordering_plan(handle, &edge_list, "cudf_interop::testing::borrowed_columns_ordering_plan")
    }
}

fn map_runtime_ordering_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to establish runtime-stream ordering for cudf interop input",
        classify_runtime_ordering_interop_error,
    )
}

fn map_prepared_lease_activation_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to activate prepared export read lease",
        classify_prepared_lease_interop_error,
    )
}

fn map_prepared_lease_completion_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to complete prepared export read lease",
        classify_prepared_lease_interop_error,
    )
}

fn map_owned_import_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to import shared interop input into the consumer cudf runtime",
        classify_owned_import_interop_error,
    )
}

fn map_borrowed_projection_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to project cudf table view into borrowed RAPIDS interop",
        classify_projection_interop_error,
    )
}

fn map_prepared_projection_interop_error(error: InteropError, operation: &'static str) -> Error {
    map_interop_boundary_error(
        error,
        operation,
        "failed to prepare cudf-owned input for prepared device export",
        classify_projection_interop_error,
    )
}

fn classify_runtime_ordering_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn classify_prepared_lease_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn classify_owned_import_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn classify_projection_interop_error(kind: InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>) {
    match kind {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => (ErrorKind::UnsupportedType, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::NullMaskUnsupported => (ErrorKind::InvalidInput, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::NotFound
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => (ErrorKind::InvalidState, Some(ErrorStatus::Permanent)),
        InteropErrorKind::ReadinessUnavailable => (ErrorKind::Unexpected, None),
    }
}

fn map_interop_boundary_error(
    error: InteropError,
    operation: &'static str,
    message: &'static str,
    classify: fn(InteropErrorKind) -> (ErrorKind, Option<ErrorStatus>),
) -> Error {
    let (kind, explicit_status) = classify(error.kind());
    let mut mapped = Error::new(kind, operation).with_message(interop_boundary_message(message, error.message()));
    if let Some(status) = explicit_status.or_else(|| interop_error_source_status(&error)) {
        mapped = apply_error_status(mapped, status);
    }
    mapped = mapped.with_context("interop_kind", format!("{:?}", error.kind()));
    mapped = mapped.with_context("interop_message", error.message().to_string());

    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }

    mapped.set_source(error)
}

fn interop_boundary_message(boundary_message: &'static str, leaf_message: &str) -> String {
    let leaf_message = leaf_message.trim();
    if leaf_message.is_empty() { boundary_message.to_string() } else { format!("{boundary_message}: {leaf_message}") }
}

fn interop_error_source_status(error: &InteropError) -> Option<ErrorStatus> {
    let mut source = error.source();
    while let Some(current) = source {
        if let Some(cudf_error) = current.downcast_ref::<cudf::Error>() {
            return Some(match cudf_error.status() {
                cudf::ErrorStatus::Permanent => ErrorStatus::Permanent,
                cudf::ErrorStatus::Temporary => ErrorStatus::Temporary,
                cudf::ErrorStatus::Persistent => ErrorStatus::Persistent,
            });
        }
        source = current.source();
    }
    None
}

fn apply_error_status(mapped: Error, status: ErrorStatus) -> Error {
    match status {
        ErrorStatus::Permanent => mapped.set_permanent(),
        ErrorStatus::Temporary => mapped.set_temporary(),
        ErrorStatus::Persistent => mapped.set_persistent(),
    }
}

fn map_cudf_error_code(code: cudf::ErrorCode) -> ErrorKind {
    match code {
        cudf::ErrorCode::MissingColumn
        | cudf::ErrorCode::DuplicateColumnName
        | cudf::ErrorCode::DuplicateKey
        | cudf::ErrorCode::ColumnLengthMismatch
        | cudf::ErrorCode::TypeMismatch
        | cudf::ErrorCode::IndexOutOfBounds
        | cudf::ErrorCode::NullUnexpected
        | cudf::ErrorCode::CrossDeviceInputs
        | cudf::ErrorCode::CrossProvenanceInputs
        | cudf::ErrorCode::ForeignInvalidArgument => ErrorKind::InvalidInput,
        cudf::ErrorCode::EmptyConstructionRequiresRuntime
        | cudf::ErrorCode::RuntimeBindingMismatch
        | cudf::ErrorCode::ReadinessUnavailable
        | cudf::ErrorCode::ExecutionBindingUnavailable
        | cudf::ErrorCode::LeaseAlreadyActive
        | cudf::ErrorCode::InvariantViolation
        | cudf::ErrorCode::ForeignInvariantViolation => ErrorKind::InvalidState,
        cudf::ErrorCode::UnsupportedDtype
        | cudf::ErrorCode::UnsupportedLayout
        | cudf::ErrorCode::UnsupportedOperation => ErrorKind::UnsupportedType,
        cudf::ErrorCode::OutOfMemory | cudf::ErrorCode::ForeignResourceExhausted => ErrorKind::OutOfMemory,
        cudf::ErrorCode::IoFailure => ErrorKind::IoError,
        cudf::ErrorCode::CudaRuntime
        | cudf::ErrorCode::FatalCuda
        | cudf::ErrorCode::ForeignTransientDependencyFailure
        | cudf::ErrorCode::ForeignPersistentDependencyFailure => ErrorKind::Unexpected,
        _ => match cudf::core::error::kind_for_code(code) {
            cudf::ErrorKind::InvalidInput => ErrorKind::InvalidInput,
            cudf::ErrorKind::NotFound => ErrorKind::IoError,
            cudf::ErrorKind::Unsupported => ErrorKind::UnsupportedType,
            cudf::ErrorKind::ExecutionContext => ErrorKind::InvalidState,
            cudf::ErrorKind::ResourceExhausted => ErrorKind::OutOfMemory,
            cudf::ErrorKind::DependencyFailure => ErrorKind::Unexpected,
            cudf::ErrorKind::InvariantViolation => ErrorKind::InvalidState,
            _ => ErrorKind::Unexpected,
        },
    }
}

fn cudf_fact_context_key(key: cudf::core::error::FactKeyName) -> &'static str {
    match key {
        cudf::core::error::FactKeyName::ColumnName => "cudf_column_name",
        cudf::core::error::FactKeyName::ColumnIndex => "cudf_column_index",
        cudf::core::error::FactKeyName::ExpectedRows => "cudf_expected_rows",
        cudf::core::error::FactKeyName::ActualRows => "cudf_actual_rows",
        cudf::core::error::FactKeyName::ExpectedCount => "cudf_expected_count",
        cudf::core::error::FactKeyName::ActualCount => "cudf_actual_count",
        cudf::core::error::FactKeyName::LeftTypeId => "cudf_left_type_id",
        cudf::core::error::FactKeyName::RightTypeId => "cudf_right_type_id",
        cudf::core::error::FactKeyName::LeftDevice => "cudf_left_device",
        cudf::core::error::FactKeyName::RightDevice => "cudf_right_device",
        cudf::core::error::FactKeyName::TableRowCount => "cudf_table_row_count",
        cudf::core::error::FactKeyName::TableColumnCount => "cudf_table_column_count",
        cudf::core::error::FactKeyName::KeysCount => "cudf_keys_count",
        cudf::core::error::FactKeyName::DistinctKeysCount => "cudf_distinct_keys_count",
        cudf::core::error::FactKeyName::TypeId => "cudf_type_id",
        _ => "cudf_fact",
    }
}

fn map_cudf_status(mut mapped: Error, status: cudf::ErrorStatus) -> Error {
    match status {
        cudf::ErrorStatus::Permanent => mapped = mapped.set_permanent(),
        cudf::ErrorStatus::Temporary => mapped = mapped.set_temporary(),
        cudf::ErrorStatus::Persistent => mapped = mapped.set_persistent(),
    }
    mapped
}

fn attach_cudf_error_context(mut mapped: Error, error: &cudf::Error) -> Error {
    mapped = mapped.with_context("cudf_code", error.code().to_string());
    mapped = mapped.with_context("cudf_kind", error.kind().to_string());
    mapped = mapped.with_context("cudf_status", error.status().to_string());
    mapped = mapped.with_context("cudf_operation", error.operation().to_string());

    for called in error.call_chain() {
        mapped = mapped.with_context("cudf_called", (*called).to_string());
    }

    for (key, value) in error.facts().iter() {
        mapped = mapped.with_context(cudf_fact_context_key(key), value.to_string());
    }

    for (key, value) in error.details() {
        mapped = mapped.with_context("cudf_detail", format!("{key}={value}"));
    }

    mapped
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
    let source_code = error.code();
    let source_status = error.status();
    let mut mapped = Error::new(map_cudf_error_code(source_code), operation).with_message(message.into());
    mapped = map_cudf_status(mapped, source_status);
    mapped = attach_cudf_error_context(mapped, &error);

    for (key, value) in context {
        mapped = mapped.with_context(key, value);
    }

    mapped.set_source(error)
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
    handle: &EnteredContext<'_>,
    producer_device: i32,
    columns: impl IntoIterator<Item = (&'a str, i32)>,
    input_kind: &str,
    operation: &'static str,
) -> Result<()> {
    for (name, device) in columns {
        if device != producer_device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("{input_kind} graph inputs must all reside on the same CUDA device"))
                .with_context("producer_device", producer_device.to_string())
                .with_context("column_name", name.to_string())
                .with_context("column_device", device.to_string()));
        }
    }

    if producer_device != handle.device() {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
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
    handle: &EnteredContext<'_>,
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
    handle: &EnteredContext<'_>,
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
        // SAFETY: `stream` comes from cudf execution metadata for a live
        // borrowed input column on the currently selected CUDA device.
        check_cudf_status(unsafe { cudf_sys::cudf_raw_cuda_stream_synchronize(stream) }, operation)?;
    }

    Ok(())
}

fn order_borrowed_inputs_for_handle(
    handle: &EnteredContext<'_>,
    edge_list: &BorrowedEdgeList<'_>,
    operation: &'static str,
) -> Result<Vec<cudf::runtime::CudaEvent>> {
    match handle.runtime() {
        Some(runtime) => order_runtime_after_borrowed_edge_list(runtime, edge_list, operation),
        None => {
            fence_producer_streams(edge_list, operation)?;
            Ok(Vec::new())
        },
    }
}

fn prepared_lease_authorities<'a>(
    handle: &'a EnteredContext<'_>,
    operation: &'static str,
) -> Result<(ReadStartAuthority<'a>, CompletionAuthority<'a>)> {
    match handle.runtime() {
        Some(runtime) => {
            let stream = cudf::interop::cuda_stream_handle(runtime.stream())
                .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;
            Ok((ReadStartAuthority::ConsumerStream { stream }, CompletionAuthority::SameDeviceStream { stream }))
        },
        None => Ok((ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)),
    }
}

fn build_graph_from_borrowed_edge_list_raw<'a, T: Transposition>(
    handle: &EnteredContext<'_>,
    edge_list: &BorrowedEdgeList<'a>,
    properties: GraphProperties,
    renumber: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    validate_edge_list_policy(handle, edge_list, operation)?;
    let src = borrowed_column_to_raw(&edge_list.src, "src", operation)?;
    let dst = borrowed_column_to_raw(&edge_list.dst, "dst", operation)?;
    let wgt = edge_list.weights.as_ref().map(|column| borrowed_column_to_raw(column, "wgt", operation)).transpose()?;
    let _barriers = order_borrowed_inputs_for_handle(handle, edge_list, operation)?;

    let mut builder = GraphBuilder::from_raw_views(properties, &src, &dst)?.renumber_flag(renumber);
    if let Some(ref weights) = wgt {
        builder = builder.weights_raw(weights)?;
    }
    builder.build_inner(handle, T::store_transposed())
}

fn complete_prepared_lease(
    result: std::result::Result<LeaseCompletionResult, InteropError>,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    operation: &'static str,
) -> Result<Option<RetainedPreparedCompletion>> {
    match result.map_err(|error| map_prepared_lease_completion_interop_error(error, operation))? {
        LeaseCompletionResult::FallbackSynchronized => Ok(None),
        LeaseCompletionResult::Completed(witness) => match retained_owner {
            Some(owner) => Ok(Some(RetainedPreparedCompletion { owner, witness })),
            None => witness
                .synchronize()
                .map_err(|error| map_prepared_lease_completion_interop_error(error, operation))
                .map(|()| None),
        },
    }
}

fn finish_prepared_graph_result<T: Transposition>(
    graph_result: Result<Graph<T>>,
    completions: impl IntoIterator<Item = Result<Option<RetainedPreparedCompletion>>>,
) -> Result<Graph<T>> {
    let mut completion_error = None;
    let mut retained_owners = Vec::new();
    let mut retained_witnesses = Vec::new();
    for completion in completions {
        match completion {
            Ok(Some(retained)) => {
                retained_owners.push(retained.owner);
                retained_witnesses.push(retained.witness);
            },
            Ok(None) => {},
            Err(error) => {
                if completion_error.is_none() {
                    completion_error = Some(error);
                }
            },
        }
    }
    let pending_input_completion = DeferredInputCompletion::new(retained_owners, retained_witnesses);

    match graph_result {
        Ok(graph) => {
            if let Some(error) = completion_error {
                if let Some(pending) = pending_input_completion {
                    pending.synchronize();
                }
                Err(error)
            } else if let Some(pending) = pending_input_completion {
                Ok(graph.with_pending_input_completion(pending))
            } else {
                Ok(graph)
            }
        },
        Err(error) => {
            if let Some(pending) = pending_input_completion {
                pending.synchronize();
            }
            Err(error)
        },
    }
}

fn canonical_edge_dataframe_from_owned_series(
    src: &cudf::Series,
    dst: &cudf::Series,
    weights: Option<&cudf::Series>,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let mut pairs = vec![(NORMALIZED_SRC_COL.to_string(), src.clone()), (NORMALIZED_DST_COL.to_string(), dst.clone())];
    if let Some(weights) = weights {
        pairs.push((NORMALIZED_WEIGHT_COL.to_string(), weights.clone()));
    }

    dataframe_from_named_series(pairs, operation)
        .map_err(|error| error.with_message("failed to assemble canonical cudf edge DataFrame for graph normalization"))
}

fn validate_table_column_selection(
    num_cols: usize,
    options: &TableGraphOptions,
    table_kind: &str,
    operation: &'static str,
) -> Result<()> {
    let validate_col_index = |idx: usize, name: &str| -> Result<()> {
        if idx >= num_cols {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} column index {} is out of bounds ({} has {} columns)",
                    name, idx, table_kind, num_cols
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

fn canonical_edge_dataframe_from_imported_table(
    table: Table,
    options: &TableGraphOptions,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let num_columns = table
        .num_columns()
        .map_err(|error| map_cudf_error_with(error, operation, "failed to inspect imported cudf table schema", []))?;
    validate_table_column_selection(num_columns, options, "imported table", operation)?;

    let names = (0..num_columns).map(|index| format!("column_{index}")).collect::<Vec<_>>();
    let imported = cudf::DataFrame::from_table(table, names.clone()).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to wrap imported cudf table for graph normalization",
            std::iter::empty(),
        )
    })?;

    let src = imported.series(&names[options.src_col]).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to select source column from imported cudf table", [])
    })?;
    let dst = imported.series(&names[options.dst_col]).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to select destination column from imported cudf table", [])
    })?;
    let weights = options
        .weight_col
        .map(|index| {
            imported.series(&names[index]).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to select weight column from imported cudf table", [])
            })
        })
        .transpose()?;

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)
}

fn normalize_python_cugraph_dataframe(
    df: &cudf::DataFrame,
    has_weights: bool,
    properties: GraphProperties,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let base = if properties.is_symmetric {
        let reverse_weights = if has_weights {
            Some(df.series(NORMALIZED_WEIGHT_COL).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to access weight column for reverse-edge expansion", [])
            })?)
        } else {
            None
        };
        let reverse = canonical_edge_dataframe_from_owned_series(
            &df.series(NORMALIZED_DST_COL).map_err(|error| {
                map_cudf_error_with(
                    error,
                    operation,
                    "failed to access destination column for reverse-edge expansion",
                    [],
                )
            })?,
            &df.series(NORMALIZED_SRC_COL).map_err(|error| {
                map_cudf_error_with(error, operation, "failed to access source column for reverse-edge expansion", [])
            })?,
            reverse_weights.as_ref(),
            operation,
        )?;

        df.concat(&reverse).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to append reverse edges for Python-style graph normalization",
                [],
            )
        })?
    } else {
        df.clone()
    };

    if properties.is_multigraph {
        return Ok(base);
    }

    let mut sort_by = vec![NORMALIZED_SRC_COL, NORMALIZED_DST_COL];
    if has_weights {
        sort_by.push(NORMALIZED_WEIGHT_COL);
    }

    let sorted = base.sort_values(sort_by, SortOptions::new().with_stable(true)).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to sort edges for Python-style graph normalization", [])
    })?;
    let deduped = sorted
        .drop_duplicates(
            DropDuplicatesOptions::builder()
                .subset(&[NORMALIZED_SRC_COL, NORMALIZED_DST_COL][..])
                .keep(DuplicateKeepOption::KeepFirst)
                .build(),
        )
        .map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to collapse duplicate edges for Python-style graph normalization",
                [],
            )
        })?;

    if has_weights {
        deduped.select([NORMALIZED_SRC_COL, NORMALIZED_DST_COL, NORMALIZED_WEIGHT_COL]).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to project normalized weighted edge columns into canonical order",
                [],
            )
        })
    } else {
        deduped.select([NORMALIZED_SRC_COL, NORMALIZED_DST_COL]).map_err(|error| {
            map_cudf_error_with(error, operation, "failed to project normalized edge columns into canonical order", [])
        })
    }
}

fn build_graph_from_owned_dataframe_raw<T: Transposition>(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
    properties: GraphProperties,
    renumber: bool,
    has_weights: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    let export = df
        .try_prepare_device_table_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
    let retained_owner = export.clone_retained_owner();
    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let lease = export
        .try_activate_read_lease(read_start, completion)
        .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;

    let columns = lease.export().columns();
    let bindings = lease.export().bindings();
    let edge_list = PreparedEdgeList {
        src: PreparedEdgeColumn { column: &columns[0], bindings },
        dst: PreparedEdgeColumn { column: &columns[1], bindings },
        weights: if has_weights { Some(PreparedEdgeColumn { column: &columns[2], bindings }) } else { None },
    };

    let graph_result = build_graph_from_prepared_edge_list_raw(handle, &edge_list, properties, renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), retained_owner, operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}

fn build_python_normalized_graph_from_dataframe<T: Transposition>(
    handle: &EnteredContext<'_>,
    df: &cudf::DataFrame,
    properties: GraphProperties,
    renumber: bool,
    has_weights: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    let normalized = normalize_python_cugraph_dataframe(df, has_weights, properties, operation)?;
    build_graph_from_owned_dataframe_raw(handle, &normalized, properties, renumber, has_weights, operation)
}

fn build_graph_from_prepared_edge_list_raw<'a, T: Transposition>(
    handle: &EnteredContext<'_>,
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

fn canonical_edge_dataframe_from_borrowed_edge_list(
    handle: &EnteredContext<'_>,
    edge_list: &BorrowedEdgeList<'_>,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    validate_edge_list_policy(handle, edge_list, operation)?;
    let _barriers = order_borrowed_inputs_for_handle(handle, edge_list, operation)?;

    let owned_runtime;
    let runtime = if let Some(runtime) = handle.runtime() {
        runtime
    } else {
        owned_runtime = runtime_for_handle(handle, operation)?;
        &owned_runtime
    };
    let src = cudf::Series::try_import_from_borrowed_column_view_into_runtime(&edge_list.src, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let dst = cudf::Series::try_import_from_borrowed_column_view_into_runtime(&edge_list.dst, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let weights = edge_list
        .weights
        .as_ref()
        .map(|column| {
            cudf::Series::try_import_from_borrowed_column_view_into_runtime(column, runtime)
                .map_err(|error| map_owned_import_interop_error(error, operation))
        })
        .transpose()?;

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)
}

fn canonical_edge_dataframe_from_prepared_exports(
    handle: &EnteredContext<'_>,
    src: &PreparedColumnExport<'_>,
    dst: &PreparedColumnExport<'_>,
    weights: Option<&PreparedColumnExport<'_>>,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let owned_runtime;
    let runtime = if let Some(runtime) = handle.runtime() {
        runtime
    } else {
        owned_runtime = runtime_for_handle(handle, operation)?;
        &owned_runtime
    };
    let src = cudf::Series::try_import_from_prepared_column_export_into_runtime(src, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let dst = cudf::Series::try_import_from_prepared_column_export_into_runtime(dst, runtime)
        .map_err(|error| map_owned_import_interop_error(error, operation))?;
    let weights = weights
        .map(|column| {
            cudf::Series::try_import_from_prepared_column_export_into_runtime(column, runtime)
                .map_err(|error| map_owned_import_interop_error(error, operation))
        })
        .transpose()?;

    canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph construction functions
// ═══════════════════════════════════════════════════════════════════════════

/// Build a cuGraph [`Graph`] directly from borrowed RAPIDS column descriptors.
pub fn graph_from_borrowed_columns<T: Transposition>(
    handle: &EnteredContext<'_>,
    src: &BorrowedColumnView<'_>,
    dst: &BorrowedColumnView<'_>,
    weights: Option<&BorrowedColumnView<'_>>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let edge_list = BorrowedEdgeList { src: src.clone(), dst: dst.clone(), weights: weights.cloned() };
        let canonical =
            canonical_edge_dataframe_from_borrowed_edge_list(handle, &edge_list, "graph_from_borrowed_columns")?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            weights.is_some(),
            "graph_from_borrowed_columns",
        );
    }

    let edge_list = BorrowedEdgeList { src: src.clone(), dst: dst.clone(), weights: weights.cloned() };
    build_graph_from_borrowed_edge_list_raw(
        handle,
        &edge_list,
        options.properties,
        options.renumber,
        "graph_from_borrowed_columns",
    )
}

/// Build a cuGraph [`Graph`] directly from a borrowed RAPIDS table descriptor.
pub fn graph_from_borrowed_table_view<T: Transposition>(
    handle: &EnteredContext<'_>,
    view: &BorrowedTableView<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_borrowed_table_view";
    let edge_list = borrowed_table_to_edge_list(view, options, operation)?;
    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_borrowed_edge_list(handle, &edge_list, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    build_graph_from_borrowed_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation)
}

/// Build a cuGraph [`Graph`] directly from prepared-device RAPIDS column exports.
///
/// This is the primary single-GPU shared-contract path. Runtime-bound handles
/// activate one read lease per input export on the consumer stream and retain
/// same-device completion ordering explicitly. Non-runtime-bound handles fall
/// back to synchronous-by-return lease authorities.
///
/// When [`GraphConstructionPolicy::PythonCugraph`] is selected, this function
/// may first import the edge columns into owned cudf objects on the consumer
/// device so Python-style normalization can run before raw graph construction.
pub fn graph_from_prepared_columns<T: Transposition>(
    handle: &EnteredContext<'_>,
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

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_prepared_exports(handle, src, dst, weights, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            weights.is_some(),
            operation,
        );
    }

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let src_retained_owner = src.clone_retained_owner();
    let dst_retained_owner = dst.clone_retained_owner();
    let wgt_retained_owner = weights.and_then(|weights| weights.clone_retained_owner());
    let mut src_lease = Some(
        src.try_activate_read_lease(read_start, completion)
            .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?,
    );
    let mut dst_lease = None;
    let mut wgt_lease = None;

    let graph_result = (|| {
        dst_lease = Some(
            dst.try_activate_read_lease(read_start, completion)
                .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?,
        );
        wgt_lease = weights
            .map(|weights| {
                weights
                    .try_activate_read_lease(read_start, completion)
                    .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))
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

        build_graph_from_prepared_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation)
    })();

    let src_complete = src_lease
        .take()
        .map(|lease| complete_prepared_lease(lease.complete(), src_retained_owner, operation))
        .unwrap_or(Ok(None));
    let dst_complete = dst_lease
        .take()
        .map(|lease| complete_prepared_lease(lease.complete(), dst_retained_owner, operation))
        .unwrap_or(Ok(None));
    let wgt_complete = wgt_lease
        .take()
        .map(|lease| complete_prepared_lease(lease.complete(), wgt_retained_owner, operation))
        .unwrap_or(Ok(None));

    finish_prepared_graph_result(graph_result, [src_complete, dst_complete, wgt_complete])
}

fn validate_prepared_table_column_selection(
    num_cols: usize,
    options: &TableGraphOptions,
    operation: &'static str,
) -> Result<()> {
    validate_table_column_selection(num_cols, options, "prepared table export", operation)
}

/// Build a cuGraph [`Graph`] directly from a prepared-device RAPIDS table export.
///
/// When [`GraphConstructionPolicy::PythonCugraph`] is selected, this function
/// may first import the selected edge columns into owned cudf objects on the
/// consumer device so Python-style normalization can run before raw graph
/// construction.
pub fn graph_from_prepared_table_export<T: Transposition>(
    handle: &EnteredContext<'_>,
    export: &PreparedTableExport<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_prepared_table_export";
    prepared_profile_supported(export.diagnostics().profile(), operation)?;
    validate_prepared_table_column_selection(export.columns().len(), options, operation)?;

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let owned_runtime;
        let runtime = if let Some(runtime) = handle.runtime() {
            runtime
        } else {
            owned_runtime = runtime_for_handle(handle, operation)?;
            &owned_runtime
        };
        let imported = Table::try_import_from_prepared_table_export_into_runtime(export, runtime)
            .map_err(|error| map_owned_import_interop_error(error, operation))?;
        let canonical = canonical_edge_dataframe_from_imported_table(imported, options, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    let (read_start, completion) = prepared_lease_authorities(handle, operation)?;
    let retained_owner = export.clone_retained_owner();
    let lease = export
        .try_activate_read_lease(read_start, completion)
        .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;

    let columns = lease.export().columns();
    let bindings = lease.export().bindings();
    let edge_list = PreparedEdgeList {
        src: PreparedEdgeColumn { column: &columns[options.src_col], bindings },
        dst: PreparedEdgeColumn { column: &columns[options.dst_col], bindings },
        weights: options.weight_col.map(|index| PreparedEdgeColumn { column: &columns[index], bindings }),
    };

    let graph_result =
        build_graph_from_prepared_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation);
    let lease_complete = complete_prepared_lease(lease.complete(), retained_owner, operation);

    finish_prepared_graph_result(graph_result, [lease_complete])
}

/// Build a cuGraph [`Graph`] from a cudf low-level [`TableView`].
///
/// **Borrowed compatibility path**: this entry point projects the cudf table
/// view into the shared RAPIDS borrowed ABI. Runtime-bound handles order the
/// consumer stream after same-device producer streams with device-side CUDA
/// dependencies; non-runtime-bound handles may still perform synchronous
/// host-side fencing before cuGraph reads the borrowed device buffers.
///
/// For new single-GPU integrations, prefer
/// [`graph_from_prepared_table_export()`], which keeps producer read-start,
/// completion, and liveness explicit at the shared-contract layer. This
/// function remains the supported compatibility path for callers that already
/// operate on low-level cudf table views and want an immediate bridge into the
/// explicit RAPIDS borrowed ABI.
///
/// **Raw-policy input path**: under [`GraphConstructionPolicy::RawLibcugraph`],
/// cuGraph reads the edge data directly from the table's column device pointers
/// and builds its own internal CSR/CSC representation. Under
/// [`GraphConstructionPolicy::PythonCugraph`], this function may first import
/// the selected edge columns into owned cudf objects on the consumer device so
/// Python-style normalization can run before raw graph construction. The
/// returned [`Graph`] is independent of the input table.
///
/// # Errors
///
/// Returns an error if:
/// - any column index is out of bounds
/// - selected columns contain null values
/// - column types are incompatible with cugraph
/// - graph construction fails
///
/// [`TableView`]: cudf::core::table::TableView
pub fn graph_from_table_view<T: Transposition>(
    handle: &EnteredContext<'_>,
    view: &TableView<'_>,
    options: &TableGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_table_view";
    let borrowed =
        view.try_borrow_table_view().map_err(|error| map_borrowed_projection_interop_error(error, operation))?;
    let edge_list = borrowed_table_to_edge_list(&borrowed, options, operation)?;
    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_borrowed_edge_list(handle, &edge_list, operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    build_graph_from_borrowed_edge_list_raw(handle, &edge_list, options.properties, options.renumber, operation)
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
    handle: &EnteredContext<'_>,
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
        construction_policy: options.construction_policy,
    };

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let src = df.series(options.src_col).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to project source column for Python-style graph normalization",
                [],
            )
        })?;
        let dst = df.series(options.dst_col).map_err(|error| {
            map_cudf_error_with(
                error,
                operation,
                "failed to project destination column for Python-style graph normalization",
                [],
            )
        })?;
        let weights = options
            .weight_col
            .map(|name| {
                df.series(name).map_err(|error| {
                    map_cudf_error_with(
                        error,
                        operation,
                        "failed to project weight column for Python-style graph normalization",
                        [],
                    )
                })
            })
            .transpose()?;

        let canonical = canonical_edge_dataframe_from_owned_series(&src, &dst, weights.as_ref(), operation)?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            options.weight_col.is_some(),
            operation,
        );
    }

    let export = df
        .try_prepare_device_table_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
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
/// let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
/// let handle =
///     ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
/// let entered = handle.enter("graph_from_series docs").unwrap();
/// let src = runtime.series(&[0i32, 1, 2]).unwrap();
/// let dst = runtime.series(&[1i32, 2, 0]).unwrap();
///
/// let opts = SeriesGraphOptions::default().renumber();
/// let graph = graph_from_series::<Forward>(&entered, &src, &dst, None, &opts)
///     .unwrap();
/// ```
///
/// [`Series`]: cudf::Series
pub fn graph_from_series<T: Transposition>(
    handle: &EnteredContext<'_>,
    src: &cudf::Series,
    dst: &cudf::Series,
    weights: Option<&cudf::Series>,
    options: &SeriesGraphOptions,
) -> Result<Graph<T>> {
    let operation = "graph_from_series";
    let src_export = src
        .try_prepare_device_column_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
    let dst_export = dst
        .try_prepare_device_column_export()
        .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
    let wgt_export = weights
        .map(|column| {
            column
                .try_prepare_device_column_export()
                .map_err(|error| map_prepared_projection_interop_error(error, operation))
        })
        .transpose()?;

    if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
        let canonical = canonical_edge_dataframe_from_prepared_exports(
            handle,
            &src_export,
            &dst_export,
            wgt_export.as_ref(),
            operation,
        )?;
        return build_python_normalized_graph_from_dataframe(
            handle,
            &canonical,
            options.properties,
            options.renumber,
            weights.is_some(),
            operation,
        );
    }

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
    handle: &EnteredContext<'_>,
    options: &ParquetGraphOptions<'_>,
) -> Result<LoadedGraph<T>> {
    let operation = "graph_from_parquet";
    let owned_runtime;
    let runtime = if let Some(runtime) = handle.runtime() {
        runtime
    } else {
        owned_runtime = runtime_for_handle(handle, operation)?;
        &owned_runtime
    };

    let reader_opts = parquet::ParquetTableReadOptions::builder().filepath(options.path).with_runtime(runtime).build();

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
        let table_opts = TableGraphOptions {
            src_col: options.src_col,
            dst_col: options.dst_col,
            weight_col: options.weight_col,
            properties: options.properties,
            renumber: options.renumber,
            construction_policy: options.construction_policy,
        };
        if options.construction_policy == GraphConstructionPolicy::PythonCugraph {
            let canonical = canonical_edge_dataframe_from_imported_table(
                table.deep_copy().map_err(|error| {
                    map_cudf_error_with(
                        error,
                        operation,
                        "failed to deep-copy loaded cudf table for Python-style graph normalization",
                        [],
                    )
                })?,
                &table_opts,
                operation,
            )?;
            build_python_normalized_graph_from_dataframe(
                handle,
                &canonical,
                options.properties,
                options.renumber,
                options.weight_col.is_some(),
                operation,
            )?
        } else {
            let export = table
                .try_prepare_device_table_export()
                .map_err(|error| map_prepared_projection_interop_error(error, operation))?;
            graph_from_prepared_table_export::<T>(handle, &export, &table_opts)?
        }
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
    use crate::ResourceHandle;

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
    fn map_cudf_error_missing_column_is_invalid_input() {
        let cudf_err = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::MissingColumn,
            "cudf::testing::missing_column",
            "column 'foo' not found",
        );
        let err =
            map_cudf_error_with(cudf_err, "test_op", "input adapter failed", [("column_role", "src".to_string())]);
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.is_permanent(), "missing columns should be permanent");
        assert_eq!(err.message(), "input adapter failed");
        assert!(err.to_string().contains("cudf_code"), "mapped error should retain cudf metadata");
    }

    #[test]
    fn map_cudf_error_runtime_binding_mismatch_is_invalid_state() {
        let cudf_err = cudf::testing::error::reanchor_for_test(
            cudf::testing::error::new_with_for_test(
                cudf::ErrorCode::RuntimeBindingMismatch,
                "cudf::testing::runtime_binding::inner",
                "runtime must not use ambient provenance",
            ),
            "cudf::testing::runtime_binding::outer",
        );
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "runtime binding mismatches should be permanent");
        assert!(err.to_string().contains("cudf_called"), "call-chain breadcrumbs should be preserved");
    }

    #[test]
    fn map_cudf_error_cuda_runtime_is_temporary_unexpected() {
        let cudf_err = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::CudaRuntime,
            "cudf::testing::cuda_runtime",
            "device launch failed",
        );
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "CUDA runtime failures should be temporary");
    }

    #[test]
    fn map_cudf_error_out_of_memory() {
        let cudf_err =
            cudf::testing::error::new_with_for_test(cudf::ErrorCode::OutOfMemory, "cudf::testing::oom", "alloc failed");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.is_temporary(), "OOM should be temporary");
    }

    #[test]
    fn map_cudf_error_io_error() {
        let cudf_err =
            cudf::testing::error::new_with_for_test(cudf::ErrorCode::IoFailure, "cudf::testing::io", "file not found");
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::IoError);
        assert!(err.is_permanent(), "I/O errors should be permanent by default");
    }

    #[test]
    fn map_cudf_error_fatal_cuda_is_persistent_unexpected() {
        let cudf_err = cudf::testing::error::new_with_for_test(
            cudf::ErrorCode::FatalCuda,
            "cudf::testing::fatal_cuda",
            "device entered fatal state",
        );
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_persistent(), "fatal CUDA failures should be persistent");
    }

    #[test]
    fn map_cudf_error_foreign_invariant_violation_is_invalid_state() {
        let cudf_err = cudf::testing::error::foreign_invariant_violation_for_test();
        let err = map_cudf_error(cudf_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "foreign invariant violations should be permanent");
        assert!(err.to_string().contains("cudf_detail"), "foreign details should be preserved");
    }

    #[test]
    fn runtime_ordering_interop_execution_binding_unavailable_is_invalid_state() {
        let interop_err = InteropError::new(
            InteropErrorKind::ExecutionBindingUnavailable,
            "producer must provide explicit execution provenance",
        );
        let err = map_runtime_ordering_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "missing runtime ordering provenance is a contract failure, not a retryable one");
    }

    #[test]
    fn runtime_ordering_interop_readiness_unavailable_uses_cudf_source_status() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to query producer event")
            .set_source(cudf::testing::error::new_with_for_test(
                cudf::ErrorCode::CudaRuntime,
                "cudf::testing::cuda_runtime",
                "device launch failed",
            ));
        let err = map_runtime_ordering_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "runtime ordering should inherit retryability from the underlying CUDA failure");
    }

    #[test]
    fn runtime_ordering_interop_readiness_unavailable_without_source_is_permanent() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to query producer event");
        let err = map_runtime_ordering_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(
            err.is_permanent(),
            "runtime ordering must not invent retryability when the interop boundary did not prove it"
        );
    }

    #[test]
    fn prepared_lease_activation_interop_lease_already_active_is_invalid_state() {
        let interop_err = InteropError::new(
            InteropErrorKind::LeaseAlreadyActive,
            "prepared export lease activation requires an inactive export root",
        );
        let err = map_prepared_lease_activation_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "lease lifecycle violations should remain permanent invalid-state failures");
    }

    #[test]
    fn prepared_lease_activation_interop_preserves_leaf_failure_message() {
        let interop_err = InteropError::new(InteropErrorKind::InvalidView, "test backend forced activation failure");
        let err = map_prepared_lease_activation_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(
            err.message().contains("test backend forced activation failure"),
            "top-level error message should preserve the leaf failure reason for fast diagnosis"
        );
    }

    #[test]
    fn prepared_lease_activation_interop_without_leaf_message_keeps_boundary_message_clean() {
        let interop_err = InteropError::new(InteropErrorKind::InvalidView, "");
        let err = map_prepared_lease_activation_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.message(), "failed to activate prepared export read lease");
    }

    #[test]
    fn prepared_lease_completion_interop_readiness_unavailable_uses_persistent_source_status() {
        let interop_err =
            InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to finalize producer lease").set_source(
                cudf::testing::error::new_with_for_test(
                    cudf::ErrorCode::FatalCuda,
                    "cudf::testing::fatal_cuda",
                    "device entered fatal state",
                ),
            );
        let err = map_prepared_lease_completion_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_persistent(), "lease completion should preserve persistent dependency failure semantics");
    }

    #[test]
    fn owned_import_interop_execution_binding_mismatch_is_invalid_input() {
        let interop_err = InteropError::new(
            InteropErrorKind::ExecutionBindingMismatch,
            "prepared export import requires the destination runtime to use the same CUDA device",
        );
        let err = map_owned_import_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.is_permanent(), "cross-device import mismatches are caller input failures");
    }

    #[test]
    fn borrowed_projection_interop_invalid_view_is_invalid_state() {
        let interop_err =
            InteropError::new(InteropErrorKind::InvalidView, "borrowed table view is internally inconsistent");
        let err = map_borrowed_projection_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert!(err.is_permanent(), "borrowed projection invariants should surface as invalid-state failures");
    }

    #[test]
    fn prepared_projection_interop_readiness_unavailable_uses_temporary_source_status() {
        let interop_err = InteropError::new(InteropErrorKind::ReadinessUnavailable, "failed to project device export")
            .set_source(cudf::testing::error::new_with_for_test(
                cudf::ErrorCode::OutOfMemory,
                "cudf::testing::oom",
                "allocation failed",
            ));
        let err = map_prepared_projection_interop_error(interop_err, "test_op");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(err.is_temporary(), "projection should preserve temporary source failures instead of flattening them");
    }

    #[test]
    fn validate_edge_list_devices_mixed_devices_is_invalid_input() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("validate_edge_list_devices_mixed_devices_is_invalid_input").expect("entered");
        let err = validate_edge_list_devices(
            &entered,
            handle.device(),
            [("dst", handle.device() + 1)],
            "borrowed",
            "test_op",
        )
        .expect_err("mixed-device inputs should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn validate_edge_list_devices_handle_mismatch_is_invalid_input() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("validate_edge_list_devices_handle_mismatch_is_invalid_input").expect("entered");
        let err = validate_edge_list_devices(
            &entered,
            handle.device() + 1,
            std::iter::empty::<(&str, i32)>(),
            "prepared",
            "test_op",
        )
        .expect_err("producer and consumer device mismatch should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
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
