//! Zero-copy interop between cudf [`DeviceSlice`] and cugraph [`DeviceBuffer`] / [`DeviceArrayView`].
//!
//! This module is only available when the `cudf-interop` feature is enabled.
//!
//! [`DeviceSlice`]: cudf::DeviceSlice
//! [`DeviceBuffer`]: crate::array::DeviceBuffer
//! [`DeviceArrayView`]: crate::array::DeviceArrayView

use crate::array::{DeviceArrayView, DeviceBuffer};
use crate::error::{Error, ErrorKind, Result};
use crate::graph::Graph;
use crate::handle::ResourceHandle;
use crate::types::{DataType, GraphProperties};
use cudf::column::DeviceSlice;
use cudf::io::parquet;
use cudf::types::TypeId;

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

// SAFETY: `DeviceSlice` exposes a valid device pointer, length, and type that
// are fully compatible with cugraph's type-erased device array API. The pointer
// is GPU memory owned by the cudf column (not host-dereferenceable), and the
// type mapping is verified at construction time via `cudf_type_to_cugraph`.
unsafe impl DeviceBuffer for DeviceSlice {
    fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn dtype(&self) -> Result<DataType> {
        let type_id = self.type_id();
        cudf_type_to_cugraph(type_id).ok_or_else(|| {
            Error::new(ErrorKind::UnsupportedType, "DeviceSlice::dtype").with_message(format!(
                "cudf TypeId {:?} has no cugraph DataType equivalent",
                type_id
            ))
        })
    }
}

/// Extension trait for creating a [`DeviceArrayView`] from a cudf [`DeviceSlice`].
pub trait DeviceSliceExt {
    /// Create a zero-copy [`DeviceArrayView`] from this [`DeviceSlice`].
    fn as_cugraph_view(&self) -> Result<DeviceArrayView<'_>>;
}

impl DeviceSliceExt for DeviceSlice {
    fn as_cugraph_view(&self) -> Result<DeviceArrayView<'_>> {
        DeviceArrayView::from_device_buffer(self)
    }
}

/// Extension trait for converting a cugraph [`DeviceArrayView`] into a cudf [`Column`].
pub trait DeviceArrayViewExt {
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

impl DeviceArrayViewExt for DeviceArrayView<'_> {
    fn to_cudf_column(&self) -> Result<cudf::Column> {
        let dtype = self.dtype()?;
        let cudf_type = cugraph_type_to_cudf(dtype).ok_or_else(|| {
            Error::new(
                ErrorKind::UnsupportedType,
                "DeviceArrayViewExt::to_cudf_column",
            )
            .with_message(format!(
                "cugraph DataType {:?} has no cudf TypeId equivalent",
                dtype
            ))
        })?;

        // SAFETY: `self.device_ptr()` points to valid GPU memory containing
        // `self.len()` elements of `dtype`, which we have verified maps to
        // `cudf_type`. `Column::from_device_ptr` performs a device-to-device
        // copy, so the returned column is independent of this view.
        let col = unsafe {
            cudf::Column::from_device_ptr(
                self.device_ptr() as *const std::ffi::c_void,
                self.len(),
                cudf_type,
            )
        }
        .map_err(|e| {
            Error::new(ErrorKind::Unexpected, "DeviceArrayViewExt::to_cudf_column")
                .with_message(format!("cudf Column::from_device_ptr failed: {}", e))
        })?;

        Ok(col)
    }
}

/// Read a Parquet edge-list and build a cuGraph [`Graph`] on the GPU.
///
/// This convenience helper eliminates the ~20 lines of boilerplate that every
/// Parquet-to-Graph workflow repeats: read table → extract column views →
/// create DeviceSlices → convert to cugraph views → build Graph.
///
/// The returned [`cudf::Table`] lets callers reuse the loaded edge-list data
/// (for additional algorithms or inspection) without re-reading Parquet.
///
/// Graph construction itself is synchronous: once this function returns, the
/// [`Graph`] owns its internal representation and no longer depends on the
/// table's buffers.
///
/// # Arguments
///
/// - `handle` — resource handle for GPU operations
/// - `path` — file-system path to the Parquet file
/// - `src_col` — zero-based column index for source vertex ids
/// - `dst_col` — zero-based column index for destination vertex ids
/// - `wgt_col` — optional zero-based column index for edge weights
/// - `props` — graph properties (symmetric / multigraph)
/// - `store_transposed` — whether to store the graph transposed (needed by
///   some algorithms like PageRank)
/// - `renumber` — whether to renumber vertices (strongly recommended when
///   vertex IDs are not sequential from 0)
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
/// use cugraph::cudf_interop::graph_from_parquet;
/// use cugraph::handle::ResourceHandle;
/// use cugraph::types::GraphProperties;
///
/// let handle = ResourceHandle::new().unwrap();
/// let props = GraphProperties { is_symmetric: false, is_multigraph: false };
///
/// let (_table, mut graph) = graph_from_parquet(
///     &handle,
///     "edges.parquet",
///     0, 1, Some(2),
///     props, false, true,
/// ).unwrap();
/// ```
pub fn graph_from_parquet(
    handle: &ResourceHandle,
    path: &str,
    src_col: usize,
    dst_col: usize,
    wgt_col: Option<usize>,
    props: GraphProperties,
    store_transposed: bool,
    renumber: bool,
) -> Result<(cudf::Table, Graph)> {
    // ── Read the Parquet file into GPU memory ───────────────────────────
    let options = parquet::ParquetReaderOptions::new(path).map_err(|e| {
        Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
            .with_message(format!("failed to create Parquet reader options: {}", e))
            .with_context("path", path.to_string())
    })?;

    let table = parquet::read_parquet(&options).map_err(|e| {
        Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
            .with_message(format!("failed to read Parquet file: {}", e))
            .with_context("path", path.to_string())
    })?;

    // Build the graph inside an inner scope so that the TableView and all
    // intermediate DeviceArrayViews are dropped before we return `table`.
    let graph = {
        let view = table.view().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                .with_message(format!("failed to get table view: {}", e))
        })?;

        let num_cols = view.num_columns().map_err(|e| {
            Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                .with_message(format!("failed to get column count: {}", e))
        })? as usize;

        // ── Validate column indices ─────────────────────────────────────
        let validate_col_index = |idx: usize, name: &str| -> Result<()> {
            if idx >= num_cols {
                return Err(Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
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

        // ── Helper: extract a cugraph DeviceArrayView from a table column
        // Rejects nullable columns (graph edges/weights must be fully defined)
        // and accounts for cudf column offsets when computing the device ptr.
        let col_to_cugraph_view = |idx: usize, name: &str| -> Result<DeviceArrayView<'_>> {
            let col = view.column(idx as i32).map_err(|e| {
                Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                    .with_message(format!("failed to get {} column: {}", name, e))
                    .with_context("column_index", idx.to_string())
            })?;

            let dt = col.data_type().map_err(|e| {
                Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                    .with_message(format!("failed to get {} column data type: {}", name, e))
                    .with_context("column_index", idx.to_string())
            })?;

            let cudf_type_id = dt.id();
            let cugraph_dtype = cudf_type_to_cugraph(cudf_type_id).ok_or_else(|| {
                Error::new(ErrorKind::UnsupportedType, "graph_from_parquet")
                    .with_message(format!(
                        "{} column has cudf type {:?} which has no cugraph equivalent",
                        name, cudf_type_id
                    ))
                    .with_context("column_index", idx.to_string())
            })?;

            let null_count = col.null_count().map_err(|e| {
                Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                    .with_message(format!("failed to get {} column null count: {}", name, e))
                    .with_context("column_index", idx.to_string())
            })?;

            if null_count != 0 {
                return Err(
                    Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
                        .with_message(format!(
                            "{} column contains {} null value(s); nulls are not supported for edge lists",
                            name, null_count
                        ))
                        .with_context("column_name", name.to_string())
                        .with_context("column_index", idx.to_string())
                        .with_context("null_count", null_count.to_string()),
                );
            }

            let base_ptr = col.device_data_ptr().map_err(|e| {
                Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                    .with_message(format!(
                        "failed to get {} column device pointer: {}",
                        name, e
                    ))
                    .with_context("column_index", idx.to_string())
            })? as *const u8;

            let len = col.len().map_err(|e| {
                Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                    .with_message(format!("failed to get {} column length: {}", name, e))
                    .with_context("column_index", idx.to_string())
            })? as usize;

            let offset = col.offset().map_err(|e| {
                Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                    .with_message(format!("failed to get {} column offset: {}", name, e))
                    .with_context("column_index", idx.to_string())
            })?;

            if offset < 0 {
                return Err(Error::new(ErrorKind::Unexpected, "graph_from_parquet")
                    .with_message(format!(
                        "{} column returned a negative offset {}",
                        name, offset
                    ))
                    .with_context("column_index", idx.to_string()));
            }
            let offset = offset as usize;

            let byte_offset = offset.checked_mul(cugraph_dtype.size_of()).ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "graph_from_parquet")
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
            Ok(unsafe { DeviceArrayView::from_raw(ptr, len, cugraph_dtype) })
        };

        let src_cg = col_to_cugraph_view(src_col, "src")?;
        let dst_cg = col_to_cugraph_view(dst_col, "dst")?;
        let wgt_cg = wgt_col.map(|w| col_to_cugraph_view(w, "wgt")).transpose()?;

        // ── Build the graph ─────────────────────────────────────────────
        Graph::from_edge_list(
            handle,
            props,
            &src_cg,
            &dst_cg,
            wgt_cg.as_ref(),
            None, // no edge IDs
            None, // no edge type IDs
            store_transposed,
            renumber,
        )?
    };

    Ok((table, graph))
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
            let cudf_tid = cugraph_type_to_cudf(dt)
                .unwrap_or_else(|| panic!("cugraph_type_to_cudf returned None for {:?}", dt));
            let back = cudf_type_to_cugraph(cudf_tid)
                .unwrap_or_else(|| panic!("cudf_type_to_cugraph returned None for {:?}", cudf_tid));
            assert_eq!(
                dt, back,
                "round-trip failed: {:?} → {:?} → {:?}",
                dt, cudf_tid, back
            );
        }
    }

    #[test]
    fn cudf_to_cugraph_roundtrip() {
        // For every mapped cudf TypeId, converting to cugraph and back yields the original.
        for &tid in MAPPED_TYPE_IDS {
            let cugraph_dt = cudf_type_to_cugraph(tid)
                .unwrap_or_else(|| panic!("cudf_type_to_cugraph returned None for {:?}", tid));
            let back = cugraph_type_to_cudf(cugraph_dt).unwrap_or_else(|| {
                panic!("cugraph_type_to_cudf returned None for {:?}", cugraph_dt)
            });
            assert_eq!(
                tid, back,
                "round-trip failed: {:?} → {:?} → {:?}",
                tid, cugraph_dt, back
            );
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
            assert_eq!(
                cudf_type_to_cugraph(tid),
                None,
                "expected None for unsupported cudf TypeId {:?}",
                tid
            );
        }
    }
}
