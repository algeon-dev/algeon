use std::{collections::BTreeSet, ffi::c_void};

use cudf::Runtime;
use rapids_interop::{BorrowedColumnView, BorrowedTableView, NullMaskKind, TypeId as InteropTypeId};

use super::{errors::map_runtime_ordering_interop_error, options::TableGraphOptions, types::borrowed_type_to_cugraph};
use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceGuard, check_cudf_status},
    error::{Error, ErrorKind, Result},
    graph::{Graph, GraphBuilder, Transposition},
    handle::EnteredContext,
    types::{DataType, GraphProperties},
};

pub(super) struct BorrowedEdgeList<'a> {
    pub(super) src: BorrowedColumnView<'a>,
    pub(super) dst: BorrowedColumnView<'a>,
    pub(super) weights: Option<BorrowedColumnView<'a>>,
}

/// Testing-only borrowed-input ordering plan.
///
/// This captures the observable runtime-bound ordering contract for borrowed
/// cudf inputs while preserving compatibility with handles that are not bound
/// to a cudf runtime stream.
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

    /// Compute the borrowed-input ordering plan for explicit borrowed columns.
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

pub(super) fn edge_list_dtype(type_id: InteropTypeId, name: &str, operation: &'static str) -> Result<DataType> {
    borrowed_type_to_cugraph(type_id).ok_or_else(|| {
        Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("{} column has interop type {:?} which has no cugraph equivalent", name, type_id))
            .with_context("column_name", name.to_string())
    })
}

pub(super) fn checked_column_offset_bytes(
    offset: usize,
    dtype: DataType,
    name: &str,
    operation: &'static str,
) -> Result<usize> {
    offset.checked_mul(dtype.size_of()).ok_or_else(|| {
        Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!("{} column offset is too large", name))
            .with_context("column_name", name.to_string())
            .with_context("offset", offset.to_string())
            .with_context("dtype_size", dtype.size_of().to_string())
    })
}

pub(super) fn checked_active_byte_len(
    len: usize,
    dtype: DataType,
    name: &str,
    operation: &'static str,
) -> Result<usize> {
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
pub(super) fn offset_device_ptr(base_ptr: *const u8, byte_offset: usize, active_len: usize) -> *mut c_void {
    if base_ptr.is_null() || active_len == 0 {
        base_ptr as *mut c_void
    } else {
        // SAFETY: callers validate that the active fixed-width range is in-bounds.
        unsafe { base_ptr.add(byte_offset) as *mut c_void }
    }
}

pub(super) fn validate_edge_list_devices<'a>(
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

pub(super) fn borrowed_column_to_raw<'a>(
    column: &BorrowedColumnView<'a>,
    name: &str,
    operation: &'static str,
) -> Result<RawDeviceArrayView<'a>> {
    let cugraph_dtype = edge_list_dtype(column.type_id(), name, operation)?;

    if let NullMaskKind::Present { null_count, .. } = column.null_mask().kind()
        && *null_count != 0
    {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message(format!(
                "{} column contains {} null value(s); nulls are not supported for edge lists",
                name, null_count
            ))
            .with_context("column_name", name.to_string())
            .with_context("null_count", null_count.to_string()));
    }

    let byte_offset = checked_column_offset_bytes(column.offset(), cugraph_dtype, name, operation)?;
    let active_len = checked_active_byte_len(column.len(), cugraph_dtype, name, operation)?;
    let ptr = offset_device_ptr(column.data_ptr() as *const u8, byte_offset, active_len);

    // SAFETY: `ptr` points at the active region of the producer-owned device buffer.
    Ok(unsafe { RawDeviceArrayView::try_from_raw(ptr, column.len(), cugraph_dtype, operation)? })
}

pub(super) fn borrowed_table_to_edge_list<'a>(
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

pub(super) fn validate_edge_list_policy(
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

pub(super) fn order_borrowed_inputs_for_handle(
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

pub(super) fn build_graph_from_borrowed_edge_list_raw<T: Transposition>(
    handle: &EnteredContext<'_>,
    edge_list: &BorrowedEdgeList<'_>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResourceHandle, error::ErrorKind};

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
}
