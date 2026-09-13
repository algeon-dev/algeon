use std::ffi::c_void;

use cudf::interop::contract::TypeId as InteropTypeId;

use super::types::borrowed_type_to_cugraph;
use crate::{
    error::{Error, ErrorKind, Result},
    handle::EnteredContext,
    types::DataType,
};

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
