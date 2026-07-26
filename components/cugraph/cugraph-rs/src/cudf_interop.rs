//! Zero-copy interop between cudf [`DeviceSlice`] and cugraph [`DeviceBuffer`] / [`DeviceArrayView`].
//!
//! This module is only available when the `cudf-interop` feature is enabled.
//!
//! [`DeviceSlice`]: cudf::DeviceSlice
//! [`DeviceBuffer`]: crate::array::DeviceBuffer
//! [`DeviceArrayView`]: crate::array::DeviceArrayView

use crate::array::{DeviceArrayView, DeviceBuffer};
use crate::error::{Error, ErrorKind, Result};
use crate::types::DataType;
use cudf::column::DeviceSlice;
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
            Error::new(ErrorKind::UnsupportedType, "DeviceSlice::dtype")
                .with_message(format!(
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
