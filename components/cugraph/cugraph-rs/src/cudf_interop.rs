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
            assert_eq!(dt, back, "round-trip failed: {:?} → {:?} → {:?}", dt, cudf_tid, back);
        }
    }

    #[test]
    fn cudf_to_cugraph_roundtrip() {
        // For every mapped cudf TypeId, converting to cugraph and back yields the original.
        for &tid in MAPPED_TYPE_IDS {
            let cugraph_dt = cudf_type_to_cugraph(tid)
                .unwrap_or_else(|| panic!("cudf_type_to_cugraph returned None for {:?}", tid));
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
            assert_eq!(
                cudf_type_to_cugraph(tid),
                None,
                "expected None for unsupported cudf TypeId {:?}",
                tid
            );
        }
    }
}
