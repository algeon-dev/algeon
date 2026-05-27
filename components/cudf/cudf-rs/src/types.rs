//! Data type definitions mirroring `cudf::type_id` and `cudf::data_type`.

use crate::error::{self, Result};
use std::ptr;

/// Logical type identifier for column elements.
///
/// Each variant maps 1:1 to `cudf::type_id` with identical discriminant values.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeId {
    Empty = 0,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool8,
    TimestampDays,
    TimestampSeconds,
    TimestampMilliseconds,
    TimestampMicroseconds,
    TimestampNanoseconds,
    DurationDays,
    DurationSeconds,
    DurationMilliseconds,
    DurationMicroseconds,
    DurationNanoseconds,
    Dictionary32,
    String,
    List,
    Decimal32,
    Decimal64,
    Decimal128,
    Struct,
}

impl TypeId {
    /// Convert from the raw FFI type_id enum.
    pub(crate) fn from_raw(raw: cudf_sys::cudf_type_id_t) -> Self {
        // The Rust enum and C enum have identical layouts.
        unsafe { std::mem::transmute(raw as i32) }
    }

    /// Convert to the raw FFI type_id enum.
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_type_id_t {
        unsafe { std::mem::transmute(self) }
    }

    /// Returns the size in bytes of a fixed-width type, or `None` for
    /// variable-width types (String, List, Struct).
    pub fn size_in_bytes(self) -> Result<Option<i32>> {
        let mut out: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_type_size(self.to_raw(), &mut out) },
            "type_id::size_in_bytes",
        )?;
        if out == 0 {
            Ok(None)
        } else {
            Ok(Some(out))
        }
    }

    /// Returns `true` if this is a fixed-width numeric type.
    pub fn is_numeric(self) -> bool {
        matches!(
            self,
            TypeId::Int8
                | TypeId::Int16
                | TypeId::Int32
                | TypeId::Int64
                | TypeId::UInt8
                | TypeId::UInt16
                | TypeId::UInt32
                | TypeId::UInt64
                | TypeId::Float32
                | TypeId::Float64
                | TypeId::Bool8
        )
    }

    /// Returns `true` if this is a timestamp type.
    pub fn is_timestamp(self) -> bool {
        matches!(
            self,
            TypeId::TimestampDays
                | TypeId::TimestampSeconds
                | TypeId::TimestampMilliseconds
                | TypeId::TimestampMicroseconds
                | TypeId::TimestampNanoseconds
        )
    }

    /// Returns `true` if this is a duration type.
    pub fn is_duration(self) -> bool {
        matches!(
            self,
            TypeId::DurationDays
                | TypeId::DurationSeconds
                | TypeId::DurationMilliseconds
                | TypeId::DurationMicroseconds
                | TypeId::DurationNanoseconds
        )
    }

    /// Returns `true` if this is a fixed-point decimal type.
    pub fn is_decimal(self) -> bool {
        matches!(
            self,
            TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128
        )
    }
}

/// A data type descriptor combining a [`TypeId`] with an optional scale
/// (for fixed-point types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataType {
    id: TypeId,
    scale: i32,
}

impl DataType {
    /// Create a new `DataType` from a type identifier (scale defaults to 0).
    pub fn new(id: TypeId) -> Self {
        Self { id, scale: 0 }
    }

    /// Create a fixed-point `DataType` with an explicit scale.
    pub fn fixed_point(id: TypeId, scale: i32) -> Self {
        Self { id, scale }
    }

    /// Returns the type identifier.
    pub fn id(&self) -> TypeId {
        self.id
    }

    /// Returns the scale (meaningful only for decimal types).
    pub fn scale(&self) -> i32 {
        self.scale
    }
}

/// Trait for Rust types that map to a cudf `TypeId`.
///
/// This enables generic column construction: `Column::from_slice::<i32>(&data)`.
pub trait CudfType: Copy + 'static {
    /// The corresponding cudf type identifier.
    const TYPE_ID: TypeId;
}

impl CudfType for i8 {
    const TYPE_ID: TypeId = TypeId::Int8;
}
impl CudfType for i16 {
    const TYPE_ID: TypeId = TypeId::Int16;
}
impl CudfType for i32 {
    const TYPE_ID: TypeId = TypeId::Int32;
}
impl CudfType for i64 {
    const TYPE_ID: TypeId = TypeId::Int64;
}
impl CudfType for u8 {
    const TYPE_ID: TypeId = TypeId::UInt8;
}
impl CudfType for u16 {
    const TYPE_ID: TypeId = TypeId::UInt16;
}
impl CudfType for u32 {
    const TYPE_ID: TypeId = TypeId::UInt32;
}
impl CudfType for u64 {
    const TYPE_ID: TypeId = TypeId::UInt64;
}
impl CudfType for f32 {
    const TYPE_ID: TypeId = TypeId::Float32;
}
impl CudfType for f64 {
    const TYPE_ID: TypeId = TypeId::Float64;
}
impl CudfType for bool {
    const TYPE_ID: TypeId = TypeId::Bool8;
}

/// Helper to create a bitmask from an iterator of validity flags.
///
/// cudf uses LSB-first bitmask packing (bit 0 of byte 0 = element 0).
pub fn validity_to_bitmask(validity: &[bool]) -> Vec<u8> {
    let num_bytes = validity.len().div_ceil(8);
    let mut mask = vec![0u8; num_bytes];
    for (i, &valid) in validity.iter().enumerate() {
        if valid {
            mask[i / 8] |= 1 << (i % 8);
        }
    }
    mask
}

/// Count the number of `false` entries in a validity slice.
pub fn count_nulls(validity: &[bool]) -> i32 {
    validity.iter().filter(|&&v| !v).count() as i32
}

// Internal helpers for FFI handle round-tripping of DataType.
impl DataType {
    /// Create a C shim data_type handle from this DataType with operation context.
    pub(crate) fn to_raw_handle_for(
        self,
        operation: &'static str,
    ) -> Result<*mut std::ffi::c_void> {
        let mut handle = ptr::null_mut();
        if self.id.is_decimal() {
            error::check_status_in_with(
                unsafe {
                    cudf_sys::cudf_data_type_create_fixed_point(
                        self.id.to_raw(),
                        self.scale,
                        &mut handle,
                    )
                },
                operation,
                [
                    ("type_id", format!("{:?}", self.id).to_lowercase()),
                    ("scale", self.scale.to_string()),
                ],
            )?;
        } else {
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_data_type_create(self.id.to_raw(), &mut handle) },
                operation,
                [
                    ("type_id", format!("{:?}", self.id).to_lowercase()),
                    ("scale", self.scale.to_string()),
                ],
            )?;
        }
        Ok(handle)
    }
}
