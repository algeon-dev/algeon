//! Data type definitions mirroring `cudf::type_id` and `cudf::data_type`.

use std::{fmt, ptr};

use crate::core::error::{self, Result};

/// Logical type identifier for column elements.
///
/// Each variant maps 1:1 to `cudf::type_id` with identical discriminant values.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeId {
    /// Placeholder for columns with no data or unknown type.
    Empty = 0,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// Unsigned 8-bit integer.
    UInt8,
    /// Unsigned 16-bit integer.
    UInt16,
    /// Unsigned 32-bit integer.
    UInt32,
    /// Unsigned 64-bit integer.
    UInt64,
    /// 32-bit IEEE 754 floating point.
    Float32,
    /// 64-bit IEEE 754 floating point.
    Float64,
    /// Boolean stored as 8-bit value (`0` = false, non-zero = true).
    Bool8,
    /// Timestamp with day resolution (days since Unix epoch).
    TimestampDays,
    /// Timestamp with second resolution (seconds since Unix epoch).
    TimestampSeconds,
    /// Timestamp with millisecond resolution (ms since Unix epoch).
    TimestampMilliseconds,
    /// Timestamp with microsecond resolution (us since Unix epoch).
    TimestampMicroseconds,
    /// Timestamp with nanosecond resolution (ns since Unix epoch).
    TimestampNanoseconds,
    /// Duration with day resolution (number of days).
    DurationDays,
    /// Duration with second resolution (number of seconds).
    DurationSeconds,
    /// Duration with millisecond resolution (number of ms).
    DurationMilliseconds,
    /// Duration with microsecond resolution (number of us).
    DurationMicroseconds,
    /// Duration with nanosecond resolution (number of ns).
    DurationNanoseconds,
    /// Dictionary-encoded column with `Int32` indices.
    Dictionary32,
    /// Variable-length UTF-8 string.
    String,
    /// Variable-length list (each element is a sub-list of another type).
    List,
    /// 32-bit fixed-point decimal (requires a scale via [`DataType::fixed_point`]).
    Decimal32,
    /// 64-bit fixed-point decimal (requires a scale via [`DataType::fixed_point`]).
    Decimal64,
    /// 128-bit fixed-point decimal (requires a scale via [`DataType::fixed_point`]).
    Decimal128,
    /// A column of structs (each element is a row of child columns).
    Struct,
}

impl TypeId {
    /// Stable machine-readable cudf type label.
    pub const fn name(self) -> &'static str {
        match self {
            TypeId::Empty => "empty",
            TypeId::Int8 => "int8",
            TypeId::Int16 => "int16",
            TypeId::Int32 => "int32",
            TypeId::Int64 => "int64",
            TypeId::UInt8 => "uint8",
            TypeId::UInt16 => "uint16",
            TypeId::UInt32 => "uint32",
            TypeId::UInt64 => "uint64",
            TypeId::Float32 => "float32",
            TypeId::Float64 => "float64",
            TypeId::Bool8 => "bool8",
            TypeId::TimestampDays => "timestamp_days",
            TypeId::TimestampSeconds => "timestamp_seconds",
            TypeId::TimestampMilliseconds => "timestamp_milliseconds",
            TypeId::TimestampMicroseconds => "timestamp_microseconds",
            TypeId::TimestampNanoseconds => "timestamp_nanoseconds",
            TypeId::DurationDays => "duration_days",
            TypeId::DurationSeconds => "duration_seconds",
            TypeId::DurationMilliseconds => "duration_milliseconds",
            TypeId::DurationMicroseconds => "duration_microseconds",
            TypeId::DurationNanoseconds => "duration_nanoseconds",
            TypeId::Dictionary32 => "dictionary32",
            TypeId::String => "string",
            TypeId::List => "list",
            TypeId::Decimal32 => "decimal32",
            TypeId::Decimal64 => "decimal64",
            TypeId::Decimal128 => "decimal128",
            TypeId::Struct => "struct",
        }
    }

    /// Every mirrored type id, in discriminant order.
    pub(crate) const ALL: [TypeId; 29] = [
        TypeId::Empty,
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

    /// Convert from a raw FFI type-id discriminant, returning an error for
    /// unknown values.
    ///
    /// The raw value arrives as `i32` (not as the mirrored FFI enum) so a
    /// libcudf type id this binding does not mirror yet stays a recoverable
    /// error instead of materializing an invalid Rust enum value.
    pub(crate) fn from_raw(raw: i32) -> Result<Self> {
        Self::ALL.iter().copied().find(|id| id.to_raw() as i32 == raw).ok_or_else(|| {
            error::foreign_invariant_violation(
                "type_id::from_raw",
                "libcudf returned a type id this binding does not mirror",
                "type_id",
                "unmirrored_foreign_type_id",
                "upgrade cudf-nexus to a version that mirrors this libcudf type id",
            )
            .with_detail("raw_type_id", raw.to_string())
        })
    }

    /// Convert to the raw FFI type_id enum.
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_type_id_t {
        use cudf_sys::cudf_type_id_t::*;
        match self {
            TypeId::Empty => CUDF_TYPE_EMPTY,
            TypeId::Int8 => CUDF_TYPE_INT8,
            TypeId::Int16 => CUDF_TYPE_INT16,
            TypeId::Int32 => CUDF_TYPE_INT32,
            TypeId::Int64 => CUDF_TYPE_INT64,
            TypeId::UInt8 => CUDF_TYPE_UINT8,
            TypeId::UInt16 => CUDF_TYPE_UINT16,
            TypeId::UInt32 => CUDF_TYPE_UINT32,
            TypeId::UInt64 => CUDF_TYPE_UINT64,
            TypeId::Float32 => CUDF_TYPE_FLOAT32,
            TypeId::Float64 => CUDF_TYPE_FLOAT64,
            TypeId::Bool8 => CUDF_TYPE_BOOL8,
            TypeId::TimestampDays => CUDF_TYPE_TIMESTAMP_DAYS,
            TypeId::TimestampSeconds => CUDF_TYPE_TIMESTAMP_SECONDS,
            TypeId::TimestampMilliseconds => CUDF_TYPE_TIMESTAMP_MILLISECONDS,
            TypeId::TimestampMicroseconds => CUDF_TYPE_TIMESTAMP_MICROSECONDS,
            TypeId::TimestampNanoseconds => CUDF_TYPE_TIMESTAMP_NANOSECONDS,
            TypeId::DurationDays => CUDF_TYPE_DURATION_DAYS,
            TypeId::DurationSeconds => CUDF_TYPE_DURATION_SECONDS,
            TypeId::DurationMilliseconds => CUDF_TYPE_DURATION_MILLISECONDS,
            TypeId::DurationMicroseconds => CUDF_TYPE_DURATION_MICROSECONDS,
            TypeId::DurationNanoseconds => CUDF_TYPE_DURATION_NANOSECONDS,
            TypeId::Dictionary32 => CUDF_TYPE_DICTIONARY32,
            TypeId::String => CUDF_TYPE_STRING,
            TypeId::List => CUDF_TYPE_LIST,
            TypeId::Decimal32 => CUDF_TYPE_DECIMAL32,
            TypeId::Decimal64 => CUDF_TYPE_DECIMAL64,
            TypeId::Decimal128 => CUDF_TYPE_DECIMAL128,
            TypeId::Struct => CUDF_TYPE_STRUCT,
        }
    }

    /// Returns the size in bytes of a fixed-width type, or `None` for
    /// variable-width types (String, List, Struct).
    pub fn size_in_bytes(self) -> Result<Option<i32>> {
        let mut out: i32 = 0;
        // SAFETY: self.to_raw() produces a valid cudf_type_id_t value.
        error::check_status_in(unsafe { cudf_sys::cudf_type_size(self.to_raw(), &mut out) }, "type_id::size_in_bytes")?;
        if out == 0 { Ok(None) } else { Ok(Some(out)) }
    }

    /// Returns `true` if this is a fixed-width numeric type (integers and floats).
    ///
    /// `Bool8` is **not** considered numeric — use [`is_boolean`](Self::is_boolean)
    /// instead, or combine with `|| is_boolean()` when both are acceptable.
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
        )
    }

    /// Returns `true` if this is a boolean type (`Bool8`).
    pub fn is_boolean(self) -> bool {
        self == TypeId::Bool8
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

    /// Returns `true` if this is a signed integer type.
    pub fn is_signed_int(self) -> bool {
        matches!(self, TypeId::Int8 | TypeId::Int16 | TypeId::Int32 | TypeId::Int64)
    }

    /// Returns `true` if this is an unsigned integer type.
    pub fn is_unsigned_int(self) -> bool {
        matches!(self, TypeId::UInt8 | TypeId::UInt16 | TypeId::UInt32 | TypeId::UInt64)
    }

    /// Returns `true` if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, TypeId::Float32 | TypeId::Float64)
    }

    /// Returns `true` if this is a signed or unsigned integer type.
    pub(crate) fn is_integral(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Returns a numeric rank used for type promotion ordering.
    ///
    /// Integers get sequential ranks by width, floats get higher ranks:
    /// - `Int8=0, UInt8=1, Int16=2, UInt16=3, Int32=4, UInt32=5, Int64=6, UInt64=7`
    /// - `Float32=8, Float64=10`
    /// - Non-numeric types → `None`
    pub(crate) fn numeric_rank(self) -> Option<u8> {
        match self {
            TypeId::Int8 => Some(0),
            TypeId::UInt8 => Some(1),
            TypeId::Int16 => Some(2),
            TypeId::UInt16 => Some(3),
            TypeId::Int32 => Some(4),
            TypeId::UInt32 => Some(5),
            TypeId::Int64 => Some(6),
            TypeId::UInt64 => Some(7),
            TypeId::Float32 => Some(8),
            TypeId::Float64 => Some(10),
            _ => None,
        }
    }

    /// Returns `true` if this is a fixed-point decimal type.
    pub fn is_decimal(self) -> bool {
        matches!(self, TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128)
    }

    pub(crate) fn timestamp_resolution(self) -> Option<TimestampResolution> {
        match self {
            TypeId::TimestampDays => Some(TimestampResolution::Days),
            TypeId::TimestampSeconds => Some(TimestampResolution::Seconds),
            TypeId::TimestampMilliseconds => Some(TimestampResolution::Milliseconds),
            TypeId::TimestampMicroseconds => Some(TimestampResolution::Microseconds),
            TypeId::TimestampNanoseconds => Some(TimestampResolution::Nanoseconds),
            _ => None,
        }
    }

    pub(crate) fn duration_resolution(self) -> Option<TimestampResolution> {
        match self {
            TypeId::DurationDays => Some(TimestampResolution::Days),
            TypeId::DurationSeconds => Some(TimestampResolution::Seconds),
            TypeId::DurationMilliseconds => Some(TimestampResolution::Milliseconds),
            TypeId::DurationMicroseconds => Some(TimestampResolution::Microseconds),
            TypeId::DurationNanoseconds => Some(TimestampResolution::Nanoseconds),
            _ => None,
        }
    }
}

impl fmt::Display for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
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

    pub(crate) fn from_raw_parts(raw_id: i32, scale: i32) -> Result<Self> {
        let id = TypeId::from_raw(raw_id)?;
        Ok(if id.is_decimal() { Self::fixed_point(id, scale) } else { Self::new(id) })
    }
}

impl From<TypeId> for DataType {
    fn from(id: TypeId) -> Self {
        DataType::new(id)
    }
}

mod private {
    pub trait Sealed {}

    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for bool {}
}

/// Trait for Rust types that map to a cudf `TypeId`.
///
/// This enables generic runtime-backed column construction:
/// `runtime.column_from_slice::<i32>(&data)`.
pub trait CudfType: private::Sealed + Copy + 'static {
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

/// Null handling policy for operations like groupby keys and scan.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullPolicy {
    Exclude = 0,
    Include = 1,
}

impl NullPolicy {
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_null_policy_t {
        match self {
            NullPolicy::Exclude => cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE,
            NullPolicy::Include => cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_INCLUDE,
        }
    }
}

/// Whether null elements should be considered equal in joins and stream compaction.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullEquality {
    /// Null keys are considered equal (null == null is true).
    Equal = 0,
    /// Null keys are considered unequal (null == null is false).
    Unequal = 1,
}

impl NullEquality {
    pub(crate) fn to_join_raw(self) -> cudf_sys::cudf_null_equality_t {
        match self {
            NullEquality::Equal => cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            NullEquality::Unequal => cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_UNEQUAL,
        }
    }

    pub(crate) fn to_stream_compaction_raw(self) -> cudf_sys::cudf_stream_compaction_null_equality_t {
        match self {
            NullEquality::Equal => {
                cudf_sys::cudf_stream_compaction_null_equality_t::CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL
            },
            NullEquality::Unequal => {
                cudf_sys::cudf_stream_compaction_null_equality_t::CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL
            },
        }
    }
}

/// Temporal resolution for timestamp and duration convenience APIs.
///
/// Maps to the corresponding timestamp or duration [`TypeId`] variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimestampResolution {
    /// Day resolution.
    Days,
    /// Second resolution.
    Seconds,
    /// Millisecond resolution.
    Milliseconds,
    /// Microsecond resolution.
    Microseconds,
    /// Nanosecond resolution.
    Nanoseconds,
}

impl TimestampResolution {
    /// Returns the corresponding timestamp [`TypeId`] for this resolution.
    pub fn type_id(self) -> TypeId {
        match self {
            Self::Days => TypeId::TimestampDays,
            Self::Seconds => TypeId::TimestampSeconds,
            Self::Milliseconds => TypeId::TimestampMilliseconds,
            Self::Microseconds => TypeId::TimestampMicroseconds,
            Self::Nanoseconds => TypeId::TimestampNanoseconds,
        }
    }

    /// Returns the corresponding duration [`TypeId`] for this resolution.
    ///
    /// libcudf requires a numeric → duration → timestamp cast path;
    /// direct numeric → timestamp casts are not supported.
    pub fn duration_type_id(self) -> TypeId {
        match self {
            Self::Days => TypeId::DurationDays,
            Self::Seconds => TypeId::DurationSeconds,
            Self::Milliseconds => TypeId::DurationMilliseconds,
            Self::Microseconds => TypeId::DurationMicroseconds,
            Self::Nanoseconds => TypeId::DurationNanoseconds,
        }
    }
}

/// Helper to create a bitmask from an iterator of validity flags.
///
/// cudf uses LSB-first bitmask packing (bit 0 of byte 0 = element 0).
pub(crate) fn validity_to_bitmask(validity: &[bool]) -> Vec<u8> {
    let num_bytes = validity.len().div_ceil(8);
    let mut mask = vec![0u8; num_bytes];
    for (i, &valid) in validity.iter().enumerate() {
        if valid {
            mask[i / 8] |= 1 << (i % 8);
        }
    }
    mask
}

pub(crate) fn cudf_null_mask_allocation_size_bytes(len: usize) -> usize {
    if len == 0 { 0 } else { len.div_ceil(8).div_ceil(64) * 64 }
}

pub(crate) fn validity_to_ffi_bitmask(validity: &[bool]) -> Vec<u8> {
    let mut mask = validity_to_bitmask(validity);
    mask.resize(cudf_null_mask_allocation_size_bytes(validity.len()), 0);
    mask
}

/// Count the number of `false` entries in a validity slice.
pub(crate) fn count_nulls(validity: &[bool]) -> i32 {
    validity.iter().filter(|&&v| !v).count() as i32
}

// Internal helpers for FFI handle round-tripping of DataType.
impl DataType {
    /// Create a C shim data_type handle from this DataType with operation context.
    pub(crate) fn to_raw_handle_for(self, operation: &'static str) -> Result<*mut std::ffi::c_void> {
        let mut handle = ptr::null_mut();
        if self.id.is_decimal() {
            error::check_status_in_with_details(
                // SAFETY: handle is initialized to null and set by the C shim on success.
                unsafe { cudf_sys::cudf_data_type_create_fixed_point(self.id.to_raw(), self.scale, &mut handle) },
                operation,
                [("type_id_label", self.id.to_string()), ("scale", self.scale.to_string())],
            )?;
        } else {
            error::check_status_in_with_details(
                // SAFETY: handle is initialized to null and set by the C shim on success.
                unsafe { cudf_sys::cudf_data_type_create(self.id.to_raw(), &mut handle) },
                operation,
                [("type_id_label", self.id.to_string()), ("scale", self.scale.to_string())],
            )?;
        }
        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::ErrorCode;

    #[test]
    fn type_id_from_raw_round_trips_every_mirrored_discriminant() {
        for id in TypeId::ALL {
            assert_eq!(TypeId::from_raw(id.to_raw() as i32).unwrap(), id);
        }
    }

    #[test]
    fn type_id_from_raw_rejects_unmirrored_discriminants() {
        // NUM_TYPE_IDS (one past the last mirrored id) is exactly what a newer
        // libcudf would hand back for a type this binding does not mirror yet.
        let one_past_last = TypeId::ALL.len() as i32;
        for raw in [one_past_last, -1, i32::MAX] {
            let err = TypeId::from_raw(raw).expect_err("unmirrored type id must be rejected");
            assert_eq!(err.code(), ErrorCode::ForeignInvariantViolation);
            assert_eq!(err.operation(), "type_id::from_raw");
            assert_eq!(err.detail_value("raw_type_id"), Some(raw.to_string().as_str()));
        }
    }
}
