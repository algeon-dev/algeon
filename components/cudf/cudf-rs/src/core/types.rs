//! Data type definitions mirroring `cudf::type_id` and `cudf::data_type`.

use std::ptr;

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
    /// Convert from the raw FFI type_id enum, returning an error for unknown values.
    pub(crate) fn from_raw(raw: cudf_sys::cudf_type_id_t) -> Result<Self> {
        use cudf_sys::cudf_type_id_t::*;
        match raw {
            CUDF_TYPE_EMPTY => Ok(TypeId::Empty),
            CUDF_TYPE_INT8 => Ok(TypeId::Int8),
            CUDF_TYPE_INT16 => Ok(TypeId::Int16),
            CUDF_TYPE_INT32 => Ok(TypeId::Int32),
            CUDF_TYPE_INT64 => Ok(TypeId::Int64),
            CUDF_TYPE_UINT8 => Ok(TypeId::UInt8),
            CUDF_TYPE_UINT16 => Ok(TypeId::UInt16),
            CUDF_TYPE_UINT32 => Ok(TypeId::UInt32),
            CUDF_TYPE_UINT64 => Ok(TypeId::UInt64),
            CUDF_TYPE_FLOAT32 => Ok(TypeId::Float32),
            CUDF_TYPE_FLOAT64 => Ok(TypeId::Float64),
            CUDF_TYPE_BOOL8 => Ok(TypeId::Bool8),
            CUDF_TYPE_TIMESTAMP_DAYS => Ok(TypeId::TimestampDays),
            CUDF_TYPE_TIMESTAMP_SECONDS => Ok(TypeId::TimestampSeconds),
            CUDF_TYPE_TIMESTAMP_MILLISECONDS => Ok(TypeId::TimestampMilliseconds),
            CUDF_TYPE_TIMESTAMP_MICROSECONDS => Ok(TypeId::TimestampMicroseconds),
            CUDF_TYPE_TIMESTAMP_NANOSECONDS => Ok(TypeId::TimestampNanoseconds),
            CUDF_TYPE_DURATION_DAYS => Ok(TypeId::DurationDays),
            CUDF_TYPE_DURATION_SECONDS => Ok(TypeId::DurationSeconds),
            CUDF_TYPE_DURATION_MILLISECONDS => Ok(TypeId::DurationMilliseconds),
            CUDF_TYPE_DURATION_MICROSECONDS => Ok(TypeId::DurationMicroseconds),
            CUDF_TYPE_DURATION_NANOSECONDS => Ok(TypeId::DurationNanoseconds),
            CUDF_TYPE_DICTIONARY32 => Ok(TypeId::Dictionary32),
            CUDF_TYPE_STRING => Ok(TypeId::String),
            CUDF_TYPE_LIST => Ok(TypeId::List),
            CUDF_TYPE_DECIMAL32 => Ok(TypeId::Decimal32),
            CUDF_TYPE_DECIMAL64 => Ok(TypeId::Decimal64),
            CUDF_TYPE_DECIMAL128 => Ok(TypeId::Decimal128),
            CUDF_TYPE_STRUCT => Ok(TypeId::Struct),
        }
    }

    /// Convert to the raw FFI type_id enum.
    pub(crate) fn to_raw(self) -> cudf_sys::cudf_type_id_t {
        match self {
            TypeId::Empty => cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY,
            TypeId::Int8 => cudf_sys::cudf_type_id_t::CUDF_TYPE_INT8,
            TypeId::Int16 => cudf_sys::cudf_type_id_t::CUDF_TYPE_INT16,
            TypeId::Int32 => cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            TypeId::Int64 => cudf_sys::cudf_type_id_t::CUDF_TYPE_INT64,
            TypeId::UInt8 => cudf_sys::cudf_type_id_t::CUDF_TYPE_UINT8,
            TypeId::UInt16 => cudf_sys::cudf_type_id_t::CUDF_TYPE_UINT16,
            TypeId::UInt32 => cudf_sys::cudf_type_id_t::CUDF_TYPE_UINT32,
            TypeId::UInt64 => cudf_sys::cudf_type_id_t::CUDF_TYPE_UINT64,
            TypeId::Float32 => cudf_sys::cudf_type_id_t::CUDF_TYPE_FLOAT32,
            TypeId::Float64 => cudf_sys::cudf_type_id_t::CUDF_TYPE_FLOAT64,
            TypeId::Bool8 => cudf_sys::cudf_type_id_t::CUDF_TYPE_BOOL8,
            TypeId::TimestampDays => cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_DAYS,
            TypeId::TimestampSeconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_SECONDS,
            TypeId::TimestampMilliseconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MILLISECONDS,
            TypeId::TimestampMicroseconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_MICROSECONDS,
            TypeId::TimestampNanoseconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_TIMESTAMP_NANOSECONDS,
            TypeId::DurationDays => cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_DAYS,
            TypeId::DurationSeconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_SECONDS,
            TypeId::DurationMilliseconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MILLISECONDS,
            TypeId::DurationMicroseconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_MICROSECONDS,
            TypeId::DurationNanoseconds => cudf_sys::cudf_type_id_t::CUDF_TYPE_DURATION_NANOSECONDS,
            TypeId::Dictionary32 => cudf_sys::cudf_type_id_t::CUDF_TYPE_DICTIONARY32,
            TypeId::String => cudf_sys::cudf_type_id_t::CUDF_TYPE_STRING,
            TypeId::List => cudf_sys::cudf_type_id_t::CUDF_TYPE_LIST,
            TypeId::Decimal32 => cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL32,
            TypeId::Decimal64 => cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL64,
            TypeId::Decimal128 => cudf_sys::cudf_type_id_t::CUDF_TYPE_DECIMAL128,
            TypeId::Struct => cudf_sys::cudf_type_id_t::CUDF_TYPE_STRUCT,
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

    pub(crate) fn from_raw_parts(raw_id: cudf_sys::cudf_type_id_t, scale: i32) -> Result<Self> {
        let id = TypeId::from_raw(raw_id)?;
        Ok(if id.is_decimal() { Self::fixed_point(id, scale) } else { Self::new(id) })
    }
}

impl From<TypeId> for DataType {
    fn from(id: TypeId) -> Self {
        DataType::new(id)
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

/// Timestamp resolution for convenience cast methods like [`Series::to_datetime`].
///
/// Maps to the corresponding [`TypeId`] timestamp variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimestampResolution {
    /// Day resolution ([`TypeId::TimestampDays`]).
    Days,
    /// Second resolution ([`TypeId::TimestampSeconds`]).
    Seconds,
    /// Millisecond resolution ([`TypeId::TimestampMilliseconds`]).
    Milliseconds,
    /// Microsecond resolution ([`TypeId::TimestampMicroseconds`]).
    Microseconds,
    /// Nanosecond resolution ([`TypeId::TimestampNanoseconds`]).
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
            error::check_status_in_with(
                // SAFETY: handle is initialized to null and set by the C shim on success.
                unsafe { cudf_sys::cudf_data_type_create_fixed_point(self.id.to_raw(), self.scale, &mut handle) },
                operation,
                [("type_id", format!("{:?}", self.id).to_lowercase()), ("scale", self.scale.to_string())],
            )?;
        } else {
            error::check_status_in_with(
                // SAFETY: handle is initialized to null and set by the C shim on success.
                unsafe { cudf_sys::cudf_data_type_create(self.id.to_raw(), &mut handle) },
                operation,
                [("type_id", format!("{:?}", self.id).to_lowercase()), ("scale", self.scale.to_string())],
            )?;
        }
        Ok(handle)
    }
}
