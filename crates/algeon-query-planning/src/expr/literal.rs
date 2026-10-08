//! Typed scalar literal IR.
//!
//! SQL nulls carry an explicit [`DataType`] in [`LiteralValue::Null`] because
//! they have no value to infer a type from. Decimal, date, timestamp,
//! duration, and interval literals use Arrow-compatible raw encodings;
//! whether a literal is supported in a given expression context is decided
//! during expression lowering, not here.

use std::fmt;

use arrow::datatypes::{DataType, IntervalUnit, TimeUnit};

/// Typed scalar literal in native expression IR; nulls carry an explicit [`DataType`].
///
/// [`Self::data_type`] is total: every variant names one Arrow type, so
/// validation types a literal without a schema. `PartialEq` only, because
/// `Float32` and `Float64` carry NaN; two literals that compare equal have the
/// same type and the same non-NaN value.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    /// Boolean scalar.
    Boolean(bool),
    /// Signed 8-bit integer scalar.
    Int8(i8),
    /// Signed 16-bit integer scalar.
    Int16(i16),
    /// Signed 32-bit integer scalar.
    Int32(i32),
    /// Signed 64-bit integer scalar.
    Int64(i64),
    /// Unsigned 8-bit integer scalar.
    UInt8(u8),
    /// Unsigned 16-bit integer scalar.
    UInt16(u16),
    /// Unsigned 32-bit integer scalar.
    UInt32(u32),
    /// Unsigned 64-bit integer scalar.
    UInt64(u64),
    /// 32-bit floating-point scalar.
    Float32(f32),
    /// 64-bit floating-point scalar.
    Float64(f64),
    /// UTF-8 string scalar with Arrow `Utf8` type.
    Utf8(String),
    /// UTF-8 string scalar with Arrow `LargeUtf8` type.
    LargeUtf8(String),
    /// Arrow `Decimal128` scalar.
    ///
    /// `raw` is the unscaled integer payload. `precision` and `scale` are the
    /// Arrow decimal metadata; the mathematical value is `raw * 10^-scale`.
    Decimal128 {
        /// Unscaled 128-bit decimal payload.
        raw: i128,
        /// Maximum number of significant decimal digits.
        precision: u8,
        /// Decimal scale in Arrow's representation.
        scale: i8,
    },
    /// Arrow `Date32` scalar, encoded as days since the Unix epoch.
    Date32(i32),
    /// Arrow `Date64` scalar, encoded as milliseconds since the Unix epoch.
    Date64(i64),
    /// Arrow `Time32` scalar.
    ///
    /// `raw` is the count of `unit` intervals since midnight.
    Time32 {
        /// Time-of-day value in the given unit.
        raw: i32,
        /// Time resolution. Arrow permits only seconds and milliseconds.
        unit: TimeUnit,
    },
    /// Arrow `Time64` scalar.
    ///
    /// `raw` is the count of `unit` intervals since midnight.
    Time64 {
        /// Time-of-day value in the given unit.
        raw: i64,
        /// Time resolution. Arrow permits only microseconds and nanoseconds.
        unit: TimeUnit,
    },
    /// Arrow timestamp scalar without a timezone.
    ///
    /// `raw` is the count of `unit` intervals since the Unix epoch.
    Timestamp {
        /// Epoch-relative timestamp value in the given unit.
        raw: i64,
        /// Timestamp resolution.
        unit: TimeUnit,
    },
    /// Arrow duration scalar.
    ///
    /// `raw` is the count of `unit` intervals.
    Duration {
        /// Duration value in the given unit.
        raw: i64,
        /// Duration resolution.
        unit: TimeUnit,
    },
    /// Arrow `Interval(MonthDayNano)` scalar.
    IntervalMonthDayNano {
        /// Month component.
        months: i32,
        /// Day component.
        days: i32,
        /// Nanosecond component.
        nanoseconds: i64,
    },
    /// Typed SQL null literal.
    Null(DataType),
}

impl LiteralValue {
    /// Return the Arrow data type represented by this literal.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Boolean(_) => DataType::Boolean,
            Self::Int8(_) => DataType::Int8,
            Self::Int16(_) => DataType::Int16,
            Self::Int32(_) => DataType::Int32,
            Self::Int64(_) => DataType::Int64,
            Self::UInt8(_) => DataType::UInt8,
            Self::UInt16(_) => DataType::UInt16,
            Self::UInt32(_) => DataType::UInt32,
            Self::UInt64(_) => DataType::UInt64,
            Self::Float32(_) => DataType::Float32,
            Self::Float64(_) => DataType::Float64,
            Self::Utf8(_) => DataType::Utf8,
            Self::LargeUtf8(_) => DataType::LargeUtf8,
            Self::Decimal128 { precision, scale, .. } => DataType::Decimal128(*precision, *scale),
            Self::Date32(_) => DataType::Date32,
            Self::Date64(_) => DataType::Date64,
            Self::Time32 { unit, .. } => DataType::Time32(*unit),
            Self::Time64 { unit, .. } => DataType::Time64(*unit),
            Self::Timestamp { unit, .. } => DataType::Timestamp(*unit, None),
            Self::Duration { unit, .. } => DataType::Duration(*unit),
            Self::IntervalMonthDayNano { .. } => DataType::Interval(IntervalUnit::MonthDayNano),
            Self::Null(data_type) => data_type.clone(),
        }
    }

    /// Zero of an integer type, or `None` for any other type. Count merging
    /// uses it to read a merged global count's null as zero.
    pub(crate) fn integer_zero(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Int8 => Some(Self::Int8(0)),
            DataType::Int16 => Some(Self::Int16(0)),
            DataType::Int32 => Some(Self::Int32(0)),
            DataType::Int64 => Some(Self::Int64(0)),
            DataType::UInt8 => Some(Self::UInt8(0)),
            DataType::UInt16 => Some(Self::UInt16(0)),
            DataType::UInt32 => Some(Self::UInt32(0)),
            DataType::UInt64 => Some(Self::UInt64(0)),
            _ => None,
        }
    }
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean(value) => write!(f, "{value}"),
            Self::Int8(value) => write!(f, "{value}"),
            Self::Int16(value) => write!(f, "{value}"),
            Self::Int32(value) => write!(f, "{value}"),
            Self::Int64(value) => write!(f, "{value}"),
            Self::UInt8(value) => write!(f, "{value}"),
            Self::UInt16(value) => write!(f, "{value}"),
            Self::UInt32(value) => write!(f, "{value}"),
            Self::UInt64(value) => write!(f, "{value}"),
            Self::Float32(value) => write!(f, "{value}"),
            Self::Float64(value) => write!(f, "{value}"),
            Self::Utf8(value) | Self::LargeUtf8(value) => write!(f, "{value:?}"),
            Self::Decimal128 { raw, precision, scale } => {
                write!(f, "DECIMAL128({raw}, precision={precision}, scale={scale})")
            },
            Self::Date32(value) => write!(f, "DATE32({value})"),
            Self::Date64(value) => write!(f, "DATE64({value})"),
            Self::Time32 { raw, unit } => write!(f, "TIME32({raw}, {unit:?})"),
            Self::Time64 { raw, unit } => write!(f, "TIME64({raw}, {unit:?})"),
            Self::Timestamp { raw, unit } => write!(f, "TIMESTAMP({raw}, {unit:?})"),
            Self::Duration { raw, unit } => write!(f, "DURATION({raw}, {unit:?})"),
            Self::IntervalMonthDayNano { months, days, nanoseconds } => {
                write!(f, "INTERVAL(months={months}, days={days}, nanoseconds={nanoseconds})")
            },
            Self::Null(data_type) => write!(f, "NULL::{data_type}"),
        }
    }
}
