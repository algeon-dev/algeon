use super::*;

fn decimal128_dtype(scale: i32) -> DataType {
    match DataType::fixed_point(TypeId::Decimal128, scale) {
        Ok(dtype) => dtype,
        Err(_) => unreachable!("Decimal128 partition values pass a decimal type"),
    }
}

/// An explicit partition column to be materialized after reading a file/chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionColumn {
    pub(super) name: String,
    pub(super) data_type: DataType,
    pub(super) nullable: bool,
}

impl PartitionColumn {
    /// Create partition column metadata.
    ///
    /// `nullable` records source schema truth. Typed null partition values
    /// are accepted only when this flag is true.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Result<Self> {
        let name = checked_name(name.into(), OP_PARTITION_NEW, "partition column")?;
        Ok(Self { name, data_type, nullable })
    }

    /// Partition column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Partition column dtype.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Source-level nullability truth.
    pub fn nullable(&self) -> bool {
        self.nullable
    }
}

/// Partition value supported by the local Parquet source contract.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PartitionValue {
    /// Typed null value. The dtype must match the partition column dtype and
    /// the partition column must be nullable.
    Null(DataType),
    /// Boolean value.
    Boolean(bool),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Signed 32-bit integer.
    Int32(i32),
    /// Signed 64-bit integer.
    Int64(i64),
    /// Unsigned 8-bit integer.
    UInt8(u8),
    /// Unsigned 16-bit integer.
    UInt16(u16),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// 32-bit floating point value.
    Float32(f32),
    /// 64-bit floating point value.
    Float64(f64),
    /// Decimal128 raw fixed-point value plus cudf scale.
    Decimal128 { raw: i128, scale: i32 },
    /// Arrow Date32-compatible day value, represented as cudf timestamp days.
    Date32(i32),
    /// Timezone-less timestamp raw value with explicit resolution.
    Timestamp { raw: i64, resolution: TimestampResolution },
    /// Duration raw value with explicit resolution.
    Duration { raw: i64, resolution: TimestampResolution },
    /// UTF-8 string value.
    String(String),
}

impl PartitionValue {
    /// cuDF dtype represented by this partition value.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Null(data_type) => *data_type,
            Self::Boolean(_) => DataType::new(TypeId::Bool8),
            Self::Int8(_) => DataType::new(TypeId::Int8),
            Self::Int16(_) => DataType::new(TypeId::Int16),
            Self::Int32(_) => DataType::new(TypeId::Int32),
            Self::Int64(_) => DataType::new(TypeId::Int64),
            Self::UInt8(_) => DataType::new(TypeId::UInt8),
            Self::UInt16(_) => DataType::new(TypeId::UInt16),
            Self::UInt32(_) => DataType::new(TypeId::UInt32),
            Self::UInt64(_) => DataType::new(TypeId::UInt64),
            Self::Float32(_) => DataType::new(TypeId::Float32),
            Self::Float64(_) => DataType::new(TypeId::Float64),
            Self::Decimal128 { scale, .. } => decimal128_dtype(*scale),
            Self::Date32(_) => DataType::new(TypeId::TimestampDays),
            Self::Timestamp { resolution, .. } => DataType::new(resolution.type_id()),
            Self::Duration { resolution, .. } => DataType::new(resolution.duration_type_id()),
            Self::String(_) => DataType::new(TypeId::String),
        }
    }

    pub(super) fn is_null(&self) -> bool {
        matches!(self, Self::Null(_))
    }
}
