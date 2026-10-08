//! Arrow-to-cuDF conversion of a source's schema, output columns, partition columns and
//! decimal reader types.
//!
//! cuDF fixed-point scale is the negative of the Arrow and Parquet decimal scale (scale 2 is
//! cuDF scale -2). `ParquetReaderDecimalType::cudf_scale` is already in cuDF convention, while
//! decimal literals carry the Arrow scale and pass through `arrow_decimal_scale_to_cudf_scale`.

use cudf::{
    DataType as CudfDataType, TimestampResolution, TypeId,
    io::source::{
        LocalParquetFile as CudfLocalParquetFile, ParquetReaderDecimalType as CudfParquetReaderDecimalType,
        PartitionColumn as CudfPartitionColumn, PartitionValue as CudfPartitionValue, SourceColumn as CudfSourceColumn,
        SourceColumnRef as CudfSourceColumnRef, SourceColumnRole as CudfSourceColumnRole,
        SourceOutputColumn as CudfSourceOutputColumn, SourceRequiredColumn as CudfSourceRequiredColumn,
        SourceSchema as CudfSourceSchema,
    },
};
use query_planning::source::{
    ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, SourceColumnRef,
};

use super::error::cudf_error_for_location;
use crate::{datatypes::arrow_decimal_scale_to_cudf_scale, exec::source::*};

/// Attaches per-column decimal storage types to a cuDF file. An empty slice leaves the file
/// unchanged.
pub(in crate::exec::source) fn attach_cudf_reader_decimal_types(
    cudf_file: CudfLocalParquetFile,
    reader_decimal_types: &[ParquetReaderDecimalType],
    operation: &'static str,
    location: &ParquetSourceLocation,
) -> Result<CudfLocalParquetFile> {
    if reader_decimal_types.is_empty() {
        return Ok(cudf_file);
    }
    let types = reader_decimal_types.iter().map(cudf_reader_decimal_type).collect::<Result<Vec<_>>>()?;
    cudf_file.with_reader_decimal_types(types).map_err(|error| {
        cudf_error_for_location(
            location,
            operation,
            "failed to attach reader decimal type metadata to cudf source contract",
            error,
        )
    })
}

pub(in crate::exec::source) fn cudf_reader_decimal_type(
    reader_type: &ParquetReaderDecimalType,
) -> Result<CudfParquetReaderDecimalType> {
    let type_id = match reader_type.type_id {
        ParquetReaderDecimalTypeId::Decimal32 => TypeId::Decimal32,
        ParquetReaderDecimalTypeId::Decimal64 => TypeId::Decimal64,
        ParquetReaderDecimalTypeId::Decimal128 => TypeId::Decimal128,
    };
    let dtype = CudfDataType::fixed_point(type_id, reader_type.cudf_scale).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.reader_decimal_type_cudf",
            "failed to convert native reader decimal dtype into cudf source contract",
            error,
        )
        .with_fact("column", reader_type.column_name.as_str())
        .with_fact("reader_type", reader_type.type_id.as_str())
    })?;
    CudfParquetReaderDecimalType::new(reader_type.column_name.as_str(), dtype).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.reader_decimal_type_cudf",
            "failed to convert native reader decimal type into cudf source contract",
            error,
        )
        .with_fact("column", reader_type.column_name.as_str())
        .with_fact("reader_type", reader_type.type_id.as_str())
    })
}

/// Converts the source's logical schema, which covers every field, including partition and
/// missing columns as well as the ones read from the file. A type cuDF cannot represent fails with
/// `UnsupportedType`. List columns keep their Arrow type, because the cuDF dtype for a list does
/// not carry the element type.
pub(in crate::exec::source) fn cudf_source_schema(source: &LocalParquetSource) -> Result<CudfSourceSchema> {
    let columns = source
        .schema
        .fields()
        .iter()
        .map(|field| {
            let dtype = cudf_dtype_from_arrow_type(field.data_type()).map_err(|error| {
                cudf_unsupported_type_error(
                    "query_engine.local_parquet_chunk.schema_cudf",
                    "local parquet source schema contains a type unsupported by cudf source reads",
                    error,
                )
                .with_fact("column", field.name().as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?;
            let column = if dtype.id() == TypeId::List {
                CudfSourceColumn::new_list(field.name().as_str(), field.data_type().clone(), field.is_nullable())
            } else {
                CudfSourceColumn::new(field.name().as_str(), dtype, field.is_nullable())
            };
            column.map_err(|error| {
                cudf_query_error(
                    "query_engine.local_parquet_chunk.schema_cudf",
                    "failed to convert native local parquet source schema into cudf source schema",
                    error,
                )
                .with_fact("column", field.name().as_str())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    CudfSourceSchema::new(columns).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_chunk.schema_cudf",
            "failed to build cudf local parquet source schema",
            error,
        )
    })
}

pub(in crate::exec::source) fn cudf_source_output_columns(
    source: &LocalParquetSource,
) -> Result<Vec<CudfSourceOutputColumn>> {
    source
        .projected_output_columns
        .iter()
        .zip(&source.projected_output_column_source_refs)
        .map(|(column, source_ref)| {
            let source_ref = cudf_source_column_ref(source, column, *source_ref)?;
            CudfSourceOutputColumn::new(column.as_str(), source_ref).map_err(|error| {
                cudf_query_error(
                    "query_engine.local_parquet_chunk.output_cudf",
                    "failed to convert native local parquet output column into cudf source output column",
                    error,
                )
                .with_fact("column", column)
            })
        })
        .collect()
}

pub(in crate::exec::source) fn cudf_source_required_columns(
    source: &LocalParquetSource,
) -> Result<Vec<CudfSourceRequiredColumn>> {
    source
        .pure_filter_columns
        .iter()
        .zip(&source.pure_filter_column_source_refs)
        .map(|(column, source_ref)| {
            let source_ref = cudf_source_column_ref(source, column, *source_ref)?;
            CudfSourceRequiredColumn::new(column.as_str(), source_ref, CudfSourceColumnRole::PureFilter).map_err(
                |error| {
                    cudf_query_error(
                        "query_engine.local_parquet_chunk.required_cudf",
                        "failed to convert native local parquet pure-filter column into cudf source required column",
                        error,
                    )
                    .with_fact("column", column)
                },
            )
        })
        .collect()
}

/// Translates a planner column reference into the cuDF one. A file column becomes the position of
/// its file index in `read_column_file_indices`, because cuDF `ReadColumn` indexes address the
/// projected read list; fails with `InvalidPlan` if the column was not projected. A missing column
/// carries no position and is materialized as nulls.
pub(in crate::exec::source) fn cudf_source_column_ref(
    source: &LocalParquetSource,
    column: &str,
    source_ref: SourceColumnRef,
) -> Result<CudfSourceColumnRef> {
    match source_ref {
        SourceColumnRef::File(file_index) => source
            .read_column_file_indices
            .iter()
            .position(|index| *index == file_index)
            .map(CudfSourceColumnRef::ReadColumn)
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.local_parquet_chunk.output_cudf",
                    "local parquet source file column cannot be resolved to a cudf read projection index",
                )
                .with_fact("column", column)
                .with_fact("file_column_index", file_index)
            }),
        SourceColumnRef::Partition(index) => Ok(CudfSourceColumnRef::Partition(index)),
        SourceColumnRef::MissingNull(_index) => Ok(CudfSourceColumnRef::MissingNull),
    }
}

pub(in crate::exec::source) fn cudf_partition_columns(source: &LocalParquetSource) -> Result<Vec<CudfPartitionColumn>> {
    source
        .partition_columns
        .iter()
        .map(|column| {
            let dtype = cudf_dtype_from_arrow_type(&column.data_type).map_err(|error| {
                cudf_unsupported_type_error(
                    "query_engine.local_parquet_chunk.partition_cudf",
                    "local parquet partition column type is unsupported by cudf source reads",
                    error,
                )
                .with_fact("partition_column", column.name.as_str())
                .with_fact("data_type", format!("{:?}", column.data_type))
            })?;
            CudfPartitionColumn::new(column.name.as_str(), dtype, column.nullable).map_err(|error| {
                cudf_query_error(
                    "query_engine.local_parquet_chunk.partition_cudf",
                    "failed to convert native local parquet partition column into cudf source partition column",
                    error,
                )
                .with_fact("partition_column", column.name.as_str())
            })
        })
        .collect()
}

/// Converts one hive-style partition value. `Date64` is carried as a millisecond timestamp and
/// `Time32`/`Time64` as durations of the same unit. Interval values fail with `UnsupportedType`.
pub(in crate::exec::source) fn cudf_partition_value(value: &LiteralValue) -> Result<CudfPartitionValue> {
    match value {
        LiteralValue::Null(data_type) => {
            let dtype = cudf_dtype_from_arrow_type(data_type).map_err(|error| {
                cudf_unsupported_type_error(
                    "query_engine.local_parquet_chunk.partition_cudf",
                    "local parquet partition null type is unsupported by cudf source reads",
                    error,
                )
                .with_fact("data_type", format!("{data_type:?}"))
            })?;
            Ok(CudfPartitionValue::Null(dtype))
        },
        LiteralValue::Boolean(value) => Ok(CudfPartitionValue::Boolean(*value)),
        LiteralValue::Int8(value) => Ok(CudfPartitionValue::Int8(*value)),
        LiteralValue::Int16(value) => Ok(CudfPartitionValue::Int16(*value)),
        LiteralValue::Int32(value) => Ok(CudfPartitionValue::Int32(*value)),
        LiteralValue::Int64(value) => Ok(CudfPartitionValue::Int64(*value)),
        LiteralValue::UInt8(value) => Ok(CudfPartitionValue::UInt8(*value)),
        LiteralValue::UInt16(value) => Ok(CudfPartitionValue::UInt16(*value)),
        LiteralValue::UInt32(value) => Ok(CudfPartitionValue::UInt32(*value)),
        LiteralValue::UInt64(value) => Ok(CudfPartitionValue::UInt64(*value)),
        LiteralValue::Float32(value) => Ok(CudfPartitionValue::Float32(*value)),
        LiteralValue::Float64(value) => Ok(CudfPartitionValue::Float64(*value)),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => Ok(CudfPartitionValue::String(value.clone())),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            Ok(CudfPartitionValue::Decimal128 { raw: *raw, scale: arrow_decimal_scale_to_cudf_scale(*scale) })
        },
        LiteralValue::Date32(value) => Ok(CudfPartitionValue::Date32(*value)),
        LiteralValue::Date64(value) => {
            Ok(CudfPartitionValue::Timestamp { raw: *value, resolution: TimestampResolution::Milliseconds })
        },
        LiteralValue::Time32 { raw, unit } => Ok(CudfPartitionValue::Duration {
            raw: i64::from(*raw),
            resolution: timestamp_unit_to_cudf_resolution(unit),
        }),
        LiteralValue::Time64 { raw, unit } => {
            Ok(CudfPartitionValue::Duration { raw: *raw, resolution: timestamp_unit_to_cudf_resolution(unit) })
        },
        LiteralValue::Timestamp { raw, unit } => {
            Ok(CudfPartitionValue::Timestamp { raw: *raw, resolution: timestamp_unit_to_cudf_resolution(unit) })
        },
        LiteralValue::Duration { raw, unit } => {
            Ok(CudfPartitionValue::Duration { raw: *raw, resolution: timestamp_unit_to_cudf_resolution(unit) })
        },
        LiteralValue::IntervalMonthDayNano { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.local_parquet_chunk.partition_cudf",
            "local parquet partition value is unsupported by cudf source reads",
        )
        .with_fact("partition_value", value)),
    }
}
