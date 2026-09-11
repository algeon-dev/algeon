use std::time::Instant;

use super::{
    ArrowDataType, DataFrame, ExecutionContext, ExecutionMetrics, GpuDataFrame, LiteralValue, LocalParquetSource,
    QueryError, QueryErrorCode, Result, cudf_query_error, timestamp_unit_to_cudf_resolution,
};

pub(super) fn inject_partition_columns(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    partition_values: &[LiteralValue],
    frame: DataFrame,
    row_count_override: Option<usize>,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    if source.partition_columns.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }
    if source.partition_columns.len() != partition_values.len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.local_parquet_source.inject_partitions",
            "partition value count must match source partition columns",
        )
        .with_fact("partition_columns", source.partition_columns.len())
        .with_fact("partition_values", partition_values.len()));
    }

    let rows = row_count_override.unwrap_or_else(|| frame.num_rows());
    let timer = Instant::now();
    let mut output = frame;
    for (partition_column, partition_value) in source.partition_columns.iter().zip(partition_values) {
        let series = partition_value_series(context, partition_value, rows).map_err(|error| {
            error
                .with_fact("partition_column", partition_column.name.as_str())
                .with_fact("partition_value", partition_value)
        })?;
        output = output.with_column(partition_column.name.as_str(), series).map_err(|error| {
            cudf_query_error(
                "query_engine.local_parquet_source.inject_partitions",
                "failed to inject native parquet partition column",
                error,
            )
            .with_fact("partition_column", partition_column.name.as_str())
            .with_fact("partition_value", partition_value)
        })?;
    }
    metrics.add_source_read_time(timer.elapsed());
    Ok(GpuDataFrame::new(output))
}

pub(super) fn partition_value_series(
    context: &ExecutionContext,
    value: &LiteralValue,
    rows: usize,
) -> Result<cudf::Series> {
    if let LiteralValue::Null(data_type) = value {
        let scalar = null_partition_value_scalar(context, data_type)?;
        return context.runtime()?.series_from_scalar(&scalar, rows).map_err(|error| {
            cudf_query_error(
                "query_engine.local_parquet_source.partition_value",
                "failed to construct native parquet partition column",
                error,
            )
        });
    }

    let scalar = match value {
        LiteralValue::Boolean(value) => context.runtime()?.scalar(*value),
        LiteralValue::Int8(value) => context.runtime()?.scalar(*value),
        LiteralValue::Int16(value) => context.runtime()?.scalar(*value),
        LiteralValue::Int32(value) => context.runtime()?.scalar(*value),
        LiteralValue::Int64(value) => context.runtime()?.scalar(*value),
        LiteralValue::UInt8(value) => context.runtime()?.scalar(*value),
        LiteralValue::UInt16(value) => context.runtime()?.scalar(*value),
        LiteralValue::UInt32(value) => context.runtime()?.scalar(*value),
        LiteralValue::UInt64(value) => context.runtime()?.scalar(*value),
        LiteralValue::Float32(value) => context.runtime()?.scalar(*value),
        LiteralValue::Float64(value) => context.runtime()?.scalar(*value),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => context.runtime()?.string_scalar(value),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            context.runtime()?.decimal128_raw_scalar(*raw, crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        LiteralValue::Date32(value) => {
            context.runtime()?.timestamp_raw_scalar(i64::from(*value), cudf::TimestampResolution::Days)
        },
        LiteralValue::Date64(value) => {
            context.runtime()?.timestamp_raw_scalar(*value, cudf::TimestampResolution::Milliseconds)
        },
        LiteralValue::Time32 { raw, unit } => {
            context.runtime()?.duration_raw_scalar(i64::from(*raw), timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Time64 { raw, unit } => {
            context.runtime()?.duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Timestamp { raw, unit } => {
            context.runtime()?.timestamp_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Duration { raw, unit } => {
            context.runtime()?.duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::IntervalMonthDayNano { .. } => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.local_parquet_source.partition_value",
                "interval partition values are not supported by native parquet sources",
            )
            .with_fact("partition_value", value));
        },
        LiteralValue::Null(_) => unreachable!("typed null partition values are handled before scalar construction"),
    }
    .map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.partition_value",
            "failed to construct native parquet partition scalar",
            error,
        )
    })?;

    context.runtime()?.series_from_scalar(&scalar, rows).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.partition_value",
            "failed to construct native parquet partition column",
            error,
        )
    })
}

fn null_partition_value_scalar(context: &ExecutionContext, data_type: &ArrowDataType) -> Result<cudf::core::Scalar> {
    let scalar = match data_type {
        ArrowDataType::Boolean => context.runtime()?.null_scalar::<bool>(),
        ArrowDataType::Int8 => context.runtime()?.null_scalar::<i8>(),
        ArrowDataType::Int16 => context.runtime()?.null_scalar::<i16>(),
        ArrowDataType::Int32 => context.runtime()?.null_scalar::<i32>(),
        ArrowDataType::Int64 => context.runtime()?.null_scalar::<i64>(),
        ArrowDataType::UInt8 => context.runtime()?.null_scalar::<u8>(),
        ArrowDataType::UInt16 => context.runtime()?.null_scalar::<u16>(),
        ArrowDataType::UInt32 => context.runtime()?.null_scalar::<u32>(),
        ArrowDataType::UInt64 => context.runtime()?.null_scalar::<u64>(),
        ArrowDataType::Float32 => context.runtime()?.null_scalar::<f32>(),
        ArrowDataType::Float64 => context.runtime()?.null_scalar::<f64>(),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            context.runtime()?.null_string_scalar()
        },
        ArrowDataType::Decimal128(_precision, scale) => {
            context.runtime()?.null_decimal128_raw_scalar(crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        ArrowDataType::Date32 => context.runtime()?.null_timestamp_raw_scalar(cudf::TimestampResolution::Days),
        ArrowDataType::Date64 => context.runtime()?.null_timestamp_raw_scalar(cudf::TimestampResolution::Milliseconds),
        ArrowDataType::Time32(unit) | ArrowDataType::Time64(unit) => {
            context.runtime()?.null_duration_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        ArrowDataType::Timestamp(unit, _) => {
            context.runtime()?.null_timestamp_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        ArrowDataType::Duration(unit) => {
            context.runtime()?.null_duration_raw_scalar(timestamp_unit_to_cudf_resolution(unit))
        },
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.local_parquet_source.partition_value",
                "native parquet partition null literal type is not supported",
            )
            .with_fact("data_type", format!("{data_type:?}")));
        },
    };

    scalar.map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.partition_value",
            "failed to construct native parquet partition null scalar",
            error,
        )
    })
}
