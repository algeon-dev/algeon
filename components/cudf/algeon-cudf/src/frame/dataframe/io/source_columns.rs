//! Source frame projection and missing column materialization.

use crate::{
    core::{
        error::{Error, ErrorCode, Result},
        scalar::Scalar,
        types::{DataType, TimestampResolution, TypeId},
    },
    frame::{dataframe::owned::DataFrame, series::Series},
    internal::hash::{FastHashSet, fast_hash_set_with_capacity},
    interop::arrow_host::all_null_list_series,
    io::source::{
        LocalParquetSource, PartitionValue, SourceChunk, SourceColumn, SourceColumnRef, SourceReadMode,
        ZERO_COLUMN_COUNT_COLUMN,
    },
    runtime::Runtime,
};

pub(super) fn missing_source_row_count_error(operation: &'static str, chunk: &SourceChunk) -> Error {
    Error::new_with(
        ErrorCode::ForeignInvalidArgument,
        operation,
        "source chunk requires row-count metadata because it has no decoded read columns",
    )
    .with_detail("reason", "missing_chunk_row_count")
    .with_detail("chunk_index", chunk.chunk_index().to_string())
    .with_detail("file_index", chunk.file_index().to_string())
    .with_detail("stable_id", chunk.stable_id().to_string())
    .with_detail("path", source_chunk_diagnostic_path(chunk))
}

pub(super) fn source_chunk_diagnostic_path(chunk: &SourceChunk) -> String {
    chunk.remote_source().map_or_else(|| chunk.path().display().to_string(), |source| source.diagnostic_location())
}

fn source_ref_series(
    frame: &DataFrame,
    source: &LocalParquetSource,
    source_ref: SourceColumnRef,
    materialized_name: &str,
) -> Result<Series> {
    match source_ref {
        SourceColumnRef::ReadColumn(index) => {
            let name = source.read_columns().get(index).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::IndexOutOfBounds,
                    "dataframe::read_parquet_source",
                    "read column reference index is out of bounds during source projection",
                )
                .with_detail("reason", "read_column_ref_out_of_bounds")
                .with_detail("read_column_count", source.read_columns().len().to_string())
            })?;
            frame.series(name)
        },
        SourceColumnRef::Partition(index) => {
            let column = source.partition_columns().get(index).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::IndexOutOfBounds,
                    "dataframe::read_parquet_source",
                    "partition column reference index is out of bounds during source projection",
                )
                .with_detail("reason", "partition_ref_out_of_bounds")
                .with_detail("partition_column_count", source.partition_columns().len().to_string())
            })?;
            frame.series(column.name())
        },
        SourceColumnRef::ZeroColumnCount => frame.series(ZERO_COLUMN_COUNT_COLUMN),
        SourceColumnRef::MissingNull => frame.series(materialized_name),
    }
}

pub(super) fn project_source_frame(
    frame: &DataFrame,
    source: &LocalParquetSource,
    mode: SourceReadMode,
) -> Result<DataFrame> {
    let mut pairs: Vec<(String, Series)> = Vec::new();
    let seen_capacity = source.output_columns().len()
        + if mode == SourceReadMode::ExecutionFrame { source.required_columns().len() } else { 0 };
    let mut seen: FastHashSet<String> = fast_hash_set_with_capacity(seen_capacity);

    for output in source.output_columns() {
        let series = source_ref_series(frame, source, output.source(), output.name())?;
        seen.insert(output.name().to_string());
        pairs.push((output.name().to_string(), series));
    }

    if mode == SourceReadMode::ExecutionFrame {
        for required in source.required_columns() {
            if seen.contains(required.name()) {
                continue;
            }
            let series = source_ref_series(frame, source, required.source(), required.name())?;
            seen.insert(required.name().to_string());
            pairs.push((required.name().to_string(), series));
        }
    }

    if pairs.is_empty() {
        return frame.select(Vec::<&str>::new()).map_err(|error| error.reanchor("dataframe::read_parquet_source"));
    }

    DataFrame::from_series(pairs).map_err(|error| error.reanchor("dataframe::read_parquet_source"))
}

pub(super) fn empty_parquet_source_frame(
    runtime: &Runtime,
    source: &LocalParquetSource,
    mode: SourceReadMode,
) -> Result<DataFrame> {
    let mut pairs: Vec<(String, Series)> = Vec::new();
    let seen_capacity = source.output_columns().len()
        + if mode == SourceReadMode::ExecutionFrame { source.required_columns().len() } else { 0 };
    let mut seen: FastHashSet<String> = fast_hash_set_with_capacity(seen_capacity);

    for output in source.output_columns() {
        let schema_column = source.schema().column(output.name()).ok_or_else(|| {
            Error::new_with(
                ErrorCode::MissingColumn,
                "dataframe::read_parquet_source",
                "source schema does not contain output column for typed empty source",
            )
            .with_column(output.name().to_string())
            .with_detail("reason", "schema_column_missing")
        })?;
        pairs.push((output.name().to_string(), empty_source_series(runtime, schema_column)?));
        seen.insert(output.name().to_string());
    }

    if mode == SourceReadMode::ExecutionFrame {
        for required in source.required_columns() {
            if seen.contains(required.name()) {
                continue;
            }
            let schema_column = source.schema().column(required.name()).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::MissingColumn,
                    "dataframe::read_parquet_source",
                    "source schema does not contain required column for typed empty source",
                )
                .with_column(required.name().to_string())
                .with_detail("reason", "schema_column_missing")
            })?;
            pairs.push((required.name().to_string(), empty_source_series(runtime, schema_column)?));
            seen.insert(required.name().to_string());
        }
    }

    if pairs.is_empty() {
        return Ok(runtime.empty_dataframe());
    }

    DataFrame::from_series(pairs).map_err(|error| error.reanchor("dataframe::read_parquet_source"))
}

fn null_source_scalar(runtime: &Runtime, data_type: DataType) -> Result<Scalar> {
    match data_type.id() {
        TypeId::Bool8 => runtime.null_scalar::<bool>(),
        TypeId::Int8 => runtime.null_scalar::<i8>(),
        TypeId::Int16 => runtime.null_scalar::<i16>(),
        TypeId::Int32 => runtime.null_scalar::<i32>(),
        TypeId::Int64 => runtime.null_scalar::<i64>(),
        TypeId::UInt8 => runtime.null_scalar::<u8>(),
        TypeId::UInt16 => runtime.null_scalar::<u16>(),
        TypeId::UInt32 => runtime.null_scalar::<u32>(),
        TypeId::UInt64 => runtime.null_scalar::<u64>(),
        TypeId::Float32 => runtime.null_scalar::<f32>(),
        TypeId::Float64 => runtime.null_scalar::<f64>(),
        TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => runtime.null_decimal_raw_scalar(data_type),
        TypeId::TimestampDays
        | TypeId::TimestampSeconds
        | TypeId::TimestampMilliseconds
        | TypeId::TimestampMicroseconds
        | TypeId::TimestampNanoseconds => {
            let resolution = data_type.id().timestamp_resolution().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    "dataframe::read_parquet_source",
                    "source null timestamp dtype is not supported",
                )
                .with_detail("reason", "unsupported_source_null_dtype")
            })?;
            runtime.null_timestamp_raw_scalar(resolution)
        },
        TypeId::DurationDays
        | TypeId::DurationSeconds
        | TypeId::DurationMilliseconds
        | TypeId::DurationMicroseconds
        | TypeId::DurationNanoseconds => {
            let resolution = data_type.id().duration_resolution().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    "dataframe::read_parquet_source",
                    "source null duration dtype is not supported",
                )
                .with_detail("reason", "unsupported_source_null_dtype")
            })?;
            runtime.null_duration_raw_scalar(resolution)
        },
        TypeId::String => runtime.null_string_scalar(),
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "dataframe::read_parquet_source",
            "source null dtype is not supported",
        )
        .with_detail("reason", "unsupported_source_null_dtype")),
    }
}

fn empty_source_series(runtime: &Runtime, column: &SourceColumn) -> Result<Series> {
    match column.arrow_data_type() {
        Some(data_type) => all_null_list_series(runtime, data_type, 0),
        None => runtime.empty_series(column.data_type()),
    }
}

fn all_null_source_series(runtime: &Runtime, column: &SourceColumn, rows: usize) -> Result<Series> {
    match column.arrow_data_type() {
        Some(data_type) => all_null_list_series(runtime, data_type, rows),
        None => {
            let scalar = null_source_scalar(runtime, column.data_type())?;
            runtime.series_from_scalar(&scalar, rows)
        },
    }
}

fn partition_scalar_series(runtime: &Runtime, value: &PartitionValue, len: usize) -> Result<Series> {
    let scalar = match value {
        PartitionValue::Null(data_type) => null_source_scalar(runtime, *data_type)?,
        PartitionValue::Boolean(value) => runtime.scalar(*value)?,
        PartitionValue::Int8(value) => runtime.scalar(*value)?,
        PartitionValue::Int16(value) => runtime.scalar(*value)?,
        PartitionValue::Int32(value) => runtime.scalar(*value)?,
        PartitionValue::Int64(value) => runtime.scalar(*value)?,
        PartitionValue::UInt8(value) => runtime.scalar(*value)?,
        PartitionValue::UInt16(value) => runtime.scalar(*value)?,
        PartitionValue::UInt32(value) => runtime.scalar(*value)?,
        PartitionValue::UInt64(value) => runtime.scalar(*value)?,
        PartitionValue::Float32(value) => runtime.scalar(*value)?,
        PartitionValue::Float64(value) => runtime.scalar(*value)?,
        PartitionValue::Decimal128 { raw, scale } => runtime.decimal128_raw_scalar(*raw, *scale)?,
        PartitionValue::Date32(value) => runtime.timestamp_raw_scalar(i64::from(*value), TimestampResolution::Days)?,
        PartitionValue::Timestamp { raw, resolution } => runtime.timestamp_raw_scalar(*raw, *resolution)?,
        PartitionValue::Duration { raw, resolution } => runtime.duration_raw_scalar(*raw, *resolution)?,
        PartitionValue::String(value) => runtime.string_scalar(value)?,
    };
    runtime.series_from_scalar(&scalar, len)
}

pub(super) fn materialize_missing_null_columns(
    runtime: &Runtime,
    source: &LocalParquetSource,
    frame: DataFrame,
    row_count_override: Option<usize>,
    mode: SourceReadMode,
) -> Result<DataFrame> {
    let row_count = row_count_override.unwrap_or_else(|| frame.num_rows());
    let mut frame = frame;
    let planned_count = source.output_columns().len()
        + if mode == SourceReadMode::ExecutionFrame { source.required_columns().len() } else { 0 };
    let mut seen = fast_hash_set_with_capacity(planned_count);

    for output in source.output_columns() {
        if output.source() != SourceColumnRef::MissingNull || !seen.insert(output.name().to_string()) {
            continue;
        }
        frame = materialize_one_missing_null_column(runtime, source, frame, output.name(), row_count)?;
    }

    if mode == SourceReadMode::ExecutionFrame {
        for required in source.required_columns() {
            if required.source() != SourceColumnRef::MissingNull || !seen.insert(required.name().to_string()) {
                continue;
            }
            frame = materialize_one_missing_null_column(runtime, source, frame, required.name(), row_count)?;
        }
    }

    Ok(frame)
}

fn materialize_one_missing_null_column(
    runtime: &Runtime,
    source: &LocalParquetSource,
    frame: DataFrame,
    name: &str,
    row_count: usize,
) -> Result<DataFrame> {
    let schema_column = source.schema().column(name).ok_or_else(|| {
        Error::new_with(
            ErrorCode::MissingColumn,
            "dataframe::read_parquet_source",
            "source schema does not contain missing nullable column",
        )
        .with_column(name.to_string())
        .with_detail("reason", "schema_column_missing")
    })?;
    if !schema_column.nullable() {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            "dataframe::read_parquet_source",
            "missing source columns can only materialize nullable target columns",
        )
        .with_column(name.to_string())
        .with_detail("reason", "missing_null_for_non_nullable_column"));
    }
    let series = all_null_source_series(runtime, schema_column, row_count).map_err(|error| {
        error
            .reanchor("dataframe::read_parquet_source")
            .with_detail("reason", "missing_null_materialization_failed")
            .with_column(name.to_string())
    })?;
    frame.with_column(name, series).map_err(|error| {
        error
            .reanchor("dataframe::read_parquet_source")
            .with_detail("reason", "missing_null_column_insert_failed")
            .with_column(name.to_string())
    })
}

pub(super) fn materialize_all_null_read_columns(
    runtime: &Runtime,
    source: &LocalParquetSource,
    frame: DataFrame,
    column_names: &[String],
    row_count: usize,
) -> Result<DataFrame> {
    let mut frame = frame;
    for name in column_names {
        frame = materialize_one_all_null_read_column(runtime, source, frame, name, row_count)?;
    }
    Ok(frame)
}

fn materialize_one_all_null_read_column(
    runtime: &Runtime,
    source: &LocalParquetSource,
    frame: DataFrame,
    name: &str,
    row_count: usize,
) -> Result<DataFrame> {
    let schema_column = source.schema().column(name).ok_or_else(|| {
        Error::new_with(
            ErrorCode::MissingColumn,
            "dataframe::read_parquet_source",
            "source schema does not contain all-null read column",
        )
        .with_column(name.to_string())
        .with_detail("reason", "schema_column_missing")
    })?;
    if !schema_column.nullable() {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            "dataframe::read_parquet_source",
            "all-null source columns can only materialize nullable target columns",
        )
        .with_column(name.to_string())
        .with_detail("reason", "all_null_read_for_non_nullable_column"));
    }
    let series = all_null_source_series(runtime, schema_column, row_count).map_err(|error| {
        error
            .reanchor("dataframe::read_parquet_source")
            .with_detail("reason", "all_null_read_materialization_failed")
            .with_column(name.to_string())
    })?;
    frame.with_column(name, series).map_err(|error| {
        error
            .reanchor("dataframe::read_parquet_source")
            .with_detail("reason", "all_null_read_column_insert_failed")
            .with_column(name.to_string())
    })
}

pub(super) fn materialize_partition_columns(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    frame: DataFrame,
    row_count_override: Option<usize>,
) -> Result<DataFrame> {
    let mut frame = frame;
    let row_count = row_count_override.unwrap_or_else(|| frame.num_rows());
    for (partition_index, (column, value)) in
        source.partition_columns().iter().zip(chunk.partition_values()).enumerate()
    {
        let series = partition_scalar_series(runtime, value, row_count).map_err(|error| {
            error
                .reanchor("dataframe::read_parquet_source")
                .with_detail("reason", "partition_materialization_failed")
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name().to_string())
                .with_detail("chunk_index", chunk.chunk_index().to_string())
                .with_detail("file_index", chunk.file_index().to_string())
                .with_detail("stable_id", chunk.stable_id().to_string())
                .with_detail("path", source_chunk_diagnostic_path(chunk))
        })?;
        frame = frame.with_column(column.name(), series).map_err(|error| {
            error
                .reanchor("dataframe::read_parquet_source")
                .with_detail("reason", "partition_column_insert_failed")
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name().to_string())
        })?;
    }
    Ok(frame)
}
