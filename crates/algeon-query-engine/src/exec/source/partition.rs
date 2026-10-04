use std::time::Instant;

use cudf::{ScalarValue, interop::arrow_host::cudf_dtype_from_arrow_type};

use super::{
    DataFrame, ExecutionContext, ExecutionMetrics, GpuDataFrame, LiteralValue, LocalParquetSource, QueryError,
    QueryErrorCode, Result, cudf_query_error,
};
use crate::datatypes::{UnrepresentableLiteral, cudf_scalar_value};

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
    const OPERATION: &str = "query_engine.local_parquet_source.partition_value";
    let scalar_value = match cudf_scalar_value(value) {
        Ok(scalar_value) => scalar_value,
        Err(UnrepresentableLiteral::Interval) => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                OPERATION,
                "interval partition values are not supported by native parquet sources",
            )
            .with_fact("partition_value", value));
        },
        Err(UnrepresentableLiteral::Null(data_type)) => {
            let dtype = cudf_dtype_from_arrow_type(data_type).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    OPERATION,
                    "native parquet partition null literal type is not supported",
                )
                .with_fact("data_type", format!("{data_type:?}"))
            })?;
            ScalarValue::Null(dtype)
        },
    };
    let runtime = context.runtime()?;
    let scalar = runtime
        .scalar(scalar_value)
        .map_err(|error| cudf_query_error(OPERATION, "failed to construct native parquet partition scalar", error))?;
    runtime
        .series_from_scalar(&scalar, rows)
        .map_err(|error| cudf_query_error(OPERATION, "failed to construct native parquet partition column", error))
}
