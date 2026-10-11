use std::time::Instant;

use cudf::{Exec, ScalarValue, interop::arrow_host::all_null_list_series};
use query_planning::{datatypes::is_supported_list_carry_type, source::MissingColumn};

use super::{
    DataFrame, ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, LocalParquetSource, QueryError,
    QueryErrorCode, Result, budgeted_value, cudf_dtype_from_arrow_type, cudf_query_error, cudf_unsupported_type_error,
};

/// Adds the source's missing columns (declared in the logical schema, absent from the files) as
/// all-null columns of the declared type. Columns already in the frame are skipped.
///
/// `row_count_override` supplies the row count when the frame has no columns to take it from.
/// Only nullable columns can be missing; a non-nullable one fails with `InvalidPlan`. A list
/// column is built from its Arrow type by `all_null_list_series`, because a cuDF list dtype does
/// not carry the element type.
pub(super) fn materialize_missing_columns(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    mut frame: DataFrame,
    row_count_override: Option<usize>,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    if source.missing_columns.is_empty() {
        return Ok(frame);
    }

    let rows = row_count_override.unwrap_or_else(|| frame.num_rows());
    let timer = Instant::now();
    for (missing_index, missing_column) in source.missing_columns.iter().enumerate() {
        if frame.column_names().iter().any(|name| name == &missing_column.name) {
            continue;
        }
        let series = missing_column_series(context, missing_column, rows).map_err(|error| {
            error.with_fact("missing_index", missing_index).with_fact("missing_column", missing_column.name.as_str())
        })?;
        frame = frame.with_column(missing_column.name.as_str(), series).map_err(|error| {
            cudf_query_error(
                "query_engine.local_parquet_source.missing_column",
                "failed to materialize native parquet missing nullable column",
                error,
            )
            .with_fact("missing_index", missing_index)
            .with_fact("missing_column", missing_column.name.as_str())
            .with_fact("missing_type", format!("{:?}", missing_column.data_type))
        })?;
    }
    metrics.add_source_read_time(timer.elapsed());
    Ok(frame)
}

fn missing_column_series(
    context: &ExecutionContext,
    missing_column: &MissingColumn,
    rows: usize,
) -> Result<cudf::Series> {
    if !missing_column.nullable {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.local_parquet_source.missing_column",
            "missing source columns can only represent nullable logical columns",
        )
        .with_fact("reason", "non_nullable_missing_column")
        .with_fact("missing_type", format!("{:?}", missing_column.data_type)));
    }

    if is_supported_list_carry_type(&missing_column.data_type) {
        return all_null_list_series(&context.runtime()?, &missing_column.data_type, rows).map_err(|error| {
            cudf_query_error(
                "query_engine.local_parquet_source.missing_column",
                "failed to construct native parquet missing nullable list column",
                error,
            )
        });
    }

    let dtype = cudf_dtype_from_arrow_type(&missing_column.data_type).map_err(|_| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.local_parquet_source.missing_column",
            "missing source column type is not supported by native execution",
        )
        .with_fact("reason", "unsupported_missing_column_type")
        .with_fact("missing_type", format!("{:?}", missing_column.data_type))
    })?;
    let scalar = context.runtime()?.scalar(ScalarValue::Null(dtype)).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.missing_column",
            "failed to construct native parquet missing nullable scalar",
            error,
        )
    })?;

    context.runtime()?.series_from_scalar(&scalar, rows).map_err(|error| {
        cudf_query_error(
            "query_engine.local_parquet_source.missing_column",
            "failed to construct native parquet missing nullable column",
            error,
        )
    })
}

/// Casts each decoded column whose cuDF dtype differs from the dtype of the matching schema field.
///
/// cuDF picks the storage type from the Parquet encoding, which can differ from the logical
/// Arrow schema; a decimal that cuDF decodes as Decimal32 or Decimal64 under a `Decimal128`
/// schema field is one case. Columns absent from the frame are skipped, so the call is valid
/// before and after partition and missing columns are added. With `token` the casts run under
/// that execution budget and are recorded against it; with `None` they run unbudgeted on the
/// retained stream.
pub(super) fn coerce_local_parquet_source_schema(
    mut frame: DataFrame,
    source: &LocalParquetSource,
    token: Option<&ExecutionBudgetToken>,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let timer = Instant::now();
    for (field_index, field) in source.schema.fields().iter().enumerate() {
        let field_name = field.name();
        if !frame.column_names().contains(&field_name.as_str()) {
            continue;
        }
        let expected_dtype = cudf_dtype_from_arrow_type(field.data_type()).map_err(|error| {
            cudf_unsupported_type_error(
                "query_engine.local_parquet_source.schema_coerce",
                "local parquet source schema type is not supported by native execution",
                error,
            )
            .with_fact("field_name", field_name.as_str())
            .with_fact("field_index", field_index)
            .with_fact("expected_type", format!("{:?}", field.data_type()))
        })?;
        let actual_dtype = frame.series(field_name.as_str()).and_then(|series| series.dtype()).map_err(|error| {
            cudf_query_error(
                "query_engine.local_parquet_source.schema_coerce",
                "failed to inspect native local parquet source column type",
                error,
            )
            .with_fact("field_name", field_name.as_str())
            .with_fact("field_index", field_index)
        })?;
        if actual_dtype == expected_dtype {
            continue;
        }
        frame = match token {
            Some(token) => budgeted_value(
                token
                    .run("dataframe::cast_column", |exec| {
                        frame.on(exec).cast_column(field_name.as_str(), expected_dtype)
                    })
                    .map_err(|error| {
                        cudf_query_error(
                            "query_engine.local_parquet_source.schema_coerce",
                            "failed to cast bounded native local parquet source column to declared schema type",
                            error,
                        )
                        .with_fact("field_name", field_name.as_str())
                        .with_fact("field_index", field_index)
                        .with_fact("actual_dtype", format!("{actual_dtype:?}"))
                        .with_fact("expected_type", format!("{:?}", field.data_type()))
                    })?,
                metrics,
                "query_engine.local_parquet_source.schema_coerce",
            ),
            None => frame.on(Exec::RETAINED).cast_column(field_name.as_str(), expected_dtype).map_err(|error| {
                cudf_query_error(
                    "query_engine.local_parquet_source.schema_coerce",
                    "failed to cast native local parquet source column to declared schema type",
                    error,
                )
                .with_fact("field_name", field_name.as_str())
                .with_fact("field_index", field_index)
                .with_fact("actual_dtype", format!("{actual_dtype:?}"))
                .with_fact("expected_type", format!("{:?}", field.data_type()))
            })?,
        };
    }
    metrics.add_source_read_time(timer.elapsed());
    Ok(frame)
}
