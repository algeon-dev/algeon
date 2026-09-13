use std::collections::BTreeMap;

use arrow::datatypes::DataType as ArrowDataType;
use cudf::interop::arrow_host::arrow_type_from_cudf_dtype as cudf_arrow_type_from_cudf_dtype;

use super::{
    BoundedExecutionAttempt, DataFrame, GpuDataFrame, JoinOutputColumn, JoinOutputColumnUpperBoundInput, JoinSide,
    NodeId, QueryError, QueryErrorCode, ResidentGpuFrame, Result, cudf_query_error,
    execution_budget_token_for_boundary, join_input_columns_with_order, join_output_upper_bound_bytes,
    join_side_projection_with_order, observe_device_memory_bytes, select_join_input,
};

pub(in crate::exec::bounded) struct BoundedJoinSideFrame {
    pub(in crate::exec::bounded) frame: DataFrame,
    pub(in crate::exec::bounded) rows: u64,
    pub(in crate::exec::bounded) bytes: u64,
}

pub(in crate::exec::bounded) fn release_bounded_join_side_frame(side: BoundedJoinSideFrame) {
    // Release the input before the caller starts another allocation on the query domain.
    drop(side);
}

pub(in crate::exec::bounded) fn prepare_bounded_join_side_input_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: ResidentGpuFrame,
    columns: &[String],
    input_side: JoinSide,
) -> Result<BoundedJoinSideFrame> {
    prepare_bounded_join_side_input(execution, input.frame, columns, input_side)
}
pub(in crate::exec::bounded) fn prepare_bounded_join_side_input_from_shared_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: GpuDataFrame,
    columns: &[String],
    input_side: JoinSide,
) -> Result<BoundedJoinSideFrame> {
    prepare_bounded_join_side_input(execution, input, columns, input_side)
}
fn prepare_bounded_join_side_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: GpuDataFrame,
    columns: &[String],
    input_side: JoinSide,
) -> Result<BoundedJoinSideFrame> {
    let input_frame = input.into_frame();
    let columns = join_input_columns_with_order(&input_frame, columns);
    let selected = match select_join_input(input_frame, &columns, join_side_label(input_side)) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error);
        },
    };
    let rows = selected.num_rows() as u64;
    let bytes = match observe_device_memory_bytes(&GpuDataFrame::new(selected.clone()), execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    Ok(BoundedJoinSideFrame { frame: selected, rows, bytes })
}

fn join_side_label(side: JoinSide) -> &'static str {
    match side {
        JoinSide::Left => "left",
        JoinSide::Right => "right",
    }
}

pub(super) fn join_side_output_upper_bound_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    expected_output_rows: u64,
    columns: &[String],
    side: JoinSide,
) -> Result<u64> {
    let bounds = join_side_output_upper_bound_bytes_for_row_counts(
        execution,
        node_id,
        frame,
        &[expected_output_rows],
        columns,
        side,
    )?;
    Ok(bounds.into_iter().next().unwrap_or(0))
}

pub(super) fn join_side_output_upper_bound_bytes_for_row_counts(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    expected_output_row_counts: &[u64],
    columns: &[String],
    side: JoinSide,
) -> Result<Vec<u64>> {
    let mut columns = columns.to_vec();
    join_side_projection_with_order(frame, &mut columns);
    if columns.is_empty() {
        return Ok(vec![0; expected_output_row_counts.len()]);
    }

    let mut upper_bound_columns = Vec::with_capacity(columns.len());
    let mut column_cache: BTreeMap<String, JoinOutputColumnBound> = BTreeMap::new();
    for column in columns {
        let upper_bound = if let Some(upper_bound) = column_cache.get(column.as_str()) {
            upper_bound.clone()
        } else {
            let upper_bound =
                join_output_upper_bound_column_for_frame_column(execution, node_id, frame, column.as_str(), side)?;
            column_cache.insert(column.clone(), upper_bound.clone());
            upper_bound
        };
        upper_bound_columns.push(upper_bound);
    }
    expected_output_row_counts
        .iter()
        .map(|expected_output_rows| {
            join_output_columns_upper_bound_bytes(*expected_output_rows, &upper_bound_columns).map_err(|error| {
                error
                    .with_fact_if_absent("node_id", node_id)
                    .with_fact_if_absent("side", side)
                    .with_fact_if_absent("expected_output_rows", *expected_output_rows)
            })
        })
        .collect()
}

#[derive(Clone)]
enum JoinOutputColumnBound {
    Standard(JoinOutputColumnUpperBoundInput),
    List { input_column_bytes: u64 },
}

fn join_output_columns_upper_bound_bytes(expected_output_rows: u64, columns: &[JoinOutputColumnBound]) -> Result<u64> {
    columns.iter().try_fold(0u64, |total, column| {
        let column_bytes = match column {
            JoinOutputColumnBound::Standard(column) => {
                join_output_upper_bound_bytes(expected_output_rows, std::slice::from_ref(column))?
            },
            JoinOutputColumnBound::List { input_column_bytes } => {
                // A gather output row cannot retain more child payload than the entire input list column.
                let gathered = expected_output_rows.checked_mul(*input_column_bytes).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::SizeOverflow,
                        "query_engine.bounded.join.output_upper_bound",
                        "list join output byte bound overflowed",
                    )
                    .with_fact("reason", "list_output_bound_overflow")
                })?;
                let offsets =
                    expected_output_rows.checked_add(1).and_then(|rows| rows.checked_mul(4)).ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::SizeOverflow,
                            "query_engine.bounded.join.output_upper_bound",
                            "list join output offset bound overflowed",
                        )
                        .with_fact("reason", "list_output_offset_bound_overflow")
                    })?;
                let validity = if expected_output_rows == 0 {
                    0
                } else {
                    expected_output_rows.div_ceil(8).div_ceil(64).saturating_mul(64)
                };
                gathered.checked_add(offsets).and_then(|bytes| bytes.checked_add(validity)).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::SizeOverflow,
                        "query_engine.bounded.join.output_upper_bound",
                        "list join output structural byte bound overflowed",
                    )
                    .with_fact("reason", "list_output_structural_bound_overflow")
                })?
            },
        };
        total.checked_add(column_bytes).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::SizeOverflow,
                "query_engine.bounded.join.output_upper_bound",
                "join output column byte bounds overflowed",
            )
            .with_fact("reason", "join_output_bound_overflow")
        })
    })
}

pub(super) fn projected_output_columns_for_retained_sizing(
    projection: &[JoinOutputColumn],
    side: JoinSide,
) -> Vec<String> {
    projection.iter().filter(|column| column.side == side).map(|column| column.column.name().to_string()).collect()
}

fn join_output_upper_bound_column_for_frame_column(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    column: &str,
    side: JoinSide,
) -> Result<JoinOutputColumnBound> {
    let series = frame.series(column).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.join.output_upper_bound",
            "failed to access join output source column for closed-form sizing",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("side", side)
        .with_fact("column", column)
    })?;
    let cudf_dtype = series.dtype().map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.join.output_upper_bound",
            "failed to inspect join output source column dtype for closed-form sizing",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("side", side)
        .with_fact("column", column)
    })?;
    if cudf_dtype.id() == cudf::TypeId::List {
        let input_column_bytes = u64::try_from(series.device_allocated_size_bytes().map_err(|error| {
            cudf_query_error(
                "query_engine.bounded.join.output_upper_bound",
                "failed to inspect list join output source bytes for closed-form sizing",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column)
        })?)
        .map_err(|_| {
            QueryError::new(
                QueryErrorCode::SizeOverflow,
                "query_engine.bounded.join.output_upper_bound",
                "list join output source bytes exceed the reservation ledger range",
            )
            .with_fact("reason", "list_source_bytes_exceed_u64")
        })?;
        return Ok(JoinOutputColumnBound::List { input_column_bytes });
    }
    let data_type = cudf_arrow_type_from_cudf_dtype(cudf_dtype).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.join.output_upper_bound",
            "failed to convert join output source column dtype for closed-form sizing",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("side", side)
        .with_fact("column", column)
        .with_fact("cudf_dtype", format!("{cudf_dtype:?}"))
    })?;
    let input_max_string_length_bytes = if join_output_dtype_needs_string_max(&data_type) {
        Some(max_string_byte_length_for_join_output_sizing(execution, node_id, frame, column, side)?)
    } else {
        None
    };
    Ok(JoinOutputColumnBound::Standard(JoinOutputColumnUpperBoundInput {
        data_type,
        input_max_string_length_bytes,
        input_total_string_bytes: None,
    }))
}

fn max_string_byte_length_for_join_output_sizing(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: &DataFrame,
    column: &str,
    side: JoinSide,
) -> Result<u64> {
    if frame.num_rows() == 0 {
        return Ok(0);
    }
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "join_output_string_length_sizing",
        "query_engine.bounded.join.output_string_length_sizing.budget_token",
        "failed to create bounded native join output string-length sizing budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error.with_fact("node_id", node_id).with_fact("side", side).with_fact("column", column));
        },
    };

    let series = match frame.series(column) {
        Ok(series) => series,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.join.output_string_length_sizing",
                "failed to access string join output source column for sizing",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column));
        },
    };
    let byte_counts = match series.str().byte_count_in(&token.execution_overrides()) {
        Ok(byte_counts) => byte_counts,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.join.output_string_length_sizing",
                "failed to compute string byte counts for join output sizing",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column));
        },
    };
    let max_len = match byte_counts.max_value_in::<i32>(&token.execution_overrides()) {
        Ok(max_len) => max_len,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.join.output_string_length_sizing",
                "failed to reduce string byte-count sizing column",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("side", side)
            .with_fact("column", column));
        },
    };
    Ok(max_len.unwrap_or(0).max(0) as u64)
}

fn join_output_dtype_needs_string_max(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}
