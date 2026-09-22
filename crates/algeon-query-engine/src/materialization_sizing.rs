use arrow::datatypes::DataType as ArrowDataType;
use cudf::ops::materialization::{
    MaterializedColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES as CUDF_PER_COLUMN_ALIGNMENT_PADDING_BYTES,
    STRING_OFFSET_BYTES_CONSERVATIVE as CUDF_STRING_OFFSET_BYTES_CONSERVATIVE,
    TABLE_UPPER_BOUND_PREFLIGHT_OPERATION as CUDF_TABLE_UPPER_BOUND_PREFLIGHT_OPERATION, table_upper_bound_preflight,
};

use crate::{
    datatypes::cudf_dtype_from_arrow_type,
    error::{QueryError, QueryErrorCode, Result},
    native_error::cudf_query_error,
};

pub const STRING_OFFSET_BYTES_CONSERVATIVE: u64 = CUDF_STRING_OFFSET_BYTES_CONSERVATIVE as u64;
pub const PER_COLUMN_ALIGNMENT_PADDING_BYTES: u64 = CUDF_PER_COLUMN_ALIGNMENT_PADDING_BYTES as u64;
pub const MATERIALIZED_TABLE_PREFLIGHT_OPERATION: &str = CUDF_TABLE_UPPER_BOUND_PREFLIGHT_OPERATION;

#[derive(Debug, Clone)]
pub struct JoinOutputColumnUpperBoundInput {
    pub data_type: ArrowDataType,
    pub input_max_string_length_bytes: Option<u64>,
    pub input_total_string_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub enum PartialAggregateShape {
    Global,
    Grouped { input_rows: u64 },
}

impl PartialAggregateShape {
    pub fn output_row_count_upper_bound(&self) -> u64 {
        match self {
            Self::Global => 1,
            Self::Grouped { input_rows } => *input_rows,
        }
    }
}

pub fn join_output_upper_bound_bytes(
    expected_output_rows: u64,
    columns: &[JoinOutputColumnUpperBoundInput],
) -> Result<u64> {
    let output_rows = usize::try_from(expected_output_rows).map_err(|_| {
        QueryError::new(
            QueryErrorCode::SizeOverflow,
            "query_engine.bounded.join.sizing",
            "materialized output row count exceeds usize",
        )
        .with_fact("output_rows", expected_output_rows)
    })?;
    let columns = columns
        .iter()
        .map(|column| {
            let data_type = match &column.data_type {
                ArrowDataType::Timestamp(_, Some(_)) => return Err(unsupported_dtype(column)),
                data_type => cudf_dtype_from_arrow_type(data_type)
                    .map_err(|error| unsupported_dtype(column).with_source(error))?,
            };
            Ok(MaterializedColumnUpperBoundInput {
                data_type,
                input_max_string_length_bytes: column
                    .input_max_string_length_bytes
                    .map(usize::try_from)
                    .transpose()
                    .map_err(|_| payload_exceeds_usize(column))?,
                input_total_string_bytes: column
                    .input_total_string_bytes
                    .map(usize::try_from)
                    .transpose()
                    .map_err(|_| payload_exceeds_usize(column))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let preflight = table_upper_bound_preflight(output_rows, &columns).map_err(materialization_preflight_error)?;
    u64::try_from(preflight.checked_peak_bytes()).map_err(|_| {
        QueryError::new(
            QueryErrorCode::SizeOverflow,
            "query_engine.bounded.join.sizing",
            "cuDF materialized-table size preflight exceeds u64",
        )
        .with_fact("component_owner_operation", preflight.component_owner_operation())
    })
}

fn materialization_preflight_error(error: cudf::Error) -> QueryError {
    let reason = error.detail_value("reason").map(str::to_owned);
    let code = match (error.code(), reason.as_deref()) {
        (cudf::ErrorCode::UnsupportedDtype, _) => QueryErrorCode::UnsupportedPlanShape,
        (cudf::ErrorCode::UnsupportedOperation, Some("string_payload_evidence_missing")) => QueryErrorCode::InvalidPlan,
        _ => {
            return cudf_query_error(
                "query_engine.bounded.join.sizing",
                "cuDF materialized-table size preflight rejected the output contract",
                error,
            );
        },
    };
    let mut mapped = QueryError::new(
        code,
        "query_engine.bounded.join.sizing",
        "cuDF materialized-table size preflight rejected the output contract",
    )
    .with_fact("component_owner_operation", error.operation());
    if let Some(reason) = reason {
        mapped = mapped.with_fact("reason", reason);
    }
    mapped.with_source(error)
}

pub fn partial_aggregate_upper_bound_bytes(
    shape: PartialAggregateShape,
    columns: &[JoinOutputColumnUpperBoundInput],
) -> Result<u64> {
    join_output_upper_bound_bytes(shape.output_row_count_upper_bound(), columns)
}

fn unsupported_dtype(column: &JoinOutputColumnUpperBoundInput) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "query_engine.bounded.join.sizing",
        "materialized output column dtype is not supported by the cuDF owner preflight",
    )
    .with_fact("dtype", format!("{:?}", column.data_type))
    .with_fact("reason", "unsupported_join_output_dtype")
}

fn payload_exceeds_usize(column: &JoinOutputColumnUpperBoundInput) -> QueryError {
    QueryError::new(
        QueryErrorCode::SizeOverflow,
        "query_engine.bounded.join.sizing",
        "string payload evidence exceeds usize",
    )
    .with_fact("dtype", format!("{:?}", column.data_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_preflight_preserves_materialized_string_bound() {
        let rows = 109_914_910;
        let total_payload = 4_706_637_611;
        let bound = join_output_upper_bound_bytes(
            rows,
            &[JoinOutputColumnUpperBoundInput {
                data_type: ArrowDataType::Utf8,
                input_max_string_length_bytes: None,
                input_total_string_bytes: Some(total_payload),
            }],
        )
        .unwrap();
        let null_mask = rows.div_ceil(8).div_ceil(64) * 64;
        assert_eq!(
            bound,
            total_payload
                + (rows + 1) * STRING_OFFSET_BYTES_CONSERVATIVE
                + null_mask
                + PER_COLUMN_ALIGNMENT_PADDING_BYTES
        );
    }
}
