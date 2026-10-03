use arrow::datatypes::DataType as ArrowDataType;
use cudf::ops::materialization::{
    MaterializedColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES as CUDF_PER_COLUMN_ALIGNMENT_PADDING_BYTES,
    STRING_OFFSET_BYTES_CONSERVATIVE as CUDF_STRING_OFFSET_BYTES_CONSERVATIVE, table_upper_bound_preflight,
};

use crate::{
    datatypes::cudf_dtype_from_arrow_type,
    error::{QueryError, QueryErrorCode, Result},
    native_error::cudf_query_error,
};

pub const STRING_OFFSET_BYTES_CONSERVATIVE: u64 = CUDF_STRING_OFFSET_BYTES_CONSERVATIVE as u64;
pub const PER_COLUMN_ALIGNMENT_PADDING_BYTES: u64 = CUDF_PER_COLUMN_ALIGNMENT_PADDING_BYTES as u64;

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

/// The bytes a materialized join output of `expected_output_rows` rows can
/// retain, or `None` when the cuDF byte model cannot bound one of its columns
/// (an unmodelled dtype, or a count that overflows). Missing string payload
/// evidence is a caller contract error.
pub fn join_output_upper_bound_bytes(
    expected_output_rows: u64,
    columns: &[JoinOutputColumnUpperBoundInput],
) -> Result<Option<u64>> {
    let Ok(output_rows) = usize::try_from(expected_output_rows) else {
        return Ok(None);
    };
    let mut inputs = Vec::with_capacity(columns.len());
    for column in columns {
        let data_type = match &column.data_type {
            ArrowDataType::Timestamp(_, Some(_)) => return Ok(None),
            data_type => match cudf_dtype_from_arrow_type(data_type) {
                Ok(data_type) => data_type,
                Err(_) => return Ok(None),
            },
        };
        let (Ok(input_max_string_length_bytes), Ok(input_total_string_bytes)) = (
            column.input_max_string_length_bytes.map(usize::try_from).transpose(),
            column.input_total_string_bytes.map(usize::try_from).transpose(),
        ) else {
            return Ok(None);
        };
        inputs.push(MaterializedColumnUpperBoundInput {
            data_type,
            input_max_string_length_bytes,
            input_total_string_bytes,
        });
    }
    let preflight = match table_upper_bound_preflight(output_rows, &inputs) {
        Ok(preflight) => preflight,
        Err(error) if error.code() == cudf::ErrorCode::UnsupportedDtype => return Ok(None),
        Err(error) => return Err(materialization_preflight_error(error)),
    };
    Ok(u64::try_from(preflight.checked_peak_bytes()).ok())
}

fn materialization_preflight_error(error: cudf::Error) -> QueryError {
    let reason = error.detail_value("reason").map(str::to_owned);
    if reason.as_deref() != Some("string_payload_evidence_missing") {
        return cudf_query_error(
            "query_engine.bounded.join.sizing",
            "cuDF materialized-table size preflight rejected the output contract",
            error,
        );
    }
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.bounded.join.sizing",
        "cuDF materialized-table size preflight rejected the output contract",
    )
    .with_fact("component_owner_operation", error.operation())
    .with_fact("reason", "string_payload_evidence_missing")
    .with_source(error)
}

/// The bytes a partial aggregate of `shape` can retain, or `None` when the
/// byte model cannot bound it.
pub fn partial_aggregate_upper_bound_bytes(
    shape: PartialAggregateShape,
    columns: &[JoinOutputColumnUpperBoundInput],
) -> Result<Option<u64>> {
    join_output_upper_bound_bytes(shape.output_row_count_upper_bound(), columns)
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
        .unwrap()
        .expect("a string column with payload evidence has a bound");
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
