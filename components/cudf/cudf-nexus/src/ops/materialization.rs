//! Allocation-free materialized-table size preflight.

use crate::{
    DataType, Error, ErrorCode, Result, TypeId,
    core::error::{FACT_COLUMN_INDEX, FACT_TYPE_ID},
};

pub const STRING_OFFSET_BYTES_CONSERVATIVE: usize = 8;
pub const PER_COLUMN_ALIGNMENT_PADDING_BYTES: usize = 256;

const NULL_MASK_BITS_PER_BYTE: usize = 8;
const NULL_MASK_ALIGNMENT_BYTES: usize = 64;
pub const TABLE_UPPER_BOUND_PREFLIGHT_OPERATION: &str = "materialization::table_upper_bound_preflight";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedColumnUpperBoundInput {
    pub data_type: DataType,
    pub input_max_string_length_bytes: Option<usize>,
    pub input_total_string_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedTableUpperBoundPreflight {
    column_bytes: Vec<usize>,
    checked_peak_bytes: usize,
}

impl MaterializedTableUpperBoundPreflight {
    pub fn column_bytes(&self) -> &[usize] {
        &self.column_bytes
    }

    pub fn checked_peak_bytes(&self) -> usize {
        self.checked_peak_bytes
    }

    pub const fn component_owner_operation(&self) -> &'static str {
        TABLE_UPPER_BOUND_PREFLIGHT_OPERATION
    }
}

pub fn table_upper_bound_preflight(
    output_rows: usize,
    columns: &[MaterializedColumnUpperBoundInput],
) -> Result<MaterializedTableUpperBoundPreflight> {
    let column_bytes = columns
        .iter()
        .enumerate()
        .map(|(column_index, column)| {
            column_upper_bound_bytes(output_rows, column)
                .map_err(|error| error.with_fact(FACT_COLUMN_INDEX, column_index))
        })
        .collect::<Result<Vec<_>>>()?;
    let checked_peak_bytes = column_bytes.iter().try_fold(0_usize, |total, bytes| {
        total.checked_add(*bytes).ok_or_else(|| size_overflow("table_bytes", "column_sum_overflow"))
    })?;
    Ok(MaterializedTableUpperBoundPreflight { column_bytes, checked_peak_bytes })
}

fn column_upper_bound_bytes(output_rows: usize, column: &MaterializedColumnUpperBoundInput) -> Result<usize> {
    let data_bytes = if column.data_type.id() == TypeId::String {
        let payload = match (column.input_max_string_length_bytes, column.input_total_string_bytes) {
            (Some(max), Some(total)) => checked_mul(output_rows, max, "string_payload")?.min(total),
            (Some(max), None) => checked_mul(output_rows, max, "string_payload")?,
            (None, Some(total)) => total,
            (None, None) => {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    TABLE_UPPER_BOUND_PREFLIGHT_OPERATION,
                    "string materialization requires maximum-value or total-payload evidence",
                )
                .with_detail("reason", "string_payload_evidence_missing"));
            },
        };
        let offsets = checked_mul(
            output_rows.checked_add(1).ok_or_else(|| size_overflow("string_offsets", "row_count_overflow"))?,
            STRING_OFFSET_BYTES_CONSERVATIVE,
            "string_offsets",
        )?;
        checked_add(payload, offsets, "string_column")?
    } else {
        let width = column.data_type.id().size_in_bytes()?.ok_or_else(|| {
            Error::new_with(
                ErrorCode::UnsupportedDtype,
                TABLE_UPPER_BOUND_PREFLIGHT_OPERATION,
                "materialized output dtype is not a supported flat column",
            )
            .with_detail("reason", "unsupported_materialized_output_dtype")
            .with_fact(FACT_TYPE_ID, column.data_type.id())
        })?;
        let width = usize::try_from(width).map_err(|_| size_overflow("fixed_width_data", "negative_type_width"))?;
        checked_mul(output_rows, width, "fixed_width_data")?
    };
    checked_add(
        checked_add(data_bytes, null_mask_bytes(output_rows)?, "column_null_mask")?,
        PER_COLUMN_ALIGNMENT_PADDING_BYTES,
        "column_alignment",
    )
}

fn null_mask_bytes(rows: usize) -> Result<usize> {
    if rows == 0 {
        return Ok(0);
    }
    let bytes = rows
        .checked_add(NULL_MASK_BITS_PER_BYTE - 1)
        .ok_or_else(|| size_overflow("null_mask", "row_count_overflow"))?
        / NULL_MASK_BITS_PER_BYTE;
    let blocks = bytes
        .checked_add(NULL_MASK_ALIGNMENT_BYTES - 1)
        .ok_or_else(|| size_overflow("null_mask", "alignment_overflow"))?
        / NULL_MASK_ALIGNMENT_BYTES;
    checked_mul(blocks, NULL_MASK_ALIGNMENT_BYTES, "null_mask")
}

fn checked_add(lhs: usize, rhs: usize, resource: &'static str) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| size_overflow(resource, "addition_overflow"))
}

fn checked_mul(lhs: usize, rhs: usize, resource: &'static str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| size_overflow(resource, "multiplication_overflow"))
}

fn size_overflow(resource: &'static str, reason: &'static str) -> Error {
    Error::new_with(
        ErrorCode::SizeOverflow,
        TABLE_UPPER_BOUND_PREFLIGHT_OPERATION,
        "materialized table size preflight overflowed usize",
    )
    .with_detail("resource", resource)
    .with_detail("reason", reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_and_string_components_form_checked_peak() {
        let preflight = table_upper_bound_preflight(
            513,
            &[
                MaterializedColumnUpperBoundInput {
                    data_type: DataType::new(TypeId::Int64),
                    input_max_string_length_bytes: None,
                    input_total_string_bytes: None,
                },
                MaterializedColumnUpperBoundInput {
                    data_type: DataType::new(TypeId::String),
                    input_max_string_length_bytes: Some(7),
                    input_total_string_bytes: None,
                },
            ],
        )
        .unwrap();

        assert_eq!(preflight.column_bytes(), &[513 * 8 + 128 + 256, 513 * 7 + 514 * 8 + 128 + 256]);
        assert_eq!(preflight.checked_peak_bytes(), preflight.column_bytes().iter().sum::<usize>());
        assert_eq!(preflight.component_owner_operation(), TABLE_UPPER_BOUND_PREFLIGHT_OPERATION);
    }

    #[test]
    fn string_payload_must_be_proved() {
        let error = table_upper_bound_preflight(
            1,
            &[MaterializedColumnUpperBoundInput {
                data_type: DataType::new(TypeId::String),
                input_max_string_length_bytes: None,
                input_total_string_bytes: None,
            }],
        )
        .unwrap_err();

        assert_eq!(error.operation(), TABLE_UPPER_BOUND_PREFLIGHT_OPERATION);
        assert_eq!(error.detail_value("reason"), Some("string_payload_evidence_missing"));
    }
}
