// Closed-form per-column upper bound for the bytes a bounded join's
// `materialize_output` call can produce on the device. The bound is
// computed without invoking libcudf; it is used by F2 (closed-form
// reservation pattern) to size the work reservation before the gather
// runs. After the materialize completes, the caller observes the actual
// `device_allocated_bytes` of the output and reconciles via
// `shrink_to_observed`; strict-bounded mode hard-fails if observed
// exceeds the upper bound.
//
// The bound is mathematically valid under cuDF's `NULLIFY` bounds
// policy: null-extended output rows (e.g. LEFT OUTER unmatched, FULL
// OUTER unmatched on either side) contribute zero data bytes, so the
// observed bytes are always ≤ the per-column upper bound. For
// duplication-capable gathers (many-to-one inner joins) the bound is
// `expected_output_rows × max_input_string_length`, which is tight for
// maximum duplication.

use arrow::datatypes::DataType as ArrowDataType;

use crate::error::{QueryError, QueryErrorCode, Result};

/// Conservative offsets-buffer width for cuDF `String` columns. cuDF
/// starts with INT32 offsets and switches to INT64 above
/// `cudf::strings::detail::get_offset64_threshold()`
/// (`../cudf-rs/cudf/cpp/include/cudf/strings/detail/strings_children.cuh:133-157`);
/// using 8 bytes per offset upper-bounds both regimes.
pub const STRING_OFFSET_BYTES_CONSERVATIVE: u64 = 8;

/// Conservative per-buffer-pair allocation padding used to upper-bound
/// RMM's allocation alignment when materializing a join output column.
pub const PER_COLUMN_ALIGNMENT_PADDING_BYTES: u64 = 256;

/// Description of one output column for the upper-bound calculation.
#[derive(Debug, Clone)]
pub struct JoinOutputColumnUpperBoundInput {
    /// Arrow type of the materialized output column.
    pub data_type: ArrowDataType,
    /// Required for string-typed columns (`Utf8`, `LargeUtf8`,
    /// `Utf8View`); ignored otherwise. The maximum byte length over
    /// input rows in the source column that flows into this output
    /// column. For LEFT/FULL OUTER joins under the NULLIFY bounds
    /// policy, null-extended output rows contribute zero data bytes;
    /// the bound stays valid because non-null output rows ≤
    /// `expected_output_rows`.
    pub input_max_string_length_bytes: Option<u64>,
    /// Optional tighter cap for operators whose string output values are a
    /// subset of one input column's values. Joins leave this unset because
    /// gathers can duplicate string payloads; grouped aggregate partials can
    /// set it because each output group/value is selected from disjoint input
    /// rows.
    pub input_total_string_bytes: Option<u64>,
}

/// Compute the closed-form upper bound (in bytes) for the device
/// allocation of a join output projection, given the expected output
/// row count and per-column descriptors.
///
/// Returns `Ok(0)` when `columns.is_empty()`. Returns
/// `QueryErrorCode::UnsupportedPlanShape` for any column whose Arrow
/// dtype is not supported by the strict-bounded materialize path; the
/// caller's admission gate is expected to reject these plans before
/// the materialize is reached, so this function as a defense-in-depth.
pub fn join_output_upper_bound_bytes(
    expected_output_rows: u64,
    columns: &[JoinOutputColumnUpperBoundInput],
) -> Result<u64> {
    if columns.is_empty() {
        return Ok(0);
    }
    let mut total = 0u64;
    for column in columns {
        total = total.saturating_add(column_upper_bound_bytes(expected_output_rows, column)?);
    }
    Ok(total)
}

fn column_upper_bound_bytes(expected_output_rows: u64, column: &JoinOutputColumnUpperBoundInput) -> Result<u64> {
    let data_bytes = match &column.data_type {
        ArrowDataType::Boolean | ArrowDataType::Int8 | ArrowDataType::UInt8 => expected_output_rows,
        ArrowDataType::Int16 | ArrowDataType::UInt16 => expected_output_rows.saturating_mul(2),
        ArrowDataType::Int32 | ArrowDataType::UInt32 | ArrowDataType::Float32 | ArrowDataType::Date32 => {
            expected_output_rows.saturating_mul(4)
        },
        ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64 => expected_output_rows.saturating_mul(8),
        ArrowDataType::Timestamp(_, None) => expected_output_rows.saturating_mul(8),
        ArrowDataType::Duration(_) => expected_output_rows.saturating_mul(8),
        ArrowDataType::Decimal128(_, _) => expected_output_rows.saturating_mul(16),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            let max_len = column.input_max_string_length_bytes.ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.bounded.join.sizing",
                    "string column upper bound requires input_max_string_length_bytes",
                )
                .with_fact("dtype", format!("{:?}", column.data_type))
            })?;
            let data = expected_output_rows.saturating_mul(max_len);
            let data = column.input_total_string_bytes.map_or(data, |total| data.min(total));
            let offsets = expected_output_rows.saturating_add(1).saturating_mul(STRING_OFFSET_BYTES_CONSERVATIVE);
            data.saturating_add(offsets)
        },
        _ => return Err(unsupported_dtype(column)),
    };
    let null_mask = null_mask_bytes(expected_output_rows);
    Ok(data_bytes.saturating_add(null_mask).saturating_add(PER_COLUMN_ALIGNMENT_PADDING_BYTES))
}

fn null_mask_bytes(rows: u64) -> u64 {
    if rows == 0 {
        return 0;
    }
    let bits = rows.saturating_add(7) / 8;
    let blocks = bits.saturating_add(63) / 64;
    blocks.saturating_mul(64)
}

fn unsupported_dtype(column: &JoinOutputColumnUpperBoundInput) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "cudf_query_engine.bounded.join.sizing",
        "join output column dtype is not supported by strict-bounded materialize upper bound",
    )
    .with_fact("dtype", format!("{:?}", column.data_type))
    .with_fact("reason", "unsupported_join_output_dtype")
}
