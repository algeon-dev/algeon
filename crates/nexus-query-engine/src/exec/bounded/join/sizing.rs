// Closed-form per-column upper bound for the bytes a bounded join's
// `materialize_output` call can produce on the device. The bound is
// computed without invoking libcudf; it is used by F2 (closed-form
// reservation pattern) to size the work reservation before the gather
// runs. After the materialize completes, the caller observes the actual
// `device_allocated_bytes` of the output and reconciles via
// `shrink_to_observed`; bounded mode hard-fails if observed
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

/// Internal policy for closed-form join and aggregate output sizing.
///
/// These values encode conservative cudf/RMM/Arrow allocation assumptions used
/// before the materialization operator can report observed device memory. They
/// are not user-facing knobs; they should move downstack when cudf can
/// expose authoritative sizing or allocation-alignment contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JoinOutputSizingPolicy {
    string_offset_bytes_conservative: u64,
    per_column_alignment_padding_bytes: u64,
    null_mask_bits_per_byte: u64,
    null_mask_alignment_bytes: u64,
}

impl JoinOutputSizingPolicy {
    /// Conservative offsets-buffer width for cuDF `String` columns. cuDF
    /// starts with INT32 offsets and switches to INT64 above
    /// `cudf::strings::detail::get_offset64_threshold()`
    /// (`components/cudf/cudf/cpp/include/cudf/strings/detail/strings_children.cuh:133-157`);
    /// using 8 bytes per offset upper-bounds both regimes.
    const DEFAULT_STRING_OFFSET_BYTES_CONSERVATIVE: u64 = 8;
    /// Conservative per-buffer-pair allocation padding used to upper-bound
    /// RMM's allocation alignment when materializing a join output column.
    const DEFAULT_PER_COLUMN_ALIGNMENT_PADDING_BYTES: u64 = 256;
    const DEFAULT_NULL_MASK_BITS_PER_BYTE: u64 = 8;
    const DEFAULT_NULL_MASK_ALIGNMENT_BYTES: u64 = 64;

    fn string_offset_bytes_conservative(self) -> u64 {
        self.string_offset_bytes_conservative
    }

    fn per_column_alignment_padding_bytes(self) -> u64 {
        self.per_column_alignment_padding_bytes
    }

    fn null_mask_bytes(self, rows: u64) -> u64 {
        if rows == 0 {
            return 0;
        }
        let bits = rows.saturating_add(self.null_mask_bits_per_byte.saturating_sub(1)) / self.null_mask_bits_per_byte;
        let blocks =
            bits.saturating_add(self.null_mask_alignment_bytes.saturating_sub(1)) / self.null_mask_alignment_bytes;
        blocks.saturating_mul(self.null_mask_alignment_bytes)
    }
}

impl Default for JoinOutputSizingPolicy {
    fn default() -> Self {
        Self {
            string_offset_bytes_conservative: Self::DEFAULT_STRING_OFFSET_BYTES_CONSERVATIVE,
            per_column_alignment_padding_bytes: Self::DEFAULT_PER_COLUMN_ALIGNMENT_PADDING_BYTES,
            null_mask_bits_per_byte: Self::DEFAULT_NULL_MASK_BITS_PER_BYTE,
            null_mask_alignment_bytes: Self::DEFAULT_NULL_MASK_ALIGNMENT_BYTES,
        }
    }
}

/// Conservative offsets-buffer width for cuDF `String` columns.
pub const STRING_OFFSET_BYTES_CONSERVATIVE: u64 = JoinOutputSizingPolicy::DEFAULT_STRING_OFFSET_BYTES_CONSERVATIVE;

/// Conservative per-buffer-pair allocation padding used to upper-bound
/// RMM's allocation alignment when materializing a join output column.
pub const PER_COLUMN_ALIGNMENT_PADDING_BYTES: u64 = JoinOutputSizingPolicy::DEFAULT_PER_COLUMN_ALIGNMENT_PADDING_BYTES;

/// Description of one output column for the upper-bound calculation.
#[derive(Debug, Clone)]
pub struct JoinOutputColumnUpperBoundInput {
    /// Arrow type of the materialized output column.
    pub data_type: ArrowDataType,
    /// Optional for string-typed columns (`Utf8`, `LargeUtf8`,
    /// `Utf8View`); ignored otherwise. The maximum byte length over
    /// input rows in the source column that flows into this output
    /// column. For LEFT/FULL OUTER joins under the NULLIFY bounds
    /// policy, null-extended output rows contribute zero data bytes;
    /// the bound stays valid because non-null output rows ≤
    /// `expected_output_rows`.
    pub input_max_string_length_bytes: Option<u64>,
    /// Optional cap for operators whose string output values are a
    /// subset of one input column's values. Generic joins leave this unset
    /// because gathers can duplicate string payloads; operators with a proven
    /// multiplicity contract may set it.
    pub input_total_string_bytes: Option<u64>,
}

/// Compute the closed-form upper bound (in bytes) for the device
/// allocation of a join output projection, given the expected output
/// row count and per-column descriptors.
///
/// Returns `Ok(0)` when `columns.is_empty()`. Returns
/// `QueryErrorCode::UnsupportedPlanShape` for any column whose Arrow
/// dtype is not supported by the bounded materialize path; the
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
    let policy = JoinOutputSizingPolicy::default();
    let data_bytes = match &column.data_type {
        ArrowDataType::Boolean | ArrowDataType::Int8 | ArrowDataType::UInt8 => expected_output_rows,
        ArrowDataType::Int16 | ArrowDataType::UInt16 => expected_output_rows.saturating_mul(2),
        ArrowDataType::Int32
        | ArrowDataType::UInt32
        | ArrowDataType::Float32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_) => expected_output_rows.saturating_mul(4),
        ArrowDataType::Int64
        | ArrowDataType::UInt64
        | ArrowDataType::Float64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_) => expected_output_rows.saturating_mul(8),
        ArrowDataType::Timestamp(_, None) => expected_output_rows.saturating_mul(8),
        ArrowDataType::Duration(_) => expected_output_rows.saturating_mul(8),
        ArrowDataType::Decimal128(_, _) => expected_output_rows.saturating_mul(16),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            let data = match (column.input_max_string_length_bytes, column.input_total_string_bytes) {
                (Some(max_len), Some(total)) => expected_output_rows.saturating_mul(max_len).min(total),
                (Some(max_len), None) => expected_output_rows.saturating_mul(max_len),
                (None, Some(total)) => total,
                (None, None) => {
                    return Err(QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.bounded.join.sizing",
                        "string column upper bound requires maximum-value or total-payload evidence",
                    )
                    .with_fact("dtype", format!("{:?}", column.data_type)));
                },
            };
            let offsets =
                expected_output_rows.saturating_add(1).saturating_mul(policy.string_offset_bytes_conservative());
            data.saturating_add(offsets)
        },
        _ => return Err(unsupported_dtype(column)),
    };
    let null_mask = policy.null_mask_bytes(expected_output_rows);
    Ok(data_bytes.saturating_add(null_mask).saturating_add(policy.per_column_alignment_padding_bytes()))
}

fn unsupported_dtype(column: &JoinOutputColumnUpperBoundInput) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.bounded.join.sizing",
        "join output column dtype is not supported by bounded materialize upper bound",
    )
    .with_fact("dtype", format!("{:?}", column.data_type))
    .with_fact("reason", "unsupported_join_output_dtype")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_output_sizing_policy_preserves_public_aliases() {
        let policy = JoinOutputSizingPolicy::default();

        assert_eq!(policy.string_offset_bytes_conservative(), STRING_OFFSET_BYTES_CONSERVATIVE);
        assert_eq!(policy.per_column_alignment_padding_bytes(), PER_COLUMN_ALIGNMENT_PADDING_BYTES);
    }

    #[test]
    fn join_output_sizing_policy_preserves_null_mask_alignment() {
        let policy = JoinOutputSizingPolicy::default();

        assert_eq!(policy.null_mask_bytes(0), 0);
        assert_eq!(policy.null_mask_bytes(1), 64);
        assert_eq!(policy.null_mask_bytes(512), 64);
        assert_eq!(policy.null_mask_bytes(513), 128);
    }

    #[test]
    fn production_shaped_total_payload_is_not_multiplied_by_rows() {
        let rows = 109_914_910;
        let total_payload = 4_706_637_611;
        let policy = JoinOutputSizingPolicy::default();
        let bound = join_output_upper_bound_bytes(
            rows,
            &[JoinOutputColumnUpperBoundInput {
                data_type: ArrowDataType::Utf8,
                input_max_string_length_bytes: None,
                input_total_string_bytes: Some(total_payload),
            }],
        )
        .expect("a total-payload cap is sufficient for subset-preserving string materialization");
        let expected = total_payload
            + (rows + 1) * policy.string_offset_bytes_conservative()
            + policy.null_mask_bytes(rows)
            + policy.per_column_alignment_padding_bytes();

        assert_eq!(bound, expected);
    }
}
