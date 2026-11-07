use arrow::datatypes::DataType as ArrowDataType;
use cudf::{
    interop::arrow_host::cudf_dtype_from_arrow_type,
    ops::materialization::{
        MaterializedColumnUpperBoundInput,
        PER_COLUMN_ALIGNMENT_PADDING_BYTES as CUDF_PER_COLUMN_ALIGNMENT_PADDING_BYTES,
        STRING_OFFSET_BYTES_CONSERVATIVE as CUDF_STRING_OFFSET_BYTES_CONSERVATIVE, table_upper_bound_preflight,
    },
};
use query_planning::error::{QueryError, QueryErrorCode, Result};
use query_runtime::native_error::cudf_query_error;

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
    use arrow::datatypes::{Field, FieldRef, Fields, TimeUnit};

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
    fn fixed_column(data_type: ArrowDataType) -> JoinOutputColumnUpperBoundInput {
        JoinOutputColumnUpperBoundInput {
            data_type,
            input_max_string_length_bytes: None,
            input_total_string_bytes: None,
        }
    }

    fn string_column(max_len: u64) -> JoinOutputColumnUpperBoundInput {
        JoinOutputColumnUpperBoundInput {
            data_type: ArrowDataType::Utf8,
            input_max_string_length_bytes: Some(max_len),
            input_total_string_bytes: None,
        }
    }

    fn expected_null_mask_bytes(rows: u64) -> u64 {
        if rows == 0 {
            return 0;
        }
        let bits = rows.div_ceil(8);
        let blocks = bits.div_ceil(64);
        blocks * 64
    }

    fn fixed_column_expected(rows: u64, dtype_size: u64) -> u64 {
        rows * dtype_size + expected_null_mask_bytes(rows) + PER_COLUMN_ALIGNMENT_PADDING_BYTES
    }

    fn string_column_expected(rows: u64, max_len: u64) -> u64 {
        let data = rows * max_len;
        let offsets = (rows + 1) * STRING_OFFSET_BYTES_CONSERVATIVE;
        data + offsets + expected_null_mask_bytes(rows) + PER_COLUMN_ALIGNMENT_PADDING_BYTES
    }

    #[test]
    fn empty_projection_returns_zero() {
        let bound = join_output_upper_bound_bytes(1_000, &[]).unwrap().expect("the byte model bounds this output");
        assert_eq!(bound, 0);
    }

    #[test]
    fn fixed_width_column_bounds_cover_supported_physical_types() {
        let mut types = vec![
            (ArrowDataType::Boolean, 1),
            (ArrowDataType::Int8, 1),
            (ArrowDataType::UInt8, 1),
            (ArrowDataType::Int16, 2),
            (ArrowDataType::UInt16, 2),
            (ArrowDataType::Int32, 4),
            (ArrowDataType::Int64, 8),
            (ArrowDataType::UInt64, 8),
            (ArrowDataType::Float64, 8),
            (ArrowDataType::Date64, 8),
            (ArrowDataType::Time64(TimeUnit::Microsecond), 8),
            (ArrowDataType::Time64(TimeUnit::Nanosecond), 8),
            (ArrowDataType::Decimal32(9, 2), 4),
            (ArrowDataType::Decimal64(18, 6), 8),
            (ArrowDataType::Decimal128(38, 6), 16),
        ];
        for unit in [TimeUnit::Second, TimeUnit::Millisecond, TimeUnit::Microsecond, TimeUnit::Nanosecond] {
            types.push((ArrowDataType::Timestamp(unit, None), 8));
            types.push((ArrowDataType::Duration(unit), 8));
        }
        for rows in [64, 100, 256, 1_024, 10_000, 1_000_000] {
            for (dtype, width) in &types {
                let bound = join_output_upper_bound_bytes(rows, &[fixed_column(dtype.clone())])
                    .unwrap()
                    .expect("supported fixed-width output has a byte bound");
                assert_eq!(bound, fixed_column_expected(rows, *width), "dtype {dtype:?}, rows {rows}");
            }
        }
    }

    #[test]
    fn date32_uses_four_bytes_and_time32_uses_the_cudf_duration_width() {
        let rows = 10_000;
        let date = join_output_upper_bound_bytes(rows, &[fixed_column(ArrowDataType::Date32)])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(date, fixed_column_expected(rows, 4));
        for dtype in [ArrowDataType::Time32(TimeUnit::Second), ArrowDataType::Time32(TimeUnit::Millisecond)] {
            let bound = join_output_upper_bound_bytes(rows, &[fixed_column(dtype.clone())])
                .unwrap()
                .expect("the byte model bounds this output");
            assert_eq!(bound, fixed_column_expected(rows, 8), "dtype {dtype:?}");
        }
    }

    #[test]
    fn timestamp_with_timezone_has_no_bound() {
        let column = fixed_column(ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())));
        assert_eq!(join_output_upper_bound_bytes(64, &[column]).unwrap(), None);
    }

    #[test]
    fn string_column_bound_uses_max_length_times_rows_plus_offsets() {
        // High-fanout many-to-one gather: 1000 output rows over an input
        // where the longest string is 1000 bytes. The safe upper bound
        // tracks the worst-case duplication (codex v1 counterexample).
        let rows = 1_000;
        let max_len = 1_000;
        let bound = join_output_upper_bound_bytes(rows, &[string_column(max_len)])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, string_column_expected(rows, max_len));
        // Sanity: bound must be ≥ the worst-case data bytes.
        assert!(bound >= rows * max_len);
    }

    #[test]
    fn large_utf8_and_utf8view_match_utf8_bound() {
        let rows = 500;
        let max_len = 64;
        let baseline = join_output_upper_bound_bytes(rows, &[string_column(max_len)])
            .unwrap()
            .expect("the byte model bounds this output");
        for dtype in [ArrowDataType::LargeUtf8, ArrowDataType::Utf8View] {
            let column = JoinOutputColumnUpperBoundInput {
                data_type: dtype.clone(),
                input_max_string_length_bytes: Some(max_len),
                input_total_string_bytes: None,
            };
            let bound =
                join_output_upper_bound_bytes(rows, &[column]).unwrap().expect("the byte model bounds this output");
            assert_eq!(bound, baseline, "dtype {dtype:?}");
        }
    }

    #[test]
    fn string_column_without_max_length_returns_invalid_plan() {
        let column = JoinOutputColumnUpperBoundInput {
            data_type: ArrowDataType::Utf8,
            input_max_string_length_bytes: None,
            input_total_string_bytes: None,
        };
        let error = join_output_upper_bound_bytes(10, &[column]).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    }

    #[test]
    fn dtypes_without_a_byte_model_have_no_bound() {
        for dtype in [
            ArrowDataType::Binary,
            ArrowDataType::LargeBinary,
            ArrowDataType::FixedSizeBinary(32),
            ArrowDataType::Float16,
            ArrowDataType::List(FieldRef::new(Field::new("item", ArrowDataType::Int32, true))),
            ArrowDataType::Struct(Fields::empty()),
        ] {
            assert_eq!(
                join_output_upper_bound_bytes(10, &[fixed_column(dtype.clone())]).unwrap(),
                None,
                "dtype {dtype:?}"
            );
        }
    }

    #[test]
    fn zero_rows_with_fixed_width_column_returns_alignment_padding_only() {
        let bound = join_output_upper_bound_bytes(0, &[fixed_column(ArrowDataType::Int64)])
            .unwrap()
            .expect("the byte model bounds this output");
        // 0 data bytes + 0 null mask + alignment padding.
        assert_eq!(bound, PER_COLUMN_ALIGNMENT_PADDING_BYTES);
    }

    #[test]
    fn zero_rows_with_string_column_includes_one_offset_entry() {
        let bound =
            join_output_upper_bound_bytes(0, &[string_column(64)]).unwrap().expect("the byte model bounds this output");
        let expected = STRING_OFFSET_BYTES_CONSERVATIVE + PER_COLUMN_ALIGNMENT_PADDING_BYTES;
        assert_eq!(bound, expected);
    }

    #[test]
    fn multi_column_bound_is_sum_of_per_column_bounds() {
        let rows = 10_000;
        let bound = join_output_upper_bound_bytes(
            rows,
            &[fixed_column(ArrowDataType::Int32), fixed_column(ArrowDataType::Float64), string_column(128)],
        )
        .unwrap()
        .expect("the byte model bounds this output");
        let expected =
            fixed_column_expected(rows, 4) + fixed_column_expected(rows, 8) + string_column_expected(rows, 128);
        assert_eq!(bound, expected);
    }

    #[test]
    fn q72_n15_shape_bound_matches_observed_order_of_magnitude() {
        // Approximation of the q72 n15 partition's projection from the
        // research note: ~73 bytes/row at ~815M rows ≈ 58 GB observed.
        // Build a synthetic 15-column fixed-width projection of 108 data bytes per
        // row (Int32, Int64, Decimal128, Date32 and Float64 columns) and verify the
        // closed-form bound is on the right order of magnitude.
        let rows = 814_891_956u64;
        let projection = [
            fixed_column(ArrowDataType::Int32),
            fixed_column(ArrowDataType::Int32),
            fixed_column(ArrowDataType::Int32),
            fixed_column(ArrowDataType::Int32),
            fixed_column(ArrowDataType::Int64),
            fixed_column(ArrowDataType::Int64),
            fixed_column(ArrowDataType::Decimal128(18, 2)),
            fixed_column(ArrowDataType::Decimal128(18, 2)),
            fixed_column(ArrowDataType::Date32),
            fixed_column(ArrowDataType::Date32),
            fixed_column(ArrowDataType::Int32),
            fixed_column(ArrowDataType::Int64),
            fixed_column(ArrowDataType::Int64),
            fixed_column(ArrowDataType::Float64),
            fixed_column(ArrowDataType::Float64),
        ];
        let bound =
            join_output_upper_bound_bytes(rows, &projection).unwrap().expect("the byte model bounds this output");
        let observed = 58_268_236_464u64; // partition 1 observed reservation bytes
        // Bound must be ≥ observed (correctness).
        assert!(bound >= observed, "bound {bound} below observed {observed}");
        // Bound must not be wildly larger (sanity; within 4× is the
        // expected over-reservation for a synthetic projection).
        assert!(bound <= observed * 4, "bound {bound} more than 4x observed {observed}");
    }

    #[test]
    fn global_partial_with_avg_state_uses_sum_plus_count_columns() {
        // AVG expands to partial SUM + COUNT state.
        // For Decimal128 input → sum is Decimal128 (16 bytes), count is Int64 (8 bytes).
        let bound = partial_aggregate_upper_bound_bytes(
            PartialAggregateShape::Global,
            &[fixed_column(ArrowDataType::Decimal128(18, 2)), fixed_column(ArrowDataType::Int64)],
        )
        .unwrap()
        .expect("the byte model bounds this output");
        // Two columns × (data + null mask 0-bytes-when-1-row + 256-byte alignment padding).
        // null_mask_bytes(1) = ceil(1/8)=1, rounded up to 64-byte alignment → 64.
        let null_mask_bytes_one_row = 64;
        let expected_sum_col = 16 + null_mask_bytes_one_row + PER_COLUMN_ALIGNMENT_PADDING_BYTES;
        let expected_count_col = 8 + null_mask_bytes_one_row + PER_COLUMN_ALIGNMENT_PADDING_BYTES;
        assert_eq!(bound, expected_sum_col + expected_count_col);
    }

    #[test]
    fn grouped_partial_with_string_keys_uses_max_string_length() {
        // q72-shape: GROUP BY two strings (`i_item_desc`, `w_warehouse_name`) plus
        // an Int32 week_seq, with a single COUNT aggregate state column.
        // Conservative grouped row count = input_rows (every row could be a unique
        // group). String key bound = input_rows × max_string_length.
        let input_rows = 1_000_000;
        let columns = [
            string_column(200), // i_item_desc
            string_column(20),  // w_warehouse_name
            fixed_column(ArrowDataType::Int32),
            fixed_column(ArrowDataType::Int64), // partial COUNT state
        ];
        let bound = partial_aggregate_upper_bound_bytes(PartialAggregateShape::Grouped { input_rows }, &columns)
            .unwrap()
            .expect("the byte model bounds this output");
        // Sanity: bound must be ≥ worst-case data bytes for the two string keys
        // (1M rows × (200 + 20) chars) without overflowing u64.
        let data_only_strings = input_rows * (200 + 20);
        assert!(bound >= data_only_strings);
        // Sanity: bound must be ≥ Int32 + Int64 fixed-width data bytes.
        let data_only_fixed = input_rows * (4 + 8);
        assert!(bound >= data_only_fixed);
    }

    #[test]
    fn grouped_partial_with_zero_input_rows_returns_offset_only_bound_for_strings() {
        // Grouped aggregate over an empty input: partial output has 0 rows. The
        // bound must still admit one offset entry per string column (cuDF's
        // empty string column carries an offsets buffer).
        let bound = partial_aggregate_upper_bound_bytes(
            PartialAggregateShape::Grouped { input_rows: 0 },
            &[string_column(64), fixed_column(ArrowDataType::Int64)],
        )
        .unwrap()
        .expect("the byte model bounds this output");
        // String: 0 data + 1 offset × 8 bytes + 0 null mask + 256 alignment.
        // Int64: 0 data + 0 null mask + 256 alignment.
        let expected_string = STRING_OFFSET_BYTES_CONSERVATIVE + PER_COLUMN_ALIGNMENT_PADDING_BYTES;
        let expected_int = PER_COLUMN_ALIGNMENT_PADDING_BYTES;
        assert_eq!(bound, expected_string + expected_int);
    }
}
