use arrow::datatypes::DataType as ArrowDataType;

use crate::materialization_sizing::{
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, PartialAggregateShape,
    STRING_OFFSET_BYTES_CONSERVATIVE, partial_aggregate_upper_bound_bytes,
};

fn fixed_column(data_type: ArrowDataType) -> JoinOutputColumnUpperBoundInput {
    JoinOutputColumnUpperBoundInput { data_type, input_max_string_length_bytes: None, input_total_string_bytes: None }
}

fn string_column(max_len: u64) -> JoinOutputColumnUpperBoundInput {
    JoinOutputColumnUpperBoundInput {
        data_type: ArrowDataType::Utf8,
        input_max_string_length_bytes: Some(max_len),
        input_total_string_bytes: None,
    }
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
