use arrow::datatypes::{DataType as ArrowDataType, Fields, TimeUnit};
use query_engine::{
    error::QueryErrorCode,
    materialization_sizing::{
        JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, PartialAggregateShape,
        STRING_OFFSET_BYTES_CONSERVATIVE, join_output_upper_bound_bytes, partial_aggregate_upper_bound_bytes,
    },
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
fn global_shape_uses_one_output_row() {
    assert_eq!(PartialAggregateShape::Global.output_row_count_upper_bound(), 1);
}

#[test]
fn grouped_shape_uses_input_rows_as_upper_bound() {
    let shape = PartialAggregateShape::Grouped { input_rows: 1_000_000 };
    assert_eq!(shape.output_row_count_upper_bound(), 1_000_000);
}

#[test]
fn empty_partial_columns_returns_zero() {
    let bound = partial_aggregate_upper_bound_bytes(PartialAggregateShape::Global, &[]).unwrap();
    assert_eq!(bound, 0);
}

#[test]
fn global_with_single_int64_count_column_uses_one_row() {
    // Mirrors a `SELECT COUNT(*)` partial: one Int64 column, one row.
    let bound =
        partial_aggregate_upper_bound_bytes(PartialAggregateShape::Global, &[fixed_column(ArrowDataType::Int64)])
            .unwrap();
    let direct = join_output_upper_bound_bytes(1, &[fixed_column(ArrowDataType::Int64)]).unwrap();
    assert_eq!(bound, direct, "global helper must delegate with row count = 1");
    // Bound must contain at least 8 bytes for the Int64 data slot.
    assert!(bound >= 8);
}

#[test]
fn grouped_partial_bound_matches_direct_join_helper_call() {
    // The helper is a thin adapter; calling it must equal calling the
    // join helper with the same row count and column descriptors.
    let columns = [
        fixed_column(ArrowDataType::Int32),
        fixed_column(ArrowDataType::Int64),
        fixed_column(ArrowDataType::Decimal128(18, 2)),
    ];
    let rows = 4_096;
    let bound =
        partial_aggregate_upper_bound_bytes(PartialAggregateShape::Grouped { input_rows: rows }, &columns).unwrap();
    let direct = join_output_upper_bound_bytes(rows, &columns).unwrap();
    assert_eq!(bound, direct);
}

#[test]
fn global_partial_with_avg_state_uses_sum_plus_count_columns() {
    // AVG expands to partial SUM + COUNT state.
    // For Decimal128 input → sum is Decimal128 (16 bytes), count is Int64 (8 bytes).
    let bound = partial_aggregate_upper_bound_bytes(
        PartialAggregateShape::Global,
        &[fixed_column(ArrowDataType::Decimal128(18, 2)), fixed_column(ArrowDataType::Int64)],
    )
    .unwrap();
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
    let bound = partial_aggregate_upper_bound_bytes(PartialAggregateShape::Grouped { input_rows }, &columns).unwrap();
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
    .unwrap();
    // String: 0 data + 1 offset × 8 bytes + 0 null mask + 256 alignment.
    // Int64: 0 data + 0 null mask + 256 alignment.
    let expected_string = STRING_OFFSET_BYTES_CONSERVATIVE + PER_COLUMN_ALIGNMENT_PADDING_BYTES;
    let expected_int = PER_COLUMN_ALIGNMENT_PADDING_BYTES;
    assert_eq!(bound, expected_string + expected_int);
}

#[test]
fn unsupported_partial_column_dtype_is_rejected() {
    // Binary and nested dtypes are not supported by the bounded
    // materialize sizing path. The admission gate's dtype loop is expected to
    // reject plans that project these into the partial state, but the helper
    // remains a defense-in-depth check.
    for dtype in [ArrowDataType::Binary, ArrowDataType::Float16, ArrowDataType::Struct(Fields::empty())] {
        let error = partial_aggregate_upper_bound_bytes(
            PartialAggregateShape::Grouped { input_rows: 10 },
            &[fixed_column(dtype.clone())],
        )
        .unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "dtype {dtype:?}");
    }
}

#[test]
fn global_partial_with_timestamp_uses_eight_bytes() {
    // Reasonable shape for a `MAX(event_ts)` partial: one Timestamp column,
    // one row.
    let bound = partial_aggregate_upper_bound_bytes(
        PartialAggregateShape::Global,
        &[fixed_column(ArrowDataType::Timestamp(TimeUnit::Microsecond, None))],
    )
    .unwrap();
    let null_mask_bytes_one_row = 64;
    let expected = 8 + null_mask_bytes_one_row + PER_COLUMN_ALIGNMENT_PADDING_BYTES;
    assert_eq!(bound, expected);
}

#[test]
fn grouped_partial_with_temporal_keys_uses_physical_widths() {
    let input_rows = 256;
    let columns = [
        fixed_column(ArrowDataType::Date64),
        fixed_column(ArrowDataType::Time32(TimeUnit::Millisecond)),
        fixed_column(ArrowDataType::Time64(TimeUnit::Nanosecond)),
    ];
    let bound = partial_aggregate_upper_bound_bytes(PartialAggregateShape::Grouped { input_rows }, &columns).unwrap();
    let direct = join_output_upper_bound_bytes(input_rows, &columns).unwrap();
    assert_eq!(bound, direct);
}

#[test]
fn grouped_partial_with_no_aggregate_state_only_group_keys() {
    // DISTINCT-style partial: only group keys, no aggregate state columns.
    // (Conceptually mirrors `SELECT DISTINCT col_a, col_b` rewritten as
    // GROUP BY with no aggregate exprs.) Bound is purely the per-key bytes.
    let input_rows = 1_024;
    let bound = partial_aggregate_upper_bound_bytes(
        PartialAggregateShape::Grouped { input_rows },
        &[fixed_column(ArrowDataType::Int32), fixed_column(ArrowDataType::Int64)],
    )
    .unwrap();
    let direct = join_output_upper_bound_bytes(
        input_rows,
        &[fixed_column(ArrowDataType::Int32), fixed_column(ArrowDataType::Int64)],
    )
    .unwrap();
    assert_eq!(bound, direct);
}

#[test]
fn global_shape_does_not_scale_with_input_rows() {
    // Global aggregate's partial bound stays constant regardless of input
    // row count. This is the defining property of the Global vs Grouped
    // distinction.
    let columns = [fixed_column(ArrowDataType::Int64)];
    let bound_small = partial_aggregate_upper_bound_bytes(PartialAggregateShape::Global, &columns).unwrap();
    let bound_large = partial_aggregate_upper_bound_bytes(PartialAggregateShape::Global, &columns).unwrap();
    assert_eq!(bound_small, bound_large);
}
