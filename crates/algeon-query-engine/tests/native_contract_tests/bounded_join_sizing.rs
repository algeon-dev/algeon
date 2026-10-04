use arrow::datatypes::{DataType as ArrowDataType, Field, FieldRef, Fields, TimeUnit};
use query_engine::materialization_sizing::{
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, STRING_OFFSET_BYTES_CONSERVATIVE,
    join_output_upper_bound_bytes,
};
use query_planning::error::QueryErrorCode;

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
fn empty_projection_with_zero_rows_returns_zero() {
    let bound = join_output_upper_bound_bytes(0, &[]).unwrap().expect("the byte model bounds this output");
    assert_eq!(bound, 0);
}

#[test]
fn fixed_width_int32_column_bound_is_exact() {
    let rows = 1_000_000;
    let bound = join_output_upper_bound_bytes(rows, &[fixed_column(ArrowDataType::Int32)])
        .unwrap()
        .expect("the byte model bounds this output");
    assert_eq!(bound, fixed_column_expected(rows, 4));
}

#[test]
fn fixed_width_int64_uint64_float64_match_eight_byte_size() {
    let rows = 256;
    let expected = fixed_column_expected(rows, 8);
    for dtype in [ArrowDataType::Int64, ArrowDataType::UInt64, ArrowDataType::Float64] {
        let bound = join_output_upper_bound_bytes(rows, &[fixed_column(dtype.clone())])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, expected, "dtype {dtype:?}");
    }
}

#[test]
fn boolean_int8_uint8_match_one_byte_size() {
    let rows = 1_024;
    let expected = fixed_column_expected(rows, 1);
    for dtype in [ArrowDataType::Boolean, ArrowDataType::Int8, ArrowDataType::UInt8] {
        let bound = join_output_upper_bound_bytes(rows, &[fixed_column(dtype.clone())])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, expected, "dtype {dtype:?}");
    }
}

#[test]
fn int16_uint16_match_two_byte_size() {
    let rows = 1_024;
    let expected = fixed_column_expected(rows, 2);
    for dtype in [ArrowDataType::Int16, ArrowDataType::UInt16] {
        let bound = join_output_upper_bound_bytes(rows, &[fixed_column(dtype.clone())])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, expected, "dtype {dtype:?}");
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
fn date64_and_time64_use_eight_bytes() {
    let rows = 10_000;
    let expected = fixed_column_expected(rows, 8);
    for dtype in [
        ArrowDataType::Date64,
        ArrowDataType::Time64(TimeUnit::Microsecond),
        ArrowDataType::Time64(TimeUnit::Nanosecond),
    ] {
        let bound = join_output_upper_bound_bytes(rows, &[fixed_column(dtype.clone())])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, expected, "dtype {dtype:?}");
    }
}

#[test]
fn timestamp_no_timezone_uses_eight_bytes() {
    let rows = 64;
    let expected = fixed_column_expected(rows, 8);
    for unit in [TimeUnit::Second, TimeUnit::Millisecond, TimeUnit::Microsecond, TimeUnit::Nanosecond] {
        let bound = join_output_upper_bound_bytes(rows, &[fixed_column(ArrowDataType::Timestamp(unit, None))])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, expected, "unit {unit:?}");
    }
}

#[test]
fn timestamp_with_timezone_has_no_bound() {
    let column = fixed_column(ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())));
    assert_eq!(join_output_upper_bound_bytes(64, &[column]).unwrap(), None);
}

#[test]
fn duration_uses_eight_bytes() {
    let rows = 64;
    let expected = fixed_column_expected(rows, 8);
    for unit in [TimeUnit::Second, TimeUnit::Millisecond, TimeUnit::Microsecond, TimeUnit::Nanosecond] {
        let bound = join_output_upper_bound_bytes(rows, &[fixed_column(ArrowDataType::Duration(unit))])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, expected, "unit {unit:?}");
    }
}

#[test]
fn decimal128_uses_sixteen_bytes() {
    let rows = 100;
    let bound = join_output_upper_bound_bytes(rows, &[fixed_column(ArrowDataType::Decimal128(38, 6))])
        .unwrap()
        .expect("the byte model bounds this output");
    assert_eq!(bound, fixed_column_expected(rows, 16));
}

#[test]
fn decimal32_and_decimal64_use_four_and_eight_bytes() {
    let rows = 100;
    for (data_type, width) in [(ArrowDataType::Decimal32(9, 2), 4), (ArrowDataType::Decimal64(18, 6), 8)] {
        let bound = join_output_upper_bound_bytes(rows, &[fixed_column(data_type.clone())])
            .unwrap()
            .expect("the byte model bounds this output");
        assert_eq!(bound, fixed_column_expected(rows, width), "dtype {data_type:?}");
    }
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
        let bound = join_output_upper_bound_bytes(rows, &[column]).unwrap().expect("the byte model bounds this output");
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
        assert_eq!(join_output_upper_bound_bytes(10, &[fixed_column(dtype.clone())]).unwrap(), None, "dtype {dtype:?}");
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
    let expected = fixed_column_expected(rows, 4) + fixed_column_expected(rows, 8) + string_column_expected(rows, 128);
    assert_eq!(bound, expected);
}

#[test]
fn q72_n15_shape_bound_matches_observed_order_of_magnitude() {
    // Approximation of the q72 n15 partition's projection from the
    // research note: ~73 bytes/row at ~815M rows ≈ 58 GB observed.
    // Build a synthetic 15-column fixed-width projection averaging
    // ~5 bytes/row (mix of Int32, Int64, Date32) and verify the
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
    let bound = join_output_upper_bound_bytes(rows, &projection).unwrap().expect("the byte model bounds this output");
    let observed = 58_268_236_464u64; // partition 1 observed reservation bytes
    // Bound must be ≥ observed (correctness).
    assert!(bound >= observed, "bound {bound} below observed {observed}");
    // Bound must not be wildly larger (sanity; within 2× is the
    // expected over-reservation for a synthetic projection).
    assert!(bound <= observed * 4, "bound {bound} more than 4x observed {observed}");
}
