//! Integration tests for the expression layer.
//!
//! Tests cover: `col()` / `lit()`, arithmetic operators, comparisons,
//! unary ops (`is_null`, `is_not_null`, `cast`, `Not`), `.alias()`,
//! `DataFrame::with_columns`, null fill, cumulative, aggregation, string ops,
//! and conditional (`when`/`then`/`otherwise`).
//!
//! All tests require a GPU.

use std::sync::Arc;

use arrow::{
    array::{Decimal64Array, Decimal128Array},
    buffer::{NullBuffer, ScalarBuffer},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    DataFrame, DataType, ErrorCode, ErrorKind, Exec, ExecutionBudgetRequest, Runtime, ScalarValue, Series,
    StringSliceOptions, TimestampResolution, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::{
        binary::BinaryOperator,
        expr::{Expr, col, lit, lit_null, when},
    },
};
use rapids_interop::{ColumnLayout, DataType as InteropDataType, Placement};

use crate::common::{error_context_value, execution_budget_token, nullable_series, run_named_cases, series};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a simple 3-row DataFrame with columns "x" (i32) and "y" (i32).
fn xy_df() -> DataFrame {
    let x = series::<i32>("x", &[10, 20, 30]).unwrap();
    let y = series::<i32>("y", &[1, 2, 3]).unwrap();
    DataFrame::from_series([("x", x), ("y", y)]).unwrap()
}

fn df_with_series(name: &str, series: Series) -> DataFrame {
    DataFrame::from_series([(name, series)]).unwrap()
}

fn i32_df(name: &str, data: &[i32]) -> DataFrame {
    df_with_series(name, series::<i32>(name, data).unwrap())
}

fn nullable_i32_df(name: &str, data: &[i32], validity: &[bool]) -> DataFrame {
    df_with_series(name, nullable_series::<i32>(name, data, validity).unwrap())
}

fn f64_df(name: &str, data: &[f64]) -> DataFrame {
    df_with_series(name, series::<f64>(name, data).unwrap())
}

fn nullable_f64_df(name: &str, data: &[f64], validity: &[bool]) -> DataFrame {
    df_with_series(name, nullable_series::<f64>(name, data, validity).unwrap())
}

fn decimal128_df_from_arrow(name: &str, values: Vec<Option<i128>>, precision: u8, scale: i8) -> DataFrame {
    let amount = Decimal128Array::from(values).with_precision_and_scale(precision, scale).unwrap();
    let schema = Schema::new(vec![Field::new(name, ArrowDataType::Decimal128(precision, scale), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(amount)]).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap()
}

fn decimal128_pair_df_from_arrow(
    left_name: &str,
    left_values: Vec<Option<i128>>,
    right_name: &str,
    right_values: Vec<Option<i128>>,
    precision: u8,
    scale: i8,
) -> DataFrame {
    let left = Decimal128Array::from(left_values).with_precision_and_scale(precision, scale).unwrap();
    let right = Decimal128Array::from(right_values).with_precision_and_scale(precision, scale).unwrap();
    let schema = Schema::new(vec![
        Field::new(left_name, ArrowDataType::Decimal128(precision, scale), true),
        Field::new(right_name, ArrowDataType::Decimal128(precision, scale), true),
    ]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(left), Arc::new(right)]).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap()
}

/// A pair of `Decimal128` columns `a` and `b` built from `(raw, valid)` rows, so a null row keeps the raw value
/// given for it under the mask.
fn decimal128_pair_df_with_masked_raw_values(
    a: &[(i128, bool)],
    b: &[(i128, bool)],
    precision: u8,
    scale: i8,
) -> DataFrame {
    let column = |rows: &[(i128, bool)]| {
        let values = ScalarBuffer::from(rows.iter().map(|(raw, _)| *raw).collect::<Vec<_>>());
        let validity = NullBuffer::from(rows.iter().map(|(_, valid)| *valid).collect::<Vec<_>>());
        Arc::new(Decimal128Array::new(values, Some(validity)).with_precision_and_scale(precision, scale).unwrap())
    };
    let schema = Schema::new(vec![
        Field::new("a", ArrowDataType::Decimal128(precision, scale), true),
        Field::new("b", ArrowDataType::Decimal128(precision, scale), true),
    ]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![column(a), column(b)]).unwrap();
    DataFrame::from_record_batch(&Runtime::builder().build().unwrap(), &batch).unwrap()
}

/// The raw values of a `Decimal128` result, `None` in a null row.
fn decimal128_raw_options(result: Series) -> Vec<Option<i128>> {
    let name = "result";
    let batch = DataFrame::from_series([(name, result)]).unwrap().to_record_batch().unwrap();
    batch.column(0).as_any().downcast_ref::<Decimal128Array>().unwrap().iter().collect()
}

fn bool_df(name: &str, data: &[bool]) -> DataFrame {
    df_with_series(name, series::<bool>(name, data).unwrap())
}

fn str_df(name: &str, data: &[&str]) -> DataFrame {
    df_with_series(name, series(name, data).unwrap())
}

fn eval(expr: Expr, df: &DataFrame) -> Series {
    expr.eval(df).unwrap()
}

fn eval_i32(expr: Expr, df: &DataFrame) -> Vec<i32> {
    eval(expr, df).to_vec_strict::<i32>().unwrap()
}

fn eval_i16(expr: Expr, df: &DataFrame) -> Vec<i16> {
    eval(expr, df).to_vec_strict::<i16>().unwrap()
}

fn eval_f64(expr: Expr, df: &DataFrame) -> Vec<f64> {
    eval(expr, df).to_vec_strict::<f64>().unwrap()
}

fn eval_bool(expr: Expr, df: &DataFrame) -> Vec<bool> {
    eval(expr, df).to_vec_strict::<bool>().unwrap()
}

fn host_export_fixed_width_bytes(series: &Series) -> Vec<u8> {
    let export = series.try_prepare_host_column_export().unwrap();
    let values_id = match export.column().layout() {
        ColumnLayout::FixedWidth { values } => *values,
        other => panic!("expected fixed-width host export, got {other:?}"),
    };
    let binding =
        export.bindings().iter().find(|binding| binding.id() == values_id).expect("missing fixed-width values binding");

    assert_eq!(*binding.placement(), Placement::Host);

    // SAFETY: prepared host exports retain their backing host buffers, and the
    // binding lifetime is tied to the export value held in this function.
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr() as *const u8, binding.bytes().byte_len()).to_vec() }
}

fn host_export_dtype(series: &Series) -> InteropDataType {
    series.try_prepare_host_column_export().unwrap().column().data_type()
}

fn host_export_i32_values(series: &Series) -> Vec<i32> {
    host_export_fixed_width_bytes(series)
        .as_chunks::<{ std::mem::size_of::<i32>() }>()
        .0
        .iter()
        .map(|chunk| i32::from_ne_bytes(*chunk))
        .collect()
}

fn host_export_i64_values(series: &Series) -> Vec<i64> {
    host_export_fixed_width_bytes(series)
        .as_chunks::<{ std::mem::size_of::<i64>() }>()
        .0
        .iter()
        .map(|chunk| i64::from_ne_bytes(*chunk))
        .collect()
}

fn host_export_i128_values(series: &Series) -> Vec<i128> {
    host_export_fixed_width_bytes(series)
        .as_chunks::<{ std::mem::size_of::<i128>() }>()
        .0
        .iter()
        .map(|chunk| i128::from_ne_bytes(*chunk))
        .collect()
}

fn assert_arrow_decimal128_column(frame: &DataFrame, name: &str, expected_raw_values: &[i128], scale: i8) {
    let batch = frame.select([name]).unwrap().to_record_batch().unwrap();
    assert_eq!(*batch.schema().field(0).data_type(), ArrowDataType::Decimal128(38, scale));
    let values = batch.column(0).as_any().downcast_ref::<Decimal128Array>().unwrap();
    assert_eq!(values.precision(), 38);
    assert_eq!(values.scale(), scale);
    assert_eq!(values.values().as_ref(), expected_raw_values);
}

fn assert_output_name(expr: Expr, expected: Option<&str>) {
    assert_eq!(expr.output_name(), expected);
}

// ===========================================================================
// col / lit basics
// ===========================================================================

fn col_resolves_existing_column() {
    let df = xy_df();
    let result = eval(col("x"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

fn col_missing_column_errors() {
    let df = xy_df();
    let err = col("zzz").eval(&df).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

fn lit_i32_broadcasts() {
    let df = xy_df();
    let result = eval(lit(42i32).alias("c"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42]);
}

fn into_lit_and_into_expr_resolve_unambiguously_after_glob_import() {
    // `cudf::ops::expr` glob-exports both `IntoLit` and `IntoExpr`; a numeric
    // literal's `into_expr` must still resolve to a single method.
    use cudf::ops::expr::*;

    let df = xy_df();
    let result = eval(5i32.into_expr().alias("c"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![5, 5, 5]);
}

fn lit_f64_broadcasts() {
    let df = xy_df();
    let result = eval(lit(3.25f64).alias("pi"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.25, 3.25, 3.25]);
}

fn lit_string_broadcasts() {
    let df = xy_df();
    let result = eval(lit("hello").alias("s"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
}

fn lit_null_broadcasts_typed_nulls() {
    let df = xy_df();
    let result = eval(lit_null(TypeId::Int32).alias("missing"), &df);

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Int32);
    assert_eq!(result.null_count(), 3);
}

fn lit_decimal32_broadcasts_dtype_scale_and_raw_values() {
    let df = xy_df();
    let result = eval(lit(ScalarValue::Decimal32 { raw: 12_345, scale: -2 }).alias("amount"), &df);

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal32, -2).unwrap());
    assert_eq!(host_export_dtype(&result), InteropDataType::fixed_point(rapids_interop::TypeId::Decimal32, -2));
    assert_eq!(host_export_i32_values(&result), vec![12_345, 12_345, 12_345]);
}

fn lit_decimal64_broadcasts_dtype_scale_and_raw_values() {
    let df = xy_df();
    let result = eval(lit(ScalarValue::Decimal64 { raw: -98_765_432, scale: -4 }).alias("amount"), &df);

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -4).unwrap());
    assert_eq!(host_export_dtype(&result), InteropDataType::fixed_point(rapids_interop::TypeId::Decimal64, -4));
    assert_eq!(host_export_i64_values(&result), vec![-98_765_432, -98_765_432, -98_765_432]);
}

fn lit_decimal128_broadcasts_dtype_scale_and_raw_values() {
    let df = xy_df();
    let raw = 123_456_789_012_345_678_901i128;
    let result = eval(lit(ScalarValue::Decimal128 { raw, scale: -6 }).alias("amount"), &df);

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal128, -6).unwrap());
    assert_eq!(host_export_dtype(&result), InteropDataType::fixed_point(rapids_interop::TypeId::Decimal128, -6));
    assert_eq!(host_export_i128_values(&result), vec![raw, raw, raw]);
}

fn binary_expr_aligns_mixed_decimal_storage_before_arithmetic() {
    let discount = series("discount", &[5i64, 25])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(DataType::fixed_point(TypeId::Decimal64, -2).unwrap())
        .unwrap();
    let df = df_with_series("discount", discount);

    let result = eval((lit(ScalarValue::Decimal128 { raw: 1, scale: 0 }) - col("discount")).alias("net"), &df);

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal128, -2).unwrap());
    assert_eq!(host_export_i128_values(&result), vec![-400, -2400]);
}

fn lit_timestamp_broadcasts_dtype_and_raw_values() {
    let df = xy_df();
    let result = eval(
        lit(ScalarValue::Timestamp { raw: 1_700_000_123_456, resolution: TimestampResolution::Microseconds })
            .alias("ts"),
        &df,
    );

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap(), DataType::new(TypeId::TimestampMicroseconds));
    assert_eq!(host_export_dtype(&result), InteropDataType::new(rapids_interop::TypeId::TimestampMicroseconds));
    assert_eq!(host_export_i64_values(&result), vec![1_700_000_123_456; 3]);
}

fn lit_duration_broadcasts_dtype_and_raw_values() {
    let df = xy_df();
    let result = eval(
        lit(ScalarValue::Duration { raw: -42_000_000, resolution: TimestampResolution::Nanoseconds }).alias("duration"),
        &df,
    );

    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap(), DataType::new(TypeId::DurationNanoseconds));
    assert_eq!(host_export_dtype(&result), InteropDataType::new(rapids_interop::TypeId::DurationNanoseconds));
    assert_eq!(host_export_i64_values(&result), vec![-42_000_000; 3]);
}

fn raw_literals_eval_in_use_resolved_override_domain_for_allocations() {
    let source_rt = Runtime::builder().build().unwrap();
    let override_rt = Runtime::builder().build().unwrap();
    let df = DataFrame::from_series([("x", source_rt.series(&[1i32, 2, 3]).unwrap())]).unwrap();
    let execution = override_rt.exec();
    let input_placement = df.placement();
    let override_placement = override_rt.placement();

    let decimal = lit(ScalarValue::Decimal64 { raw: 12_345, scale: -2 }).eval_in(&df, execution).unwrap();
    let timestamp =
        lit(ScalarValue::Timestamp { raw: 1_700_000_123_456, resolution: TimestampResolution::Microseconds })
            .eval_in(&df, execution)
            .unwrap();
    let duration = lit(ScalarValue::Duration { raw: -42_000_000, resolution: TimestampResolution::Nanoseconds })
        .eval_in(&df, execution)
        .unwrap();

    for result in [&decimal, &timestamp, &duration] {
        let placement = result.placement();
        assert_eq!(placement.device, input_placement.device);
        assert_eq!(placement.stream_debug, override_placement.stream_debug);
        assert_eq!(placement.memory_resource_debug, override_placement.memory_resource_debug);
    }

    assert_eq!(decimal.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal64, -2).unwrap());
    assert_eq!(host_export_i64_values(&decimal), vec![12_345; 3]);
    assert_eq!(timestamp.dtype().unwrap(), DataType::new(TypeId::TimestampMicroseconds));
    assert_eq!(host_export_i64_values(&timestamp), vec![1_700_000_123_456; 3]);
    assert_eq!(duration.dtype().unwrap(), DataType::new(TypeId::DurationNanoseconds));
    assert_eq!(host_export_i64_values(&duration), vec![-42_000_000; 3]);
}

fn alias_sets_output_name_without_materializing_series_metadata() {
    let df = xy_df();
    let expr = col("x").alias("renamed");

    assert_output_name(expr.clone(), Some("renamed"));

    let result = eval(expr, &df);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

// ===========================================================================
// Arithmetic operators
// ===========================================================================

fn add_col_col() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") + col("y")).alias("sum"), &df), vec![11, 22, 33]);
}

fn add_col_lit() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") + lit(100i32)).alias("plus"), &df), vec![110, 120, 130]);
}

fn add_col_scalar_without_lit() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") + 100i32).alias("plus"), &df), vec![110, 120, 130]);
}

fn add_scalar_lhs_without_lit() {
    let df = xy_df();
    assert_eq!(eval_i32((100i32 + col("x")).alias("plus"), &df), vec![110, 120, 130]);
}

fn compound_expr_add_mul() {
    let df = xy_df();
    // (x + y) * lit(2)
    assert_eq!(eval_i32(((col("x") + col("y")) * lit(2i32)).alias("r"), &df), vec![22, 44, 66]);
}

fn sub_col_col() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") - col("y")).alias("diff"), &df), vec![9, 18, 27]);
}

fn mul_col_col() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") * col("y")).alias("prod"), &df), vec![10, 40, 90]);
}

fn div_col_col() {
    let df = xy_df();
    // cuDF integer division: 10/1=10, 20/2=10, 30/3=10
    assert_eq!(eval_i32((col("x") / col("y")).alias("quot"), &df), vec![10, 10, 10]);
}

fn binary_expr_decimal128_div_with_explicit_type_preserves_fractional_scale() {
    let df = decimal128_pair_df_from_arrow(
        "numerator",
        vec![Some(100_000_000), Some(100_000_000)],
        "denominator",
        vec![Some(400_000_000), Some(200_000_000)],
        38,
        8,
    );
    let output_type = DataType::fixed_point(TypeId::Decimal128, -8).unwrap();

    let ratio = col("numerator")
        .binary_op_with_type(col("denominator"), BinaryOperator::Div, output_type)
        .alias("ratio")
        .eval(&df)
        .unwrap();

    assert_eq!(ratio.dtype().unwrap(), output_type);
    assert_eq!(host_export_i128_values(&ratio), vec![25_000_000, 50_000_000]);

    let ratio_frame = DataFrame::from_series([("ratio", ratio)]).unwrap();
    assert_arrow_decimal128_column(&ratio_frame, "ratio", &[25_000_000, 50_000_000], 8);
}

fn with_columns_with_budget_decimal128_div_with_explicit_type_materializes_fractional_arrow_output() {
    let df = decimal128_pair_df_from_arrow(
        "numerator",
        vec![Some(100_000_000), Some(300_000_000)],
        "denominator",
        vec![Some(400_000_000), Some(600_000_000)],
        38,
        8,
    );
    let output_type = DataType::fixed_point(TypeId::Decimal128, -8).unwrap();
    let token = execution_budget_token(
        &Runtime::builder().build().unwrap(),
        ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("decimal_div_expr"),
    )
    .unwrap();

    let outcome = token
        .run("dataframe::with_columns", |exec| {
            df.on(exec).with_columns([(
                "ratio",
                col("numerator").binary_op_with_type(col("denominator"), BinaryOperator::Div, output_type),
            )])
        })
        .unwrap();
    let result = outcome.value();

    assert_eq!(outcome.report().request().boundary(), "decimal_div_expr");
    assert_eq!(result.series("ratio").unwrap().dtype().unwrap(), output_type);
    assert_arrow_decimal128_column(result, "ratio", &[25_000_000, 50_000_000], 8);
}

// Decimal division follows arrow-arith, which DataFusion uses: `lhs * 10^k / rhs` in checked i128 arithmetic over
// the rows that are valid in both operands. Decimal128(38, 8) / Decimal128(38, 8) has the arrow result type
// Decimal128(38, 12), so `k` is 12.
const DECIMAL_DIV_FACTOR: i128 = 1_000_000_000_000;

fn decimal128_div_expr() -> Expr {
    let output_type = DataType::fixed_point(TypeId::Decimal128, -12).unwrap();
    col("a").binary_op_with_type(col("b"), BinaryOperator::Div, output_type)
}

fn decimal_div_error_with_code(result: cudf::Result<Series>, code: ErrorCode) -> cudf::Error {
    let error = result.expect_err("the division must fail");
    assert_eq!(error.code(), code);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.detail_value("operator"), Some("div"));
    error
}

fn binary_expr_decimal128_div_truncates_toward_zero_and_skips_rows_with_a_null_operand() {
    // Rows 5 to 7 carry a zero divisor under or next to a null operand, which arrow never evaluates.
    let df = decimal128_pair_df_with_masked_raw_values(
        &[
            (100_000_000, true),
            (700_000_000, true),
            (-100_000_000, true),
            (5, true),
            (9, false),
            (11, true),
            (13, false),
        ],
        &[(400_000_000, true), (-200_000_000, true), (300_000_000, true), (7, true), (0, true), (0, false), (0, false)],
        38,
        8,
    );

    let quotient = decimal128_div_expr().eval(&df).unwrap();

    let expected = |lhs: i128, rhs: i128| Some(lhs * DECIMAL_DIV_FACTOR / rhs);
    assert_eq!(
        decimal128_raw_options(quotient),
        vec![
            expected(100_000_000, 400_000_000),
            expected(700_000_000, -200_000_000),
            expected(-100_000_000, 300_000_000),
            expected(5, 7),
            None,
            None,
            None
        ]
    );
}

fn binary_expr_decimal128_div_fails_on_a_zero_divisor_in_a_valid_row() {
    let df = decimal128_pair_df_with_masked_raw_values(&[(1, true), (2, true)], &[(1, true), (0, true)], 38, 8);

    let error = decimal_div_error_with_code(decimal128_div_expr().eval(&df), ErrorCode::DivisionByZero);

    assert!(error.detail_value("divisor_type_id").is_some());
}

fn binary_expr_decimal128_div_fails_when_scaling_a_valid_dividend_overflows_i128() {
    let highest = i128::MAX / DECIMAL_DIV_FACTOR;
    let lowest = i128::MIN / DECIMAL_DIV_FACTOR;
    let one = 100_000_000;

    let at_the_bounds = decimal128_pair_df_with_masked_raw_values(
        &[(highest, true), (lowest, true), (i128::MAX, false)],
        &[(one, true), (one, true), (one, true)],
        38,
        8,
    );
    assert_eq!(
        decimal128_raw_options(decimal128_div_expr().eval(&at_the_bounds).unwrap()),
        vec![Some(highest * DECIMAL_DIV_FACTOR / one), Some(lowest * DECIMAL_DIV_FACTOR / one), None]
    );

    for (label, dividend) in [("above", highest + 1), ("below", lowest - 1)] {
        let df = decimal128_pair_df_with_masked_raw_values(
            &[(1, true), (dividend, true)],
            &[(one, true), (one, true)],
            38,
            8,
        );
        let error = decimal_div_error_with_code(decimal128_div_expr().eval(&df), ErrorCode::ValueOutOfRange);
        assert_eq!(error.detail_value("reason"), Some("overflow"), "{label}");
    }

    let null_divisor = decimal128_pair_df_with_masked_raw_values(&[(highest + 1, true)], &[(0, false)], 38, 8);
    assert_eq!(decimal128_raw_options(decimal128_div_expr().eval(&null_divisor).unwrap()), vec![None]);
}

fn binary_expr_decimal128_div_fails_on_the_minimum_value_divided_by_minus_one() {
    // Equal scales make the result scale equal the dividend scale, so the dividend is not scaled.
    let df = |rows: &[(i128, bool)], divisors: &[(i128, bool)]| {
        decimal128_pair_df_with_masked_raw_values(rows, divisors, 38, 0)
    };
    let quotient = || col("a") / col("b");

    let overflow = df(&[(i128::MIN, true), (6, true)], &[(-1, true), (-1, true)]);
    let error = decimal_div_error_with_code(quotient().eval(&overflow), ErrorCode::ValueOutOfRange);
    assert_eq!(error.detail_value("reason"), Some("overflow"));

    let safe = df(&[(i128::MIN, true), (i128::MIN, false), (6, true)], &[(-2, true), (-1, true), (-1, true)]);
    assert_eq!(decimal128_raw_options(quotient().eval(&safe).unwrap()), vec![Some(i128::MIN / -2), None, Some(-6)]);
}

fn binary_expr_decimal64_div_checks_zero_divisors_and_dividend_scaling_in_its_storage_width() {
    let decimal64_df = |a: Vec<i64>, b: Vec<i64>| {
        let column = |values: Vec<i64>| Arc::new(Decimal64Array::from(values).with_precision_and_scale(18, 2).unwrap());
        let schema = Schema::new(vec![
            Field::new("a", ArrowDataType::Decimal64(18, 2), true),
            Field::new("b", ArrowDataType::Decimal64(18, 2), true),
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![column(a), column(b)]).unwrap();
        DataFrame::from_record_batch(&Runtime::builder().build().unwrap(), &batch).unwrap()
    };

    let zero_divisor = decimal64_df(vec![5, 6], vec![1, 0]);
    decimal_div_error_with_code((col("a") / col("b")).eval(&zero_divisor), ErrorCode::DivisionByZero);

    // Dividing at two more digits multiplies the dividend by 10^4, whose limit in an i64 is i64::MAX / 10^4.
    let output_type = DataType::fixed_point(TypeId::Decimal64, -4).unwrap();
    let divide = || col("a").binary_op_with_type(col("b"), BinaryOperator::Div, output_type);
    let highest = i64::MAX / 10_000;
    let at_the_bound = divide().eval(&decimal64_df(vec![highest], vec![100])).unwrap();
    assert_eq!(host_export_i64_values(&at_the_bound), vec![highest * 10_000 / 100]);
    let error = decimal_div_error_with_code(
        divide().eval(&decimal64_df(vec![highest + 1], vec![100])),
        ErrorCode::ValueOutOfRange,
    );
    assert_eq!(error.detail_value("reason"), Some("overflow"));
}

fn rem_col_lit() {
    let df = xy_df(); // x = [10, 20, 30]
    assert_eq!(eval_i32((col("x") % lit(3i32)).alias("mod3"), &df), vec![1, 2, 0]);
}

fn binary_expr_type_error_preserves_eval_operation_and_operator_detail() {
    let df = xy_df();

    let err = (col("x") + lit("not_numeric")).eval(&df).unwrap_err();

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "expr::eval_in");
    assert_eq!(err.detail_value("operator"), Some("add"));
}

fn neg_expr() {
    let df = xy_df(); // x = [10, 20, 30]
    assert_eq!(eval_i32((-col("x")).alias("neg"), &df), vec![-10, -20, -30]);
}

fn bitand_bool() {
    let a = series::<bool>("a", &[true, true, false, false]).unwrap();
    let b = series::<bool>("b", &[true, false, true, false]).unwrap();
    let df = DataFrame::from_series([("a", a), ("b", b)]).unwrap();
    let result = (col("a") & col("b")).alias("and").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false, false]);
}

fn bitand_bool_scalar_lhs_without_lit() {
    let df = bool_df("b", &[true, false, true, false]);
    let result = eval((true & col("b")).alias("and"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true, false]);
}

fn bitor_bool() {
    let a = series::<bool>("a", &[true, true, false, false]).unwrap();
    let b = series::<bool>("b", &[true, false, true, false]).unwrap();
    let df = DataFrame::from_series([("a", a), ("b", b)]).unwrap();
    let result = (col("a") | col("b")).alias("or").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, true, false]);
}

fn pow_builder() {
    let df = f64_df("x", &[2.0, 3.0, 4.0]);
    assert_eq!(eval_f64(col("x").pow(lit(2.0f64)).alias("sq"), &df), vec![4.0, 9.0, 16.0]);
}

fn pow_builder_scalar_without_lit() {
    let df = f64_df("x", &[2.0, 3.0, 4.0]);
    assert_eq!(eval_f64(col("x").pow(2.0f64).alias("sq"), &df), vec![4.0, 9.0, 16.0]);
}

fn floor_div_builder() {
    let df = xy_df(); // x = [10, 20, 30]
    assert_eq!(eval_i32(col("x").floor_div(lit(3i32)).alias("fd"), &df), vec![3, 6, 10]);
}

fn abs_builder() {
    let df = i32_df("v", &[-5, 0, 7]);
    assert_eq!(eval_i32(col("v").abs().alias("a"), &df), vec![5, 0, 7]);
}

fn sqrt_builder() {
    let df = f64_df("v", &[4.0, 9.0, 16.0]);
    assert_eq!(eval_f64(col("v").sqrt().alias("r"), &df), vec![2.0, 3.0, 4.0]);
}

fn ceil_floor_builder() {
    let df = f64_df("v", &[1.2, 2.7, -0.5]);
    assert_eq!(eval_f64(col("v").ceil().alias("c"), &df), vec![2.0, 3.0, 0.0]);
    assert_eq!(eval_f64(col("v").floor().alias("f"), &df), vec![1.0, 2.0, -1.0]);
}

fn round_builder_uses_half_up_semantics() {
    let df = f64_df("v", &[1.235, 2.225, -1.235, -2.225]);

    assert_eq!(eval_f64(col("v").round(2).alias("r"), &df), vec![1.24, 2.23, -1.24, -2.23]);
}

fn round_builder_supports_decimal128() {
    let df = decimal128_df_from_arrow("amount", vec![Some(12345), Some(12344), Some(-12345)], 20, 3);

    let result = eval(col("amount").round(2).alias("rounded_amount"), &df);

    assert_eq!(result.dtype().unwrap(), DataType::fixed_point(TypeId::Decimal128, -2).unwrap());
    assert_eq!(host_export_i128_values(&result), vec![1235, 1234, -1235]);
}

// ===========================================================================
// Comparisons
// ===========================================================================

fn gt_lit_produces_bool() {
    let df = xy_df();
    let result = eval(col("x").gt(lit(15i32)).alias("big"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

fn gt_scalar_without_lit_produces_bool() {
    let df = xy_df();
    let result = eval(col("x").gt(15i32).alias("big"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

fn eq_col_col() {
    let x = series::<i32>("a", &[1, 2, 3]).unwrap();
    let y = series::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_series([("a", x), ("b", y)]).unwrap();
    let result = col("a").eq(col("b")).alias("same").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn lt_lit_with_alias() {
    let df = xy_df();
    assert_eq!(eval_bool(col("x").lt(lit(25i32)).alias("small"), &df), vec![true, true, false]);
}

fn ne_col_col() {
    let x = series::<i32>("a", &[1, 2, 3]).unwrap();
    let y = series::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_series([("a", x), ("b", y)]).unwrap();
    let result = col("a").ne(col("b")).alias("diff").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn ge_and_le() {
    let df = xy_df();
    assert_eq!(eval_bool(col("x").ge(lit(20i32)).alias("ge20"), &df), vec![false, true, true]);
    assert_eq!(eval_bool(col("x").le(lit(20i32)).alias("le20"), &df), vec![true, true, false]);
}

fn timestamp_days_literal_comparison_matches_tpch_date_filter_shape() {
    let shipdate = series("shipdate", &[18_262i32, 18_263, 18_264])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();
    let df = df_with_series("shipdate", shipdate);

    let result = eval(
        col("shipdate")
            .lt(lit(ScalarValue::Timestamp { raw: 18_263, resolution: TimestampResolution::Days }))
            .alias("before_cutoff"),
        &df,
    );

    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn timestamp_days_plus_duration_expr_preserves_timestamp_resolution() {
    let shipdate = series("shipdate", &[18_262i32, 18_263, 18_264])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();
    let df = df_with_series("shipdate", shipdate);

    let result = eval(
        (col("shipdate") + lit(ScalarValue::Duration { raw: 5, resolution: TimestampResolution::Days }))
            .alias("later_shipdate"),
        &df,
    );

    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampDays);
    assert_eq!(host_export_i32_values(&result), vec![18_267, 18_268, 18_269]);
}

fn timestamp_days_datetime_offset_adds_calendrical_months() {
    let shipdate = series("shipdate", &[18_262i32, 18_293, 18_322])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();
    let df = df_with_series("shipdate", shipdate);

    let result = eval(col("shipdate").dt().offset(1, 2, 0).alias("shifted_shipdate"), &df);

    assert_eq!(result.dtype().unwrap().id(), TypeId::TimestampDays);
    assert_eq!(host_export_i32_values(&result), vec![18_295, 18_324, 18_355]);
}

fn timestamp_days_datetime_offset_rejects_non_exact_duration_resolution() {
    let shipdate = series("shipdate", &[18_262i32])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();
    let df = df_with_series("shipdate", shipdate);

    let err = col("shipdate").dt().offset(0, 0, 1).alias("shifted_shipdate").eval(&df).unwrap_err();

    assert_eq!(err.code(), ErrorCode::ValueOutOfRange);
    assert_eq!(err.operation(), "expr::eval_in");
    assert_eq!(err.detail_value("resolution"), Some("Days"));
    assert_eq!(err.detail_value("days"), Some("0"));
    assert_eq!(err.detail_value("nanoseconds"), Some("1"));
}

fn datetime_offset_zero_interval_still_rejects_non_timestamp_input() {
    let df = i32_df("x", &[1, 2, 3]);

    let err = col("x").dt().offset(0, 0, 0).alias("x").eval(&df).unwrap_err();

    assert_eq!(err.code(), ErrorCode::TypeMismatch);
    assert_eq!(err.operation(), "expr::eval_in");
    assert_eq!(err.detail_value("actual_type_id"), Some("int32"));
    assert_eq!(err.detail_value("expected_family"), Some("timestamp"));
}

fn timestamp_expr_datetime_namespace_extracts_components() {
    let dates = series("ts", &[0i32, 31, 365])
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::DurationDays)
        .unwrap()
        .on(Exec::RETAINED)
        .astype(TypeId::TimestampDays)
        .unwrap();
    let df = df_with_series("ts", dates);

    assert_eq!(eval_i16(col("ts").dt().year().alias("year"), &df), vec![1970, 1970, 1971]);
    assert_eq!(eval_i16(col("ts").dt().month().alias("month"), &df), vec![1, 2, 1]);
    assert_eq!(eval_i16(col("ts").dt().day().alias("day"), &df), vec![1, 1, 1]);
    assert_output_name(col("ts").dt().year(), Some("ts"));
}

// ===========================================================================
// Unary ops
// ===========================================================================

fn is_null_expr() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let result = eval(col("v").is_null().alias("nulls"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn is_not_null_expr() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    assert_eq!(eval_bool(col("v").is_not_null().alias("valid"), &df), vec![true, false, true]);
}

fn cast_i32_to_f64() {
    let df = xy_df();
    let result = eval(col("x").cast(TypeId::Float64).alias("xf"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

fn cast_f64_to_i64_truncates_and_preserves_nulls() {
    let df = nullable_f64_df("x", &[1.9, -1.9, 0.0], &[true, true, false]);

    let result = eval(col("x").cast(TypeId::Int64).alias("xi"), &df);

    assert_eq!(result.dtype().unwrap().id(), TypeId::Int64);
    assert_eq!(result.to_optional_vec::<i64>().unwrap(), vec![Some(1), Some(-1), None]);
}

fn cast_f64_to_i64_rejects_nan_and_unrepresentable_values() {
    let df = f64_df("x", &[f64::NAN, f64::INFINITY, 1.0e40]);

    let err = col("x").cast(TypeId::Int64).alias("xi").eval(&df).unwrap_err();

    assert_eq!(err.code(), ErrorCode::ValueOutOfRange);
}

fn dataframe_cast_column_f64_to_i64_uses_checked_semantics() {
    let df = nullable_f64_df("x", &[1.9, -1.9, 0.0], &[true, true, false]);

    let result = df.on(Exec::RETAINED).cast_column("x", TypeId::Int64).unwrap();

    assert_eq!(result.series("x").unwrap().dtype().unwrap().id(), TypeId::Int64);
    assert_eq!(result.series("x").unwrap().to_optional_vec::<i64>().unwrap(), vec![Some(1), Some(-1), None]);

    let invalid = f64_df("x", &[f64::NAN, f64::INFINITY, 1.0e40]);
    let err = invalid.on(Exec::RETAINED).cast_column("x", TypeId::Int64).unwrap_err();
    assert_eq!(err.code(), ErrorCode::ValueOutOfRange);
}

fn cast_decimal128_to_f64_uses_decimal_scale() {
    let df = decimal128_df_from_arrow("amount", vec![Some(12345), Some(-678), None], 20, 2);

    let result = eval(col("amount").cast(TypeId::Float64).alias("amount_f64"), &df);

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_optional_vec::<f64>().unwrap(), vec![Some(123.45), Some(-6.78), None]);
}

fn filter_expr_falls_back_to_boolean_mask_for_cast_predicates() {
    let df = xy_df();

    let filtered = df.on(Exec::RETAINED).filter_expr(col("x").cast(TypeId::Float64).gt(15.0f64)).unwrap();

    assert_eq!(filtered.series("x").unwrap().to_vec_strict::<i32>().unwrap(), vec![20, 30]);
    assert_eq!(filtered.series("y").unwrap().to_vec_strict::<i32>().unwrap(), vec![2, 3]);
}

fn not_operator() {
    let df = bool_df("mask", &[true, false, true]);
    assert_eq!(eval_bool((!col("mask")).alias("inv"), &df), vec![false, true, false]);
}

// ===========================================================================
// DataFrame::with_columns
// ===========================================================================

fn with_columns_adds_new_column() {
    let df = xy_df();
    let df2 = df.on(Exec::RETAINED).with_columns([(col("x") + col("y")).alias("sum")]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.column_names(), vec!["x", "y", "sum"]);
    let sum = df2.series("sum").unwrap();
    assert_eq!(sum.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

fn with_columns_replaces_existing() {
    let df = xy_df();
    let df2 = df.on(Exec::RETAINED).with_columns([(col("x") * lit(2i32)).alias("x")]).unwrap();
    assert_eq!(df2.num_columns(), 2);
    let x = df2.series("x").unwrap();
    assert_eq!(x.to_vec_strict::<i32>().unwrap(), vec![20, 40, 60]);
}

fn with_columns_multiple_exprs() {
    let df = xy_df();
    let df2 = df
        .on(Exec::RETAINED)
        .with_columns([(col("x") + lit(1i32)).alias("x1"), (col("y") * lit(10i32)).alias("y10")])
        .unwrap();
    assert_eq!(df2.num_columns(), 4);
    assert_eq!(df2.series("x1").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
    assert_eq!(df2.series("y10").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

fn with_columns_named_pair_allows_binary_expr_without_alias() {
    let df = xy_df();
    let df2 = df.on(Exec::RETAINED).with_columns([("x1", col("x") + 1i32)]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.series("x1").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
}

fn with_columns_named_pair_allows_if_else_without_alias() {
    let df = i32_df("x", &[1, 2, 3]);
    let df2 =
        df.on(Exec::RETAINED).with_columns([("big", when(col("x").gt(1i32)).then(100i32).otherwise(0i32))]).unwrap();
    assert_eq!(df2.series("big").unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 100, 100]);
}

fn with_columns_named_pair_overrides_inner_alias() {
    let df = xy_df();
    let df2 = df.on(Exec::RETAINED).with_columns([("x1", (col("x") + 1i32).alias("ignored"))]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.column_names(), vec!["x", "y", "x1"]);
    assert_eq!(df2.series("x1").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
}

fn with_columns_named_pair_overrides_inherited_output_name() {
    let df = xy_df();
    let df2 = df.on(Exec::RETAINED).with_columns([("x_copy", col("x").cast(TypeId::Float64))]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.column_names(), vec!["x", "y", "x_copy"]);
    assert_eq!(df2.series("x_copy").unwrap().dtype().unwrap().id(), TypeId::Float64);
}

fn with_columns_no_alias_errors() {
    let df = xy_df();
    // (col + lit) has no alias and is not a bare col(); should error.
    let err = df.on(Exec::RETAINED).with_columns([col("x") + lit(1i32)]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.operation(), "dataframe::with_columns");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("output_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("missing_output_name"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use .alias(...) or provide an explicit (name, expr) pair"),
    );
}

fn with_columns_sees_original_state() {
    let df = xy_df(); // x = [10,20,30], y = [1,2,3]
    // Both expressions see the original x, even though the first replaces x.
    let df2 =
        df.on(Exec::RETAINED).with_columns([lit(0i32).alias("x"), (col("x") + lit(1i32)).alias("x_plus_one")]).unwrap();
    let x = df2.series("x").unwrap();
    assert_eq!(x.to_vec_strict::<i32>().unwrap(), vec![0, 0, 0]);
    let xp = df2.series("x_plus_one").unwrap();
    assert_eq!(xp.to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
}

fn with_columns_bare_col_uses_column_name() {
    let df = xy_df();
    // Bare col("x") has output_name = Some("x"), so it should work without .alias().
    let df2 = df.on(Exec::RETAINED).with_columns([col("x")]).unwrap();
    assert_eq!(df2.num_columns(), 2);
}

// ===========================================================================
// Output name propagation for single-input nodes
// ===========================================================================

fn output_name_propagates_through_cast() {
    assert_output_name(col("x").cast(TypeId::Float64), Some("x"));
}

fn output_name_propagates_through_is_null() {
    assert_output_name(col("x").is_null(), Some("x"));
}

fn output_name_propagates_through_is_not_null() {
    assert_output_name(col("x").is_not_null(), Some("x"));
}

fn output_name_propagates_through_unary_op() {
    assert_output_name(col("x").abs(), Some("x"));
}

fn output_name_propagates_through_log() {
    assert_output_name(col("x").log(), Some("x"));
}

fn with_columns_cast_no_alias_needed() {
    let df = xy_df();
    let df2 = df.on(Exec::RETAINED).with_columns([col("x").cast(TypeId::Float64)]).unwrap();
    assert_eq!(df2.num_columns(), 2);
    assert_eq!(df2.series("x").unwrap().dtype().unwrap().id(), TypeId::Float64);
}

fn with_columns_is_null_no_alias_needed() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let df2 = df.on(Exec::RETAINED).with_columns([col("v").is_null()]).unwrap();
    assert_eq!(df2.series("v").unwrap().dtype().unwrap().id(), TypeId::Bool8);
}

fn in_list_expression_evaluates_literal_membership_with_nulls() {
    let df = nullable_i32_df("v", &[1, 0, 3, 4], &[true, false, true, true]);

    let result = eval(col("v").is_in_list(vec![lit(1i32), lit(4i32)]).alias("member"), &df);
    assert_eq!(result.to_optional_vec::<bool>().unwrap(), vec![Some(true), None, Some(false), Some(true)]);

    let result = eval(col("v").is_not_in_list(vec![lit(1i32), lit(4i32)]).alias("not_member"), &df);
    assert_eq!(result.to_optional_vec::<bool>().unwrap(), vec![Some(false), None, Some(true), Some(false)]);
}

fn in_list_expression_rejects_empty_haystack() {
    let df = i32_df("v", &[1, 2, 3]);

    let err = col("v").is_in_list(vec![]).alias("member").eval(&df).unwrap_err();

    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.operation(), "expr::eval_in");
    assert_eq!(err.detail_value("reason"), Some("empty_in_list"));
}

fn is_not_in_list_expression_propagates_nulls_for_all_null_input() {
    let df = nullable_i32_df("v", &[0, 0, 0], &[false, false, false]);

    let result = eval(col("v").is_not_in_list(vec![lit(1i32)]).alias("not_member"), &df);

    assert_eq!(result.to_optional_vec::<bool>().unwrap(), vec![None, None, None]);
}

fn is_not_in_list_expression_propagates_nulls_for_all_null_haystack() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);

    let result = eval(col("v").is_not_in_list(vec![lit_null(TypeId::Int32)]).alias("not_member"), &df);

    assert_eq!(result.to_optional_vec::<bool>().unwrap(), vec![None, None, None]);
}

fn eval_unary_op_display_omits_input_name() {
    let df = xy_df();
    let result = eval(col("x").abs(), &df);
    let output = format!("{}", result);
    assert!(!output.contains("x"), "expr unary output should not expose schema names: {}", output);
}

fn log_expr() {
    let df = f64_df("x", &[1.0, std::f64::consts::E, std::f64::consts::E * std::f64::consts::E]);
    let result = eval(col("x").log().alias("ln_x"), &df);
    let values = result.to_vec_strict::<f64>().unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[1] - 1.0).abs() < 1e-10);
    assert!((values[2] - 2.0).abs() < 1e-10);
}

fn exp_expr() {
    let df = f64_df("x", &[0.0, 1.0, f64::NEG_INFINITY]);
    let result = eval(col("x").exp().alias("exp_x"), &df);
    let values = result.to_vec_strict::<f64>().unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(values[0], 1.0);
    assert!((values[1] - std::f64::consts::E).abs() < 1e-12);
    assert_eq!(values[2], 0.0);
}

fn eval_fillna_display_omits_input_name() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let result = eval(col("v").fillna(99i32), &df);
    let output = format!("{}", result);
    assert!(!output.contains("v"), "expr fillna output should not expose schema names: {}", output);
}

fn eval_cumsum_preserves_name() {
    let df = i32_df("x", &[1, 2, 3]);
    let _result = eval(col("x").cumsum(), &df);
}

fn eval_reduce_has_no_implicit_name() {
    let df = i32_df("x", &[10, 20, 30]);
    let _result = eval(col("x").agg().sum(), &df);
}

fn eval_string_op_returns_values() {
    let df = str_df("s", &["Hello", "WORLD"]);
    let _result = eval(col("s").str().lower(), &df);
}

// ===========================================================================
// Null fill and cumulative expressions
// ===========================================================================

fn fillna_scalar_replaces_nulls() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let result = eval(col("v").fillna(0i32).alias("filled"), &df);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 0, 3]);
    assert_eq!(result.null_count(), 0);
}

fn fillna_scalar_preserves_output_name() {
    assert_output_name(col("v").fillna(0i32), Some("v"));
}

fn fillna_with_column_path() {
    let v = nullable_series::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let backup = series::<i32>("backup", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([("v", v), ("backup", backup)]).unwrap();
    let result = col("v").fillna_with(col("backup")).alias("filled").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(result.null_count(), 0);
}

fn fillna_with_preserves_output_name() {
    assert_output_name(col("v").fillna_with(col("backup")), Some("v"));
}

fn ffill_expr() {
    let df = nullable_i32_df("v", &[1, 0, 3, 0], &[true, false, true, false]);
    assert_eq!(eval_i32(col("v").ffill().alias("ff"), &df), vec![1, 1, 3, 3]);
}

fn bfill_expr() {
    let df = nullable_i32_df("v", &[0, 0, 3, 0], &[false, false, true, false]);
    let result = eval(col("v").bfill().alias("bf"), &df);
    // Last element remains null since there's no following value.
    assert_eq!(result.null_count(), 1);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(3), Some(3), Some(3), None]);
}

fn cumsum_expr() {
    // cumsum output type for i32 is i32 (scan preserves type)
    let df = i32_df("x", &[1, 2, 3, 4]);
    assert_eq!(eval_i32(col("x").cumsum().alias("cs"), &df), vec![1, 3, 6, 10]);
}

fn cummin_expr() {
    let df = i32_df("x", &[3, 1, 4, 1]);
    assert_eq!(eval_i32(col("x").cummin().alias("cm"), &df), vec![3, 1, 1, 1]);
}

fn cummax_expr() {
    let df = i32_df("x", &[1, 3, 2, 5]);
    assert_eq!(eval_i32(col("x").cummax().alias("cm"), &df), vec![1, 3, 3, 5]);
}

fn cumprod_expr() {
    let df = i32_df("x", &[1, 2, 3, 4]);
    assert_eq!(eval_i32(col("x").cumprod().alias("cp"), &df), vec![1, 2, 6, 24]);
}

fn cumsum_preserves_output_name() {
    assert_output_name(col("x").cumsum(), Some("x"));
}

fn with_columns_fillna_no_alias_needed() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let df2 = df.on(Exec::RETAINED).with_columns([col("v").fillna(99i32)]).unwrap();
    assert_eq!(df2.series("v").unwrap().null_count(), 0);
    assert_eq!(df2.series("v").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 99, 3]);
}

fn with_columns_cumsum_no_alias_needed() {
    let df = i32_df("x", &[1, 2, 3]);
    let df2 = df.on(Exec::RETAINED).with_columns([col("x").cumsum()]).unwrap();
    assert_eq!(df2.series("x").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3, 6]);
}

// ===========================================================================
// Aggregation expressions with broadcast results
// ===========================================================================

fn agg_sum() {
    let df = i32_df("x", &[10, 20, 30]);
    let result = eval(col("x").agg().sum().alias("s"), &df);
    assert_eq!(result.len(), 3);
    // i32 sum → i64
    assert_eq!(result.to_vec_strict::<i64>().unwrap(), vec![60, 60, 60]);
}

fn agg_mean() {
    let df = i32_df("x", &[10, 20, 30]);
    let result = eval(col("x").agg().mean().alias("m"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![20.0, 20.0, 20.0]);
}

fn agg_min_max() {
    let df = i32_df("x", &[10, 20, 30]);
    assert_eq!(eval_i32(col("x").agg().min().alias("mn"), &df), vec![10, 10, 10]);
    assert_eq!(eval_i32(col("x").agg().max().alias("mx"), &df), vec![30, 30, 30]);
}

fn agg_std_var() {
    let df = f64_df("x", &[2.0, 4.0, 6.0]);
    let std_r = eval(col("x").agg().std(1).alias("s"), &df);
    assert_eq!(std_r.dtype().unwrap().id(), TypeId::Float64);
    let var_r = eval(col("x").agg().var(1).alias("v"), &df);
    assert_eq!(var_r.dtype().unwrap().id(), TypeId::Float64);
    // var = 4.0, std = 2.0 for [2,4,6] with ddof=1
    let std_vals = std_r.to_vec_strict::<f64>().unwrap();
    assert!((std_vals[0] - 2.0).abs() < 1e-10);
    let var_vals = var_r.to_vec_strict::<f64>().unwrap();
    assert!((var_vals[0] - 4.0).abs() < 1e-10);
}

fn agg_median() {
    let df = i32_df("x", &[10, 20, 30]);
    let result = eval(col("x").agg().median().alias("med"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![20.0, 20.0, 20.0]);
}

fn agg_any_all() {
    let df = bool_df("f", &[true, false, true]);
    assert_eq!(eval_bool(col("f").agg().any().alias("a"), &df), vec![true, true, true]);
    assert_eq!(eval_bool(col("f").agg().all().alias("b"), &df), vec![false, false, false]);
}

fn agg_product() {
    let df = i32_df("x", &[2, 3, 5]);
    assert_eq!(eval_i32(col("x").agg().product().alias("p"), &df), vec![30, 30, 30]);
}

fn agg_sum_string_reports_expr_reduce_operation_on_type_error() {
    let df = str_df("s", &["a", "b"]);
    let err = col("s").agg().sum().alias("total").eval(&df).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "expr::reduce");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(error_context_value(&err, "input_type").as_deref(), None);
}

fn reduce_requires_explicit_output_name() {
    assert_output_name(col("x").agg().sum(), None);
}

fn pct_pattern() {
    let df = f64_df("x", &[10.0, 20.0, 30.0]);
    // (x / x.agg().sum()) should give each row's fraction of total.
    let vals = eval_f64((col("x") / col("x").agg().sum()).alias("pct"), &df);
    assert!((vals[0] - 10.0 / 60.0).abs() < 1e-10);
    assert!((vals[1] - 20.0 / 60.0).abs() < 1e-10);
    assert!((vals[2] - 30.0 / 60.0).abs() < 1e-10);
}

fn with_columns_rejects_unaliased_reduce() {
    let df = i32_df("x", &[10, 20, 30]);
    let err = df.on(Exec::RETAINED).with_columns([col("x").agg().sum()]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.operation(), "dataframe::with_columns");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("output_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("missing_output_name"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use .alias(...) or provide an explicit (name, expr) pair"),
    );
}

fn with_columns_reduce_requires_alias_or_explicit_name() {
    let df = i32_df("x", &[10, 20, 30]);
    let df2 = df.on(Exec::RETAINED).with_columns([("x_total", col("x").agg().sum())]).unwrap();
    assert_eq!(df2.series("x_total").unwrap().to_vec_strict::<i64>().unwrap(), vec![60, 60, 60]);
}

// ===========================================================================
// String expressions
// ===========================================================================

fn str_lower_expr() {
    let df = str_df("s", &["Hello", "WORLD", "Foo"]);
    let result = eval(col("s").str().lower().alias("lo"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
    assert_eq!(result.len(), 3);
}

fn str_upper_expr() {
    let df = str_df("s", &["Hello", "world", "Foo"]);
    let result = eval(col("s").str().upper().alias("up"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
}

fn str_strip_expr() {
    let df = str_df("s", &["  hi  ", " lo", "ok "]);
    let stripped = eval(col("s").str().strip().alias("st"), &df);
    assert_eq!(stripped.dtype().unwrap().id(), TypeId::String);
    let l = eval(col("s").str().lstrip().alias("ls"), &df);
    assert_eq!(l.dtype().unwrap().id(), TypeId::String);
    let r = eval(col("s").str().rstrip().alias("rs"), &df);
    assert_eq!(r.dtype().unwrap().id(), TypeId::String);
}

fn str_len_expr() {
    let df = str_df("s", &["ab", "cde", "f"]);
    assert_eq!(eval_i32(col("s").str().len().alias("n"), &df), vec![2, 3, 1]);
}

fn str_byte_count_expr() {
    let df = str_df("s", &["ab", "cde", "f"]);
    assert_eq!(eval_i32(col("s").str().byte_count().alias("n"), &df), vec![2, 3, 1]);
}

fn str_contains_expr() {
    let df = str_df("s", &["foobar", "baz", "foo"]);
    let result = eval(col("s").str().contains("foo").alias("has"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn str_like_expr() {
    let df = str_df("s", &["abc", "axc", "abbc", "literal%"]);
    let result = eval(col("s").str().like("a_c").alias("matches"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, false, false]);

    let escaped = eval(col("s").str().like_escape("literal/%", "/").alias("escaped"), &df);
    assert_eq!(escaped.to_vec_strict::<bool>().unwrap(), vec![false, false, false, true]);
}

fn str_slice_expr() {
    let df = str_df("s", &["abcdef", "\u{4F60}\u{597D}\u{4E16}\u{754C}"]);
    let result = eval(col("s").str().slice(StringSliceOptions::new(Some(1), Some(3))).alias("part"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
    assert_eq!(result.to_strings_strict().unwrap(), vec!["bc".to_string(), "\u{597D}\u{4E16}".to_string()]);
}

fn str_startswith_expr() {
    let df = str_df("s", &["abc", "bca", "axy"]);
    assert_eq!(eval_bool(col("s").str().starts_with("a").alias("sw"), &df), vec![true, false, true]);
}

fn str_endswith_expr() {
    let df = str_df("s", &["abc", "bca", "xya"]);
    assert_eq!(eval_bool(col("s").str().ends_with("a").alias("ew"), &df), vec![false, true, true]);
}

fn str_preserves_output_name() {
    assert_output_name(col("s").str().lower(), Some("s"));
}

fn with_columns_str_lower_no_alias() {
    let df = str_df("s", &["Hello", "WORLD"]);
    let df2 = df.on(Exec::RETAINED).with_columns([col("s").str().lower()]).unwrap();
    assert_eq!(df2.series("s").unwrap().dtype().unwrap().id(), TypeId::String);
}

// ===========================================================================
// Conditional expressions
// ===========================================================================

fn when_then_otherwise_basic() {
    let df = i32_df("x", &[1, 2, 3]);
    assert_eq!(
        eval_i32(when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)).alias("out"), &df,),
        vec![0, 100, 100]
    );
}

fn when_then_otherwise_scalar_literals_without_lit() {
    let df = i32_df("x", &[1, 2, 3]);
    assert_eq!(eval_i32(when(col("x").gt(1i32)).then(100i32).otherwise(0i32).alias("out"), &df), vec![0, 100, 100]);
}

fn when_then_otherwise_column_branches() {
    let x = series::<i32>("x", &[10, 20, 30]).unwrap();
    let y = series::<i32>("y", &[1, 2, 3]).unwrap();
    let flag = series::<bool>("f", &[true, false, true]).unwrap();
    let df = DataFrame::from_series([("x", x), ("y", y), ("f", flag)]).unwrap();
    let result = when(col("f")).then(col("x")).otherwise(col("y")).alias("pick").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 2, 30]);
}

fn when_then_otherwise_with_nulls_in_condition() {
    // libcudf copy_if_else: null mask positions produce the false_val
    let cond = nullable_series::<bool>("c", &[true, false, false], &[true, false, true]).unwrap();
    let x = series::<i32>("x", &[10, 20, 30]).unwrap();
    let y = series::<i32>("y", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([("c", cond), ("x", x), ("y", y)]).unwrap();
    let result = when(col("c")).then(col("x")).otherwise(col("y")).alias("out").eval(&df).unwrap();
    // row 0: cond valid & true  → x[0]=10
    // row 1: cond null          → y[1]=2
    // row 2: cond valid & false → y[2]=3
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 2, 3]);
}

fn when_then_otherwise_string_branches() {
    let df = i32_df("age", &[20, 10, 30]);
    let result =
        eval(when(col("age").ge(lit(18i32))).then(lit("adult")).otherwise(lit("minor")).alias("category"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_strings_strict().unwrap(), vec!["adult", "minor", "adult"]);
}

fn if_else_output_name_is_none() {
    assert_output_name(when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)), None);
}

fn if_else_requires_alias_in_with_columns() {
    let df = i32_df("x", &[1, 2, 3]);
    let err = df
        .on(Exec::RETAINED)
        .with_columns([when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32))])
        .unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.operation(), "dataframe::with_columns");
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("output_name"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("missing_output_name"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("use .alias(...) or provide an explicit (name, expr) pair"),
    );
}

fn if_else_with_alias_in_with_columns() {
    let df = i32_df("x", &[1, 2, 3]);
    let df2 = df
        .on(Exec::RETAINED)
        .with_columns([when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)).alias("big")])
        .unwrap();
    assert_eq!(df2.series("big").unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 100, 100]);
}

fn when_then_otherwise_composite() {
    let df = i32_df("age", &[25, 12, 18]);
    let df2 = df
        .on(Exec::RETAINED)
        .with_columns([when(col("age").ge(lit(18i32))).then(lit("adult")).otherwise(lit("minor")).alias("category")])
        .unwrap();
    assert_eq!(df2.series("category").unwrap().dtype().unwrap().id(), TypeId::String);
    assert_eq!(df2.series("category").unwrap().len(), 3);
    assert_eq!(df2.series("category").unwrap().to_strings_strict().unwrap(), vec!["adult", "minor", "adult"]);
}

#[test]
fn expr_tests_cases() {
    run_named_cases(&[
        ("col_resolves_existing_column", col_resolves_existing_column),
        ("col_missing_column_errors", col_missing_column_errors),
        ("lit_i32_broadcasts", lit_i32_broadcasts),
        (
            "into_lit_and_into_expr_resolve_unambiguously_after_glob_import",
            into_lit_and_into_expr_resolve_unambiguously_after_glob_import,
        ),
        ("lit_f64_broadcasts", lit_f64_broadcasts),
        ("lit_string_broadcasts", lit_string_broadcasts),
        ("lit_null_broadcasts_typed_nulls", lit_null_broadcasts_typed_nulls),
        ("lit_decimal32_broadcasts_dtype_scale_and_raw_values", lit_decimal32_broadcasts_dtype_scale_and_raw_values),
        ("lit_decimal64_broadcasts_dtype_scale_and_raw_values", lit_decimal64_broadcasts_dtype_scale_and_raw_values),
        ("lit_decimal128_broadcasts_dtype_scale_and_raw_values", lit_decimal128_broadcasts_dtype_scale_and_raw_values),
        (
            "binary_expr_aligns_mixed_decimal_storage_before_arithmetic",
            binary_expr_aligns_mixed_decimal_storage_before_arithmetic,
        ),
        ("lit_timestamp_broadcasts_dtype_and_raw_values", lit_timestamp_broadcasts_dtype_and_raw_values),
        ("lit_duration_broadcasts_dtype_and_raw_values", lit_duration_broadcasts_dtype_and_raw_values),
        (
            "raw_literals_eval_in_use_resolved_override_domain_for_allocations",
            raw_literals_eval_in_use_resolved_override_domain_for_allocations,
        ),
        (
            "alias_sets_output_name_without_materializing_series_metadata",
            alias_sets_output_name_without_materializing_series_metadata,
        ),
        ("add_col_col", add_col_col),
        ("add_col_lit", add_col_lit),
        ("add_col_scalar_without_lit", add_col_scalar_without_lit),
        ("add_scalar_lhs_without_lit", add_scalar_lhs_without_lit),
        ("compound_expr_add_mul", compound_expr_add_mul),
        ("sub_col_col", sub_col_col),
        ("mul_col_col", mul_col_col),
        ("div_col_col", div_col_col),
        (
            "binary_expr_decimal128_div_with_explicit_type_preserves_fractional_scale",
            binary_expr_decimal128_div_with_explicit_type_preserves_fractional_scale,
        ),
        (
            "with_columns_with_budget_decimal128_div_with_explicit_type_materializes_fractional_arrow_output",
            with_columns_with_budget_decimal128_div_with_explicit_type_materializes_fractional_arrow_output,
        ),
        (
            "binary_expr_decimal128_div_truncates_toward_zero_and_skips_rows_with_a_null_operand",
            binary_expr_decimal128_div_truncates_toward_zero_and_skips_rows_with_a_null_operand,
        ),
        (
            "binary_expr_decimal128_div_fails_on_a_zero_divisor_in_a_valid_row",
            binary_expr_decimal128_div_fails_on_a_zero_divisor_in_a_valid_row,
        ),
        (
            "binary_expr_decimal128_div_fails_when_scaling_a_valid_dividend_overflows_i128",
            binary_expr_decimal128_div_fails_when_scaling_a_valid_dividend_overflows_i128,
        ),
        (
            "binary_expr_decimal128_div_fails_on_the_minimum_value_divided_by_minus_one",
            binary_expr_decimal128_div_fails_on_the_minimum_value_divided_by_minus_one,
        ),
        (
            "binary_expr_decimal64_div_checks_zero_divisors_and_dividend_scaling_in_its_storage_width",
            binary_expr_decimal64_div_checks_zero_divisors_and_dividend_scaling_in_its_storage_width,
        ),
        ("rem_col_lit", rem_col_lit),
        (
            "binary_expr_type_error_preserves_eval_operation_and_operator_detail",
            binary_expr_type_error_preserves_eval_operation_and_operator_detail,
        ),
        ("neg_expr", neg_expr),
        ("bitand_bool", bitand_bool),
        ("bitand_bool_scalar_lhs_without_lit", bitand_bool_scalar_lhs_without_lit),
        ("bitor_bool", bitor_bool),
        ("pow_builder", pow_builder),
        ("pow_builder_scalar_without_lit", pow_builder_scalar_without_lit),
        ("floor_div_builder", floor_div_builder),
        ("abs_builder", abs_builder),
        ("sqrt_builder", sqrt_builder),
        ("ceil_floor_builder", ceil_floor_builder),
        ("round_builder_uses_half_up_semantics", round_builder_uses_half_up_semantics),
        ("round_builder_supports_decimal128", round_builder_supports_decimal128),
        ("gt_lit_produces_bool", gt_lit_produces_bool),
        ("gt_scalar_without_lit_produces_bool", gt_scalar_without_lit_produces_bool),
        ("eq_col_col", eq_col_col),
        ("lt_lit_with_alias", lt_lit_with_alias),
        ("ne_col_col", ne_col_col),
        ("ge_and_le", ge_and_le),
        (
            "timestamp_days_literal_comparison_matches_tpch_date_filter_shape",
            timestamp_days_literal_comparison_matches_tpch_date_filter_shape,
        ),
        (
            "timestamp_days_plus_duration_expr_preserves_timestamp_resolution",
            timestamp_days_plus_duration_expr_preserves_timestamp_resolution,
        ),
        (
            "timestamp_days_datetime_offset_adds_calendrical_months",
            timestamp_days_datetime_offset_adds_calendrical_months,
        ),
        (
            "timestamp_days_datetime_offset_rejects_non_exact_duration_resolution",
            timestamp_days_datetime_offset_rejects_non_exact_duration_resolution,
        ),
        (
            "datetime_offset_zero_interval_still_rejects_non_timestamp_input",
            datetime_offset_zero_interval_still_rejects_non_timestamp_input,
        ),
        (
            "timestamp_expr_datetime_namespace_extracts_components",
            timestamp_expr_datetime_namespace_extracts_components,
        ),
        ("is_null_expr", is_null_expr),
        ("is_not_null_expr", is_not_null_expr),
        ("cast_i32_to_f64", cast_i32_to_f64),
        ("cast_f64_to_i64_truncates_and_preserves_nulls", cast_f64_to_i64_truncates_and_preserves_nulls),
        (
            "cast_f64_to_i64_rejects_nan_and_unrepresentable_values",
            cast_f64_to_i64_rejects_nan_and_unrepresentable_values,
        ),
        (
            "dataframe_cast_column_f64_to_i64_uses_checked_semantics",
            dataframe_cast_column_f64_to_i64_uses_checked_semantics,
        ),
        ("cast_decimal128_to_f64_uses_decimal_scale", cast_decimal128_to_f64_uses_decimal_scale),
        (
            "filter_expr_falls_back_to_boolean_mask_for_cast_predicates",
            filter_expr_falls_back_to_boolean_mask_for_cast_predicates,
        ),
        ("not_operator", not_operator),
        ("with_columns_adds_new_column", with_columns_adds_new_column),
        ("with_columns_replaces_existing", with_columns_replaces_existing),
        ("with_columns_multiple_exprs", with_columns_multiple_exprs),
        (
            "with_columns_named_pair_allows_binary_expr_without_alias",
            with_columns_named_pair_allows_binary_expr_without_alias,
        ),
        ("with_columns_named_pair_allows_if_else_without_alias", with_columns_named_pair_allows_if_else_without_alias),
        ("with_columns_named_pair_overrides_inner_alias", with_columns_named_pair_overrides_inner_alias),
        (
            "with_columns_named_pair_overrides_inherited_output_name",
            with_columns_named_pair_overrides_inherited_output_name,
        ),
        ("with_columns_no_alias_errors", with_columns_no_alias_errors),
        ("with_columns_sees_original_state", with_columns_sees_original_state),
        ("with_columns_bare_col_uses_column_name", with_columns_bare_col_uses_column_name),
        ("output_name_propagates_through_cast", output_name_propagates_through_cast),
        ("output_name_propagates_through_is_null", output_name_propagates_through_is_null),
        ("output_name_propagates_through_is_not_null", output_name_propagates_through_is_not_null),
        ("output_name_propagates_through_unary_op", output_name_propagates_through_unary_op),
        ("output_name_propagates_through_log", output_name_propagates_through_log),
        ("with_columns_cast_no_alias_needed", with_columns_cast_no_alias_needed),
        ("with_columns_is_null_no_alias_needed", with_columns_is_null_no_alias_needed),
        (
            "in_list_expression_evaluates_literal_membership_with_nulls",
            in_list_expression_evaluates_literal_membership_with_nulls,
        ),
        ("in_list_expression_rejects_empty_haystack", in_list_expression_rejects_empty_haystack),
        (
            "is_not_in_list_expression_propagates_nulls_for_all_null_input",
            is_not_in_list_expression_propagates_nulls_for_all_null_input,
        ),
        (
            "is_not_in_list_expression_propagates_nulls_for_all_null_haystack",
            is_not_in_list_expression_propagates_nulls_for_all_null_haystack,
        ),
        ("eval_unary_op_display_omits_input_name", eval_unary_op_display_omits_input_name),
        ("log_expr", log_expr),
        ("exp_expr", exp_expr),
        ("eval_fillna_display_omits_input_name", eval_fillna_display_omits_input_name),
        ("eval_cumsum_preserves_name", eval_cumsum_preserves_name),
        ("eval_reduce_has_no_implicit_name", eval_reduce_has_no_implicit_name),
        ("eval_string_op_returns_values", eval_string_op_returns_values),
        ("fillna_scalar_replaces_nulls", fillna_scalar_replaces_nulls),
        ("fillna_scalar_preserves_output_name", fillna_scalar_preserves_output_name),
        ("fillna_with_column_path", fillna_with_column_path),
        ("fillna_with_preserves_output_name", fillna_with_preserves_output_name),
        ("ffill_expr", ffill_expr),
        ("bfill_expr", bfill_expr),
        ("cumsum_expr", cumsum_expr),
        ("cummin_expr", cummin_expr),
        ("cummax_expr", cummax_expr),
        ("cumprod_expr", cumprod_expr),
        ("cumsum_preserves_output_name", cumsum_preserves_output_name),
        ("with_columns_fillna_no_alias_needed", with_columns_fillna_no_alias_needed),
        ("with_columns_cumsum_no_alias_needed", with_columns_cumsum_no_alias_needed),
        ("agg_sum", agg_sum),
        ("agg_mean", agg_mean),
        ("agg_min_max", agg_min_max),
        ("agg_std_var", agg_std_var),
        ("agg_median", agg_median),
        ("agg_any_all", agg_any_all),
        ("agg_product", agg_product),
        (
            "agg_sum_string_reports_expr_reduce_operation_on_type_error",
            agg_sum_string_reports_expr_reduce_operation_on_type_error,
        ),
        ("reduce_requires_explicit_output_name", reduce_requires_explicit_output_name),
        ("pct_pattern", pct_pattern),
        ("with_columns_rejects_unaliased_reduce", with_columns_rejects_unaliased_reduce),
        ("with_columns_reduce_requires_alias_or_explicit_name", with_columns_reduce_requires_alias_or_explicit_name),
        ("str_lower_expr", str_lower_expr),
        ("str_upper_expr", str_upper_expr),
        ("str_strip_expr", str_strip_expr),
        ("str_len_expr", str_len_expr),
        ("str_byte_count_expr", str_byte_count_expr),
        ("str_contains_expr", str_contains_expr),
        ("str_like_expr", str_like_expr),
        ("str_slice_expr", str_slice_expr),
        ("str_startswith_expr", str_startswith_expr),
        ("str_endswith_expr", str_endswith_expr),
        ("str_preserves_output_name", str_preserves_output_name),
        ("with_columns_str_lower_no_alias", with_columns_str_lower_no_alias),
        ("when_then_otherwise_basic", when_then_otherwise_basic),
        ("when_then_otherwise_scalar_literals_without_lit", when_then_otherwise_scalar_literals_without_lit),
        ("when_then_otherwise_column_branches", when_then_otherwise_column_branches),
        ("when_then_otherwise_with_nulls_in_condition", when_then_otherwise_with_nulls_in_condition),
        ("when_then_otherwise_string_branches", when_then_otherwise_string_branches),
        ("if_else_output_name_is_none", if_else_output_name_is_none),
        ("if_else_requires_alias_in_with_columns", if_else_requires_alias_in_with_columns),
        ("if_else_with_alias_in_with_columns", if_else_with_alias_in_with_columns),
        ("when_then_otherwise_composite", when_then_otherwise_composite),
    ]);
}
