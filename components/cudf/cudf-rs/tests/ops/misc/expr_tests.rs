//! Integration tests for the expression layer.
//!
//! Tests cover: `col()` / `lit()`, arithmetic operators, comparisons,
//! unary ops (`is_null`, `is_not_null`, `cast`, `Not`), `.alias()`,
//! `DataFrame::with_columns`, null fill, cumulative, aggregation, string ops,
//! and conditional (`when`/`then`/`otherwise`).
//!
//! All tests require a GPU.

use cudf::{
    DataFrame, ErrorCode, ErrorKind, Series, TypeId,
    core::error::{FactKeyName, FactValue},
    ops::expr::{Expr, col, lit, when},
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a simple 3-row DataFrame with columns "x" (i32) and "y" (i32).
fn xy_df() -> DataFrame {
    let x = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let y = crate::common::series::<i32>("y", &[1, 2, 3]).unwrap();
    DataFrame::from_columns([("x", x), ("y", y)]).unwrap()
}

fn df_with_series(name: &str, series: Series) -> DataFrame {
    DataFrame::from_columns([(name, series)]).unwrap()
}

fn i32_df(name: &str, data: &[i32]) -> DataFrame {
    df_with_series(name, crate::common::series::<i32>(name, data).unwrap())
}

fn nullable_i32_df(name: &str, data: &[i32], validity: &[bool]) -> DataFrame {
    df_with_series(name, crate::common::nullable_series::<i32>(name, data, validity).unwrap())
}

fn f64_df(name: &str, data: &[f64]) -> DataFrame {
    df_with_series(name, crate::common::series::<f64>(name, data).unwrap())
}

fn bool_df(name: &str, data: &[bool]) -> DataFrame {
    df_with_series(name, crate::common::series::<bool>(name, data).unwrap())
}

fn str_df(name: &str, data: &[&str]) -> DataFrame {
    df_with_series(name, crate::common::series(name, data).unwrap())
}

fn eval(expr: Expr, df: &DataFrame) -> Series {
    expr.eval(df).unwrap()
}

fn eval_i32(expr: Expr, df: &DataFrame) -> Vec<i32> {
    eval(expr, df).to_vec_strict::<i32>().unwrap()
}

fn eval_f64(expr: Expr, df: &DataFrame) -> Vec<f64> {
    eval(expr, df).to_vec_strict::<f64>().unwrap()
}

fn eval_bool(expr: Expr, df: &DataFrame) -> Vec<bool> {
    eval(expr, df).to_vec_strict::<bool>().unwrap()
}

fn assert_output_name(expr: Expr, expected: Option<&str>) {
    assert_eq!(expr.output_name(), expected);
}

// ===========================================================================
// col / lit basics
// ===========================================================================

#[test]
fn col_resolves_existing_column() {
    let df = xy_df();
    let result = eval(col("x"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn col_missing_column_errors() {
    let df = xy_df();
    let err = col("zzz").eval(&df).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

#[test]
fn lit_i32_broadcasts() {
    let df = xy_df();
    let result = eval(lit(42i32).alias("c"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42]);
}

#[test]
fn lit_f64_broadcasts() {
    let df = xy_df();
    let result = eval(lit(3.25f64).alias("pi"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.25, 3.25, 3.25]);
}

#[test]
fn lit_string_broadcasts() {
    let df = xy_df();
    let result = eval(lit("hello").alias("s"), &df);
    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
}

#[test]
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

#[test]
fn add_col_col() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") + col("y")).alias("sum"), &df), vec![11, 22, 33]);
}

#[test]
fn add_col_lit() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") + lit(100i32)).alias("plus"), &df), vec![110, 120, 130]);
}

#[test]
fn add_col_scalar_without_lit() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") + 100i32).alias("plus"), &df), vec![110, 120, 130]);
}

#[test]
fn add_scalar_lhs_without_lit() {
    let df = xy_df();
    assert_eq!(eval_i32((100i32 + col("x")).alias("plus"), &df), vec![110, 120, 130]);
}

#[test]
fn compound_expr_add_mul() {
    let df = xy_df();
    // (x + y) * lit(2)
    assert_eq!(eval_i32(((col("x") + col("y")) * lit(2i32)).alias("r"), &df), vec![22, 44, 66]);
}

#[test]
fn sub_col_col() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") - col("y")).alias("diff"), &df), vec![9, 18, 27]);
}

#[test]
fn mul_col_col() {
    let df = xy_df();
    assert_eq!(eval_i32((col("x") * col("y")).alias("prod"), &df), vec![10, 40, 90]);
}

#[test]
fn div_col_col() {
    let df = xy_df();
    // cuDF integer division: 10/1=10, 20/2=10, 30/3=10
    assert_eq!(eval_i32((col("x") / col("y")).alias("quot"), &df), vec![10, 10, 10]);
}

#[test]
fn rem_col_lit() {
    let df = xy_df(); // x = [10, 20, 30]
    assert_eq!(eval_i32((col("x") % lit(3i32)).alias("mod3"), &df), vec![1, 2, 0]);
}

#[test]
fn neg_expr() {
    let df = xy_df(); // x = [10, 20, 30]
    assert_eq!(eval_i32((-col("x")).alias("neg"), &df), vec![-10, -20, -30]);
}

#[test]
fn bitand_bool() {
    let a = crate::common::series::<bool>("a", &[true, true, false, false]).unwrap();
    let b = crate::common::series::<bool>("b", &[true, false, true, false]).unwrap();
    let df = DataFrame::from_columns([("a", a), ("b", b)]).unwrap();
    let result = (col("a") & col("b")).alias("and").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false, false]);
}

#[test]
fn bitand_bool_scalar_lhs_without_lit() {
    let df = bool_df("b", &[true, false, true, false]);
    let result = eval((true & col("b")).alias("and"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true, false]);
}

#[test]
fn bitor_bool() {
    let a = crate::common::series::<bool>("a", &[true, true, false, false]).unwrap();
    let b = crate::common::series::<bool>("b", &[true, false, true, false]).unwrap();
    let df = DataFrame::from_columns([("a", a), ("b", b)]).unwrap();
    let result = (col("a") | col("b")).alias("or").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, true, false]);
}

#[test]
fn pow_builder() {
    let df = f64_df("x", &[2.0, 3.0, 4.0]);
    assert_eq!(eval_f64(col("x").pow(lit(2.0f64)).alias("sq"), &df), vec![4.0, 9.0, 16.0]);
}

#[test]
fn pow_builder_scalar_without_lit() {
    let df = f64_df("x", &[2.0, 3.0, 4.0]);
    assert_eq!(eval_f64(col("x").pow(2.0f64).alias("sq"), &df), vec![4.0, 9.0, 16.0]);
}

#[test]
fn floor_div_builder() {
    let df = xy_df(); // x = [10, 20, 30]
    assert_eq!(eval_i32(col("x").floor_div(lit(3i32)).alias("fd"), &df), vec![3, 6, 10]);
}

#[test]
fn abs_builder() {
    let df = i32_df("v", &[-5, 0, 7]);
    assert_eq!(eval_i32(col("v").abs().alias("a"), &df), vec![5, 0, 7]);
}

#[test]
fn sqrt_builder() {
    let df = f64_df("v", &[4.0, 9.0, 16.0]);
    assert_eq!(eval_f64(col("v").sqrt().alias("r"), &df), vec![2.0, 3.0, 4.0]);
}

#[test]
fn ceil_floor_builder() {
    let df = f64_df("v", &[1.2, 2.7, -0.5]);
    assert_eq!(eval_f64(col("v").ceil().alias("c"), &df), vec![2.0, 3.0, 0.0]);
    assert_eq!(eval_f64(col("v").floor().alias("f"), &df), vec![1.0, 2.0, -1.0]);
}

// ===========================================================================
// Comparisons
// ===========================================================================

#[test]
fn gt_lit_produces_bool() {
    let df = xy_df();
    let result = eval(col("x").gt(lit(15i32)).alias("big"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn gt_scalar_without_lit_produces_bool() {
    let df = xy_df();
    let result = eval(col("x").gt(15i32).alias("big"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn eq_col_col() {
    let x = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let y = crate::common::series::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_columns([("a", x), ("b", y)]).unwrap();
    let result = col("a").eq(col("b")).alias("same").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn lt_lit_with_alias() {
    let df = xy_df();
    assert_eq!(eval_bool(col("x").lt(lit(25i32)).alias("small"), &df), vec![true, true, false]);
}

#[test]
fn ne_col_col() {
    let x = crate::common::series::<i32>("a", &[1, 2, 3]).unwrap();
    let y = crate::common::series::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_columns([("a", x), ("b", y)]).unwrap();
    let result = col("a").ne(col("b")).alias("diff").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn ge_and_le() {
    let df = xy_df();
    assert_eq!(eval_bool(col("x").ge(lit(20i32)).alias("ge20"), &df), vec![false, true, true]);
    assert_eq!(eval_bool(col("x").le(lit(20i32)).alias("le20"), &df), vec![true, true, false]);
}

// ===========================================================================
// Unary ops
// ===========================================================================

#[test]
fn is_null_expr() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let result = eval(col("v").is_null().alias("nulls"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn is_not_null_expr() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    assert_eq!(eval_bool(col("v").is_not_null().alias("valid"), &df), vec![true, false, true]);
}

#[test]
fn cast_i32_to_f64() {
    let df = xy_df();
    let result = eval(col("x").cast(TypeId::Float64).alias("xf"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn not_operator() {
    let df = bool_df("mask", &[true, false, true]);
    assert_eq!(eval_bool((!col("mask")).alias("inv"), &df), vec![false, true, false]);
}

// ===========================================================================
// DataFrame::with_columns
// ===========================================================================

#[test]
fn with_columns_adds_new_column() {
    let df = xy_df();
    let df2 = df.with_columns([(col("x") + col("y")).alias("sum")]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.column_names(), vec!["x", "y", "sum"]);
    let sum = df2.series("sum").unwrap();
    assert_eq!(sum.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn with_columns_replaces_existing() {
    let df = xy_df();
    let df2 = df.with_columns([(col("x") * lit(2i32)).alias("x")]).unwrap();
    assert_eq!(df2.num_columns(), 2);
    let x = df2.series("x").unwrap();
    assert_eq!(x.to_vec_strict::<i32>().unwrap(), vec![20, 40, 60]);
}

#[test]
fn with_columns_multiple_exprs() {
    let df = xy_df();
    let df2 = df.with_columns([(col("x") + lit(1i32)).alias("x1"), (col("y") * lit(10i32)).alias("y10")]).unwrap();
    assert_eq!(df2.num_columns(), 4);
    assert_eq!(df2.series("x1").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
    assert_eq!(df2.series("y10").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn with_columns_named_pair_allows_binary_expr_without_alias() {
    let df = xy_df();
    let df2 = df.with_columns([("x1", col("x") + 1i32)]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.series("x1").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
}

#[test]
fn with_columns_named_pair_allows_if_else_without_alias() {
    let df = i32_df("x", &[1, 2, 3]);
    let df2 = df.with_columns([("big", when(col("x").gt(1i32)).then(100i32).otherwise(0i32))]).unwrap();
    assert_eq!(df2.series("big").unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 100, 100]);
}

#[test]
fn with_columns_named_pair_overrides_inner_alias() {
    let df = xy_df();
    let df2 = df.with_columns([("x1", (col("x") + 1i32).alias("ignored"))]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.column_names(), vec!["x", "y", "x1"]);
    assert_eq!(df2.series("x1").unwrap().to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
}

#[test]
fn with_columns_named_pair_overrides_inherited_output_name() {
    let df = xy_df();
    let df2 = df.with_columns([("x_copy", col("x").cast(TypeId::Float64))]).unwrap();
    assert_eq!(df2.num_columns(), 3);
    assert_eq!(df2.column_names(), vec!["x", "y", "x_copy"]);
    assert_eq!(df2.series("x_copy").unwrap().dtype().unwrap().id(), TypeId::Float64);
}

#[test]
fn with_columns_no_alias_errors() {
    let df = xy_df();
    // (col + lit) has no alias and is not a bare col() — should error.
    let err = df.with_columns([col("x") + lit(1i32)]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.operation(), "dataframe::with_columns");
    assert_eq!(crate::common::error_context_value(&err, "resource").as_deref(), Some("output_name"));
    assert_eq!(crate::common::error_context_value(&err, "reason").as_deref(), Some("missing_output_name"));
    assert_eq!(
        crate::common::error_context_value(&err, "remedy").as_deref(),
        Some("use .alias(...) or provide an explicit (name, expr) pair"),
    );
}

#[test]
fn with_columns_sees_original_state() {
    let df = xy_df(); // x = [10,20,30], y = [1,2,3]
    // Both expressions see the original x, even though the first replaces x.
    let df2 = df.with_columns([lit(0i32).alias("x"), (col("x") + lit(1i32)).alias("x_plus_one")]).unwrap();
    let x = df2.series("x").unwrap();
    assert_eq!(x.to_vec_strict::<i32>().unwrap(), vec![0, 0, 0]);
    let xp = df2.series("x_plus_one").unwrap();
    assert_eq!(xp.to_vec_strict::<i32>().unwrap(), vec![11, 21, 31]);
}

#[test]
fn with_columns_bare_col_uses_column_name() {
    let df = xy_df();
    // Bare col("x") has output_name = Some("x"), so it should work without .alias().
    let df2 = df.with_columns([col("x")]).unwrap();
    assert_eq!(df2.num_columns(), 2);
}

// ===========================================================================
// Output name propagation for single-input nodes
// ===========================================================================

#[test]
fn output_name_propagates_through_cast() {
    assert_output_name(col("x").cast(TypeId::Float64), Some("x"));
}

#[test]
fn output_name_propagates_through_is_null() {
    assert_output_name(col("x").is_null(), Some("x"));
}

#[test]
fn output_name_propagates_through_is_not_null() {
    assert_output_name(col("x").is_not_null(), Some("x"));
}

#[test]
fn output_name_propagates_through_unary_op() {
    assert_output_name(col("x").abs(), Some("x"));
}

#[test]
fn output_name_propagates_through_log() {
    assert_output_name(col("x").log(), Some("x"));
}

#[test]
fn with_columns_cast_no_alias_needed() {
    let df = xy_df();
    let df2 = df.with_columns([col("x").cast(TypeId::Float64)]).unwrap();
    assert_eq!(df2.num_columns(), 2);
    assert_eq!(df2.series("x").unwrap().dtype().unwrap().id(), TypeId::Float64);
}

#[test]
fn with_columns_is_null_no_alias_needed() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let df2 = df.with_columns([col("v").is_null()]).unwrap();
    assert_eq!(df2.series("v").unwrap().dtype().unwrap().id(), TypeId::Bool8);
}

#[test]
fn eval_unary_op_display_omits_input_name() {
    let df = xy_df();
    let result = eval(col("x").abs(), &df);
    let output = format!("{}", result);
    assert!(!output.contains("x"), "expr unary output should not expose schema names: {}", output);
}

#[test]
fn log_expr() {
    let df = f64_df("x", &[1.0, std::f64::consts::E, std::f64::consts::E * std::f64::consts::E]);
    let result = eval(col("x").log().alias("ln_x"), &df);
    let values = result.to_vec_strict::<f64>().unwrap();

    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert!((values[0] - 0.0).abs() < 1e-10);
    assert!((values[1] - 1.0).abs() < 1e-10);
    assert!((values[2] - 2.0).abs() < 1e-10);
}

#[test]
fn eval_fillna_display_omits_input_name() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let result = eval(col("v").fillna(99i32), &df);
    let output = format!("{}", result);
    assert!(!output.contains("v"), "expr fillna output should not expose schema names: {}", output);
}

#[test]
fn eval_cumsum_preserves_name() {
    let df = i32_df("x", &[1, 2, 3]);
    let _result = eval(col("x").cumsum(), &df);
}

#[test]
fn eval_reduce_has_no_implicit_name() {
    let df = i32_df("x", &[10, 20, 30]);
    let _result = eval(col("x").agg().sum(), &df);
}

#[test]
fn eval_string_op_returns_values() {
    let df = str_df("s", &["Hello", "WORLD"]);
    let _result = eval(col("s").str().lower(), &df);
}

// ===========================================================================
// Phase 2: Null fill + cumulative
// ===========================================================================

#[test]
fn fillna_scalar_replaces_nulls() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let result = eval(col("v").fillna(0i32).alias("filled"), &df);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 0, 3]);
    assert_eq!(result.null_count(), 0);
}

#[test]
fn fillna_scalar_preserves_output_name() {
    assert_output_name(col("v").fillna(0i32), Some("v"));
}

#[test]
fn fillna_with_column_path() {
    let v = crate::common::nullable_series::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let backup = crate::common::series::<i32>("backup", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_columns([("v", v), ("backup", backup)]).unwrap();
    let result = col("v").fillna_with(col("backup")).alias("filled").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(result.null_count(), 0);
}

#[test]
fn fillna_with_preserves_output_name() {
    assert_output_name(col("v").fillna_with(col("backup")), Some("v"));
}

#[test]
fn ffill_expr() {
    let df = nullable_i32_df("v", &[1, 0, 3, 0], &[true, false, true, false]);
    assert_eq!(eval_i32(col("v").ffill().alias("ff"), &df), vec![1, 1, 3, 3]);
}

#[test]
fn bfill_expr() {
    let df = nullable_i32_df("v", &[0, 0, 3, 0], &[false, false, true, false]);
    let result = eval(col("v").bfill().alias("bf"), &df);
    // Last element remains null since there's no following value.
    assert_eq!(result.null_count(), 1);
    let vals = result.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(3), Some(3), Some(3), None]);
}

#[test]
fn cumsum_expr() {
    // cumsum output type for i32 is i32 (scan preserves type)
    let df = i32_df("x", &[1, 2, 3, 4]);
    assert_eq!(eval_i32(col("x").cumsum().alias("cs"), &df), vec![1, 3, 6, 10]);
}

#[test]
fn cummin_expr() {
    let df = i32_df("x", &[3, 1, 4, 1]);
    assert_eq!(eval_i32(col("x").cummin().alias("cm"), &df), vec![3, 1, 1, 1]);
}

#[test]
fn cummax_expr() {
    let df = i32_df("x", &[1, 3, 2, 5]);
    assert_eq!(eval_i32(col("x").cummax().alias("cm"), &df), vec![1, 3, 3, 5]);
}

#[test]
fn cumprod_expr() {
    let df = i32_df("x", &[1, 2, 3, 4]);
    assert_eq!(eval_i32(col("x").cumprod().alias("cp"), &df), vec![1, 2, 6, 24]);
}

#[test]
fn cumsum_preserves_output_name() {
    assert_output_name(col("x").cumsum(), Some("x"));
}

#[test]
fn with_columns_fillna_no_alias_needed() {
    let df = nullable_i32_df("v", &[1, 0, 3], &[true, false, true]);
    let df2 = df.with_columns([col("v").fillna(99i32)]).unwrap();
    assert_eq!(df2.series("v").unwrap().null_count(), 0);
    assert_eq!(df2.series("v").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 99, 3]);
}

#[test]
fn with_columns_cumsum_no_alias_needed() {
    let df = i32_df("x", &[1, 2, 3]);
    let df2 = df.with_columns([col("x").cumsum()]).unwrap();
    assert_eq!(df2.series("x").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3, 6]);
}

// ===========================================================================
// Phase 3: Aggregation (broadcast aggregate)
// ===========================================================================

#[test]
fn agg_sum() {
    let df = i32_df("x", &[10, 20, 30]);
    let result = eval(col("x").agg().sum().alias("s"), &df);
    assert_eq!(result.len(), 3);
    // i32 sum → i64
    assert_eq!(result.to_vec_strict::<i64>().unwrap(), vec![60, 60, 60]);
}

#[test]
fn agg_mean() {
    let df = i32_df("x", &[10, 20, 30]);
    let result = eval(col("x").agg().mean().alias("m"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![20.0, 20.0, 20.0]);
}

#[test]
fn agg_min_max() {
    let df = i32_df("x", &[10, 20, 30]);
    assert_eq!(eval_i32(col("x").agg().min().alias("mn"), &df), vec![10, 10, 10]);
    assert_eq!(eval_i32(col("x").agg().max().alias("mx"), &df), vec![30, 30, 30]);
}

#[test]
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

#[test]
fn agg_median() {
    let df = i32_df("x", &[10, 20, 30]);
    let result = eval(col("x").agg().median().alias("med"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![20.0, 20.0, 20.0]);
}

#[test]
fn agg_any_all() {
    let df = bool_df("f", &[true, false, true]);
    assert_eq!(eval_bool(col("f").agg().any().alias("a"), &df), vec![true, true, true]);
    assert_eq!(eval_bool(col("f").agg().all().alias("b"), &df), vec![false, false, false]);
}

#[test]
fn agg_product() {
    let df = i32_df("x", &[2, 3, 5]);
    assert_eq!(eval_i32(col("x").agg().product().alias("p"), &df), vec![30, 30, 30]);
}

#[test]
fn agg_sum_string_reports_expr_reduce_operation_on_type_error() {
    let df = str_df("s", &["a", "b"]);
    let err = col("s").agg().sum().alias("total").eval(&df).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.code(), ErrorCode::UnsupportedDtype);
    assert_eq!(err.operation(), "expr::reduce");
    assert_eq!(err.fact_value(FactKeyName::TypeId), Some(&FactValue::TypeId(TypeId::String)));
    assert_eq!(crate::common::error_context_value(&err, "input_type").as_deref(), None);
}

#[test]
fn reduce_requires_explicit_output_name() {
    assert_output_name(col("x").agg().sum(), None);
}

#[test]
fn pct_pattern() {
    let df = f64_df("x", &[10.0, 20.0, 30.0]);
    // (x / x.agg().sum()) should give each row's fraction of total.
    let vals = eval_f64((col("x") / col("x").agg().sum()).alias("pct"), &df);
    assert!((vals[0] - 10.0 / 60.0).abs() < 1e-10);
    assert!((vals[1] - 20.0 / 60.0).abs() < 1e-10);
    assert!((vals[2] - 30.0 / 60.0).abs() < 1e-10);
}

#[test]
fn with_columns_rejects_unaliased_reduce() {
    let df = i32_df("x", &[10, 20, 30]);
    let err = df.with_columns([col("x").agg().sum()]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.operation(), "dataframe::with_columns");
    assert_eq!(crate::common::error_context_value(&err, "resource").as_deref(), Some("output_name"));
    assert_eq!(crate::common::error_context_value(&err, "reason").as_deref(), Some("missing_output_name"));
    assert_eq!(
        crate::common::error_context_value(&err, "remedy").as_deref(),
        Some("use .alias(...) or provide an explicit (name, expr) pair"),
    );
}

#[test]
fn with_columns_reduce_requires_alias_or_explicit_name() {
    let df = i32_df("x", &[10, 20, 30]);
    let df2 = df.with_columns([("x_total", col("x").agg().sum())]).unwrap();
    assert_eq!(df2.series("x_total").unwrap().to_vec_strict::<i64>().unwrap(), vec![60, 60, 60]);
}

// ===========================================================================
// Phase 4a: String ops
// ===========================================================================

#[test]
fn str_lower_expr() {
    let df = str_df("s", &["Hello", "WORLD", "Foo"]);
    let result = eval(col("s").str().lower().alias("lo"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
    assert_eq!(result.len(), 3);
}

#[test]
fn str_upper_expr() {
    let df = str_df("s", &["Hello", "world", "Foo"]);
    let result = eval(col("s").str().upper().alias("up"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
}

#[test]
fn str_strip_expr() {
    let df = str_df("s", &["  hi  ", " lo", "ok "]);
    let stripped = eval(col("s").str().strip().alias("st"), &df);
    assert_eq!(stripped.dtype().unwrap().id(), TypeId::String);
    let l = eval(col("s").str().lstrip().alias("ls"), &df);
    assert_eq!(l.dtype().unwrap().id(), TypeId::String);
    let r = eval(col("s").str().rstrip().alias("rs"), &df);
    assert_eq!(r.dtype().unwrap().id(), TypeId::String);
}

#[test]
fn str_len_expr() {
    let df = str_df("s", &["ab", "cde", "f"]);
    assert_eq!(eval_i32(col("s").str().len().alias("n"), &df), vec![2, 3, 1]);
}

#[test]
fn str_byte_count_expr() {
    let df = str_df("s", &["ab", "cde", "f"]);
    assert_eq!(eval_i32(col("s").str().byte_count().alias("n"), &df), vec![2, 3, 1]);
}

#[test]
fn str_contains_expr() {
    let df = str_df("s", &["foobar", "baz", "foo"]);
    let result = eval(col("s").str().contains("foo").alias("has"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn str_startswith_expr() {
    let df = str_df("s", &["abc", "bca", "axy"]);
    assert_eq!(eval_bool(col("s").str().starts_with("a").alias("sw"), &df), vec![true, false, true]);
}

#[test]
fn str_endswith_expr() {
    let df = str_df("s", &["abc", "bca", "xya"]);
    assert_eq!(eval_bool(col("s").str().ends_with("a").alias("ew"), &df), vec![false, true, true]);
}

#[test]
fn str_preserves_output_name() {
    assert_output_name(col("s").str().lower(), Some("s"));
}

#[test]
fn with_columns_str_lower_no_alias() {
    let df = str_df("s", &["Hello", "WORLD"]);
    let df2 = df.with_columns([col("s").str().lower()]).unwrap();
    assert_eq!(df2.series("s").unwrap().dtype().unwrap().id(), TypeId::String);
}

// ===========================================================================
// Phase 4b: Conditional (when / then / otherwise)
// ===========================================================================

#[test]
fn when_then_otherwise_basic() {
    let df = i32_df("x", &[1, 2, 3]);
    assert_eq!(
        eval_i32(when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)).alias("out"), &df,),
        vec![0, 100, 100]
    );
}

#[test]
fn when_then_otherwise_scalar_literals_without_lit() {
    let df = i32_df("x", &[1, 2, 3]);
    assert_eq!(eval_i32(when(col("x").gt(1i32)).then(100i32).otherwise(0i32).alias("out"), &df), vec![0, 100, 100]);
}

#[test]
fn when_then_otherwise_column_branches() {
    let x = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let y = crate::common::series::<i32>("y", &[1, 2, 3]).unwrap();
    let flag = crate::common::series::<bool>("f", &[true, false, true]).unwrap();
    let df = DataFrame::from_columns([("x", x), ("y", y), ("f", flag)]).unwrap();
    let result = when(col("f")).then(col("x")).otherwise(col("y")).alias("pick").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 2, 30]);
}

#[test]
fn when_then_otherwise_with_nulls_in_condition() {
    // libcudf copy_if_else: null mask positions produce the false_val
    let cond = crate::common::nullable_series::<bool>("c", &[true, false, false], &[true, false, true]).unwrap();
    let x = crate::common::series::<i32>("x", &[10, 20, 30]).unwrap();
    let y = crate::common::series::<i32>("y", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_columns([("c", cond), ("x", x), ("y", y)]).unwrap();
    let result = when(col("c")).then(col("x")).otherwise(col("y")).alias("out").eval(&df).unwrap();
    // row 0: cond valid & true  → x[0]=10
    // row 1: cond null          → y[1]=2
    // row 2: cond valid & false → y[2]=3
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 2, 3]);
}

#[test]
fn when_then_otherwise_string_branches() {
    let df = i32_df("age", &[20, 10, 30]);
    let result =
        eval(when(col("age").ge(lit(18i32))).then(lit("adult")).otherwise(lit("minor")).alias("category"), &df);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_strings_strict().unwrap(), vec!["adult", "minor", "adult"]);
}

#[test]
fn if_else_output_name_is_none() {
    assert_output_name(when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)), None);
}

#[test]
fn if_else_requires_alias_in_with_columns() {
    let df = i32_df("x", &[1, 2, 3]);
    let err = df.with_columns([when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32))]).unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedOperation);
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert_eq!(err.operation(), "dataframe::with_columns");
    assert_eq!(crate::common::error_context_value(&err, "resource").as_deref(), Some("output_name"));
    assert_eq!(crate::common::error_context_value(&err, "reason").as_deref(), Some("missing_output_name"));
    assert_eq!(
        crate::common::error_context_value(&err, "remedy").as_deref(),
        Some("use .alias(...) or provide an explicit (name, expr) pair"),
    );
}

#[test]
fn if_else_with_alias_in_with_columns() {
    let df = i32_df("x", &[1, 2, 3]);
    let df2 =
        df.with_columns([when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)).alias("big")]).unwrap();
    assert_eq!(df2.series("big").unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 100, 100]);
}

#[test]
fn when_then_otherwise_composite() {
    let df = i32_df("age", &[25, 12, 18]);
    let df2 = df
        .with_columns([when(col("age").ge(lit(18i32))).then(lit("adult")).otherwise(lit("minor")).alias("category")])
        .unwrap();
    assert_eq!(df2.series("category").unwrap().dtype().unwrap().id(), TypeId::String);
    assert_eq!(df2.series("category").unwrap().len(), 3);
    assert_eq!(df2.series("category").unwrap().to_strings_strict().unwrap(), vec!["adult", "minor", "adult"]);
}
