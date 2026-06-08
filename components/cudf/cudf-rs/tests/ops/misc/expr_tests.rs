//! Integration tests for the expression layer.
//!
//! Tests cover: `col()` / `lit()`, arithmetic operators, comparisons,
//! unary ops (`is_null`, `is_not_null`, `cast`, `Not`), `.alias()`,
//! `DataFrame::with_columns`, null fill, cumulative, aggregation, string ops,
//! and conditional (`when`/`then`/`otherwise`).
//!
//! All tests require a GPU.

use cudf::{
    DataFrame, ErrorKind, Series, TypeId,
    ops::expr::{col, lit, when},
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a simple 3-row DataFrame with columns "x" (i32) and "y" (i32).
fn xy_df() -> DataFrame {
    let x = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let y = Series::new::<i32>("y", &[1, 2, 3]).unwrap();
    DataFrame::from_series([x, y]).unwrap()
}

// ===========================================================================
// col / lit basics
// ===========================================================================

#[test]
fn col_resolves_existing_column() {
    let df = xy_df();
    let result = col("x").eval(&df).unwrap();
    assert_eq!(result.name(), Some("x"));
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn col_missing_column_errors() {
    let df = xy_df();
    let err = col("zzz").eval(&df).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn lit_i32_broadcasts() {
    let df = xy_df();
    let result = lit(42i32).alias("c").eval(&df).unwrap();
    assert_eq!(result.name(), Some("c"));
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![42, 42, 42]);
}

#[test]
fn lit_f64_broadcasts() {
    let df = xy_df();
    let result = lit(3.25f64).alias("pi").eval(&df).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![3.25, 3.25, 3.25]);
}

#[test]
fn lit_string_broadcasts() {
    let df = xy_df();
    let result = lit("hello").alias("s").eval(&df).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
}

#[test]
fn alias_renames_output() {
    let df = xy_df();
    let result = col("x").alias("renamed").eval(&df).unwrap();
    assert_eq!(result.name(), Some("renamed"));
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
}

// ===========================================================================
// Arithmetic operators
// ===========================================================================

#[test]
fn add_col_col() {
    let df = xy_df();
    let result = (col("x") + col("y")).alias("sum").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn add_col_lit() {
    let df = xy_df();
    let result = (col("x") + lit(100i32)).alias("plus").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![110, 120, 130]);
}

#[test]
fn add_col_scalar_without_lit() {
    let df = xy_df();
    let result = (col("x") + 100i32).alias("plus").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![110, 120, 130]);
}

#[test]
fn add_scalar_lhs_without_lit() {
    let df = xy_df();
    let result = (100i32 + col("x")).alias("plus").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![110, 120, 130]);
}

#[test]
fn compound_expr_add_mul() {
    let df = xy_df();
    // (x + y) * lit(2)
    let result = ((col("x") + col("y")) * lit(2i32)).alias("r").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![22, 44, 66]);
}

#[test]
fn sub_col_col() {
    let df = xy_df();
    let result = (col("x") - col("y")).alias("diff").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![9, 18, 27]);
}

#[test]
fn mul_col_col() {
    let df = xy_df();
    let result = (col("x") * col("y")).alias("prod").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 40, 90]);
}

#[test]
fn div_col_col() {
    let df = xy_df();
    let result = (col("x") / col("y")).alias("quot").eval(&df).unwrap();
    // cuDF integer division: 10/1=10, 20/2=10, 30/3=10
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 10, 10]);
}

#[test]
fn rem_col_lit() {
    let df = xy_df(); // x = [10, 20, 30]
    let result = (col("x") % lit(3i32)).alias("mod3").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 0]);
}

#[test]
fn neg_expr() {
    let df = xy_df(); // x = [10, 20, 30]
    let result = (-col("x")).alias("neg").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![-10, -20, -30]);
}

#[test]
fn bitand_bool() {
    let a = Series::new::<bool>("a", &[true, true, false, false]).unwrap();
    let b = Series::new::<bool>("b", &[true, false, true, false]).unwrap();
    let df = DataFrame::from_series([a, b]).unwrap();
    let result = (col("a") & col("b")).alias("and").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, false, false]);
}

#[test]
fn bitand_bool_scalar_lhs_without_lit() {
    let b = Series::new::<bool>("b", &[true, false, true, false]).unwrap();
    let df = DataFrame::from_series([b]).unwrap();
    let result = (true & col("b")).alias("and").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true, false]);
}

#[test]
fn bitor_bool() {
    let a = Series::new::<bool>("a", &[true, true, false, false]).unwrap();
    let b = Series::new::<bool>("b", &[true, false, true, false]).unwrap();
    let df = DataFrame::from_series([a, b]).unwrap();
    let result = (col("a") | col("b")).alias("or").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, true, false]);
}

#[test]
fn pow_builder() {
    let s = Series::new::<f64>("x", &[2.0, 3.0, 4.0]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").pow(lit(2.0f64)).alias("sq").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![4.0, 9.0, 16.0]);
}

#[test]
fn pow_builder_scalar_without_lit() {
    let s = Series::new::<f64>("x", &[2.0, 3.0, 4.0]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").pow(2.0f64).alias("sq").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![4.0, 9.0, 16.0]);
}

#[test]
fn floor_div_builder() {
    let df = xy_df(); // x = [10, 20, 30]
    let result = col("x").floor_div(lit(3i32)).alias("fd").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![3, 6, 10]);
}

#[test]
fn abs_builder() {
    let s = Series::new::<i32>("v", &[-5, 0, 7]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").abs().alias("a").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![5, 0, 7]);
}

#[test]
fn sqrt_builder() {
    let s = Series::new::<f64>("v", &[4.0, 9.0, 16.0]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").sqrt().alias("r").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn ceil_floor_builder() {
    let s = Series::new::<f64>("v", &[1.2, 2.7, -0.5]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let c = col("v").ceil().alias("c").eval(&df).unwrap();
    assert_eq!(c.to_vec_strict::<f64>().unwrap(), vec![2.0, 3.0, 0.0]);
    let f = col("v").floor().alias("f").eval(&df).unwrap();
    assert_eq!(f.to_vec_strict::<f64>().unwrap(), vec![1.0, 2.0, -1.0]);
}

// ===========================================================================
// Comparisons
// ===========================================================================

#[test]
fn gt_lit_produces_bool() {
    let df = xy_df();
    let result = col("x").gt(lit(15i32)).alias("big").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn gt_scalar_without_lit_produces_bool() {
    let df = xy_df();
    let result = col("x").gt(15i32).alias("big").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn eq_col_col() {
    let x = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let y = Series::new::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_series([x, y]).unwrap();
    let result = col("a").eq(col("b")).alias("same").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn lt_lit_with_alias() {
    let df = xy_df();
    let result = col("x").lt(lit(25i32)).alias("small").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn ne_col_col() {
    let x = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let y = Series::new::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_series([x, y]).unwrap();
    let result = col("a").ne(col("b")).alias("diff").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn ge_and_le() {
    let df = xy_df();
    let ge = col("x").ge(lit(20i32)).alias("ge20").eval(&df).unwrap();
    assert_eq!(ge.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
    let le = col("x").le(lit(20i32)).alias("le20").eval(&df).unwrap();
    assert_eq!(le.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

// ===========================================================================
// Unary ops
// ===========================================================================

#[test]
fn is_null_expr() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").is_null().alias("nulls").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn is_not_null_expr() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").is_not_null().alias("valid").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn cast_i32_to_f64() {
    let df = xy_df();
    let result = col("x").cast(TypeId::Float64).alias("xf").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn not_operator() {
    let s = Series::new::<bool>("mask", &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = (!col("mask")).alias("inv").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
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
    let x = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([x]).unwrap();
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
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
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
    let expr = col("x").cast(TypeId::Float64);
    assert_eq!(expr.output_name(), Some("x"));
}

#[test]
fn output_name_propagates_through_is_null() {
    let expr = col("x").is_null();
    assert_eq!(expr.output_name(), Some("x"));
}

#[test]
fn output_name_propagates_through_is_not_null() {
    let expr = col("x").is_not_null();
    assert_eq!(expr.output_name(), Some("x"));
}

#[test]
fn output_name_propagates_through_unary_op() {
    let expr = col("x").abs();
    assert_eq!(expr.output_name(), Some("x"));
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
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let df2 = df.with_columns([col("v").is_null()]).unwrap();
    assert_eq!(df2.series("v").unwrap().dtype().unwrap().id(), TypeId::Bool8);
}

#[test]
fn eval_unary_op_preserves_name() {
    let df = xy_df();
    let result = col("x").abs().eval(&df).unwrap();
    assert_eq!(result.name(), Some("x"));
}

#[test]
fn eval_fillna_preserves_name() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").fillna(99i32).eval(&df).unwrap();
    assert_eq!(result.name(), Some("v"));
}

#[test]
fn eval_cumsum_preserves_name() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").cumsum().eval(&df).unwrap();
    assert_eq!(result.name(), Some("x"));
}

#[test]
fn eval_reduce_has_no_implicit_name() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").agg().sum().eval(&df).unwrap();
    assert_eq!(result.name(), None);
}

#[test]
fn eval_string_op_preserves_name() {
    let s = Series::from_strings("s", &["Hello", "WORLD"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().lower().eval(&df).unwrap();
    assert_eq!(result.name(), Some("s"));
}

// ===========================================================================
// Phase 2: Null fill + cumulative
// ===========================================================================

#[test]
fn fillna_scalar_replaces_nulls() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").fillna(0i32).alias("filled").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 0, 3]);
    assert_eq!(result.null_count(), 0);
}

#[test]
fn fillna_scalar_preserves_output_name() {
    let expr = col("v").fillna(0i32);
    assert_eq!(expr.output_name(), Some("v"));
}

#[test]
fn fillna_with_column_path() {
    let v = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let backup = Series::new::<i32>("backup", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([v, backup]).unwrap();
    let result = col("v").fillna_with(col("backup")).alias("filled").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 20, 3]);
    assert_eq!(result.null_count(), 0);
}

#[test]
fn fillna_with_preserves_output_name() {
    let expr = col("v").fillna_with(col("backup"));
    assert_eq!(expr.output_name(), Some("v"));
}

#[test]
fn ffill_expr() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3, 0], &[true, false, true, false]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").ffill().alias("ff").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 1, 3, 3]);
}

#[test]
fn bfill_expr() {
    let s = Series::new_nullable::<i32>("v", &[0, 0, 3, 0], &[false, false, true, false]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("v").bfill().alias("bf").eval(&df).unwrap();
    // Last element remains null since there's no following value.
    assert_eq!(result.null_count(), 1);
    let vals = result.to_vec_optional::<i32>().unwrap();
    assert_eq!(vals, vec![Some(3), Some(3), Some(3), None]);
}

#[test]
fn cumsum_expr() {
    let s = Series::new::<i32>("x", &[1, 2, 3, 4]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").cumsum().alias("cs").eval(&df).unwrap();
    // cumsum output type for i32 is i32 (scan preserves type)
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 3, 6, 10]);
}

#[test]
fn cummin_expr() {
    let s = Series::new::<i32>("x", &[3, 1, 4, 1]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").cummin().alias("cm").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![3, 1, 1, 1]);
}

#[test]
fn cummax_expr() {
    let s = Series::new::<i32>("x", &[1, 3, 2, 5]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").cummax().alias("cm").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 3, 3, 5]);
}

#[test]
fn cumprod_expr() {
    let s = Series::new::<i32>("x", &[1, 2, 3, 4]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").cumprod().alias("cp").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![1, 2, 6, 24]);
}

#[test]
fn cumsum_preserves_output_name() {
    let expr = col("x").cumsum();
    assert_eq!(expr.output_name(), Some("x"));
}

#[test]
fn with_columns_fillna_no_alias_needed() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let df2 = df.with_columns([col("v").fillna(99i32)]).unwrap();
    assert_eq!(df2.series("v").unwrap().null_count(), 0);
    assert_eq!(df2.series("v").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 99, 3]);
}

#[test]
fn with_columns_cumsum_no_alias_needed() {
    let s = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let df2 = df.with_columns([col("x").cumsum()]).unwrap();
    assert_eq!(df2.series("x").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 3, 6]);
}

// ===========================================================================
// Phase 3: Aggregation (broadcast aggregate)
// ===========================================================================

#[test]
fn agg_sum() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").agg().sum().alias("s").eval(&df).unwrap();
    assert_eq!(result.len(), 3);
    // i32 sum → i64
    assert_eq!(result.to_vec_strict::<i64>().unwrap(), vec![60, 60, 60]);
}

#[test]
fn agg_mean() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").agg().mean().alias("m").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![20.0, 20.0, 20.0]);
}

#[test]
fn agg_min_max() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let mn = col("x").agg().min().alias("mn").eval(&df).unwrap();
    assert_eq!(mn.to_vec_strict::<i32>().unwrap(), vec![10, 10, 10]);
    let mx = col("x").agg().max().alias("mx").eval(&df).unwrap();
    assert_eq!(mx.to_vec_strict::<i32>().unwrap(), vec![30, 30, 30]);
}

#[test]
fn agg_std_var() {
    let s = Series::new::<f64>("x", &[2.0, 4.0, 6.0]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let std_r = col("x").agg().std(1).alias("s").eval(&df).unwrap();
    assert_eq!(std_r.dtype().unwrap().id(), TypeId::Float64);
    let var_r = col("x").agg().var(1).alias("v").eval(&df).unwrap();
    assert_eq!(var_r.dtype().unwrap().id(), TypeId::Float64);
    // var = 4.0, std = 2.0 for [2,4,6] with ddof=1
    let std_vals = std_r.to_vec_strict::<f64>().unwrap();
    assert!((std_vals[0] - 2.0).abs() < 1e-10);
    let var_vals = var_r.to_vec_strict::<f64>().unwrap();
    assert!((var_vals[0] - 4.0).abs() < 1e-10);
}

#[test]
fn agg_median() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").agg().median().alias("med").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![20.0, 20.0, 20.0]);
}

#[test]
fn agg_any_all() {
    let s = Series::new::<bool>("f", &[true, false, true]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let any_r = col("f").agg().any().alias("a").eval(&df).unwrap();
    assert_eq!(any_r.to_vec_strict::<bool>().unwrap(), vec![true, true, true]);
    let all_r = col("f").agg().all().alias("b").eval(&df).unwrap();
    assert_eq!(all_r.to_vec_strict::<bool>().unwrap(), vec![false, false, false]);
}

#[test]
fn agg_product() {
    let s = Series::new::<i32>("x", &[2, 3, 5]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("x").agg().product().alias("p").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![30, 30, 30]);
}

#[test]
fn reduce_requires_explicit_output_name() {
    let expr = col("x").agg().sum();
    assert_eq!(expr.output_name(), None);
}

#[test]
fn pct_pattern() {
    let s = Series::new::<f64>("x", &[10.0, 20.0, 30.0]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    // (x / x.agg().sum()) should give each row's fraction of total.
    let result = (col("x") / col("x").agg().sum()).alias("pct").eval(&df).unwrap();
    let vals = result.to_vec_strict::<f64>().unwrap();
    assert!((vals[0] - 10.0 / 60.0).abs() < 1e-10);
    assert!((vals[1] - 20.0 / 60.0).abs() < 1e-10);
    assert!((vals[2] - 30.0 / 60.0).abs() < 1e-10);
}

#[test]
fn with_columns_rejects_unaliased_reduce() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let err = df.with_columns([col("x").agg().sum()]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn with_columns_reduce_requires_alias_or_explicit_name() {
    let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let df2 = df.with_columns([("x_total", col("x").agg().sum())]).unwrap();
    assert_eq!(df2.series("x_total").unwrap().to_vec_strict::<i64>().unwrap(), vec![60, 60, 60]);
}

// ===========================================================================
// Phase 4a: String ops
// ===========================================================================

#[test]
fn str_lower_expr() {
    let s = Series::from_strings("s", &["Hello", "WORLD", "Foo"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().lower().alias("lo").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
    assert_eq!(result.len(), 3);
}

#[test]
fn str_upper_expr() {
    let s = Series::from_strings("s", &["Hello", "world", "Foo"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().upper().alias("up").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
}

#[test]
fn str_strip_expr() {
    let s = Series::from_strings("s", &["  hi  ", " lo", "ok "]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let stripped = col("s").str().strip().alias("st").eval(&df).unwrap();
    assert_eq!(stripped.dtype().unwrap().id(), TypeId::String);
    let l = col("s").str().lstrip().alias("ls").eval(&df).unwrap();
    assert_eq!(l.dtype().unwrap().id(), TypeId::String);
    let r = col("s").str().rstrip().alias("rs").eval(&df).unwrap();
    assert_eq!(r.dtype().unwrap().id(), TypeId::String);
}

#[test]
fn str_len_expr() {
    let s = Series::from_strings("s", &["ab", "cde", "f"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().len().alias("n").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![2, 3, 1]);
}

#[test]
fn str_byte_count_expr() {
    let s = Series::from_strings("s", &["ab", "cde", "f"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().byte_count().alias("n").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![2, 3, 1]);
}

#[test]
fn str_contains_expr() {
    let s = Series::from_strings("s", &["foobar", "baz", "foo"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().contains("foo").alias("has").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn str_startswith_expr() {
    let s = Series::from_strings("s", &["abc", "bca", "axy"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().starts_with("a").alias("sw").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn str_endswith_expr() {
    let s = Series::from_strings("s", &["abc", "bca", "xya"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let result = col("s").str().ends_with("a").alias("ew").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn str_preserves_output_name() {
    let expr = col("s").str().lower();
    assert_eq!(expr.output_name(), Some("s"));
}

#[test]
fn with_columns_str_lower_no_alias() {
    let s = Series::from_strings("s", &["Hello", "WORLD"]).unwrap();
    let df = DataFrame::from_series([s]).unwrap();
    let df2 = df.with_columns([col("s").str().lower()]).unwrap();
    assert_eq!(df2.series("s").unwrap().dtype().unwrap().id(), TypeId::String);
}

// ===========================================================================
// Phase 4b: Conditional (when / then / otherwise)
// ===========================================================================

#[test]
fn when_then_otherwise_basic() {
    let x = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([x]).unwrap();
    let result = when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)).alias("out").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![0, 100, 100]);
}

#[test]
fn when_then_otherwise_scalar_literals_without_lit() {
    let x = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([x]).unwrap();
    let result = when(col("x").gt(1i32)).then(100i32).otherwise(0i32).alias("out").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![0, 100, 100]);
}

#[test]
fn when_then_otherwise_column_branches() {
    let x = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let y = Series::new::<i32>("y", &[1, 2, 3]).unwrap();
    let flag = Series::new::<bool>("f", &[true, false, true]).unwrap();
    let df = DataFrame::from_series([x, y, flag]).unwrap();
    let result = when(col("f")).then(col("x")).otherwise(col("y")).alias("pick").eval(&df).unwrap();
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 2, 30]);
}

#[test]
fn when_then_otherwise_with_nulls_in_condition() {
    // libcudf copy_if_else: null mask positions produce the false_val
    let cond = Series::new_nullable::<bool>("c", &[true, false, false], &[true, false, true]).unwrap();
    let x = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let y = Series::new::<i32>("y", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([cond, x, y]).unwrap();
    let result = when(col("c")).then(col("x")).otherwise(col("y")).alias("out").eval(&df).unwrap();
    // row 0: cond valid & true  → x[0]=10
    // row 1: cond null          → y[1]=2
    // row 2: cond valid & false → y[2]=3
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 2, 3]);
}

#[test]
fn when_then_otherwise_string_branches() {
    let age = Series::new::<i32>("age", &[20, 10, 30]).unwrap();
    let df = DataFrame::from_series([age]).unwrap();
    let result =
        when(col("age").ge(lit(18i32))).then(lit("adult")).otherwise(lit("minor")).alias("category").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::String);
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_strings_strict().unwrap(), vec!["adult", "minor", "adult"]);
}

#[test]
fn if_else_output_name_is_none() {
    let expr = when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32));
    assert_eq!(expr.output_name(), None);
}

#[test]
fn if_else_requires_alias_in_with_columns() {
    let x = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([x]).unwrap();
    let err = df.with_columns([when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32))]).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn if_else_with_alias_in_with_columns() {
    let x = Series::new::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([x]).unwrap();
    let df2 =
        df.with_columns([when(col("x").gt(lit(1i32))).then(lit(100i32)).otherwise(lit(0i32)).alias("big")]).unwrap();
    assert_eq!(df2.series("big").unwrap().to_vec_strict::<i32>().unwrap(), vec![0, 100, 100]);
}

#[test]
fn when_then_otherwise_composite() {
    let age = Series::new::<i32>("age", &[25, 12, 18]).unwrap();
    let df = DataFrame::from_series([age]).unwrap();
    let df2 = df
        .with_columns([when(col("age").ge(lit(18i32))).then(lit("adult")).otherwise(lit("minor")).alias("category")])
        .unwrap();
    assert_eq!(df2.series("category").unwrap().dtype().unwrap().id(), TypeId::String);
    assert_eq!(df2.series("category").unwrap().len(), 3);
    assert_eq!(df2.series("category").unwrap().to_strings_strict().unwrap(), vec!["adult", "minor", "adult"]);
}
