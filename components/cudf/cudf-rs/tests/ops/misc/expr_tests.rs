//! Integration tests for the expression layer.
//!
//! Tests cover: `col()` / `lit()`, arithmetic operators, comparisons,
//! unary ops (`is_null`, `is_not_null`, `cast`, `Not`), `.alias()`,
//! and `DataFrame::with_columns`.
//!
//! All tests require a GPU.

use cudf::{
    DataFrame, ErrorKind, Series, TypeId,
    ops::expr::{col, lit},
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a simple 3-row DataFrame with columns "x" (i32) and "y" (i32).
fn xy_df() -> DataFrame {
    let x = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
    let y = Series::new::<i32>("y", &[1, 2, 3]).unwrap();
    DataFrame::from_series([("x", x), ("y", y)]).unwrap()
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
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
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
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![42, 42, 42]);
}

#[test]
fn lit_f64_broadcasts() {
    let df = xy_df();
    let result = lit(3.25f64).alias("pi").eval(&df).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![3.25, 3.25, 3.25]);
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
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

// ===========================================================================
// Arithmetic operators
// ===========================================================================

#[test]
fn add_col_col() {
    let df = xy_df();
    let result = (col("x") + col("y")).alias("sum").eval(&df).unwrap();
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn add_col_lit() {
    let df = xy_df();
    let result = (col("x") + lit(100i32)).alias("plus").eval(&df).unwrap();
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![110, 120, 130]);
}

#[test]
fn compound_expr_add_mul() {
    let df = xy_df();
    // (x + y) * lit(2)
    let result = ((col("x") + col("y")) * lit(2i32)).alias("r").eval(&df).unwrap();
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![22, 44, 66]);
}

#[test]
fn sub_col_col() {
    let df = xy_df();
    let result = (col("x") - col("y")).alias("diff").eval(&df).unwrap();
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![9, 18, 27]);
}

#[test]
fn mul_col_col() {
    let df = xy_df();
    let result = (col("x") * col("y")).alias("prod").eval(&df).unwrap();
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![10, 40, 90]);
}

#[test]
fn div_col_col() {
    let df = xy_df();
    let result = (col("x") / col("y")).alias("quot").eval(&df).unwrap();
    // cuDF integer division: 10/1=10, 20/2=10, 30/3=10
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![10, 10, 10]);
}

// ===========================================================================
// Comparisons
// ===========================================================================

#[test]
fn gt_lit_produces_bool() {
    let df = xy_df();
    let result = col("x").gt(lit(15i32)).alias("big").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn eq_col_col() {
    let x = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let y = Series::new::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_series([("a", x), ("b", y)]).unwrap();
    let result = col("a").eq(col("b")).alias("same").eval(&df).unwrap();
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn lt_lit_with_alias() {
    let df = xy_df();
    let result = col("x").lt(lit(25i32)).alias("small").eval(&df).unwrap();
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn ne_col_col() {
    let x = Series::new::<i32>("a", &[1, 2, 3]).unwrap();
    let y = Series::new::<i32>("b", &[1, 99, 3]).unwrap();
    let df = DataFrame::from_series([("a", x), ("b", y)]).unwrap();
    let result = col("a").ne(col("b")).alias("diff").eval(&df).unwrap();
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn ge_and_le() {
    let df = xy_df();
    let ge = col("x").ge(lit(20i32)).alias("ge20").eval(&df).unwrap();
    assert_eq!(ge.to_vec::<bool>().unwrap(), vec![false, true, true]);
    let le = col("x").le(lit(20i32)).alias("le20").eval(&df).unwrap();
    assert_eq!(le.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

// ===========================================================================
// Unary ops
// ===========================================================================

#[test]
fn is_null_expr() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([("v", s)]).unwrap();
    let result = col("v").is_null().alias("nulls").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn is_not_null_expr() {
    let s = Series::new_nullable::<i32>("v", &[1, 0, 3], &[true, false, true]).unwrap();
    let df = DataFrame::from_series([("v", s)]).unwrap();
    let result = col("v").is_not_null().alias("valid").eval(&df).unwrap();
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn cast_i32_to_f64() {
    let df = xy_df();
    let result = col("x").cast(TypeId::Float64).alias("xf").eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn not_operator() {
    let s = Series::new::<bool>("mask", &[true, false, true]).unwrap();
    let df = DataFrame::from_series([("mask", s)]).unwrap();
    let result = (!col("mask")).alias("inv").eval(&df).unwrap();
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![false, true, false]);
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
    assert_eq!(sum.to_vec::<i32>().unwrap(), vec![11, 22, 33]);
}

#[test]
fn with_columns_replaces_existing() {
    let df = xy_df();
    let df2 = df.with_columns([(col("x") * lit(2i32)).alias("x")]).unwrap();
    assert_eq!(df2.num_columns(), 2);
    let x = df2.series("x").unwrap();
    assert_eq!(x.to_vec::<i32>().unwrap(), vec![20, 40, 60]);
}

#[test]
fn with_columns_multiple_exprs() {
    let df = xy_df();
    let df2 = df.with_columns([(col("x") + lit(1i32)).alias("x1"), (col("y") * lit(10i32)).alias("y10")]).unwrap();
    assert_eq!(df2.num_columns(), 4);
    assert_eq!(df2.series("x1").unwrap().to_vec::<i32>().unwrap(), vec![11, 21, 31]);
    assert_eq!(df2.series("y10").unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
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
    assert_eq!(x.to_vec::<i32>().unwrap(), vec![0, 0, 0]);
    let xp = df2.series("x_plus_one").unwrap();
    assert_eq!(xp.to_vec::<i32>().unwrap(), vec![11, 21, 31]);
}

#[test]
fn with_columns_bare_col_uses_column_name() {
    let df = xy_df();
    // Bare col("x") has output_name = Some("x"), so it should work without .alias().
    let df2 = df.with_columns([col("x")]).unwrap();
    assert_eq!(df2.num_columns(), 2);
}
