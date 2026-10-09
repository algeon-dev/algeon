//! Cast API consistency tests.
//!
//! Verifies that all cast entry points accept `DataType` (the canonical type),
//! while `TypeId` remains accepted via `Into<DataType>` for convenience.

use cudf::{
    DataFrame, DataType, Exec, TypeId,
    ops::{expr::col, unary::is_supported_cast},
};

use crate::common::series;

// ===========================================================================
// Series::astype; accepts DataType directly
// ===========================================================================

fn series_astype_accepts_datatype() {
    let s = series::<i32>("x", &[100, 200, 300]).unwrap();
    let dt = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();
    let result = s.on(Exec::RETAINED).astype(dt).unwrap();
    assert_eq!(result.dtype().unwrap(), dt);
}

fn series_astype_accepts_typeid() {
    // Backward-compat: TypeId still works via Into<DataType>.
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let result = s.on(Exec::RETAINED).astype(TypeId::Float64).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
}

// ===========================================================================
// Expr::cast; accepts DataType directly
// ===========================================================================

fn expr_cast_accepts_datatype() {
    let dt = DataType::fixed_point(TypeId::Decimal128, -2).unwrap();
    let expr = col("x").cast(dt).alias("x");
    let s = series::<i32>("x", &[100, 200, 300]).unwrap();
    let df = DataFrame::from_series([("x", s)]).unwrap();
    let result = expr.eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap(), dt);
}

fn expr_cast_accepts_typeid() {
    // Backward-compat: TypeId still works via Into<DataType>.
    let expr = col("x").cast(TypeId::Float64).alias("xf");
    let s = series::<i32>("x", &[10, 20, 30]).unwrap();
    let df = DataFrame::from_series([("x", s)]).unwrap();
    let result = expr.eval(&df).unwrap();
    assert_eq!(result.dtype().unwrap().id(), TypeId::Float64);
    assert_eq!(result.to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ===========================================================================
// Symmetry: series.astype(dt) == df.cast_column(col, dt)
// ===========================================================================

fn cast_symmetry() {
    let dt = DataType::new(TypeId::Float64);
    let s = series::<i32>("x", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([("x", s.clone())]).unwrap();

    let via_series = s.on(Exec::RETAINED).astype(dt).unwrap();
    let via_df = df.on(Exec::RETAINED).cast_column("x", dt).unwrap();
    let via_df_col = via_df.series("x").unwrap();

    assert_eq!(via_series.dtype().unwrap(), via_df_col.dtype().unwrap());
    assert_eq!(via_series.len(), via_df_col.len());
}

// ===========================================================================
// unary::is_supported_cast; already uses DataType (no change needed)
// ===========================================================================

fn is_supported_cast_uses_datatype() {
    let from = DataType::new(TypeId::Int32);
    let to = DataType::new(TypeId::Float64);
    let supported = is_supported_cast(&from, &to).unwrap();
    assert!(supported);
}

// ===========================================================================
// From<TypeId> for DataType
// ===========================================================================

fn typeid_converts_to_datatype() {
    let dt: DataType = TypeId::Float64.into();
    assert_eq!(dt.id(), TypeId::Float64);
    assert_eq!(dt.scale(), 0);
}

fn typeid_converts_to_datatype_decimal() {
    // From<TypeId> gives scale 0; use DataType::fixed_point for explicit scale.
    let dt: DataType = TypeId::Decimal128.into();
    assert_eq!(dt.id(), TypeId::Decimal128);
    assert_eq!(dt.scale(), 0);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("cast_consistency_tests::series_astype_accepts_datatype", series_astype_accepts_datatype),
    ("cast_consistency_tests::series_astype_accepts_typeid", series_astype_accepts_typeid),
    ("cast_consistency_tests::expr_cast_accepts_datatype", expr_cast_accepts_datatype),
    ("cast_consistency_tests::expr_cast_accepts_typeid", expr_cast_accepts_typeid),
    ("cast_consistency_tests::cast_symmetry", cast_symmetry),
    ("cast_consistency_tests::is_supported_cast_uses_datatype", is_supported_cast_uses_datatype),
    ("cast_consistency_tests::typeid_converts_to_datatype", typeid_converts_to_datatype),
    ("cast_consistency_tests::typeid_converts_to_datatype_decimal", typeid_converts_to_datatype_decimal),
];
