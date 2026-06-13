//! Integration tests for the unified `Series::*_scalar` API.
//!
//! Verifies that the six generic scalar-compare methods (`eq`, `ne`, `gt`,
//! `lt`, `ge`, `le`) accept both numeric primitives (via `CudfType`) and
//! string slices (`&str`) through the `IntoRuntimeScalar` conversion trait,
//! and that the explicit `Series::*_in(..., &ExecutionOverrides)` API stays in
//! sync with the owned `Series` API.
//!
//! All tests require a GPU.

use cudf::TypeId;

// ===========================================================================
// Primitive path: numeric `CudfType` values still work through the generic API
// ===========================================================================

#[test]
fn eq_scalar_accepts_primitive() {
    let s = crate::common::series::<i32>("z", &[1, 2, 2, 3]).unwrap();
    let mask = s.eq_scalar(2i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true, false]);
}

#[test]
fn ne_scalar_accepts_primitive() {
    let s = crate::common::series::<i32>("z", &[1, 2, 2, 3]).unwrap();
    let mask = s.ne_scalar(2i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, false, true]);
}

#[test]
fn gt_scalar_accepts_primitive() {
    let s = crate::common::series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.gt_scalar(2i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true, true]);
}

#[test]
fn lt_scalar_accepts_primitive() {
    let s = crate::common::series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.lt_scalar(3i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false, false]);
}

#[test]
fn ge_scalar_accepts_primitive() {
    let s = crate::common::series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.ge_scalar(3i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true, true]);
}

#[test]
fn le_scalar_accepts_primitive() {
    let s = crate::common::series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.le_scalar(2i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false, false]);
}

// ===========================================================================
// String path: `&str` literals now work through the generic API
// ===========================================================================

#[test]
fn eq_scalar_accepts_str_literal() {
    let s = crate::common::series("status", &["active", "inactive", "active"]).unwrap();
    let mask = s.eq_scalar("active").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn ne_scalar_accepts_str_literal() {
    let s = crate::common::series("status", &["active", "inactive", "active"]).unwrap();
    let mask = s.ne_scalar("active").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn gt_scalar_accepts_str_literal() {
    let s = crate::common::series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.gt_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true]);
}

#[test]
fn lt_scalar_accepts_str_literal() {
    let s = crate::common::series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.lt_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn ge_scalar_accepts_str_literal() {
    let s = crate::common::series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.ge_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn le_scalar_accepts_str_literal() {
    let s = crate::common::series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.le_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

// ===========================================================================
// Owned `String` and `&String` work too
// ===========================================================================

#[test]
fn eq_scalar_accepts_owned_string() {
    let s = crate::common::series("status", &["active", "inactive", "active"]).unwrap();
    let needle = String::from("active");
    let mask = s.eq_scalar(&needle).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);

    // Now pass owned String by value.
    let mask2 = s.eq_scalar(needle).unwrap();
    assert_eq!(mask2.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

// ===========================================================================
// Explicit execution-context path stays in sync with the owned API
// ===========================================================================

#[test]
fn eq_scalar_in_accepts_str_literal() {
    let s = crate::common::series("status", &["active", "inactive", "active"]).unwrap();
    let ctx = cudf::ExecutionOverrides::new();
    let mask = s.eq_scalar_in("active", &ctx).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn eq_scalar_in_accepts_owned_string() {
    let s = crate::common::series("status", &["active", "inactive", "active"]).unwrap();
    let ctx = cudf::ExecutionOverrides::new();
    let needle = String::from("active");

    let mask = s.eq_scalar_in(needle, &ctx).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn gt_scalar_in_accepts_str_literal() {
    let s = crate::common::series("name", &["alpha", "beta", "gamma"]).unwrap();
    let ctx = cudf::ExecutionOverrides::new();
    let mask = s.gt_scalar_in("beta", &ctx).unwrap();

    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true]);
}

#[test]
fn eq_scalar_in_accepts_primitive() {
    let s = crate::common::series::<i32>("z", &[1, 2, 2, 3]).unwrap();
    let ctx = cudf::ExecutionOverrides::new();
    let mask = s.eq_scalar_in(2i32, &ctx).unwrap();

    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true, false]);
}
