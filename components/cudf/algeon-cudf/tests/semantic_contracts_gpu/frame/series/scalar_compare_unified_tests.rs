//! Verifies that the six generic scalar-compare methods (`eq`, `ne`, `gt`,
//! `lt`, `ge`, `le`) accept both numeric primitives (via `CudfType`) and
//! string slices (`&str`) through the `Into<ScalarValue>` conversion,
//! and that the explicit `.on(Exec)` path stays in sync with the owned
//! `Series` API.

use cudf::{Exec, TypeId};

use crate::common::series;

fn eq_scalar_accepts_primitive() {
    let s = series::<i32>("z", &[1, 2, 2, 3]).unwrap();
    let mask = s.on(Exec::RETAINED).eq_scalar(2i32).unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true, false]);
}

fn ne_scalar_accepts_primitive() {
    let s = series::<i32>("z", &[1, 2, 2, 3]).unwrap();
    let mask = s.on(Exec::RETAINED).ne_scalar(2i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, false, true]);
}

fn gt_scalar_accepts_primitive() {
    let s = series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.on(Exec::RETAINED).gt_scalar(2i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true, true]);
}

fn lt_scalar_accepts_primitive() {
    let s = series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.on(Exec::RETAINED).lt_scalar(3i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false, false]);
}

fn ge_scalar_accepts_primitive() {
    let s = series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.on(Exec::RETAINED).ge_scalar(3i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true, true]);
}

fn le_scalar_accepts_primitive() {
    let s = series::<i32>("z", &[1, 2, 3, 4]).unwrap();
    let mask = s.on(Exec::RETAINED).le_scalar(2i32).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false, false]);
}

fn eq_scalar_accepts_str_literal() {
    let s = series("status", &["active", "inactive", "active"]).unwrap();
    let mask = s.on(Exec::RETAINED).eq_scalar("active").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn ne_scalar_accepts_str_literal() {
    let s = series("status", &["active", "inactive", "active"]).unwrap();
    let mask = s.on(Exec::RETAINED).ne_scalar("active").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, false]);
}

fn gt_scalar_accepts_str_literal() {
    let s = series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.on(Exec::RETAINED).gt_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true]);
}

fn lt_scalar_accepts_str_literal() {
    let s = series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.on(Exec::RETAINED).lt_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, false]);
}

fn ge_scalar_accepts_str_literal() {
    let s = series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.on(Exec::RETAINED).ge_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true]);
}

fn le_scalar_accepts_str_literal() {
    let s = series("name", &["alpha", "beta", "gamma"]).unwrap();
    let mask = s.on(Exec::RETAINED).le_scalar("beta").unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, true, false]);
}

// Owned `String` and `&String` work too.

fn eq_scalar_accepts_owned_string() {
    let s = series("status", &["active", "inactive", "active"]).unwrap();
    let needle = String::from("active");
    let mask = s.on(Exec::RETAINED).eq_scalar(&needle).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);

    let mask2 = s.on(Exec::RETAINED).eq_scalar(needle).unwrap();
    assert_eq!(mask2.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

// The explicit execution-context path stays in sync with the owned API.

fn eq_scalar_in_accepts_str_literal() {
    let s = series("status", &["active", "inactive", "active"]).unwrap();
    let ctx = cudf::Exec::RETAINED;
    let mask = s.on(ctx).eq_scalar("active").unwrap();

    assert_eq!(mask.dtype().unwrap().id(), TypeId::Bool8);
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn eq_scalar_in_accepts_owned_string() {
    let s = series("status", &["active", "inactive", "active"]).unwrap();
    let ctx = cudf::Exec::RETAINED;
    let needle = String::from("active");

    let mask = s.on(ctx).eq_scalar(needle).unwrap();
    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn gt_scalar_in_accepts_str_literal() {
    let s = series("name", &["alpha", "beta", "gamma"]).unwrap();
    let ctx = cudf::Exec::RETAINED;
    let mask = s.on(ctx).gt_scalar("beta").unwrap();

    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, false, true]);
}

fn eq_scalar_in_accepts_primitive() {
    let s = series::<i32>("z", &[1, 2, 2, 3]).unwrap();
    let ctx = cudf::Exec::RETAINED;
    let mask = s.on(ctx).eq_scalar(2i32).unwrap();

    assert_eq!(mask.to_vec_strict::<bool>().unwrap(), vec![false, true, true, false]);
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("scalar_compare_unified_tests::eq_scalar_accepts_primitive", eq_scalar_accepts_primitive),
    ("scalar_compare_unified_tests::ne_scalar_accepts_primitive", ne_scalar_accepts_primitive),
    ("scalar_compare_unified_tests::gt_scalar_accepts_primitive", gt_scalar_accepts_primitive),
    ("scalar_compare_unified_tests::lt_scalar_accepts_primitive", lt_scalar_accepts_primitive),
    ("scalar_compare_unified_tests::ge_scalar_accepts_primitive", ge_scalar_accepts_primitive),
    ("scalar_compare_unified_tests::le_scalar_accepts_primitive", le_scalar_accepts_primitive),
    ("scalar_compare_unified_tests::eq_scalar_accepts_str_literal", eq_scalar_accepts_str_literal),
    ("scalar_compare_unified_tests::ne_scalar_accepts_str_literal", ne_scalar_accepts_str_literal),
    ("scalar_compare_unified_tests::gt_scalar_accepts_str_literal", gt_scalar_accepts_str_literal),
    ("scalar_compare_unified_tests::lt_scalar_accepts_str_literal", lt_scalar_accepts_str_literal),
    ("scalar_compare_unified_tests::ge_scalar_accepts_str_literal", ge_scalar_accepts_str_literal),
    ("scalar_compare_unified_tests::le_scalar_accepts_str_literal", le_scalar_accepts_str_literal),
    ("scalar_compare_unified_tests::eq_scalar_accepts_owned_string", eq_scalar_accepts_owned_string),
    ("scalar_compare_unified_tests::eq_scalar_in_accepts_str_literal", eq_scalar_in_accepts_str_literal),
    ("scalar_compare_unified_tests::eq_scalar_in_accepts_owned_string", eq_scalar_in_accepts_owned_string),
    ("scalar_compare_unified_tests::gt_scalar_in_accepts_str_literal", gt_scalar_in_accepts_str_literal),
    ("scalar_compare_unified_tests::eq_scalar_in_accepts_primitive", eq_scalar_in_accepts_primitive),
];
