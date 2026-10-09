//! Tests for explicit placement before `fillna_with_series` and runtime-seeded constructors.
//!
//! Coverage:
//! - `DataFrame::copy_into_runtime` + `fillna_with_series`: explicit placement
//! - `Series::copy_into_runtime` + `fillna_with_series`: explicit placement
//! - `DataFrame::fillna_with_series`: v2: mixed-domain default path succeeds
//! - `SeriesOps::fillna_with_series`: v2: mixed-domain default path succeeds
//! - `Runtime::empty_column`: retains explicit domain
//! - `Runtime::column_from_device_ptr`: uses `_ex` FFI, retains explicit domain
use std::{ffi::c_void, ptr};

use cudf::{
    DataFrame, Exec, Series, TypeId,
    testing::{prelude::*, run_named_cases},
};
use cudf_sys::test_support::{
    cudf_test_current_device_resource_raw, cudf_test_exec_last_mr_raw, cudf_test_exec_last_stream_value,
    cudf_test_exec_reset_observation,
};

use crate::common::sample_runtime;

fn obs_reset() {
    unsafe { cudf_test_exec_reset_observation() };
}

fn obs_last_stream() -> usize {
    let mut out: usize = 0;
    let status = unsafe { cudf_test_exec_last_stream_value(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_stream_value failed");
    out
}

fn obs_last_mr() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_exec_last_mr_raw(&mut out) };
    assert_eq!(status, 0, "cudf_test_exec_last_mr_raw failed");
    out
}

fn obs_current_device_resource() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, 0, "cudf_test_current_device_resource_raw failed");
    out
}

// ---------------------------------------------------------------------------
// copy_into_runtime + fillna_with_series; DataFrame
// ---------------------------------------------------------------------------

/// Left DataFrame in domain A, fill series in domain B, target runtime C.
/// Explicit placement makes the target runtime step visible before
/// `fillna_with_series` runs.
fn df_fillna_after_copy_into_runtime_places_inputs_in_target_runtime() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let target_rt = sample_runtime();

    // Build a DataFrame with a nullable column in domain A.
    let df = DataFrame::from_series([
        ("id", rt_a.series(&[1i32, 2, 3]).unwrap()),
        ("val", rt_a.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap()),
    ])
    .unwrap();

    // Fill series in domain B.
    let fill = rt_b.series(&[99i32, 99, 99]).unwrap();

    let df = df.copy_into_runtime(&target_rt).unwrap();
    let fill = fill.copy_into_runtime(&target_rt).unwrap();

    obs_reset();
    let result = df.on(Exec::RETAINED).fillna_with_series("val", &fill).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().testing_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(result.testing_execution_domain_id(), target_rt.testing_execution_domain_id());
    assert_eq!(result.testing_execution_domain_mr_value(), target_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(result.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 99, 30]);
}

/// v2 semantics: mixed-domain fillna_with_series succeeds via the default
/// path (without `_with_runtime`). Output provenance follows the input
/// DataFrame (primary input for replace-nulls).
fn df_fillna_with_series_succeeds_with_mixed_domains_default_path() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();

    let df = DataFrame::from_series([
        ("id", rt_a.series(&[1i32, 2, 3]).unwrap()),
        ("val", rt_a.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap()),
    ])
    .unwrap();

    let fill = rt_b.series(&[99i32, 99, 99]).unwrap();
    let expected_mr = df.testing_execution_domain_mr_value();

    // Without runtime, mixed domains now succeed under v2.
    obs_reset();
    let result = df.on(Exec::RETAINED).fillna_with_series("val", &fill).unwrap();

    // Output provenance inherits from the input DataFrame.
    assert_eq!(result.testing_execution_domain_id(), rt_a.testing_execution_domain_id());
    assert_eq!(obs_last_stream(), rt_a.stream().testing_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(result.testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 99, 30]);
}

/// Both inputs already in the same runtime; no copy needed.
fn df_fillna_after_copy_into_runtime_same_domain_works() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", rt.series(&[1i32, 2, 3]).unwrap()),
        ("val", rt.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap()),
    ])
    .unwrap();
    let fill = rt.series(&[99i32, 99, 99]).unwrap();

    obs_reset();
    let result = df
        .copy_into_runtime(&rt)
        .unwrap()
        .on(Exec::RETAINED)
        .fillna_with_series("val", &fill.copy_into_runtime(&rt).unwrap())
        .unwrap();

    assert_eq!(result.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(result.series("val").unwrap().to_vec_strict::<i32>().unwrap(), vec![10, 99, 30]);
}

// ---------------------------------------------------------------------------
// copy_into_runtime + fillna_with_series; Series
// ---------------------------------------------------------------------------

/// Same as the DataFrame test but at the Series level.
/// Explicit placement makes the target runtime step visible.
fn series_fillna_after_copy_into_runtime_places_inputs_in_target_runtime() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let target_rt = sample_runtime();

    let series = rt_a.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = rt_b.series(&[99i32, 99, 99]).unwrap();

    let series = series.copy_into_runtime(&target_rt).unwrap();
    let fill = fill.copy_into_runtime(&target_rt).unwrap();

    obs_reset();
    let result = series.on(Exec::RETAINED).fillna_with_series(&fill).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().testing_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().testing_raw_handle_value());
    assert_eq!(result.as_column().testing_execution_domain_id(), target_rt.testing_execution_domain_id());
    assert_eq!(
        result.as_column().testing_execution_domain_mr_value(),
        target_rt.memory_resource().testing_raw_handle_value()
    );
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 99, 30]);
}

/// v2 semantics: mixed-domain SeriesOps::fillna_with_series succeeds via the
/// default path (without `_with_runtime`). Output provenance follows the
/// input series (primary input for replace-nulls).
fn series_fillna_with_series_succeeds_with_mixed_domains_default_path() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();

    let series = rt_a.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = rt_b.series(&[99i32, 99, 99]).unwrap();
    let expected_mr = series.as_column().testing_execution_domain_mr_value();

    // Without runtime, mixed domains now succeed under v2.
    obs_reset();
    let result = series.on(Exec::RETAINED).fillna_with_series(&fill).unwrap();

    // Output provenance inherits from the input series.
    assert_eq!(result.as_column().testing_execution_domain_id(), rt_a.testing_execution_domain_id());
    assert_eq!(obs_last_stream(), rt_a.stream().testing_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(result.as_column().testing_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.to_vec_strict::<i32>().unwrap(), vec![10, 99, 30]);
}

// ---------------------------------------------------------------------------
// Runtime::empty_column
// ---------------------------------------------------------------------------

/// An empty column created via `Runtime::empty_column` carries the runtime's
/// explicit domain instead of relying on any implicit fallback.
fn runtime_empty_column_retains_explicit_domain() {
    let rt = sample_runtime();
    obs_reset();
    let col = rt.empty_column(TypeId::Int32).unwrap();

    assert_eq!(col.testing_execution_domain_kind(), "explicit");
    assert_eq!(col.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(col.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
    assert_eq!(obs_last_stream(), rt.stream().testing_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, rt.memory_resource().testing_raw_handle_value());
}

/// A `Runtime::empty_column` result can be added to an explicit-domain
/// DataFrame via `with_column` without mixed-domain rejection.
fn runtime_empty_column_in_dataframe_no_mixed_domain_error() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([("id", rt.series(&[1i32; 0]).unwrap())]).unwrap();
    let empty_col = rt.empty_column(TypeId::Int64).unwrap();

    // Adding an empty column in the same domain should succeed without
    // the mixed-domain error.
    let result = df.with_column("extra", Series::from_column(empty_col).unwrap());
    assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// Runtime::column_from_device_ptr
// ---------------------------------------------------------------------------

/// `Runtime::column_from_device_ptr` tags the result with the runtime's
/// explicit domain.
fn runtime_column_from_device_ptr_retains_explicit_domain() {
    let rt = sample_runtime();

    // Allocate some device memory via a Series (easiest way to get a device ptr).
    let source = rt.series(&[10i32, 20, 30]).unwrap();
    let device_ptr = source.testing_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let col = unsafe { rt.column_from_device_ptr(device_ptr, 3, TypeId::Int32).unwrap() };

    assert_eq!(col.testing_execution_domain_kind(), "explicit");
    assert_eq!(col.testing_execution_domain_id(), rt.testing_execution_domain_id());
    assert_eq!(col.testing_execution_domain_stream_value(), rt.stream().testing_raw_handle_value());
}

/// `Runtime::column_from_device_ptr` routes through the `_ex` FFI shim,
/// which records the explicit stream in the observation hooks.
fn runtime_column_from_device_ptr_uses_ex_ffi() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().testing_raw_handle_value();
    let default_mr = obs_current_device_resource() as usize;

    // Allocate source data in the runtime's domain.
    let source = rt.series(&[1i64, 2, 3, 4]).unwrap();
    let device_ptr = source.testing_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let col = unsafe { rt.column_from_device_ptr(device_ptr, 4, TypeId::Int64).unwrap() };

    // The observation hook should have recorded the runtime's stream (not default).
    assert_eq!(obs_last_stream(), expected_stream);
    // The observation hook should have recorded the runtime's MR (not default).
    assert_ne!(obs_last_mr() as usize, default_mr);
    assert_eq!(col.testing_execution_domain_mr_value(), rt.memory_resource().testing_raw_handle_value());
}

#[test]
fn runtime_v1_fillna_and_seed_tests_cases() {
    run_named_cases(&[
        (
            "df_fillna_after_copy_into_runtime_places_inputs_in_target_runtime",
            df_fillna_after_copy_into_runtime_places_inputs_in_target_runtime,
        ),
        (
            "df_fillna_with_series_succeeds_with_mixed_domains_default_path",
            df_fillna_with_series_succeeds_with_mixed_domains_default_path,
        ),
        ("df_fillna_after_copy_into_runtime_same_domain_works", df_fillna_after_copy_into_runtime_same_domain_works),
        (
            "series_fillna_after_copy_into_runtime_places_inputs_in_target_runtime",
            series_fillna_after_copy_into_runtime_places_inputs_in_target_runtime,
        ),
        (
            "series_fillna_with_series_succeeds_with_mixed_domains_default_path",
            series_fillna_with_series_succeeds_with_mixed_domains_default_path,
        ),
        ("runtime_empty_column_retains_explicit_domain", runtime_empty_column_retains_explicit_domain),
        (
            "runtime_empty_column_in_dataframe_no_mixed_domain_error",
            runtime_empty_column_in_dataframe_no_mixed_domain_error,
        ),
        (
            "runtime_column_from_device_ptr_retains_explicit_domain",
            runtime_column_from_device_ptr_retains_explicit_domain,
        ),
        ("runtime_column_from_device_ptr_uses_ex_ffi", runtime_column_from_device_ptr_uses_ex_ffi),
    ]);
}
