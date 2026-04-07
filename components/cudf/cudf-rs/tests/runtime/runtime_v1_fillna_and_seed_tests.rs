//! Tests for explicit placement before `fillna_with_series` and ambient constructor variants.
//!
//! Coverage:
//! - `DataFrame::copy_into_runtime` + `fillna_with_series` — explicit placement
//! - `Series::copy_into_runtime` + `fillna_with_series` — explicit placement
//! - `DataFrame::fillna_with_series` — v2: mixed-domain default path succeeds
//! - `Series::fillna_with_series` — v2: mixed-domain default path succeeds
//! - `Column::empty_with_runtime` — retains explicit domain
//! - `Column::from_device_ptr_with_runtime` — uses `_ex` FFI, retains explicit domain

use std::{ffi::c_void, ptr};

use cudf::{DataFrame, Series, TypeId, core::Column};
use cudf_sys::{
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
// copy_into_runtime + fillna_with_series — DataFrame
// ---------------------------------------------------------------------------

/// Left DataFrame in domain A, fill series in domain B, target runtime C.
/// Explicit placement makes the target runtime step visible before
/// `fillna_with_series` runs.
#[test]
fn df_fillna_after_copy_into_runtime_places_inputs_in_target_runtime() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let target_rt = sample_runtime();

    // Build a DataFrame with a nullable column in domain A.
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt_a, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_nullable_with_runtime(&rt_a, "val", &[10i32, 0, 30], &[true, false, true]).unwrap()),
    ])
    .unwrap();

    // Fill series in domain B.
    let fill = Series::new_with_runtime(&rt_b, "fill", &[99i32, 99, 99]).unwrap();

    let df = df.copy_into_runtime(&target_rt).unwrap();
    let fill = fill.copy_into_runtime(&target_rt).unwrap();

    obs_reset();
    let result = df.fillna_with_series("val", &fill).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(result.__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(result.__debug_execution_domain_mr_value(), target_rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(result.series("val").unwrap().to_vec::<i32>().unwrap(), vec![10, 99, 30]);
}

/// v2 semantics: mixed-domain fillna_with_series succeeds via the default
/// path (without `_with_runtime`). Output provenance follows the input
/// DataFrame (primary input for replace-nulls).
#[test]
fn df_fillna_with_series_succeeds_with_mixed_domains_default_path() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();

    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt_a, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_nullable_with_runtime(&rt_a, "val", &[10i32, 0, 30], &[true, false, true]).unwrap()),
    ])
    .unwrap();

    let fill = Series::new_with_runtime(&rt_b, "fill", &[99i32, 99, 99]).unwrap();
    let expected_mr = df.__debug_execution_domain_mr_value();

    // Without runtime, mixed domains now succeed under v2.
    obs_reset();
    let result = df.fillna_with_series("val", &fill).unwrap();

    // Output provenance inherits from the input DataFrame.
    assert_eq!(result.__debug_execution_domain_id(), rt_a.__debug_execution_domain_id());
    assert_eq!(obs_last_stream(), rt_a.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(result.__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.series("val").unwrap().to_vec::<i32>().unwrap(), vec![10, 99, 30]);
}

/// Both inputs already in the same runtime — no copy needed.
#[test]
fn df_fillna_after_copy_into_runtime_same_domain_works() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([
        ("id", Series::new_with_runtime(&rt, "id", &[1i32, 2, 3]).unwrap()),
        ("val", Series::new_nullable_with_runtime(&rt, "val", &[10i32, 0, 30], &[true, false, true]).unwrap()),
    ])
    .unwrap();
    let fill = Series::new_with_runtime(&rt, "fill", &[99i32, 99, 99]).unwrap();

    obs_reset();
    let result =
        df.copy_into_runtime(&rt).unwrap().fillna_with_series("val", &fill.copy_into_runtime(&rt).unwrap()).unwrap();

    assert_eq!(result.__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(result.series("val").unwrap().to_vec::<i32>().unwrap(), vec![10, 99, 30]);
}

// ---------------------------------------------------------------------------
// copy_into_runtime + fillna_with_series — Series
// ---------------------------------------------------------------------------

/// Same as the DataFrame test but at the Series level.
/// Explicit placement makes the target runtime step visible.
#[test]
fn series_fillna_after_copy_into_runtime_places_inputs_in_target_runtime() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();
    let target_rt = sample_runtime();

    let series = Series::new_nullable_with_runtime(&rt_a, "val", &[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = Series::new_with_runtime(&rt_b, "fill", &[99i32, 99, 99]).unwrap();

    let series = series.copy_into_runtime(&target_rt).unwrap();
    let fill = fill.copy_into_runtime(&target_rt).unwrap();

    obs_reset();
    let result = series.fillna_with_series(&fill).unwrap();

    assert_eq!(obs_last_stream(), target_rt.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, target_rt.memory_resource().__debug_raw_handle_value());
    assert_eq!(result.as_column().__debug_execution_domain_id(), target_rt.__debug_execution_domain_id());
    assert_eq!(
        result.as_column().__debug_execution_domain_mr_value(),
        target_rt.memory_resource().__debug_raw_handle_value()
    );
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![10, 99, 30]);
}

/// v2 semantics: mixed-domain Series::fillna_with_series succeeds via the
/// default path (without `_with_runtime`). Output provenance follows the
/// input series (primary input for replace-nulls).
#[test]
fn series_fillna_with_series_succeeds_with_mixed_domains_default_path() {
    let rt_a = sample_runtime();
    let rt_b = sample_runtime();

    let series = Series::new_nullable_with_runtime(&rt_a, "val", &[10i32, 0, 30], &[true, false, true]).unwrap();
    let fill = Series::new_with_runtime(&rt_b, "fill", &[99i32, 99, 99]).unwrap();
    let expected_mr = series.as_column().__debug_execution_domain_mr_value();

    // Without runtime, mixed domains now succeed under v2.
    obs_reset();
    let result = series.fillna_with_series(&fill).unwrap();

    // Output provenance inherits from the input series.
    assert_eq!(result.as_column().__debug_execution_domain_id(), rt_a.__debug_execution_domain_id());
    assert_eq!(obs_last_stream(), rt_a.stream().__debug_raw_handle_value());
    assert_eq!(obs_last_mr() as usize, expected_mr);
    assert_eq!(result.as_column().__debug_execution_domain_mr_value(), expected_mr);
    assert_eq!(result.to_vec::<i32>().unwrap(), vec![10, 99, 30]);
}

// ---------------------------------------------------------------------------
// Column::empty_with_runtime
// ---------------------------------------------------------------------------

/// An empty column created via `empty_with_runtime` carries the runtime's
/// explicit domain, not the ambient default.
#[test]
fn column_empty_with_runtime_retains_explicit_domain() {
    let rt = sample_runtime();
    let col = Column::empty_with_runtime(&rt, TypeId::Int32).unwrap();

    assert_eq!(col.__debug_execution_domain_kind(), "explicit");
    assert_eq!(col.__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(col.__debug_execution_domain_stream_value(), rt.stream().__debug_raw_handle_value());
}

/// An `empty_with_runtime` column can be added to an explicit-domain
/// DataFrame via `with_column` without mixed-domain rejection.
#[test]
fn empty_with_runtime_column_in_dataframe_no_mixed_domain_error() {
    let rt = sample_runtime();
    let df = DataFrame::from_series([("id", Series::new_with_runtime(&rt, "id", &[1i32; 0]).unwrap())]).unwrap();
    let empty_col = Column::empty_with_runtime(&rt, TypeId::Int64).unwrap();

    // Adding an empty column in the same domain should succeed without
    // the mixed-domain error.
    let result = df.with_column("extra", Series::from_column(empty_col).unwrap());
    assert!(result.is_ok(), "expected Ok but got: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// Column::from_device_ptr_with_runtime
// ---------------------------------------------------------------------------

/// `from_device_ptr_with_runtime` tags the result with the runtime's
/// explicit domain.
#[test]
fn column_from_device_ptr_with_runtime_retains_explicit_domain() {
    let rt = sample_runtime();

    // Allocate some device memory via a Series (easiest way to get a device ptr).
    let source = Series::new_with_runtime(&rt, "src", &[10i32, 20, 30]).unwrap();
    let device_ptr = source.as_column_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let col = unsafe { Column::from_device_ptr_with_runtime(&rt, device_ptr, 3, TypeId::Int32).unwrap() };

    assert_eq!(col.__debug_execution_domain_kind(), "explicit");
    assert_eq!(col.__debug_execution_domain_id(), rt.__debug_execution_domain_id());
    assert_eq!(col.__debug_execution_domain_stream_value(), rt.stream().__debug_raw_handle_value());
}

/// `from_device_ptr_with_runtime` routes through the `_ex` FFI shim,
/// which records the explicit stream in the observation hooks.
#[test]
fn column_from_device_ptr_with_runtime_uses_ex_ffi() {
    let rt = sample_runtime();
    let expected_stream = rt.stream().__debug_raw_handle_value();
    let default_mr = obs_current_device_resource() as usize;

    // Allocate source data in the runtime's domain.
    let source = Series::new_with_runtime(&rt, "src", &[1i64, 2, 3, 4]).unwrap();
    let device_ptr = source.as_column_view().unwrap().device_data_ptr().unwrap();

    obs_reset();
    let col = unsafe { Column::from_device_ptr_with_runtime(&rt, device_ptr, 4, TypeId::Int64).unwrap() };

    // The observation hook should have recorded the runtime's stream (not default).
    assert_eq!(obs_last_stream(), expected_stream);
    // The observation hook should have recorded the runtime's MR (not default).
    assert_ne!(obs_last_mr() as usize, default_mr);
    assert_eq!(col.__debug_execution_domain_mr_value(), rt.memory_resource().__debug_raw_handle_value());
}
