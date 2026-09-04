mod support;

#[path = "ffi_regressions/helpers.rs"]
mod helpers;

use std::ptr;

use helpers::{make_i32_column, make_table_from_one_column, make_table_view};
use support::{OwnedMemoryResource, OwnedStream, current_device, current_device_resource_raw, lock_shim_state};

#[test]
fn column_destroy_keeps_explicit_memory_resource_alive_after_handle_destroy() {
    let _shim_state = lock_shim_state();
    let mut mr: cudf_sys::cudf_memory_resource_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::test_support::cudf_test_memory_resource_create_stats(&mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());

    let borrowed = unsafe { cudf_sys::BorrowedMemoryResource::from_raw(mr) };
    let stream = OwnedStream::new();
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new()
        .with_device(current_device())
        .with_stream(&borrowed_stream)
        .with_memory_resource(&borrowed);
    let data = [1_i32, 2, 3, 4];
    let mut column: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::CUDF_TYPE_INT32,
            data.as_ptr().cast(),
            data.len() as i32,
            ptr::null(),
            0,
            exec.as_raw(),
            &mut column,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!column.is_null());

    let status = unsafe { cudf_sys::cudf_memory_resource_destroy(mr, current_device()) };
    assert_eq!(
        status,
        cudf_sys::CUDF_SUCCESS,
        "destroying the MR owner handle should leave outstanding output handles' retained MR lifetime alive"
    );

    unsafe { cudf_sys::cudf_column_destroy(column) };
}

#[test]
fn distinct_hash_join_destroy_uses_build_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let build_mr = OwnedMemoryResource::new_test_stats();

    let build_col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let build_borrowed = build_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new()
        .with_device(current_device())
        .with_stream(&target_stream)
        .with_memory_resource(&build_borrowed);

    let mut join: cudf_sys::cudf_distinct_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_hash_join_create_ex(
            build_tv,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            &mut join,
            exec.as_raw(),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    stream.synchronize();
    // Device-domain model: the build constructor takes no memory resource, so
    // its retained hash table allocates through (and later releases through)
    // the device's current resource. The explicit execution memory resource is
    // outputs-only and must stay untouched by construction.
    assert_eq!(
        build_mr.active_allocation_count(),
        0,
        "distinct hash-join construction must not charge build allocations to the outputs-only execution memory resource"
    );

    unsafe { cudf_sys::cudf_distinct_hash_join_destroy(join) };

    assert_eq!(
        current_device_resource_raw(),
        previous_raw,
        "distinct hash-join destroy must leave the device's current resource unchanged"
    );
    assert_eq!(
        build_mr.active_allocation_count(),
        0,
        "distinct hash-join destroy must not touch the outputs-only execution memory resource"
    );

    unsafe {
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

#[test]
fn hash_join_destroy_uses_build_memory_resource() {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let build_mr = OwnedMemoryResource::new_test_stats();

    let build_col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let build_borrowed = build_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new()
        .with_device(current_device())
        .with_stream(&target_stream)
        .with_memory_resource(&build_borrowed);

    let mut join: cudf_sys::cudf_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_create_ex(
            build_tv,
            cudf_sys::CUDF_NULLABLE_JOIN_YES,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            0.5,
            &mut join,
            exec.as_raw(),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    stream.synchronize();
    assert!(
        build_mr.active_allocation_count() > 0,
        "hash-join construction must charge retained build allocations to the explicit execution memory resource"
    );

    unsafe { cudf_sys::cudf_hash_join_destroy(join) };

    assert_eq!(
        current_device_resource_raw(),
        previous_raw,
        "hash-join destroy must leave the device's current resource unchanged"
    );
    assert_eq!(
        build_mr.active_allocation_count(),
        0,
        "hash-join destroy must release retained build allocations from the explicit execution memory resource"
    );

    unsafe {
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}
