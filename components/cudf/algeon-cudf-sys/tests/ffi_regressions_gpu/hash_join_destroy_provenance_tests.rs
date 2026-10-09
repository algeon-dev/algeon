use std::ptr;

use cudf_sys::test_support;

use crate::{
    helpers::{make_i32_column, make_table_from_one_column, make_table_view},
    support::{OwnedMemoryResource, OwnedStream, current_device, required_exec_options},
};

fn column_destroy_keeps_explicit_memory_resource_alive_after_handle_destroy() {
    let mut mr: cudf_sys::cudf_memory_resource_handle = ptr::null_mut();
    let status = unsafe { test_support::cudf_test_memory_resource_create_stats(&mut mr) };
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

    unsafe { assert_eq!(cudf_sys::cudf_column_destroy(column), cudf_sys::CUDF_SUCCESS) };
}

fn distinct_hash_join_destroy_uses_build_memory_resource() {
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
    assert!(
        build_mr.active_allocation_count() > 0,
        "distinct hash-join construction must charge retained build allocations to the explicit execution memory resource"
    );

    unsafe { cudf_sys::cudf_distinct_hash_join_destroy(join) };

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

fn hash_join_destroy_uses_build_memory_resource() {
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
        build_mr.active_allocation_count(),
        0,
        "hash-join destroy must release retained build allocations from the explicit execution memory resource"
    );

    unsafe {
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

pub(super) const CASES: &[(&str, fn())] = &[
    (
        "hash_join_destroy_provenance_tests::hash_join_destroy_uses_build_memory_resource",
        hash_join_destroy_uses_build_memory_resource,
    ),
    (
        "hash_join_destroy_provenance_tests::column_destroy_keeps_explicit_memory_resource_alive_after_handle_destroy",
        column_destroy_keeps_explicit_memory_resource_alive_after_handle_destroy,
    ),
    (
        "hash_join_destroy_provenance_tests::distinct_hash_join_destroy_uses_build_memory_resource",
        distinct_hash_join_destroy_uses_build_memory_resource,
    ),
    (
        "table_destroy_retains_original_column_provenance_after_owner_drop",
        table_destroy_retains_original_column_provenance_after_owner_drop,
    ),
    ("scalar_destroy_retains_provenance_after_owner_drop", scalar_destroy_retains_provenance_after_owner_drop),
    ("buffer_destroy_retains_provenance_after_owner_drop", buffer_destroy_retains_provenance_after_owner_drop),
    (
        "released_columns_retain_original_provenance_after_resource_owner_drop",
        released_columns_retain_original_provenance_after_resource_owner_drop,
    ),
];

fn assert_destroy_after_resource_owner_drop<H>(
    create: impl FnOnce(&cudf_sys::cudf_exec_options_t) -> H,
    destroy: unsafe extern "C" fn(H) -> cudf_sys::cudf_status_t,
) {
    let mr = OwnedMemoryResource::new_test_stats();
    let stream = OwnedStream::new();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let output = create(&exec);
    stream.synchronize();
    assert!(mr.active_allocation_count() > 0);
    drop(stream);
    drop(mr);
    unsafe {
        assert_eq!(destroy(output), cudf_sys::CUDF_SUCCESS);
    }
}

fn table_destroy_retains_original_column_provenance_after_owner_drop() {
    assert_destroy_after_resource_owner_drop(
        |exec| {
            let data = [1_i32, 2, 3];
            let mut column = ptr::null_mut();
            let mut table = ptr::null_mut();
            unsafe {
                assert_eq!(
                    cudf_sys::cudf_column_create_numeric_ex(
                        cudf_sys::CUDF_TYPE_INT32,
                        data.as_ptr().cast(),
                        data.len() as i32,
                        ptr::null(),
                        0,
                        exec,
                        &mut column
                    ),
                    cudf_sys::CUDF_SUCCESS
                );
                assert_eq!(cudf_sys::cudf_table_create(&mut column, 1, &mut table), cudf_sys::CUDF_SUCCESS);
            }
            assert!(!table.is_null());
            table
        },
        cudf_sys::cudf_table_destroy,
    );
}

fn scalar_destroy_retains_provenance_after_owner_drop() {
    assert_destroy_after_resource_owner_drop(
        |exec| {
            let value = 42_i32;
            let mut scalar = ptr::null_mut();
            unsafe {
                assert_eq!(
                    cudf_sys::cudf_scalar_create_numeric_ex(
                        cudf_sys::CUDF_TYPE_INT32,
                        (&value as *const i32).cast(),
                        1,
                        exec,
                        &mut scalar
                    ),
                    cudf_sys::CUDF_SUCCESS
                );
            }
            assert!(!scalar.is_null());
            scalar
        },
        cudf_sys::cudf_scalar_destroy,
    );
}

fn buffer_destroy_retains_provenance_after_owner_drop() {
    assert_destroy_after_resource_owner_drop(
        |exec| {
            let mut buffer = ptr::null_mut();
            unsafe {
                assert_eq!(cudf_sys::cudf_device_buffer_create_ex(4096, exec, &mut buffer), cudf_sys::CUDF_SUCCESS);
            }
            assert!(!buffer.is_null());
            buffer
        },
        cudf_sys::cudf_device_buffer_destroy,
    );
}

fn released_columns_retain_original_provenance_after_resource_owner_drop() {
    let mr = OwnedMemoryResource::new_test_stats();
    let stream = OwnedStream::new();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let data = [4_i32, 5, 6];
    let mut column = ptr::null_mut();
    unsafe {
        assert_eq!(
            cudf_sys::cudf_column_create_numeric_ex(
                cudf_sys::CUDF_TYPE_INT32,
                data.as_ptr().cast(),
                data.len() as i32,
                ptr::null(),
                0,
                &exec,
                &mut column
            ),
            cudf_sys::CUDF_SUCCESS
        );
    }
    let mut table = ptr::null_mut();
    unsafe {
        assert_eq!(cudf_sys::cudf_table_create(&mut column, 1, &mut table), cudf_sys::CUDF_SUCCESS);
    }
    stream.synchronize();
    drop(stream);
    drop(mr);
    let mut columns = ptr::null_mut();
    let mut count = -1;
    unsafe {
        assert_eq!(cudf_sys::cudf_table_release_columns(table, &mut columns, &mut count), cudf_sys::CUDF_SUCCESS);
        assert_eq!(count, 1);
        assert!(!columns.is_null());
        assert_eq!(cudf_sys::cudf_column_destroy(*columns), cudf_sys::CUDF_SUCCESS);
        cudf_sys::cudf_free_column_array(columns);
    }
}
