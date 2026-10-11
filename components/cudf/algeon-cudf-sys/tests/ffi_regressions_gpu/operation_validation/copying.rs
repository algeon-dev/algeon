use std::{os::raw::c_void, ptr};

use super::*;
use crate::support::run_named_cases;

fn make_bool8_column(data: &[u8]) -> cudf_sys::cudf_column_handle {
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::CUDF_TYPE_BOOL8,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &exec,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn test_concatenate_columns_null_element_returns_null_error() {
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);

    let views = [view, ptr::null_mut()];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_concatenate_columns_ex(views.as_ptr(), views.len() as i32, &exec, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

fn test_concatenate_tables_null_element_returns_null_error() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let tables = [tv, ptr::null_mut()];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_concatenate_tables_ex(tables.as_ptr(), tables.len() as i32, &exec, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_sort_negative_num_keys_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_sort_ex(tv, ptr::null(), ptr::null(), -1, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_sort_invalid_order_enum_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let column_order = [99i32];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sort_raw(raw_handle(tv), column_order.as_ptr(), ptr::null(), 1, raw_out_handle(&mut out), &exec)
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_sort_invalid_null_order_enum_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let null_order = [99i32];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status =
        unsafe { cudf_sort_raw(raw_handle(tv), ptr::null(), null_order.as_ptr(), 1, raw_out_handle(&mut out), &exec) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_drop_nulls_negative_keep_threshold_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls_ex(tv, ptr::null(), 0, -1, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_drop_nulls_negative_num_keys_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls_ex(tv, keys.as_ptr(), -1, 1, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_distinct_negative_num_keys_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status =
        unsafe { cudf_distinct_raw(raw_handle(tv), keys.as_ptr(), -1, 0, 0, 0, raw_out_handle(&mut out), &exec) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_gather_invalid_bounds_policy_rejected() {
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let map_col = make_i32_column(&[0, 2]);
    let map_view = make_column_view(map_col);

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_gather_raw(raw_handle(tv), raw_handle(map_view), 99, raw_out_handle(&mut out), &exec) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(map_view);
        cudf_sys::cudf_column_destroy(map_col);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_distinct_invalid_policy_values_rejected() {
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let (_stream, _mr, exec) = strict_exec_owners();
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();

    let status =
        unsafe { cudf_distinct_raw(raw_handle(tv), keys.as_ptr(), 1, 99, 0, 0, raw_out_handle(&mut out), &exec) };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    let status =
        unsafe { cudf_distinct_raw(raw_handle(tv), keys.as_ptr(), 1, 0, 99, 0, raw_out_handle(&mut out), &exec) };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    let status =
        unsafe { cudf_distinct_raw(raw_handle(tv), keys.as_ptr(), 1, 0, 0, 99, raw_out_handle(&mut out), &exec) };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_drop_nulls_zero_num_keys_allows_null_keys_pointer() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls_ex(tv, ptr::null(), 0, 0, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_distinct_zero_num_keys_allows_null_keys_pointer() {
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_ex(
            tv,
            ptr::null(),
            0,
            cudf_sys::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            cudf_sys::CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL,
            cudf_sys::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL,
            &mut out,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

fn test_gather_ex_uses_explicit_memory_resource() {
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input_col = make_i32_column(&[10, 20, 30, 40]);
    let input_tbl = make_table_from_one_column(input_col);
    let input_view = make_table_view(input_tbl);
    let gather_map = make_i32_column(&[3, 1, 0, 2]);
    let gather_map_view = make_column_view(gather_map);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_gather_ex(
            input_view,
            gather_map_view,
            cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            &mut out,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "gather allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_column_view_destroy(gather_map_view);
        cudf_sys::cudf_column_destroy(gather_map);
        cudf_sys::cudf_table_view_destroy(input_view);
        cudf_sys::cudf_table_destroy(input_tbl);
    }
}

fn test_cross_join_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 3, 4]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[10, 20, 30]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_cross_join_ex(left_tv, right_tv, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "cross_join allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

fn test_scatter_null_scatter_map_returns_libcudf_error() {
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let source_tbl = make_table_from_one_column(make_i32_column(&[10, 20]));
    let target_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 3, 4]));
    let source_tv = make_table_view(source_tbl);
    let target_tv = make_table_view(target_tbl);
    let scatter_map = make_i32_column_nullable(&[0, 1], &[true, false]);
    let scatter_map_view = make_column_view(scatter_map);

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_scatter_ex(source_tv, scatter_map_view, target_tv, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(scatter_map_view);
        cudf_sys::cudf_column_destroy(scatter_map);
        cudf_sys::cudf_table_view_destroy(target_tv);
        cudf_sys::cudf_table_view_destroy(source_tv);
        cudf_sys::cudf_table_destroy(target_tbl);
        cudf_sys::cudf_table_destroy(source_tbl);
    }
}

fn test_scatter_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let target_mr = OwnedMemoryResource::new_test_stats();

    let source_tbl = make_table_from_one_column(make_i32_column(&[10, 20, 30]));
    let target_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 3, 4, 5]));
    let source_tv = make_table_view(source_tbl);
    let target_tv = make_table_view(target_tbl);
    let scatter_map = make_i32_column(&[4, 2, 0]);
    let scatter_map_view = make_column_view(scatter_map);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_scatter_ex(source_tv, scatter_map_view, target_tv, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "scatter allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_column_view_destroy(scatter_map_view);
        cudf_sys::cudf_column_destroy(scatter_map);
        cudf_sys::cudf_table_view_destroy(target_tv);
        cudf_sys::cudf_table_view_destroy(source_tv);
        cudf_sys::cudf_table_destroy(target_tbl);
        cudf_sys::cudf_table_destroy(source_tbl);
    }
}

fn test_apply_boolean_mask_ex_uses_explicit_memory_resource() {
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input_col = make_i32_column(&[10, 20, 30, 40]);
    let input_tbl = make_table_from_one_column(input_col);
    let input_view = make_table_view(input_tbl);
    let mask = make_bool8_column(&[1, 0, 1, 0]);
    let mask_view = make_column_view(mask);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_apply_boolean_mask_ex(input_view, mask_view, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "apply_boolean_mask allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_column_view_destroy(mask_view);
        cudf_sys::cudf_column_destroy(mask);
        cudf_sys::cudf_table_view_destroy(input_view);
        cudf_sys::cudf_table_destroy(input_tbl);
    }
}

fn test_replace_nulls_policy_ex_uses_explicit_memory_resource() {
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input = make_i32_column_nullable(&[10, 20, 30, 40], &[true, false, true, false]);
    let input_view = make_column_view(input);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_replace_nulls_policy_ex(input_view, 0, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "replace_nulls policy allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_column_view_destroy(input_view);
        cudf_sys::cudf_column_destroy(input);
    }
}

fn test_distinct_ex_uses_explicit_memory_resource() {
    let target_mr = OwnedMemoryResource::new_test_stats();

    let input_col = make_i32_column(&[1, 2, 1, 3, 2, 4, 4, 5]);
    let input_tbl = make_table_from_one_column(input_col);
    let input_view = make_table_view(input_tbl);
    let keys = [0_i32];
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let target_before = target_mr.allocation_count();

    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_ex(
            input_view,
            keys.as_ptr(),
            keys.len() as i32,
            cudf_sys::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            cudf_sys::CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL,
            cudf_sys::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL,
            &mut out,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "distinct allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(input_view);
        cudf_sys::cudf_table_destroy(input_tbl);
    }
}

#[test]
fn copying_cases() {
    run_named_cases(&[
        ("test_distinct_invalid_policy_values_rejected", test_distinct_invalid_policy_values_rejected),
        (
            "test_scatter_ex_uses_explicit_memory_resource_for_allocations_and_provenance",
            test_scatter_ex_uses_explicit_memory_resource_for_allocations_and_provenance,
        ),
        (
            "test_cross_join_ex_uses_explicit_memory_resource_for_allocations_and_provenance",
            test_cross_join_ex_uses_explicit_memory_resource_for_allocations_and_provenance,
        ),
        ("test_distinct_negative_num_keys_rejected", test_distinct_negative_num_keys_rejected),
        ("test_sort_negative_num_keys_rejected", test_sort_negative_num_keys_rejected),
        (
            "test_replace_nulls_policy_ex_uses_explicit_memory_resource",
            test_replace_nulls_policy_ex_uses_explicit_memory_resource,
        ),
        ("test_scatter_null_scatter_map_returns_libcudf_error", test_scatter_null_scatter_map_returns_libcudf_error),
        ("test_distinct_ex_uses_explicit_memory_resource", test_distinct_ex_uses_explicit_memory_resource),
        (
            "test_concatenate_columns_null_element_returns_null_error",
            test_concatenate_columns_null_element_returns_null_error,
        ),
        (
            "test_apply_boolean_mask_ex_uses_explicit_memory_resource",
            test_apply_boolean_mask_ex_uses_explicit_memory_resource,
        ),
        ("test_drop_nulls_negative_keep_threshold_rejected", test_drop_nulls_negative_keep_threshold_rejected),
        ("test_sort_invalid_null_order_enum_rejected", test_sort_invalid_null_order_enum_rejected),
        ("test_gather_invalid_bounds_policy_rejected", test_gather_invalid_bounds_policy_rejected),
        (
            "test_concatenate_tables_null_element_returns_null_error",
            test_concatenate_tables_null_element_returns_null_error,
        ),
        ("test_gather_ex_uses_explicit_memory_resource", test_gather_ex_uses_explicit_memory_resource),
        ("test_drop_nulls_negative_num_keys_rejected", test_drop_nulls_negative_num_keys_rejected),
        ("test_sort_invalid_order_enum_rejected", test_sort_invalid_order_enum_rejected),
        ("test_distinct_zero_num_keys_allows_null_keys_pointer", test_distinct_zero_num_keys_allows_null_keys_pointer),
        (
            "test_drop_nulls_zero_num_keys_allows_null_keys_pointer",
            test_drop_nulls_zero_num_keys_allows_null_keys_pointer,
        ),
    ]);
}
