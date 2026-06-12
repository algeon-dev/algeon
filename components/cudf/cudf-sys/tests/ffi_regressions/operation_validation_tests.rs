use std::{os::raw::c_void, ptr};

use crate::{
    helpers::*,
    support::{last_error_message, lock_shim_state},
};

#[test]
fn test_column_view_strings_to_host_empty_rejects_undersized_offsets_buffer() {
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let mut offsets = vec![-1i32; 1];
    let mut total_chars = -1;

    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(view, offsets.as_mut_ptr(), 0, ptr::null_mut(), 0, &mut total_chars)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert_eq!(offsets, vec![-1]);
    assert_eq!(total_chars, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_concatenate_columns_null_element_returns_null_error() {
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);

    let views = [view, ptr::null_mut()];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_concatenate_columns(views.as_ptr(), views.len() as i32, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_concatenate_tables_null_element_returns_null_error() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let tables = [tv, ptr::null_mut()];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_concatenate_tables(tables.as_ptr(), tables.len() as i32, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_sort_negative_num_keys_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_sort(tv, ptr::null(), ptr::null(), -1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_sort_invalid_order_enum_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let column_order = [99i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sort_raw(tv, column_order.as_ptr(), ptr::null(), 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_sort_invalid_null_order_enum_rejected() {
    let col = make_i32_column(&[3, 2, 1]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let null_order = [99i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sort_raw(tv, ptr::null(), null_order.as_ptr(), 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_drop_nulls_negative_keep_threshold_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls(tv, ptr::null(), 0, -1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_drop_nulls_negative_num_keys_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls(tv, keys.as_ptr(), -1, 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_distinct_negative_num_keys_rejected() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), -1, 0, 0, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_gather_invalid_bounds_policy_rejected() {
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let map_col = make_i32_column(&[0, 2]);
    let map_view = make_column_view(map_col);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_gather_raw(tv, map_view, 99, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(map_view);
        cudf_sys::cudf_column_destroy(map_col);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_distinct_invalid_policy_values_rejected() {
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let keys = [0i32];
    let mut out: *mut c_void = ptr::null_mut();

    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 99, 0, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 0, 99, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 0, 0, 99, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_column_create_numeric_negative_size_rejected() {
    let data: [i32; 1] = [42];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            -1,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_column_create_numeric_null_count_out_of_range_rejected() {
    let data = [1i32, 2, 3];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            4,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_column_create_numeric_null_count_requires_mask() {
    let data = [1i32, 2, 3];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            2,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_column_view_to_host_string_type_rejected() {
    let strings = [std::ffi::CString::new("ab").unwrap(), std::ffi::CString::new("cd").unwrap()];
    let string_ptrs = [strings[0].as_ptr(), strings[1].as_ptr()];
    let lengths = [2i32, 2i32];

    let mut col: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(string_ptrs.as_ptr(), lengths.as_ptr(), 2, ptr::null(), 0, &mut col)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!col.is_null());

    let view = make_column_view(col);
    let mut buf = [0u8; 16];
    let status = unsafe { cudf_sys::cudf_column_view_to_host(view, buf.as_mut_ptr() as *mut c_void, buf.len()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_TYPE);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_create_strings_negative_length_rejected() {
    let strings = [std::ffi::CString::new("ab").unwrap()];
    let string_ptrs = [strings[0].as_ptr()];
    let lengths = [-1i32];

    let mut col: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(string_ptrs.as_ptr(), lengths.as_ptr(), 1, ptr::null(), 0, &mut col)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(col.is_null());
}

#[test]
fn test_column_create_strings_positive_length_requires_non_null_string_pointer() {
    let string_ptrs = [ptr::null()];
    let lengths = [2i32];

    let mut col: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(string_ptrs.as_ptr(), lengths.as_ptr(), 1, ptr::null(), 0, &mut col)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(col.is_null());
}

#[test]
fn test_drop_nulls_zero_num_keys_allows_null_keys_pointer() {
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_drop_nulls(tv, ptr::null(), 0, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_distinct_zero_num_keys_allows_null_keys_pointer() {
    let col = make_i32_column(&[1, 2, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct(
            tv,
            ptr::null(),
            0,
            cudf_sys::cudf_duplicate_keep_option_t::CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY,
            cudf_sys::cudf_stream_compaction_null_equality_t::CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL,
            cudf_sys::cudf_stream_compaction_nan_equality_t::CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL,
            &mut out,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    unsafe {
        cudf_sys::cudf_table_destroy(out);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_groupby_negative_num_requests_rejected() {
    let key_col = make_i32_column(&[1, 1, 2]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = ptr::null_mut();
    let mut out_total: i32 = -1;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys_view,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            -1,
            0,
            cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    unsafe {
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

#[test]
fn test_groupby_aggregation_count_mismatch_rejected() {
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);

    let values = [val_view];
    let agg_counts = [1i32];
    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::cudf_aggregation_kind_t::CUDF_AGG_SUM, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());
    let aggregations = [agg];

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = ptr::null_mut();
    let mut out_total: i32 = -1;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys_view,
            values.as_ptr(),
            aggregations.as_ptr(),
            agg_counts.as_ptr(),
            1,
            0,
            cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    unsafe {
        cudf_sys::cudf_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(val_view);
        cudf_sys::cudf_column_destroy(val_col);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

#[test]
fn test_aggregation_create_count_invalid_null_policy_rejected() {
    let _shim_state = lock_shim_state();
    let mut agg: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_aggregation_create_count_raw(99, &mut agg) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(agg.is_null());
    assert!(last_error_message().contains("invalid null_policy"));
}

#[test]
fn test_groupby_invalid_keys_null_policy_rejected() {
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);
    let values = [val_view];
    let agg_counts = [1i32];

    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::cudf_aggregation_kind_t::CUDF_AGG_SUM, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());
    let aggregations = [agg];

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = ptr::null_mut();
    let mut out_total: i32 = -1;
    let status = unsafe {
        cudf_groupby_aggregate_raw(
            keys_view,
            values.as_ptr(),
            aggregations.as_ptr(),
            agg_counts.as_ptr(),
            1,
            1,
            99,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);
    assert!(last_error_message().contains("invalid keys_null_policy"));

    unsafe {
        cudf_sys::cudf_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(val_view);
        cudf_sys::cudf_column_destroy(val_col);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

#[test]
fn test_groupby_zero_aggregations_returns_null_output_array() {
    let key_col = make_i32_column(&[1, 1, 2]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);

    let mut out_keys: *mut c_void = ptr::null_mut();
    let mut out_vals: *mut *mut c_void = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_total: i32 = -1;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate(
            keys_view,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            0,
            cudf_sys::cudf_null_policy_t::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    unsafe {
        cudf_sys::cudf_table_destroy(out_keys);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

fn assert_join_invalid_null_equality_rejected(
    join_fn: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        i32,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
) {
    let left_key_col = make_i32_column(&[1, 2]);
    let right_key_col = make_i32_column(&[2, 3]);
    let left_keys_tbl = make_table_from_one_column(left_key_col);
    let right_keys_tbl = make_table_from_one_column(right_key_col);
    let left_keys_view = make_table_view(left_keys_tbl);
    let right_keys_view = make_table_view(right_keys_tbl);
    let left_payload_tbl = make_empty_table();
    let right_payload_tbl = make_empty_table();
    let left_payload_view = make_table_view(left_payload_tbl);
    let right_payload_view = make_table_view(right_payload_tbl);

    let mut out_left: *mut c_void = ptr::null_mut();
    let mut out_right: *mut c_void = ptr::null_mut();
    let status = unsafe {
        join_fn(
            left_keys_view,
            right_keys_view,
            left_payload_view,
            right_payload_view,
            99,
            &mut out_left,
            &mut out_right,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_left.is_null());
    assert!(out_right.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(right_payload_view);
        cudf_sys::cudf_table_view_destroy(left_payload_view);
        cudf_sys::cudf_table_view_destroy(right_keys_view);
        cudf_sys::cudf_table_view_destroy(left_keys_view);
        cudf_sys::cudf_table_destroy(right_payload_tbl);
        cudf_sys::cudf_table_destroy(left_payload_tbl);
        cudf_sys::cudf_table_destroy(right_keys_tbl);
        cudf_sys::cudf_table_destroy(left_keys_tbl);
    }
}

#[test]
fn test_inner_join_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_invalid_null_equality_rejected(cudf_inner_join_raw);
}

#[test]
fn test_left_join_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_invalid_null_equality_rejected(cudf_left_join_raw);
}

#[test]
fn test_full_join_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_invalid_null_equality_rejected(cudf_full_join_raw);
}

#[test]
fn test_distinct_hash_join_create_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_distinct_hash_join_create_raw(tv, 99, &mut join) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(join.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_distinct_hash_join_left_join_returns_build_indices() {
    let build_col = make_i32_column(&[10, 20, 30]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);

    let mut join: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_hash_join_create(
            build_tv,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            &mut join,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_columns(vec![make_i32_column(&[20, 99, 10, 30])]);
    let probe_tv = make_table_view(probe_tbl);
    let mut out: *mut c_void = ptr::null_mut();

    let status = unsafe { cudf_distinct_hash_join_left_join_raw(join, probe_tv, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(read_i32_column_data(out, 4), vec![1, i32::MIN, 0, 2]);

    unsafe {
        cudf_sys::cudf_column_destroy(out);
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_distinct_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

#[test]
fn test_scan_invalid_scan_type_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);
    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scan_aggregation_create(cudf_sys::cudf_reduce_aggregation_kind_t::CUDF_REDUCE_AGG_SUM, &mut agg)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_scan_raw(view, agg, 99, 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid scan_type"));

    unsafe {
        cudf_sys::cudf_scan_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scan_invalid_null_policy_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);
    let mut agg: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scan_aggregation_create(cudf_sys::cudf_reduce_aggregation_kind_t::CUDF_REDUCE_AGG_SUM, &mut agg)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());

    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_scan(view, agg, cudf_sys::cudf_scan_type_t::CUDF_SCAN_INCLUSIVE, 99, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid null_policy"));

    unsafe {
        cudf_sys::cudf_scan_aggregation_destroy(agg);
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_unary_operation_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let view = make_column_view(col);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_unary_operation_raw(view, 99, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid unary operator"));

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_binary_operation_vv_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let lhs_col = make_i32_column(&[1, 2, 3]);
    let rhs_col = make_i32_column(&[4, 5, 6]);
    let lhs_view = make_column_view(lhs_col);
    let rhs_view = make_column_view(rhs_col);
    let dtype = make_i32_data_type();

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_binary_operation_vv_raw(lhs_view, rhs_view, 99, dtype, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid binary operator"));

    unsafe {
        cudf_sys::cudf_data_type_destroy(dtype);
        cudf_sys::cudf_column_view_destroy(rhs_view);
        cudf_sys::cudf_column_view_destroy(lhs_view);
        cudf_sys::cudf_column_destroy(rhs_col);
        cudf_sys::cudf_column_destroy(lhs_col);
    }
}

#[test]
fn test_binary_operation_vs_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let lhs_col = make_i32_column(&[1, 2, 3]);
    let lhs_view = make_column_view(lhs_col);
    let rhs_scalar = make_i32_scalar(4);
    let dtype = make_i32_data_type();

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_binary_operation_vs_raw(lhs_view, rhs_scalar, 99, dtype, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid binary operator"));

    unsafe {
        cudf_sys::cudf_data_type_destroy(dtype);
        cudf_sys::cudf_scalar_destroy(rhs_scalar);
        cudf_sys::cudf_column_view_destroy(lhs_view);
        cudf_sys::cudf_column_destroy(lhs_col);
    }
}

#[test]
fn test_binary_operation_sv_invalid_operator_rejected() {
    let _shim_state = lock_shim_state();
    let lhs_scalar = make_i32_scalar(4);
    let rhs_col = make_i32_column(&[1, 2, 3]);
    let rhs_view = make_column_view(rhs_col);
    let dtype = make_i32_data_type();

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_binary_operation_sv_raw(lhs_scalar, rhs_view, 99, dtype, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid binary operator"));

    unsafe {
        cudf_sys::cudf_data_type_destroy(dtype);
        cudf_sys::cudf_column_view_destroy(rhs_view);
        cudf_sys::cudf_column_destroy(rhs_col);
        cudf_sys::cudf_scalar_destroy(lhs_scalar);
    }
}
