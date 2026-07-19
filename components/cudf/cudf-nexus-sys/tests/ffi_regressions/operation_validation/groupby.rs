use super::*;

#[test]
fn test_groupby_negative_num_requests_rejected() {
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);

    let mut out_keys: cudf_sys::cudf_table_handle = ptr::null_mut();
    let mut out_vals: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_total: i32 = -1;

    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate_ex(
            keys_view,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            -1,
            0,
            cudf_sys::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
            &exec,
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
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);

    let values = [val_view];
    let agg_counts = [1i32];
    let mut agg: cudf_sys::cudf_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::CUDF_AGG_SUM, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());
    let aggregations = [agg];

    let mut out_keys: cudf_sys::cudf_table_handle = ptr::null_mut();
    let mut out_vals: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_total: i32 = -1;

    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate_ex(
            keys_view,
            values.as_ptr(),
            aggregations.as_ptr(),
            agg_counts.as_ptr(),
            1,
            0,
            cudf_sys::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
            &exec,
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
    let mut agg: cudf_sys::cudf_aggregation_handle = ptr::null_mut();
    let status = unsafe { cudf_aggregation_create_count_raw(99, raw_out_handle(&mut agg)) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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

    let mut agg: cudf_sys::cudf_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::CUDF_AGG_SUM, &mut agg) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!agg.is_null());
    let aggregations = [agg];

    let mut out_keys: cudf_sys::cudf_table_handle = ptr::null_mut();
    let mut out_vals: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_total: i32 = -1;
    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe {
        cudf_groupby_aggregate_raw(
            raw_handle(keys_view),
            values.as_ptr().cast(),
            aggregations.as_ptr().cast(),
            agg_counts.as_ptr(),
            1,
            1,
            99,
            raw_out_handle(&mut out_keys),
            (&mut out_vals as *mut *mut cudf_sys::cudf_column_handle).cast(),
            &mut out_total,
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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
    let _shim_state = lock_shim_state();
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let key_col = make_i32_column(&[1, 1, 2]);
    let keys_tbl = make_table_from_one_column(key_col);
    let keys_view = make_table_view(keys_tbl);

    let mut out_keys: cudf_sys::cudf_table_handle = ptr::null_mut();
    let mut out_vals = std::ptr::NonNull::<cudf_sys::cudf_column_handle>::dangling().as_ptr();
    let mut out_total: i32 = -1;

    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate_ex(
            keys_view,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            0,
            cudf_sys::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_destroy(out_keys);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

#[test]
fn test_groupby_aggregate_publish_failure_does_not_publish_partial_outputs() {
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_columns(vec![key_col]);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);
    let values = [val_view];
    let agg_counts = [2i32];

    let mut sum: cudf_sys::cudf_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::CUDF_AGG_SUM, &mut sum) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let mut min: cudf_sys::cudf_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::CUDF_AGG_MIN, &mut min) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let aggregations = [sum, min];

    let mut out_keys = std::ptr::NonNull::<cudf_sys::cudf_table_handle_state>::dangling().as_ptr();
    let mut out_vals = std::ptr::NonNull::<cudf_sys::cudf_column_handle>::dangling().as_ptr();
    let mut out_total: i32 = -1;
    let (_stream, _mr, exec) = strict_exec_owners();

    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(1) };
    let status = unsafe {
        cudf_sys::cudf_groupby_aggregate_ex(
            keys_view,
            values.as_ptr(),
            aggregations.as_ptr(),
            agg_counts.as_ptr(),
            1,
            2,
            cudf_sys::CUDF_NULL_POLICY_EXCLUDE,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
            &exec,
        )
    };
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(-1) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    unsafe {
        cudf_sys::cudf_aggregation_destroy(sum);
        cudf_sys::cudf_aggregation_destroy(min);
        cudf_sys::cudf_column_view_destroy(val_view);
        cudf_sys::cudf_column_destroy(val_col);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}

#[test]
fn test_prepared_groupby_aggregate_publish_failure_does_not_publish_partial_outputs() {
    let _shim_state = lock_shim_state();
    let key_col = make_i32_column(&[1, 1, 2]);
    let val_col = make_i32_column(&[10, 20, 30]);
    let keys_tbl = make_table_from_columns(vec![key_col]);
    let keys_view = make_table_view(keys_tbl);
    let val_view = make_column_view(val_col);
    let values = [val_view];
    let agg_counts = [2i32];

    let mut sum: cudf_sys::cudf_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::CUDF_AGG_SUM, &mut sum) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let mut min: cudf_sys::cudf_aggregation_handle = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_aggregation_create(cudf_sys::CUDF_AGG_MIN, &mut min) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let aggregations = [sum, min];

    let (_stream, _mr, exec) = strict_exec_owners();
    let mut prepared: cudf_sys::cudf_prepared_groupby_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_prepared_groupby_create_ex(
            keys_view,
            cudf_sys::CUDF_NULL_POLICY_EXCLUDE,
            0,
            &mut prepared,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!prepared.is_null());

    let mut out_keys = std::ptr::NonNull::<cudf_sys::cudf_table_handle_state>::dangling().as_ptr();
    let mut out_vals = std::ptr::NonNull::<cudf_sys::cudf_column_handle>::dangling().as_ptr();
    let mut out_total: i32 = -1;

    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(1) };
    let status = unsafe {
        cudf_sys::cudf_prepared_groupby_aggregate_ex(
            prepared,
            values.as_ptr(),
            aggregations.as_ptr(),
            agg_counts.as_ptr(),
            1,
            2,
            &mut out_keys,
            &mut out_vals,
            &mut out_total,
            &exec,
        )
    };
    unsafe { cudf_sys::test_support::cudf_test_fail_column_handle_release_after(-1) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out_keys.is_null());
    assert!(out_vals.is_null());
    assert_eq!(out_total, 0);

    unsafe {
        cudf_sys::cudf_prepared_groupby_destroy(prepared);
        cudf_sys::cudf_aggregation_destroy(sum);
        cudf_sys::cudf_aggregation_destroy(min);
        cudf_sys::cudf_column_view_destroy(val_view);
        cudf_sys::cudf_column_destroy(val_col);
        cudf_sys::cudf_table_view_destroy(keys_view);
        cudf_sys::cudf_table_destroy(keys_tbl);
    }
}
