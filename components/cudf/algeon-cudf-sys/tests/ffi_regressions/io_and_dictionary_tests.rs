use std::{
    ffi::CStr,
    mem::{align_of, offset_of, size_of},
    os::raw::c_char,
    ptr,
};

use cudf_sys::test_support;

use crate::{
    helpers::*,
    support::{
        OwnedMemoryResource, OwnedStream, last_error_message, last_exec_mr_raw, lock_shim_state, required_exec_options,
    },
};

fn io_exec_options() -> (OwnedStream, OwnedMemoryResource, cudf_sys::cudf_exec_options_t) {
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    (stream, mr, exec)
}

#[test]
fn test_parquet_pruning_evidence_ffi_layout() {
    type Evidence = cudf_sys::cudf_io_parquet_pruning_evidence_t;

    assert_eq!(size_of::<Evidence>(), 40);
    assert_eq!(align_of::<Evidence>(), 8);
    assert_eq!(offset_of!(Evidence, input_row_groups), 0);
    assert_eq!(offset_of!(Evidence, has_row_groups_after_stats_filter), 8);
    assert_eq!(offset_of!(Evidence, row_groups_after_stats_filter), 16);
    assert_eq!(offset_of!(Evidence, has_row_groups_after_bloom_filter), 24);
    assert_eq!(offset_of!(Evidence, row_groups_after_bloom_filter), 32);
}

#[test]
fn test_error_detail_ffi_layout() {
    type Detail = cudf_sys::cudf_error_detail_t;

    assert_eq!(size_of::<Detail>(), 568);
    assert_eq!(align_of::<Detail>(), 8);
    assert_eq!(offset_of!(Detail, status), 0);
    assert_eq!(offset_of!(Detail, failure_source), 4);
    assert_eq!(offset_of!(Detail, domain_id), 8);
    assert_eq!(offset_of!(Detail, member_id), 16);
    assert_eq!(offset_of!(Detail, generation), 24);
    assert_eq!(offset_of!(Detail, requested_bytes), 32);
    assert_eq!(offset_of!(Detail, alignment), 40);
    assert_eq!(offset_of!(Detail, cuda_error), 48);
    assert_eq!(offset_of!(Detail, message), 52);
    assert_eq!(size_of::<[c_char; cudf_sys::CUDF_ERROR_MESSAGE_CAPACITY as usize]>(), 512);
}

#[test]
fn test_table_view_null_handle_clears_out() {
    let _shim_state = lock_shim_state();
    let mut out = std::ptr::dangling_mut::<cudf_sys::cudf_table_view_handle_state>();
    let status = unsafe { cudf_sys::cudf_table_view(ptr::null_mut(), &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_missing_file_clears_out() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out = std::ptr::dangling_mut::<cudf_sys::cudf_table_handle_state>();
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_ex(path.as_ptr(), ptr::null_mut(), 0, 0, -1, ptr::null(), 0, &mut out, &exec)
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_IO);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_from_host_buffer_invalid_args_clear_out() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let mut out = std::ptr::dangling_mut::<cudf_sys::cudf_table_handle_state>();
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_from_host_buffer_ex(
            ptr::null_mut(),
            16,
            ptr::null_mut(),
            0,
            0,
            -1,
            ptr::null_mut(),
            0,
            &mut out,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_from_host_buffer_empty_buffer_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let bytes = [0u8; 1];
    let mut out = std::ptr::dangling_mut::<cudf_sys::cudf_table_handle_state>();
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_from_host_buffer_ex(
            bytes.as_ptr().cast(),
            0,
            ptr::null_mut(),
            0,
            0,
            -1,
            ptr::null_mut(),
            0,
            &mut out,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_non_io_runtime_error_maps_to_unknown() {
    let _shim_state = lock_shim_state();
    let status = unsafe { test_support::cudf_test_throw_runtime_error() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_UNKNOWN);
    assert!(last_error_message().contains("non-io runtime failure"));
}

#[test]
fn test_bad_alloc_maps_to_out_of_memory() {
    let _shim_state = lock_shim_state();
    let status = unsafe { test_support::cudf_test_throw_bad_alloc() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(!last_error_message().is_empty());
}

#[test]
fn test_bad_alloc_publishes_copyable_typed_error_detail() {
    let _shim_state = lock_shim_state();
    let status = unsafe { test_support::cudf_test_throw_bad_alloc() };
    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);

    let detail = unsafe {
        let ptr = cudf_sys::cudf_last_error_detail();
        assert!(!ptr.is_null());
        ptr::read(ptr)
    };
    assert_eq!(detail.status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert_eq!(detail.failure_source, cudf_sys::CUDF_ERROR_FAILURE_SOURCE_UNKNOWN);
    assert_eq!(detail.domain_id, 0);
    assert_eq!(detail.member_id, 0);
    assert_eq!(detail.generation, 0);
    assert_eq!(detail.requested_bytes, 0);
    assert_eq!(detail.alignment, 0);
    assert_eq!(detail.cuda_error, -1);
    let message = unsafe { CStr::from_ptr(detail.message.as_ptr()) }.to_string_lossy();
    assert!(!message.is_empty());

    unsafe { cudf_sys::cudf_clear_error() };
    assert!(unsafe { cudf_sys::cudf_last_error_detail() }.is_null());
}

#[test]
fn test_invalid_argument_maps_to_invalid_argument() {
    let _shim_state = lock_shim_state();
    let status = unsafe { test_support::cudf_test_throw_invalid_argument() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(last_error_message().contains("invalid argument"));
}

#[test]
fn test_owning_handle_size_arithmetic_overflow_is_typed_and_does_not_wrap() {
    let _shim_state = lock_shim_state();
    let mut out = usize::MAX;

    let add_status = unsafe { test_support::cudf_test_checked_owning_handle_size_add(usize::MAX, 1, &mut out) };
    assert_eq!(add_status, cudf_sys::CUDF_ERROR_SIZE_OVERFLOW);
    assert_eq!(out, 0, "failed checked addition must leave a deterministic output");
    assert!(last_error_message().contains("overflow"));

    out = usize::MAX;
    let mul_status = unsafe { test_support::cudf_test_checked_owning_handle_size_mul(usize::MAX, 2, &mut out) };
    assert_eq!(mul_status, cudf_sys::CUDF_ERROR_SIZE_OVERFLOW);
    assert_eq!(out, 0, "failed checked multiplication must leave a deterministic output");
    assert!(last_error_message().contains("overflow"));
}

#[test]
fn test_io_read_parquet_with_metadata_invalid_args_clear_outputs() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out_table = std::ptr::dangling_mut::<cudf_sys::cudf_table_handle_state>();
    let mut out_names = std::ptr::dangling_mut::<*mut c_char>();
    let mut out_num_names = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata_ex(
            path.as_ptr(),
            ptr::null_mut(),
            -1,
            0,
            -1,
            ptr::null_mut(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
}

#[test]
fn test_io_read_parquet_multi_mixed_all_local_flags_allow_null_profiles() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut filepaths = [path.as_ptr()];
    let remote_source_flags = [0_i32];
    let mut out_table = std::ptr::dangling_mut::<cudf_sys::cudf_table_handle_state>();
    let mut out_names = std::ptr::dangling_mut::<*mut c_char>();
    let mut out_num_names = -1;
    let mut evidence = cudf_sys::cudf_io_remote_parquet_read_evidence_t {
        backend: -99,
        fallback_reason: -99,
        supports_device_read: -99,
        device_read_used: -99,
        remote_bytes: -99,
        remote_range_count: -99,
        host_read_bytes: -99,
        host_read_count: -99,
        device_read_bytes: -99,
        device_read_count: -99,
    };
    let mut pruning_evidence = cudf_sys::cudf_io_parquet_pruning_evidence_t {
        input_row_groups: -99,
        has_row_groups_after_stats_filter: -99,
        row_groups_after_stats_filter: -99,
        has_row_groups_after_bloom_filter: -99,
        row_groups_after_bloom_filter: -99,
    };

    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex(
            filepaths.as_mut_ptr(),
            remote_source_flags.as_ptr(),
            ptr::null_mut(),
            filepaths.len() as i32,
            ptr::null_mut(),
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null_mut(),
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
            &mut evidence,
            &mut pruning_evidence,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_IO);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
    assert_eq!(evidence.backend, 0);
    assert_eq!(evidence.fallback_reason, 1);
    assert_eq!(pruning_evidence.input_row_groups, -1);
}

#[test]
fn test_io_read_parquet_from_host_buffer_with_metadata_invalid_args_clear_outputs() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let bytes = [0u8; 8];
    let mut out_table = std::ptr::dangling_mut::<cudf_sys::cudf_table_handle_state>();
    let mut out_names = std::ptr::dangling_mut::<*mut c_char>();
    let mut out_num_names = -1;
    let mut pruning_evidence = cudf_sys::cudf_io_parquet_pruning_evidence_t {
        input_row_groups: -99,
        has_row_groups_after_stats_filter: -99,
        row_groups_after_stats_filter: -99,
        has_row_groups_after_bloom_filter: -99,
        row_groups_after_bloom_filter: -99,
    };
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
            bytes.as_ptr().cast(),
            bytes.len(),
            ptr::null_mut(),
            -1,
            0,
            -1,
            ptr::null(),
            0,
            ptr::null_mut(),
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
            &mut pruning_evidence,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
    assert_eq!(pruning_evidence.input_row_groups, -1);
}

#[test]
fn test_io_read_parquet_negative_num_columns_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_ex(path.as_ptr(), ptr::null_mut(), -1, 0, -1, ptr::null(), 0, &mut out, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_null_column_name_entry_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut columns = [ptr::null()];
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_ex(
            path.as_ptr(),
            columns.as_mut_ptr(),
            1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_negative_skip_rows_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_ex(path.as_ptr(), ptr::null_mut(), 0, -1, -1, ptr::null(), 0, &mut out, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_invalid_num_rows_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out: cudf_sys::cudf_table_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_ex(path.as_ptr(), ptr::null_mut(), 0, 0, -2, ptr::null(), 0, &mut out, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_write_parquet_negative_num_columns_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");

    let status = unsafe { cudf_sys::cudf_io_write_parquet_ex(tv, path.as_ptr(), ptr::null_mut(), -1, 0, -1, &exec) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_write_parquet_null_column_name_entry_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");
    let mut names = [ptr::null()];

    let status = unsafe { cudf_sys::cudf_io_write_parquet_ex(tv, path.as_ptr(), names.as_mut_ptr(), 1, 0, -1, &exec) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_write_parquet_ex_uses_explicit_memory_resource_for_writer_temporaries() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let col = make_i32_column(&[10, 20, 30, 40, 50, 60]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");
    let path_buf = std::path::PathBuf::from(path.as_c_str().to_string_lossy().into_owned());
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let status = unsafe { cudf_sys::cudf_io_write_parquet_ex(tv, path.as_ptr(), ptr::null_mut(), 0, 0, -1, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{}", last_error_message());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());

    stream.synchronize();
    let _ = std::fs::remove_file(path_buf);
    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_probe_path_datasource_reports_local_file_capability_and_size() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");
    let path_buf = std::path::PathBuf::from(path.as_c_str().to_string_lossy().into_owned());

    let status = unsafe { cudf_sys::cudf_io_write_parquet_ex(tv, path.as_ptr(), ptr::null_mut(), 0, 0, -1, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{}", last_error_message());

    let mut probe = cudf_sys::cudf_io_datasource_probe_result_t {
        can_open: -1,
        supports_device_read: -1,
        device_read_preferred: -1,
        size_bytes: -1,
    };
    let status = unsafe { cudf_sys::cudf_io_probe_path_datasource(path.as_ptr(), 4096, &mut probe) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{}", last_error_message());
    assert_eq!(probe.can_open, 1);
    assert!(probe.size_bytes > 0);
    assert_eq!(probe.size_bytes as u64, std::fs::metadata(&path_buf).unwrap().len());
    assert!(matches!(probe.supports_device_read, 0 | 1));
    assert!(matches!(probe.device_read_preferred, 0 | 1));

    let mut null_path_probe = cudf_sys::cudf_io_datasource_probe_result_t {
        can_open: -1,
        supports_device_read: -1,
        device_read_preferred: -1,
        size_bytes: -2,
    };
    let status = unsafe { cudf_sys::cudf_io_probe_path_datasource(ptr::null(), 4096, &mut null_path_probe) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert_eq!(null_path_probe.can_open, 0);
    assert_eq!(null_path_probe.supports_device_read, 0);
    assert_eq!(null_path_probe.device_read_preferred, 0);
    assert_eq!(null_path_probe.size_bytes, -1);

    let _ = std::fs::remove_file(path_buf);
    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_read_parquet_with_metadata_negative_num_columns_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out_table: cudf_sys::cudf_table_handle = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata_ex(
            path.as_ptr(),
            ptr::null_mut(),
            -1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_io_read_parquet_with_metadata_null_column_name_entry_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut columns = [ptr::null()];
    let mut out_table: cudf_sys::cudf_table_handle = std::ptr::dangling_mut::<cudf_sys::cudf_table_handle_state>();
    let mut out_names: *mut *mut i8 = std::ptr::dangling_mut::<*mut i8>();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata_ex(
            path.as_ptr(),
            columns.as_mut_ptr(),
            1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
}

#[test]
fn test_io_read_parquet_with_metadata_negative_skip_rows_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out_table: cudf_sys::cudf_table_handle = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata_ex(
            path.as_ptr(),
            ptr::null_mut(),
            0,
            -1,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_io_read_parquet_with_metadata_invalid_num_rows_rejected() {
    let _shim_state = lock_shim_state();
    let (_stream, _mr, exec) = io_exec_options();
    let path = unique_missing_path("parquet");
    let mut out_table: cudf_sys::cudf_table_handle = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata_ex(
            path.as_ptr(),
            ptr::null_mut(),
            0,
            0,
            -2,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_dictionary_match_dictionaries_invalid_num_columns_clears_outputs() {
    let _shim_state = lock_shim_state();
    let dict_columns = std::ptr::NonNull::<cudf_sys::cudf_column_view_handle>::dangling().as_ptr();
    let mut out_columns = std::ptr::NonNull::<cudf_sys::cudf_column_handle>::dangling().as_ptr();
    let mut out_count: i32 = -1;

    let (_stream, _mr, exec) = io_exec_options();
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries_ex(dict_columns, 0, &mut out_columns, &mut out_count, &exec)
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);
}

#[test]
fn test_replace_nulls_policy_rejects_invalid_policy_value() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column_nullable(&[1, 0, 3], &[true, false, true]);
    let view = make_column_view(col);

    let (_stream, _mr, exec) = io_exec_options();
    let mut out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe { cudf_sys::cudf_replace_nulls_policy_ex(view, 42, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null(), "invalid policy must not produce an output column");
    assert!(
        last_error_message().contains("policy"),
        "expected invalid policy error message, got: {}",
        last_error_message()
    );

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_dictionary_match_dictionaries_null_entry_clears_outputs() {
    let _shim_state = lock_shim_state();
    let dict = make_dictionary_i32_column(&[1, 2, 1]);
    let dict_view = make_column_view(dict);
    let dict_columns = [dict_view, ptr::null_mut()];

    let mut out_columns = std::ptr::NonNull::<cudf_sys::cudf_column_handle>::dangling().as_ptr();
    let mut out_count: i32 = -1;
    let (_stream, _mr, exec) = io_exec_options();
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries_ex(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(dict_view);
        cudf_sys::cudf_column_destroy(dict);
    }
}

#[test]
fn test_dictionary_match_dictionaries_returns_heap_array() {
    let _shim_state = lock_shim_state();
    let left = make_dictionary_i32_column(&[10, 20, 10, 30]);
    let right = make_dictionary_i32_column(&[20, 30, 40, 20]);
    let left_view = make_column_view(left);
    let right_view = make_column_view(right);
    let dict_columns = [left_view, right_view];
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries_ex(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, dict_columns.len() as i32);
    assert!(!out_columns.is_null());

    let matched = unsafe { std::slice::from_raw_parts(out_columns, out_count as usize) };
    stream.synchronize();
    assert_eq!(decode_dictionary_i32_column(matched[0], 4), vec![10, 20, 10, 30]);
    assert_eq!(decode_dictionary_i32_column(matched[1], 4), vec![20, 30, 40, 20]);

    unsafe {
        for &col in matched {
            cudf_sys::cudf_column_destroy(col);
        }
        cudf_sys::cudf_free_column_array(out_columns);
        cudf_sys::cudf_column_view_destroy(right_view);
        cudf_sys::cudf_column_view_destroy(left_view);
        cudf_sys::cudf_column_destroy(right);
        cudf_sys::cudf_column_destroy(left);
    }
}
