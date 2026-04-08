mod support;

use std::{
    os::raw::{c_char, c_void},
    ptr,
};

// Re-declare selected FFI functions with `i32` parameters so that tests can
// pass deliberately invalid enum discriminants to the C shim without
// constructing invalid Rust enum values (which is instant UB).
#[allow(dead_code)]
extern "C" {
    #[link_name = "cudf_sort"]
    fn cudf_sort_raw(
        input: *mut c_void,
        column_order: *const i32,
        null_precedence: *const i32,
        num_key_columns: i32,
        out: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_distinct"]
    fn cudf_distinct_raw(
        input: *mut c_void,
        keys: *const i32,
        num_keys: i32,
        keep: i32,
        null_equality: i32,
        nan_equality: i32,
        out: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_gather"]
    fn cudf_gather_raw(input: *mut c_void, gather_map: *mut c_void, bounds_policy: i32, out: *mut *mut c_void) -> i32;
    #[link_name = "cudf_aggregation_create_count"]
    fn cudf_aggregation_create_count_raw(null_policy: i32, out: *mut *mut c_void) -> i32;
    #[link_name = "cudf_groupby_aggregate"]
    fn cudf_groupby_aggregate_raw(
        keys: *mut c_void,
        values: *const *mut c_void,
        aggregations: *const *mut c_void,
        agg_counts: *const i32,
        num_requests: i32,
        num_aggregations: i32,
        keys_null_policy: i32,
        out_keys: *mut *mut c_void,
        out_values: *mut *mut *mut c_void,
        out_total_columns: *mut i32,
    ) -> i32;
    #[link_name = "cudf_inner_join"]
    fn cudf_inner_join_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        left_payload: *mut c_void,
        right_payload: *mut c_void,
        null_equality: i32,
        out_left: *mut *mut c_void,
        out_right: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_left_join"]
    fn cudf_left_join_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        left_payload: *mut c_void,
        right_payload: *mut c_void,
        null_equality: i32,
        out_left: *mut *mut c_void,
        out_right: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_full_join"]
    fn cudf_full_join_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        left_payload: *mut c_void,
        right_payload: *mut c_void,
        null_equality: i32,
        out_left: *mut *mut c_void,
        out_right: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_scan_aggregation_create"]
    fn cudf_scan_aggregation_create_raw(kind: i32, out: *mut *mut c_void) -> i32;
    #[link_name = "cudf_scan"]
    fn cudf_scan_raw(
        input: *mut c_void,
        agg: *mut c_void,
        scan_type: i32,
        null_policy: i32,
        out: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_unary_operation"]
    fn cudf_unary_operation_raw(input: *mut c_void, op: i32, out: *mut *mut c_void) -> i32;
    #[link_name = "cudf_binary_operation_vv"]
    fn cudf_binary_operation_vv_raw(
        lhs: *mut c_void,
        rhs: *mut c_void,
        op: i32,
        output_type: *mut c_void,
        out: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_binary_operation_vs"]
    fn cudf_binary_operation_vs_raw(
        lhs: *mut c_void,
        rhs: *mut c_void,
        op: i32,
        output_type: *mut c_void,
        out: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_binary_operation_sv"]
    fn cudf_binary_operation_sv_raw(
        lhs: *mut c_void,
        rhs: *mut c_void,
        op: i32,
        output_type: *mut c_void,
        out: *mut *mut c_void,
    ) -> i32;
}

use support::{
    NO_SYNC_RECORDED, OwnedEvent, OwnedStream, current_device_resource_raw, default_stream_value, last_error_message,
    last_exec_mr_raw, last_exec_stream_value, last_exec_sync_stream_value, lock_shim_state, make_stats_memory_resource,
    owned_string_value, reset_exec_observation, stream_value,
};

fn make_i32_column(data: &[i32]) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn validity_to_bitmask(validity: &[bool]) -> Vec<u8> {
    let mut mask = vec![0u8; validity.len().div_ceil(8)];
    for (index, is_valid) in validity.iter().copied().enumerate() {
        if is_valid {
            mask[index / 8] |= 1 << (index % 8);
        }
    }
    mask
}

fn make_i32_column_nullable(data: &[i32], validity: &[bool]) -> *mut c_void {
    assert_eq!(data.len(), validity.len());
    let null_mask = validity_to_bitmask(validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;

    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_string_column(data: &[&str]) -> *mut c_void {
    let c_strings: Vec<_> = data.iter().map(|value| std::ffi::CString::new(*value).unwrap()).collect();
    let string_ptrs: Vec<_> = c_strings.iter().map(|value| value.as_ptr()).collect();
    let lengths: Vec<_> = data.iter().map(|value| value.len() as i32).collect();

    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_string_scalar(value: &str) -> *mut c_void {
    let c_value = std::ffi::CString::new(value).unwrap();
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(c_value.as_ptr(), 1, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_empty_i32_column() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_create_empty(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_empty_table() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(ptr::null_mut(), 0, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_column_view(col: *mut c_void) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view(col, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_table_from_one_column(col: *mut c_void) -> *mut c_void {
    let mut col_arr = [col];
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(col_arr.as_mut_ptr(), 1, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    // ownership moved into table
    assert!(col_arr[0].is_null());
    out
}

fn read_i32_column_data(col: *mut c_void, expected_len: usize) -> Vec<i32> {
    let view = make_column_view(col);
    let mut buf = vec![0i32; expected_len];
    let buf_size = expected_len * std::mem::size_of::<i32>();
    let st = unsafe { cudf_sys::cudf_column_view_to_host(view, buf.as_mut_ptr() as *mut c_void, buf_size) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    buf
}

fn read_string_column_data(col: *mut c_void) -> Vec<String> {
    let view = make_column_view(col);
    let mut size = 0i32;
    let st = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut offsets = vec![0i32; size as usize + 1];
    let mut total_chars = -1i32;
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut total_chars,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut chars = vec![0u8; total_chars.max(0) as usize];
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            chars.as_mut_ptr() as *mut c_char,
            chars.len(),
            &mut total_chars,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };

    offsets
        .windows(2)
        .map(|window| {
            let start = window[0] as usize;
            let end = window[1] as usize;
            String::from_utf8(chars[start..end].to_vec()).unwrap()
        })
        .collect()
}

fn make_table_view(tbl: *mut c_void) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_view(tbl, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_i32_scalar(value: i32) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            &value as *const i32 as *const c_void,
            1,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_i32_data_type() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_data_type_create(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_dictionary_i32_column(data: &[i32]) -> *mut c_void {
    let col = make_i32_column(data);
    let view = make_column_view(col);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_dictionary_encode(view, cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
    out
}

fn decode_dictionary_i32_column(dict_col: *mut c_void, expected_len: usize) -> Vec<i32> {
    let dict_view = make_column_view(dict_col);
    let mut decoded: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_dictionary_decode(dict_view, &mut decoded) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!decoded.is_null());
    unsafe { cudf_sys::cudf_column_view_destroy(dict_view) };

    let data = read_i32_column_data(decoded, expected_len);
    unsafe { cudf_sys::cudf_column_destroy(decoded) };
    data
}

fn make_sliced_column_view(col: *mut c_void, offset: usize, len: usize) -> *mut c_void {
    let base = make_column_view(col);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice(base, offset as i32, len as i32, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    unsafe { cudf_sys::cudf_column_view_destroy(base) };
    out
}

fn unique_missing_path(extension: &str) -> std::ffi::CString {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!(
        "cudf-sys-ffi-regressions-missing-{}-{}.{}",
        std::process::id(),
        nanos,
        extension
    ));
    let _ = std::fs::remove_file(&path);
    std::ffi::CString::new(path.to_string_lossy().into_owned()).unwrap()
}

#[test]
fn test_scalar_string_value_round_trips_empty_and_non_empty_values() {
    for expected in ["hello", ""] {
        let s = std::ffi::CString::new(expected).unwrap();
        let mut scalar: *mut c_void = ptr::null_mut();
        let st = unsafe { cudf_sys::cudf_scalar_create_string(s.as_ptr(), 1, &mut scalar) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);

        let mut out: *mut c_char = ptr::null_mut();
        let mut len: i32 = -1;
        let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut out, &mut len) };
        assert_eq!(st, cudf_sys::CUDF_SUCCESS);
        assert_eq!(len as usize, expected.len());
        assert_eq!(owned_string_value(out, len), expected);

        unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
    }
}

#[test]
fn test_column_view_null_mask_to_host_ex_supports_sliced_nullable_views() {
    let col = make_i32_column_nullable(
        &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        &[true, false, true, true, false, true, false, true, true, false],
    );
    let sliced = make_sliced_column_view(col, 3, 6);

    let mut mask = vec![0xAAu8; 64];
    let st =
        unsafe { cudf_sys::cudf_column_view_null_mask_to_host_ex(sliced, mask.as_mut_ptr(), mask.len(), ptr::null()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS, "sliced nullable null-mask export should succeed");

    assert_eq!(mask[0], 0b0011_0101, "mask bits should be normalized to the logical slice starting at bit 0");
    assert!(mask[1..].iter().all(|&byte| byte == 0), "no extra bits outside the logical slice should be set");

    unsafe {
        cudf_sys::cudf_column_view_destroy(sliced);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scalar_string_value_owned_output_survives_subsequent_shim_calls() {
    let s = std::ffi::CString::new("hello").unwrap();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(s.as_ptr(), 1, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut ptr1: *mut c_char = ptr::null_mut();
    let mut len1: i32 = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut ptr1, &mut len1) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut type_id = cudf_sys::cudf_type_id_t::CUDF_TYPE_EMPTY;
    let st = unsafe { cudf_sys::cudf_scalar_type(scalar, &mut type_id) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(type_id, cudf_sys::cudf_type_id_t::CUDF_TYPE_STRING);

    let mut ptr2: *mut c_char = ptr::null_mut();
    let mut len2: i32 = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut ptr2, &mut len2) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    assert_eq!(owned_string_value(ptr1, len1), "hello");
    assert_eq!(owned_string_value(ptr2, len2), "hello");

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_string_value_null_handle_initializes_outputs_before_failure() {
    let mut out = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut len: i32 = 123;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(ptr::null_mut(), &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
    assert_eq!(len, 0);
}

#[test]
fn test_scalar_string_value_wrong_type_initializes_outputs_before_failure() {
    let scalar = make_i32_scalar(42);

    let mut out = std::ptr::NonNull::<c_char>::dangling().as_ptr();
    let mut len: i32 = 123;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_TYPE);
    assert!(out.is_null());
    assert_eq!(len, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_numeric_invalid_allows_null_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, ptr::null(), 0, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());

    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid(scalar, &mut is_valid) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_numeric_valid_requires_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, ptr::null(), 1, &mut scalar)
    };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

#[test]
fn test_scalar_create_string_valid_requires_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(ptr::null(), 1, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);
    assert!(scalar.is_null());
}

#[test]
fn test_scalar_create_string_invalid_allows_null_value() {
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(ptr::null(), 0, &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());

    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid(scalar, &mut is_valid) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 0);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_stream_is_ready_null_out_rejected() {
    let mut stream: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_stream_create(&mut stream) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let st = unsafe { cudf_sys::cudf_stream_is_ready(stream, ptr::null_mut()) };
    assert_eq!(st, cudf_sys::CUDF_ERROR_NULL);

    unsafe { cudf_sys::cudf_stream_destroy(stream) };
}

#[test]
fn test_stream_create_is_ready_and_synchronize_smoke() {
    let mut stream: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_stream_create(&mut stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let mut ready = 0;
    let status = unsafe { cudf_sys::cudf_stream_is_ready(stream, &mut ready) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(ready, 1);

    let status = unsafe { cudf_sys::cudf_stream_synchronize(stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    unsafe { cudf_sys::cudf_stream_destroy(stream) };
}

#[test]
fn test_event_create_null_out_rejected() {
    let status = unsafe { cudf_sys::cudf_event_create(ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
}

#[test]
fn test_event_create_record_wait_and_synchronize_smoke() {
    let _shim_state = lock_shim_state();
    unsafe { cudf_sys::cudf_test_event_reset_observation() };

    let mut stream: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_stream_create(&mut stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!stream.is_null());

    let mut event: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_event_create(&mut event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!event.is_null());

    let status = unsafe { cudf_sys::cudf_event_record(event, stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let mut ready = 0;
    let status = unsafe { cudf_sys::cudf_event_is_ready(event, &mut ready) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(ready == 0 || ready == 1);

    let status = unsafe { cudf_sys::cudf_stream_wait_event(stream, event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let status = unsafe { cudf_sys::cudf_event_synchronize(event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let mut synchronize_calls = -1;
    let status = unsafe { cudf_sys::cudf_test_event_synchronize_call_count(&mut synchronize_calls) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(synchronize_calls, 1);

    let status = unsafe { cudf_sys::cudf_event_is_ready(event, &mut ready) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(ready, 1);

    let mut raw: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_event_raw(event, &mut raw) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!raw.is_null());

    unsafe {
        cudf_sys::cudf_event_destroy(event);
        cudf_sys::cudf_stream_destroy(stream);
    }
}

#[test]
fn test_raw_cuda_event_record_and_wait_smoke() {
    let _shim_state = lock_shim_state();
    let stream = OwnedStream::new();
    let event = OwnedEvent::new();

    let raw_stream = stream.raw_cuda_stream();
    let raw_event = event.raw_cuda_event();

    let status = unsafe { cudf_sys::cudf_raw_cuda_event_record(raw_event, raw_stream) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let status = unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(raw_stream, raw_event) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    let status = unsafe { cudf_sys::cudf_event_synchronize(event.as_raw()) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}

#[test]
fn test_execution_context_defaults_match_legacy_column_copy() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3, 4]);
    let default_stream = default_stream_value();
    let default_mr = current_device_resource_raw();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy(col, &mut legacy_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i32_column_data(legacy_out, 4), vec![1, 2, 3, 4]);

    reset_exec_observation();
    let mut null_opts_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, ptr::null(), &mut null_opts_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i32_column_data(null_opts_out, 4), vec![1, 2, 3, 4]);

    let exec = cudf_sys::ExecutionContext::new();
    reset_exec_observation();
    let mut explicit_defaults_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, exec.as_raw(), &mut explicit_defaults_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    assert_eq!(read_i32_column_data(explicit_defaults_out, 4), vec![1, 2, 3, 4]);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_defaults_out);
        cudf_sys::cudf_column_destroy(null_opts_out);
        cudf_sys::cudf_column_destroy(legacy_out);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scalar_create_numeric_remains_unsynchronized_before_return() {
    let _shim_state = lock_shim_state();
    reset_exec_observation();
    let scalar = make_i32_scalar(42);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut value = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value(scalar, &mut value as *mut i32 as *mut c_void) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(value, 42);
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_string_synchronizes_default_stream_before_return() {
    let _shim_state = lock_shim_state();
    let default_stream = default_stream_value();
    let hello = std::ffi::CString::new("hello").unwrap();
    reset_exec_observation();
    let mut string_scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string(hello.as_ptr(), 1, &mut string_scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!string_scalar.is_null());
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(string_scalar, &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    unsafe { cudf_sys::cudf_scalar_destroy(string_scalar) };
}

#[test]
fn test_scalar_create_numeric_ex_uses_explicit_stream_and_mr_without_syncing_before_return() {
    let _shim_state = lock_shim_state();
    let value = 42i32;
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    reset_exec_observation();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            &value as *const i32 as *const c_void,
            1,
            exec.as_raw(),
            &mut scalar,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_ne!(last_exec_mr_raw(), ptr::null_mut());
    assert_ne!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let mut out = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value(scalar, &mut out as *mut i32 as *mut c_void) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, value);
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_create_string_ex_uses_explicit_stream_and_mr_and_synchronizes_before_return() {
    let _shim_state = lock_shim_state();
    let hello = std::ffi::CString::new("hello").unwrap();
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream).with_memory_resource(&borrowed_mr);

    reset_exec_observation();
    let mut scalar: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string_ex(hello.as_ptr(), 1, exec.as_raw(), &mut scalar) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!scalar.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_ne!(last_exec_mr_raw(), ptr::null_mut());
    assert_ne!(last_exec_mr_raw(), default_mr);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value(scalar, &mut out, &mut len) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_create_numeric_host_source_sync_contract() {
    let _shim_state = lock_shim_state();
    let data = [31i32, 32, 33];
    let validity = [true, false, true];
    let null_mask = validity_to_bitmask(&validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;
    let default_stream = default_stream_value();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            &mut legacy_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!legacy_out.is_null());
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);
    assert_eq!(read_i32_column_data(legacy_out, data.len()), data);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            exec.as_raw(),
            &mut explicit_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!explicit_out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
    assert_eq!(read_i32_column_data(explicit_out, data.len()), data);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
    }
}

#[test]
fn test_column_create_strings_host_source_sync_contract() {
    let _shim_state = lock_shim_state();
    let values = ["ab", "", "cde"];
    let c_strings: Vec<_> = values.iter().map(|value| std::ffi::CString::new(*value).unwrap()).collect();
    let string_ptrs: Vec<_> = c_strings.iter().map(|value| value.as_ptr()).collect();
    let lengths: Vec<_> = values.iter().map(|value| value.len() as i32).collect();
    let default_stream = default_stream_value();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            values.len() as i32,
            ptr::null(),
            0,
            &mut legacy_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!legacy_out.is_null());
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);
    assert_eq!(read_string_column_data(legacy_out), values);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            values.len() as i32,
            ptr::null(),
            0,
            exec.as_raw(),
            &mut explicit_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!explicit_out.is_null());
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);
    assert_eq!(read_string_column_data(explicit_out), values);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
    }
}

#[test]
fn test_stream_handle_has_no_effect_until_attached_to_execution_context() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[7, 8, 9]);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy(col, &mut legacy_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_ne!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, exec.as_raw(), &mut explicit_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_to_host_execution_context_defaults_and_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[11, 12, 13]);
    let view = make_column_view(col);
    let default_stream = default_stream_value();

    let mut legacy_buf = vec![0i32; 3];
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_to_host(
            view,
            legacy_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(legacy_buf.as_slice()),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(legacy_buf, vec![11, 12, 13]);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let mut explicit_buf = vec![0i32; 3];
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_to_host_ex(
            view,
            explicit_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(explicit_buf.as_slice()),
            exec.as_raw(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(explicit_buf, vec![11, 12, 13]);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    let explicit_defaults = cudf_sys::ExecutionContext::new();
    let mut default_opts_buf = vec![0i32; 3];
    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_to_host_ex(
            view,
            default_opts_buf.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(default_opts_buf.as_slice()),
            explicit_defaults.as_raw(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(default_opts_buf, vec![11, 12, 13]);
    assert_eq!(last_exec_stream_value(), default_stream);
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_null_mask_to_host_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column_nullable(&[1, 2, 3, 4, 5], &[true, false, true, false, true]);
    let view = make_column_view(col);
    let default_stream = default_stream_value();
    let mut host_mask = [0u8; 64];

    reset_exec_observation();
    let st = unsafe { cudf_sys::cudf_column_view_null_mask_to_host(view, host_mask.as_mut_ptr(), host_mask.len()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(host_mask[0] & 0b0001_1111, 0b0001_0101);
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_slice_nested_nullable_preserves_logical_offset_in_null_count() {
    let col = make_i32_column_nullable(&[10, 20, 30, 40, 50], &[true, false, true, false, true]);
    let base_view = make_column_view(col);

    let mut first_slice: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice(base_view, 1, 3, &mut first_slice) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!first_slice.is_null());

    let mut first_offset = -1i32;
    let mut first_null_count = -1i32;
    let st = unsafe { cudf_sys::cudf_column_view_offset(first_slice, &mut first_offset) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let st = unsafe { cudf_sys::cudf_column_view_null_count(first_slice, &mut first_null_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(first_offset, 1);
    assert_eq!(first_null_count, 2);

    let mut second_slice: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice(first_slice, 1, 1, &mut second_slice) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!second_slice.is_null());

    let mut second_offset = -1i32;
    let mut second_null_count = -1i32;
    let st = unsafe { cudf_sys::cudf_column_view_offset(second_slice, &mut second_offset) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    let st = unsafe { cudf_sys::cudf_column_view_null_count(second_slice, &mut second_null_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(second_offset, 2);
    assert_eq!(second_null_count, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(second_slice);
        cudf_sys::cudf_column_view_destroy(first_slice);
        cudf_sys::cudf_column_view_destroy(base_view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["ab", "", "cde"]);
    let view = make_column_view(col);
    let default_stream = default_stream_value();
    let mut offsets = vec![-1i32; 4];
    let mut chars = vec![0u8; 5];
    let mut total_chars = -1;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            chars.as_mut_ptr() as *mut c_char,
            chars.len(),
            &mut total_chars,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(offsets, vec![0, 2, 2, 5]);
    assert_eq!(total_chars, 5);
    assert_eq!(chars.as_slice(), b"abcde");
    assert_eq!(last_exec_sync_stream_value(), default_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_empty_writes_single_zero_offset() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let mut offsets = vec![-1i32; 1];
    let mut total_chars = -1;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut total_chars,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(offsets, vec![0]);
    assert_eq!(total_chars, 0);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

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
fn test_column_view_string_chars_ptr_and_size_ex_reports_active_sliced_chars_span() {
    let col = make_string_column(&["zero", "one", "two", "three"]);
    let full_view = make_column_view(col);
    let sliced_view = make_sliced_column_view(col, 1, 2);

    let mut full_ptr: *const c_void = ptr::null();
    let mut full_size: usize = 0;
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(full_view, ptr::null(), &mut full_ptr, &mut full_size)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!full_ptr.is_null());
    assert_eq!(full_size, "zeroonetwothree".len());

    let mut sliced_ptr: *const c_void = ptr::null();
    let mut sliced_size: usize = 0;
    let st = unsafe {
        cudf_sys::cudf_column_view_string_chars_ptr_and_size_ex(
            sliced_view,
            ptr::null(),
            &mut sliced_ptr,
            &mut sliced_size,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!sliced_ptr.is_null());
    assert_eq!(sliced_ptr, full_ptr);
    assert_eq!(sliced_size, "zeroonetwo".len());

    unsafe {
        cudf_sys::cudf_column_view_destroy(sliced_view);
        cudf_sys::cudf_column_view_destroy(full_view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_empty_ex_matches_legacy_contract() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&[]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let mut legacy_offsets = vec![-1i32; 1];
    let mut legacy_total_chars = -1;
    let mut ex_offsets = vec![-1i32; 1];
    let mut ex_total_chars = -1;

    let legacy_st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host(
            view,
            legacy_offsets.as_mut_ptr(),
            std::mem::size_of_val(legacy_offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut legacy_total_chars,
        )
    };
    assert_eq!(legacy_st, cudf_sys::CUDF_SUCCESS);

    reset_exec_observation();
    let ex_st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            ex_offsets.as_mut_ptr(),
            std::mem::size_of_val(ex_offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut ex_total_chars,
            exec.as_raw(),
        )
    };
    assert_eq!(ex_st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(ex_offsets, legacy_offsets);
    assert_eq!(ex_total_chars, legacy_total_chars);
    assert_eq!(ex_offsets, vec![0]);
    assert_eq!(ex_total_chars, 0);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_column_view_strings_to_host_ex_sync_contract() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["ab", "", "cde"]);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let mut offsets = vec![-1i32; 4];
    let mut chars = vec![0u8; 5];
    let mut total_chars = -1;

    reset_exec_observation();
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            chars.as_mut_ptr() as *mut c_char,
            chars.len(),
            &mut total_chars,
            exec.as_raw(),
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(offsets, vec![0, 2, 2, 5]);
    assert_eq!(total_chars, 5);
    assert_eq!(chars.as_slice(), b"abcde");
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), explicit_stream);

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_scalar_is_valid_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_i32_scalar(42);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut is_valid = -1;
    let st = unsafe { cudf_sys::cudf_scalar_is_valid_ex(scalar, &mut is_valid, exec.as_raw()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(is_valid, 1);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_value_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_i32_scalar(42);
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut out = 0i32;
    let st = unsafe { cudf_sys::cudf_scalar_value_ex(scalar, &mut out as *mut i32 as *mut c_void, exec.as_raw()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 42);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_scalar_string_value_ex_uses_explicit_stream_without_extra_shim_sync() {
    let _shim_state = lock_shim_state();
    let scalar = make_string_scalar("hello");
    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut out: *mut c_char = ptr::null_mut();
    let mut len = -1;
    let st = unsafe { cudf_sys::cudf_scalar_string_value_ex(scalar, &mut out, &mut len, exec.as_raw()) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(owned_string_value(out, len), "hello");
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);

    unsafe { cudf_sys::cudf_scalar_destroy(scalar) };
}

#[test]
fn test_column_create_from_device_data_sync_and_async_contracts() {
    let _shim_state = lock_shim_state();
    let source = make_i32_column(&[21, 22, 23]);
    let source_view = make_column_view(source);
    let default_stream = default_stream_value();

    let mut device_ptr: *const c_void = ptr::null();
    let st = unsafe { cudf_sys::cudf_column_view_data_ptr(source_view, &mut device_ptr) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!device_ptr.is_null());

    reset_exec_observation();
    let mut sync_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_from_device_data(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            device_ptr,
            3,
            &mut sync_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_sync_stream_value(), default_stream);
    assert_eq!(read_i32_column_data(sync_out, 3), vec![21, 22, 23]);

    let stream = OwnedStream::new();
    let explicit_stream = stream_value(stream.as_raw());
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);

    reset_exec_observation();
    let mut async_out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_from_device_data_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            device_ptr,
            3,
            exec.as_raw(),
            &mut async_out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(last_exec_stream_value(), explicit_stream);
    assert_eq!(last_exec_sync_stream_value(), NO_SYNC_RECORDED);
    stream.synchronize();
    assert_eq!(read_i32_column_data(async_out, 3), vec![21, 22, 23]);

    unsafe {
        cudf_sys::cudf_column_destroy(async_out);
        cudf_sys::cudf_column_destroy(sync_out);
        cudf_sys::cudf_column_view_destroy(source_view);
        cudf_sys::cudf_column_destroy(source);
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

    // Invalid keep policy
    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 99, 0, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    // Invalid null equality policy
    let status = unsafe { cudf_distinct_raw(tv, keys.as_ptr(), 1, 0, 99, 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());

    // Invalid nan equality policy
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

#[test]
fn test_memory_resource_pool_create_destroy_smoke() {
    let mut mr: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_pool(1 << 20, 1 << 24, &mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());

    unsafe { cudf_sys::cudf_memory_resource_destroy(mr) };
}

#[test]
fn test_memory_resource_cuda_create_destroy_smoke() {
    let mut mr: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_memory_resource_create_cuda(&mut mr) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!mr.is_null());

    unsafe { cudf_sys::cudf_memory_resource_destroy(mr) };
}

#[test]
fn test_memory_resource_has_no_effect_until_attached_to_execution_context() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[21, 22, 23]);
    let stats_mr = make_stats_memory_resource();
    let default_mr = current_device_resource_raw();

    reset_exec_observation();
    let mut legacy_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy(col, &mut legacy_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(stats_mr.allocation_count(), 0);
    assert_eq!(last_exec_mr_raw(), default_mr);

    let borrowed_mr = stats_mr.borrowed();
    let exec_opts = cudf_sys::ExecutionContext::new().with_memory_resource(&borrowed_mr);
    reset_exec_observation();
    let mut explicit_out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_copy_ex(col, exec_opts.as_raw(), &mut explicit_out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(stats_mr.allocation_count() > 0);
    assert_ne!(last_exec_mr_raw(), default_mr);

    unsafe {
        cudf_sys::cudf_column_destroy(explicit_out);
        cudf_sys::cudf_column_destroy(legacy_out);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_table_view_null_handle_clears_out() {
    let mut out = 1usize as *mut c_void;
    let status = unsafe { cudf_sys::cudf_table_view(ptr::null_mut(), &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_io_read_csv_missing_file_clears_out() {
    let path = unique_missing_path("csv");
    let mut out = 1usize as *mut c_void;
    let status = unsafe { cudf_sys::cudf_io_read_csv(path.as_ptr(), b',' as c_char, 1, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_IO);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_missing_file_clears_out() {
    let path = unique_missing_path("parquet");
    let mut out = 1usize as *mut c_void;
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, 0, -1, ptr::null(), 0, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_IO);
    assert!(out.is_null());
}

#[test]
fn test_non_io_runtime_error_maps_to_unknown() {
    let _shim_state = lock_shim_state();
    let status = unsafe { cudf_sys::cudf_test_throw_runtime_error() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_UNKNOWN);
    assert!(last_error_message().contains("non-io runtime failure"));
}

#[test]
fn test_bad_alloc_maps_to_out_of_memory() {
    let _shim_state = lock_shim_state();
    let status = unsafe { cudf_sys::cudf_test_throw_bad_alloc() };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(!last_error_message().is_empty());
}

#[test]
fn test_io_read_parquet_with_metadata_invalid_args_clear_outputs() {
    let path = unique_missing_path("parquet");
    let mut out_table = 1usize as *mut c_void;
    let mut out_names = 1usize as *mut *mut c_char;
    let mut out_num_names = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            -1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
}

#[test]
fn test_io_write_csv_negative_num_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("csv");

    let out_status = unsafe { cudf_sys::cudf_io_write_csv(tv, path.as_ptr(), ptr::null(), -1, b',' as i8, 1) };
    assert_eq!(out_status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_write_csv_null_column_name_entry_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("csv");
    let names = [ptr::null()];

    let status = unsafe { cudf_sys::cudf_io_write_csv(tv, path.as_ptr(), names.as_ptr(), 1, b',' as i8, 1) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_read_parquet_negative_num_columns_rejected() {
    let path = unique_missing_path("parquet");
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), -1, 0, -1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_null_column_name_entry_rejected() {
    let path = unique_missing_path("parquet");
    let columns = [ptr::null()];
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), columns.as_ptr(), 1, 0, -1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_negative_skip_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, -1, -1, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_read_parquet_invalid_num_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out: *mut c_void = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_io_read_parquet(path.as_ptr(), ptr::null(), 0, 0, -2, ptr::null(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_io_write_parquet_negative_num_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");

    let status = unsafe { cudf_sys::cudf_io_write_parquet(tv, path.as_ptr(), ptr::null(), -1, 0) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_write_parquet_null_column_name_entry_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let path = unique_missing_path("parquet");
    let names = [ptr::null()];

    let status = unsafe { cudf_sys::cudf_io_write_parquet(tv, path.as_ptr(), names.as_ptr(), 1, 0) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_io_read_parquet_with_metadata_negative_num_columns_rejected() {
    let path = unique_missing_path("parquet");
    let mut out_table: *mut c_void = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            -1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_io_read_parquet_with_metadata_null_column_name_entry_rejected() {
    let path = unique_missing_path("parquet");
    let columns = [ptr::null()];
    let mut out_table: *mut c_void = 1usize as *mut c_void;
    let mut out_names: *mut *mut i8 = 1usize as *mut *mut i8;
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            columns.as_ptr(),
            1,
            0,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
    assert_eq!(out_num_names, 0);
}

#[test]
fn test_io_read_parquet_with_metadata_negative_skip_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out_table: *mut c_void = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            0,
            -1,
            -1,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_io_read_parquet_with_metadata_invalid_num_rows_rejected() {
    let path = unique_missing_path("parquet");
    let mut out_table: *mut c_void = ptr::null_mut();
    let mut out_names: *mut *mut i8 = ptr::null_mut();
    let mut out_num_names: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_io_read_parquet_with_metadata(
            path.as_ptr(),
            ptr::null(),
            0,
            0,
            -2,
            ptr::null(),
            0,
            &mut out_table,
            &mut out_names,
            &mut out_num_names,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_table.is_null());
    assert!(out_names.is_null());
}

#[test]
fn test_dictionary_match_dictionaries_invalid_num_columns_clears_outputs() {
    let dict_columns = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_columns = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_count: i32 = -1;

    let status =
        unsafe { cudf_sys::cudf_dictionary_match_dictionaries(dict_columns, 0, &mut out_columns, &mut out_count) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);
}

#[test]
fn test_replace_nulls_policy_rejects_invalid_policy_value() {
    let col = make_i32_column_nullable(&[1, 0, 3], &[true, false, true]);
    let view = make_column_view(col);

    let mut out = std::ptr::NonNull::<c_void>::dangling().as_ptr();
    let status = unsafe { cudf_sys::cudf_replace_nulls_policy(view, 42, &mut out) };

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
    let dict = make_dictionary_i32_column(&[1, 2, 1]);
    let dict_view = make_column_view(dict);
    let dict_columns = [dict_view, ptr::null_mut()];

    let mut out_columns = std::ptr::NonNull::<*mut c_void>::dangling().as_ptr();
    let mut out_count: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
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
    let left = make_dictionary_i32_column(&[10, 20, 10, 30]);
    let right = make_dictionary_i32_column(&[20, 30, 40, 20]);
    let left_view = make_column_view(left);
    let right_view = make_column_view(right);
    let dict_columns = [left_view, right_view];

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe {
        cudf_sys::cudf_dictionary_match_dictionaries(
            dict_columns.as_ptr(),
            dict_columns.len() as i32,
            &mut out_columns,
            &mut out_count,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, dict_columns.len() as i32);
    assert!(!out_columns.is_null());

    let matched = unsafe { std::slice::from_raw_parts(out_columns, out_count as usize) };
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

// ===========================================================================
// cudf_table_create tests
// ===========================================================================

#[test]
fn test_table_create_failure_does_not_partially_consume_columns() {
    let _shim_state = lock_shim_state();
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let mut columns = [c1, ptr::null_mut(), c2];

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), columns.len() as i32, &mut out) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
    assert_eq!(columns[0], c1);
    assert!(columns[1].is_null());
    assert_eq!(columns[2], c2);
    assert!(last_error_message().contains("column handle is null"));

    let data1 = read_i32_column_data(columns[0], 3);
    assert_eq!(data1, vec![1, 2, 3]);

    let data2 = read_i32_column_data(columns[2], 3);
    assert_eq!(data2, vec![4, 5, 6]);

    let mut valid_columns = [columns[0], columns[2]];
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(valid_columns.as_mut_ptr(), valid_columns.len() as i32, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());
    assert!(valid_columns.iter().all(|col| col.is_null()));

    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}

// ===========================================================================
// cudf_table_release_columns tests
// ===========================================================================

#[test]
fn test_table_release_columns_single_column() {
    let col = make_i32_column(&[10, 20, 30]);
    let tbl = make_table_from_one_column(col);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 1);
    assert!(!out_columns.is_null());

    // Read the released column and verify data.
    let released_col = unsafe { *out_columns };
    assert!(!released_col.is_null());

    let data = read_i32_column_data(released_col, 3);
    assert_eq!(data, vec![10, 20, 30]);

    // Cleanup: destroy the released column and the array.
    unsafe {
        cudf_sys::cudf_column_destroy(released_col);
        cudf_sys::cudf_free_column_array(out_columns);
    }
    // Note: tbl is consumed by release_columns, no need to destroy it.
}

#[test]
fn test_table_release_columns_multiple_columns() {
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let mut col_arr = [c1, c2];
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(col_arr.as_mut_ptr(), 2, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 2);
    assert!(!out_columns.is_null());

    let col_ptrs = unsafe { std::slice::from_raw_parts(out_columns, 2) };

    let data1 = read_i32_column_data(col_ptrs[0], 3);
    assert_eq!(data1, vec![1, 2, 3]);

    let data2 = read_i32_column_data(col_ptrs[1], 3);
    assert_eq!(data2, vec![4, 5, 6]);

    unsafe {
        cudf_sys::cudf_column_destroy(col_ptrs[0]);
        cudf_sys::cudf_column_destroy(col_ptrs[1]);
        cudf_sys::cudf_free_column_array(out_columns);
    }
}

#[test]
fn test_table_release_columns_empty_table() {
    // Create an empty table (0 columns).
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(ptr::null_mut(), 0, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out_count, 0);
    assert!(out_columns.is_null());
}

#[test]
fn test_table_release_columns_null_table_rejected() {
    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(ptr::null_mut(), &mut out_columns, &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
}

#[test]
fn test_table_release_columns_null_out_columns_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);

    let mut out_count: i32 = 0;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, ptr::null_mut(), &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    // Table was not consumed on error — must still be destroyed.
    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}

#[test]
fn test_table_release_columns_null_out_count_rejected() {
    let col = make_i32_column(&[1, 2]);
    let tbl = make_table_from_one_column(col);

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    // Table was not consumed on error — must still be destroyed.
    unsafe { cudf_sys::cudf_table_destroy(tbl) };
}

#[test]
fn test_table_release_columns_failure_does_not_consume_table() {
    let col = make_i32_column(&[7, 8, 9]);
    let tbl = make_table_from_one_column(col);

    let mut out_count: i32 = -1;
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, ptr::null_mut(), &mut out_count) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert_eq!(out_count, -1);

    // Regression check: on failure, the table must remain valid and usable.
    let tv = make_table_view(tbl);
    let mut num_cols: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(tv, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 1);

    let mut released_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut released_count: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut released_columns, &mut released_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(released_count, 1);
    assert!(!released_columns.is_null());

    let released_col = unsafe { *released_columns };
    let data = read_i32_column_data(released_col, 3);
    assert_eq!(data, vec![7, 8, 9]);

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_column_destroy(released_col);
        cudf_sys::cudf_free_column_array(released_columns);
    }
}

#[test]
fn test_table_release_columns_allocation_failure_does_not_consume_table() {
    let _shim_state = lock_shim_state();
    const NUM_COLUMNS: usize = 2;

    let mut columns: Vec<*mut c_void> = (0..NUM_COLUMNS).map(|_| make_empty_i32_column()).collect();
    let mut tbl: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), NUM_COLUMNS as i32, &mut tbl) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!tbl.is_null());
    assert!(columns.iter().all(|col| col.is_null()));

    let mut out_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_count: i32 = -1;
    unsafe { cudf_sys::cudf_test_fail_next_table_release_columns_handle_array_alloc() };
    let status = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut out_columns, &mut out_count) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(out_columns.is_null());
    assert_eq!(out_count, 0);
    assert!(last_error_message().contains("failed to allocate column handle array"));

    // Regression check: a failure after entering the function body must not
    // consume the table; the caller should still own and be able to use it.
    let tv = make_table_view(tbl);
    let mut num_cols: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(tv, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, NUM_COLUMNS as i32);

    let mut released_columns: *mut cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut released_count: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_release_columns(tbl, &mut released_columns, &mut released_count) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(released_count, NUM_COLUMNS as i32);
    assert!(!released_columns.is_null());

    let released = unsafe { std::slice::from_raw_parts(released_columns, released_count as usize) };
    for &col in released {
        unsafe { cudf_sys::cudf_column_destroy(col) };
    }

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_free_column_array(released_columns);
    }
}

// ===========================================================================
// cudf_table_view_from_column_views tests
// ===========================================================================

#[test]
fn test_table_view_from_column_views_two_columns() {
    let c1 = make_i32_column(&[1, 2, 3]);
    let c2 = make_i32_column(&[4, 5, 6]);
    let cv1 = make_column_view(c1);
    let cv2 = make_column_view(c2);

    let mut views = [cv1, cv2];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    // Verify num_columns == 2.
    let mut num_cols: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 2);

    // Verify num_rows == 3.
    let mut num_rows: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_rows(out, &mut num_rows) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_rows, 3);

    unsafe {
        cudf_sys::cudf_table_view_destroy(out);
        cudf_sys::cudf_column_view_destroy(cv1);
        cudf_sys::cudf_column_view_destroy(cv2);
        cudf_sys::cudf_column_destroy(c1);
        cudf_sys::cudf_column_destroy(c2);
    }
}

#[test]
fn test_table_view_from_column_views_single_column() {
    let col = make_i32_column(&[10, 20]);
    let cv = make_column_view(col);

    let mut views = [cv];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let mut num_cols: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 1);

    let mut num_rows: i32 = 0;
    let st = unsafe { cudf_sys::cudf_table_view_num_rows(out, &mut num_rows) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_rows, 2);

    unsafe {
        cudf_sys::cudf_table_view_destroy(out);
        cudf_sys::cudf_column_view_destroy(cv);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_table_view_from_column_views_null_out_rejected() {
    let col = make_i32_column(&[1]);
    let cv = make_column_view(col);

    let mut views = [cv];
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 1, ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    unsafe {
        cudf_sys::cudf_column_view_destroy(cv);
        cudf_sys::cudf_column_destroy(col);
    }
}

#[test]
fn test_table_view_from_column_views_negative_count_rejected() {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), -1, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
}

#[test]
fn test_table_view_from_column_views_null_array_with_nonzero_count_rejected() {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());
}

#[test]
fn test_table_view_from_column_views_null_element_rejected() {
    let c1 = make_i32_column(&[1, 2, 3]);
    let cv1 = make_column_view(c1);

    let mut views = [cv1, ptr::null_mut()];
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(views.as_mut_ptr(), 2, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(cv1);
        cudf_sys::cudf_column_destroy(c1);
    }
}

#[test]
fn test_table_view_from_column_views_zero_columns_succeeds() {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_view_from_column_views(ptr::null_mut(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    let mut num_cols: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_columns(out, &mut num_cols) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_cols, 0);

    let mut num_rows: i32 = -1;
    let st = unsafe { cudf_sys::cudf_table_view_num_rows(out, &mut num_rows) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert_eq!(num_rows, 0);

    unsafe {
        cudf_sys::cudf_table_view_destroy(out);
    }
}
