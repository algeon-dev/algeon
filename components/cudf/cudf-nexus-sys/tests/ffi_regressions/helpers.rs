use std::{
    os::raw::{c_char, c_void},
    ptr,
};

use crate::support::{OwnedStream, make_stats_memory_resource, required_exec_options};

// Re-declare selected FFI functions with `i32` parameters so that tests can
// pass deliberately invalid enum discriminants to the C shim without
// constructing invalid Rust enum values (which is instant UB).
#[allow(dead_code)]
unsafe extern "C" {
    #[link_name = "cudf_ast_column_reference_create"]
    pub(crate) fn cudf_ast_column_reference_create_raw(
        column_index: i32,
        table_source: i32,
        out: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_ast_operation_unary_create"]
    pub(crate) fn cudf_ast_operation_unary_create_raw(op: i32, input: *mut c_void, out: *mut *mut c_void) -> i32;
    #[link_name = "cudf_ast_operation_binary_create"]
    pub(crate) fn cudf_ast_operation_binary_create_raw(
        op: i32,
        left: *mut c_void,
        right: *mut c_void,
        out: *mut *mut c_void,
    ) -> i32;
    #[link_name = "cudf_sort_ex"]
    pub(crate) fn cudf_sort_raw(
        input: *mut c_void,
        column_order: *const i32,
        null_precedence: *const i32,
        num_key_columns: i32,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_distinct_ex"]
    pub(crate) fn cudf_distinct_raw(
        input: *mut c_void,
        keys: *const i32,
        num_keys: i32,
        keep: i32,
        null_equality: i32,
        nan_equality: i32,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_gather_ex"]
    pub(crate) fn cudf_gather_raw(
        input: *mut c_void,
        gather_map: *mut c_void,
        bounds_policy: i32,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_scatter_ex"]
    pub(crate) fn cudf_scatter_raw(
        source: *mut c_void,
        scatter_map: *mut c_void,
        target: *mut c_void,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_aggregation_create_count"]
    pub(crate) fn cudf_aggregation_create_count_raw(null_policy: i32, out: *mut *mut c_void) -> i32;
    #[link_name = "cudf_groupby_aggregate_ex"]
    pub(crate) fn cudf_groupby_aggregate_raw(
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
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_inner_join_ex"]
    pub(crate) fn cudf_inner_join_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        left_payload: *mut c_void,
        right_payload: *mut c_void,
        null_equality: i32,
        out_left: *mut *mut c_void,
        out_right: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_left_join_ex"]
    pub(crate) fn cudf_left_join_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        left_payload: *mut c_void,
        right_payload: *mut c_void,
        null_equality: i32,
        out_left: *mut *mut c_void,
        out_right: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_full_join_ex"]
    pub(crate) fn cudf_full_join_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        left_payload: *mut c_void,
        right_payload: *mut c_void,
        null_equality: i32,
        out_left: *mut *mut c_void,
        out_right: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_inner_join_indices_ex"]
    pub(crate) fn cudf_inner_join_indices_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        null_equality: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_left_join_indices_ex"]
    pub(crate) fn cudf_left_join_indices_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        null_equality: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_full_join_indices_ex"]
    pub(crate) fn cudf_full_join_indices_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        null_equality: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_left_semi_join_indices_ex"]
    pub(crate) fn cudf_left_semi_join_indices_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        null_equality: i32,
        out_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_left_anti_join_indices_ex"]
    pub(crate) fn cudf_left_anti_join_indices_raw(
        left_keys: *mut c_void,
        right_keys: *mut c_void,
        null_equality: i32,
        out_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_distinct_hash_join_create_ex"]
    pub(crate) fn cudf_distinct_hash_join_create_raw(
        build: *mut c_void,
        null_equality: i32,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_distinct_hash_join_left_join_ex"]
    pub(crate) fn cudf_distinct_hash_join_left_join_raw(
        join: *mut c_void,
        probe: *mut c_void,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_hash_join_create_ex"]
    pub(crate) fn cudf_hash_join_create_raw(
        build: *mut c_void,
        nullable_join: i32,
        null_equality: i32,
        load_factor: f64,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
}

pub(crate) type ConditionalPairJoinRaw = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
    *const cudf_sys::cudf_exec_options_t,
) -> i32;
pub(crate) type ConditionalSingleJoinRaw = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut *mut c_void,
    *const cudf_sys::cudf_exec_options_t,
) -> i32;
pub(crate) type FilterJoinIndicesRaw = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    *mut *mut c_void,
    *mut *mut c_void,
    *const cudf_sys::cudf_exec_options_t,
) -> i32;
pub(crate) type FilterJoinIndicesExRaw = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    *mut *mut c_void,
    *mut *mut c_void,
    *const cudf_sys::cudf_exec_options_t,
) -> i32;

#[allow(dead_code)]
unsafe extern "C" {
    #[link_name = "cudf_conditional_inner_join_indices_ex"]
    pub(crate) fn cudf_conditional_inner_join_indices_raw(
        left: *mut c_void,
        right: *mut c_void,
        predicate: *mut c_void,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_conditional_left_join_indices_ex"]
    pub(crate) fn cudf_conditional_left_join_indices_raw(
        left: *mut c_void,
        right: *mut c_void,
        predicate: *mut c_void,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_conditional_full_join_indices_ex"]
    pub(crate) fn cudf_conditional_full_join_indices_raw(
        left: *mut c_void,
        right: *mut c_void,
        predicate: *mut c_void,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_conditional_left_semi_join_indices_ex"]
    pub(crate) fn cudf_conditional_left_semi_join_indices_raw(
        left: *mut c_void,
        right: *mut c_void,
        predicate: *mut c_void,
        out_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_conditional_left_anti_join_indices_ex"]
    pub(crate) fn cudf_conditional_left_anti_join_indices_raw(
        left: *mut c_void,
        right: *mut c_void,
        predicate: *mut c_void,
        out_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_filter_join_indices_ex"]
    pub(crate) fn cudf_filter_join_indices_raw(
        left: *mut c_void,
        right: *mut c_void,
        left_indices: *mut c_void,
        right_indices: *mut c_void,
        predicate: *mut c_void,
        join_kind: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_filter_join_indices_ex"]
    pub(crate) fn cudf_filter_join_indices_ex_raw(
        left: *mut c_void,
        right: *mut c_void,
        left_indices: *mut c_void,
        right_indices: *mut c_void,
        predicate: *mut c_void,
        join_kind: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
}

pub(crate) type MixedPairJoinRaw = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    *mut *mut c_void,
    *mut *mut c_void,
    *const cudf_sys::cudf_exec_options_t,
) -> i32;
pub(crate) type MixedSingleJoinRaw = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    *mut *mut c_void,
    *const cudf_sys::cudf_exec_options_t,
) -> i32;

#[allow(dead_code)]
unsafe extern "C" {
    #[link_name = "cudf_mixed_inner_join_indices_ex"]
    pub(crate) fn cudf_mixed_inner_join_indices_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_mixed_left_join_indices_ex"]
    pub(crate) fn cudf_mixed_left_join_indices_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_mixed_full_join_indices_ex"]
    pub(crate) fn cudf_mixed_full_join_indices_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: i32,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_mixed_left_semi_join_indices_ex"]
    pub(crate) fn cudf_mixed_left_semi_join_indices_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: i32,
        out_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_mixed_left_anti_join_indices_ex"]
    pub(crate) fn cudf_mixed_left_anti_join_indices_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: i32,
        out_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
}

#[allow(dead_code)]
unsafe extern "C" {
    #[link_name = "cudf_scan_aggregation_create"]
    pub(crate) fn cudf_scan_aggregation_create_raw(kind: i32, out: *mut *mut c_void) -> i32;
    #[link_name = "cudf_scan_ex"]
    pub(crate) fn cudf_scan_raw(
        input: *mut c_void,
        agg: *mut c_void,
        scan_type: i32,
        null_policy: i32,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_unary_operation_ex"]
    pub(crate) fn cudf_unary_operation_raw(
        input: *mut c_void,
        op: i32,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_binary_operation_vv_ex"]
    pub(crate) fn cudf_binary_operation_vv_raw(
        lhs: *mut c_void,
        rhs: *mut c_void,
        op: i32,
        output_type: *mut c_void,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_binary_operation_vs_ex"]
    pub(crate) fn cudf_binary_operation_vs_raw(
        lhs: *mut c_void,
        rhs: *mut c_void,
        op: i32,
        output_type: *mut c_void,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
    #[link_name = "cudf_binary_operation_sv_ex"]
    pub(crate) fn cudf_binary_operation_sv_raw(
        lhs: *mut c_void,
        rhs: *mut c_void,
        op: i32,
        output_type: *mut c_void,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> i32;
}

pub(crate) fn make_i32_column(data: &[i32]) -> *mut c_void {
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
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

pub(crate) fn validity_to_bitmask(validity: &[bool]) -> Vec<u8> {
    let mut mask = vec![0u8; validity.len().div_ceil(8).div_ceil(64) * 64];
    for (index, is_valid) in validity.iter().copied().enumerate() {
        if is_valid {
            mask[index / 8] |= 1 << (index % 8);
        }
    }
    mask
}

pub(crate) fn make_i32_column_nullable(data: &[i32], validity: &[bool]) -> *mut c_void {
    assert_eq!(data.len(), validity.len());
    let null_mask = validity_to_bitmask(validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;

    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            null_mask.as_ptr(),
            null_count,
            &exec,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

pub(crate) fn make_string_column(data: &[&str]) -> *mut c_void {
    let c_strings: Vec<_> = data.iter().map(|value| std::ffi::CString::new(*value).unwrap()).collect();
    let string_ptrs: Vec<_> = c_strings.iter().map(|value| value.as_ptr()).collect();
    let lengths: Vec<_> = data.iter().map(|value| value.len() as i32).collect();

    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
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

pub(crate) fn make_string_scalar(value: &str) -> *mut c_void {
    let c_value = std::ffi::CString::new(value).unwrap();
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_scalar_create_string_ex(c_value.as_ptr(), 1, &exec, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

pub(crate) fn make_empty_i32_column() -> *mut c_void {
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st =
        unsafe { cudf_sys::cudf_column_create_empty_ex(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &exec, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

pub(crate) fn make_empty_table() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(ptr::null_mut(), 0, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

pub(crate) fn make_column_view(col: *mut c_void) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view(col, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

pub(crate) fn make_table_from_one_column(col: *mut c_void) -> *mut c_void {
    let mut col_arr = [col];
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(col_arr.as_mut_ptr(), 1, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    // ownership moved into table
    assert!(col_arr[0].is_null());
    out
}

pub(crate) fn read_i32_column_data(col: *mut c_void, expected_len: usize) -> Vec<i32> {
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    read_i32_column_data_with_exec(col, expected_len, &exec)
}

pub(crate) fn read_i32_column_data_with_exec(
    col: *mut c_void,
    expected_len: usize,
    exec: &cudf_sys::cudf_exec_options_t,
) -> Vec<i32> {
    let view = make_column_view(col);
    let mut buf = vec![0i32; expected_len];
    let buf_size = expected_len * std::mem::size_of::<i32>();
    let st = unsafe { cudf_sys::cudf_column_view_to_host_ex(view, buf.as_mut_ptr() as *mut c_void, buf_size, exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    buf
}

pub(crate) fn read_string_column_data(col: *mut c_void) -> Vec<String> {
    let view = make_column_view(col);
    let mut size = 0i32;
    let st = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut offsets = vec![0i32; size as usize + 1];
    let mut total_chars = -1i32;
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            ptr::null_mut(),
            0,
            &mut total_chars,
            &exec,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);

    let mut chars = vec![0u8; total_chars.max(0) as usize];
    let st = unsafe {
        cudf_sys::cudf_column_view_strings_to_host_ex(
            view,
            offsets.as_mut_ptr(),
            std::mem::size_of_val(offsets.as_slice()),
            chars.as_mut_ptr() as *mut c_char,
            chars.len(),
            &mut total_chars,
            &exec,
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

pub(crate) fn make_table_view(tbl: *mut c_void) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_view(tbl, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

pub(crate) fn make_table_from_columns(mut columns: Vec<*mut c_void>) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), columns.len() as i32, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert!(columns.iter().all(|handle| handle.is_null()));
    out
}

pub(crate) fn make_i32_scalar(value: i32) -> *mut c_void {
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_scalar_create_numeric_ex(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            &value as *const i32 as *const c_void,
            1,
            &exec,
            &mut out,
        )
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    stream.synchronize();
    out
}

pub(crate) fn make_i32_data_type() -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_data_type_create(cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

pub(crate) fn make_dictionary_i32_column(data: &[i32]) -> *mut c_void {
    let col = make_i32_column(data);
    let view = make_column_view(col);
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe {
        cudf_sys::cudf_dictionary_encode_ex(view, cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out, &exec)
    };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }
    out
}

pub(crate) fn decode_dictionary_i32_column(dict_col: *mut c_void, expected_len: usize) -> Vec<i32> {
    let dict_view = make_column_view(dict_col);
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut decoded: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_dictionary_decode_ex(dict_view, &mut decoded, &exec) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!decoded.is_null());
    stream.synchronize();
    unsafe { cudf_sys::cudf_column_view_destroy(dict_view) };

    let data = read_i32_column_data(decoded, expected_len);
    unsafe { cudf_sys::cudf_column_destroy(decoded) };
    data
}

pub(crate) fn make_sliced_column_view(col: *mut c_void, offset: usize, len: usize) -> *mut c_void {
    let base = make_column_view(col);
    let stream = OwnedStream::new();
    let mr = make_stats_memory_resource();
    let borrowed_stream = stream.borrowed();
    let borrowed_mr = mr.borrowed();
    let exec = required_exec_options(&borrowed_stream, &borrowed_mr);
    let mut out: *mut c_void = ptr::null_mut();
    let st = unsafe { cudf_sys::cudf_column_view_slice_ex(base, offset as i32, len as i32, &exec, &mut out) };
    assert_eq!(st, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    unsafe { cudf_sys::cudf_column_view_destroy(base) };
    out
}

pub(crate) fn unique_missing_path(extension: &str) -> std::ffi::CString {
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
