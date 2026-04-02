mod support;

use std::{
    ffi::{CString, c_void},
    ptr,
};

use support::{last_exec_stream_value, lock_shim_state, reset_exec_observation, stream_value};

fn validity_to_bitmask(validity: &[bool]) -> Vec<u8> {
    let mut mask = vec![0u8; validity.len().div_ceil(8)];
    for (index, is_valid) in validity.iter().copied().enumerate() {
        if is_valid {
            mask[index / 8] |= 1 << (index % 8);
        }
    }
    mask
}

fn nullable_bitmask_payload_bytes(len: usize) -> usize {
    len.div_ceil(32) * std::mem::size_of::<u32>()
}

fn make_i32_column(data: &[i32]) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_i32_column_nullable(data: &[i32], validity: &[bool]) -> *mut c_void {
    assert_eq!(data.len(), validity.len());
    let mask = validity_to_bitmask(validity);
    let null_count = validity.iter().filter(|&&is_valid| !is_valid).count() as i32;

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_numeric(
            cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32,
            data.as_ptr() as *const c_void,
            data.len() as i32,
            mask.as_ptr(),
            null_count,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_string_column(data: &[&str]) -> *mut c_void {
    let c_strings: Vec<_> = data.iter().map(|value| CString::new(*value).unwrap()).collect();
    let string_ptrs: Vec<_> = c_strings.iter().map(|value| value.as_ptr()).collect();
    let lengths: Vec<_> = data.iter().map(|value| value.len() as i32).collect();

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings(
            string_ptrs.as_ptr(),
            lengths.as_ptr(),
            data.len() as i32,
            ptr::null(),
            0,
            &mut out,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_column_view(column: *mut c_void) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_view(column, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_string_dictionary_i32_column(data: &[&str]) -> *mut c_void {
    let col = make_string_column(data);
    let view = make_column_view(col);

    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_dictionary_encode(view, cudf_sys::cudf_type_id_t::CUDF_TYPE_INT32, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    unsafe {
        cudf_sys::cudf_column_view_destroy(view);
        cudf_sys::cudf_column_destroy(col);
    }

    out
}

fn make_table(columns: &mut [*mut c_void]) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_create(columns.as_mut_ptr(), columns.len() as i32, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn column_payload_size(handle: *mut c_void) -> usize {
    let mut out = 0usize;
    let status = unsafe { cudf_sys::cudf_column_payload_size(handle, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out
}

fn column_device_allocated_size(handle: *mut c_void) -> usize {
    let mut out = 0usize;
    let status = unsafe { cudf_sys::cudf_column_device_allocated_size(handle, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out
}

fn table_payload_size(handle: *mut c_void) -> usize {
    let mut out = 0usize;
    let status = unsafe { cudf_sys::cudf_table_payload_size(handle, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out
}

fn table_device_allocated_size(handle: *mut c_void) -> usize {
    let mut out = 0usize;
    let status = unsafe { cudf_sys::cudf_table_device_allocated_size(handle, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out
}

#[test]
fn numeric_column_size_hooks_report_expected_payload_and_device_allocated_bytes() {
    let col = make_i32_column(&[1, 2, 3, 4]);

    let payload = column_payload_size(col);
    let device_allocated = column_device_allocated_size(col);

    assert_eq!(payload, 4 * std::mem::size_of::<i32>());
    assert!(device_allocated >= payload);

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

#[test]
fn nullable_numeric_column_payload_size_counts_bitmask_payload_not_allocator_rounding() {
    let col = make_i32_column_nullable(&[10, 0, 30], &[true, false, true]);

    let payload = column_payload_size(col);
    let device_allocated = column_device_allocated_size(col);

    assert_eq!(payload, (3 * std::mem::size_of::<i32>()) + nullable_bitmask_payload_bytes(3));
    assert!(device_allocated >= payload);

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

#[test]
fn string_column_size_hooks_account_for_offsets_and_chars_payload() {
    let col = make_string_column(&["ab", "cde", ""]);

    let payload = column_payload_size(col);
    let device_allocated = column_device_allocated_size(col);

    assert_eq!(payload, (4 * std::mem::size_of::<i32>()) + 5);
    assert!(device_allocated >= payload);

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

#[test]
fn dictionary_column_size_hooks_include_indices_and_recursive_key_payload() {
    let col = make_string_dictionary_i32_column(&["beta", "alpha", "beta", "gamma"]);

    let payload = column_payload_size(col);
    let device_allocated = column_device_allocated_size(col);

    assert_eq!(payload, (4 * std::mem::size_of::<i32>()) + (4 * std::mem::size_of::<i32>()) + 14);
    assert!(device_allocated >= payload);

    unsafe { cudf_sys::cudf_column_destroy(col) };
}

#[test]
fn table_size_hooks_sum_column_payloads() {
    let int_col = make_i32_column(&[1, 2, 3]);
    let str_col = make_string_column(&["ab", "cde", ""]);
    let int_payload = column_payload_size(int_col);
    let str_payload = column_payload_size(str_col);

    let mut columns = [int_col, str_col];
    let table = make_table(&mut columns);

    let payload = table_payload_size(table);
    let device_allocated = table_device_allocated_size(table);

    assert_eq!(payload, int_payload + str_payload);
    assert!(device_allocated >= payload);

    unsafe { cudf_sys::cudf_table_destroy(table) };
}

#[test]
fn empty_table_size_hooks_return_zero() {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_table_create(ptr::null_mut(), 0, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());

    assert_eq!(table_payload_size(out), 0);
    assert_eq!(table_device_allocated_size(out), 0);

    unsafe { cudf_sys::cudf_table_destroy(out) };
}

#[test]
fn column_payload_size_ex_uses_explicit_stream_for_string_columns() {
    let _shim_state = lock_shim_state();
    let col = make_string_column(&["ab", "cde", ""]);
    let stream = cudf_sys::Stream::new().unwrap();
    let ctx = cudf_sys::ExecutionContext::new().with_stream(&stream);
    let expected_stream = stream_value(stream.as_raw());

    reset_exec_observation();

    let mut payload = 0usize;
    let status = unsafe { cudf_sys::cudf_column_payload_size_ex(col, ctx.as_raw(), &mut payload) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);

    assert_eq!(payload, (4 * std::mem::size_of::<i32>()) + 5);
    assert_eq!(last_exec_stream_value(), expected_stream);

    unsafe { cudf_sys::cudf_column_destroy(col) };
}
