use std::{ffi::c_void, ptr};

use crate::{
    helpers::{make_i32_column, make_table_from_one_column, make_table_view, read_i32_column_data},
    support::{
        CurrentDeviceResourceRestore, OwnedMemoryResource, OwnedStream, current_device_resource_raw, last_exec_mr_raw,
        last_exec_stream_value, lock_shim_state, reset_exec_observation, stream_value,
    },
};

const JOIN_NO_MATCH: i32 = i32::MIN;

extern "C" {
    #[link_name = "cudf_mixed_inner_join_begin_size_data"]
    fn cudf_mixed_inner_join_begin_size_data_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: cudf_sys::cudf_null_equality_t,
        out: *mut *mut c_void,
    ) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_inner_join_begin_size_data_ex"]
    fn cudf_mixed_inner_join_begin_size_data_ex_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: cudf_sys::cudf_null_equality_t,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_left_join_begin_size_data"]
    fn cudf_mixed_left_join_begin_size_data_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: cudf_sys::cudf_null_equality_t,
        out: *mut *mut c_void,
    ) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_left_join_begin_size_data_ex"]
    fn cudf_mixed_left_join_begin_size_data_ex_raw(
        left_equality: *mut c_void,
        right_equality: *mut c_void,
        left_conditional: *mut c_void,
        right_conditional: *mut c_void,
        predicate: *mut c_void,
        null_equality: cudf_sys::cudf_null_equality_t,
        out: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_join_size_data_output_size"]
    fn cudf_mixed_join_size_data_output_size_raw(handle: *mut c_void, out: *mut usize) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_join_size_data_device_allocated_size"]
    fn cudf_mixed_join_size_data_device_allocated_size_raw(
        handle: *mut c_void,
        out: *mut usize,
    ) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_join_size_data_materialize_indices"]
    fn cudf_mixed_join_size_data_materialize_indices_raw(
        handle: *mut c_void,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
    ) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_join_size_data_materialize_indices_ex"]
    fn cudf_mixed_join_size_data_materialize_indices_ex_raw(
        handle: *mut c_void,
        out_left_indices: *mut *mut c_void,
        out_right_indices: *mut *mut c_void,
        options: *const cudf_sys::cudf_exec_options_t,
    ) -> cudf_sys::cudf_status_t;

    #[link_name = "cudf_mixed_join_size_data_destroy"]
    fn cudf_mixed_join_size_data_destroy_raw(handle: *mut c_void);
}

type BeginSizeDataExRaw = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    cudf_sys::cudf_null_equality_t,
    *mut *mut c_void,
    *const cudf_sys::cudf_exec_options_t,
) -> cudf_sys::cudf_status_t;

struct MixedJoinInputs {
    left_equality_tbl: *mut c_void,
    right_equality_tbl: *mut c_void,
    left_conditional_tbl: *mut c_void,
    right_conditional_tbl: *mut c_void,
    left_equality: *mut c_void,
    right_equality: *mut c_void,
    left_conditional: *mut c_void,
    right_conditional: *mut c_void,
    predicate: *mut c_void,
}

impl MixedJoinInputs {
    fn new() -> Self {
        let left_equality_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 3]));
        let right_equality_tbl = make_table_from_one_column(make_i32_column(&[1, 2]));
        let left_conditional_tbl = make_table_from_one_column(make_i32_column(&[1, 5, 9]));
        let right_conditional_tbl = make_table_from_one_column(make_i32_column(&[2, 4]));
        let left_equality = make_table_view(left_equality_tbl);
        let right_equality = make_table_view(right_equality_tbl);
        let left_conditional = make_table_view(left_conditional_tbl);
        let right_conditional = make_table_view(right_conditional_tbl);
        let predicate = make_left_less_than_right_predicate();

        Self {
            left_equality_tbl,
            right_equality_tbl,
            left_conditional_tbl,
            right_conditional_tbl,
            left_equality,
            right_equality,
            left_conditional,
            right_conditional,
            predicate,
        }
    }
}

impl Drop for MixedJoinInputs {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_ast_expression_destroy(self.predicate);
            cudf_sys::cudf_table_view_destroy(self.right_conditional);
            cudf_sys::cudf_table_view_destroy(self.left_conditional);
            cudf_sys::cudf_table_view_destroy(self.right_equality);
            cudf_sys::cudf_table_view_destroy(self.left_equality);
            cudf_sys::cudf_table_destroy(self.right_conditional_tbl);
            cudf_sys::cudf_table_destroy(self.left_conditional_tbl);
            cudf_sys::cudf_table_destroy(self.right_equality_tbl);
            cudf_sys::cudf_table_destroy(self.left_equality_tbl);
        }
    }
}

fn make_ast_column_reference(table_reference: cudf_sys::cudf_ast_table_reference_t) -> *mut c_void {
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_ast_column_reference_create(0, table_reference, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_left_less_than_right_predicate() -> *mut c_void {
    let left = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_LEFT);
    let right = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_RIGHT);
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_ast_operation_binary_create(cudf_sys::cudf_ast_operator_t::CUDF_AST_LESS, left, right, &mut out)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    unsafe {
        cudf_sys::cudf_ast_expression_destroy(right);
        cudf_sys::cudf_ast_expression_destroy(left);
    }
    out
}

fn column_len(col: *mut c_void) -> usize {
    let mut view: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_column_view(col, &mut view) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!view.is_null());

    let mut size = 0;
    let status = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(size >= 0);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    size as usize
}

fn read_all_i32(col: *mut c_void) -> Vec<i32> {
    read_i32_column_data(col, column_len(col))
}

fn assert_size_data_payload(
    name: &str,
    size_data: *mut c_void,
    expected_size: usize,
    expected_left_indices: &[i32],
    expected_right_indices: &[i32],
    exec: &cudf_sys::ExecutionContext<'_>,
    expected_stream: usize,
    expected_mr_raw: *mut c_void,
) {
    let mut output_size = usize::MAX;
    let status = unsafe { cudf_mixed_join_size_data_output_size_raw(size_data, &mut output_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} output_size failed");
    assert_eq!(output_size, expected_size, "{name} output size");

    let mut device_allocated_size = usize::MAX;
    let status = unsafe { cudf_mixed_join_size_data_device_allocated_size_raw(size_data, &mut device_allocated_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} device_allocated_size failed");
    assert!(device_allocated_size > 0, "{name} non-empty mixed join size data should retain device bytes");

    reset_exec_observation();
    let mut left_indices: *mut c_void = ptr::null_mut();
    let mut right_indices: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_mixed_join_size_data_materialize_indices_ex_raw(
            size_data,
            &mut left_indices,
            &mut right_indices,
            exec.as_raw(),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} materialize_indices failed");
    assert!(!left_indices.is_null(), "{name} did not publish left indices");
    assert!(!right_indices.is_null(), "{name} did not publish right indices");
    assert_eq!(last_exec_stream_value(), expected_stream, "{name} materialize stream");
    assert_eq!(last_exec_mr_raw(), expected_mr_raw, "{name} materialize memory resource");
    assert_eq!(read_all_i32(left_indices), expected_left_indices, "{name} left indices");
    assert_eq!(read_all_i32(right_indices), expected_right_indices, "{name} right indices");

    unsafe {
        cudf_sys::cudf_column_destroy(right_indices);
        cudf_sys::cudf_column_destroy(left_indices);
    }
}

fn assert_size_data_case(
    name: &str,
    begin: BeginSizeDataExRaw,
    expected_size: usize,
    expected_left_indices: &[i32],
    expected_right_indices: &[i32],
) {
    let _shim_state = lock_shim_state();
    let previous_raw = current_device_resource_raw();
    let current_mr = OwnedMemoryResource::new_test_stats();
    let _restore = CurrentDeviceResourceRestore::new(previous_raw);
    let target_mr = OwnedMemoryResource::new_test_stats();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&target_stream).with_memory_resource(&target_borrowed);
    let expected_stream = stream_value(stream.as_raw());
    let expected_mr_raw = target_mr.raw_device_resource();
    let inputs = MixedJoinInputs::new();

    let target_before = target_mr.allocation_count();
    let current_before = current_mr.allocation_count();
    let current_borrowed = current_mr.borrowed();
    reset_exec_observation();
    cudf_sys::test_set_current_device_resource(&current_borrowed)
        .expect("failed to install current sentinel memory resource");

    let mut size_data: *mut c_void = ptr::null_mut();
    let status = unsafe {
        begin(
            inputs.left_equality,
            inputs.right_equality,
            inputs.left_conditional,
            inputs.right_conditional,
            inputs.predicate,
            cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
            &mut size_data,
            exec.as_raw(),
        )
    };
    let restored_current = current_device_resource_raw();
    unsafe { cudf_sys::test_set_current_device_resource_raw(previous_raw) }
        .expect("failed to restore previous current device resource");

    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} begin_size_data failed");
    assert!(!size_data.is_null(), "{name} did not publish size-data handle");
    assert_eq!(last_exec_stream_value(), expected_stream, "{name} begin stream");
    assert_eq!(last_exec_mr_raw(), expected_mr_raw, "{name} begin memory resource");
    assert_eq!(
        restored_current,
        current_mr.raw_device_resource(),
        "{name} begin must restore the caller's current memory resource"
    );
    assert_eq!(
        current_mr.allocation_count(),
        current_before,
        "{name} begin temporaries must not allocate from the caller's current memory resource"
    );
    assert!(
        target_mr.allocation_count() > target_before,
        "{name} begin must allocate through the explicit execution memory resource"
    );

    assert_size_data_payload(
        name,
        size_data,
        expected_size,
        expected_left_indices,
        expected_right_indices,
        &exec,
        expected_stream,
        expected_mr_raw,
    );

    stream.synchronize();
    unsafe { cudf_mixed_join_size_data_destroy_raw(size_data) };
}

#[test]
fn mixed_inner_join_size_data_reports_exact_size_and_materializes_indices() {
    assert_size_data_case("inner", cudf_mixed_inner_join_begin_size_data_ex_raw, 1, &[0], &[0]);
}

#[test]
fn mixed_left_join_size_data_reports_exact_size_and_materializes_indices() {
    assert_size_data_case(
        "left",
        cudf_mixed_left_join_begin_size_data_ex_raw,
        3,
        &[0, 1, 2],
        &[0, JOIN_NO_MATCH, JOIN_NO_MATCH],
    );
}

#[test]
fn mixed_join_size_data_default_forms_match_ex_contract() {
    let _shim_state = lock_shim_state();
    let inputs = MixedJoinInputs::new();

    for (name, begin, expected_size) in [
        (
            "inner",
            cudf_mixed_inner_join_begin_size_data_raw
                as unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    cudf_sys::cudf_null_equality_t,
                    *mut *mut c_void,
                ) -> cudf_sys::cudf_status_t,
            1usize,
        ),
        ("left", cudf_mixed_left_join_begin_size_data_raw, 3usize),
    ] {
        let mut size_data: *mut c_void = ptr::null_mut();
        let status = unsafe {
            begin(
                inputs.left_equality,
                inputs.right_equality,
                inputs.left_conditional,
                inputs.right_conditional,
                inputs.predicate,
                cudf_sys::cudf_null_equality_t::CUDF_NULL_EQUALITY_EQUAL,
                &mut size_data,
            )
        };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} default begin_size_data failed");
        assert!(!size_data.is_null(), "{name} default begin did not publish size-data handle");

        let mut output_size = usize::MAX;
        let status = unsafe { cudf_mixed_join_size_data_output_size_raw(size_data, &mut output_size) };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} default output_size failed");
        assert_eq!(output_size, expected_size, "{name} default output size");

        let mut left_indices: *mut c_void = ptr::null_mut();
        let mut right_indices: *mut c_void = ptr::null_mut();
        let status = unsafe {
            cudf_mixed_join_size_data_materialize_indices_raw(size_data, &mut left_indices, &mut right_indices)
        };
        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} default materialize_indices failed");
        assert_eq!(column_len(left_indices), expected_size);
        assert_eq!(column_len(right_indices), expected_size);

        unsafe {
            cudf_sys::cudf_column_destroy(right_indices);
            cudf_sys::cudf_column_destroy(left_indices);
            cudf_mixed_join_size_data_destroy_raw(size_data);
        }
    }
}
