use std::{ffi::c_void, ptr};

use crate::{
    helpers::{make_i32_column, make_table_from_one_column, make_table_view, read_i32_column_data_with_exec},
    support::{
        OwnedMemoryResource, OwnedStream, last_exec_mr_raw, last_exec_stream_value, lock_shim_state,
        required_exec_options, reset_exec_observation, stream_value,
    },
};

const JOIN_NO_MATCH: i32 = i32::MIN;

#[test]
fn mixed_left_state_requires_eof_and_retains_predicate_owner() {
    let _lock = lock_shim_state();
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_test_stats();
    let target_stream = stream.borrowed();
    let target_mr = mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_mr);
    let inputs = MixedJoinInputs::new();
    let predicate = make_left_less_than_right_predicate();
    let mut state = ptr::null_mut();
    unsafe {
        assert_eq!(
            cudf_sys::cudf_mixed_left_join_state_create_ex(
                inputs.left_equality,
                inputs.left_conditional,
                predicate,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                &mut state,
                &exec,
            ),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(cudf_sys::cudf_ast_expression_destroy(predicate), cudf_sys::CUDF_SUCCESS);
        let mut rows = usize::MAX;
        assert_ne!(cudf_sys::cudf_mixed_left_join_state_output_size(state, &mut rows), cudf_sys::CUDF_SUCCESS);
        assert_eq!(rows, 0);
        let mut output = ptr::null_mut();
        assert_ne!(
            cudf_sys::cudf_mixed_left_join_state_materialize_indices_ex(state, &mut output, &exec),
            cudf_sys::CUDF_SUCCESS
        );
        assert!(output.is_null());
        let other_stream = OwnedStream::new();
        let other_borrowed = other_stream.borrowed();
        let other_exec = required_exec_options(&other_borrowed, &target_mr);
        assert_ne!(
            cudf_sys::cudf_mixed_left_join_state_consume_ex(
                state,
                inputs.right_equality,
                inputs.right_conditional,
                &other_exec,
            ),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(
            cudf_sys::cudf_mixed_left_join_state_consume_ex(
                state,
                inputs.right_equality,
                inputs.right_conditional,
                &exec,
            ),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(
            cudf_sys::cudf_mixed_left_join_state_seal_ex(state, cudf_sys::CUDF_JOIN_KIND_LEFT_ANTI_JOIN, &exec,),
            cudf_sys::CUDF_SUCCESS
        );
        assert_ne!(
            cudf_sys::cudf_mixed_left_join_state_consume_ex(
                state,
                inputs.right_equality,
                inputs.right_conditional,
                &exec,
            ),
            cudf_sys::CUDF_SUCCESS
        );
        assert_ne!(
            cudf_sys::cudf_mixed_left_join_state_seal_ex(state, cudf_sys::CUDF_JOIN_KIND_LEFT_SEMI_JOIN, &exec,),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(cudf_sys::cudf_mixed_left_join_state_output_size(state, &mut rows), cudf_sys::CUDF_SUCCESS);
        assert_eq!(rows, 2);
        assert_eq!(
            cudf_sys::cudf_mixed_left_join_state_materialize_indices_ex(state, &mut output, &exec),
            cudf_sys::CUDF_SUCCESS
        );
        assert_eq!(cudf_sys::cudf_mixed_left_join_state_destroy(state), cudf_sys::CUDF_SUCCESS);
        assert_eq!(read_all_i32(output, &exec), vec![1, 2]);
        assert_eq!(cudf_sys::cudf_column_destroy(output), cudf_sys::CUDF_SUCCESS);
    }
}

type BeginSizeDataExRaw = unsafe extern "C" fn(
    cudf_sys::cudf_table_view_handle,
    cudf_sys::cudf_table_view_handle,
    cudf_sys::cudf_table_view_handle,
    cudf_sys::cudf_table_view_handle,
    cudf_sys::cudf_ast_expression_handle,
    cudf_sys::cudf_null_equality_t,
    *mut cudf_sys::cudf_mixed_join_size_data_handle,
    *const cudf_sys::cudf_exec_options_t,
) -> cudf_sys::cudf_status_t;

struct MixedJoinInputs {
    left_equality_tbl: cudf_sys::cudf_table_handle,
    right_equality_tbl: cudf_sys::cudf_table_handle,
    left_conditional_tbl: cudf_sys::cudf_table_handle,
    right_conditional_tbl: cudf_sys::cudf_table_handle,
    left_equality: cudf_sys::cudf_table_view_handle,
    right_equality: cudf_sys::cudf_table_view_handle,
    left_conditional: cudf_sys::cudf_table_view_handle,
    right_conditional: cudf_sys::cudf_table_view_handle,
    predicate: cudf_sys::cudf_ast_expression_handle,
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

fn make_ast_column_reference(
    table_reference: cudf_sys::cudf_ast_table_reference_t,
) -> cudf_sys::cudf_ast_expression_handle {
    let mut out: cudf_sys::cudf_ast_expression_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_ast_column_reference_create(0, table_reference, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    out
}

fn make_left_less_than_right_predicate() -> cudf_sys::cudf_ast_expression_handle {
    let left = make_ast_column_reference(cudf_sys::CUDF_AST_TABLE_REFERENCE_LEFT);
    let right = make_ast_column_reference(cudf_sys::CUDF_AST_TABLE_REFERENCE_RIGHT);
    let mut out: cudf_sys::cudf_ast_expression_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_ast_operation_binary_create(cudf_sys::CUDF_AST_LESS, left, right, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    unsafe {
        cudf_sys::cudf_ast_expression_destroy(right);
        cudf_sys::cudf_ast_expression_destroy(left);
    }
    out
}

fn column_len(col: cudf_sys::cudf_column_handle) -> usize {
    let mut view: cudf_sys::cudf_column_view_handle = ptr::null_mut();
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

fn read_all_i32(col: cudf_sys::cudf_column_handle, exec: &cudf_sys::cudf_exec_options_t) -> Vec<i32> {
    read_i32_column_data_with_exec(col, column_len(col), exec)
}

fn assert_size_data_payload(
    name: &str,
    size_data: cudf_sys::cudf_mixed_join_size_data_handle,
    expected_size: usize,
    expected_left_indices: &[i32],
    expected_right_indices: &[i32],
    exec: &cudf_sys::cudf_exec_options_t,
    expected_stream: usize,
    expected_mr_raw: *mut c_void,
) {
    let mut output_size = usize::MAX;
    let status = unsafe { cudf_sys::cudf_mixed_join_size_data_output_size(size_data, &mut output_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} output_size failed");
    assert_eq!(output_size, expected_size, "{name} output size");

    let mut device_allocated_size = usize::MAX;
    let status =
        unsafe { cudf_sys::cudf_mixed_join_size_data_device_allocated_size(size_data, &mut device_allocated_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} device_allocated_size failed");
    assert!(device_allocated_size > 0, "{name} non-empty mixed join size data should retain device bytes");

    let mut gather_map_size = usize::MAX;
    let status = unsafe { cudf_sys::cudf_mixed_join_size_data_gather_map_size(size_data, &mut gather_map_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} gather_map_size failed");
    assert_eq!(gather_map_size, expected_size * 2 * std::mem::size_of::<i32>(), "{name} exact pair gather-map bytes");

    reset_exec_observation();
    let mut left_indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut right_indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_mixed_join_size_data_materialize_indices_ex(
            size_data,
            &mut left_indices,
            &mut right_indices,
            exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} materialize_indices failed");
    assert!(!left_indices.is_null(), "{name} did not publish left indices");
    assert!(!right_indices.is_null(), "{name} did not publish right indices");
    assert_eq!(last_exec_stream_value(), expected_stream, "{name} materialize stream");
    assert_eq!(last_exec_mr_raw(), expected_mr_raw, "{name} materialize memory resource");
    let mut actual_pairs =
        read_all_i32(left_indices, exec).into_iter().zip(read_all_i32(right_indices, exec)).collect::<Vec<_>>();
    let mut expected_pairs =
        expected_left_indices.iter().copied().zip(expected_right_indices.iter().copied()).collect::<Vec<_>>();
    actual_pairs.sort_unstable();
    expected_pairs.sort_unstable();
    assert_eq!(actual_pairs, expected_pairs, "{name} indices");

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
    let target_mr = OwnedMemoryResource::new_test_stats();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let expected_stream = stream_value(stream.as_raw());
    let expected_mr_raw = target_mr.raw_device_resource();
    let inputs = MixedJoinInputs::new();

    let target_before = target_mr.allocation_count();
    reset_exec_observation();

    let mut size_data: cudf_sys::cudf_mixed_join_size_data_handle = ptr::null_mut();
    let status = unsafe {
        begin(
            inputs.left_equality,
            inputs.right_equality,
            inputs.left_conditional,
            inputs.right_conditional,
            inputs.predicate,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            &mut size_data,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} begin_size_data failed");
    assert!(!size_data.is_null(), "{name} did not publish size-data handle");
    assert_eq!(last_exec_stream_value(), expected_stream, "{name} begin stream");
    assert_eq!(last_exec_mr_raw(), expected_mr_raw, "{name} begin memory resource");
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
    unsafe { cudf_sys::cudf_mixed_join_size_data_destroy(size_data) };
}

fn assert_single_size_data_case(name: &str, begin: BeginSizeDataExRaw, expected_size: usize, expected_indices: &[i32]) {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let expected_stream = stream_value(stream.as_raw());
    let expected_mr_raw = target_mr.raw_device_resource();
    let inputs = MixedJoinInputs::new();

    let target_before = target_mr.allocation_count();
    reset_exec_observation();

    let mut size_data: cudf_sys::cudf_mixed_join_size_data_handle = ptr::null_mut();
    let status = unsafe {
        begin(
            inputs.left_equality,
            inputs.right_equality,
            inputs.left_conditional,
            inputs.right_conditional,
            inputs.predicate,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            &mut size_data,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} begin_size_data failed");
    assert!(!size_data.is_null(), "{name} did not publish size-data handle");
    assert_eq!(last_exec_stream_value(), expected_stream, "{name} begin stream");
    assert_eq!(last_exec_mr_raw(), expected_mr_raw, "{name} begin memory resource");
    assert!(
        target_mr.allocation_count() > target_before,
        "{name} begin must allocate through the explicit execution memory resource"
    );

    let mut output_size = usize::MAX;
    let status = unsafe { cudf_sys::cudf_mixed_join_size_data_output_size(size_data, &mut output_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} output_size failed");
    assert_eq!(output_size, expected_size, "{name} output size");

    let mut device_allocated_size = usize::MAX;
    let status =
        unsafe { cudf_sys::cudf_mixed_join_size_data_device_allocated_size(size_data, &mut device_allocated_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} device_allocated_size failed");
    assert!(device_allocated_size > 0, "{name} retained mask should report device bytes");

    let mut gather_map_size = usize::MAX;
    let status = unsafe { cudf_sys::cudf_mixed_join_size_data_gather_map_size(size_data, &mut gather_map_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} gather_map_size failed");
    assert_eq!(gather_map_size, expected_size * std::mem::size_of::<i32>(), "{name} exact single gather-map bytes");

    reset_exec_observation();
    let mut indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status =
        unsafe { cudf_sys::cudf_mixed_join_size_data_materialize_single_indices_ex(size_data, &mut indices, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} materialize_single_indices failed");
    assert!(!indices.is_null(), "{name} did not publish indices");
    assert_eq!(last_exec_stream_value(), expected_stream, "{name} materialize stream");
    assert_eq!(last_exec_mr_raw(), expected_mr_raw, "{name} materialize memory resource");
    assert_eq!(read_all_i32(indices, &exec), expected_indices, "{name} indices");

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_column_destroy(indices);
        cudf_sys::cudf_mixed_join_size_data_destroy(size_data);
    }
}

#[test]
fn mixed_inner_join_size_data_reports_exact_size_and_materializes_indices() {
    assert_size_data_case("inner", cudf_sys::cudf_mixed_inner_join_begin_size_data_ex, 1, &[0], &[0]);
}

#[test]
fn mixed_left_join_size_data_reports_exact_size_and_materializes_indices() {
    assert_size_data_case(
        "left",
        cudf_sys::cudf_mixed_left_join_begin_size_data_ex,
        3,
        &[0, 1, 2],
        &[0, JOIN_NO_MATCH, JOIN_NO_MATCH],
    );
}

#[test]
fn mixed_full_join_size_data_reports_exact_size_and_materializes_indices() {
    assert_size_data_case(
        "full",
        cudf_sys::cudf_mixed_full_join_begin_size_data_ex,
        4,
        &[0, 1, 2, JOIN_NO_MATCH],
        &[0, JOIN_NO_MATCH, JOIN_NO_MATCH, 1],
    );
}

#[test]
fn mixed_left_semi_join_size_data_reports_exact_size_and_materializes_indices() {
    assert_single_size_data_case("semi", cudf_sys::cudf_mixed_left_semi_join_begin_size_data_ex, 1, &[0]);
}

#[test]
fn mixed_left_anti_join_size_data_reports_exact_size_and_materializes_indices() {
    assert_single_size_data_case("anti", cudf_sys::cudf_mixed_left_anti_join_begin_size_data_ex, 2, &[1, 2]);
}

#[test]
fn mixed_join_size_data_ex_forms_reject_null_exec_options() {
    let _shim_state = lock_shim_state();
    let inputs = MixedJoinInputs::new();

    for (name, begin) in [
        ("inner", cudf_sys::cudf_mixed_inner_join_begin_size_data_ex as BeginSizeDataExRaw),
        ("left", cudf_sys::cudf_mixed_left_join_begin_size_data_ex),
        ("full", cudf_sys::cudf_mixed_full_join_begin_size_data_ex),
    ] {
        let mut size_data: cudf_sys::cudf_mixed_join_size_data_handle = ptr::null_mut();
        let status = unsafe {
            begin(
                inputs.left_equality,
                inputs.right_equality,
                inputs.left_conditional,
                inputs.right_conditional,
                inputs.predicate,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                &mut size_data,
                ptr::null(),
            )
        };
        assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION, "{name} begin must reject null options");
        assert!(size_data.is_null(), "{name} begin must not publish size-data handle");
    }

    for (name, begin) in [
        ("semi", cudf_sys::cudf_mixed_left_semi_join_begin_size_data_ex as BeginSizeDataExRaw),
        ("anti", cudf_sys::cudf_mixed_left_anti_join_begin_size_data_ex),
    ] {
        let mut size_data: cudf_sys::cudf_mixed_join_size_data_handle = ptr::null_mut();
        let status = unsafe {
            begin(
                inputs.left_equality,
                inputs.right_equality,
                inputs.left_conditional,
                inputs.right_conditional,
                inputs.predicate,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                &mut size_data,
                ptr::null(),
            )
        };
        assert_eq!(status, cudf_sys::CUDF_ERROR_INVALID_OPERATION, "{name} begin must reject null options");
        assert!(size_data.is_null(), "{name} begin must not publish size-data handle");
    }
}
