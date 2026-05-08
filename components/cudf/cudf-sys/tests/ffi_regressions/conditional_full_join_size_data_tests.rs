use std::{ffi::c_void, ptr};

use crate::{
    helpers::{make_i32_column, make_table_from_one_column, make_table_view, read_i32_column_data},
    support::{OwnedStream, lock_shim_state},
};

const JOIN_NO_MATCH: i32 = i32::MIN;

struct ConditionalJoinInputs {
    left_tbl: *mut c_void,
    right_tbl: *mut c_void,
    left: *mut c_void,
    right: *mut c_void,
    predicate: *mut c_void,
}

impl ConditionalJoinInputs {
    fn new(left_values: &[i32], right_values: &[i32]) -> Self {
        Self::new_with_predicate(left_values, right_values, make_left_less_than_right_predicate())
    }

    fn new_with_predicate(left_values: &[i32], right_values: &[i32], predicate: *mut c_void) -> Self {
        let left_tbl = make_table_from_one_column(make_i32_column(left_values));
        let right_tbl = make_table_from_one_column(make_i32_column(right_values));
        let left = make_table_view(left_tbl);
        let right = make_table_view(right_tbl);
        Self { left_tbl, right_tbl, left, right, predicate }
    }
}

impl Drop for ConditionalJoinInputs {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_ast_expression_destroy(self.predicate);
            cudf_sys::cudf_table_view_destroy(self.right);
            cudf_sys::cudf_table_view_destroy(self.left);
            cudf_sys::cudf_table_destroy(self.right_tbl);
            cudf_sys::cudf_table_destroy(self.left_tbl);
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
    make_left_binary_right_predicate(cudf_sys::cudf_ast_operator_t::CUDF_AST_LESS)
}

fn make_left_add_right_predicate() -> *mut c_void {
    make_left_binary_right_predicate(cudf_sys::cudf_ast_operator_t::CUDF_AST_ADD)
}

fn make_left_binary_right_predicate(op: cudf_sys::cudf_ast_operator_t) -> *mut c_void {
    let left = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_LEFT);
    let right = make_ast_column_reference(cudf_sys::cudf_ast_table_reference_t::CUDF_AST_TABLE_REFERENCE_RIGHT);
    let mut out: *mut c_void = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_ast_operation_binary_create(op, left, right, &mut out) };
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
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    size as usize
}

fn read_all_i32(col: *mut c_void) -> Vec<i32> {
    read_i32_column_data(col, column_len(col))
}

#[test]
fn conditional_full_join_size_data_reports_and_materializes_pair_contract() {
    let _shim_state = lock_shim_state();
    let inputs = ConditionalJoinInputs::new(&[2, 5, 9], &[1, 3, 6]);
    let mut size_data: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_full_join_begin_size_data(
            inputs.left,
            inputs.right,
            inputs.predicate,
            &mut size_data,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!size_data.is_null());

    let mut output_size = usize::MAX;
    let status = unsafe { cudf_sys::cudf_conditional_full_join_size_data_output_size(size_data, &mut output_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(output_size, 5);

    let mut gather_map_size = usize::MAX;
    let status =
        unsafe { cudf_sys::cudf_conditional_full_join_size_data_gather_map_size(size_data, &mut gather_map_size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(gather_map_size, 2 * output_size * std::mem::size_of::<i32>());

    let mut left_indices: *mut c_void = ptr::null_mut();
    let mut right_indices: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_full_join_size_data_materialize_indices(
            size_data,
            &mut left_indices,
            &mut right_indices,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(read_all_i32(left_indices), vec![0, 0, 1, 2, JOIN_NO_MATCH]);
    assert_eq!(read_all_i32(right_indices), vec![1, 2, 2, JOIN_NO_MATCH, 0]);

    unsafe {
        cudf_sys::cudf_column_destroy(right_indices);
        cudf_sys::cudf_column_destroy(left_indices);
        cudf_sys::cudf_conditional_full_join_size_data_destroy(size_data);
    }
}

#[test]
fn conditional_full_join_size_data_empty_input_rejects_non_boolean_predicate() {
    let _shim_state = lock_shim_state();
    let inputs = ConditionalJoinInputs::new_with_predicate(&[], &[1, 2], make_left_add_right_predicate());
    let mut size_data: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_full_join_begin_size_data(
            inputs.left,
            inputs.right,
            inputs.predicate,
            &mut size_data,
        )
    };

    assert_ne!(status, cudf_sys::CUDF_SUCCESS);
    assert!(size_data.is_null());
}

#[test]
fn conditional_full_join_size_data_materialized_indices_survive_destroy_before_stream_sync() {
    let _shim_state = lock_shim_state();
    let stream = OwnedStream::new();
    let borrowed_stream = stream.borrowed();
    let exec = cudf_sys::ExecutionContext::new().with_stream(&borrowed_stream);
    let inputs = ConditionalJoinInputs::new(&[2, 5, 9], &[1, 3, 6]);

    let mut size_data: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_full_join_begin_size_data_ex(
            inputs.left,
            inputs.right,
            inputs.predicate,
            &mut size_data,
            exec.as_raw(),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!size_data.is_null());

    let mut left_indices: *mut c_void = ptr::null_mut();
    let mut right_indices: *mut c_void = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_full_join_size_data_materialize_indices_ex(
            size_data,
            &mut left_indices,
            &mut right_indices,
            exec.as_raw(),
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!left_indices.is_null());
    assert!(!right_indices.is_null());

    unsafe { cudf_sys::cudf_conditional_full_join_size_data_destroy(size_data) };
    stream.synchronize();

    assert_eq!(read_all_i32(left_indices), vec![0, 0, 1, 2, JOIN_NO_MATCH]);
    assert_eq!(read_all_i32(right_indices), vec![1, 2, 2, JOIN_NO_MATCH, 0]);

    unsafe {
        cudf_sys::cudf_column_destroy(right_indices);
        cudf_sys::cudf_column_destroy(left_indices);
    }
}
