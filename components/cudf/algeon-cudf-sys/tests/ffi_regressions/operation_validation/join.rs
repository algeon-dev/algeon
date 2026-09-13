use std::{os::raw::c_void, ptr};

use cudf_sys::test_support;

use super::*;

fn column_len(col: cudf_sys::cudf_column_handle) -> usize {
    let view = make_column_view(col);
    let mut size = 0;
    let status = unsafe { cudf_sys::cudf_column_view_size(view, &mut size) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(size >= 0);
    unsafe { cudf_sys::cudf_column_view_destroy(view) };
    size as usize
}

fn read_all_i32_column_data(col: cudf_sys::cudf_column_handle) -> Vec<i32> {
    read_i32_column_data(col, column_len(col))
}

fn assert_readable_nonempty_i32_indices(col: cudf_sys::cudf_column_handle, context: &str) {
    let indices = read_all_i32_column_data(col);
    assert!(!indices.is_empty(), "{context} must return at least one index");
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

const FILTER_JOIN_KIND_INNER: i32 = 0;
const FILTER_JOIN_KIND_LEFT_SEMI: i32 = 3;
const FILTER_JOIN_KIND_LEFT_ANTI: i32 = 4;
const FILTER_JOIN_KIND_INVALID: i32 = 99;

struct FilterJoinTestInputs {
    left_tbl: cudf_sys::cudf_table_handle,
    right_tbl: cudf_sys::cudf_table_handle,
    left_tv: cudf_sys::cudf_table_view_handle,
    right_tv: cudf_sys::cudf_table_view_handle,
    left_indices_col: cudf_sys::cudf_column_handle,
    right_indices_col: cudf_sys::cudf_column_handle,
    left_indices_view: cudf_sys::cudf_column_view_handle,
    right_indices_view: cudf_sys::cudf_column_view_handle,
    predicate: cudf_sys::cudf_ast_expression_handle,
}

impl FilterJoinTestInputs {
    fn new() -> Self {
        let left_tbl = make_table_from_one_column(make_i32_column(&[10, 30, 5, 7]));
        let right_tbl = make_table_from_one_column(make_i32_column(&[20, 10, 6, 8]));
        let left_tv = make_table_view(left_tbl);
        let right_tv = make_table_view(right_tbl);
        let left_indices_col = make_i32_column(&[0, 1, 2, 3]);
        let right_indices_col = make_i32_column(&[0, 1, 2, 3]);
        let left_indices_view = make_column_view(left_indices_col);
        let right_indices_view = make_column_view(right_indices_col);
        let predicate = make_left_less_than_right_predicate();

        Self {
            left_tbl,
            right_tbl,
            left_tv,
            right_tv,
            left_indices_col,
            right_indices_col,
            left_indices_view,
            right_indices_view,
            predicate,
        }
    }
}

impl Drop for FilterJoinTestInputs {
    fn drop(&mut self) {
        unsafe {
            cudf_sys::cudf_column_view_destroy(self.right_indices_view);
            cudf_sys::cudf_column_view_destroy(self.left_indices_view);
            cudf_sys::cudf_column_destroy(self.right_indices_col);
            cudf_sys::cudf_column_destroy(self.left_indices_col);
            cudf_sys::cudf_ast_expression_destroy(self.predicate);
            cudf_sys::cudf_table_view_destroy(self.right_tv);
            cudf_sys::cudf_table_view_destroy(self.left_tv);
            cudf_sys::cudf_table_destroy(self.right_tbl);
            cudf_sys::cudf_table_destroy(self.left_tbl);
        }
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
        *const cudf_sys::cudf_exec_options_t,
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

    let mut out_left: cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_right: cudf_sys::cudf_column_handle = ptr::null_mut();
    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe {
        join_fn(
            raw_handle(left_keys_view),
            raw_handle(right_keys_view),
            raw_handle(left_payload_view),
            raw_handle(right_payload_view),
            99,
            raw_out_handle(&mut out_left),
            raw_out_handle(&mut out_right),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
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

fn assert_join_indices_invalid_null_equality_rejected(
    join_fn: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        i32,
        *mut *mut c_void,
        *mut *mut c_void,
        *const cudf_sys::cudf_exec_options_t,
    ) -> i32,
) {
    let left_key_col = make_i32_column(&[1, 2]);
    let right_key_col = make_i32_column(&[2, 3]);
    let left_keys_tbl = make_table_from_one_column(left_key_col);
    let right_keys_tbl = make_table_from_one_column(right_key_col);
    let left_keys_view = make_table_view(left_keys_tbl);
    let right_keys_view = make_table_view(right_keys_tbl);

    let mut out_left: cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_right: cudf_sys::cudf_column_handle = ptr::null_mut();
    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe {
        join_fn(
            raw_handle(left_keys_view),
            raw_handle(right_keys_view),
            99,
            raw_out_handle(&mut out_left),
            raw_out_handle(&mut out_right),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_left.is_null());
    assert!(out_right.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(right_keys_view);
        cudf_sys::cudf_table_view_destroy(left_keys_view);
        cudf_sys::cudf_table_destroy(right_keys_tbl);
        cudf_sys::cudf_table_destroy(left_keys_tbl);
    }
}

fn assert_filtered_join_indices_invalid_null_equality_rejected(
    join_fn: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        i32,
        *mut *mut c_void,
        *const cudf_sys::cudf_exec_options_t,
    ) -> i32,
) {
    let left_key_col = make_i32_column(&[1, 2]);
    let right_key_col = make_i32_column(&[2, 3]);
    let left_keys_tbl = make_table_from_one_column(left_key_col);
    let right_keys_tbl = make_table_from_one_column(right_key_col);
    let left_keys_view = make_table_view(left_keys_tbl);
    let right_keys_view = make_table_view(right_keys_tbl);

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe {
        join_fn(raw_handle(left_keys_view), raw_handle(right_keys_view), 99, raw_out_handle(&mut out), &exec)
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(right_keys_view);
        cudf_sys::cudf_table_view_destroy(left_keys_view);
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
fn test_inner_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_indices_invalid_null_equality_rejected(cudf_inner_join_indices_raw);
}

#[test]
fn test_left_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_indices_invalid_null_equality_rejected(cudf_left_join_indices_raw);
}

#[test]
fn test_full_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_join_indices_invalid_null_equality_rejected(cudf_full_join_indices_raw);
}

#[test]
fn test_left_semi_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_filtered_join_indices_invalid_null_equality_rejected(cudf_left_semi_join_indices_raw);
}

#[test]
fn test_left_anti_join_indices_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    assert_filtered_join_indices_invalid_null_equality_rejected(cudf_left_anti_join_indices_raw);
}

#[test]
fn test_distinct_hash_join_create_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut join: cudf_sys::cudf_distinct_hash_join_handle = ptr::null_mut();
    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe { cudf_distinct_hash_join_create_raw(raw_handle(tv), 99, raw_out_handle(&mut join), &exec) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(join.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_hash_join_create_invalid_nullable_join_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut join: cudf_sys::cudf_hash_join_handle = ptr::null_mut();
    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe {
        cudf_hash_join_create_raw(
            raw_handle(tv),
            99,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL as i32,
            0.5,
            raw_out_handle(&mut join),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(join.is_null());
    assert!(last_error_message().contains("invalid nullable_join"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_hash_join_create_invalid_null_equality_rejected() {
    let _shim_state = lock_shim_state();
    let col = make_i32_column(&[1, 2, 3]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let mut join: cudf_sys::cudf_hash_join_handle = ptr::null_mut();
    let (_stream, _mr, exec) = strict_exec_owners();
    let status = unsafe { cudf_hash_join_create_raw(raw_handle(tv), 0, 99, 0.5, raw_out_handle(&mut join), &exec) };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(join.is_null());
    assert!(last_error_message().contains("invalid null_equality"));

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_ast_invalid_table_reference_operator_and_null_handle_clear_output() {
    let _shim_state = lock_shim_state();

    let mut out = std::ptr::NonNull::<cudf_sys::cudf_ast_expression_handle_state>::dangling().as_ptr();
    let status = unsafe { cudf_ast_column_reference_create_raw(0, 99, raw_out_handle(&mut out)) };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid AST table reference"));

    let left = make_ast_column_reference(cudf_sys::CUDF_AST_TABLE_REFERENCE_LEFT);
    let right = make_ast_column_reference(cudf_sys::CUDF_AST_TABLE_REFERENCE_RIGHT);

    out = std::ptr::NonNull::<cudf_sys::cudf_ast_expression_handle_state>::dangling().as_ptr();
    let status = unsafe {
        cudf_ast_operation_binary_create_raw(99, raw_handle(left), raw_handle(right), raw_out_handle(&mut out))
    };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid AST operator"));

    out = std::ptr::NonNull::<cudf_sys::cudf_ast_expression_handle_state>::dangling().as_ptr();
    let status =
        unsafe { cudf_sys::cudf_ast_operation_unary_create(cudf_sys::CUDF_AST_NOT, ptr::null_mut(), &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert!(out.is_null());

    out = std::ptr::NonNull::<cudf_sys::cudf_ast_expression_handle_state>::dangling().as_ptr();
    let status = unsafe { cudf_ast_operation_unary_create_raw(99, raw_handle(left), raw_out_handle(&mut out)) };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out.is_null());
    assert!(last_error_message().contains("invalid AST operator"));

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(right);
        cudf_sys::cudf_ast_expression_destroy(left);
    }
}

#[test]
fn test_conditional_join_default_forms_validate_null_handles_and_clear_outputs() {
    let _shim_state = lock_shim_state();
    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 4]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let predicate = make_left_less_than_right_predicate();
    let (_stream, _mr, exec) = strict_exec_owners();

    for (name, join_indices) in [
        ("inner", cudf_conditional_inner_join_indices_raw as ConditionalPairJoinRaw),
        ("left", cudf_conditional_left_join_indices_raw),
        ("full", cudf_conditional_full_join_indices_raw),
    ] {
        let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(left_tv),
                raw_handle(right_tv),
                raw_handle(predicate),
                ptr::null_mut(),
                raw_out_handle(&mut out_right),
                &exec,
            )
        };
        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null left output");
        assert!(last_error_message().contains("out_left_indices is null"));

        let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(left_tv),
                raw_handle(right_tv),
                raw_handle(predicate),
                raw_out_handle(&mut out_left),
                ptr::null_mut(),
                &exec,
            )
        };
        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null right output");
        assert!(out_left.is_null(), "{name} conditional join must clear left output first");
        assert!(last_error_message().contains("out_right_indices is null"));

        for (case, left, right, pred, message) in [
            ("left", ptr::null_mut(), right_tv, predicate, "left is null"),
            ("right", left_tv, ptr::null_mut(), predicate, "right is null"),
            ("predicate", left_tv, right_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
            let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
            let status = unsafe {
                join_indices(
                    raw_handle(left),
                    raw_handle(right),
                    raw_handle(pred),
                    raw_out_handle(&mut out_left),
                    raw_out_handle(&mut out_right),
                    &exec,
                )
            };
            assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null {case} status");
            assert!(out_left.is_null(), "{name} conditional join must clear left output for null {case}");
            assert!(out_right.is_null(), "{name} conditional join must clear right output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    for (name, join_indices) in [
        ("semi", cudf_conditional_left_semi_join_indices_raw as ConditionalSingleJoinRaw),
        ("anti", cudf_conditional_left_anti_join_indices_raw),
    ] {
        let status = unsafe {
            join_indices(raw_handle(left_tv), raw_handle(right_tv), raw_handle(predicate), ptr::null_mut(), &exec)
        };
        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null output");
        assert!(last_error_message().contains("out_indices is null"));

        for (case, left, right, pred, message) in [
            ("left", ptr::null_mut(), right_tv, predicate, "left is null"),
            ("right", left_tv, ptr::null_mut(), predicate, "right is null"),
            ("predicate", left_tv, right_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
            let status = unsafe {
                join_indices(raw_handle(left), raw_handle(right), raw_handle(pred), raw_out_handle(&mut out), &exec)
            };
            assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} conditional join null {case} status");
            assert!(out.is_null(), "{name} conditional join must clear output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_filter_join_indices_invalid_join_kind_rejected_and_clears_outputs() {
    let _shim_state = lock_shim_state();
    let inputs = FilterJoinTestInputs::new();
    let (_stream, _mr, exec) = strict_exec_owners();
    let join_indices = cudf_filter_join_indices_raw as FilterJoinIndicesRaw;

    let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        join_indices(
            raw_handle(inputs.left_tv),
            raw_handle(inputs.right_tv),
            raw_handle(inputs.left_indices_view),
            raw_handle(inputs.right_indices_view),
            raw_handle(inputs.predicate),
            FILTER_JOIN_KIND_INVALID,
            raw_out_handle(&mut out_left),
            raw_out_handle(&mut out_right),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION);
    assert!(out_left.is_null());
    assert!(out_right.is_null());
    assert!(last_error_message().contains("invalid join_kind"));
}

#[test]
fn test_filter_join_indices_unsupported_join_kinds_rejected_and_clear_outputs() {
    let _shim_state = lock_shim_state();
    let inputs = FilterJoinTestInputs::new();
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let join_indices = cudf_filter_join_indices_ex_raw as FilterJoinIndicesExRaw;

    for (name, join_kind) in [("left_semi", FILTER_JOIN_KIND_LEFT_SEMI), ("left_anti", FILTER_JOIN_KIND_LEFT_ANTI)] {
        let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(inputs.left_tv),
                raw_handle(inputs.right_tv),
                raw_handle(inputs.left_indices_view),
                raw_handle(inputs.right_indices_view),
                raw_handle(inputs.predicate),
                join_kind,
                raw_out_handle(&mut out_left),
                raw_out_handle(&mut out_right),
                &exec,
            )
        };

        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_ARGUMENT, "{name} filter join status");
        assert!(out_left.is_null(), "{name} filter join must leave left output null");
        assert!(out_right.is_null(), "{name} filter join must leave right output null");
    }
}

#[test]
fn test_filter_join_indices_mismatched_index_lengths_rejected_and_clear_outputs() {
    let _shim_state = lock_shim_state();
    let inputs = FilterJoinTestInputs::new();
    let short_right_indices_col = make_i32_column(&[0, 1]);
    let short_right_indices_view = make_column_view(short_right_indices_col);
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());
    let join_indices = cudf_filter_join_indices_ex_raw as FilterJoinIndicesExRaw;

    let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        join_indices(
            raw_handle(inputs.left_tv),
            raw_handle(inputs.right_tv),
            raw_handle(inputs.left_indices_view),
            raw_handle(short_right_indices_view),
            raw_handle(inputs.predicate),
            FILTER_JOIN_KIND_INNER,
            raw_out_handle(&mut out_left),
            raw_out_handle(&mut out_right),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_ARGUMENT);
    assert!(out_left.is_null());
    assert!(out_right.is_null());
    assert!(last_error_message().contains("same size"));

    unsafe {
        cudf_sys::cudf_column_view_destroy(short_right_indices_view);
        cudf_sys::cudf_column_destroy(short_right_indices_col);
    }
}

#[test]
fn test_filter_join_indices_null_handles_clear_outputs() {
    let _shim_state = lock_shim_state();
    let inputs = FilterJoinTestInputs::new();
    let (_stream, _mr, exec) = strict_exec_owners();
    let join_indices = cudf_filter_join_indices_raw as FilterJoinIndicesRaw;

    let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        join_indices(
            raw_handle(inputs.left_tv),
            raw_handle(inputs.right_tv),
            raw_handle(inputs.left_indices_view),
            raw_handle(inputs.right_indices_view),
            raw_handle(inputs.predicate),
            FILTER_JOIN_KIND_INNER,
            ptr::null_mut(),
            raw_out_handle(&mut out_right),
            &exec,
        )
    };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL);
    assert!(last_error_message().contains("out_left_indices is null"));

    let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let status = unsafe {
        join_indices(
            raw_handle(inputs.left_tv),
            raw_handle(inputs.right_tv),
            raw_handle(inputs.left_indices_view),
            raw_handle(inputs.right_indices_view),
            raw_handle(inputs.predicate),
            FILTER_JOIN_KIND_INNER,
            raw_out_handle(&mut out_left),
            ptr::null_mut(),
            &exec,
        )
    };
    assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL);
    assert!(out_left.is_null(), "filter join must clear left output before validating right output");
    assert!(last_error_message().contains("out_right_indices is null"));

    for (case, left, right, left_indices, right_indices, predicate, message) in [
        (
            "left",
            ptr::null_mut(),
            inputs.right_tv,
            inputs.left_indices_view,
            inputs.right_indices_view,
            inputs.predicate,
            "left is null",
        ),
        (
            "right",
            inputs.left_tv,
            ptr::null_mut(),
            inputs.left_indices_view,
            inputs.right_indices_view,
            inputs.predicate,
            "right is null",
        ),
        (
            "left_indices",
            inputs.left_tv,
            inputs.right_tv,
            ptr::null_mut(),
            inputs.right_indices_view,
            inputs.predicate,
            "left_indices is null",
        ),
        (
            "right_indices",
            inputs.left_tv,
            inputs.right_tv,
            inputs.left_indices_view,
            ptr::null_mut(),
            inputs.predicate,
            "right_indices is null",
        ),
        (
            "predicate",
            inputs.left_tv,
            inputs.right_tv,
            inputs.left_indices_view,
            inputs.right_indices_view,
            ptr::null_mut(),
            "predicate is null",
        ),
    ] {
        let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(left),
                raw_handle(right),
                raw_handle(left_indices),
                raw_handle(right_indices),
                raw_handle(predicate),
                FILTER_JOIN_KIND_INNER,
                raw_out_handle(&mut out_left),
                raw_out_handle(&mut out_right),
                &exec,
            )
        };

        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "filter join null {case} status");
        assert!(out_left.is_null(), "filter join must clear left output for null {case}");
        assert!(out_right.is_null(), "filter join must clear right output for null {case}");
        assert!(last_error_message().contains(message));
    }
}

#[test]
fn test_filter_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();
    let inputs = FilterJoinTestInputs::new();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);
    let join_indices = cudf_filter_join_indices_ex_raw as FilterJoinIndicesExRaw;

    let target_before = target_mr.allocation_count();
    reset_exec_observation();

    let mut out_left: cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut out_right: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        join_indices(
            raw_handle(inputs.left_tv),
            raw_handle(inputs.right_tv),
            raw_handle(inputs.left_indices_view),
            raw_handle(inputs.right_indices_view),
            raw_handle(inputs.predicate),
            FILTER_JOIN_KIND_INNER,
            raw_out_handle(&mut out_left),
            raw_out_handle(&mut out_right),
            &exec,
        )
    };

    assert_eq!(raw_status(status), cudf_sys::CUDF_SUCCESS);
    assert!(!out_left.is_null(), "filter join did not publish left indices");
    assert!(!out_right.is_null(), "filter join did not publish right indices");
    assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "filter join allocations must be charged to the explicit execution memory resource"
    );
    assert_readable_nonempty_i32_indices(out_left, "filter join left indices");
    assert_readable_nonempty_i32_indices(out_right, "filter join right indices");

    stream.synchronize();
    let target_active_before_destroy = target_mr.active_allocation_count();
    assert!(
        target_active_before_destroy > 0,
        "filter join outputs should retain active allocations on the explicit memory resource"
    );
    unsafe {
        cudf_sys::cudf_column_destroy(out_right);
        cudf_sys::cudf_column_destroy(out_left);
    }
    assert!(
        target_mr.active_allocation_count() < target_active_before_destroy,
        "filter join output destroy must release through the explicit execution memory resource"
    );
}

#[test]
fn test_hash_join_pre_build_sizing_allocates_no_device_payload_and_uses_explicit_context() {
    let _shim_state = lock_shim_state();

    let strings = [
        std::ffi::CString::new("alpha").unwrap(),
        std::ffi::CString::new("null-payload").unwrap(),
        std::ffi::CString::new("omega").unwrap(),
    ];
    let mut string_ptrs = strings.iter().map(|value| value.as_ptr()).collect::<Vec<_>>();
    let lengths = strings.iter().map(|value| value.as_bytes().len() as i32).collect::<Vec<_>>();
    let validity = [0b0000_0101u8];
    let (input_stream, input_mr, input_exec) = strict_exec_owners();
    let mut col: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_column_create_strings_ex(
            string_ptrs.as_mut_ptr(),
            lengths.as_ptr(),
            string_ptrs.len() as i32,
            validity.as_ptr(),
            1,
            &input_exec,
            &mut col,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!col.is_null());
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);

    let target_mr = OwnedMemoryResource::new_test_stats();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let target_before = target_mr.allocation_count();
    reset_exec_observation();
    let mut reservation = 0usize;
    let status = unsafe { cudf_sys::cudf_hash_join_pre_build_reservation_size_ex(tv, 0.5, &mut reservation, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(reservation > 0);
    assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert_eq!(
        target_mr.allocation_count(),
        target_before,
        "pre-build admission sizing must not allocate a sanitized payload through explicit MR"
    );

    unsafe {
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
    drop(input_mr);
    drop(input_stream);
}

#[test]
fn test_hash_join_create_ex_uses_explicit_memory_resource_for_build_allocations() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    reset_exec_observation();

    let mut join: cudf_sys::cudf_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_create_ex(
            tv,
            cudf_sys::CUDF_NULLABLE_JOIN_YES,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            0.5,
            &mut join,
            &exec,
        )
    };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    // Device-domain model: the hash-join build constructor takes no memory
    // resource, so its retained hash table allocates through the device's
    // current resource. The explicit execution memory resource may therefore
    // see no build allocations; retained bytes are reported via the
    // pre-build sizing formula instead.
    let mut retained_bytes = 0usize;
    let status = unsafe { cudf_sys::cudf_hash_join_device_allocated_size(join, &mut retained_bytes) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(retained_bytes > 0, "hash-join build must report formula-derived retained device bytes");

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_hash_join_device_allocated_size_reports_retained_build_artifact_bytes() {
    let _shim_state = lock_shim_state();

    let col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let tbl = make_table_from_one_column(col);
    let tv = make_table_view(tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_mr = OwnedMemoryResource::new_test_stats();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let mut out = usize::MAX;
    let status = unsafe { cudf_sys::cudf_hash_join_device_allocated_size(ptr::null_mut(), &mut out) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);
    assert_eq!(out, 0);

    let status = unsafe { cudf_sys::cudf_hash_join_device_allocated_size(ptr::null_mut(), ptr::null_mut()) };
    assert_eq!(status, cudf_sys::CUDF_ERROR_NULL);

    let mut join: cudf_sys::cudf_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_create_ex(
            tv,
            cudf_sys::CUDF_NULLABLE_JOIN_YES,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            0.5,
            &mut join,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let mut bytes = 0usize;
    let status = unsafe { cudf_sys::cudf_hash_join_device_allocated_size(join, &mut bytes) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(bytes > 0, "hash-join build artifact should retain device allocations");

    unsafe {
        cudf_sys::cudf_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(tv);
        cudf_sys::cudf_table_destroy(tbl);
    }
}

#[test]
fn test_hash_join_size_apis_report_exact_output_rows() {
    let _shim_state = lock_shim_state();
    let build_col = make_i32_column(&[2, 2, 4, 6]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_mr = OwnedMemoryResource::new_test_stats();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let mut join: cudf_sys::cudf_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_create_ex(
            build_tv,
            cudf_sys::CUDF_NULLABLE_JOIN_YES,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            0.5,
            &mut join,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 4, 5]));
    let probe_tv = make_table_view(probe_tbl);

    let mut out = 0usize;
    let status = unsafe { cudf_sys::cudf_hash_join_inner_join_size_ex(join, probe_tv, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 3);

    let status = unsafe { cudf_sys::cudf_hash_join_left_join_size_ex(join, probe_tv, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 5);

    let status = unsafe { cudf_sys::cudf_hash_join_full_join_size_ex(join, probe_tv, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 6);

    let mut left_indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut right_indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_inner_join_with_output_size_ex(
            join,
            probe_tv,
            3,
            &mut left_indices,
            &mut right_indices,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(column_len(left_indices), 3);
    assert_eq!(column_len(right_indices), 3);
    unsafe {
        cudf_sys::cudf_column_destroy(left_indices);
        cudf_sys::cudf_column_destroy(right_indices);
    }

    let status = unsafe {
        cudf_sys::cudf_hash_join_left_join_with_output_size_ex(
            join,
            probe_tv,
            5,
            &mut left_indices,
            &mut right_indices,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(column_len(left_indices), 5);
    assert_eq!(column_len(right_indices), 5);
    unsafe {
        cudf_sys::cudf_column_destroy(left_indices);
        cudf_sys::cudf_column_destroy(right_indices);
    }

    let status = unsafe {
        cudf_sys::cudf_hash_join_full_join_with_output_size_ex(
            join,
            probe_tv,
            6,
            &mut left_indices,
            &mut right_indices,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(column_len(left_indices), 6);
    assert_eq!(column_len(right_indices), 6);
    unsafe {
        cudf_sys::cudf_column_destroy(left_indices);
        cudf_sys::cudf_column_destroy(right_indices);
    }

    unsafe {
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

#[test]
fn test_conditional_join_size_apis_report_exact_output_rows() {
    let _shim_state = lock_shim_state();
    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 3, 5]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2, 4]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let predicate = make_left_less_than_right_predicate();
    let stream = OwnedStream::new();
    let mr = OwnedMemoryResource::new_cuda();
    let exec = required_exec_options(&stream.borrowed(), &mr.borrowed());

    let mut out = 0usize;
    let status =
        unsafe { cudf_sys::cudf_conditional_inner_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 3);

    let status = unsafe { cudf_sys::cudf_conditional_left_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 4);

    let status =
        unsafe { cudf_sys::cudf_conditional_left_semi_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 2);

    let status =
        unsafe { cudf_sys::cudf_conditional_left_anti_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 1);

    let status =
        unsafe { cudf_sys::cudf_conditional_inner_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 3);

    let status = unsafe { cudf_sys::cudf_conditional_left_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 4);

    let status =
        unsafe { cudf_sys::cudf_conditional_left_semi_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 2);

    let status =
        unsafe { cudf_sys::cudf_conditional_left_anti_join_size_ex(left_tv, right_tv, predicate, &mut out, &exec) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(out, 1);

    let mut left_indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let mut right_indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_inner_join_indices_with_output_size_ex(
            left_tv,
            right_tv,
            predicate,
            3,
            &mut left_indices,
            &mut right_indices,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(column_len(left_indices), 3);
    assert_eq!(column_len(right_indices), 3);
    unsafe {
        cudf_sys::cudf_column_destroy(left_indices);
        cudf_sys::cudf_column_destroy(right_indices);
    }

    let status = unsafe {
        cudf_sys::cudf_conditional_left_join_indices_with_output_size_ex(
            left_tv,
            right_tv,
            predicate,
            4,
            &mut left_indices,
            &mut right_indices,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(column_len(left_indices), 4);
    assert_eq!(column_len(right_indices), 4);
    unsafe {
        cudf_sys::cudf_column_destroy(left_indices);
        cudf_sys::cudf_column_destroy(right_indices);
    }

    let mut indices: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_conditional_left_semi_join_indices_with_output_size_ex(
            left_tv,
            right_tv,
            predicate,
            2,
            &mut indices,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(column_len(indices), 2);
    unsafe { cudf_sys::cudf_column_destroy(indices) };

    let status = unsafe {
        cudf_sys::cudf_conditional_left_anti_join_indices_with_output_size_ex(
            left_tv,
            right_tv,
            predicate,
            1,
            &mut indices,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert_eq!(column_len(indices), 1);

    unsafe {
        cudf_sys::cudf_column_destroy(indices);
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_hash_join_probe_ex_uses_explicit_memory_resource_for_probe_allocations() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let build_col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let mut join: cudf_sys::cudf_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_hash_join_create_ex(
            build_tv,
            cudf_sys::CUDF_NULLABLE_JOIN_YES,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            0.5,
            &mut join,
            &exec,
        )
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_one_column(make_i32_column(&[8, 7, 42, 1, 2, 99, 3, 4]));
    let probe_tv = make_table_view(probe_tbl);

    for (name, probe) in [
        (
            "inner",
            cudf_sys::cudf_hash_join_inner_join_ex
                as unsafe extern "C" fn(
                    cudf_sys::cudf_hash_join_handle,
                    cudf_sys::cudf_table_view_handle,
                    *mut cudf_sys::cudf_column_handle,
                    *mut cudf_sys::cudf_column_handle,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_hash_join_left_join_ex),
        ("full", cudf_sys::cudf_hash_join_full_join_ex),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: cudf_sys::cudf_column_handle = ptr::null_mut();
        let mut out_right: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe { probe(join, probe_tv, &mut out_left, &mut out_right, &exec) };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} hash-join probe failed");
        assert!(!out_left.is_null(), "{name} hash-join probe did not publish left indices");
        assert!(!out_right.is_null(), "{name} hash-join probe did not publish right indices");
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} hash-join probe allocations must be charged to the explicit execution memory resource"
        );

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_view_destroy(probe_tv);
        cudf_sys::cudf_table_destroy(probe_tbl);
        cudf_sys::cudf_hash_join_destroy(join);
        cudf_sys::cudf_table_view_destroy(build_tv);
        cudf_sys::cudf_table_destroy(build_tbl);
    }
}

#[test]
fn test_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 3]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2, 2, 4]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    for (name, join_indices) in [
        (
            "inner",
            cudf_sys::cudf_inner_join_indices_ex
                as unsafe extern "C" fn(
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_null_equality_t,
                    *mut cudf_sys::cudf_column_handle,
                    *mut cudf_sys::cudf_column_handle,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_left_join_indices_ex),
        ("full", cudf_sys::cudf_full_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: cudf_sys::cudf_column_handle = ptr::null_mut();
        let mut out_right: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe {
            join_indices(left_tv, right_tv, cudf_sys::CUDF_NULL_EQUALITY_EQUAL, &mut out_left, &mut out_right, &exec)
        };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} join indices failed");
        assert!(!out_left.is_null(), "{name} join indices did not publish left indices");
        assert!(!out_right.is_null(), "{name} join indices did not publish right indices");
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} join-index allocations must be charged to the explicit execution memory resource"
        );

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_filtered_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 3, 5, 8, 13, 21]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2, 3, 13]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    for (name, join_indices) in [
        (
            "semi",
            cudf_sys::cudf_left_semi_join_indices_ex
                as unsafe extern "C" fn(
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_null_equality_t,
                    *mut cudf_sys::cudf_column_handle,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("anti", cudf_sys::cudf_left_anti_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe { join_indices(left_tv, right_tv, cudf_sys::CUDF_NULL_EQUALITY_EQUAL, &mut out, &exec) };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} filtered join indices failed");
        assert!(!out.is_null(), "{name} filtered join indices did not publish indices");
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} filtered join allocations must be charged to the explicit execution memory resource"
        );

        unsafe { cudf_sys::cudf_column_destroy(out) };
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_conditional_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_tbl = make_table_from_one_column(make_i32_column(&[1, 4]));
    let right_tbl = make_table_from_one_column(make_i32_column(&[2]));
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let predicate = make_left_less_than_right_predicate();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    for (name, join_indices) in [
        (
            "inner",
            cudf_sys::cudf_conditional_inner_join_indices_ex
                as unsafe extern "C" fn(
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_ast_expression_handle,
                    *mut cudf_sys::cudf_column_handle,
                    *mut cudf_sys::cudf_column_handle,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_conditional_left_join_indices_ex),
        ("full", cudf_sys::cudf_conditional_full_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: cudf_sys::cudf_column_handle = ptr::null_mut();
        let mut out_right: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe { join_indices(left_tv, right_tv, predicate, &mut out_left, &mut out_right, &exec) };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} conditional join failed");
        assert!(!out_left.is_null(), "{name} conditional join did not publish left indices");
        assert!(!out_right.is_null(), "{name} conditional join did not publish right indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} conditional join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out_left, &format!("{name} conditional left indices"));
        assert_readable_nonempty_i32_indices(out_right, &format!("{name} conditional right indices"));

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    for (name, join_indices) in [
        (
            "semi",
            cudf_sys::cudf_conditional_left_semi_join_indices_ex
                as unsafe extern "C" fn(
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_ast_expression_handle,
                    *mut cudf_sys::cudf_column_handle,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("anti", cudf_sys::cudf_conditional_left_anti_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe { join_indices(left_tv, right_tv, predicate, &mut out, &exec) };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} conditional join failed");
        assert!(!out.is_null(), "{name} conditional join did not publish indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} conditional join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out, &format!("{name} conditional indices"));

        unsafe { cudf_sys::cudf_column_destroy(out) };
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_destroy(right_tbl);
        cudf_sys::cudf_table_destroy(left_tbl);
    }
}

#[test]
fn test_mixed_join_invalid_null_equality_rejected_and_clears_outputs() {
    let _shim_state = lock_shim_state();
    let left_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 1, 2, 3]));
    let right_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 4]));
    let left_cond_tbl = make_table_from_one_column(make_i32_column(&[10, 30, 5, 7]));
    let right_cond_tbl = make_table_from_one_column(make_i32_column(&[20, 10, 6, 8]));
    let left_eq_tv = make_table_view(left_eq_tbl);
    let right_eq_tv = make_table_view(right_eq_tbl);
    let left_cond_tv = make_table_view(left_cond_tbl);
    let right_cond_tv = make_table_view(right_cond_tbl);
    let predicate = make_left_less_than_right_predicate();
    let (_stream, _mr, exec) = strict_exec_owners();

    for (name, join_indices) in [
        ("inner", cudf_mixed_inner_join_indices_raw as MixedPairJoinRaw),
        ("left", cudf_mixed_left_join_indices_raw),
        ("full", cudf_mixed_full_join_indices_raw),
    ] {
        let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(left_eq_tv),
                raw_handle(right_eq_tv),
                raw_handle(left_cond_tv),
                raw_handle(right_cond_tv),
                raw_handle(predicate),
                99,
                raw_out_handle(&mut out_left),
                raw_out_handle(&mut out_right),
                &exec,
            )
        };

        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION, "{name} mixed join status");
        assert!(out_left.is_null(), "{name} mixed join must clear left output");
        assert!(out_right.is_null(), "{name} mixed join must clear right output");
        assert!(last_error_message().contains("invalid null_equality"));
    }

    for (name, join_indices) in [
        ("semi", cudf_mixed_left_semi_join_indices_raw as MixedSingleJoinRaw),
        ("anti", cudf_mixed_left_anti_join_indices_raw),
    ] {
        let mut out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(left_eq_tv),
                raw_handle(right_eq_tv),
                raw_handle(left_cond_tv),
                raw_handle(right_cond_tv),
                raw_handle(predicate),
                99,
                raw_out_handle(&mut out),
                &exec,
            )
        };

        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_INVALID_OPERATION, "{name} mixed join status");
        assert!(out.is_null(), "{name} mixed join must clear output");
        assert!(last_error_message().contains("invalid null_equality"));
    }

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_cond_tv);
        cudf_sys::cudf_table_view_destroy(left_cond_tv);
        cudf_sys::cudf_table_view_destroy(right_eq_tv);
        cudf_sys::cudf_table_view_destroy(left_eq_tv);
        cudf_sys::cudf_table_destroy(right_cond_tbl);
        cudf_sys::cudf_table_destroy(left_cond_tbl);
        cudf_sys::cudf_table_destroy(right_eq_tbl);
        cudf_sys::cudf_table_destroy(left_eq_tbl);
    }
}

#[test]
fn test_mixed_join_null_handles_clear_outputs() {
    let _shim_state = lock_shim_state();
    let left_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 1, 2, 3]));
    let right_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 4]));
    let left_cond_tbl = make_table_from_one_column(make_i32_column(&[10, 30, 5, 7]));
    let right_cond_tbl = make_table_from_one_column(make_i32_column(&[20, 10, 6, 8]));
    let left_eq_tv = make_table_view(left_eq_tbl);
    let right_eq_tv = make_table_view(right_eq_tbl);
    let left_cond_tv = make_table_view(left_cond_tbl);
    let right_cond_tv = make_table_view(right_cond_tbl);
    let predicate = make_left_less_than_right_predicate();
    let null_equality = cudf_sys::CUDF_NULL_EQUALITY_EQUAL as i32;
    let (_stream, _mr, exec) = strict_exec_owners();

    for (name, join_indices) in [
        ("inner", cudf_mixed_inner_join_indices_raw as MixedPairJoinRaw),
        ("left", cudf_mixed_left_join_indices_raw),
        ("full", cudf_mixed_full_join_indices_raw),
    ] {
        let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(left_eq_tv),
                raw_handle(right_eq_tv),
                raw_handle(left_cond_tv),
                raw_handle(right_cond_tv),
                raw_handle(predicate),
                null_equality,
                ptr::null_mut(),
                raw_out_handle(&mut out_right),
                &exec,
            )
        };
        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null left output");
        assert!(last_error_message().contains("out_left_indices is null"));

        let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
        let status = unsafe {
            join_indices(
                raw_handle(left_eq_tv),
                raw_handle(right_eq_tv),
                raw_handle(left_cond_tv),
                raw_handle(right_cond_tv),
                raw_handle(predicate),
                null_equality,
                raw_out_handle(&mut out_left),
                ptr::null_mut(),
                &exec,
            )
        };
        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null right output");
        assert!(out_left.is_null(), "{name} mixed join must clear left output first");
        assert!(last_error_message().contains("out_right_indices is null"));

        for (case, left_eq, right_eq, left_cond, right_cond, pred, message) in [
            (
                "left_equality",
                ptr::null_mut(),
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                "left_equality is null",
            ),
            (
                "right_equality",
                left_eq_tv,
                ptr::null_mut(),
                left_cond_tv,
                right_cond_tv,
                predicate,
                "right_equality is null",
            ),
            (
                "left_conditional",
                left_eq_tv,
                right_eq_tv,
                ptr::null_mut(),
                right_cond_tv,
                predicate,
                "left_conditional is null",
            ),
            (
                "right_conditional",
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                ptr::null_mut(),
                predicate,
                "right_conditional is null",
            ),
            ("predicate", left_eq_tv, right_eq_tv, left_cond_tv, right_cond_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out_left = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
            let mut out_right = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
            let status = unsafe {
                join_indices(
                    raw_handle(left_eq),
                    raw_handle(right_eq),
                    raw_handle(left_cond),
                    raw_handle(right_cond),
                    raw_handle(pred),
                    null_equality,
                    raw_out_handle(&mut out_left),
                    raw_out_handle(&mut out_right),
                    &exec,
                )
            };
            assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null {case} status");
            assert!(out_left.is_null(), "{name} mixed join must clear left output for null {case}");
            assert!(out_right.is_null(), "{name} mixed join must clear right output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    for (name, join_indices) in [
        ("semi", cudf_mixed_left_semi_join_indices_raw as MixedSingleJoinRaw),
        ("anti", cudf_mixed_left_anti_join_indices_raw),
    ] {
        let status = unsafe {
            join_indices(
                raw_handle(left_eq_tv),
                raw_handle(right_eq_tv),
                raw_handle(left_cond_tv),
                raw_handle(right_cond_tv),
                raw_handle(predicate),
                null_equality,
                ptr::null_mut(),
                &exec,
            )
        };
        assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null output");
        assert!(last_error_message().contains("out_indices is null"));

        for (case, left_eq, right_eq, left_cond, right_cond, pred, message) in [
            (
                "left_equality",
                ptr::null_mut(),
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                "left_equality is null",
            ),
            (
                "right_equality",
                left_eq_tv,
                ptr::null_mut(),
                left_cond_tv,
                right_cond_tv,
                predicate,
                "right_equality is null",
            ),
            (
                "left_conditional",
                left_eq_tv,
                right_eq_tv,
                ptr::null_mut(),
                right_cond_tv,
                predicate,
                "left_conditional is null",
            ),
            (
                "right_conditional",
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                ptr::null_mut(),
                predicate,
                "right_conditional is null",
            ),
            ("predicate", left_eq_tv, right_eq_tv, left_cond_tv, right_cond_tv, ptr::null_mut(), "predicate is null"),
        ] {
            let mut out = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
            let status = unsafe {
                join_indices(
                    raw_handle(left_eq),
                    raw_handle(right_eq),
                    raw_handle(left_cond),
                    raw_handle(right_cond),
                    raw_handle(pred),
                    null_equality,
                    raw_out_handle(&mut out),
                    &exec,
                )
            };
            assert_eq!(raw_status(status), cudf_sys::CUDF_ERROR_NULL, "{name} mixed join null {case} status");
            assert!(out.is_null(), "{name} mixed join must clear output for null {case}");
            assert!(last_error_message().contains(message));
        }
    }

    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_cond_tv);
        cudf_sys::cudf_table_view_destroy(left_cond_tv);
        cudf_sys::cudf_table_view_destroy(right_eq_tv);
        cudf_sys::cudf_table_view_destroy(left_eq_tv);
        cudf_sys::cudf_table_destroy(right_cond_tbl);
        cudf_sys::cudf_table_destroy(left_cond_tbl);
        cudf_sys::cudf_table_destroy(right_eq_tbl);
        cudf_sys::cudf_table_destroy(left_eq_tbl);
    }
}

#[test]
fn test_mixed_join_indices_ex_uses_explicit_memory_resource_for_allocations_and_provenance() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let left_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 1, 2, 3]));
    let right_eq_tbl = make_table_from_one_column(make_i32_column(&[1, 2, 2, 4]));
    let left_cond_tbl = make_table_from_one_column(make_i32_column(&[10, 30, 5, 7]));
    let right_cond_tbl = make_table_from_one_column(make_i32_column(&[20, 10, 6, 8]));
    let left_eq_tv = make_table_view(left_eq_tbl);
    let right_eq_tv = make_table_view(right_eq_tbl);
    let left_cond_tv = make_table_view(left_cond_tbl);
    let right_cond_tv = make_table_view(right_cond_tbl);
    let predicate = make_left_less_than_right_predicate();
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    for (name, join_indices) in [
        (
            "inner",
            cudf_sys::cudf_mixed_inner_join_indices_ex
                as unsafe extern "C" fn(
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_ast_expression_handle,
                    cudf_sys::cudf_null_equality_t,
                    *mut cudf_sys::cudf_column_handle,
                    *mut cudf_sys::cudf_column_handle,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("left", cudf_sys::cudf_mixed_left_join_indices_ex),
        ("full", cudf_sys::cudf_mixed_full_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut out_left: cudf_sys::cudf_column_handle = ptr::null_mut();
        let mut out_right: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                &mut out_left,
                &mut out_right,
                &exec,
            )
        };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} mixed join failed");
        assert!(!out_left.is_null(), "{name} mixed join did not publish left indices");
        assert!(!out_right.is_null(), "{name} mixed join did not publish right indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} mixed join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out_left, &format!("{name} mixed left indices"));
        assert_readable_nonempty_i32_indices(out_right, &format!("{name} mixed right indices"));

        unsafe {
            cudf_sys::cudf_column_destroy(out_left);
            cudf_sys::cudf_column_destroy(out_right);
        }
    }

    for (name, join_indices) in [
        (
            "semi",
            cudf_sys::cudf_mixed_left_semi_join_indices_ex
                as unsafe extern "C" fn(
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_table_view_handle,
                    cudf_sys::cudf_ast_expression_handle,
                    cudf_sys::cudf_null_equality_t,
                    *mut cudf_sys::cudf_column_handle,
                    *const cudf_sys::cudf_exec_options_t,
                ) -> cudf_sys::cudf_status_t,
        ),
        ("anti", cudf_sys::cudf_mixed_left_anti_join_indices_ex),
    ] {
        let target_before = target_mr.allocation_count();
        reset_exec_observation();

        let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
        let status = unsafe {
            join_indices(
                left_eq_tv,
                right_eq_tv,
                left_cond_tv,
                right_cond_tv,
                predicate,
                cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
                &mut out,
                &exec,
            )
        };

        assert_eq!(status, cudf_sys::CUDF_SUCCESS, "{name} mixed join failed");
        assert!(!out.is_null(), "{name} mixed join did not publish indices");
        assert_eq!(last_exec_stream_value(), stream_value(stream.as_raw()));
        assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
        assert!(
            target_mr.allocation_count() > target_before,
            "{name} mixed join allocations must be charged to the explicit execution memory resource"
        );
        assert_readable_nonempty_i32_indices(out, &format!("{name} mixed indices"));

        unsafe { cudf_sys::cudf_column_destroy(out) };
    }

    stream.synchronize();
    unsafe {
        cudf_sys::cudf_ast_expression_destroy(predicate);
        cudf_sys::cudf_table_view_destroy(right_cond_tv);
        cudf_sys::cudf_table_view_destroy(left_cond_tv);
        cudf_sys::cudf_table_view_destroy(right_eq_tv);
        cudf_sys::cudf_table_view_destroy(left_eq_tv);
        cudf_sys::cudf_table_destroy(right_cond_tbl);
        cudf_sys::cudf_table_destroy(left_cond_tbl);
        cudf_sys::cudf_table_destroy(right_eq_tbl);
        cudf_sys::cudf_table_destroy(left_eq_tbl);
    }
}

#[test]
fn test_distinct_hash_join_probe_ex_uses_explicit_memory_resource_for_probe_allocations() {
    let _shim_state = lock_shim_state();
    let target_mr = OwnedMemoryResource::new_test_stats();

    let build_col = make_i32_column(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let mut join: cudf_sys::cudf_distinct_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_hash_join_create_ex(build_tv, cudf_sys::CUDF_NULL_EQUALITY_EQUAL, &mut join, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_one_column(make_i32_column(&[8, 7, 42, 1, 2, 99, 3, 4]));
    let probe_tv = make_table_view(probe_tbl);

    let target_before = target_mr.allocation_count();
    reset_exec_observation();

    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_distinct_hash_join_left_join_ex(join, probe_tv, &mut out, &exec) };

    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!out.is_null());
    assert_eq!(last_exec_mr_raw(), target_mr.raw_device_resource());
    assert!(
        target_mr.allocation_count() > target_before,
        "distinct hash-join probe allocations must be charged to the explicit execution memory resource"
    );

    stream.synchronize();
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
fn test_join_indices_publish_no_outputs_when_second_handle_release_fails() {
    let _shim_state = lock_shim_state();
    let left_tbl = make_table_from_columns(vec![make_i32_column(&[1, 2, 3])]);
    let right_tbl = make_table_from_columns(vec![make_i32_column(&[2, 3, 4])]);
    let left_tv = make_table_view(left_tbl);
    let right_tv = make_table_view(right_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_mr = OwnedMemoryResource::new_test_stats();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let mut left_indices = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    let mut right_indices = std::ptr::NonNull::<cudf_sys::cudf_column_handle_state>::dangling().as_ptr();
    unsafe { test_support::cudf_test_fail_column_handle_release_after(1) };

    let status = unsafe {
        cudf_sys::cudf_inner_join_indices_ex(
            left_tv,
            right_tv,
            cudf_sys::CUDF_NULL_EQUALITY_EQUAL,
            &mut left_indices,
            &mut right_indices,
            &exec,
        )
    };
    unsafe { test_support::cudf_test_fail_column_handle_release_after(-1) };

    assert_eq!(status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(left_indices.is_null());
    assert!(right_indices.is_null());

    unsafe {
        cudf_sys::cudf_table_view_destroy(left_tv);
        cudf_sys::cudf_table_view_destroy(right_tv);
        cudf_sys::cudf_table_destroy(left_tbl);
        cudf_sys::cudf_table_destroy(right_tbl);
    }
}

#[test]
fn test_distinct_hash_join_left_join_returns_build_indices() {
    let _shim_state = lock_shim_state();
    let build_col = make_i32_column(&[10, 20, 30]);
    let build_tbl = make_table_from_one_column(build_col);
    let build_tv = make_table_view(build_tbl);
    let stream = OwnedStream::new();
    let target_stream = stream.borrowed();
    let target_mr = OwnedMemoryResource::new_test_stats();
    let target_borrowed = target_mr.borrowed();
    let exec = required_exec_options(&target_stream, &target_borrowed);

    let mut join: cudf_sys::cudf_distinct_hash_join_handle = ptr::null_mut();
    let status = unsafe {
        cudf_sys::cudf_distinct_hash_join_create_ex(build_tv, cudf_sys::CUDF_NULL_EQUALITY_EQUAL, &mut join, &exec)
    };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    assert!(!join.is_null());

    let probe_tbl = make_table_from_columns(vec![make_i32_column(&[20, 99, 10, 30])]);
    let probe_tv = make_table_view(probe_tbl);
    let mut out: cudf_sys::cudf_column_handle = ptr::null_mut();

    let status = unsafe { cudf_sys::cudf_distinct_hash_join_left_join_ex(join, probe_tv, &mut out, &exec) };
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
