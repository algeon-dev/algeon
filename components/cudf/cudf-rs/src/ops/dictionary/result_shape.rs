use super::errors::foreign_dictionary_match_result_shape_error;
use crate::core::error::{FACT_ACTUAL_COUNT, FACT_TABLE_COLUMN_COUNT, Result};

unsafe fn destroy_returned_columns_array(out_columns: *mut *mut std::ffi::c_void, out_count: i32) {
    if out_columns.is_null() {
        return;
    }

    let returned = unsafe { std::slice::from_raw_parts(out_columns, out_count.max(0) as usize) };
    for &handle in returned {
        if !handle.is_null() {
            unsafe { cudf_sys::cudf_column_destroy(handle) };
        }
    }
    unsafe { cudf_sys::cudf_free_column_array(out_columns) };
}

pub(super) fn validate_match_dictionaries_result_shape(
    out_columns: *mut *mut std::ffi::c_void,
    out_count: i32,
    expected_count: i32,
    operation: &'static str,
) -> Result<()> {
    if out_count != expected_count {
        unsafe { destroy_returned_columns_array(out_columns, out_count) };
        let mut err = foreign_dictionary_match_result_shape_error(
            operation,
            "ffi returned unexpected dictionary match result count",
            "result_count_mismatch",
        )
        .with_fact(FACT_TABLE_COLUMN_COUNT, expected_count as usize)
        .with_detail("result_column_count", out_count.to_string());
        if out_count >= 0 {
            err = err.with_fact(FACT_ACTUAL_COUNT, out_count as usize);
        }
        return Err(err);
    }

    if out_count > 0 && out_columns.is_null() {
        return Err(foreign_dictionary_match_result_shape_error(
            operation,
            "ffi returned null dictionary match result array with positive count",
            "null_result_array",
        )
        .with_fact(FACT_TABLE_COLUMN_COUNT, expected_count as usize)
        .with_detail("result_column_count", out_count.to_string())
        .with_fact(FACT_ACTUAL_COUNT, out_count as usize));
    }

    Ok(())
}
