use super::errors::foreign_dictionary_match_result_shape_error;
use crate::{
    core::{
        error::{Error, FACT_ACTUAL_COUNT, FACT_TABLE_COLUMN_COUNT, Result},
        native_teardown::drop_device_owned_handle,
    },
    runtime::DeviceId,
};

unsafe fn destroy_returned_column_handles(returned: &[cudf_sys::cudf_column_handle], device: DeviceId) {
    for &handle in returned {
        if !handle.is_null() {
            drop_device_owned_handle(
                handle,
                device,
                "dictionary::match_dictionaries::cleanup_malformed_result",
                |owned| unsafe { cudf_sys::cudf_column_destroy(owned) },
            );
        }
    }
}

unsafe fn destroy_returned_columns_array(
    out_columns: *mut cudf_sys::cudf_column_handle,
    out_count: i32,
    device: DeviceId,
) {
    if out_columns.is_null() {
        return;
    }

    let returned = unsafe { std::slice::from_raw_parts(out_columns, out_count.max(0) as usize) };
    unsafe { destroy_returned_column_handles(returned, device) };
    unsafe { cudf_sys::cudf_free_column_array(out_columns) };
}

fn result_count_mismatch_error(out_count: i32, expected_count: i32, operation: &'static str) -> Error {
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
    err
}

pub(super) fn validate_match_dictionaries_result_shape(
    out_columns: *mut cudf_sys::cudf_column_handle,
    out_count: i32,
    expected_count: i32,
    device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    if out_count != expected_count {
        unsafe { destroy_returned_columns_array(out_columns, out_count, device) };
        return Err(result_count_mismatch_error(out_count, expected_count, operation));
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

#[cfg(any(test, feature = "test-hooks"))]
pub(super) fn validate_owned_handles_result_shape_for_test(
    handles: &[cudf_sys::cudf_column_handle],
    expected_count: i32,
    device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    let out_count = i32::try_from(handles.len()).expect("test handle count fits i32");
    unsafe { destroy_returned_column_handles(handles, device) };
    Err(result_count_mismatch_error(out_count, expected_count, operation))
}
