use crate::{core::error::Result, runtime::DeviceId};

pub(crate) fn trigger_match_dictionaries_result_count_mismatch_for_test() -> Result<()> {
    super::result_shape::validate_match_dictionaries_result_shape(
        std::ptr::null_mut(),
        1,
        2,
        DeviceId::DEFAULT,
        "dictionary::match_dictionaries",
    )
}

pub(crate) fn trigger_match_dictionaries_null_result_array_for_test() -> Result<()> {
    super::result_shape::validate_match_dictionaries_result_shape(
        std::ptr::null_mut(),
        1,
        1,
        DeviceId::DEFAULT,
        "dictionary::match_dictionaries",
    )
}

pub(crate) fn trigger_match_dictionaries_column_count_limit_error_for_test() -> Result<()> {
    Err(super::errors::dictionary_column_count_limit_error("dictionary::match_dictionaries", usize::MAX))
}

/// Exercise malformed-result cleanup after test code has armed a native
/// teardown failure through `cudf_sys::test_support`.
pub(crate) fn trigger_match_dictionaries_fatal_malformed_cleanup_for_test() -> Result<()> {
    let runtime = crate::Runtime::new()?;
    let column = runtime.column_from_slice(&[7_i32])?;
    let device = runtime.device();
    let (handle, retained_owner) = column.into_raw_parts();
    drop(retained_owner);

    super::result_shape::validate_owned_handles_result_shape_for_test(
        &[handle],
        2,
        device,
        "dictionary::match_dictionaries",
    )
}
