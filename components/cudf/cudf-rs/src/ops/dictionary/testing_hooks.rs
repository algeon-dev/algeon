use crate::core::error::Result;

pub(crate) fn trigger_match_dictionaries_result_count_mismatch_for_test() -> Result<()> {
    super::result_shape::validate_match_dictionaries_result_shape(
        std::ptr::null_mut(),
        1,
        2,
        crate::runtime::DeviceId::DEFAULT,
        "dictionary::match_dictionaries",
    )
}

pub(crate) fn trigger_match_dictionaries_null_result_array_for_test() -> Result<()> {
    super::result_shape::validate_match_dictionaries_result_shape(
        std::ptr::null_mut(),
        1,
        1,
        crate::runtime::DeviceId::DEFAULT,
        "dictionary::match_dictionaries",
    )
}

pub(crate) fn trigger_match_dictionaries_column_count_limit_error_for_test() -> Result<()> {
    Err(super::errors::dictionary_column_count_limit_error("dictionary::match_dictionaries", usize::MAX))
}

pub(crate) fn trigger_match_dictionaries_fatal_malformed_cleanup_for_test() -> Result<()> {
    let runtime = crate::Runtime::new()?;
    let column = runtime.column_from_slice(&[7_i32])?;
    let device = runtime.device();
    let (handle, retained_owner) = column.into_raw_parts();
    drop(retained_owner);

    // SAFETY: the one-shot native hook consumes only scalar thread-local test
    // state. The raw column ownership was transferred above and is consumed
    // exactly once by the malformed-result cleanup below.
    unsafe { cudf_sys::cudf_test_fail_next_provenance_destroy(cudf_sys::CUDF_ERROR_FATAL_CUDA) };
    super::result_shape::validate_owned_handles_result_shape_for_test(
        &[handle],
        2,
        device,
        "dictionary::match_dictionaries",
    )
}
