use crate::core::error::Result;

pub(crate) fn trigger_match_dictionaries_result_count_mismatch_for_test() -> Result<()> {
    super::result_shape::validate_match_dictionaries_result_shape(
        std::ptr::null_mut(),
        1,
        2,
        "dictionary::match_dictionaries",
    )
}

pub(crate) fn trigger_match_dictionaries_null_result_array_for_test() -> Result<()> {
    super::result_shape::validate_match_dictionaries_result_shape(
        std::ptr::null_mut(),
        1,
        1,
        "dictionary::match_dictionaries",
    )
}

pub(crate) fn trigger_match_dictionaries_column_count_limit_error_for_test() -> Result<()> {
    Err(super::errors::dictionary_column_count_limit_error("dictionary::match_dictionaries", usize::MAX))
}
