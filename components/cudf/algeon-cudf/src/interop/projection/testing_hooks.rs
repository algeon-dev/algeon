use super::{InteropError, support};

pub(crate) fn trigger_active_range_overflow_for_test() -> Result<(), InteropError> {
    Err(support::active_projection_range_overflow_error(
        "testing::interop::projection::active_range_overflow",
        usize::MAX,
        1,
    ))
}

pub(crate) fn trigger_negative_offset_for_test() -> Result<(), InteropError> {
    Err(support::negative_projection_offset_error("testing::interop::projection::negative_offset", -1))
}

pub(crate) fn trigger_missing_string_offsets_child_for_test() -> Result<(), InteropError> {
    Err(support::projection_child_count_error(
        "testing::interop::projection::missing_string_offsets_child",
        "string column view must expose an offsets child",
        "string",
        0,
        1,
        "requires_offsets_child",
        "expose an offsets child before projecting string data",
    ))
}

pub(crate) fn trigger_list_offsets_width_mismatch_for_test() -> Result<(), InteropError> {
    Err(support::projection_child_width_error(
        "testing::interop::projection::list_offsets_width_mismatch",
        "list offsets child must use 32-bit offsets",
        "offsets_layout",
        8,
        std::mem::size_of::<i32>(),
        "materialize list offsets as 32-bit signed integers before projection",
    ))
}

pub(crate) fn trigger_dictionary_indices_width_mismatch_for_test() -> Result<(), InteropError> {
    Err(support::projection_child_width_error(
        "testing::interop::projection::dictionary_indices_width_mismatch",
        "dictionary indices child must use 32-bit indices",
        "indices_layout",
        2,
        std::mem::size_of::<i32>(),
        "materialize dictionary indices as 32-bit signed integers before projection",
    ))
}

pub(crate) fn trigger_host_offsets_too_small_for_test() -> Result<(), InteropError> {
    Err(support::projection_offsets_too_small_error(
        "testing::interop::projection::host_offsets_too_small",
        "list offsets child is too small for the active logical range",
        3,
        2,
        5,
    ))
}

pub(crate) fn trigger_missing_null_mask_for_test() -> Result<(), InteropError> {
    Err(support::projection_missing_null_mask_error("testing::interop::projection::missing_null_mask", 3))
}

pub(crate) fn trigger_struct_missing_child_for_test() -> Result<(), InteropError> {
    Err(support::projection_struct_requires_child_error("testing::interop::projection::missing_struct_child"))
}
