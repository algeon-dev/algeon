fn main() {
    let _ = cudf::core::column::testing_hooks::slice_view_for_test;
    let _ = cudf::core::testing_hooks::trigger_invariant_violation_for_test();
    let _ = cudf::io::arrow::testing_hooks::trigger_numeric_to_arrow_error_for_test();
    let _ = cudf::ops::dictionary::testing_hooks::trigger_match_dictionaries_result_count_mismatch_for_test();
    let _ = cudf::ops::dictionary::testing_hooks::trigger_match_dictionaries_column_count_limit_error_for_test();
    let _ = cudf::ops::sorting::testing_hooks::trigger_sort_key_count_limit_error_for_test();
    let _ = cudf::interop::arrow_device::testing_hooks::arrow_device_cpu_type_for_test();
    let _ = cudf::interop::projection::testing_hooks::trigger_negative_offset_for_test();
}
