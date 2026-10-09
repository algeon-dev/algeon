mod arrow_host_tests;
mod conditional_full_join_size_data_tests;
mod datetime_tests;
mod execution_context_tests;
mod filtered_join_probe_tests;
#[path = "../support/ffi_helpers.rs"]
mod helpers;
mod io_and_dictionary_tests;
mod mixed_join_size_data_tests;
mod operation_validation;
mod scalar_and_column_tests;

mod stream_and_memory_resource_tests;
#[path = "../support/mod.rs"]
mod support;

mod table_tests;

mod hash_join_destroy_provenance_tests;

mod metadata_size_hooks_tests;

mod memory_resource_hooks_tests;

#[test]
fn hooks_and_metadata_cases() {
    let cases = [
        filtered_join_probe_tests::CASES,
        arrow_host_tests::CASES,
        stream_and_memory_resource_tests::CASES,
        datetime_tests::CASES,
        conditional_full_join_size_data_tests::CASES,
        hash_join_destroy_provenance_tests::CASES,
        metadata_size_hooks_tests::CASES,
        memory_resource_hooks_tests::CASES,
    ]
    .concat();
    support::run_named_cases(&cases);
}
