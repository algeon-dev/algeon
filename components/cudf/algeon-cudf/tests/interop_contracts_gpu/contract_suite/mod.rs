use cudf::testing::run_named_cases;

mod import_tests;
mod lease_cleanup_tests;
mod prepared_export_tests;
mod runtime_execution_binding_tests;
mod send_sync_contract_tests;
mod single_gpu_contract_validation_tests;

#[test]
fn prepared_lease_and_runtime_binding_cases() {
    let cases = [
        import_tests::error_mapping_contracts::CASES,
        prepared_export_tests::lease_contracts::CASES,
        lease_cleanup_tests::CASES,
        runtime_execution_binding_tests::CASES,
        single_gpu_contract_validation_tests::CASES,
    ]
    .concat();
    run_named_cases(&cases);
}
