use cudf::testing::run_named_cases;

#[path = "../support/common/mod.rs"]
mod common;

mod core_contracts;
mod frame_contracts;
mod native_teardown_contracts;
mod ops_contracts;
mod query_backend_contract_tests;
mod runtime_contracts;
mod source_contract_tests;
mod tpch_pipeline_contract_tests;

#[test]
fn runtime_and_operator_binding_cases() {
    let cases = [
        frame_contracts::same_provenance_tests::CASES,
        ops_contracts::datetime_execution_tests::CASES,
        ops_contracts::strings_execution_tests::CASES,
        runtime_contracts::device_guard_tests::CASES,
        runtime_contracts::device_id_tests::CASES,
        runtime_contracts::runtime_device_tests::CASES,
        runtime_contracts::runtime_contract_tests::CASES,
        tpch_pipeline_contract_tests::CASES,
    ]
    .concat();
    run_named_cases(&cases);
}
