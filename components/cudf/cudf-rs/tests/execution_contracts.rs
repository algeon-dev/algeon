#[path = "common/mod.rs"]
mod common;

#[path = "execution/core_contracts.rs"]
mod core_contracts;
#[path = "execution/frame_contracts.rs"]
mod frame_contracts;
#[path = "execution/native_teardown_contracts.rs"]
mod native_teardown_contracts;
#[path = "execution/ops_contracts.rs"]
mod ops_contracts;
#[path = "execution/query_backend_contract_tests.rs"]
mod query_backend_contract_tests;
#[path = "execution/runtime_contracts.rs"]
mod runtime_contracts;
#[path = "execution/source_contract_tests.rs"]
mod source_contract_tests;
#[path = "execution/tpch_pipeline_contract_tests.rs"]
mod tpch_pipeline_contract_tests;
