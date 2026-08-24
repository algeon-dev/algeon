#[path = "cugraph/cugraph_contract.rs"]
mod cugraph_contract;
#[path = "cugraph/optional_fixture.rs"]
mod cugraph_optional_fixture;
#[cfg(feature = "bench-support")]
#[path = "cugraph/cugraph_parity.rs"]
mod cugraph_parity;
#[path = "cugraph/cugraph_sql_tests.rs"]
mod cugraph_sql_tests;
