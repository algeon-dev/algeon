#![allow(dead_code)] // Explicit tools runners share parser/support modules with this Cargo contract target.

#[path = "cugraph/cugraph_contract.rs"]
mod cugraph_contract;
#[path = "cugraph/optional_fixture.rs"]
mod cugraph_optional_fixture;
#[path = "cugraph/cugraph_parity.rs"]
mod cugraph_parity;
#[path = "cugraph/cugraph_sql_tests.rs"]
mod cugraph_sql_tests;
