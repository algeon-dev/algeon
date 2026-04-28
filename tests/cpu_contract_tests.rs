//! CPU contract coverage: errors, policy reports, public facade, table-format scan.

#[cfg(not(feature = "cugraph"))]
#[path = "cpu_contract/cugraph_sql_feature_off_tests.rs"]
mod cugraph_sql_feature_off_tests;
#[path = "cpu_contract/cuvs_sql_tests.rs"]
mod cuvs_sql_tests;
#[path = "cpu_contract/error_tests.rs"]
mod error_tests;
#[path = "cpu_contract/policy_report_tests.rs"]
mod policy_report_tests;
#[path = "cpu_contract/public_facade_tests.rs"]
mod public_facade_tests;
#[path = "cpu_contract/table_format_scan_tests.rs"]
mod table_format_scan_tests;
