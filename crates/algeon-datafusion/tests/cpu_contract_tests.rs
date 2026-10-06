//! CPU contract coverage: errors, policy reports, public facade.

#[cfg(not(feature = "cugraph"))]
#[path = "cpu_contract/cugraph_sql_feature_off_tests.rs"]
mod cugraph_sql_feature_off_tests;
#[cfg(feature = "cugraph")]
#[path = "cpu_contract/cugraph_sql_option_tests.rs"]
mod cugraph_sql_option_tests;
#[path = "cpu_contract/cuml_sql_tests.rs"]
mod cuml_sql_tests;
#[path = "cpu_contract/cuvs_sql_tests.rs"]
mod cuvs_sql_tests;
#[path = "cpu_contract/observability_facade_tests.rs"]
mod observability_facade_tests;
#[path = "cpu_contract/policy_report_tests.rs"]
mod policy_report_tests;
