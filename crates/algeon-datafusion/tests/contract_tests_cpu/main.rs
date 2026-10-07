//! CPU contract coverage: errors, policy reports, public facade.

#[cfg(not(feature = "cugraph"))]
mod cugraph_sql_feature_off_tests;
#[cfg(feature = "cugraph")]
mod cugraph_sql_option_tests;
mod cuml_sql_tests;
mod cuvs_sql_tests;
mod observability_facade_tests;

mod policy_report_tests;
