use datafusion_nexus::{
    engine_contracts, error_contracts, execution_contracts, prelude, report_contracts, table_format_contracts,
};

#[test]
fn adapter_facade_modules_expose_intentional_contract_groups() {
    let _rule_name: &str = prelude::CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME;
    let _error_code = error_contracts::ErrorCode::InvalidExecutionPlan;
    let _metric_names: &[&str] = execution_contracts::NATIVE_CUDF_DATAFUSION_METRIC_NAMES;
    let _fallback_header: &str = report_contracts::NATIVE_FALLBACK_PRESSURE_TSV_HEADER;
    let _table_format = table_format_contracts::TableFormat::Iceberg;
    let _mode = engine_contracts::NativePipelineExecutionMode::StrictBounded;
}

#[cfg(feature = "iceberg")]
#[test]
fn iceberg_facade_exposes_catalog_contracts() {
    let _s3_access_key_env: &str = datafusion_nexus::iceberg_contracts::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV;
}

#[cfg(feature = "cugraph")]
#[test]
fn graph_facade_exposes_sql_contracts() {
    let _bfs_function: &str = datafusion_nexus::graph_contracts::CUGRAPH_BFS_FUNCTION;
}

#[cfg(feature = "server")]
#[test]
fn server_facade_exposes_session_contracts() {
    let _profile = datafusion_nexus::server_contracts::ServerSessionProfile::Native;
}
