use std::mem::size_of;

#[test]
fn canonical_adapter_paths_expose_stable_contract_symbols() {
    let _rule_name: &str = datafusion_nexus::session::NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME;
    let _metric_names: &[&str] = datafusion_nexus::native_exec::NATIVE_NEXUS_DATAFUSION_METRIC_NAMES;
    let _headers: [&str; 4] = [
        datafusion_nexus::native_report::NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
        datafusion_nexus::native_report::NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        datafusion_nexus::native_report::NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        datafusion_nexus::native_report::NATIVE_TPCDS_PARITY_TSV_HEADER,
    ];
    let _wrappers = (
        size_of::<datafusion_nexus::native_exec::NexusNativeExec>(),
        size_of::<datafusion_nexus::native_exec::NexusNativeComposedExec>(),
        size_of::<datafusion_nexus::native_exec::NexusNativeGpuProducerExec>(),
    );
    let _planning_report = datafusion_nexus::report::PlanningReport::default();
    let _workload_contracts = (
        nexus_workloads::NEXUS_PARQUET_DIR_ENV,
        nexus_workloads::tpcds::NEXUS_TPCDS_SCALE_FACTOR_ENV,
        nexus_workloads::NEXUS_PARQUET_DIR_ENV,
        size_of::<nexus_workloads::tpch::TpchQuery>(),
        size_of::<nexus_workloads::tpcds::TpcdsQuery>(),
        size_of::<nexus_workloads::clickbench::ClickbenchQuery>(),
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn canonical_iceberg_path_exposes_catalog_contracts() {
    let _s3_access_key_env: &str = datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV;
}

#[test]
fn canonical_graph_path_exposes_sql_contracts() {
    let _bfs_function: &str = datafusion_nexus::cugraph_sql::CUGRAPH_BFS_FUNCTION;
    let _parser = datafusion_nexus::cugraph_sql::parse_cugraph_algorithm_spec;
    let _config = datafusion_nexus::cugraph_sql::CugraphSqlConfig::default();
}

#[cfg(feature = "server")]
#[test]
fn canonical_server_path_exposes_session_contracts() {
    let _auth_contracts = (
        size_of::<datafusion_nexus::server::AuthConfig>(),
        size_of::<datafusion_nexus::server::AuthInterceptor>(),
        size_of::<datafusion_nexus::server::AuthMode>(),
        size_of::<datafusion_nexus::server::PasswordHashError>(),
        size_of::<datafusion_nexus::server::Principal>(),
        size_of::<datafusion_nexus::server::StaticUserConfig>(),
        size_of::<datafusion_nexus::server::TokenStore>(),
        datafusion_nexus::server::handle_handshake,
        datafusion_nexus::server::hash_password,
        datafusion_nexus::server::verify_password,
    );
    let _diagnostic_contracts = (
        datafusion_nexus::server::FlightSqlNativeNotSupportedClass::Other,
        datafusion_nexus::server::log_native_planning_report,
        datafusion_nexus::server::log_native_planning_report_for_tables,
    );
    let _error_contracts = (
        datafusion_nexus::server::ERROR_CODE_METADATA_KEY,
        datafusion_nexus::server::ERROR_KIND_METADATA_KEY,
        datafusion_nexus::server::ERROR_STATUS_METADATA_KEY,
        datafusion_nexus::server::ERROR_OPERATION_METADATA_KEY,
        datafusion_nexus::server::UNSUPPORTED_COLUMN_METADATA_KEY,
        datafusion_nexus::server::UNSUPPORTED_COLUMN_TYPE_METADATA_KEY,
        datafusion_nexus::server::UNSUPPORTED_COLUMN_ROLE_METADATA_KEY,
        datafusion_nexus::server::datafusion_error_to_status,
        datafusion_nexus::server::datafusion_error_to_status_for_operation,
        size_of::<datafusion_nexus::server::ServerError>(),
    );
    let _result: datafusion_nexus::server::Result<()> = Ok(());
}

#[cfg(all(feature = "server", feature = "iceberg"))]
#[test]
fn canonical_server_path_exposes_iceberg_config_contracts() {
    let _contracts = (
        size_of::<datafusion_nexus::server::GlueServerBackendConfig>(),
        size_of::<datafusion_nexus::server::IcebergCatalogBackendConfig>(),
        size_of::<datafusion_nexus::server::IcebergServerConfig>(),
        size_of::<datafusion_nexus::server::IcebergTableRegistrationConfig>(),
    );
}

#[cfg(feature = "server")]
#[test]
fn canonical_server_path_exposes_cugraph_config_contract() {
    let _contract = size_of::<datafusion_nexus::server::CugraphServerConfig>();
}
