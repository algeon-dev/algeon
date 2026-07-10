use datafusion_nexus::{
    admission_contracts, coverage_contracts, engine_contracts, error_contracts, execution_contracts, prelude,
    report_contracts, table_format_contracts,
};

#[test]
fn adapter_facade_modules_expose_intentional_contract_groups() {
    let _rule_name: &str = prelude::CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME;
    let _error_code = error_contracts::ErrorCode::InvalidExecutionPlan;
    let _metric_names: &[&str] = execution_contracts::NATIVE_CUDF_DATAFUSION_METRIC_NAMES;
    let _fallback_header: &str = report_contracts::NATIVE_FALLBACK_PRESSURE_TSV_HEADER;
    let _mapping_row = report_contracts::NativePlanTaskMappingMetricRow {
        query_id: "q".to_string(),
        plan_node_id: None,
        compiled_task_id: None,
        operator_kind: None,
        datafusion_shape: None,
    };
    let _table_format = table_format_contracts::TableFormat::Iceberg;
    let _mode = engine_contracts::NativePipelineExecutionMode::StrictBounded;
    let _admission_config = admission_contracts::QueryServiceConfig::default();
    let _tenant = admission_contracts::QueryServiceTenantConfig::new("tenant", 1024);
    let _admission_error_code = admission_contracts::QueryErrorCode::ServiceOverloaded;
    let _coverage_sql_function: &str = coverage_contracts::NEXUS_EXPLAIN_COVERAGE_FUNCTION;
    let _coverage_verdict = coverage_contracts::QueryGpuCoverageOverall::CpuOnly;
    let _backend_builder = prelude::NexusGpuBackend::builder()
        .fallback_policy(prelude::GpuFallbackPolicy::GpuPreferred)
        .admission(prelude::NexusAdmissionPolicy::default());
    let _memory_policy = prelude::NexusMemoryPolicy::default();
    let _grant_band =
        prelude::GrantBand { min: prelude::MemoryShare::Fraction(0.125), want: prelude::MemoryShare::Fraction(0.5) };
    let _slot_guarantee = prelude::SlotGuarantee::GuaranteedMinimum;
    let _priority = prelude::Priority::Normal;
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
