#![cfg(feature = "server")]

use std::collections::BTreeMap;

use datafusion_cudf::server::{
    DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, DATAFUSION_CUDF_SERVER_BIND_ENV,
    DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV, DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
    ServerConfig,
};

#[test]
fn server_config_defaults_to_single_flight_sql_admission() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
    assert_eq!(config.max_in_flight, 1);
    assert_eq!(config.statement_ticket_ttl.as_secs(), 300);
    assert_eq!(config.log_filter, "info");
}

#[test]
fn server_config_rejects_zero_admission_limit() {
    let env = BTreeMap::from([(DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV));
}

#[test]
fn server_config_accepts_statement_ticket_ttl() {
    let env = BTreeMap::from([(DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "30".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.statement_ticket_ttl.as_secs(), 30);
}

#[test]
fn server_config_rejects_zero_statement_ticket_ttl() {
    let env = BTreeMap::from([(DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS_ENV));
}

#[test]
fn server_config_requires_complete_bounded_native_profile() {
    let env = BTreeMap::from([(DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV));
}

#[test]
fn server_config_accepts_bounded_native_profile() {
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
    ]);

    ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
}

#[cfg(not(feature = "iceberg"))]
#[test]
fn server_config_rejects_iceberg_env_without_iceberg_feature() {
    let env = BTreeMap::from([(DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("--features server,iceberg"));
}

#[test]
fn server_config_rejects_malformed_bind_address() {
    let env = BTreeMap::from([(DATAFUSION_CUDF_SERVER_BIND_ENV, "not-a-socket".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_CUDF_SERVER_BIND_ENV));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_ignores_aws_env_without_catalog_request() {
    let env = BTreeMap::from([(datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert!(config.iceberg.is_none());
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_dynamic_catalog_without_explicit_tables() {
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert!(iceberg.tables.is_empty(), "dynamic catalog mode: tables must be empty when no tables env var is set");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_backend_neutral_namespace_env() {
    use datafusion_cudf::server::{
        DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV, DATAFUSION_CUDF_ICEBERG_TABLES_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        (DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (DATAFUSION_CUDF_ICEBERG_TABLES_ENV, "orders=tpch_sf1_orders,nation".to_string()),
        (DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.catalog_name, "iceberg");
    // glue_catalog_handle is independent of catalog_name; defaults to "glue"
    assert_eq!(iceberg.glue_catalog_handle, "glue");
    assert_eq!(iceberg.namespace, "analytics");
    assert_eq!(iceberg.tables.len(), 2);
    assert_eq!(iceberg.tables[0].alias, "orders");
    assert_eq!(iceberg.tables[0].table, "tpch_sf1_orders");
    assert_eq!(iceberg.tables[1].alias, "nation");
    assert_eq!(iceberg.tables[1].table, "nation");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_catalog_name_does_not_affect_glue_catalog_handle() {
    use datafusion_cudf::server::{
        DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV,
        DATAFUSION_CUDF_ICEBERG_TABLES_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        // Glue-specific catalog handle is set independently.
        (DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV, "my_glue_handle".to_string()),
        (DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (DATAFUSION_CUDF_ICEBERG_TABLES_ENV, "nation".to_string()),
        (DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.catalog_name, "iceberg", "DataFusion connector name");
    assert_eq!(iceberg.glue_catalog_handle, "my_glue_handle", "Glue catalog handle is independent");
}

#[test]
fn server_config_session_profile_defaults_to_cpu_fallback_without_iceberg() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_cudf::server::ServerSessionProfile::CpuFallback,
        "no Iceberg + no explicit profile should not change DataFusion's CPU-style defaults"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_config_session_profile_defaults_to_native_when_iceberg_configured() {
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_cudf::server::ServerSessionProfile::Native,
        "Iceberg presence must force the native profile so CudfTableFormatScanExec is not over-repartitioned"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_config_session_profile_env_overrides_iceberg_default() {
    use datafusion_cudf::server::DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV;
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        (DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV, "cpu_fallback".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_cudf::server::ServerSessionProfile::CpuFallback,
        "explicit DATAFUSION_CUDF_SERVER_SESSION_PROFILE=cpu_fallback must override the Iceberg default"
    );
}

#[test]
fn server_config_session_profile_rejects_unknown_value() {
    use datafusion_cudf::server::DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV;
    let env = BTreeMap::from([(DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV, "bogus".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV));
}

#[test]
fn apply_session_profile_native_pins_all_repartition_and_pushdown_flags_off() {
    // Construct a base that simulates a CPU-style operator config
    // (target_partitions=16, all repartition/pushdown defaults left at
    // DataFusion's defaults of `true`). The native profile must override every
    // flag the audit/plan calls out, including the join-specific dynamic
    // filter flag that DataFusion's hash-join path reads directly.
    let base = datafusion::prelude::SessionConfig::new().with_target_partitions(16);
    let session_config =
        datafusion_cudf::server::apply_session_profile(base, datafusion_cudf::server::ServerSessionProfile::Native);
    let options = session_config.options();

    assert_eq!(options.execution.target_partitions, 1, "native profile must pin target_partitions=1");
    assert!(!options.optimizer.repartition_file_scans);
    assert!(!options.optimizer.repartition_aggregations);
    assert!(!options.optimizer.repartition_joins);
    assert!(!options.optimizer.repartition_sorts);
    assert!(!options.optimizer.enable_topk_dynamic_filter_pushdown);
    assert!(
        !options.optimizer.enable_join_dynamic_filter_pushdown,
        "DataFusion's hash join checks this field directly; the umbrella flag does not suppress it"
    );
    assert!(!options.optimizer.enable_aggregate_dynamic_filter_pushdown);
    assert!(!options.optimizer.enable_dynamic_filter_pushdown);
    assert!(options.catalog.information_schema, "information_schema must remain enabled");
}

#[test]
fn apply_session_profile_cpu_fallback_preserves_base_target_partitions() {
    let base = datafusion::prelude::SessionConfig::new().with_target_partitions(16);
    let session_config = datafusion_cudf::server::apply_session_profile(
        base,
        datafusion_cudf::server::ServerSessionProfile::CpuFallback,
    );
    let options = session_config.options();

    assert_eq!(
        options.execution.target_partitions, 16,
        "cpu_fallback profile must leave the base SessionConfig's target_partitions untouched"
    );
    assert!(
        options.catalog.information_schema,
        "information_schema must be enabled for both profiles so Flight SQL clients can introspect"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn build_session_config_picks_native_profile_when_iceberg_configured() {
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let session_config = datafusion_cudf::server::build_session_config(&config).unwrap();
    let options = session_config.options();

    // End-to-end check: even when DATAFUSION_EXECUTION_* process env vars are
    // unset (or set to CPU-style values), the native profile resolved from the
    // Iceberg presence must pin target_partitions=1 and disable repartitioning.
    assert_eq!(options.execution.target_partitions, 1);
    assert!(!options.optimizer.repartition_file_scans);
    assert!(!options.optimizer.repartition_joins);
    assert!(options.catalog.information_schema);
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_backward_compat_glue_database_maps_to_namespace() {
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV, "nation".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.namespace, "demo_db", "GLUE_DATABASE must map to namespace field");
    assert_eq!(iceberg.catalog_name, "glue", "default catalog name must be glue");
    assert_eq!(iceberg.glue_catalog_handle, "glue", "default glue catalog handle must be glue");
}
