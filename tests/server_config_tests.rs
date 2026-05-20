#![cfg(feature = "server")]

use std::collections::BTreeMap;

use datafusion_cudf::server::{
    DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, DATAFUSION_CUDF_SERVER_BIND_ENV,
    DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV, DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, ServerConfig,
};

#[test]
fn server_config_defaults_to_single_flight_sql_admission() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
    assert_eq!(config.max_in_flight, 1);
    assert_eq!(config.log_filter, "info");
}

#[test]
fn server_config_rejects_zero_admission_limit() {
    let env = BTreeMap::from([(DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV));
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
fn server_iceberg_config_requires_tables_when_catalog_requested() {
    let env = BTreeMap::from([
        (DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_cudf::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV));
}
