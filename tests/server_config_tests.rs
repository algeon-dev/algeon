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

    // Error mentions the tables env var (either neutral or Glue-specific).
    let msg = error.to_string();
    assert!(
        msg.contains(datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_TABLES_ENV)
            || msg.contains(datafusion_cudf::server::DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV),
        "error should mention a tables env var: {msg}"
    );
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
