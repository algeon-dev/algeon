use std::collections::HashMap;

use datafusion::prelude::SessionContext;
use datafusion_nexus::iceberg::{
    GlueIcebergCredentialSource, GlueIcebergTableConfig, load_glue_table, register_glue_table, table_provider,
};

#[test]
fn glue_config_requires_caller_owned_table_identity() {
    let config = GlueIcebergTableConfig::new(
        "prod_catalog",
        "prod_database",
        "orders",
        "s3://prod-warehouse/tables/",
        "ap-northeast-1",
    )
    .with_catalog_id("123456789012")
    .with_profile("analytics-prod");

    assert_eq!(config.catalog_name, "prod_catalog");
    assert_eq!(config.database, "prod_database");
    assert_eq!(config.table, "orders");
    assert_eq!(config.warehouse, "s3://prod-warehouse/tables/");
    assert_eq!(config.catalog_id.as_deref(), Some("123456789012"));
    assert_eq!(config.region, "ap-northeast-1");
    assert_eq!(config.profile.as_deref(), Some("analytics-prod"));
    assert_eq!(config.table_ident_display(), "prod_database.orders");
}

#[test]
fn glue_config_env_overlay_keeps_only_profile_and_region_diagnostics() {
    let env = HashMap::from([
        ("CUDF_ICEBERG_GLUE_DATABASE", "analytics_db"),
        ("CUDF_ICEBERG_GLUE_TABLE", "orders_sf10"),
        ("AWS_PROFILE", "local-dev"),
        ("AWS_REGION", "us-west-2"),
    ]);

    let config = GlueIcebergTableConfig::new(
        "glue",
        env["CUDF_ICEBERG_GLUE_DATABASE"],
        env["CUDF_ICEBERG_GLUE_TABLE"],
        "s3://warehouse/root/",
        env["AWS_REGION"],
    )
    .with_profile(env["AWS_PROFILE"]);
    let diagnostics = config.diagnostics_with_env(|name| env.get(name).map(|value| value.to_string()));

    assert_eq!(config.table_ident_display(), "analytics_db.orders_sf10");
    assert_eq!(diagnostics.region, "us-west-2");
    assert_eq!(diagnostics.profile.as_deref(), Some("local-dev"));
    assert_eq!(diagnostics.credential_source, GlueIcebergCredentialSource::AwsProfileEnv);
    assert_eq!(diagnostics.credential_source.code(), "aws_profile_env");
}

#[test]
fn glue_diagnostics_do_not_store_or_debug_print_credential_values() {
    let env = HashMap::from([
        ("AWS_ACCESS_KEY_ID", "AKIA_TEST_VALUE_SHOULD_NOT_LEAK"),
        ("AWS_SECRET_ACCESS_KEY", "SECRET_VALUE_SHOULD_NOT_LEAK"),
        ("AWS_SESSION_TOKEN", "SESSION_VALUE_SHOULD_NOT_LEAK"),
    ]);

    let config = GlueIcebergTableConfig::new("glue", "database", "table", "s3://warehouse/root/", "us-east-1");
    let diagnostics = config.diagnostics_with_env(|name| env.get(name).map(|value| value.to_string()));
    let debug = format!("{diagnostics:?}");

    assert_eq!(diagnostics.credential_source, GlueIcebergCredentialSource::StsEnv);
    assert!(!debug.contains("AKIA_TEST_VALUE_SHOULD_NOT_LEAK"));
    assert!(!debug.contains("SECRET_VALUE_SHOULD_NOT_LEAK"));
    assert!(!debug.contains("SESSION_VALUE_SHOULD_NOT_LEAK"));
}

#[test]
fn public_imports_compile_under_iceberg_feature() {
    let ctx = SessionContext::new();
    let config = GlueIcebergTableConfig::new("glue", "database", "table", "s3://warehouse/root/", "us-east-1");

    let _load_future = load_glue_table(config.clone());
    let _register_future = register_glue_table(&ctx, "table", config);
    let _provider_fn = table_provider;
}
