#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_ignores_aws_env_without_catalog_request() {
    let env = BTreeMap::from([(datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();

    assert!(config.iceberg.is_none());
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_dynamic_catalog_without_explicit_tables() {
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert!(iceberg.tables.is_empty(), "dynamic catalog mode: tables must be empty when no tables env var is set");
    assert_eq!(iceberg.scan_planning_timeout.as_secs(), 60);
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_scan_planning_timeout() {
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV, "7".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.scan_planning_timeout.as_secs(), 7);
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_rejects_zero_scan_planning_timeout() {
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV, "0".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains(datafusion_nexus::server::NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_backend_neutral_namespace_env() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_NAME_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_TABLES_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (NEXUS_ICEBERG_TABLES_ENV, "orders=tpch_sf1_orders,nation".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.catalog_name, "iceberg");
    // Glue catalog handle is independent of catalog_name; defaults to "glue"
    assert_eq!(expect_glue_backend(&iceberg).catalog_handle, "glue");
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
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_TABLES_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        // Glue-specific catalog handle is set independently.
        (NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, "my_glue_handle".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (NEXUS_ICEBERG_TABLES_ENV, "nation".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.catalog_name, "iceberg", "DataFusion connector name");
    assert_eq!(expect_glue_backend(&iceberg).catalog_handle, "my_glue_handle", "Glue catalog handle is independent");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_glue_explicit_selects_glue_backend() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-2".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV, "AKIA_GLUE".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV, "glue-secret".to_string()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, "AKIA_LOCAL".to_string()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, "local-secret".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let profiles = config.native_optimizer_config.native_execution_config().source_storage_profiles();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].id(), "table_format_s3");
    assert_eq!(profiles[0].s3_profile().region.as_deref(), Some("us-east-2"));
    assert_eq!(profiles[0].s3_profile().endpoint.as_deref(), Some("http://localhost:9000"));
    assert_eq!(profiles[0].s3_profile().virtual_hosted_style_request, Some(false));
    match &profiles[0].s3_profile().credential {
        S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token } => {
            assert_eq!(access_key_id, "AKIA_LOCAL");
            assert_eq!(secret_access_key, "local-secret");
            assert_eq!(session_token, &None);
        },
        other => panic!("expected Glue static source profile credentials, got {other:?}"),
    }
    let iceberg = config.iceberg.expect("iceberg config must be present");
    let glue = expect_glue_backend(&iceberg);
    let source_access = iceberg.source_access_policy().s3_access();

    assert_eq!(glue.region, "us-east-1");
    assert_eq!(glue.uri.as_deref(), Some("http://localhost:5001"));
    assert_eq!(glue.s3_endpoint.as_deref(), Some("http://localhost:9000"));
    assert_eq!(glue.s3_region.as_deref(), Some("us-east-2"));
    assert_eq!(glue.s3_path_style_access, Some(true));
    assert_eq!(source_access.endpoint, Some("http://localhost:9000"));
    assert_eq!(source_access.region, Some("us-east-2"));
    assert_eq!(source_access.path_style_access, Some(true));
    assert_eq!(format!("{:?}", source_access.credential.mode()), "ExplicitStatic");
    assert!(!format!("{:?}", iceberg.source_access_policy()).contains("local-secret"));
    assert_eq!(iceberg.catalog_name, "glue", "default catalog name for glue kind");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_s3_credential_source_default_chain_keeps_default_chain_with_endpoint() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-1".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "default_chain".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        // Ambient AWS env credentials must flow through the provider-side
        // default chain, not be converted into static profile credentials.
        (datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV, "AKIA_AMBIENT".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV, "ambient-secret".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let profiles = config.native_optimizer_config.native_execution_config().source_storage_profiles();
    assert_eq!(profiles.len(), 1);
    let s3_profile = profiles[0].s3_profile();
    assert_eq!(s3_profile.endpoint.as_deref(), Some("http://localhost:9000"));
    assert_eq!(s3_profile.virtual_hosted_style_request, Some(false));
    assert!(
        matches!(s3_profile.credential, S3SourceStorageCredential::DefaultChain),
        "explicit default_chain opt-in must keep the default-chain credential with a custom endpoint, got {:?}",
        s3_profile.credential
    );
    let iceberg = config.iceberg.expect("iceberg config must be present");
    assert_eq!(format!("{:?}", iceberg.source_access_policy().s3_access().credential.mode()), "DefaultChain",);
    assert_eq!(
        iceberg.parquet_read_strategy_config.credential_mode,
        nexus_query_engine::source::CredentialMode::ScopedResolved
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_custom_endpoint_freezes_unsigned_source_policy_and_native_profile() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-1".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        // A custom endpoint must not silently use these ambient credentials.
        (datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV, "AKIA_AMBIENT".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV, "ambient-secret".to_string()),
    ]);

    let config = server_config_with_default_device_profile(env).unwrap();
    let iceberg = config.iceberg.as_ref().expect("iceberg config must be present");
    let source_access = iceberg.source_access_policy().s3_access();
    assert_eq!(source_access.endpoint, Some("http://localhost:9000"));
    assert_eq!(source_access.region, Some("us-east-1"));
    assert_eq!(source_access.path_style_access, Some(true));
    assert_eq!(format!("{:?}", source_access.credential.mode()), "Unsigned");

    let profile = config.native_optimizer_config.native_execution_config().source_storage_profiles();
    assert_eq!(profile.len(), 1);
    let native = profile[0].s3_profile();
    assert_eq!(native.endpoint.as_deref(), source_access.endpoint);
    assert_eq!(native.region.as_deref(), source_access.region);
    assert_eq!(native.virtual_hosted_style_request, Some(false));
    assert!(matches!(native.credential, S3SourceStorageCredential::Unsigned));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_s3_credential_source_default_chain_applies_to_rest_backend() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "default_chain".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let profiles = config.native_optimizer_config.native_execution_config().source_storage_profiles();
    assert_eq!(profiles.len(), 1);
    let s3_profile = profiles[0].s3_profile();
    assert_eq!(s3_profile.endpoint.as_deref(), Some("http://localhost:9000"));
    assert!(
        matches!(s3_profile.credential, S3SourceStorageCredential::DefaultChain),
        "REST default_chain opt-in must keep the default-chain credential with a custom endpoint, got {:?}",
        s3_profile.credential
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_s3_credential_source_rejects_unknown_value() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "imds".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();
    assert!(error.to_string().contains(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV), "{error}");
    assert!(error.to_string().contains("default_chain"), "error should name the accepted value: {error}");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_s3_credential_source_rejects_static_credential_combination() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "default_chain".to_string()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, "AKIA_LOCAL".to_string()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, "local-secret".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();
    assert!(error.to_string().contains(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV), "{error}");
    assert!(
        error.to_string().contains("NEXUS_ICEBERG_S3_ACCESS_KEY_ID"),
        "error should name the conflicting static credentials: {error}"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_glue_config_debug_redacts_local_endpoint_urls() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "https://user:glue-secret@glue.local".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "https://user:s3-secret@s3.local".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();

    let rendered = format!("{config:?}");

    assert!(!rendered.contains("glue-secret"), "Glue endpoint credentials must be redacted: {rendered}");
    assert!(!rendered.contains("glue.local"), "Glue endpoint host must be redacted: {rendered}");
    assert!(!rendered.contains("s3-secret"), "S3 endpoint credentials must be redacted: {rendered}");
    assert!(!rendered.contains("s3.local"), "S3 endpoint host must be redacted: {rendered}");
    assert!(rendered.contains("<redacted>"), "redaction marker should be visible: {rendered}");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_invalid_kind_is_rejected() {
    use datafusion_nexus::server::{NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV};
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "delta".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();
    assert!(error.to_string().contains("glue"), "error should name the accepted kinds: {error}");
    assert!(error.to_string().contains("rest"), "error should name the accepted kinds: {error}");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_happy_path() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-1".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, "AKIA_REST".to_string()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, "rest-secret".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let profiles = config.native_optimizer_config.native_execution_config().source_storage_profiles();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].id(), "table_format_s3");
    let s3_profile = profiles[0].s3_profile();
    assert_eq!(s3_profile.region.as_deref(), Some("us-east-1"));
    assert_eq!(s3_profile.endpoint.as_deref(), Some("http://localhost:9000"));
    assert_eq!(s3_profile.virtual_hosted_style_request, Some(false));
    match &s3_profile.credential {
        S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token } => {
            assert_eq!(access_key_id, "AKIA_REST");
            assert_eq!(secret_access_key, "rest-secret");
            assert_eq!(session_token, &None);
        },
        other => panic!("expected REST static source profile credentials, got {other:?}"),
    }
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.namespace, "tpch_sf1");
    // REST kind defaults the DataFusion catalog name to "rest".
    assert_eq!(iceberg.catalog_name, "rest");
    let rest = expect_rest_backend(&iceberg);
    assert_eq!(rest.uri(), "http://localhost:8181");
    assert_eq!(rest.warehouse, "s3://lakehouse/warehouse");
    let source_access = iceberg.source_access_policy().s3_access();
    assert_eq!(source_access.endpoint, Some("http://localhost:9000"));
    assert_eq!(source_access.region, Some("us-east-1"));
    assert_eq!(source_access.path_style_access, Some(true));
    assert_eq!(format!("{:?}", source_access.credential.mode()), "ExplicitStatic");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_owns_table_format_parquet_strategy() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://user:secret@localhost:9000/rustfs?sig=secret".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-west-2".to_string()),
        (table_format::NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, "/tmp/nexus-cache".to_string()),
        (table_format::NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, "read_through".to_string()),
        (table_format::NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, "4096".to_string()),
    ]);

    let config = server_config_with_default_device_profile(env).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");
    let strategy = iceberg.parquet_read_strategy_config;

    assert_eq!(strategy.lakehouse_cache_root, Some(PathBuf::from("/tmp/nexus-cache")));
    assert_eq!(strategy.lakehouse_cache_policy, TableFormatLakehouseCachePolicy::ReadThrough);
    assert_eq!(strategy.lakehouse_cache_max_bytes, Some(4096));
    assert_eq!(strategy.storage_profile_id.as_deref(), Some("table_format_s3"));
    assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000/rustfs"));
    assert_eq!(strategy.credential_mode, nexus_query_engine::source::CredentialMode::ScopedResolved);
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_rejects_unknown_lakehouse_cache_policy() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (table_format::NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, "eventual".to_string()),
    ]);

    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains(table_format::NEXUS_LAKEHOUSE_CACHE_POLICY_ENV));
    assert!(error.to_string().contains("fill_before_read"));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_rejects_invalid_table_format_strategy_size() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let required = [
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ];
    let zero_size = required
        .iter()
        .cloned()
        .chain([(table_format::NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, "0".to_string())])
        .collect::<BTreeMap<_, _>>();
    let zero_size_error = server_config_with_default_device_profile(zero_size).unwrap_err();
    assert!(zero_size_error.to_string().contains(table_format::NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_does_not_require_region() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    // No AWS_REGION_ENV present — REST must still succeed.
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");
    assert_eq!(expect_rest_backend(&iceberg).uri(), "http://localhost:8181");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_requires_uri() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();
    assert!(
        error.to_string().contains(datafusion_nexus::server::NEXUS_ICEBERG_REST_URI_ENV),
        "error should name the missing REST URI env var: {error}"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_requires_warehouse() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_REST_URI_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();
    assert!(
        error.to_string().contains(datafusion_nexus::server::NEXUS_ICEBERG_WAREHOUSE_ENV),
        "error should name the missing warehouse env var: {error}"
    );
}
