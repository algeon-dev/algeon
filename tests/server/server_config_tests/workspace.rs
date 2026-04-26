#[test]
fn server_config_accepts_workspace_overlay_defaults_and_aliases() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string()),
        (NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, "nexus_graph".to_string()),
        (NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV, "edges_by_src,edges_by_dst,profiles_by_user_id".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let workspace = config.workspace.expect("workspace overlay config");

    assert_eq!(workspace.catalog, "datafusion");
    assert_eq!(workspace.schema, "public");
    assert_eq!(workspace.backing_catalog, "glue");
    assert_eq!(workspace.backing_schema, "nexus_graph");
    assert_eq!(
        workspace.backing_aliases.unwrap(),
        vec!["edges_by_src".to_string(), "edges_by_dst".to_string(), "profiles_by_user_id".to_string()]
    );
}

#[test]
fn server_config_accepts_explicit_workspace_identity() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_WORKSPACE_CATALOG_ENV, "workspace".to_string()),
        (NEXUS_SERVER_WORKSPACE_SCHEMA_ENV, "session".to_string()),
        (NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string()),
        (NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, "nexus_graph".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let workspace = config.workspace.expect("workspace overlay config");

    assert_eq!(workspace.catalog, "workspace");
    assert_eq!(workspace.schema, "session");
    assert_eq!(workspace.backing_aliases, None);
}

#[test]
fn server_config_rejects_incomplete_workspace_backing() {
    let env = BTreeMap::from([(NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV), "{error}");
}

#[test]
fn server_config_rejects_workspace_equal_to_backing() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_WORKSPACE_CATALOG_ENV, "glue".to_string()),
        (NEXUS_SERVER_WORKSPACE_SCHEMA_ENV, "nexus_graph".to_string()),
        (NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string()),
        (NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, "nexus_graph".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("must differ from backing"), "{error}");
}

#[tokio::test]
async fn create_view_fails_when_default_schema_is_read_only_source_catalog() {
    use datafusion::prelude::{SessionConfig, SessionContext};

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_default_catalog_and_schema("glue", "nexus_graph"));
    ctx.register_catalog("glue", Arc::new(ReadOnlyCatalog::new("nexus_graph", [("edges_by_src", edge_table())])));

    let error = ctx.sql("CREATE VIEW target_edges AS SELECT src, dst FROM edges_by_src").await.unwrap_err();

    assert!(error.to_string().contains("schema provider does not support registering tables"), "{error}");
}

#[tokio::test]
async fn workspace_default_create_view_can_read_qualified_source_catalog() {
    use datafusion::prelude::{SessionConfig, SessionContext};

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_default_catalog_and_schema("datafusion", "public"));
    ctx.register_catalog("glue", Arc::new(ReadOnlyCatalog::new("nexus_graph", [("edges_by_src", edge_table())])));

    ctx.sql("CREATE VIEW target_edges AS SELECT src, dst FROM glue.nexus_graph.edges_by_src")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batches = ctx.sql("SELECT COUNT(*) FROM target_edges").await.unwrap().collect().await.unwrap();

    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batch_i64_value(&batches[0], 0), 2);
}

#[test]
fn server_config_session_profile_defaults_to_cpu_fallback_without_iceberg() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_nexus::server::ServerSessionProfile::CpuFallback,
        "no Iceberg + no explicit profile should not change DataFusion's CPU-style defaults"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_config_session_profile_defaults_to_native_when_iceberg_configured() {
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_nexus::server::ServerSessionProfile::Native,
        "Iceberg presence must force the native profile so NexusTableFormatScanExec is not over-repartitioned"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_config_session_profile_env_overrides_iceberg_default() {
    use datafusion_nexus::server::NEXUS_SERVER_SESSION_PROFILE_ENV;
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        (NEXUS_SERVER_SESSION_PROFILE_ENV, "cpu_fallback".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_nexus::server::ServerSessionProfile::CpuFallback,
        "explicit NEXUS_SERVER_SESSION_PROFILE=cpu_fallback must override the Iceberg default"
    );
}

#[test]
fn server_config_session_profile_rejects_unknown_value() {
    use datafusion_nexus::server::NEXUS_SERVER_SESSION_PROFILE_ENV;
    let env = BTreeMap::from([(NEXUS_SERVER_SESSION_PROFILE_ENV, "bogus".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_SESSION_PROFILE_ENV));
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
        datafusion_nexus::server::apply_session_profile(base, datafusion_nexus::server::ServerSessionProfile::Native);
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
    let session_config = datafusion_nexus::server::apply_session_profile(
        base,
        datafusion_nexus::server::ServerSessionProfile::CpuFallback,
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
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let session_config = datafusion_nexus::server::build_session_config(&config).unwrap();
    let options = session_config.options();

    // End-to-end check: even when DATAFUSION_EXECUTION_* process env vars are
    // unset (or set to CPU-style values), the native profile resolved from the
    // Iceberg presence must pin target_partitions=1 and disable repartitioning.
    assert_eq!(options.execution.target_partitions, 1);
    assert!(!options.optimizer.repartition_file_scans);
    assert!(!options.optimizer.repartition_joins);
    assert!(options.catalog.information_schema);
}
