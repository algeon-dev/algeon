#![cfg(feature = "server")]

use std::collections::BTreeMap;

use datafusion_nexus::server::{
    DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV, DATAFUSION_NEXUS_SERVER_BIND_ENV,
    DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT_ENV,
    DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
    DATAFUSION_NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, ServerConfig,
};
#[cfg(feature = "cugraph")]
use datafusion_nexus::server::{
    DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
};
use nexus_query_engine::{MemoryControlPolicy, S3SourceStorageCredential};

#[test]
fn server_config_defaults_to_single_flight_sql_admission() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
    assert_eq!(config.max_in_flight, 1);
    assert_eq!(config.statement_ticket_ttl.as_secs(), 300);
    assert_eq!(config.log_filter, "info");
    assert!(
        config.native_optimizer_config.native_execution_config().memory_control_policy().is_bounded(),
        "server production default must be bounded when no native budget env vars are supplied"
    );
    #[cfg(feature = "cugraph")]
    assert!(config.cugraph.is_none(), "cuGraph Flight SQL exposure must be disabled by default");
}

#[test]
fn server_config_rejects_zero_admission_limit() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT_ENV));
}

#[test]
fn server_config_accepts_statement_ticket_ttl() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "30".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.statement_ticket_ttl.as_secs(), 30);
}

#[test]
fn server_config_rejects_zero_statement_ticket_ttl() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV));
}

#[test]
fn server_config_requires_complete_bounded_native_profile() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV));
}

#[test]
fn server_config_accepts_bounded_native_profile() {
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.native_optimizer_config.native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        }
    );
}

#[test]
fn server_config_accepts_explicit_unbounded_native_profile() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.native_optimizer_config.native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Unbounded
    );
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_bounded_knobs() {
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
}

#[test]
fn server_config_rejects_unknown_native_memory_policy() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "best_effort".to_string())]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
}

#[cfg(not(feature = "iceberg"))]
#[test]
fn server_config_rejects_iceberg_env_without_iceberg_feature() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("--features server,iceberg"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_cugraph_env_without_cugraph_feature() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("--features server,cugraph"));
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_accepts_enabled_env() {
    use datafusion_nexus::{CugraphAlgorithm, CugraphGraphConstructionPolicy};
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, "cugraph_pagerank,cugraph_bfs".to_string()),
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, "false".to_string()),
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER_ENV, "false".to_string()),
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, "raw_libcugraph".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let cugraph = config.cugraph.expect("enabled env must materialize cuGraph server config");
    let allowed = cugraph.allowed_algorithms.expect("allowlist env must materialize an explicit set");

    assert_eq!(allowed.len(), 2);
    assert!(allowed.contains(&CugraphAlgorithm::PageRank));
    assert!(allowed.contains(&CugraphAlgorithm::Bfs));
    assert!(!cugraph.default_graph_options.directed);
    assert!(!cugraph.default_graph_options.renumber);
    assert_eq!(cugraph.default_graph_options.construction_policy, CugraphGraphConstructionPolicy::RawLibcugraph);
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_rejects_unknown_algorithm() {
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, "cugraph_pagerank,cugraph_missing".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV));
    assert!(error.to_string().contains("cugraph_missing"));
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_rejects_settings_without_enabled_true() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, "cugraph_pagerank".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV));
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn server_cugraph_session_registers_algorithm_udtfs_when_enabled() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let ctx = datafusion_nexus::server::build_session_context(&config).await.unwrap();

    assert!(
        ctx.state().table_functions().contains_key("cugraph_pagerank"),
        "enabled server config must install cuGraph SQL UDTFs into the Flight SQL SessionContext"
    );
    assert!(
        ctx.state().table_functions().contains_key("cugraph_list_algorithms"),
        "enabled server config must install cuGraph metadata UDTFs into the Flight SQL SessionContext"
    );

    let df = ctx.sql("SELECT * FROM cugraph_pagerank('g')").await.unwrap();
    let error = df.create_physical_plan().await.unwrap_err();
    assert!(
        error.to_string().contains("No table named 'g'"),
        "query should reach cuGraph edge-source catalog resolution instead of failing with a missing table function: {error}"
    );

    let batches = ctx
        .sql("SELECT function_name, enabled FROM cugraph_list_algorithms() WHERE function_name = 'cugraph_pagerank'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].num_rows(), 1, "metadata UDTF must execute in the server session");
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn server_cugraph_sql_exec_inherits_server_native_memory_policy() {
    use datafusion::physical_plan::displayable;

    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let ctx = datafusion_nexus::server::build_session_context(&config).await.unwrap();
    ctx.sql(
        "CREATE VIEW target_edges AS \
         SELECT 0::BIGINT AS src, 1::BIGINT AS dst, 1.0::DOUBLE AS weight",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let plan = ctx
        .sql("SELECT * FROM cugraph_bfs('target_edges', 0, 'src', 'dst', 'weight', '{\"depth_limit\":2}')")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let display = displayable(plan.as_ref()).indent(false).to_string();

    assert!(
        display.contains("memory_control_policy=unbounded"),
        "cuGraph SQL physical plan must inherit the server native memory policy:\n{display}"
    );
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn server_cugraph_session_does_not_register_udtfs_when_disabled() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let ctx = datafusion_nexus::server::build_session_context(&config).await.unwrap();

    assert!(!ctx.state().table_functions().contains_key("cugraph_pagerank"));
    assert!(!ctx.state().table_functions().contains_key("cugraph_list_algorithms"));
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn server_cugraph_allowlist_rejects_disallowed_algorithm_at_planning() {
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, "cugraph_pagerank".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let ctx = datafusion_nexus::server::build_session_context(&config).await.unwrap();

    let error = ctx.sql("SELECT * FROM cugraph_louvain('g')").await.unwrap_err();
    assert!(error.to_string().contains("not enabled in this session"));
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn server_cugraph_udtfs_and_views_survive_session_state_rebuild() {
    use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
    use datafusion_nexus::{CudfOptimizerConfig, CudfSessionStateBuilderExt};

    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let ctx = datafusion_nexus::server::build_session_context(&config).await.unwrap();
    ctx.sql("CREATE VIEW target_edges AS SELECT 1::BIGINT AS src, 2::BIGINT AS dst")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let rebuilt_state =
        SessionStateBuilder::new_from_existing(ctx.state()).with_cudf_native(CudfOptimizerConfig::default()).build();
    assert!(
        rebuilt_state.table_functions().contains_key("cugraph_pagerank"),
        "Flight SQL per-query state rebuild must retain registered cuGraph UDTFs"
    );
    assert!(
        rebuilt_state.table_functions().contains_key("cugraph_validate_call"),
        "Flight SQL per-query state rebuild must retain registered cuGraph metadata UDTFs"
    );
    let rebuilt = SessionContext::new_with_state(rebuilt_state);
    let batches = rebuilt.sql("SELECT src, dst FROM target_edges").await.unwrap().collect().await.unwrap();
    assert_eq!(batches[0].num_rows(), 1, "session-local view must survive the state rebuild");
}

#[test]
fn server_config_rejects_malformed_bind_address() {
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_BIND_ENV, "not-a-socket".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_BIND_ENV));
}

#[cfg(feature = "iceberg")]
fn expect_glue_backend(
    iceberg: &datafusion_nexus::server::IcebergServerConfig,
) -> &datafusion_nexus::server::GlueServerBackendConfig {
    match &iceberg.backend {
        datafusion_nexus::server::IcebergCatalogBackendConfig::Glue(glue) => glue,
        other => panic!("expected Glue backend, got {other:?}"),
    }
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_ignores_aws_env_without_catalog_request() {
    let env = BTreeMap::from([(datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert!(config.iceberg.is_none());
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_dynamic_catalog_without_explicit_tables() {
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert!(iceberg.tables.is_empty(), "dynamic catalog mode: tables must be empty when no tables env var is set");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_backend_neutral_namespace_env() {
    use datafusion_nexus::server::{
        DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV, DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, DATAFUSION_NEXUS_ICEBERG_TABLES_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_TABLES_ENV, "orders=tpch_sf1_orders,nation".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
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
        DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV, DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
        DATAFUSION_NEXUS_ICEBERG_TABLES_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        // Glue-specific catalog handle is set independently.
        (DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, "my_glue_handle".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_TABLES_ENV, "nation".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.catalog_name, "iceberg", "DataFusion connector name");
    assert_eq!(expect_glue_backend(&iceberg).catalog_handle, "my_glue_handle", "Glue catalog handle is independent");
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
        (DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_nexus::server::ServerSessionProfile::Native,
        "Iceberg presence must force the native profile so CudfTableFormatScanExec is not over-repartitioned"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_config_session_profile_env_overrides_iceberg_default() {
    use datafusion_nexus::server::DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV;
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        (DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV, "cpu_fallback".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.session_profile,
        datafusion_nexus::server::ServerSessionProfile::CpuFallback,
        "explicit DATAFUSION_NEXUS_SERVER_SESSION_PROFILE=cpu_fallback must override the Iceberg default"
    );
}

#[test]
fn server_config_session_profile_rejects_unknown_value() {
    use datafusion_nexus::server::DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV;
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV, "bogus".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV));
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
        (DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
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

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_backward_compat_glue_database_maps_to_namespace() {
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV, "demo_db".to_string()),
        (datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV, "nation".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.namespace, "demo_db", "GLUE_DATABASE must map to namespace field");
    assert_eq!(iceberg.catalog_name, "glue", "default catalog name must be glue");
    assert_eq!(expect_glue_backend(&iceberg).catalog_handle, "glue", "default glue catalog handle must be glue");
}

#[cfg(feature = "iceberg")]
fn expect_rest_backend(
    iceberg: &datafusion_nexus::server::IcebergServerConfig,
) -> &datafusion_nexus::iceberg::catalog::RestIcebergCatalogConfig {
    match &iceberg.backend {
        datafusion_nexus::server::IcebergCatalogBackendConfig::Rest(rest) => rest,
        other => panic!("expected REST backend, got {other:?}"),
    }
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_glue_explicit_selects_glue_backend() {
    use datafusion_nexus::server::{
        DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "demo_db".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        (datafusion_nexus::iceberg::AWS_ACCESS_KEY_ID_ENV, "AKIA_GLUE".to_string()),
        (datafusion_nexus::iceberg::AWS_SECRET_ACCESS_KEY_ENV, "glue-secret".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let profiles = config.native_optimizer_config.native_execution_config().source_storage_profiles();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].id(), "table_format_s3");
    assert_eq!(profiles[0].s3_profile().region.as_deref(), Some("us-east-1"));
    match &profiles[0].s3_profile().credential {
        S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token } => {
            assert_eq!(access_key_id, "AKIA_GLUE");
            assert_eq!(secret_access_key, "glue-secret");
            assert_eq!(session_token, &None);
        },
        other => panic!("expected Glue static source profile credentials, got {other:?}"),
    }
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(expect_glue_backend(&iceberg).region, "us-east-1");
    assert_eq!(iceberg.catalog_name, "glue", "default catalog name for glue kind");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_invalid_kind_is_rejected() {
    use datafusion_nexus::server::{DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV};
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, "delta".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "demo_db".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains("glue"), "error should name the accepted kinds: {error}");
    assert!(error.to_string().contains("rest"), "error should name the accepted kinds: {error}");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_happy_path() {
    use datafusion_nexus::server::{
        DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV, DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_REGION_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_S3_REGION_ENV, "us-east-1".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (datafusion_nexus::iceberg::CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV, "AKIA_REST".to_string()),
        (datafusion_nexus::iceberg::CUDF_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, "rest-secret".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
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
    assert_eq!(rest.uri, "http://localhost:8181");
    assert_eq!(rest.warehouse, "s3://lakehouse/warehouse");
    assert_eq!(rest.s3_endpoint.as_deref(), Some("http://localhost:9000"));
    assert_eq!(rest.s3_region.as_deref(), Some("us-east-1"));
    assert_eq!(rest.s3_path_style_access, Some(true));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_does_not_require_region() {
    use datafusion_nexus::server::{
        DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV,
    };
    // No AWS_REGION_ENV present — REST must still succeed.
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");
    assert_eq!(expect_rest_backend(&iceberg).uri, "http://localhost:8181");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_requires_uri() {
    use datafusion_nexus::server::{
        DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(
        error.to_string().contains(datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV),
        "error should name the missing REST URI env var: {error}"
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_requires_warehouse() {
    use datafusion_nexus::server::{
        DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV,
    };
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(
        error.to_string().contains(datafusion_nexus::server::DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV),
        "error should name the missing warehouse env var: {error}"
    );
}
