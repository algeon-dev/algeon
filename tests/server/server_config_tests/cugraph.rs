#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_accepts_enabled_env() {
    use datafusion_nexus::graph_sql::CugraphGraphConstructionPolicy;
    let env = BTreeMap::from([
        (NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, "false".to_string()),
        (NEXUS_SERVER_CUGRAPH_RENUMBER_ENV, "false".to_string()),
        (NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, "raw_libcugraph".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let cugraph = config.cugraph.expect("enabled env must materialize cuGraph server config");

    assert!(!cugraph.default_graph_options.directed);
    assert!(!cugraph.default_graph_options.renumber);
    assert_eq!(cugraph.default_graph_options.construction_policy, CugraphGraphConstructionPolicy::RawLibcugraph);
}

#[tokio::test]
async fn server_session_exposes_gpu_coverage_table_function() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let ctx = datafusion_nexus::server::build_session_context(&config).await.unwrap();

    assert!(
        ctx.state().table_functions().contains_key(NEXUS_EXPLAIN_COVERAGE_FUNCTION),
        "server sessions must expose GPU coverage validation through the Flight SQL table-function path"
    );

    let batches = ctx
        .sql(
            "SELECT row_kind, gpu_path, coverage_json \
             FROM nexus_explain_coverage('SELECT 1') \
             WHERE row_kind = 'summary'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batch_string_value(&batches[0], 0), "summary");
    assert_eq!(batch_string_value(&batches[0], 1), "native");
    let json: serde_json::Value = serde_json::from_str(batch_string_value(&batches[0], 2)).unwrap();
    assert_eq!(json["overall"]["verdict"], "no_cpu_in_planned_path");
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_rejects_settings_without_enabled_true() {
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, "false".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_CUGRAPH_ENABLED_ENV));
}

#[cfg(feature = "cugraph")]
#[tokio::test]
async fn server_cugraph_session_registers_algorithm_udtfs_when_enabled() {
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
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
        .sql("SELECT function_name FROM cugraph_list_algorithms() WHERE function_name = 'cugraph_pagerank'")
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
        (NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
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
async fn server_cugraph_udtfs_and_views_survive_session_state_rebuild() {
    use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
    use datafusion_nexus::{planner::NexusNativeOptimizerConfig, session::NexusSessionStateBuilderExt};

    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let ctx = datafusion_nexus::server::build_session_context(&config).await.unwrap();
    ctx.sql("CREATE VIEW target_edges AS SELECT 1::BIGINT AS src, 2::BIGINT AS dst")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let rebuilt_state =
        SessionStateBuilder::new_from_existing(ctx.state()).with_nexus_native(NexusNativeOptimizerConfig::default()).build();
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
