#[test]
fn server_native_config_option_descriptors_cover_toml_env_and_session_keys() {
    let expected = [
        ("native.device_budget_bytes", NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV),
        ("native.max_source_chunk_bytes", NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV),
        ("native.max_row_groups_per_chunk", NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV),
        ("native.max_retry_attempts", NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV),
        ("native.min_retry_chunk_row_groups", NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV),
        ("native.max_source_reads_in_flight", NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV),
        ("native.source_chunk_planning_mode", NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV),
        ("native.statistical_aggregate_execution_mode", NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV),
        ("native.final_plan_requirement", NEXUS_SERVER_FINAL_PLAN_REQUIREMENT_ENV),
    ];

    assert_eq!(NEXUS_SERVER_NATIVE_CONFIG_OPTIONS.len(), expected.len());
    for (toml_field, env_var) in expected {
        let option = NEXUS_SERVER_NATIVE_CONFIG_OPTIONS
            .iter()
            .find(|option| option.toml_field == toml_field)
            .unwrap_or_else(|| panic!("missing descriptor for {toml_field}"));
        assert_eq!(option.env_var, env_var);
    }

    let session_keys = DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
        .iter()
        .map(|option| option.full_key())
        .filter(|key| key.starts_with("datafusion_nexus.native."))
        .collect::<BTreeSet<_>>();
    let server_session_keys = NEXUS_SERVER_NATIVE_CONFIG_OPTIONS
        .iter()
        .filter_map(|option| option.session_key.map(str::to_string))
        .collect::<BTreeSet<_>>();
    assert_eq!(server_session_keys, session_keys);
}

#[test]
fn server_config_requires_complete_bounded_native_profile() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV));
}

#[test]
fn server_config_accepts_bounded_native_profile() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let projection = DataFusionNexusSessionConfig::from_optimizer_config(&config.native_optimizer_config);
    assert_eq!(
        config.gpu_device_profiles[0].memory_limit(),
        datafusion_nexus::backend::GpuDeviceMemoryLimit::default()
    );
    assert_eq!(projection.native_device_budget_bytes, 1024);
    assert_eq!(projection.native_max_source_chunk_bytes, 512);
    assert_eq!(projection.native_max_row_groups_per_chunk, 8);
    assert_eq!(
        projection.native_max_retry_attempts,
        datafusion_nexus::native_exec::NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS
    );
    assert_eq!(projection.native_min_retry_chunk_row_groups, 1);
}

#[test]
fn server_config_accepts_extended_native_query_knobs() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "5".to_string()),
        (NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, "2".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "4".to_string()),
        (NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "local_parquet_throughput".to_string()),
        (NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, "gpu_native_tolerant".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let native = config.native_optimizer_config.native_execution_config();
    assert_eq!(native.source_scheduler_policy().max_source_reads_in_flight(), 4);
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(native.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
}

#[test]
fn server_config_rejects_removed_legacy_cache_profile_env() {
    for name in
        ["NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES", "NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES"]
    {
        let env = BTreeMap::from([(name, "256".to_string())]);
        let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();
        assert_eq!(error.code(), datafusion_nexus::error::ErrorCode::InvalidRequest);
        assert_eq!(error.operation(), "datafusion_nexus.server.config");
        assert_eq!(error.fact_value("config_field"), Some(name));
        assert!(error.message().contains("removed by the resource-only admission contract"), "{error}");
    }
}

#[test]
fn server_config_rejects_removed_legacy_cache_profile_toml_fields() {
    for field in ["source_cache_max_retained_bytes", "graph_cache_max_retained_bytes"] {
        let toml = format!("[native]\n{field} = 256\n");
        let (_dir, env) = env_with_server_toml(&toml);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert!(error.to_string().contains(field), "{error}");
    }
}

#[test]
fn server_config_accepts_toml_native_and_resource_profiles() {
    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
max_queued_attempts = 2

[native]
device_budget_bytes = 1024
max_source_chunk_bytes = 512
max_row_groups_per_chunk = 8
max_retry_attempts = 5
min_retry_chunk_row_groups = 2
"#,
    );
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert_eq!(config.queue_policy.max_queued_attempts, 2);
    let projection = DataFusionNexusSessionConfig::from_optimizer_config(&config.native_optimizer_config);
    assert_eq!(projection.native_device_budget_bytes, 1024);
    assert_eq!(projection.native_max_source_chunk_bytes, 512);
    assert_eq!(projection.native_max_row_groups_per_chunk, 8);
    assert_eq!(projection.native_max_retry_attempts, 5);
    assert_eq!(projection.native_min_retry_chunk_row_groups, 2);
}

#[test]
fn server_config_rejects_zero_extended_native_query_knobs() {
    for name in [
        NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
    ] {
        let env = BTreeMap::from([(name, "0".to_string())]);
        let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();
        assert!(error.to_string().contains(name));
    }
}

#[test]
fn server_config_leaves_final_plan_unconstrained_by_default() {
    let config = server_config_with_default_device_profile(BTreeMap::new()).unwrap();
    assert!(config.native_optimizer_config.final_plan_requirements().is_empty());
}

#[test]
fn server_config_accepts_no_datafusion_cpu_final_plan_requirement_env() {
    let env = BTreeMap::from([(NEXUS_SERVER_FINAL_PLAN_REQUIREMENT_ENV, "no_datafusion_cpu".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();
    assert_eq!(config.native_optimizer_config.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);
}

#[test]
fn server_config_accepts_no_datafusion_cpu_final_plan_requirement_toml() {
    let (_dir, env) = env_with_server_toml("[native]\nfinal_plan_requirement = \"no_datafusion_cpu\"\n");
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert_eq!(config.native_optimizer_config.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);
}

#[test]
fn server_config_rejects_unknown_final_plan_requirement() {
    let env = BTreeMap::from([(NEXUS_SERVER_FINAL_PLAN_REQUIREMENT_ENV, "native_only".to_string())]);
    let error = server_config_with_default_device_profile(env).unwrap_err();
    assert_eq!(error.code(), datafusion_nexus::error::ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("config_field"), Some(NEXUS_SERVER_FINAL_PLAN_REQUIREMENT_ENV));
    assert!(error.message().contains("no_datafusion_cpu"), "{error}");
}

#[test]
fn server_config_rejects_malformed_remaining_native_execution_modes() {
    let aggregate_env = BTreeMap::from([(
        NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
        "bit_identical_gpu".to_string(),
    )]);
    let error = ServerConfig::from_env_get(|name| aggregate_env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV));
}
