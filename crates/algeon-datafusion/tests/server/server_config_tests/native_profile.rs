#[test]
fn server_native_config_option_descriptors_cover_toml_and_env_keys() {
    let expected = [
        ("native.execution_mode", ALGEON_SERVER_EXECUTION_MODE_ENV),
        ("native.max_source_chunk_bytes", ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV),
        ("native.max_row_groups_per_chunk", ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV),
        ("native.max_retry_attempts", ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV),
        ("native.max_source_reads_in_flight", ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV),
        ("native.source_chunk_planning_mode", ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV),
        ("native.statistical_aggregate_execution_mode", ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV),
        ("native.lakehouse_footer_pruning", ALGEON_SERVER_NATIVE_LAKEHOUSE_FOOTER_PRUNING_ENV),
    ];

    assert_eq!(ALGEON_SERVER_NATIVE_CONFIG_OPTIONS.len(), expected.len());
    for (toml_field, env_var) in expected {
        let option = ALGEON_SERVER_NATIVE_CONFIG_OPTIONS
            .iter()
            .find(|option| option.toml_field == toml_field)
            .unwrap_or_else(|| panic!("missing descriptor for {toml_field}"));
        assert_eq!(option.env_var, env_var);
    }
}

#[test]
fn server_config_rejects_removed_native_budget_and_retry_floor_env() {
    for name in ["ALGEON_SERVER_NATIVE_DEVICE_BUDGET_BYTES", "ALGEON_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS"] {
        let env = BTreeMap::from([(name, "1024".to_string())]);
        let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();
        assert_eq!(error.fact_value("config_field"), Some(name));
        assert!(error.message().contains("removed by the resource-only admission contract"), "{error}");
    }
}

#[test]
fn server_config_accepts_bounded_native_profile() {
    let env = BTreeMap::from([
        (ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let native = config.native_optimizer_config.native_execution_config();
    assert_eq!(
        config.gpu_device_profiles[0].memory_limit(),
        GpuDeviceMemoryLimit::default()
    );
    assert_eq!(native.source_chunking().max_source_chunk_bytes.get(), 512);
    assert_eq!(native.source_chunking().max_row_groups_per_chunk.get(), 8);
    assert_eq!(
        native.source_chunking().max_retry_attempts.get(),
        SourceChunking::DEFAULT_MAX_RETRY_ATTEMPTS.get()
    );
}

#[test]
fn server_config_accepts_independent_source_chunk_limits() {
    let env = BTreeMap::from([(ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let native = config.native_optimizer_config.native_execution_config();
    assert_eq!(native.source_chunking().max_source_chunk_bytes.get(), 512);
    assert_eq!(
        native.source_chunking().max_row_groups_per_chunk.get(),
        SourceChunking::DEFAULT_MAX_ROW_GROUPS_PER_CHUNK.get()
    );
}

#[test]
fn server_config_accepts_extended_native_query_knobs() {
    let env = BTreeMap::from([
        (ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
        (ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "5".to_string()),
        (ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "4".to_string()),
        (ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "local_parquet_throughput".to_string()),
        (ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, "gpu_native_tolerant".to_string()),
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
        ["ALGEON_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES", "ALGEON_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES"]
    {
        let env = BTreeMap::from([(name, "256".to_string())]);
        let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
        assert_eq!(error.operation(), "algeon_datafusion.server.config");
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
max_source_chunk_bytes = 512
max_row_groups_per_chunk = 8
max_retry_attempts = 5
"#,
    );
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert_eq!(config.queue_policy.max_queued_attempts(), 2);
    let native = config.native_optimizer_config.native_execution_config();
    assert_eq!(native.source_chunking().max_source_chunk_bytes.get(), 512);
    assert_eq!(native.source_chunking().max_row_groups_per_chunk.get(), 8);
    assert_eq!(native.source_chunking().max_retry_attempts.get(), 5);
}

#[test]
fn server_config_rejects_zero_extended_native_query_knobs() {
    for name in [
        ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    ] {
        let env = BTreeMap::from([(name, "0".to_string())]);
        let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();
        assert!(error.to_string().contains(name));
    }
}

#[test]
fn server_config_defaults_to_native_preferred() {
    let config = server_config_with_default_device_profile(BTreeMap::new()).unwrap();
    assert_eq!(config.native_optimizer_config.execution_mode(), AlgeonGpuExecutionMode::NativePreferred);
}

#[test]
fn server_config_maps_execution_mode_from_env_and_toml() {
    let env = BTreeMap::from([(ALGEON_SERVER_EXECUTION_MODE_ENV, "functions_only".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();
    assert_eq!(config.native_optimizer_config.execution_mode(), AlgeonGpuExecutionMode::FunctionsOnly);

    let (_dir, env) = env_with_server_toml("[native]\nexecution_mode = \"native_required\"\n");
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert_eq!(config.native_optimizer_config.execution_mode(), AlgeonGpuExecutionMode::NativeRequired);
}

#[test]
fn server_config_sets_footer_pruning_from_env_or_toml() {
    let config = server_config_with_default_device_profile(BTreeMap::new()).unwrap();
    assert!(!config.native_optimizer_config.native_execution_config().lakehouse_footer_pruning());

    let env = BTreeMap::from([(ALGEON_SERVER_NATIVE_LAKEHOUSE_FOOTER_PRUNING_ENV, "true".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();
    assert!(config.native_optimizer_config.native_execution_config().lakehouse_footer_pruning());

    let (_dir, env) = env_with_server_toml("[native]\nlakehouse_footer_pruning = true\n");
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert!(config.native_optimizer_config.native_execution_config().lakehouse_footer_pruning());
}

#[test]
fn server_config_builds_one_s3_storage_from_toml_and_env() {
    let config = server_config_with_default_device_profile(BTreeMap::new()).unwrap();
    assert!(config.native_optimizer_config.native_execution_config().s3_storage().is_none());

    let (_dir, mut env) = env_with_server_toml("[s3]\nendpoint = \"http://127.0.0.1:9000\"\npath_style = true\n");
    env.insert(ALGEON_SERVER_S3_REGION_ENV, "us-west-2".to_string());
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let storage = config.native_optimizer_config.native_execution_config().s3_storage().expect("S3 storage").clone();
    assert_eq!(storage.endpoint.as_deref(), Some("http://127.0.0.1:9000"));
    assert_eq!(storage.region.as_deref(), Some("us-west-2"));
    assert_eq!(storage.virtual_hosted_style_request, Some(false));
    assert_eq!(storage.credential, S3SourceStorageCredential::DefaultChain);

    let env = BTreeMap::from([
        (ALGEON_SERVER_S3_ENDPOINT_ENV, "http://127.0.0.1:9001".to_string()),
        (ALGEON_SERVER_S3_UNSIGNED_ENV, "true".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let storage = config.native_optimizer_config.native_execution_config().s3_storage().expect("S3 storage").clone();
    assert_eq!(storage.endpoint.as_deref(), Some("http://127.0.0.1:9001"));
    assert_eq!(storage.credential, S3SourceStorageCredential::Unsigned);
}

#[test]
fn server_config_rejects_removed_final_plan_requirement_env() {
    let name = "ALGEON_SERVER_FINAL_PLAN_REQUIREMENT";
    let env = BTreeMap::from([(name, "no_datafusion_cpu".to_string())]);
    let error = server_config_with_default_device_profile(env).unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    assert_eq!(error.fact_value("config_field"), Some(name));
    assert!(error.message().contains("ALGEON_SERVER_EXECUTION_MODE=native_required"), "{error}");
}

#[test]
fn server_config_rejects_malformed_remaining_native_execution_modes() {
    let aggregate_env = BTreeMap::from([(
        ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
        "bit_identical_gpu".to_string(),
    )]);
    let error = ServerConfig::from_env_get(|name| aggregate_env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains(ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV));
}
