#[test]
fn server_native_config_option_descriptors_cover_toml_env_and_session_keys() {
    let expected = [
        ("native.memory_policy", NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV),
        ("native.device_budget_bytes", NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV),
        ("native.max_source_chunk_bytes", NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV),
        ("native.max_row_groups_per_chunk", NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV),
        ("native.max_retry_attempts", NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV),
        ("native.min_retry_chunk_row_groups", NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV),
        ("native.max_source_reads_in_flight", NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV),
        ("native.source_chunk_planning_mode", NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV),
        ("native.source_cache_max_retained_bytes", NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV),
        ("native.graph_cache_max_retained_bytes", NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV),
        (
            "native.query_memory_domain_ceiling_multiplier",
            NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV,
        ),
        ("native.pipeline_execution_mode", NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV),
        ("native.statistical_aggregate_execution_mode", NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV),
    ];

    assert_eq!(NEXUS_SERVER_NATIVE_CONFIG_OPTIONS.len(), expected.len());
    let mut toml_fields = BTreeSet::new();
    let mut env_vars = BTreeSet::new();
    for option in NEXUS_SERVER_NATIVE_CONFIG_OPTIONS {
        assert!(toml_fields.insert(option.toml_field), "duplicate TOML field {}", option.toml_field);
        assert!(env_vars.insert(option.env_var), "duplicate env var {}", option.env_var);
        assert!(!option.value_type.is_empty(), "{} should document a value type", option.toml_field);
        assert!(!option.description.is_empty(), "{} should document a description", option.toml_field);
    }
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
fn server_native_config_option_descriptors_document_scope() {
    let bounded_fields = NEXUS_SERVER_NATIVE_CONFIG_OPTIONS
        .iter()
        .filter(|option| option.scope.is_bounded_only())
        .map(|option| option.toml_field)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        bounded_fields,
        BTreeSet::from([
            "native.device_budget_bytes",
            "native.max_source_chunk_bytes",
            "native.max_row_groups_per_chunk",
            "native.max_retry_attempts",
            "native.min_retry_chunk_row_groups",
            "native.max_source_reads_in_flight",
            "native.source_chunk_planning_mode",
            "native.source_cache_max_retained_bytes",
            "native.graph_cache_max_retained_bytes",
            "native.query_memory_domain_ceiling_multiplier",
            "native.pipeline_execution_mode",
        ])
    );

    let statistical = NEXUS_SERVER_NATIVE_CONFIG_OPTIONS
        .iter()
        .find(|option| option.toml_field == "native.statistical_aggregate_execution_mode")
        .expect("statistical aggregate descriptor");
    assert_eq!(statistical.scope, ServerNativeConfigScope::NativeExecution);
    assert!(!statistical.scope.is_bounded_only());
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

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.native_optimizer_config.native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 8,
            max_retry_attempts:
                datafusion_nexus::native_exec::NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: 1,
        }
    );
}

#[test]
fn server_config_accepts_extended_native_query_knobs() {
    let mut env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "5".to_string()),
        (NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, "2".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "4".to_string()),
        (NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "local_parquet_throughput".to_string()),
        (NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, "256".to_string()),
        (NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV, "3".to_string()),
        (NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV, "adaptive".to_string()),
        (NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, "gpu_native_tolerant".to_string()),
    ]);
    if cfg!(feature = "cugraph") {
        env.insert(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string());
        env.insert(NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV, "512".to_string());
    }

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let native = config.native_optimizer_config.native_execution_config();

    assert_eq!(
        native.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 5,
            min_retry_chunk_row_groups: 2,
        }
    );
    assert_eq!(native.source_scheduler_policy().max_source_reads_in_flight(), 4);
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(native.query_service_local_parquet_source_cache_max_retained_bytes(), Some(256));
    assert_eq!(native.query_service_query_memory_domain().unwrap().ceiling_multiplier, 3);
    #[cfg(feature = "cugraph")]
    assert_eq!(native.query_service_resident_graph_cache_max_retained_bytes(), Some(512));
    #[cfg(not(feature = "cugraph"))]
    assert_eq!(native.query_service_resident_graph_cache_max_retained_bytes(), None);
    assert_eq!(native.native_pipeline_execution_mode(), NativePipelineExecutionMode::Adaptive);
    assert_eq!(native.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
}

#[test]
fn server_config_accepts_toml_native_query_knobs() {
    let graph_cache_toml = if cfg!(feature = "cugraph") { "graph_cache_max_retained_bytes = 512\n" } else { "" };
    let toml = format!(
        r#"
[native]
device_budget_bytes = 1024
max_source_chunk_bytes = 512
max_row_groups_per_chunk = 8
max_retry_attempts = 5
min_retry_chunk_row_groups = 2
max_source_reads_in_flight = 4
source_chunk_planning_mode = "local_parquet_throughput"
source_cache_max_retained_bytes = 256
query_memory_domain_ceiling_multiplier = 3
{graph_cache_toml}
pipeline_execution_mode = "adaptive"
statistical_aggregate_execution_mode = "gpu_native_tolerant"
"#
    );
    let (_dir, mut env) = env_with_server_toml(&toml);
    if cfg!(feature = "cugraph") {
        env.insert(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string());
    }

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let native = config.native_optimizer_config.native_execution_config();

    assert_eq!(
        native.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 5,
            min_retry_chunk_row_groups: 2,
        }
    );
    assert_eq!(native.source_scheduler_policy().max_source_reads_in_flight(), 4);
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(native.query_service_local_parquet_source_cache_max_retained_bytes(), Some(256));
    assert_eq!(native.query_service_query_memory_domain().unwrap().ceiling_multiplier, 3);
    #[cfg(feature = "cugraph")]
    assert_eq!(native.query_service_resident_graph_cache_max_retained_bytes(), Some(512));
    #[cfg(not(feature = "cugraph"))]
    assert_eq!(native.query_service_resident_graph_cache_max_retained_bytes(), None);
    assert_eq!(native.native_pipeline_execution_mode(), NativePipelineExecutionMode::Adaptive);
    assert_eq!(native.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
}

#[test]
fn server_config_env_overrides_toml_native_query_knobs() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[native]
device_budget_bytes = 1024
max_source_chunk_bytes = 512
max_row_groups_per_chunk = 8
max_retry_attempts = 0
min_retry_chunk_row_groups = 2
source_chunk_planning_mode = "local_parquet_throughput"
"#,
    );
    env.insert(NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "7".to_string());
    env.insert(NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "conservative".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let native = config.native_optimizer_config.native_execution_config();

    assert_eq!(
        native.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 7,
            min_retry_chunk_row_groups: 2,
        }
    );
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::Conservative);
}

#[test]
fn server_config_env_unbounded_overrides_toml_bounded_native_profile() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[native]
device_budget_bytes = 1024
max_source_chunk_bytes = 512
max_row_groups_per_chunk = 8
max_retry_attempts = 5
min_retry_chunk_row_groups = 2
max_source_reads_in_flight = 4
source_chunk_planning_mode = "local_parquet_throughput"
pipeline_execution_mode = "adaptive"
statistical_aggregate_execution_mode = "gpu_native_tolerant"
"#,
    );
    env.insert(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let native = config.native_optimizer_config.native_execution_config();

    assert_eq!(native.memory_control_policy(), MemoryControlPolicy::Unbounded);
    assert_eq!(
        native.source_scheduler_policy().max_source_reads_in_flight(),
        nexus_query_engine::runtime::SourceSchedulerPolicy::default().max_source_reads_in_flight()
    );
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::Conservative);
    assert_eq!(native.native_pipeline_execution_mode(), NativePipelineExecutionMode::StrictBounded);
    assert_eq!(native.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
}

#[test]
fn server_config_env_completes_toml_bounded_native_profile() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[native]
device_budget_bytes = 1024
max_source_chunk_bytes = 512
"#,
    );
    env.insert(NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(
        config.native_optimizer_config.native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 512,
            max_row_groups_per_chunk: 8,
            max_retry_attempts:
                datafusion_nexus::native_exec::NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: 1,
        }
    );
}

#[test]
fn server_config_accepts_retry_knobs_with_auto_sized_bounded_profile() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "7".to_string()),
        (NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, "3".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        config.native_optimizer_config.native_execution_config().memory_control_policy()
    else {
        panic!("server default must remain bounded");
    };
    assert_eq!(max_retry_attempts, 7);
    assert_eq!(min_retry_chunk_row_groups, 3);
}

#[test]
fn server_config_accepts_explicit_unbounded_native_profile() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 1);
    assert_eq!(
        config.native_optimizer_config.native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Unbounded
    );
}

#[test]
fn server_config_accepts_neutral_admission_defaults_with_unbounded_native_profile() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "balanced".to_string()),
        (NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "least_loaded".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 1);
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::LeastLoaded);
    assert_eq!(
        config.native_optimizer_config.native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Unbounded
    );
}

#[test]
fn server_config_accepts_neutral_toml_admission_defaults_with_unbounded_native_profile() {
    let (_dir, env) = env_with_server_toml(
        r#"
[native]
memory_policy = "unbounded"

[admission]
workload_profile = "balanced"
placement_policy = "least_loaded"
"#,
    );

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 1);
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::LeastLoaded);
}

#[test]
fn server_config_neutral_env_admission_values_shadow_non_neutral_toml_values() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[admission]
workload_profile = "small_query"
placement_policy = "best_fit"
"#,
    );
    env.insert(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "balanced".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "least_loaded".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 1);
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::LeastLoaded);
}

#[test]
fn server_config_rejects_concurrent_unbounded_native_profile() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "2".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV));
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_admission_memory_env() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, "0.125".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV));
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_admission_workload_profile() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "small_query".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV));
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_toml_admission_memory_knobs() {
    let (_dir, env) = env_with_server_toml(
        r#"
[native]
memory_policy = "unbounded"

[admission]
slot_guarantee = "optimistic"
"#,
    );

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("admission.slot_guarantee"));
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_placement_knob() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "best_fit".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV));
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_bounded_knobs() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_bounded_scheduler_knobs() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, "2".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_cache_knobs() {
    for cache_env in [
        NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV,
        NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
    ] {
        let env = BTreeMap::from([
            (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
            (cache_env, "1024".to_string()),
        ]);

        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

        assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV), "{cache_env}: {error}");
    }
}

#[test]
fn server_config_rejects_unbounded_native_profile_with_bounded_retry_or_pipeline_knobs() {
    for (name, value) in [
        (NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "2"),
        (NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, "2"),
        (NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV, "adaptive"),
    ] {
        let env = BTreeMap::from([
            (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
            (name, value.to_string()),
        ]);

        let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();

        assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
    }
}

#[test]
fn server_config_rejects_toml_unbounded_native_profile_with_bounded_knobs() {
    let (_dir, env) = env_with_server_toml(
        r#"
[native]
memory_policy = "unbounded"
max_source_reads_in_flight = 2
"#,
    );

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("native.memory_policy"));
}

#[test]
fn server_config_rejects_partial_toml_bounded_native_profile() {
    let (_dir, env) = env_with_server_toml(
        r#"
[native]
device_budget_bytes = 1024
"#,
    );

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("native.device_budget_bytes"));
}

#[test]
fn server_config_rejects_unknown_native_memory_policy() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "best_effort".to_string())]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
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
fn server_config_rejects_zero_cache_max_retained_bytes() {
    for cache_env in [
        NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV,
        NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
    ] {
        let env = BTreeMap::from([(cache_env, "0".to_string())]);

        let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();

        assert!(error.to_string().contains(cache_env), "{cache_env}: {error}");
    }
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_graph_cache_without_cugraph_feature() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV, "1024".to_string())]);

    let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV));
}

#[cfg(feature = "cugraph")]
#[test]
fn server_config_rejects_graph_cache_when_cugraph_sql_is_disabled() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV, "1024".to_string())]);

    let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV));
    assert!(error.to_string().contains(NEXUS_SERVER_CUGRAPH_ENABLED_ENV));
}

#[cfg(feature = "cugraph")]
#[test]
fn server_config_rejects_toml_graph_cache_when_cugraph_sql_is_disabled() {
    let (_dir, env) = env_with_server_toml(
        r#"
[native]
graph_cache_max_retained_bytes = 1024
"#,
    );

    let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();

    assert!(error.to_string().contains("native.graph_cache_max_retained_bytes"));
    assert!(error.to_string().contains(NEXUS_SERVER_CUGRAPH_ENABLED_ENV));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_toml_graph_cache_without_cugraph_feature() {
    let (_dir, env) = env_with_server_toml(
        r#"
[native]
graph_cache_max_retained_bytes = 1024
"#,
    );

    let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();

    assert!(error.to_string().contains("native.graph_cache_max_retained_bytes"));
}

#[test]
fn server_config_rejects_zero_toml_native_query_knobs() {
    for (toml_body, field) in [
        (
            r#"
[native]
max_source_reads_in_flight = 0
"#,
            "native.max_source_reads_in_flight",
        ),
        (
            r#"
[native]
source_cache_max_retained_bytes = 0
"#,
            "native.source_cache_max_retained_bytes",
        ),
        (
            r#"
[native]
graph_cache_max_retained_bytes = 0
"#,
            "native.graph_cache_max_retained_bytes",
        ),
        (
            r#"
[native]
query_memory_domain_ceiling_multiplier = 0
"#,
            "native.query_memory_domain_ceiling_multiplier",
        ),
    ] {
        let (_dir, env) = env_with_server_toml(toml_body);

        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

        assert!(error.to_string().contains(field));
    }
}

#[test]
fn server_config_rejects_malformed_native_execution_modes() {
    let pipeline_env = BTreeMap::from([(NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV, "sometimes".to_string())]);
    let pipeline_error = ServerConfig::from_env_get(|name| pipeline_env.get(name).cloned()).unwrap_err();
    assert!(pipeline_error.to_string().contains(NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV));

    let aggregate_env = BTreeMap::from([(
        NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
        "bit_identical_gpu".to_string(),
    )]);
    let aggregate_error = ServerConfig::from_env_get(|name| aggregate_env.get(name).cloned()).unwrap_err();
    assert!(aggregate_error.to_string().contains(NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV));
}

#[test]
fn server_config_rejects_undocumented_native_execution_mode_aliases() {
    let pipeline_env = BTreeMap::from([(NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV, "strict".to_string())]);
    let pipeline_error = ServerConfig::from_env_get(|name| pipeline_env.get(name).cloned()).unwrap_err();
    assert!(pipeline_error.to_string().contains(NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV));

    for alias in ["exact", "gpu_native"] {
        let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, alias.to_string())]);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV));
    }
}

#[test]
fn server_config_rejects_malformed_source_chunk_planning_mode() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "throughput".to_string())]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV));
}

#[test]
fn server_config_rejects_malformed_toml_native_execution_modes() {
    for (toml_body, field) in [
        (
            r#"
[native]
source_chunk_planning_mode = "throughput"
"#,
            "native.source_chunk_planning_mode",
        ),
        (
            r#"
[native]
pipeline_execution_mode = "strict"
"#,
            "native.pipeline_execution_mode",
        ),
        (
            r#"
[native]
statistical_aggregate_execution_mode = "gpu_native"
"#,
            "native.statistical_aggregate_execution_mode",
        ),
    ] {
        let (_dir, env) = env_with_server_toml(toml_body);

        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

        assert!(error.to_string().contains(field), "expected {field} in error: {error}");
    }
}
