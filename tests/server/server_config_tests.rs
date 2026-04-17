#![cfg(feature = "server")]

#[cfg(feature = "iceberg")]
use std::path::PathBuf;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use datafusion::common::config::ConfigOptions;
#[cfg(feature = "cugraph")]
use datafusion_nexus::server::{
    NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
};
use datafusion_nexus::{
    CudfOptimizerConfig, DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS, DataFusionNexusSessionConfig, MemoryShare,
    NEXUS_EXPLAIN_COVERAGE_FUNCTION, NativeCudfExecutionConfig, SlotGuarantee,
    server::{
        NEXUS_ICEBERG_GLUE_DATABASE_ENV, NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV,
        NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV, NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV,
        NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE_ENV,
        NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV, NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, NEXUS_SERVER_BIND_ENV,
        NEXUS_SERVER_CONFIG_FILE_ENV, NEXUS_SERVER_CPU_FALLBACK_MAX_CONCURRENT_STATEMENTS_ENV,
        NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS_ENV, NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
        NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV, NEXUS_SERVER_GPU_DEVICES_ENV, NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV,
        NEXUS_SERVER_MAX_IN_FLIGHT_ENV, NEXUS_SERVER_NATIVE_CONFIG_OPTIONS,
        NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
        NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV, NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
        NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV,
        NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV,
        NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
        NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV, NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, NEXUS_SERVER_WORKSPACE_CATALOG_ENV,
        NEXUS_SERVER_WORKSPACE_SCHEMA_ENV, ServerConfig, ServerNativeConfigScope,
    },
};
#[cfg(feature = "iceberg")]
use datafusion_nexus::{TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyPreference};
#[cfg(feature = "iceberg")]
use nexus_query_engine::S3SourceStorageCredential;
use nexus_query_engine::{
    MemoryControlPolicy, NativePipelineExecutionMode, QueryPlacementPolicy, QueryService, QueryServiceConfig,
    QueryServiceTenantConfig, SourceChunkPlanningMode, StatisticalAggregateExecutionMode,
};

#[test]
fn server_config_defaults_to_two_native_query_admissions() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
    assert_eq!(config.max_in_flight, 2);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 2);
    assert_eq!(config.cpu_fallback_max_concurrent_statements.map(|value| value.get()), Some(2));
    assert_eq!(
        config.native_optimizer_config.cpu_fallback_max_concurrent_statements().map(|value| value.get()),
        Some(2)
    );
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Fraction(1.0 / 8.0));
    assert_eq!(config.admission_policy.grant().want, MemoryShare::PolicyDefault);
    assert_eq!(config.admission_policy.max_queries_per_device(), None);
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::LeastLoaded);
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::GuaranteedMinimum);
    assert_eq!(config.admission_policy.wait_timeout(), Duration::from_secs(60));
    assert_eq!(config.cpu_fallback_wait_timeout, Some(Duration::from_secs(30)));
    assert_eq!(config.native_optimizer_config.cpu_fallback_wait_timeout(), Some(Duration::from_secs(30)));
    assert_eq!(config.gpu_devices, vec![0]);
    assert_eq!(config.device_budget_fraction, None);
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
fn server_config_small_query_admission_profile_sets_serving_defaults() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "small_query".to_string()),
        (NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 16);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 16);
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Bytes(1 << 30));
    assert_eq!(config.admission_policy.grant().want, MemoryShare::PolicyDefault);
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::Optimistic);
}

#[test]
fn server_config_accepts_small_query_when_one_minimum_grant_fits_device_capacity() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "small_query".to_string()),
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, (2_u64 << 30).to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, (1_u64 << 30).to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 8);
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Bytes(1 << 30));
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::Optimistic);
}

#[test]
fn server_config_rejects_profile_minimum_above_device_capacity_with_profile_facts() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "small_query".to_string()),
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, (512_u64 << 20).to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, (256_u64 << 20).to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert_eq!(error.fact_value("config_field"), Some(NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV));
    assert_eq!(error.fact_value("reason"), Some("admission_grant_min_above_device_capacity"));
    assert_eq!(error.fact_value("grant_min_bytes"), Some("1073741824"));
    assert_eq!(error.fact_value("device_capacity_bytes"), Some("536870912"));
}

#[test]
fn server_config_rejects_explicit_minimum_above_capacity_with_minimum_field() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "small_query".to_string()),
        (NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, (768_u64 << 20).to_string()),
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, (512_u64 << 20).to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, (256_u64 << 20).to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert_eq!(error.fact_value("config_field"), Some(NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV));
    assert_eq!(error.fact_value("reason"), Some("admission_grant_min_above_device_capacity"));
    assert_eq!(error.fact_value("grant_min_bytes"), Some("805306368"));
}

#[test]
fn server_config_small_query_admission_profile_counts_only_pinned_device() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "small_query".to_string()),
        (NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string()),
        (NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "pinned".to_string()),
        (NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV, "1".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 8);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 8);
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::Pinned { device_ordinal: 1 });
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Bytes(1 << 30));
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::Optimistic);
}

#[test]
fn server_config_explicit_admission_limit_and_grant_min_override_small_query_profile() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "small_query".to_string()),
        (NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string()),
        (NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "3".to_string()),
        (NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, "0.25".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 3);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 3);
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Fraction(0.25));
}

#[test]
fn server_config_rejects_invalid_admission_workload_profile() {
    let env = BTreeMap::from([(NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, "graph_burst".to_string())]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV));
}

#[test]
fn server_config_accepts_cpu_fallback_statement_limit_env() {
    let env = BTreeMap::from([(NEXUS_SERVER_CPU_FALLBACK_MAX_CONCURRENT_STATEMENTS_ENV, "4".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.cpu_fallback_max_concurrent_statements.map(|value| value.get()), Some(4));
    assert_eq!(
        config.native_optimizer_config.cpu_fallback_max_concurrent_statements().map(|value| value.get()),
        Some(4)
    );
}

#[test]
fn server_config_accepts_cpu_fallback_wait_timeout_env() {
    let env = BTreeMap::from([(NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS_ENV, "250".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.cpu_fallback_wait_timeout, Some(Duration::from_millis(250)));
    assert_eq!(config.native_optimizer_config.cpu_fallback_wait_timeout(), Some(Duration::from_millis(250)));
}

#[test]
fn server_config_zero_cpu_fallback_wait_timeout_waits_indefinitely() {
    let env = BTreeMap::from([(NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS_ENV, "0".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.cpu_fallback_wait_timeout, None);
    assert_eq!(config.native_optimizer_config.cpu_fallback_wait_timeout(), None);
}

#[test]
fn server_config_zero_cpu_fallback_statement_limit_disables_bound() {
    let env = BTreeMap::from([(NEXUS_SERVER_CPU_FALLBACK_MAX_CONCURRENT_STATEMENTS_ENV, "0".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.cpu_fallback_max_concurrent_statements, None);
    assert_eq!(config.native_optimizer_config.cpu_fallback_max_concurrent_statements(), None);
    assert_eq!(config.cpu_fallback_wait_timeout, Some(Duration::from_secs(30)));
    assert_eq!(config.native_optimizer_config.cpu_fallback_wait_timeout(), None);
}

#[test]
fn server_config_accepts_toml_cpu_fallback_statement_limit() {
    let (_dir, env) = env_with_server_toml(
        r#"
[fallback]
max_concurrent_statements = 3
"#,
    );

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.cpu_fallback_max_concurrent_statements.map(|value| value.get()), Some(3));
    assert_eq!(
        config.native_optimizer_config.cpu_fallback_max_concurrent_statements().map(|value| value.get()),
        Some(3)
    );
}

#[test]
fn server_config_accepts_toml_cpu_fallback_wait_timeout() {
    let (_dir, env) = env_with_server_toml(
        r#"
[fallback]
wait_timeout_ms = 125
"#,
    );

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.cpu_fallback_wait_timeout, Some(Duration::from_millis(125)));
    assert_eq!(config.native_optimizer_config.cpu_fallback_wait_timeout(), Some(Duration::from_millis(125)));
}

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
        ("native.host_fetch_acquire_timeout_secs", NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV),
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

    let timeout = NEXUS_SERVER_NATIVE_CONFIG_OPTIONS
        .iter()
        .find(|option| option.toml_field == "native.host_fetch_acquire_timeout_secs")
        .expect("host-fetch timeout descriptor");
    assert_eq!(timeout.scope, ServerNativeConfigScope::SourceIoResource);
    assert_eq!(timeout.session_key, None);
    assert!(!timeout.scope.is_bounded_only());

    let statistical = NEXUS_SERVER_NATIVE_CONFIG_OPTIONS
        .iter()
        .find(|option| option.toml_field == "native.statistical_aggregate_execution_mode")
        .expect("statistical aggregate descriptor");
    assert_eq!(statistical.scope, ServerNativeConfigScope::NativeExecution);
    assert!(!statistical.scope.is_bounded_only());
}

#[test]
fn server_config_rejects_zero_admission_limit() {
    let env = BTreeMap::from([(NEXUS_SERVER_MAX_IN_FLIGHT_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_MAX_IN_FLIGHT_ENV));
}

#[test]
fn server_config_accepts_preferred_concurrency_knob() {
    let env = BTreeMap::from([(NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "3".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 3);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 3);
}

#[test]
fn server_config_accepts_admission_grant_band_and_slot_guarantee() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, "1024".to_string()),
        (NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV, "0.75".to_string()),
        (NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE_ENV, "optimistic".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.admission_policy.grant().min, MemoryShare::Bytes(1024));
    assert_eq!(config.admission_policy.grant().want, MemoryShare::Fraction(0.75));
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::Optimistic);
}

#[test]
fn server_config_accepts_admission_placement_policy_and_per_device_cap() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "best_fit".to_string()),
        (NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV, "8".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::BestFit);
    assert_eq!(config.admission_policy.max_queries_per_device(), Some(8));
}

#[test]
fn server_config_accepts_pinned_admission_placement() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string()),
        (NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "pinned".to_string()),
        (NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV, "1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::Pinned { device_ordinal: 1 });
}

#[test]
fn server_config_accepts_grant_min_without_pinning_policy_default_want() {
    let env = BTreeMap::from([(NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, "1024".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.admission_policy.grant().min, MemoryShare::Bytes(1024));
    assert_eq!(config.admission_policy.grant().want, MemoryShare::PolicyDefault);
}

#[test]
fn server_config_accepts_toml_admission_policy() {
    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
workload_profile = "small_query"
max_concurrent_queries = 4
max_queries_per_device = 2
placement_policy = "pinned"
pinned_device_ordinal = 0
grant_min = 0.25
grant_want = 0.75
slot_guarantee = "optimistic"
"#,
    );
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 4);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 4);
    assert_eq!(config.admission_policy.max_queries_per_device(), Some(2));
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::Pinned { device_ordinal: 0 });
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Fraction(0.25));
    assert_eq!(config.admission_policy.grant().want, MemoryShare::Fraction(0.75));
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::Optimistic);
}

#[test]
fn server_config_toml_profile_capacity_error_names_toml_profile_field() {
    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
workload_profile = "small_query"

[native]
device_budget_bytes = 536870912
max_source_chunk_bytes = 268435456
max_row_groups_per_chunk = 1
"#,
    );

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert_eq!(error.fact_value("config_field"), Some("admission.workload_profile"));
    assert_eq!(error.fact_value("reason"), Some("admission_grant_min_above_device_capacity"));
}

#[test]
fn server_config_env_overrides_toml_admission_policy() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[admission]
max_concurrent_queries = 4
max_queries_per_device = 2
placement_policy = "least_loaded"
grant_min = 0.25
grant_want = 0.50
slot_guarantee = "guaranteed_minimum"
wait_timeout_ms = 45000
"#,
    );
    env.insert(NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "3".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, "1024".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV, "0.75".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV, "3".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "best_fit".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE_ENV, "optimistic".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV, "5000".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.max_in_flight, 3);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 3);
    assert_eq!(config.admission_policy.max_queries_per_device(), Some(3));
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::BestFit);
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Bytes(1024));
    assert_eq!(config.admission_policy.grant().want, MemoryShare::Fraction(0.75));
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::Optimistic);
    assert_eq!(config.admission_policy.wait_timeout(), Duration::from_millis(5000));
}

#[test]
fn server_config_env_placement_policy_override_ignores_toml_pinned_device() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[admission]
placement_policy = "pinned"
pinned_device_ordinal = 0
"#,
    );
    env.insert(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "best_fit".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::BestFit);
}

#[test]
fn server_config_rejects_conflicting_admission_aliases() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, "3".to_string()),
        (NEXUS_SERVER_MAX_IN_FLIGHT_ENV, "2".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV));
    assert!(error.to_string().contains(NEXUS_SERVER_MAX_IN_FLIGHT_ENV));
}

#[test]
fn server_config_rejects_zero_admission_wait_timeout() {
    let env = BTreeMap::from([(NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV));
}

#[test]
fn server_config_rejects_zero_max_queries_per_device() {
    let env = BTreeMap::from([(NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV));
}

#[test]
fn server_config_rejects_invalid_placement_policy() {
    let env = BTreeMap::from([(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "packed".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV));
}

#[test]
fn server_config_rejects_pinned_placement_without_device() {
    let env = BTreeMap::from([(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "pinned".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV));
    assert_eq!(error.fact_value("reason"), Some("pinned_placement_missing_device_ordinal"));
    assert_eq!(error.fact_value("required_config_field"), Some(NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV));
}

#[test]
fn server_config_rejects_pinned_device_without_pinned_policy() {
    let env = BTreeMap::from([(NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV));
    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV));
    assert_eq!(error.fact_value("reason"), Some("pinned_device_requires_pinned_placement"));
    assert_eq!(error.fact_value("conflicting_config_field"), Some(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV));
}

#[test]
fn server_config_toml_pinned_errors_name_toml_companion_fields() {
    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
placement_policy = "pinned"
"#,
    );
    let missing_device = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert_eq!(missing_device.fact_value("config_field"), Some("admission.placement_policy"));
    assert_eq!(missing_device.fact_value("required_config_field"), Some("admission.pinned_device_ordinal"));
    assert!(!missing_device.to_string().contains(NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV));

    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
pinned_device_ordinal = 0
"#,
    );
    let missing_policy = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert_eq!(missing_policy.fact_value("config_field"), Some("admission.pinned_device_ordinal"));
    assert_eq!(missing_policy.fact_value("conflicting_config_field"), Some("admission.placement_policy"));
    assert!(!missing_policy.to_string().contains(NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV));
}

#[test]
fn server_config_rejects_pinned_device_outside_gpu_devices() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_GPU_DEVICES_ENV, "0".to_string()),
        (NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, "pinned".to_string()),
        (NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV, "1".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV));
    assert!(error.to_string().contains(NEXUS_SERVER_GPU_DEVICES_ENV));
}

#[test]
fn server_config_accepts_single_gpu_device_ordinal() {
    let env = BTreeMap::from([(NEXUS_SERVER_GPU_DEVICES_ENV, "0".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.gpu_devices, vec![0]);
}

#[test]
fn server_config_accepts_multi_gpu_device_subset() {
    let env = BTreeMap::from([(NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.gpu_devices, vec![0, 1]);
}

#[test]
fn server_config_rejects_duplicate_gpu_device_subset() {
    let env = BTreeMap::from([(NEXUS_SERVER_GPU_DEVICES_ENV, "0,0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_GPU_DEVICES_ENV));
}

#[test]
fn server_config_rejects_invalid_device_budget_fraction() {
    for value in ["0", "1.1", "nan"] {
        let env = BTreeMap::from([(NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV, value.to_string())]);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

        assert!(error.to_string().contains(NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV));
    }
}

#[test]
fn server_config_accepts_device_budget_fraction_for_auto_sized_budget() {
    let env = BTreeMap::from([(NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV, "0.5".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.device_budget_fraction, Some(0.5));
    assert!(
        config.native_optimizer_config.native_execution_config().memory_control_policy().is_bounded(),
        "device budget fraction should keep the bounded auto-sized native profile"
    );
}

#[test]
fn server_config_rejects_device_budget_fraction_with_explicit_device_budget() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV, "0.8".to_string()),
        (NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, "1024".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "512".to_string()),
        (NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV));
}

#[test]
fn query_service_handle_rejects_stale_session_budget_extension() {
    let service = QueryService::new(
        QueryServiceConfig { max_concurrent_queries: 1, stream_pool_size: Some(3), ..QueryServiceConfig::default() },
        vec![QueryServiceTenantConfig::new("tenant", 3 * 1024 * 1024)],
    )
    .unwrap();
    let handle = Arc::new(service.try_admit_query("tenant").unwrap().expect("query should admit"));
    let admitted_policy = handle.memory_control_policy();
    let query_config = CudfOptimizerConfig::default().with_native_execution_config(
        NativeCudfExecutionConfig::bounded_auto_sized().with_query_service_handle(handle),
    );
    let stale_server_config = CudfOptimizerConfig::default()
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(12 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1));
    let mut options = ConfigOptions::default();
    options.extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(&stale_server_config));

    let error = query_config.with_datafusion_config_options(&options).unwrap_err();

    assert!(error.to_string().contains("native.device_budget_bytes"));
    assert!(error.to_string().contains("query-service admission"));
    assert_eq!(query_config.native_execution_config().memory_control_policy(), admitted_policy);
}

#[test]
fn server_config_accepts_statement_ticket_ttl() {
    let env = BTreeMap::from([(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "30".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.statement_ticket_ttl.as_secs(), 30);
}

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

#[test]
fn server_config_rejects_zero_statement_ticket_ttl() {
    let env = BTreeMap::from([(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV));
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
            max_retry_attempts: datafusion_nexus::NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
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
        (NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV, "12".to_string()),
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
    assert_eq!(native.source_fetch_config().host_fetch_acquire_timeout, Duration::from_secs(12));
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
host_fetch_acquire_timeout_secs = 12
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
    assert_eq!(native.source_fetch_config().host_fetch_acquire_timeout, Duration::from_secs(12));
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
host_fetch_acquire_timeout_secs = 0
"#,
    );
    env.insert(NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "7".to_string());
    env.insert(NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "conservative".to_string());
    env.insert(NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV, "11".to_string());

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
    assert_eq!(native.source_fetch_config().host_fetch_acquire_timeout, Duration::from_secs(11));
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
        nexus_query_engine::SourceSchedulerPolicy::default().max_source_reads_in_flight()
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
            max_retry_attempts: datafusion_nexus::NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
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
fn server_config_rejects_zero_host_fetch_acquire_timeout() {
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV, "0".to_string())]);

    let error = ServerConfig::from_env_get(|env_name| env.get(env_name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV));
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
host_fetch_acquire_timeout_secs = 0
"#,
            "native.host_fetch_acquire_timeout_secs",
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

fn env_with_server_toml(toml_body: &str) -> (tempfile::TempDir, BTreeMap<&'static str, String>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, toml_body).expect("write config file");
    let path_str = path.to_string_lossy().into_owned();
    (dir, BTreeMap::from([(NEXUS_SERVER_CONFIG_FILE_ENV, path_str)]))
}

#[cfg(not(feature = "iceberg"))]
#[test]
fn server_config_rejects_iceberg_env_without_iceberg_feature() {
    let env = BTreeMap::from([(NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("--features server,iceberg"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_cugraph_env_without_cugraph_feature() {
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("--features server,cugraph"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_accepts_explicit_cugraph_disabled_without_cugraph_feature() {
    // One env file must be able to serve both artifact flavors: an explicit
    // `false` requests no capability, so a no-cugraph build accepts it.
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "false".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_cugraph_settings_alongside_disabled_env_without_cugraph_feature() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "false".to_string()),
        (datafusion_nexus::server::NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, "false".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("--features server,cugraph"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_malformed_cugraph_enabled_without_cugraph_feature() {
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "maybe".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("must be a boolean"));
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_accepts_enabled_env() {
    use datafusion_nexus::CugraphGraphConstructionPolicy;
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
        .sql("SELECT overall, coverage_json FROM nexus_explain_coverage('SELECT 1')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches[0].num_rows(), 1);
    let json: serde_json::Value = serde_json::from_str(batch_string_value(&batches[0], 1)).unwrap();
    assert_eq!(json["overall"]["verdict"], batch_string_value(&batches[0], 0));
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
    use datafusion_nexus::{CudfOptimizerConfig, CudfSessionStateBuilderExt};

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

#[derive(Debug)]
struct ReadOnlyCatalog {
    schema_name: String,
    schema: Arc<ReadOnlySchema>,
}

impl ReadOnlyCatalog {
    fn new(
        schema_name: impl Into<String>,
        tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>,
    ) -> Self {
        Self { schema_name: schema_name.into(), schema: Arc::new(ReadOnlySchema::new(tables)) }
    }
}

impl datafusion::catalog::CatalogProvider for ReadOnlyCatalog {
    fn schema_names(&self) -> Vec<String> {
        vec![self.schema_name.clone()]
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn datafusion::catalog::SchemaProvider>> {
        (name == self.schema_name).then(|| Arc::clone(&self.schema) as Arc<dyn datafusion::catalog::SchemaProvider>)
    }
}

#[derive(Debug)]
struct ReadOnlySchema {
    tables: BTreeMap<String, Arc<dyn datafusion::datasource::TableProvider>>,
}

impl ReadOnlySchema {
    fn new(tables: impl IntoIterator<Item = (&'static str, Arc<dyn datafusion::datasource::TableProvider>)>) -> Self {
        Self { tables: tables.into_iter().map(|(name, table)| (name.to_string(), table)).collect() }
    }
}

#[async_trait::async_trait]
impl datafusion::catalog::SchemaProvider for ReadOnlySchema {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table(
        &self,
        name: &str,
    ) -> datafusion::common::Result<Option<Arc<dyn datafusion::datasource::TableProvider>>> {
        Ok(self.tables.get(name).cloned())
    }

    async fn table_type(&self, name: &str) -> datafusion::common::Result<Option<datafusion::datasource::TableType>> {
        Ok(self.tables.get(name).map(|table| table.table_type()))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

fn edge_table() -> Arc<dyn datafusion::datasource::TableProvider> {
    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::datasource::MemTable;

    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![0, 1])), Arc::new(Int64Array::from(vec![1, 2]))],
    )
    .unwrap();
    Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
}

fn batch_i64_value(batch: &arrow::record_batch::RecordBatch, column: usize) -> i64 {
    batch.column(column).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0)
}

fn batch_string_value(batch: &arrow::record_batch::RecordBatch, column: usize) -> &str {
    batch.column(column).as_any().downcast_ref::<arrow::array::StringArray>().unwrap().value(0)
}

#[test]
fn server_config_rejects_malformed_bind_address() {
    let env = BTreeMap::from([(NEXUS_SERVER_BIND_ENV, "not-a-socket".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_BIND_ENV));
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
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert!(iceberg.tables.is_empty(), "dynamic catalog mode: tables must be empty when no tables env var is set");
    assert_eq!(iceberg.scan_planning_timeout.as_secs(), 60);
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_scan_planning_timeout() {
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV, "7".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.scan_planning_timeout.as_secs(), 7);
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_rejects_zero_scan_planning_timeout() {
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV, "0".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(datafusion_nexus::server::NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_accepts_backend_neutral_namespace_env() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_TABLES_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (NEXUS_ICEBERG_TABLES_ENV, "orders=tpch_sf1_orders,nation".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
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
        NEXUS_ICEBERG_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_TABLES_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_NAME_ENV, "iceberg".to_string()),
        // Glue-specific catalog handle is set independently.
        (NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, "my_glue_handle".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "analytics".to_string()),
        (NEXUS_ICEBERG_TABLES_ENV, "nation".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
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
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
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
    use datafusion_nexus::server::NEXUS_SERVER_SESSION_PROFILE_ENV;
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
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
        (NEXUS_ICEBERG_GLUE_DATABASE_ENV, "tpch_sf1".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (datafusion_nexus::server::NEXUS_ICEBERG_GLUE_TABLES_ENV, "nation".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");

    assert_eq!(iceberg.namespace, "tpch_sf1", "GLUE_DATABASE must map to namespace field");
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
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        NEXUS_ICEBERG_S3_REGION_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-2".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        (datafusion_nexus::iceberg::AWS_ACCESS_KEY_ID_ENV, "AKIA_GLUE".to_string()),
        (datafusion_nexus::iceberg::AWS_SECRET_ACCESS_KEY_ENV, "glue-secret".to_string()),
        (datafusion_nexus::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, "AKIA_LOCAL".to_string()),
        (datafusion_nexus::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, "local-secret".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
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

    assert_eq!(glue.region, "us-east-1");
    assert_eq!(glue.uri.as_deref(), Some("http://localhost:5001"));
    assert_eq!(glue.s3_endpoint.as_deref(), Some("http://localhost:9000"));
    assert_eq!(glue.s3_region.as_deref(), Some("us-east-2"));
    assert_eq!(glue.s3_path_style_access, Some(true));
    assert_eq!(iceberg.catalog_name, "glue", "default catalog name for glue kind");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_s3_credential_source_default_chain_keeps_default_chain_with_endpoint() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-1".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "default_chain".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        // Ambient AWS env credentials must flow through the provider-side
        // default chain, not be converted into static profile credentials.
        (datafusion_nexus::iceberg::AWS_ACCESS_KEY_ID_ENV, "AKIA_AMBIENT".to_string()),
        (datafusion_nexus::iceberg::AWS_SECRET_ACCESS_KEY_ENV, "ambient-secret".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
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
    assert_eq!(
        iceberg.parquet_read_strategy_config.credential_mode,
        nexus_query_engine::CredentialMode::ScopedResolved
    );
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_s3_credential_source_default_chain_applies_to_rest_backend() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_REST_URI_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "default_chain".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
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
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "imds".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV), "{error}");
    assert!(error.to_string().contains("default_chain"), "error should name the accepted value: {error}");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_s3_credential_source_rejects_static_credential_combination() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "http://localhost:5001".to_string()),
        (NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, "default_chain".to_string()),
        (datafusion_nexus::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, "AKIA_LOCAL".to_string()),
        (datafusion_nexus::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, "local-secret".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
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
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://bucket/warehouse/".to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, "https://user:glue-secret@glue.local".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "https://user:s3-secret@s3.local".to_string()),
        (datafusion_nexus::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

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
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains("glue"), "error should name the accepted kinds: {error}");
    assert!(error.to_string().contains("rest"), "error should name the accepted kinds: {error}");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_happy_path() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_REST_URI_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        NEXUS_ICEBERG_S3_REGION_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://localhost:9000".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-1".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (datafusion_nexus::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, "AKIA_REST".to_string()),
        (datafusion_nexus::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, "rest-secret".to_string()),
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
fn server_iceberg_config_owns_table_format_parquet_strategy() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, NEXUS_ICEBERG_REST_URI_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
        NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, "http://user:secret@localhost:9000/rustfs?sig=secret".to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-west-2".to_string()),
        (NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, "local_cache".to_string()),
        (NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, "/tmp/nexus-cache".to_string()),
        (NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, "read_through".to_string()),
        (NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, "4096".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");
    let strategy = iceberg.parquet_read_strategy_config;

    assert_eq!(strategy.preference, TableFormatParquetReadStrategyPreference::LocalCache);
    assert_eq!(strategy.lakehouse_cache_root, Some(PathBuf::from("/tmp/nexus-cache")));
    assert_eq!(strategy.lakehouse_cache_policy, TableFormatLakehouseCachePolicy::ReadThrough);
    assert_eq!(strategy.lakehouse_cache_max_bytes, Some(4096));
    assert_eq!(strategy.storage_profile_id.as_deref(), Some("table_format_s3"));
    assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000/rustfs"));
    assert_eq!(strategy.credential_mode, nexus_query_engine::CredentialMode::ScopedResolved);
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_rejects_unknown_table_format_parquet_strategy() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, NEXUS_ICEBERG_REST_URI_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, "surprise".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV));
    assert!(error.to_string().contains("host_buffer"));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_rejects_unknown_lakehouse_cache_policy() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_REST_URI_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
        (NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, "eventual".to_string()),
    ]);

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_LAKEHOUSE_CACHE_POLICY_ENV));
    assert!(error.to_string().contains("fill_before_read"));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_rejects_invalid_table_format_strategy_size() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_REST_URI_ENV, NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
    };
    let required = [
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ];
    let zero_size = required
        .iter()
        .cloned()
        .chain([(NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, "0".to_string())])
        .collect::<BTreeMap<_, _>>();
    let zero_size_error = ServerConfig::from_env_get(|name| zero_size.get(name).cloned()).unwrap_err();
    assert!(zero_size_error.to_string().contains(NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV));
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_does_not_require_region() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_REST_URI_ENV,
    };
    // No AWS_REGION_ENV present — REST must still succeed.
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_REST_URI_ENV, "http://localhost:8181".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let iceberg = config.iceberg.expect("iceberg config must be present");
    assert_eq!(expect_rest_backend(&iceberg).uri, "http://localhost:8181");
}

#[cfg(feature = "iceberg")]
#[test]
fn server_iceberg_config_kind_rest_requires_uri() {
    use datafusion_nexus::server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
    };
    let env = BTreeMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "rest".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, "s3://lakehouse/warehouse".to_string()),
    ]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
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
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(
        error.to_string().contains(datafusion_nexus::server::NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV),
        "error should name the missing warehouse env var: {error}"
    );
}
