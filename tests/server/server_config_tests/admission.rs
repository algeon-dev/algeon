#[test]
fn server_config_defaults_to_two_native_query_admissions() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
    assert_eq!(config.max_in_flight, 2);
    assert_eq!(config.admission_policy.max_concurrent_queries(), 2);
    let host_execution_qos = config.host_execution_qos.as_ref().expect("server defaults install host execution QoS");
    assert_eq!(host_execution_qos.max_concurrent_statements().get(), 2);
    assert_eq!(config.admission_policy.grant().min, MemoryShare::Fraction(1.0 / 8.0));
    assert_eq!(config.admission_policy.grant().want, MemoryShare::PolicyDefault);
    assert_eq!(config.admission_policy.max_queries_per_device(), None);
    assert_eq!(config.admission_policy.placement_policy(), QueryPlacementPolicy::LeastLoaded);
    assert_eq!(config.admission_policy.slot_guarantee(), SlotGuarantee::GuaranteedMinimum);
    assert_eq!(config.admission_policy.wait_timeout(), Duration::from_secs(60));
    assert_eq!(host_execution_qos.wait_timeout(), Some(Duration::from_secs(30)));
    assert_eq!(config.gpu_devices, vec![0]);
    assert!(config.gpu_memory_ownership.is_empty(), "server config loading must not invent deployment ownership");
    assert_eq!(config.device_budget_fraction, None);
    assert_eq!(config.statement_ticket_ttl.as_secs(), 300);
    assert_eq!(config.observability.per_query_max_records, 65_536);
    assert_eq!(config.observability.per_query_max_bytes, 8 * 1024 * 1024);
    assert_eq!(config.observability.active_logical_capture_max_bytes, 1024 * 1024 * 1024);
    assert_eq!(config.observability.global_sealed_max_count, 1_024);
    assert_eq!(config.observability.global_sealed_max_bytes, 1024 * 1024 * 1024);
    assert_eq!(config.observability.sealed_ttl, Duration::from_secs(60 * 60));
    assert_eq!(config.observability.shutdown_grace, Duration::from_secs(30));
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
fn server_config_accepts_host_execution_qos_statement_limit_env() {
    let env = BTreeMap::from([(NEXUS_SERVER_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS_ENV, "4".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.host_execution_qos.as_ref().map(|qos| qos.max_concurrent_statements().get()), Some(4));
}

#[test]
fn server_config_accepts_host_execution_qos_wait_timeout_env() {
    let env = BTreeMap::from([(NEXUS_SERVER_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS_ENV, "250".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.host_execution_qos.as_ref().and_then(|qos| qos.wait_timeout()), Some(Duration::from_millis(250)));
}

#[test]
fn server_config_zero_host_execution_qos_wait_timeout_waits_indefinitely() {
    let env = BTreeMap::from([(NEXUS_SERVER_HOST_EXECUTION_QOS_WAIT_TIMEOUT_MS_ENV, "0".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.host_execution_qos.as_ref().and_then(|qos| qos.wait_timeout()), None);
}

#[test]
fn server_config_zero_host_execution_qos_statement_limit_disables_bound() {
    let env = BTreeMap::from([(NEXUS_SERVER_HOST_EXECUTION_QOS_MAX_CONCURRENT_STATEMENTS_ENV, "0".to_string())]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert!(config.host_execution_qos.is_none());
}

#[test]
fn server_config_accepts_toml_host_execution_qos_statement_limit() {
    let (_dir, env) = env_with_server_toml(
        r#"
[host_execution_qos]
max_concurrent_statements = 3
"#,
    );

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.host_execution_qos.as_ref().map(|qos| qos.max_concurrent_statements().get()), Some(3));
}

#[test]
fn server_config_accepts_toml_host_execution_qos_wait_timeout() {
    let (_dir, env) = env_with_server_toml(
        r#"
[host_execution_qos]
wait_timeout_ms = 125
"#,
    );

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.host_execution_qos.as_ref().and_then(|qos| qos.wait_timeout()), Some(Duration::from_millis(125)));
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
fn server_config_accepts_statement_ticket_ttl() {
    let env = BTreeMap::from([(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "30".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.statement_ticket_ttl.as_secs(), 30);
}

#[test]
fn server_config_accepts_runtime_observability_limits() {
    let env = BTreeMap::from([
        (NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV, "123".to_string()),
        (NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV, "4567".to_string()),
        (NEXUS_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV, "7654".to_string()),
        (NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV, "8".to_string()),
        (NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV, "9999".to_string()),
        (NEXUS_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV, "42".to_string()),
        (NEXUS_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV, "7".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.observability.per_query_max_records, 123);
    assert_eq!(config.observability.per_query_max_bytes, 4567);
    assert_eq!(config.observability.active_logical_capture_max_bytes, 7654);
    assert_eq!(config.observability.global_sealed_max_count, 8);
    assert_eq!(config.observability.global_sealed_max_bytes, 9999);
    assert_eq!(config.observability.sealed_ttl, Duration::from_secs(42));
    assert_eq!(config.observability.shutdown_grace, Duration::from_secs(7));
}

#[test]
fn server_config_rejects_zero_runtime_observability_limits() {
    for field in [
        NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV,
        NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV,
        NEXUS_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV,
        NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV,
        NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV,
        NEXUS_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV,
        NEXUS_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV,
    ] {
        let env = BTreeMap::from([(field, "0".to_string())]);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert!(error.to_string().contains(field), "{error}");
    }
}

#[test]
fn server_config_rejects_zero_statement_ticket_ttl() {
    let env = BTreeMap::from([(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV));
}

#[test]
fn server_config_rejects_unrepresentable_statement_ticket_ttl() {
    let env = BTreeMap::from([(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, u64::MAX.to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV), "{error}");
    assert!(error.to_string().contains("monotonic clock range"), "{error}");
}

#[test]
fn server_config_loads_runtime_observability_toml_and_env_overrides_it() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[observability]
per_query_max_records = 10
per_query_max_bytes = 20
active_logical_capture_max_bytes = 30
global_sealed_max_count = 3
global_sealed_max_bytes = 40
sealed_ttl_secs = 50
shutdown_grace_secs = 60
"#,
    );
    env.insert(NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV, "11".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.observability.per_query_max_records, 11);
    assert_eq!(config.observability.per_query_max_bytes, 20);
    assert_eq!(config.observability.active_logical_capture_max_bytes, 30);
    assert_eq!(config.observability.global_sealed_max_count, 3);
    assert_eq!(config.observability.global_sealed_max_bytes, 40);
    assert_eq!(config.observability.sealed_ttl, Duration::from_secs(50));
    assert_eq!(config.observability.shutdown_grace, Duration::from_secs(60));
}

#[test]
fn server_config_rejects_zero_runtime_observability_toml_values() {
    for (field, diagnostic) in [
        ("per_query_max_records", NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV),
        ("per_query_max_bytes", NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV),
        ("active_logical_capture_max_bytes", NEXUS_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV),
        ("global_sealed_max_count", NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV),
        ("global_sealed_max_bytes", NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV),
        ("sealed_ttl_secs", NEXUS_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV),
        ("shutdown_grace_secs", NEXUS_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV),
    ] {
        let body = format!("[observability]\n{field} = 0\n");
        let (_dir, env) = env_with_server_toml(&body);

        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

        assert!(error.to_string().contains(diagnostic), "expected {diagnostic} in error: {error}");
    }
}

#[test]
fn server_config_rejects_unrepresentable_runtime_observability_durations_from_env_and_toml() {
    for field in [NEXUS_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV, NEXUS_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV] {
        let env = BTreeMap::from([(field, u64::MAX.to_string())]);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert!(error.to_string().contains(field), "{error}");
        assert!(error.to_string().contains("monotonic clock range"), "{error}");
    }

    for (field, diagnostic) in [
        ("sealed_ttl_secs", NEXUS_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV),
        ("shutdown_grace_secs", NEXUS_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV),
    ] {
        let body = format!("[observability]\n{field} = {}\n", u64::MAX);
        let (_dir, env) = env_with_server_toml(&body);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert!(error.to_string().contains(diagnostic), "{error}");
        assert!(error.to_string().contains("monotonic clock range"), "{error}");
    }
}

#[test]
fn server_config_rejects_malformed_bind_address() {
    let env = BTreeMap::from([(NEXUS_SERVER_BIND_ENV, "not-a-socket".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_BIND_ENV));
}
