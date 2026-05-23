#[test]
fn server_config_requires_explicit_device_profiles() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("missing_device_profiles"));
    assert_eq!(error.fact_value("device_ordinals"), Some("0"));
}

#[test]
fn server_config_rejects_removed_workload_profile_toml_field() {
    let (_dir, env) = env_with_server_toml("[admission]\nworkload_profile = \"small_query\"\n");

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("workload_profile"), "{error}");
}

#[test]
fn server_config_accepts_queue_admission_environment() {
    let (_dir, mut env) = env_with_default_device_profile();
    env.extend([
        (NEXUS_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV, "8".to_string()),
        (NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV, "5000".to_string()),
        (NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER_ENV, "7".to_string()),
    ]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert_eq!(config.queue_policy.max_queued_attempts(), 8);
    assert_eq!(config.queue_policy.max_queue_wait(), Duration::from_millis(5000));
    assert_eq!(config.queue_policy.max_overtakes_per_waiter(), 7);
}

#[test]
fn server_config_accepts_resource_admission_toml_and_env_overrides_it() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[admission]
max_queued_attempts = 4
max_queue_wait_ms = 1000
max_overtakes_per_waiter = 2
"#,
    );
    env.insert(NEXUS_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV, "8".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV, "5000".to_string());
    env.insert(NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER_ENV, "7".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert_eq!(config.queue_policy.max_queued_attempts(), 8);
    assert_eq!(config.queue_policy.max_queue_wait(), Duration::from_millis(5000));
    assert_eq!(config.queue_policy.max_overtakes_per_waiter(), 7);
}

#[test]
fn server_config_rejects_zero_resource_admission_values() {
    for field in [NEXUS_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV, NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV] {
        let env = BTreeMap::from([(field, "0".to_string())]);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert!(error.to_string().contains(field), "{error}");
    }
}

#[test]
fn server_config_rejects_removed_scalar_admission_environment() {
    for field in [
        "NEXUS_SERVER_ADMISSION_MAX_ACTIVE_ATTEMPTS_PER_DEVICE",
        "NEXUS_SERVER_ADMISSION_ATTEMPT_FLOOR_BYTES",
        "NEXUS_SERVER_ADMISSION_SOURCE_WORK_PER_ATTEMPT",
    ] {
        let env = BTreeMap::from([(field, "1".to_string())]);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert_eq!(error.fact_value("config_field"), Some(field), "{error}");
    }
}

#[test]
fn server_config_rejects_overtake_budget_above_u32() {
    let env =
        BTreeMap::from([(NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER_ENV, (u64::from(u32::MAX) + 1).to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains(NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER_ENV));
}

#[test]
fn server_config_rejects_removed_host_rate_limit() {
    let env = BTreeMap::from([("NEXUS_SERVER_MAX_IN_FLIGHT", "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains("NEXUS_SERVER_MAX_IN_FLIGHT"));
}

#[test]
fn server_config_rejects_single_gpu_device_ordinal_without_profile() {
    let env = BTreeMap::from([(NEXUS_SERVER_GPU_DEVICES_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("missing_device_profiles"));
    assert_eq!(error.fact_value("device_ordinals"), Some("0"));
}

#[test]
fn server_config_rejects_multi_gpu_device_subset_without_ordinal_profiles() {
    let env = BTreeMap::from([(NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("missing_device_profiles"));
    assert_eq!(error.fact_value("device_ordinals"), Some("0,1"));
}

#[test]
fn server_config_projects_each_explicit_multi_gpu_profile_by_ordinal() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 4096
backend_reserve_bytes = 0
attempt_floor_bytes = 512
cache_cap_bytes = 0
source_work_per_attempt = 2
max_active_attempts = 2
stream_pool_size = 6
worker_count = 2
lane_count = 2
install_telemetry = false

[[admission.device_profiles]]
device_ordinal = 3
memory_limit_bytes = 8192
backend_reserve_bytes = 128
attempt_floor_bytes = 1024
cache_cap_bytes = 2048
source_work_per_attempt = 1
max_active_attempts = 1
stream_pool_size = 2
worker_count = 1
lane_count = 1
install_telemetry = true
"#,
    );
    env.insert(NEXUS_SERVER_GPU_DEVICES_ENV, "0,3".to_string());

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert_eq!(config.gpu_devices, vec![0, 3]);
    assert_eq!(config.gpu_device_profiles.len(), 2);
    let first = &config.gpu_device_profiles[0];
    let second = &config.gpu_device_profiles[1];
    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(first.memory_limit(), datafusion_nexus::backend::GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 });
    assert_eq!(first.backend_reserve_bytes(), 0);
    assert_eq!(first.attempt_floor_bytes(), Some(512));
    assert_eq!(first.cache_cap_bytes(), 0);
    assert_eq!(first.source_work_per_attempt(), 2);
    assert_eq!(first.max_active_attempts(), 2);
    assert_eq!(first.installation_bounds().unwrap().stream_pool_size, 6);
    assert!(!first.install_telemetry());

    assert_eq!(second.device_ordinal(), 3);
    assert_eq!(second.memory_limit(), datafusion_nexus::backend::GpuDeviceMemoryLimit::Explicit { limit_bytes: 8192 });
    assert_eq!(second.backend_reserve_bytes(), 128);
    assert_eq!(second.attempt_floor_bytes(), Some(1024));
    assert_eq!(second.cache_cap_bytes(), 2048);
    assert_eq!(second.source_work_per_attempt(), 1);
    assert_eq!(second.max_active_attempts(), 1);
    assert_eq!(second.installation_bounds().unwrap().stream_pool_size, 2);
    assert!(second.install_telemetry());
}

#[test]
fn shipped_stress_server_fixtures_parse_as_valid_server_config() {
    for fixture in [
        include_str!("../../../../nexus-tools/fixtures/stress/server-s8-s2-scan-sf1.toml"),
        include_str!("../../../../nexus-tools/fixtures/stress/server-s8-s2-knee-sf10.toml"),
    ] {
        let (_dir, mut env) = env_with_server_toml(fixture);
        env.insert(NEXUS_SERVER_GPU_DEVICES_ENV, "0,1,2,3".to_string());
        ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("shipped stress fixture");
    }
}

#[test]
fn server_config_requires_backend_reserve_with_explicit_memory_limit() {
    let (_dir, env) = env_with_server_toml(
        r#"
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 4096
"#,
    );

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains("backend_reserve_bytes"), "{error}");
    assert!(error.to_string().contains("memory_limit_bytes is explicit"), "{error}");
}

#[test]
fn server_config_rejects_missing_duplicate_or_unselected_device_profiles() {
    let (_dir, mut missing_env) = env_with_server_toml(
        r#"
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 4096
backend_reserve_bytes = 0
"#,
    );
    missing_env.insert(NEXUS_SERVER_GPU_DEVICES_ENV, "0,1".to_string());
    let missing = ServerConfig::from_env_get(|name| missing_env.get(name).cloned()).unwrap_err();
    assert_eq!(missing.fact_value("reason"), Some("missing_devices"));
    assert_eq!(missing.fact_value("device_ordinals"), Some("1"));

    let (_dir, duplicate_env) = env_with_server_toml(
        r#"
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 4096
backend_reserve_bytes = 0

[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 8192
backend_reserve_bytes = 0
"#,
    );
    let duplicate = ServerConfig::from_env_get(|name| duplicate_env.get(name).cloned()).unwrap_err();
    assert_eq!(duplicate.fact_value("reason"), Some("duplicate_device"));
    assert_eq!(duplicate.fact_value("device_ordinal"), Some("0"));

    let (_dir, unselected_env) = env_with_server_toml(
        r#"
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 4096
backend_reserve_bytes = 0

[[admission.device_profiles]]
device_ordinal = 1
memory_limit_bytes = 8192
backend_reserve_bytes = 0
"#,
    );
    let unselected = ServerConfig::from_env_get(|name| unselected_env.get(name).cloned()).unwrap_err();
    assert_eq!(unselected.fact_value("reason"), Some("unconfigured_devices"));
    assert_eq!(unselected.fact_value("device_ordinals"), Some("1"));
}

#[test]
fn server_config_rejects_legacy_scalar_device_fields_with_explicit_profiles() {
    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
max_active_attempts_per_device = 2

[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 4096
backend_reserve_bytes = 0
"#,
    );

    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains("max_active_attempts_per_device"), "{error}");
}

#[test]
fn server_config_rejects_duplicate_gpu_device_subset() {
    let env = BTreeMap::from([(NEXUS_SERVER_GPU_DEVICES_ENV, "0,0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains(NEXUS_SERVER_GPU_DEVICES_ENV));
}

#[test]
fn server_config_rejects_removed_global_device_budget_fraction() {
    let env = BTreeMap::from([("NEXUS_SERVER_DEVICE_BUDGET_FRACTION", "0.8".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert_eq!(error.fact_value("config_field"), Some("NEXUS_SERVER_DEVICE_BUDGET_FRACTION"));
}

#[test]
fn server_config_accepts_statement_ticket_ttl() {
    let (_dir, mut env) = env_with_default_device_profile();
    env.insert(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "30".to_string());
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    assert_eq!(config.statement_ticket_ttl.as_secs(), 30);
}

#[test]
fn server_config_rejects_zero_statement_ticket_ttl() {
    let env = BTreeMap::from([(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, "0".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert!(error.to_string().contains(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV));
}
