use serde_json::Value;

use super::*;

#[test]
fn device_profiles_select_the_gpus_and_default_to_gpu_zero() {
    let loaded = LoadedServerConfig::from_vars(BTreeMap::<&str, String>::new()).expect("no file and no environment");
    assert!(loaded.device_profiles_defaulted());
    assert_eq!(loaded.config.gpu_devices(), vec![0]);
    assert_eq!(loaded.config.gpu_device_profiles[0].max_active_attempts(), 2, "the default profile sets nothing");

    let config = load_toml(
        "[[admission.device_profiles]]\ndevice_ordinal = 1\n\n[[admission.device_profiles]]\ndevice_ordinal = 0\n",
    )
    .expect("two profiles");
    assert_eq!(config.gpu_devices(), vec![1, 0], "profile order is selection order");

    let error = load_toml(
        "[[admission.device_profiles]]\ndevice_ordinal = 0\n\n[[admission.device_profiles]]\ndevice_ordinal = 0\n",
    )
    .expect_err("two profiles for one GPU");
    assert_eq!(error.fact_value("reason"), Some("duplicate_device"));
    assert_eq!(error.fact_value("device_ordinal"), Some("0"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles"));
}

#[test]
fn server_config_rejects_an_unknown_toml_field() {
    let (_dir, env) = env_with_server_toml("[admission]\nworkload_profile = \"small_query\"\n");

    let error = ServerConfig::from_vars(env.clone()).unwrap_err();

    assert!(error.to_string().contains("workload_profile"), "{error}");
}

#[test]
fn server_config_accepts_queue_admission_environment() {
    let mut env = BTreeMap::new();
    env.extend([
        ("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS", "8".to_string()),
        ("ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS", "5000".to_string()),
    ]);
    let config = ServerConfig::from_vars(env.clone()).unwrap();

    assert_eq!(config.queue_policy.max_queued_attempts(), 8);
    assert_eq!(config.queue_policy.max_queue_wait(), Duration::from_millis(5000));
}

#[test]
fn server_config_accepts_resource_admission_toml_and_env_overrides_it() {
    let (_dir, mut env) = env_with_server_toml(
        r#"
[admission]
max_queued_attempts = 4
max_queue_wait_ms = 1000
"#,
    );
    env.insert("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS", "8".to_string());
    env.insert("ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS", "5000".to_string());

    let config = ServerConfig::from_vars(env.clone()).unwrap();
    assert_eq!(config.queue_policy.max_queued_attempts(), 8);
    assert_eq!(config.queue_policy.max_queue_wait(), Duration::from_millis(5000));
}

#[test]
fn server_config_accepts_a_logical_capture_budget_below_the_admission_bounds() {
    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
max_queued_attempts = 64

[observability]
active_logical_capture_max_bytes = 1073741824

[[admission.device_profiles]]
device_ordinal = 0
max_active_attempts = 1
"#,
    );

    // A budget for 25 live captures: queries beyond it run without capture.
    let config = ServerConfig::from_vars(env.clone()).expect("the capture budget does not bound admission");
    assert_eq!(config.observability.active_logical_capture_max_bytes, Some(1_073_741_824));
}

#[test]
fn server_config_rejects_admission_bounds_the_coordinator_cannot_register() {
    let (_dir, env) = env_with_server_toml(
        r#"
[admission]
max_queued_attempts = 1023

[[admission.device_profiles]]
device_ordinal = 0
max_active_attempts = 1
"#,
    );

    let error = ServerConfig::from_vars(env.clone())
        .expect_err("1024 admitted or queued attempts fill every registration slot");
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued_attempts"));
    assert_eq!(error.fact_value("reason"), Some("admission_exceeds_live_query_capacity"));
    assert_eq!(error.fact_value("admission_live_attempts"), Some("1024"));
    assert_eq!(error.fact_value("max_live_queries"), Some("1024"));
}

#[test]
fn server_config_rejects_zero_resource_admission_values() {
    for field in ["ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS", "ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS"] {
        let env = BTreeMap::from([(field, "0".to_string())]);
        let error = ServerConfig::from_vars(env.clone()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
        assert!(error.to_string().contains(field), "{error}");
    }
}

#[test]
fn server_config_names_the_device_profile_key_in_configuration_errors() {
    for (profile, field) in [
        (
            "device_ordinal = 0\nmemory_limit_bytes = 0\nbackend_reserve_bytes = 0\n",
            "admission.device_profiles.memory_limit_bytes",
        ),
        ("device_ordinal = 0\nmemory_limit_bytes = 4096\n", "admission.device_profiles.backend_reserve_bytes"),
        ("device_ordinal = 0\nmax_active_attempts = 0\n", "admission.device_profiles.max_active_attempts"),
    ] {
        let (_dir, env) = env_with_server_toml(&format!("[[admission.device_profiles]]\n{profile}"));
        let error = ServerConfig::from_vars(env.clone()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
        assert_eq!(error.fact_value("config_field"), Some(field), "{error}");
        assert!(error.message().contains(field), "{error}");
    }
}

#[test]
fn server_config_projects_each_explicit_multi_gpu_profile_by_ordinal() {
    let (_dir, env) = env_with_server_toml(
        r#"
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 4096
backend_reserve_bytes = 0
default_query_limit_bytes = 512
cache_cap_bytes = 0
source_work_per_attempt = 2
max_active_attempts = 2
stream_pool_size = 6
worker_count = 3
lane_count = 2
install_telemetry = false

[[admission.device_profiles]]
device_ordinal = 3
memory_limit_bytes = 8192
backend_reserve_bytes = 128
default_query_limit_bytes = 1024
cache_cap_bytes = 2048
source_work_per_attempt = 1
max_active_attempts = 1
stream_pool_size = 2
worker_count = 2
lane_count = 1
install_telemetry = true
"#,
    );

    let config = ServerConfig::from_vars(env.clone()).unwrap();
    assert_eq!(config.gpu_devices(), vec![0, 3]);
    assert_eq!(config.gpu_device_profiles.len(), 2);
    let first = &config.gpu_device_profiles[0];
    let second = &config.gpu_device_profiles[1];
    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(first.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 });
    assert_eq!(first.backend_reserve_bytes(), 0);
    assert_eq!(first.default_query_limit_bytes(), Some(512));
    assert_eq!(first.cache_cap_bytes(), 0);
    assert_eq!(first.source_work_per_attempt(), 2);
    assert_eq!(first.max_active_attempts(), 2);
    assert_eq!(first.installation_bounds().unwrap().stream_pool_size, 6);
    assert!(!first.install_telemetry());

    assert_eq!(second.device_ordinal(), 3);
    assert_eq!(second.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 8192 });
    assert_eq!(second.backend_reserve_bytes(), 128);
    assert_eq!(second.default_query_limit_bytes(), Some(1024));
    assert_eq!(second.cache_cap_bytes(), 2048);
    assert_eq!(second.source_work_per_attempt(), 1);
    assert_eq!(second.max_active_attempts(), 1);
    assert_eq!(second.installation_bounds().unwrap().stream_pool_size, 2);
    assert!(second.install_telemetry());
}

#[test]
fn shipped_stress_server_fixtures_parse_as_valid_server_config() {
    let fixture = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/fixtures/stress/server.toml"));
    let config = load_toml(fixture).expect("shipped stress fixture");
    assert_eq!(config.gpu_devices(), vec![0]);
    assert_eq!(config.gpu_device_profiles[0].backend_reserve_bytes(), 0);
    let baseline: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/fixtures/stress/flight-baseline.json"
    )))
    .expect("Flight baseline");
    let devices = baseline["backend"]["device_profiles"].as_array().expect("device profiles");
    assert_eq!(config.gpu_device_profiles.len(), devices.len());
    for (server, device) in config.gpu_device_profiles.iter().zip(devices) {
        let GpuDeviceMemoryLimit::Explicit { limit_bytes } = server.memory_limit() else {
            panic!("Flight baseline requires an explicit device limit");
        };
        let installation = server.installation_bounds().expect("explicit installation");
        assert_eq!(serde_json::json!(server.device_ordinal()), device["ordinal"]);
        assert_eq!(serde_json::json!(limit_bytes), device["memory_limit_bytes"]);
        assert_eq!(serde_json::json!(server.backend_reserve_bytes()), device["backend_reserve_bytes"]);
        assert_eq!(serde_json::json!(server.cache_cap_bytes()), device["cache_cap_bytes"]);
        assert_eq!(serde_json::json!(server.default_query_limit_bytes()), device["default_query_limit_bytes"]);
        assert_eq!(serde_json::json!(server.source_work_per_attempt()), device["source_work_per_attempt"]);
        assert_eq!(serde_json::json!(server.max_active_attempts()), device["max_active_attempts"]);
        assert_eq!(serde_json::json!(server.install_telemetry()), device["install_telemetry"]);
        assert_eq!(
            serde_json::json!({
                "stream_pool_size": installation.stream_pool_size,
                "worker_count": installation.worker_count,
                "lane_count": installation.lane_count
            }),
            device["installation"]
        );
    }
    assert_eq!(
        serde_json::json!({
            "max_queued_attempts": config.queue_policy.max_queued_attempts(),
            "max_queue_wait_ms": config.queue_policy.max_queue_wait().as_millis()
        }),
        baseline["backend"]["queue"]
    );
    assert_eq!(config.native_optimizer_config.execution_mode(), AlgeonGpuExecutionMode::NativeRequired);
    let chunking = config.native_optimizer_config.native_execution_config().source_chunking();
    assert_eq!(
        serde_json::json!({
            "max_source_chunk_bytes": chunking.max_source_chunk_bytes.get(),
            "max_row_groups_per_chunk": chunking.max_row_groups_per_chunk.get(),
            "max_retry_attempts": chunking.max_retry_attempts.get(),
            "min_retry_chunk_row_groups": chunking.min_retry_chunk_row_groups.get()
        }),
        baseline["backend"]["native_execution"]
    );
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

    let error = ServerConfig::from_vars(env.clone()).unwrap_err();

    assert!(error.to_string().contains("backend_reserve_bytes"), "{error}");
    assert!(error.to_string().contains("memory_limit_bytes is explicit"), "{error}");
}

#[test]
fn server_config_accepts_statement_ticket_ttl() {
    let mut env = BTreeMap::new();
    env.insert("ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS", "30".to_string());
    let config = ServerConfig::from_vars(env.clone()).unwrap();
    assert_eq!(config.statement_ticket_ttl.as_secs(), 30);
}

#[test]
fn server_config_rejects_zero_statement_ticket_ttl() {
    let env = BTreeMap::from([("ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS", "0".to_string())]);
    let error = ServerConfig::from_vars(env.clone()).unwrap_err();
    assert!(error.to_string().contains("ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS"));
}

#[test]
fn server_config_parses_http2_keepalive_and_rejects_invalid_durations() {
    let mut env = BTreeMap::new();
    env.insert("ALGEON_SERVER_HTTP2_KEEPALIVE_INTERVAL_SECS", "2".to_string());
    env.insert("ALGEON_SERVER_HTTP2_KEEPALIVE_TIMEOUT_SECS", "3".to_string());
    let config = ServerConfig::from_vars(env.clone()).unwrap();
    assert_eq!(config.http2_keepalive.interval, Duration::from_secs(2));
    assert_eq!(config.http2_keepalive.timeout, Duration::from_secs(3));
    for field in ["ALGEON_SERVER_HTTP2_KEEPALIVE_INTERVAL_SECS", "ALGEON_SERVER_HTTP2_KEEPALIVE_TIMEOUT_SECS"] {
        let original = env[field].clone();
        for value in ["0", "invalid", "18446744073709551615"] {
            env.insert(field, value.to_string());
            let error = ServerConfig::from_vars(env.clone()).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
        env.insert(field, original);
    }
}
