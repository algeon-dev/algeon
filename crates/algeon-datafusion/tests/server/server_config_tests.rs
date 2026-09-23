use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[cfg(feature = "cugraph")]
use algeon_datafusion::cugraph_sql::CugraphGraphOptions;
use algeon_datafusion::{
    backend::GpuDeviceMemoryLimit,
    cuml_sql::CumlModelStoreConfig,
    error::ErrorCode,
    native_exec::SourceChunking,
    planner::{AlgeonGpuExecutionMode, DataFusionCpuExecutionPolicy},
    server::{
        ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV, ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV,
        ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
        ALGEON_SERVER_CUGRAPH_ENABLED_ENV, ALGEON_SERVER_EXECUTION_MODE_ENV, ALGEON_SERVER_FINAL_PLAN_REQUIREMENT_ENV,
        ALGEON_SERVER_GPU_DEVICES_ENV, ALGEON_SERVER_NATIVE_CONFIG_OPTIONS,
        ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
        ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
        ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
        ALGEON_SERVER_WORKSPACE_BACKING_ALIASES_ENV, ALGEON_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
        ALGEON_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, ALGEON_SERVER_WORKSPACE_CATALOG_ENV,
        ALGEON_SERVER_WORKSPACE_SCHEMA_ENV, AuthMode, ServerConfig, hash_password,
    },
};
use query_engine::{mode::StatisticalAggregateExecutionMode, runtime::SourceChunkPlanningMode};

// Server config contract tests are split by concern while staying in this
// integration-test module via include! so existing test names remain stable.
include!("server_config_tests/support.rs");
include!("server_config_tests/admission.rs");
include!("server_config_tests/native_profile.rs");
include!("server_config_tests/workspace.rs");
include!("server_config_tests/feature_gates.rs");
include!("server_config_tests/cugraph.rs");
include!("server_config_tests/models.rs");

#[test]
fn auth_toml_overlay_loads_static_basic_bearer_user() {
    let alice_hash = hash_password("from-toml").expect("hash");
    let toml_body = format!(
        r#"
[auth.mode]
kind = "basic_bearer"
token_ttl_secs = 600

[[auth.mode.users]]
username = "alice"
password_hash = "{alice_hash}"
"#,
    );
    let (_dir, env) = env_with_server_toml(&toml_body);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("load auth TOML overlay");

    let AuthMode::BasicBearer { users, token_ttl_secs } = &config.auth.mode else {
        panic!("expected basic bearer auth mode, got {:?}", config.auth.mode);
    };
    assert_eq!(*token_ttl_secs, 600);
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].username, "alice");
}

#[test]
fn auth_toml_overlay_requires_nonzero_token_ttl() {
    let alice_hash = hash_password("s3cret").expect("hash");
    let missing_ttl = format!(
        r#"
[auth.mode]
kind = "basic_bearer"

[[auth.mode.users]]
username = "alice"
password_hash = "{alice_hash}"
"#,
    );
    let (_dir, env) = env_with_server_toml(&missing_ttl);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect_err("missing TTL must reject");
    assert!(error.to_string().contains("token_ttl_secs"), "unexpected error: {error}");

    let zero_ttl = format!(
        r#"
[auth.mode]
kind = "basic_bearer"
token_ttl_secs = 0

[[auth.mode.users]]
username = "alice"
password_hash = "{alice_hash}"
"#,
    );
    let (_dir, env) = env_with_server_toml(&zero_ttl);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect_err("zero TTL must reject");
    assert!(error.to_string().contains("token_ttl_secs must be greater than zero"), "unexpected error: {error}");
}

#[test]
fn auth_toml_overlay_rejects_token_ttl_beyond_the_monotonic_clock_range() {
    let alice_hash = hash_password("s3cret").expect("hash");
    let toml_body = format!(
        r#"
[auth.mode]
kind = "basic_bearer"
token_ttl_secs = {}

[[auth.mode.users]]
username = "alice"
password_hash = "{alice_hash}"
"#,
        i64::MAX,
    );
    let (_dir, env) = env_with_server_toml(&toml_body);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect_err("oversized TTL must reject");
    assert!(error.to_string().contains("token_ttl_secs is too large"), "unexpected error: {error}");
}

#[test]
fn auth_toml_overlay_rejects_removed_roles_field() {
    let alice_hash = hash_password("s3cret").expect("hash");
    let toml_body = format!(
        r#"
[auth.mode]
kind = "basic_bearer"
token_ttl_secs = 600

[[auth.mode.users]]
username = "alice"
password_hash = "{alice_hash}"
roles = ["reader"]
"#,
    );
    let (_dir, env) = env_with_server_toml(&toml_body);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect_err("removed roles field must reject");
    assert!(error.to_string().contains("roles"), "unexpected error: {error}");
}

#[test]
fn auth_toml_overlay_rejects_plaintext_password() {
    let toml_body = r#"
[auth.mode]
kind = "basic_bearer"
token_ttl_secs = 600

[[auth.mode.users]]
username = "alice"
password_hash = "plaintext-not-allowed"
"#;
    let (_dir, env) = env_with_server_toml(toml_body);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect_err("plaintext password must reject");
    assert!(error.to_string().contains("Argon2"), "unexpected error: {error}");
}

#[test]
fn auth_toml_overlay_rejects_malformed_argon2_phc() {
    let toml_body = r#"
[auth.mode]
kind = "basic_bearer"
token_ttl_secs = 600

[[auth.mode.users]]
username = "alice"
password_hash = "$argon2id$v=19$m=19456,t=2,p=1$not-a-complete-phc"
"#;
    let (_dir, env) = env_with_server_toml(toml_body);
    let error =
        ServerConfig::from_env_get(|name| env.get(name).cloned()).expect_err("malformed Argon2 PHC must reject");
    assert!(error.to_string().contains("valid Argon2 PHC string"), "unexpected error: {error}");
}
