use std::{collections::BTreeMap, time::Duration};

use algeon_datafusion::{
    backend::GpuDeviceMemoryLimit,
    cuml_sql::CumlModelStoreConfig,
    error::ErrorCode,
    native_exec::SourceChunking,
    planner::AlgeonGpuExecutionMode,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, AuthMode, ConfigValueSource, LoadedServerConfig, Result as ServerResult,
        ServerConfig, hash_password,
    },
};
use query_planning::{execution_contracts::SourceChunkPlanningMode, mode::StatisticalAggregateExecutionMode};
use support::{env_with_server_toml, load, load_toml};

mod admission;
mod cugraph;
mod feature_gates;
mod models;
mod native_profile;
mod startup;
mod support;
mod tls;

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
    let config = ServerConfig::from_vars(env.clone()).expect("load auth TOML overlay");

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
    let error = ServerConfig::from_vars(env.clone()).expect_err("missing TTL must reject");
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
    let error = ServerConfig::from_vars(env.clone()).expect_err("zero TTL must reject");
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
    let error = ServerConfig::from_vars(env.clone()).expect_err("oversized TTL must reject");
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
    let error = ServerConfig::from_vars(env.clone()).expect_err("removed roles field must reject");
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
    let error = ServerConfig::from_vars(env.clone()).expect_err("plaintext password must reject");
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
    let error = ServerConfig::from_vars(env.clone()).expect_err("malformed Argon2 PHC must reject");
    assert!(error.to_string().contains("valid Argon2 PHC string"), "unexpected error: {error}");
}

#[test]
fn server_config_binds_loopback_but_accepts_an_explicit_remote_address() {
    let config = load(BTreeMap::new()).unwrap();
    assert!(config.bind_addr.ip().is_loopback(), "{}", config.bind_addr);

    // Container port mappings and TLS-terminating proxies need a remote bind,
    // so no TLS or auth check may refuse it.
    let env = BTreeMap::from([("ALGEON_SERVER_BIND", "0.0.0.0:50051".to_string())]);
    let config = load(env).unwrap();
    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
}
