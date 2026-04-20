#![cfg(feature = "server")]

use std::{io::Write, net::TcpListener, time::Duration};

use arrow_flight::{error::FlightError, sql::client::FlightSqlServiceClient};
use datafusion_nexus::server::{
    AuthConfig, AuthMode, ServerConfig, StaticUserConfig, hash_password, serve_with_listener,
};
use tonic::{Code, transport::Endpoint};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_disabled_passes_handshake_and_query_through() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    // Default mode is Disabled — explicit for clarity.
    config.auth = AuthConfig { mode: AuthMode::Disabled };

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    // Handshake with disabled auth still succeeds (the client always
    // calls it before queries; the server must not reject it).
    client.handshake("ignored", "ignored").await.expect("handshake must succeed in disabled mode");

    let flight_info = client.execute("select 1".to_string(), None).await.expect("query in disabled mode");
    assert!(!flight_info.endpoint.is_empty());

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn basic_bearer_happy_path() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.auth = basic_bearer_config("alice", "s3cret", 3600);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    client.handshake("alice", "s3cret").await.expect("valid credentials must hand back a bearer token");

    // The client has now stashed the bearer token internally and
    // attaches it to subsequent RPCs.
    let flight_info = client.execute("select 1".to_string(), None).await.expect("authenticated query must succeed");
    assert!(!flight_info.endpoint.is_empty());

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn basic_handshake_rejects_wrong_password() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.auth = basic_bearer_config("alice", "s3cret", 3600);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let err = client.handshake("alice", "wrong").await.expect_err("bad password must fail handshake");
    server.abort();

    // FlightSqlServiceClient::handshake wraps tonic Status in
    // ArrowError::IpcError. We only need to assert the call failed and
    // the message reflects unauthenticated; that's what a real client
    // would surface to the user.
    let display = err.to_string();
    assert!(
        display.contains("Unauthenticated") || display.to_ascii_lowercase().contains("invalid"),
        "handshake error must mention auth failure, got: {display}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn basic_handshake_rejects_unknown_user() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.auth = basic_bearer_config("alice", "s3cret", 3600);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let err = client.handshake("mallory", "s3cret").await.expect_err("unknown user must fail");
    server.abort();
    assert!(err.to_string().to_ascii_lowercase().contains("unauthenticated") || err.to_string().contains("invalid"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_without_handshake_rejects_with_unauthenticated() {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.auth = basic_bearer_config("alice", "s3cret", 3600);

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let err = client.execute("select 1".to_string(), None).await.expect_err("query without token must fail");
    server.abort();

    let FlightError::Tonic(status) = err else {
        panic!("expected tonic status, got {err:?}");
    };
    assert_eq!(status.code(), Code::Unauthenticated);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn toml_overlay_loads_basic_bearer_users() {
    let alice_hash = hash_password("from-toml").expect("hash");
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
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    let mut file = std::fs::File::create(&path).expect("create config file");
    file.write_all(toml_body.as_bytes()).expect("write config file");
    let path_str = path.to_string_lossy().into_owned();

    let config =
        ServerConfig::from_env_get(
            |name| {
                if name == "NEXUS_SERVER_CONFIG_FILE" { Some(path_str.clone()) } else { None }
            },
        )
        .expect("load config with TOML overlay");

    let AuthMode::BasicBearer { users, token_ttl_secs } = &config.auth.mode else {
        panic!("expected BasicBearer mode after TOML load, got {:?}", config.auth.mode);
    };
    assert_eq!(*token_ttl_secs, 600);
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].username, "alice");
    assert_eq!(users[0].roles, vec!["reader".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn toml_overlay_rejects_plaintext_password() {
    let toml_body = r#"
[auth.mode]
kind = "basic_bearer"

[[auth.mode.users]]
username = "alice"
password_hash = "plaintext-not-allowed"
"#;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, toml_body).expect("write config file");
    let path_str = path.to_string_lossy().into_owned();

    let err =
        ServerConfig::from_env_get(
            |name| {
                if name == "NEXUS_SERVER_CONFIG_FILE" { Some(path_str.clone()) } else { None }
            },
        )
        .expect_err("plaintext password must be rejected at load time");
    let msg = err.to_string();
    assert!(msg.contains("Argon2"), "error must mention argon2: {msg}");
}

fn basic_bearer_config(username: &str, password: &str, ttl_secs: u64) -> AuthConfig {
    let hash = hash_password(password).expect("hash test password");
    AuthConfig {
        mode: AuthMode::BasicBearer {
            users: vec![StaticUserConfig {
                username: username.to_string(),
                password_hash: hash,
                roles: vec!["reader".to_string()],
            }],
            token_ttl_secs: ttl_secs,
        },
    }
}

/// Warm the process-wide CUDA context exactly once. The first CUDA call pays a
/// multi-second `cuInit()` cost; doing it here (every test reserves an address
/// before spawning a server) keeps that cost off each server's connect window,
/// and the `OnceLock` collapses concurrent callers onto a single init.
fn warm_cuda_once() {
    static CUDA_WARM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        // Best-effort: serve_with_listener re-runs (and surfaces) any error.
        let _ = datafusion_nexus::server::init_cuda_device(0);
    });
}

/// Reserve a loopback port and hand back the live listener so the caller can
/// pass it straight to `serve_with_listener`. Keeping the socket bound closes
/// the bind/serve gap where a parallel test could steal the port.
fn free_local_addr() -> (TcpListener, std::net::SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

async fn connect_with_retry(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let uri = format!("http://{addr}");
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut backoff = Duration::from_millis(10);
    loop {
        match Endpoint::from_shared(uri.clone()).unwrap().connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) => {
                if std::time::Instant::now() >= deadline {
                    panic!("Flight SQL server did not accept connections at {addr} within 15s: {error:?}");
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
            },
        }
    }
}
