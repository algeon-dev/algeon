use std::{io::Write, net::TcpListener, time::Duration};

use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    server::{AuthConfig, AuthMode, ServerConfig, StaticUserConfig, hash_password, serve_with_listener},
};
use arrow_flight::{
    error::FlightError,
    sql::{ProstMessageExt, client::FlightSqlServiceClient},
};
use prost::Message;
use tonic::{Code, transport::Endpoint};

fn declare_test_gpu_ownership(config: &mut ServerConfig) {
    config.gpu_memory_ownership =
        config.gpu_devices.iter().copied().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)).collect();
}

fn test_server_config() -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, "[[admission.device_profiles]]\ndevice_ordinal = 0\n").expect("write server config");
    let path = path.to_string_lossy().into_owned();
    ServerConfig::from_env_get(|name| {
        (name == algeon_datafusion::server::ALGEON_SERVER_CONFIG_FILE_ENV).then(|| path.clone())
    })
    .expect("explicit test device profile")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_disabled_passes_handshake_and_query_through() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    // Default mode is Disabled; explicit for clarity.
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
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

[[admission.device_profiles]]
device_ordinal = 0
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
                if name == "ALGEON_SERVER_CONFIG_FILE" { Some(path_str.clone()) } else { None }
            },
        )
        .expect("load config with TOML overlay");

    let AuthMode::BasicBearer { users, token_ttl_secs } = &config.auth.mode else {
        panic!("expected BasicBearer mode after TOML load, got {:?}", config.auth.mode);
    };
    assert_eq!(*token_ttl_secs, 600);
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].username, "alice");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn toml_overlay_rejects_plaintext_password() {
    let toml_body = r#"
[auth.mode]
kind = "basic_bearer"
token_ttl_secs = 600

[[auth.mode.users]]
username = "alice"
password_hash = "plaintext-not-allowed"

[[admission.device_profiles]]
device_ordinal = 0
"#;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, toml_body).expect("write config file");
    let path_str = path.to_string_lossy().into_owned();

    let err =
        ServerConfig::from_env_get(
            |name| {
                if name == "ALGEON_SERVER_CONFIG_FILE" { Some(path_str.clone()) } else { None }
            },
        )
        .expect_err("plaintext password must be rejected at load time");
    let msg = err.to_string();
    assert!(msg.contains("Argon2"), "error must mention argon2: {msg}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_user_cannot_redeem_another_users_ticket() {
    // A query id is an opaque bearer reference handed back on the wire.
    // Authentication alone would let any other logged-in user redeem it.
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.auth = two_user_config();

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });

    let mut alice = connect_with_retry(addr).await;
    alice.handshake("alice", "alice-pw").await.expect("alice authenticates");
    let flight_info = alice.execute("select 1".to_string(), None).await.expect("alice plans her query");
    let ticket = flight_info.endpoint[0].ticket.clone().expect("statement ticket");

    let mut mallory = connect_with_retry(addr).await;
    mallory.handshake("mallory", "mallory-pw").await.expect("mallory authenticates");
    let error = mallory.do_get(ticket).await.expect_err("another user must not redeem alice's ticket");
    server.abort();

    assert_eq!(tonic_status(error).code(), Code::PermissionDenied);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_user_cannot_read_another_users_query_stats() {
    // Retained stats carry the query's SQL text, which may hold the
    // credentials of a CREATE EXTERNAL TABLE ... OPTIONS statement.
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.auth = two_user_config();

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });

    let mut alice = connect_with_retry(addr).await;
    alice.handshake("alice", "alice-pw").await.expect("alice authenticates");
    let flight_info = alice.execute("select 1".to_string(), None).await.expect("alice plans her query");
    let query_id = query_id_of(&flight_info);

    let mut mallory = connect_with_retry(addr).await;
    mallory.handshake("mallory", "mallory-pw").await.expect("mallory authenticates");
    let error = mallory
        .do_action(arrow_flight::Action {
            r#type: "GetQueryStats".to_string(),
            body: query_id.clone().into_bytes().into(),
        })
        .await
        .expect_err("another user must not read alice's query stats");
    server.abort();

    assert_eq!(tonic_status(error).code(), Code::PermissionDenied);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_user_still_reaches_their_own_query_stats() {
    // Ownership must narrow access without breaking the owner's own path.
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.auth = two_user_config();

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });

    let mut alice = connect_with_retry(addr).await;
    alice.handshake("alice", "alice-pw").await.expect("alice authenticates");
    let flight_info = alice.execute("select 1".to_string(), None).await.expect("alice plans her query");
    let query_id = query_id_of(&flight_info);

    let stats = alice
        .do_action(arrow_flight::Action { r#type: "GetQueryStats".to_string(), body: query_id.into_bytes().into() })
        .await;
    server.abort();
    stats.expect("the issuing principal reads its own query stats");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_user_cannot_use_another_users_prepared_statement() {
    // A prepared-statement handle is likewise an opaque bearer reference,
    // and it carries the prepared SQL text.
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.auth = two_user_config();

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });

    let mut alice = connect_with_retry(addr).await;
    alice.handshake("alice", "alice-pw").await.expect("alice authenticates");
    let handle = create_prepared_handle(&mut alice, "select 1").await;

    let mut mallory = connect_with_retry(addr).await;
    mallory.handshake("mallory", "mallory-pw").await.expect("mallory authenticates");
    let error = get_flight_info_for_prepared(&mut mallory, &handle)
        .await
        .expect_err("another user must not use alice's prepared statement");

    // The owner still reaches their own handle.
    let own = get_flight_info_for_prepared(&mut alice, &handle).await;
    server.abort();

    assert_eq!(error.code(), Code::PermissionDenied);
    own.expect("the issuing principal still uses its own prepared statement");
}

/// Ask for a prepared statement's `FlightInfo` over the raw Flight channel,
/// carrying the client's own bearer token. The Flight SQL client builds this
/// request only from a `PreparedStatement` it owns, which is exactly what this
/// test must not do.
async fn get_flight_info_for_prepared(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    handle: &[u8],
) -> Result<arrow_flight::FlightInfo, tonic::Status> {
    let token = client.token().cloned().expect("an authenticated client holds a bearer token");
    let descriptor = arrow_flight::FlightDescriptor::new_cmd(
        arrow_flight::sql::CommandPreparedStatementQuery { prepared_statement_handle: handle.to_vec().into() }
            .as_any()
            .encode_to_vec(),
    );
    let mut request = tonic::Request::new(descriptor);
    request.metadata_mut().insert("authorization", format!("Bearer {token}").parse().expect("bearer header is ASCII"));
    client.inner_mut().get_flight_info(request).await.map(tonic::Response::into_inner)
}

/// Create a prepared statement through the raw action channel so the test can
/// hold the opaque handle the server issued.
async fn create_prepared_handle(client: &mut FlightSqlServiceClient<tonic::transport::Channel>, sql: &str) -> Vec<u8> {
    let mut results = client
        .do_action(arrow_flight::Action {
            r#type: "CreatePreparedStatement".to_string(),
            body: arrow_flight::sql::ActionCreatePreparedStatementRequest {
                query: sql.to_string(),
                transaction_id: None,
            }
            .as_any()
            .encode_to_vec()
            .into(),
        })
        .await
        .expect("create prepared statement");
    let body = results.message().await.expect("action stream").expect("action result").body;
    let any = arrow_flight::sql::Any::decode(&*body).expect("action result envelope");
    let result: arrow_flight::sql::ActionCreatePreparedStatementResult =
        any.unpack().expect("unpack result").expect("prepared statement result");
    result.prepared_statement_handle.to_vec()
}

/// The `query_id` the server attaches to a successful `GetFlightInfo`.
fn query_id_of(info: &arrow_flight::FlightInfo) -> String {
    let ticket = info.endpoint[0].ticket.as_ref().expect("statement ticket");
    let any = arrow_flight::sql::Any::decode(&*ticket.ticket).expect("ticket envelope");
    let statement: arrow_flight::sql::TicketStatementQuery =
        any.unpack().expect("unpack ticket").expect("statement ticket kind");
    String::from_utf8(statement.statement_handle.to_vec()).expect("query id is UTF-8")
}

fn tonic_status(error: FlightError) -> tonic::Status {
    match error {
        FlightError::Tonic(status) => *status,
        other => panic!("expected tonic status, got {other:?}"),
    }
}

fn two_user_config() -> AuthConfig {
    AuthConfig {
        mode: AuthMode::BasicBearer {
            users: vec![
                StaticUserConfig {
                    username: "alice".to_string(),
                    password_hash: hash_password("alice-pw").expect("hash alice"),
                },
                StaticUserConfig {
                    username: "mallory".to_string(),
                    password_hash: hash_password("mallory-pw").expect("hash mallory"),
                },
            ],
            token_ttl_secs: 3600,
        },
    }
}

fn basic_bearer_config(username: &str, password: &str, ttl_secs: u64) -> AuthConfig {
    let hash = hash_password(password).expect("hash test password");
    AuthConfig {
        mode: AuthMode::BasicBearer {
            users: vec![StaticUserConfig { username: username.to_string(), password_hash: hash }],
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
        let _ = algeon_datafusion::server::init_cuda_device(0);
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
