use std::{
    net::{SocketAddr, TcpListener},
    sync::OnceLock,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, AuthConfig, AuthMode, ServerConfig, StaticUserConfig, hash_password,
        init_cuda_device, serve_with_listener,
    },
};
use arrow_flight::{
    Action, FlightDescriptor, FlightInfo,
    error::FlightError,
    sql::{
        ActionCreatePreparedStatementRequest, ActionCreatePreparedStatementResult, Any, CommandPreparedStatementQuery,
        ProstMessageExt, TicketStatementQuery, client::FlightSqlServiceClient,
    },
};
use futures::TryStreamExt;
use prost::Message;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair,
    KeyUsagePurpose,
};
use tokio::time::{sleep, timeout};
use tonic::{
    Code,
    transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity},
};

fn declare_test_gpu_ownership(config: &mut ServerConfig) {
    config.gpu_memory_ownership =
        config.gpu_devices().into_iter().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)).collect();
}

fn test_server_config() -> ServerConfig {
    test_server_config_with_toml("")
}

fn test_server_config_with_toml(extra_toml: &str) -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, format!("[[admission.device_profiles]]\ndevice_ordinal = 0\n\n{extra_toml}"))
        .expect("write server config");
    let path = path.to_string_lossy().into_owned();
    ServerConfig::from_vars([(ALGEON_SERVER_CONFIG_FILE_ENV, path)]).expect("explicit test device profile")
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
    // ArrowError::IpcError. The test asserts only that the call failed and
    // that the message reflects unauthenticated, which is the message a
    // client surfaces to the user.
    let display = err.to_string();
    assert!(
        display.contains("Unauthenticated") || display.to_ascii_lowercase().contains("invalid"),
        "handshake error must mention auth failure, got: {display}"
    );
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
async fn one_user_cannot_redeem_another_users_ticket() {
    // A query id is an opaque bearer reference that the server returns on the wire.
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
        .do_action(Action { r#type: "GetQueryStats".to_string(), body: query_id.clone().into_bytes().into() })
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

    let stats =
        alice.do_action(Action { r#type: "GetQueryStats".to_string(), body: query_id.into_bytes().into() }).await;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tls_listener_serves_tls_clients_and_rejects_plaintext_clients() {
    let pki = TestPki::new();
    let (listener, addr) = free_local_addr();
    let mut config = tls_server_config(&pki, false);
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });

    let tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(&pki.ca_pem)).domain_name("localhost");
    let mut client = connect_endpoint_with_retry(tls_endpoint(addr, tls), addr).await;
    let rows = select_one_rows(&mut client).await.expect("a TLS client completes a query");
    // The TLS client has proven the server is serving, so a plaintext failure
    // below cannot come from a server that is still starting.
    let plaintext = timeout(Duration::from_secs(15), plaintext_query(addr)).await;
    server.abort();

    assert_eq!(rows, 1);
    let outcome = plaintext.expect("the server must refuse a plaintext client without hanging");
    assert!(outcome.is_err(), "a plaintext client must not complete a query against a TLS listener");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mtls_listener_requires_a_client_certificate() {
    let pki = TestPki::new();
    let (listener, addr) = free_local_addr();
    let mut config = tls_server_config(&pki, true);
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });

    let trusted_server =
        ClientTlsConfig::new().ca_certificate(Certificate::from_pem(&pki.ca_pem)).domain_name("localhost");
    let with_identity = trusted_server.clone().identity(Identity::from_pem(&pki.client_cert_pem, &pki.client_key_pem));
    let mut client = connect_endpoint_with_retry(tls_endpoint(addr, with_identity), addr).await;
    let rows = select_one_rows(&mut client).await.expect("a client with a CA-signed certificate completes a query");
    // The identified client has proven the server is serving, so the
    // certificate-less failure below is the server's verdict.
    let anonymous = timeout(Duration::from_secs(15), anonymous_tls_query(addr, trusted_server)).await;
    server.abort();

    assert_eq!(rows, 1);
    let outcome = anonymous.expect("the server must refuse a client without a certificate without hanging");
    assert!(outcome.is_err(), "a client without a certificate must not complete a query against an mTLS listener");
}

async fn select_one_rows(client: &mut FlightSqlServiceClient<Channel>) -> Result<usize, FlightError> {
    let info = client.execute("select 1".to_string(), None).await?;
    let ticket = info.endpoint[0].ticket.clone().expect("statement ticket");
    let batches: Vec<_> = client.do_get(ticket).await?.try_collect().await?;
    Ok(batches.iter().map(|batch| batch.num_rows()).sum())
}

async fn plaintext_query(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let channel = Endpoint::from_shared(format!("http://{addr}"))?.connect().await?;
    select_one_rows(&mut FlightSqlServiceClient::new(channel)).await?;
    Ok(())
}

async fn anonymous_tls_query(addr: SocketAddr, tls: ClientTlsConfig) -> Result<(), Box<dyn std::error::Error>> {
    let channel = tls_endpoint(addr, tls).connect().await?;
    select_one_rows(&mut FlightSqlServiceClient::new(channel)).await?;
    Ok(())
}

fn tls_endpoint(addr: SocketAddr, tls: ClientTlsConfig) -> Endpoint {
    Endpoint::from_shared(format!("https://{addr}")).expect("endpoint URI").tls_config(tls).expect("client TLS config")
}

/// Generates certificates per test to avoid expiring checked-in fixtures.
struct TestPki {
    ca_pem: String,
    server_cert_pem: String,
    server_key_pem: String,
    client_cert_pem: String,
    client_key_pem: String,
}

impl TestPki {
    fn new() -> Self {
        let mut ca_params = CertificateParams::new(Vec::new()).expect("CA params");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        ca_params.distinguished_name.push(DnType::CommonName, "algeon-test-ca");
        let ca = CertifiedIssuer::self_signed(ca_params, KeyPair::generate().expect("CA key")).expect("CA certificate");
        let (server_cert_pem, server_key_pem) =
            Self::leaf(&ca, "algeon-test-server", vec!["localhost".to_string()], ExtendedKeyUsagePurpose::ServerAuth);
        let (client_cert_pem, client_key_pem) =
            Self::leaf(&ca, "algeon-test-client", Vec::new(), ExtendedKeyUsagePurpose::ClientAuth);
        Self { ca_pem: ca.pem(), server_cert_pem, server_key_pem, client_cert_pem, client_key_pem }
    }

    fn leaf(
        ca: &CertifiedIssuer<'_, KeyPair>,
        common_name: &str,
        names: Vec<String>,
        usage: ExtendedKeyUsagePurpose,
    ) -> (String, String) {
        let mut params = CertificateParams::new(names).expect("leaf params");
        params.extended_key_usages = vec![usage];
        params.distinguished_name.push(DnType::CommonName, common_name);
        let key = KeyPair::generate().expect("leaf key");
        let cert = params.signed_by(&key, ca).expect("leaf certificate");
        (cert.pem(), key.serialize_pem())
    }
}

fn tls_server_config(pki: &TestPki, require_client_certificate: bool) -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let write = |name: &str, pem: &str| {
        let path = dir.path().join(name);
        std::fs::write(&path, pem).expect("write PEM");
        path
    };
    let cert = write("server.pem", &pki.server_cert_pem);
    let key = write("server.key", &pki.server_key_pem);
    let mut toml = format!("[tls]\ncert_path = {cert:?}\nkey_path = {key:?}\n");
    if require_client_certificate {
        toml.push_str(&format!("client_ca_path = {:?}\n", write("clients.pem", &pki.ca_pem)));
    }
    test_server_config_with_toml(&toml)
}

/// Ask for a prepared statement's `FlightInfo` over the raw Flight channel,
/// carrying the client's own bearer token. The Flight SQL client builds this
/// request only from a `PreparedStatement` it owns, which is what this test
/// must not do.
async fn get_flight_info_for_prepared(
    client: &mut FlightSqlServiceClient<Channel>,
    handle: &[u8],
) -> Result<FlightInfo, tonic::Status> {
    let token = client.token().cloned().expect("an authenticated client holds a bearer token");
    let descriptor = FlightDescriptor::new_cmd(
        CommandPreparedStatementQuery { prepared_statement_handle: handle.to_vec().into() }.as_any().encode_to_vec(),
    );
    let mut request = tonic::Request::new(descriptor);
    request.metadata_mut().insert("authorization", format!("Bearer {token}").parse().expect("bearer header is ASCII"));
    client.inner_mut().get_flight_info(request).await.map(tonic::Response::into_inner)
}

/// Create a prepared statement through the raw action channel so the test can
/// hold the opaque handle the server issued.
async fn create_prepared_handle(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> Vec<u8> {
    let mut results = client
        .do_action(Action {
            r#type: "CreatePreparedStatement".to_string(),
            body: ActionCreatePreparedStatementRequest { query: sql.to_string(), transaction_id: None }
                .as_any()
                .encode_to_vec()
                .into(),
        })
        .await
        .expect("create prepared statement");
    let body = results.message().await.expect("action stream").expect("action result").body;
    let any = Any::decode(&*body).expect("action result envelope");
    let result: ActionCreatePreparedStatementResult =
        any.unpack().expect("unpack result").expect("prepared statement result");
    result.prepared_statement_handle.to_vec()
}

/// The `query_id` the server attaches to a successful `GetFlightInfo`.
fn query_id_of(info: &FlightInfo) -> String {
    let ticket = info.endpoint[0].ticket.as_ref().expect("statement ticket");
    let any = Any::decode(&*ticket.ticket).expect("ticket envelope");
    let statement: TicketStatementQuery = any.unpack().expect("unpack ticket").expect("statement ticket kind");
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
    static CUDA_WARM: OnceLock<()> = OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        // Best-effort: serve_with_listener re-runs (and surfaces) any error.
        let _ = init_cuda_device(0);
    });
}

/// Reserve a loopback port and return the live listener so the caller can
/// pass it straight to `serve_with_listener`. Keeping the socket bound closes
/// the bind/serve gap where a parallel test could steal the port.
fn free_local_addr() -> (TcpListener, SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

async fn connect_with_retry(addr: SocketAddr) -> FlightSqlServiceClient<Channel> {
    connect_endpoint_with_retry(Endpoint::from_shared(format!("http://{addr}")).unwrap(), addr).await
}

async fn connect_endpoint_with_retry(endpoint: Endpoint, addr: SocketAddr) -> FlightSqlServiceClient<Channel> {
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut backoff = Duration::from_millis(10);
    loop {
        match endpoint.connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) => {
                if Instant::now() >= deadline {
                    panic!("Flight SQL server did not accept connections at {addr} within 15s: {error:?}");
                }
                sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
            },
        }
    }
}
