//! `do_handshake` implementation for [`AuthMode::BasicBearer`].
//!
//! Implements the same handshake pattern as the official arrow-rs
//! `flight_sql_server.rs` example: read `authorization: Basic
//! <base64(user:pass)>` from the request metadata, validate against the
//! configured static user list, mint a bearer token, and return it on
//! the response metadata as `authorization: Bearer <token>`. The Rust
//! `FlightSqlServiceClient` automatically captures that header for
//! subsequent RPCs.

use std::sync::LazyLock;

use argon2::Params;
use arrow_flight::{HandshakeRequest, HandshakeResponse};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use futures::stream::{self, BoxStream};
use tokio::task::spawn_blocking;
use tonic::{
    Request, Response, Status, Streaming,
    metadata::{AsciiMetadataValue, MetadataMap},
};

use super::{AuthInterceptor, AuthMode, Principal, StaticUserConfig, hash_password, verify_password};

const AUTHORIZATION_HEADER: &str = "authorization";
const BASIC_PREFIX: &str = "Basic ";

/// Memory budget for handshakes verifying concurrently on the blocking pool.
/// `password.rs` hashes with Argon2's default params (`m_cost` = 19 MiB per
/// verification, including the decoy path for unknown users); the tokio
/// blocking pool defaults to 512 threads, so with no cap a handshake flood
/// could schedule ~9.5 GiB of concurrent Argon2 work. 256 MiB keeps peak
/// verification memory small while still allowing real concurrency.
const VERIFICATION_MEMORY_BUDGET_KIB: u32 = 256 * 1024;

/// Number of Argon2 verifications allowed to run concurrently, derived from
/// [`VERIFICATION_MEMORY_BUDGET_KIB`] and the `m_cost` that `password.rs` uses,
/// which is Argon2's library default.
pub(crate) fn verification_permits() -> usize {
    (VERIFICATION_MEMORY_BUDGET_KIB / Params::DEFAULT_M_COST).max(1) as usize
}

/// Decoy Argon2 hash that gives unknown users the same password-check cost as
/// known users with wrong passwords.
static DUMMY_PASSWORD_HASH: LazyLock<String> = LazyLock::new(|| {
    hash_password("algeon-datafusion dummy password for constant-time handshake")
        .expect("hashing a constant password with default Argon2 parameters cannot fail")
});

/// Warm the decoy hash at startup so the first unknown-user handshake is not a
/// timing outlier.
pub(crate) fn warm_unknown_user_timing_decoy() {
    LazyLock::force(&DUMMY_PASSWORD_HASH);
}

/// Type alias for the streaming handshake response returned by
/// [`arrow_flight::sql::server::FlightSqlService::do_handshake`].
type HandshakeResponseStream = BoxStream<'static, Result<HandshakeResponse, Status>>;

/// Handle a Flight SQL handshake. Returns the same response stream type
/// expected by [`arrow_flight::sql::server::FlightSqlService::do_handshake`].
///
/// When [`AuthMode::Disabled`] is configured, the handshake is a no-op
/// passthrough that emits a single empty response; clients can still
/// call it (some drivers do unconditionally).
///
/// Credential verification runs Argon2 (real or decoy), which is
/// CPU/memory-heavy and must not run inline on an async worker thread.
/// It is dispatched to [`tokio::task::spawn_blocking`] gated by
/// `interceptor`'s verification semaphore; when the semaphore is
/// saturated the handshake fails fast with `resource_exhausted`, without
/// queuing. An unbounded queue of pending verifications is the same memory
/// DoS as running them inline.
pub async fn handle_handshake(
    interceptor: &AuthInterceptor,
    request: Request<Streaming<HandshakeRequest>>,
) -> Result<Response<HandshakeResponseStream>, Status> {
    let token = authenticate_and_issue_token(interceptor, request.metadata()).await?;
    let mut response = empty_response();
    if let Some(token) = token {
        let bearer = AsciiMetadataValue::try_from(format!("Bearer {token}"))
            .map_err(|_| Status::internal("issued token contains invalid header bytes"))?;
        response.metadata_mut().insert(AUTHORIZATION_HEADER, bearer);
    }
    Ok(response)
}

/// Metadata-only handshake core: validates credentials (off the async
/// worker) and mints a bearer token. Split out from [`handle_handshake`] so
/// it is testable without constructing a real [`Streaming`] request body,
/// which handshake handling never reads.
async fn authenticate_and_issue_token(
    interceptor: &AuthInterceptor,
    metadata: &MetadataMap,
) -> Result<Option<String>, Status> {
    let mode = interceptor.mode();
    let tokens = interceptor.tokens();
    match mode {
        AuthMode::Disabled => Ok(None),
        AuthMode::BasicBearer { users, .. } => {
            let (username, password) = extract_basic_credentials(metadata)?;
            let permit = interceptor
                .verification_semaphore()
                .try_acquire()
                .map_err(|_| Status::resource_exhausted("too many concurrent handshakes, retry later"))?;

            let users = users.clone();
            let username = spawn_blocking(move || authenticate_user(&users, &username, &password).map(|()| username))
                .await
                .map_err(|_| Status::internal("password verification task panicked"))??;
            // Return the verification slot once Argon2 finishes; issuing a token runs no Argon2.
            drop(permit);

            let principal = Principal { username };
            let token = tokens.issue(principal, mode.token_ttl())?;
            Ok(Some(token))
        },
    }
}

/// Authenticate while spending Argon2 work for unknown usernames, hiding
/// username existence from handshake timing. Takes an owned user list so
/// the check can run inside `spawn_blocking`, which requires `'static`.
fn authenticate_user(users: &[StaticUserConfig], username: &str, password: &str) -> Result<(), Status> {
    match users.iter().find(|u| u.username == username) {
        Some(user) => verify_credential(user, password),
        None => {
            // Spend decoy Argon2 work, then return the same error as a wrong password.
            let _ = verify_password(password, &DUMMY_PASSWORD_HASH);
            Err(Status::unauthenticated("invalid username or password"))
        },
    }
}

fn verify_credential(user: &StaticUserConfig, password: &str) -> Result<(), Status> {
    match verify_password(password, &user.password_hash) {
        Ok(true) => Ok(()),
        Ok(false) => Err(Status::unauthenticated("invalid username or password")),
        Err(error) => {
            // A misconfigured hash is a server bug. The error is logged and
            // surfaced as `internal`, so an operator can find it in the logs
            // without leaking the hash to the client.
            tracing::error!(username = %user.username, %error, "auth: password hash failed to parse");
            Err(Status::internal("server auth configuration error"))
        },
    }
}

fn extract_basic_credentials(metadata: &MetadataMap) -> Result<(String, String), Status> {
    let header =
        metadata.get(AUTHORIZATION_HEADER).ok_or_else(|| Status::unauthenticated("missing authorization header"))?;
    let header = header.to_str().map_err(|_| Status::unauthenticated("authorization header is not ASCII"))?;
    let encoded = header
        .strip_prefix(BASIC_PREFIX)
        .ok_or_else(|| Status::unauthenticated("authorization header must use Basic scheme"))?;
    let decoded = BASE64_STANDARD
        .decode(encoded.trim())
        .map_err(|_| Status::unauthenticated("authorization header base64 decode failed"))?;
    let decoded =
        std::str::from_utf8(&decoded).map_err(|_| Status::unauthenticated("authorization header is not UTF-8"))?;
    let (user, pass) =
        decoded.split_once(':').ok_or_else(|| Status::unauthenticated("authorization header missing `:` separator"))?;
    Ok((user.to_string(), pass.to_string()))
}

fn empty_response() -> Response<HandshakeResponseStream> {
    let stream = stream::iter([Ok(HandshakeResponse { protocol_version: 0, payload: vec![].into() })]);
    Response::new(Box::pin(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::auth::TokenStore;

    fn make_metadata(header: Option<&str>) -> MetadataMap {
        let mut md = MetadataMap::new();
        if let Some(value) = header {
            md.insert(AUTHORIZATION_HEADER, AsciiMetadataValue::try_from(value).expect("test header"));
        }
        md
    }

    #[test]
    fn extract_basic_credentials_round_trip() {
        let encoded = BASE64_STANDARD.encode("alice:s3cret");
        let header = format!("Basic {encoded}");
        let md = make_metadata(Some(&header));
        let (user, pass) = extract_basic_credentials(&md).expect("decode");
        assert_eq!(user, "alice");
        assert_eq!(pass, "s3cret");
    }

    #[test]
    fn missing_header_reports_unauthenticated() {
        let md = make_metadata(None);
        let status = extract_basic_credentials(&md).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn wrong_scheme_rejects() {
        let md = make_metadata(Some("Bearer abc"));
        let status = extract_basic_credentials(&md).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn missing_colon_rejects() {
        let encoded = BASE64_STANDARD.encode("no-colon-here");
        let header = format!("Basic {encoded}");
        let md = make_metadata(Some(&header));
        let status = extract_basic_credentials(&md).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    fn test_users() -> Vec<StaticUserConfig> {
        vec![StaticUserConfig {
            username: "alice".to_string(),
            password_hash:
                "$argon2id$v=19$m=19456,t=2,p=1$YWxnZW9uLXRlc3Qtc2FsdCE$fZ4tz7ti8qbyORcQw+9ta8d98CY+97cbxsddNlnzJt4"
                    .to_string(),
        }]
    }

    #[test]
    fn warm_unknown_user_timing_decoy_initializes_genuine_argon2_hash() {
        // Warmup is idempotent and leaves a usable Argon2 decoy.
        warm_unknown_user_timing_decoy();
        warm_unknown_user_timing_decoy();
        assert!(DUMMY_PASSWORD_HASH.starts_with("$argon2"), "decoy must be an Argon2 PHC string");
        assert!(matches!(verify_password("any password", &DUMMY_PASSWORD_HASH), Ok(false)));
    }

    fn basic_bearer_interceptor() -> AuthInterceptor {
        let mode = AuthMode::BasicBearer { users: test_users(), token_ttl_secs: 60 };
        AuthInterceptor::new(mode, TokenStore::new())
    }

    fn basic_auth_header(username: &str, password: &str) -> String {
        format!("Basic {}", BASE64_STANDARD.encode(format!("{username}:{password}")))
    }

    #[tokio::test]
    async fn authenticate_and_issue_token_succeeds_for_correct_credentials() {
        let interceptor = basic_bearer_interceptor();
        let md = make_metadata(Some(&basic_auth_header("alice", "s3cret")));
        let token = authenticate_and_issue_token(&interceptor, &md).await.expect("correct credentials succeed");
        assert!(token.is_some(), "basic_bearer mode must issue a token");
    }

    #[tokio::test]
    async fn authenticate_and_issue_token_rejects_unknown_and_wrong_password_identically() {
        let interceptor = basic_bearer_interceptor();
        let unknown_md = make_metadata(Some(&basic_auth_header("mallory", "s3cret")));
        let wrong_md = make_metadata(Some(&basic_auth_header("alice", "wrong")));
        let unknown = authenticate_and_issue_token(&interceptor, &unknown_md).await.unwrap_err();
        let wrong = authenticate_and_issue_token(&interceptor, &wrong_md).await.unwrap_err();
        assert_eq!(unknown.code(), tonic::Code::Unauthenticated);
        assert_eq!(unknown.code(), wrong.code());
        assert_eq!(unknown.message(), wrong.message());
    }

    #[tokio::test]
    async fn saturated_verification_semaphore_fails_fast_with_resource_exhausted() {
        let interceptor = basic_bearer_interceptor();
        // Hold every permit up front so the handshake below cannot acquire
        // one; this must reject immediately, without queuing.
        let semaphore = interceptor.verification_semaphore();
        let permits = semaphore.try_acquire_many(semaphore.available_permits() as u32).expect("acquire all permits");

        let md = make_metadata(Some(&basic_auth_header("alice", "s3cret")));
        let status = authenticate_and_issue_token(&interceptor, &md).await.unwrap_err();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);

        drop(permits);
    }
}
