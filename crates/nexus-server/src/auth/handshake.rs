//! `do_handshake` implementation for [`AuthMode::BasicBearer`].
//!
//! Implements the same handshake pattern as the official arrow-rs
//! `flight_sql_server.rs` example: read `authorization: Basic
//! <base64(user:pass)>` from the request metadata, validate against the
//! configured static user list, mint a bearer token, and return it on
//! the response metadata as `authorization: Bearer <token>`. The Rust
//! `FlightSqlServiceClient` automatically captures that header for
//! subsequent RPCs.

use std::{pin::Pin, sync::LazyLock};

use arrow_flight::{HandshakeRequest, HandshakeResponse};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use futures::Stream;
use tonic::{
    Request, Response, Status, Streaming,
    metadata::{AsciiMetadataValue, MetadataMap},
};

use super::{AuthMode, Principal, StaticUserConfig, TokenStore, hash_password, verify_password};

const AUTHORIZATION_HEADER: &str = "authorization";
const BASIC_PREFIX: &str = "Basic ";

/// Decoy Argon2 hash that gives unknown users the same password-check cost as
/// known users with wrong passwords.
static DUMMY_PASSWORD_HASH: LazyLock<String> = LazyLock::new(|| {
    hash_password("datafusion-nexus dummy password for constant-time handshake")
        .expect("hashing a constant password with default Argon2 parameters cannot fail")
});

/// Warm the decoy hash at startup so the first unknown-user handshake is not a
/// timing outlier.
pub(crate) fn warm_unknown_user_timing_decoy() {
    LazyLock::force(&DUMMY_PASSWORD_HASH);
}

/// Type alias for the streaming handshake response returned by
/// [`arrow_flight::sql::server::FlightSqlService::do_handshake`].
type HandshakeResponseStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;

/// Handle a Flight SQL handshake. Returns the same response stream type
/// expected by [`arrow_flight::sql::server::FlightSqlService::do_handshake`].
///
/// When [`AuthMode::Disabled`] is configured, the handshake is a no-op
/// passthrough that emits a single empty response — clients can still
/// call it (some drivers do unconditionally), and the result is the
/// same as the pre-auth stub.
pub async fn handle_handshake(
    mode: &AuthMode,
    tokens: &TokenStore,
    request: Request<Streaming<HandshakeRequest>>,
) -> Result<Response<HandshakeResponseStream>, Status> {
    match mode {
        AuthMode::Disabled => Ok(empty_response()),
        AuthMode::BasicBearer { users, .. } => {
            let (username, password) = extract_basic_credentials(request.metadata())?;
            let user = authenticate_user(users, &username, &password)?;

            let principal = Principal { username: user.username.clone() };
            let token = tokens.issue(principal, mode.token_ttl());

            let mut response = empty_response();
            let bearer = AsciiMetadataValue::try_from(format!("Bearer {token}"))
                .map_err(|_| Status::internal("issued token contains invalid header bytes"))?;
            response.metadata_mut().insert(AUTHORIZATION_HEADER, bearer);
            Ok(response)
        },
    }
}

/// Authenticate while spending Argon2 work for unknown usernames, hiding
/// username existence from handshake timing.
fn authenticate_user<'a>(
    users: &'a [StaticUserConfig],
    username: &str,
    password: &str,
) -> Result<&'a StaticUserConfig, Status> {
    match users.iter().find(|u| u.username == username) {
        Some(user) => {
            verify_credential(user, password)?;
            Ok(user)
        },
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
            // Misconfigured hash is a server bug, not a client error;
            // log it and surface as `internal` so an operator can find
            // it in the logs without leaking the hash to the client.
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
    let stream = futures::stream::iter([Ok(HandshakeResponse { protocol_version: 0, payload: vec![].into() })]);
    Response::new(Box::pin(stream))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            password_hash: super::super::hash_password("s3cret").expect("hash"),
        }]
    }

    #[test]
    fn authenticate_user_accepts_correct_credentials() {
        let users = test_users();
        let user = authenticate_user(&users, "alice", "s3cret").expect("correct credentials authenticate");
        assert_eq!(user.username, "alice");
    }

    #[test]
    fn authenticate_user_rejects_wrong_password() {
        let users = test_users();
        let status = authenticate_user(&users, "alice", "wrong").unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn authenticate_user_rejects_unknown_user_with_identical_message() {
        // Unknown users and wrong passwords must be indistinguishable to clients.
        let users = test_users();
        let unknown = authenticate_user(&users, "mallory", "s3cret").unwrap_err();
        let wrong_password = authenticate_user(&users, "alice", "wrong").unwrap_err();
        assert_eq!(unknown.code(), tonic::Code::Unauthenticated);
        assert_eq!(unknown.message(), wrong_password.message());
    }

    #[test]
    fn warm_unknown_user_timing_decoy_initializes_genuine_argon2_hash() {
        // Warmup is idempotent and leaves a usable Argon2 decoy.
        warm_unknown_user_timing_decoy();
        warm_unknown_user_timing_decoy();
        assert!(DUMMY_PASSWORD_HASH.starts_with("$argon2"), "decoy must be an Argon2 PHC string");
        assert!(matches!(verify_password("any password", &DUMMY_PASSWORD_HASH), Ok(false)));
    }

    #[test]
    fn dummy_password_hash_is_genuine_argon2_work() {
        // The decoy must be a real PHC hash so verification performs Argon2 work.
        let outcome = verify_password("any password", &DUMMY_PASSWORD_HASH);
        assert!(matches!(outcome, Ok(false)), "dummy hash must verify as a real Argon2 mismatch, got: {outcome:?}");
    }
}
