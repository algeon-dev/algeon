//! Bearer-token validator used by the Flight SQL RPC methods.
//!
//! Following the reference pattern from arrow-rs's `flight_sql_server.rs`
//! example, the server impl calls [`AuthInterceptor::check`] at the top
//! of every non-handshake RPC. We do not use a tonic [`tonic::service::Interceptor`]
//! because the handshake call shares the same tower service and an
//! interceptor cannot reliably tell from the [`tonic::Request<()>`]
//! alone which RPC it is gating.

use tonic::{Status, metadata::MetadataMap};

use super::{AuthMode, Principal, TokenStore};

const AUTHORIZATION_HEADER: &str = "authorization";
const BEARER_PREFIX: &str = "Bearer ";

/// Clone-cheap handle over `(mode, token_store)`. Constructed once at
/// server start and cloned into the Flight SQL service.
#[derive(Clone)]
pub struct AuthInterceptor {
    mode: AuthMode,
    tokens: TokenStore,
}

impl AuthInterceptor {
    pub fn new(mode: AuthMode, tokens: TokenStore) -> Self {
        Self { mode, tokens }
    }

    pub fn mode(&self) -> &AuthMode {
        &self.mode
    }

    pub fn tokens(&self) -> &TokenStore {
        &self.tokens
    }

    /// Validate the request metadata against the configured mode.
    ///
    /// - In [`AuthMode::Disabled`]: returns `Ok(None)` unconditionally.
    /// - In [`AuthMode::BasicBearer`]: extracts the bearer token,
    ///   looks it up in the [`TokenStore`], and returns the bound
    ///   [`Principal`].
    ///
    /// Callers should invoke this at the top of every Flight SQL RPC
    /// method other than `do_handshake` (which establishes the token
    /// in the first place).
    pub fn check(&self, metadata: &MetadataMap) -> Result<Option<Principal>, Status> {
        match &self.mode {
            AuthMode::Disabled => Ok(None),
            AuthMode::BasicBearer { .. } => {
                let token = extract_bearer_token(metadata)?;
                let principal = self
                    .tokens
                    .validate(&token)
                    .ok_or_else(|| Status::unauthenticated("invalid or expired bearer token"))?;
                Ok(Some(principal))
            },
        }
    }
}

fn extract_bearer_token(metadata: &MetadataMap) -> Result<String, Status> {
    let header =
        metadata.get(AUTHORIZATION_HEADER).ok_or_else(|| Status::unauthenticated("missing authorization header"))?;
    let header = header.to_str().map_err(|_| Status::unauthenticated("authorization header is not ASCII"))?;
    let token = header
        .strip_prefix(BEARER_PREFIX)
        .ok_or_else(|| Status::unauthenticated("authorization header must use Bearer scheme"))?;
    if token.is_empty() {
        return Err(Status::unauthenticated("bearer token is empty"));
    }
    Ok(token.to_string())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tonic::metadata::AsciiMetadataValue;

    use super::*;

    fn principal(name: &str) -> Principal {
        Principal { username: name.to_string(), roles: vec![] }
    }

    fn metadata_with(value: Option<&str>) -> MetadataMap {
        let mut md = MetadataMap::new();
        if let Some(value) = value {
            md.insert(AUTHORIZATION_HEADER, AsciiMetadataValue::try_from(value).expect("ascii"));
        }
        md
    }

    #[test]
    fn disabled_mode_returns_no_principal() {
        let interceptor = AuthInterceptor::new(AuthMode::Disabled, TokenStore::new());
        assert!(interceptor.check(&metadata_with(None)).expect("disabled").is_none());
    }

    #[test]
    fn valid_bearer_returns_principal() {
        let tokens = TokenStore::new();
        let token = tokens.issue(principal("alice"), Some(Duration::from_secs(60)));
        let mode = AuthMode::BasicBearer { users: vec![], token_ttl_secs: 60 };
        let interceptor = AuthInterceptor::new(mode, tokens);

        let md = metadata_with(Some(&format!("Bearer {token}")));
        let p = interceptor.check(&md).expect("valid token").expect("principal");
        assert_eq!(p.username, "alice");
    }

    #[test]
    fn invalid_bearer_rejects() {
        let mode = AuthMode::BasicBearer { users: vec![], token_ttl_secs: 60 };
        let interceptor = AuthInterceptor::new(mode, TokenStore::new());

        let md = metadata_with(Some("Bearer not-a-real-token"));
        let status = interceptor.check(&md).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn missing_authorization_rejects() {
        let mode = AuthMode::BasicBearer { users: vec![], token_ttl_secs: 60 };
        let interceptor = AuthInterceptor::new(mode, TokenStore::new());

        let status = interceptor.check(&metadata_with(None)).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn wrong_scheme_rejects() {
        let mode = AuthMode::BasicBearer { users: vec![], token_ttl_secs: 60 };
        let interceptor = AuthInterceptor::new(mode, TokenStore::new());

        let md = metadata_with(Some("Basic dXNlcjpwYXNz"));
        let status = interceptor.check(&md).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn empty_bearer_rejects() {
        let mode = AuthMode::BasicBearer { users: vec![], token_ttl_secs: 60 };
        let interceptor = AuthInterceptor::new(mode, TokenStore::new());

        let md = metadata_with(Some("Bearer "));
        let status = interceptor.check(&md).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }
}
