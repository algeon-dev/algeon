//! Server-issued bearer token store with optional TTL.
//!
//! The store is shared across the [`crate::server::auth::handle_handshake`]
//! path (which inserts on successful credential check) and the
//! [`crate::server::auth::AuthInterceptor`] path (which looks up on every
//! authenticated RPC). Both paths share a single [`dashmap::DashMap`] so
//! contention on the hot per-RPC lookup is minimal.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use rand::{RngCore, rngs::OsRng};

/// Identity attached to an authenticated request. Populated by the
/// interceptor and read by any downstream handler that wants to log /
/// authorize against the calling user. Carried inside the
/// [`tonic::Request`] extensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    pub username: String,
    pub roles: Vec<String>,
}

#[derive(Debug, Clone)]
struct TokenEntry {
    principal: Principal,
    /// Wall-clock instant after which the token must be rejected. `None`
    /// means the token never expires.
    expires_at: Option<Instant>,
}

/// Thread-safe bearer token store with optional per-token TTL.
///
/// # Clone semantics
///
/// `TokenStore` wraps a [`std::sync::Arc`] internally; cloning is cheap
/// and every clone shares the same backing map. The interceptor and the
/// handshake handler hold separate clones of the same store.
///
/// # Token format
///
/// Tokens are 32 random bytes produced by [`rand::rngs::OsRng`], encoded
/// as URL-safe base64 without padding. That is 43 characters of
/// `[A-Za-z0-9_-]`, fits cleanly inside an `Authorization: Bearer <...>`
/// header, and has 256 bits of entropy.
#[derive(Debug, Clone, Default)]
pub struct TokenStore {
    inner: Arc<DashMap<String, TokenEntry>>,
}

impl TokenStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a fresh random token bound to `principal`. The token expires
    /// after `ttl` if provided, otherwise never. Returns the bearer
    /// string the server should hand to the client.
    pub fn issue(&self, principal: Principal, ttl: Option<Duration>) -> String {
        let mut bytes = [0u8; 32];
        OsRng.fill_bytes(&mut bytes);
        let token = base64_url_no_pad(&bytes);
        let expires_at = ttl.map(|ttl| Instant::now() + ttl);
        self.inner.insert(token.clone(), TokenEntry { principal, expires_at });
        token
    }

    /// Look up `token`. Expired entries are removed on hit, so a single
    /// caller that races against expiry observes the same outcome
    /// across retries. Returns `None` if the token is unknown or
    /// expired.
    pub fn validate(&self, token: &str) -> Option<Principal> {
        let entry = self.inner.get(token)?;
        if let Some(expires_at) = entry.expires_at
            && Instant::now() >= expires_at
        {
            drop(entry);
            self.inner.remove(token);
            return None;
        }
        Some(entry.principal.clone())
    }

    /// Drop expired entries. Optional — `validate` cleans up lazily —
    /// but useful as a periodic sweep to bound memory in long-running
    /// servers that issue tokens faster than they are used.
    pub fn sweep_expired(&self) {
        let now = Instant::now();
        self.inner.retain(|_, entry| match entry.expires_at {
            Some(expires_at) => now < expires_at,
            None => true,
        });
    }

    /// Number of live tokens (including any not yet swept). Test-only
    /// hook; production callers do not need this.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
}

fn base64_url_no_pad(bytes: &[u8]) -> String {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal(name: &str) -> Principal {
        Principal { username: name.to_string(), roles: vec!["reader".to_string()] }
    }

    #[test]
    fn issue_and_validate_round_trip() {
        let store = TokenStore::new();
        let token = store.issue(principal("alice"), None);
        assert_eq!(token.len(), 43, "32 raw bytes -> 43 base64url chars");
        let resolved = store.validate(&token).expect("token must validate");
        assert_eq!(resolved.username, "alice");
    }

    #[test]
    fn unknown_token_rejects() {
        let store = TokenStore::new();
        assert!(store.validate("not-a-token").is_none());
    }

    #[test]
    fn expired_token_rejects_and_is_removed() {
        let store = TokenStore::new();
        let token = store.issue(principal("alice"), Some(Duration::from_millis(0)));
        std::thread::sleep(Duration::from_millis(1));
        assert!(store.validate(&token).is_none(), "expired token must reject");
        assert_eq!(store.len(), 0, "expired entry must be removed from the map");
    }

    #[test]
    fn sweep_expired_removes_stale_entries() {
        let store = TokenStore::new();
        let live = store.issue(principal("alice"), Some(Duration::from_secs(60)));
        let _stale = store.issue(principal("bob"), Some(Duration::from_millis(0)));
        std::thread::sleep(Duration::from_millis(1));
        assert_eq!(store.len(), 2);
        store.sweep_expired();
        assert_eq!(store.len(), 1);
        assert!(store.validate(&live).is_some());
    }

    #[test]
    fn tokens_are_unique() {
        let store = TokenStore::new();
        let a = store.issue(principal("alice"), None);
        let b = store.issue(principal("alice"), None);
        assert_ne!(a, b);
    }
}
