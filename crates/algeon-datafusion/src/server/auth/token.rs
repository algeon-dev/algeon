//! Server-issued bearer token store with optional TTL.
//!
//! The store is shared across the [`super::handle_handshake`]
//! path (which inserts on successful credential check) and the
//! [`super::AuthInterceptor`] path (which looks up on every
//! authenticated RPC). Both paths share a single [`dashmap::DashMap`] so
//! contention on the hot per-RPC lookup is minimal.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;
use rand::{TryRng, rngs::SysRng};
use tonic::Status;

/// Hard cap on live (not-yet-expired-and-swept) bearer tokens. Each entry is
/// a small fixed-size struct (a `String` token key, a username, an
/// `Option<Instant>`), so this bounds the store to a few MiB even if clients
/// never reconnect to let old tokens expire and get swept. Sized well above
/// any realistic single-deployment client count while still being a real
/// bound, not an unbounded map.
const MAX_LIVE_TOKENS: usize = 100_000;

/// Identity attached to an authenticated request. Populated by the
/// interceptor and read by any downstream handler that wants to log /
/// authorize against the calling user. Carried inside the
/// [`tonic::Request`] extensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    pub username: String,
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
/// Tokens are 32 random bytes produced by [`rand::rngs::SysRng`], encoded
/// as URL-safe base64 without padding. That is 43 characters of
/// `[A-Za-z0-9_-]`, fits cleanly inside an `Authorization: Bearer <...>`
/// header, and has 256 bits of entropy.
#[derive(Debug, Clone)]
pub struct TokenStore {
    inner: Arc<DashMap<String, TokenEntry>>,
    max_live_tokens: usize,
}

impl Default for TokenStore {
    fn default() -> Self {
        Self { inner: Arc::default(), max_live_tokens: MAX_LIVE_TOKENS }
    }
}

impl TokenStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Same as [`TokenStore::new`] but with an overridden capacity, so
    /// capacity-boundary tests do not need to actually mint
    /// [`MAX_LIVE_TOKENS`] tokens to exercise the limit.
    #[cfg(test)]
    pub(crate) fn with_capacity_for_tests(max_live_tokens: usize) -> Self {
        Self { max_live_tokens, ..Self::default() }
    }

    /// Mint a fresh random token bound to `principal`. The token expires
    /// after `ttl` if provided, otherwise never. Returns the bearer
    /// string the server should hand to the client.
    ///
    /// At capacity, sweeps expired entries to reclaim space from tokens that
    /// were never looked up again (`validate` only sweeps on hit) and retries
    /// once; if the store is still full, issuance fails rather than growing
    /// the map without bound.
    ///
    /// The sweep is a full scan, so it runs only on the capacity edge rather
    /// than on every handshake: below the cap the map is already bounded and
    /// scanning buys nothing, while sweeping unconditionally would make a
    /// login storm quadratic in the number of live tokens.
    pub fn issue(&self, principal: Principal, ttl: Option<Duration>) -> Result<String, Status> {
        if self.inner.len() >= self.max_live_tokens {
            self.sweep_expired();
            if self.inner.len() >= self.max_live_tokens {
                return Err(Status::resource_exhausted("bearer token store is at capacity, retry later"));
            }
        }
        let mut bytes = [0u8; 32];
        SysRng.try_fill_bytes(&mut bytes).expect("system random source must be available to issue bearer tokens");
        let token = base64_url_no_pad(&bytes);
        let expires_at = ttl
            .map(|ttl| {
                expiry_after(ttl).ok_or_else(|| Status::internal("bearer token TTL exceeds the representable expiry"))
            })
            .transpose()?;
        self.inner.insert(token.clone(), TokenEntry { principal, expires_at });
        Ok(token)
    }

    /// Remove every entry whose TTL has passed. Called from `issue`, the
    /// only place the map grows, so the map is bounded without a
    /// background sweeper task.
    ///
    /// A token whose TTL has passed is already rejected by `validate`, so
    /// sweeping only reclaims memory and never revokes a usable token.
    fn sweep_expired(&self) {
        let now = Instant::now();
        self.inner.retain(|_, entry| entry.expires_at.is_none_or(|expires_at| now < expires_at));
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

    /// Number of live tokens (including any not yet swept). Test-only
    /// hook; production callers do not need this.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Expiry of a token issued now with `ttl`, or `None` when the sum exceeds
/// [`Instant`]'s range. Config validation and [`TokenStore::issue`] apply the
/// same bound.
pub(super) fn expiry_after(ttl: Duration) -> Option<Instant> {
    Instant::now().checked_add(ttl)
}

fn base64_url_no_pad(bytes: &[u8]) -> String {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal(name: &str) -> Principal {
        Principal { username: name.to_string() }
    }

    #[test]
    fn issue_and_validate_round_trip() {
        let store = TokenStore::new();
        let token = store.issue(principal("alice"), None).expect("issue under capacity");
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
        let token = store.issue(principal("alice"), Some(Duration::from_millis(0))).expect("issue under capacity");
        std::thread::sleep(Duration::from_millis(1));
        assert!(store.validate(&token).is_none(), "expired token must reject");
        assert_eq!(store.len(), 0, "expired entry must be removed from the map");
    }

    #[test]
    fn issue_rejects_ttl_beyond_the_instant_range() {
        let store = TokenStore::new();
        let status = store.issue(principal("alice"), Some(Duration::MAX)).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(store.len(), 0, "a refused issuance must not insert a token");
    }

    #[test]
    fn tokens_are_unique() {
        let store = TokenStore::new();
        let a = store.issue(principal("alice"), None).expect("issue under capacity");
        let b = store.issue(principal("alice"), None).expect("issue under capacity");
        assert_ne!(a, b);
    }

    #[test]
    fn issue_refuses_beyond_capacity() {
        const CAPACITY: usize = 8;
        let store = TokenStore::with_capacity_for_tests(CAPACITY);
        for _ in 0..CAPACITY {
            store.issue(principal("alice"), None).expect("issue under capacity");
        }
        let status = store.issue(principal("alice"), None).unwrap_err();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
        assert_eq!(store.len(), CAPACITY, "refused issuance must not grow the map");
    }

    #[test]
    fn issue_at_capacity_sweeps_expired_entries_to_reclaim_it() {
        const CAPACITY: usize = 8;
        let store = TokenStore::with_capacity_for_tests(CAPACITY);
        // Fill the store to capacity with already-expired tokens.
        for _ in 0..CAPACITY {
            store.issue(principal("alice"), Some(Duration::from_millis(0))).expect("issue under capacity");
        }
        std::thread::sleep(Duration::from_millis(1));
        // Hitting the cap must sweep rather than report the store as full:
        // every entry in it is already unusable.
        let token = store.issue(principal("bob"), None).expect("sweep reclaims capacity for a fresh issuance");
        assert_eq!(store.len(), 1, "the sweep must clear all expired entries, not just enough for one slot");
        let resolved = store.validate(&token).expect("token must validate");
        assert_eq!(resolved.username, "bob");
    }

    #[test]
    fn live_tokens_at_capacity_are_not_swept() {
        // The sweep runs on the capacity edge, so it must still distinguish
        // expired entries from live ones rather than clearing the store.
        const CAPACITY: usize = 8;
        let store = TokenStore::with_capacity_for_tests(CAPACITY);
        for _ in 0..CAPACITY {
            store.issue(principal("alice"), Some(Duration::from_secs(3600))).expect("issue under capacity");
        }
        let status = store.issue(principal("bob"), None).unwrap_err();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
        assert_eq!(store.len(), CAPACITY, "live tokens must survive the sweep");
    }
}
