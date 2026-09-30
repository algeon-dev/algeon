//! Keyed source-location identities for retained topology.

use crate::observability::RuntimeSourceLocationId;

pub(super) trait SourceLocationDigestKeyProvider {
    fn source_location_digest_key(&self) -> Result<SourceLocationDigestKey, SourceLocationDigestKeyUnavailable>;
}

pub(super) struct OsSourceLocationDigestKeyProvider;

impl SourceLocationDigestKeyProvider for OsSourceLocationDigestKeyProvider {
    fn source_location_digest_key(&self) -> Result<SourceLocationDigestKey, SourceLocationDigestKeyUnavailable> {
        let mut key = [0_u8; blake3::KEY_LEN];
        getrandom::fill(&mut key).map_err(|_| SourceLocationDigestKeyUnavailable)?;
        Ok(SourceLocationDigestKey(key))
    }
}

pub(super) struct SourceLocationDigestKeyUnavailable;

pub(super) enum SourceLocationDigestKeyState {
    Available(SourceLocationDigestKey),
    Unavailable,
}

pub(super) struct SourceLocationDigestKey([u8; blake3::KEY_LEN]);

impl SourceLocationDigestKey {
    fn source_location_id(&self, value: &str) -> RuntimeSourceLocationId {
        let mut hasher = blake3::Hasher::new_keyed(&self.0);
        hasher.update(b"query-engine/runtime-source-location/v2\0");
        hasher.update(value.as_bytes());
        RuntimeSourceLocationId::from_keyed_digest(format!(
            "source_location:blake3_keyed_256:{}",
            hasher.finalize().to_hex()
        ))
    }
}

impl SourceLocationDigestKeyState {
    pub(super) fn source_location_id(&self, value: &str) -> Option<RuntimeSourceLocationId> {
        match self {
            Self::Available(key) => Some(key.source_location_id(value)),
            Self::Unavailable => None,
        }
    }
}

#[cfg(any(test, feature = "test-hooks"))]
pub(super) struct TestSourceLocationDigestKeyProvider {
    pub(super) key: Option<[u8; blake3::KEY_LEN]>,
}

#[cfg(any(test, feature = "test-hooks"))]
impl SourceLocationDigestKeyProvider for TestSourceLocationDigestKeyProvider {
    fn source_location_digest_key(&self) -> Result<SourceLocationDigestKey, SourceLocationDigestKeyUnavailable> {
        self.key.map(SourceLocationDigestKey).ok_or(SourceLocationDigestKeyUnavailable)
    }
}
