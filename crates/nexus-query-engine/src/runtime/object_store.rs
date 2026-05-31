//! Object-store source fetch handles and shared runtime registry.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use bytes::Bytes;

use crate::error::{QueryError, QueryErrorCode, Result};

/// Storage profile used by native source fetches.
///
/// The profile is format-neutral: callers such as Iceberg REST/Glue translate
/// catalog/FileIO settings into this shape before execution. Secrets are held
/// only as object-store builder inputs and are redacted from [`Debug`].
#[derive(Clone, PartialEq, Eq)]
pub struct SourceStorageProfile {
    id: String,
    s3: S3SourceStorageProfile,
}

impl SourceStorageProfile {
    /// Builds an S3 storage profile with a stable non-secret identifier.
    pub fn s3(id: impl Into<String>, s3: S3SourceStorageProfile) -> Self {
        Self { id: id.into(), s3 }
    }

    /// Stable profile identifier referenced by resolved source contracts.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// S3-specific source storage settings.
    pub fn s3_profile(&self) -> &S3SourceStorageProfile {
        &self.s3
    }
}

impl fmt::Debug for SourceStorageProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceStorageProfile").field("id", &self.id).field("s3", &self.s3).finish()
    }
}

/// S3-compatible storage settings for source object-store fetches.
#[derive(Clone, PartialEq, Eq)]
pub struct S3SourceStorageProfile {
    /// Region passed to the S3 client. When absent, the parsed source/fallback
    /// region is used.
    pub region: Option<String>,
    /// Optional S3-compatible endpoint such as `http://localhost:9000`.
    pub endpoint: Option<String>,
    /// Optional virtual-hosted addressing flag. Set to `Some(false)` for
    /// path-style S3-compatible stores such as RustFS/MinIO.
    pub virtual_hosted_style_request: Option<bool>,
    /// Optional HTTP allowance. If unset, `http://` endpoints imply `true`.
    pub allow_http: Option<bool>,
    /// Credential source used by this profile.
    pub credential: S3SourceStorageCredential,
}

impl S3SourceStorageProfile {
    /// Builds an S3 profile that uses the object-store crate's default
    /// credential/config chain unless fields are overridden.
    pub fn default_chain() -> Self {
        Self {
            region: None,
            endpoint: None,
            virtual_hosted_style_request: None,
            allow_http: None,
            credential: S3SourceStorageCredential::DefaultChain,
        }
    }

    /// Overrides the S3 region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Overrides the S3 endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Selects path-style or virtual-hosted-style request addressing.
    pub fn with_path_style_access(mut self, path_style: bool) -> Self {
        self.virtual_hosted_style_request = Some(!path_style);
        self
    }

    /// Allows or rejects HTTP endpoints.
    pub fn with_allow_http(mut self, allow_http: bool) -> Self {
        self.allow_http = Some(allow_http);
        self
    }

    /// Uses explicit static credentials for this profile.
    pub fn with_static_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        self.credential = S3SourceStorageCredential::Static {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
        };
        self
    }
}

impl fmt::Debug for S3SourceStorageProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3SourceStorageProfile")
            .field("region", &self.region)
            .field("endpoint", &self.endpoint.as_ref().map(|_| "<configured>"))
            .field("virtual_hosted_style_request", &self.virtual_hosted_style_request)
            .field("allow_http", &self.allow_http)
            .field("credential", &self.credential)
            .finish()
    }
}

/// Credential source for an S3 source storage profile.
#[derive(Clone, PartialEq, Eq)]
pub enum S3SourceStorageCredential {
    /// Use the object-store crate's default AWS credential/config chain.
    DefaultChain,
    /// Use explicit static credentials supplied by the adapter/catalog layer.
    Static {
        /// Access key id. Redacted from [`Debug`] to keep logs uniformly secret-free.
        access_key_id: String,
        /// Secret access key. Never render in diagnostics.
        secret_access_key: String,
        /// Optional session token. Never render in diagnostics.
        session_token: Option<String>,
    },
}

impl fmt::Debug for S3SourceStorageCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DefaultChain => f.write_str("DefaultChain"),
            Self::Static { session_token, .. } => f
                .debug_struct("Static")
                .field("access_key_id", &"<redacted>")
                .field("secret_access_key", &"<redacted>")
                .field("session_token", &session_token.as_ref().map(|_| "<redacted>"))
                .finish(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct S3ObjectStoreKey {
    pub(crate) bucket: String,
    pub(crate) region: String,
    pub(crate) storage_profile_id: Option<String>,
}

pub(crate) struct ObjectStoreFetchResult {
    pub(crate) bytes: Result<Bytes>,
    pub(crate) elapsed: Duration,
}

pub(crate) struct ObjectStoreCustomFetchResult<T> {
    pub(crate) value: Result<T>,
    pub(crate) elapsed: Duration,
}

pub(super) struct SourceObjectStoreRegistry {
    profiles: HashMap<String, SourceStorageProfile>,
    stores: Mutex<HashMap<S3ObjectStoreKey, Arc<dyn object_store::ObjectStore>>>,
}

impl SourceObjectStoreRegistry {
    pub(super) fn new() -> Self {
        Self { profiles: HashMap::new(), stores: Mutex::new(HashMap::new()) }
    }

    pub(super) fn with_profiles(profiles: impl IntoIterator<Item = SourceStorageProfile>) -> Result<Self> {
        let mut by_id = HashMap::new();
        for profile in profiles {
            if profile.id().trim().is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.source_object_store.profile",
                    "source storage profile id must not be empty",
                )
                .with_fact("reason", "empty_profile_id"));
            }
            let id = profile.id().to_string();
            if by_id.insert(id.clone(), profile).is_some() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.source_object_store.profile",
                    "source storage profile ids must be unique",
                )
                .with_fact("storage_profile_id", id)
                .with_fact("reason", "duplicate_profile_id"));
            }
        }
        Ok(Self { profiles: by_id, stores: Mutex::new(HashMap::new()) })
    }

    pub(super) fn s3_store(&self, key: S3ObjectStoreKey) -> Result<Arc<dyn object_store::ObjectStore>> {
        let mut stores = self.stores.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.source_object_store.registry",
                "source object-store registry mutex was poisoned",
            )
        })?;
        if let Some(store) = stores.get(&key) {
            return Ok(Arc::clone(store));
        }
        let profile = key
            .storage_profile_id
            .as_ref()
            .map(|id| {
                self.profiles.get(id).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.source_object_store.profile",
                        "source storage profile id was not registered",
                    )
                    .with_fact("storage_profile_id", id.as_str())
                })
            })
            .transpose()?;
        let store = build_s3_store(&key, profile)?;
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(store);
        stores.insert(key, Arc::clone(&store));
        Ok(store)
    }
}

fn build_s3_store(
    key: &S3ObjectStoreKey,
    profile: Option<&SourceStorageProfile>,
) -> Result<object_store::aws::AmazonS3> {
    let s3_profile = profile.map(SourceStorageProfile::s3_profile);
    let region = s3_profile
        .and_then(|profile| profile.region.as_deref())
        .filter(|region| !region.is_empty())
        .unwrap_or(key.region.as_str());
    let mut builder = match s3_profile.map(|profile| &profile.credential) {
        Some(S3SourceStorageCredential::Static { access_key_id, secret_access_key, session_token }) => {
            let mut builder = object_store::aws::AmazonS3Builder::new()
                .with_access_key_id(access_key_id)
                .with_secret_access_key(secret_access_key);
            if let Some(token) = session_token.as_ref().filter(|token| !token.is_empty()) {
                builder = builder.with_token(token);
            }
            builder
        },
        _ => object_store::aws::AmazonS3Builder::from_env(),
    }
    .with_bucket_name(key.bucket.as_str())
    .with_region(region);
    if let Some(profile) = s3_profile {
        if let Some(endpoint) = profile.endpoint.as_ref().filter(|endpoint| !endpoint.trim().is_empty()) {
            builder = builder.with_endpoint(endpoint);
            if profile.allow_http.unwrap_or_else(|| endpoint.trim_start().starts_with("http://")) {
                builder = builder.with_allow_http(true);
            }
        } else if let Some(allow_http) = profile.allow_http {
            builder = builder.with_allow_http(allow_http);
        }
        if let Some(virtual_hosted_style_request) = profile.virtual_hosted_style_request {
            builder = builder.with_virtual_hosted_style_request(virtual_hosted_style_request);
        }
    }
    builder.build().map_err(|error| {
        let mut query_error = QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_object_store.s3",
            "failed to build S3 object-store client for native source fetch",
        )
        .with_fact("bucket", key.bucket.as_str())
        .with_fact("region", region);
        if let Some(profile_id) = key.storage_profile_id.as_deref() {
            query_error = query_error.with_fact("storage_profile_id", profile_id);
        }
        query_error.with_source(error)
    })
}

impl fmt::Debug for SourceObjectStoreRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cached_store_count = self.stores.lock().map_or(0, |stores| stores.len());
        f.debug_struct("SourceObjectStoreRegistry")
            .field("profile_count", &self.profiles.len())
            .field("cached_store_count", &cached_store_count)
            .finish()
    }
}

pub(super) fn default_source_fetch_runtime() -> Arc<tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
    Arc::clone(RUNTIME.get_or_init(|| {
        Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .thread_name("cudf-source-fetch")
                .enable_all()
                .build()
                .expect("source fetch runtime should initialize"),
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_storage_profile_debug_redacts_endpoint_and_credentials() {
        let profile = SourceStorageProfile::s3(
            "rest-s3",
            S3SourceStorageProfile::default_chain()
                .with_region("us-east-1")
                .with_endpoint("http://user:password@localhost:9000")
                .with_path_style_access(true)
                .with_static_credentials(
                    "access-key-value",
                    "secret-key-value",
                    Some("session-token-value".to_string()),
                ),
        );

        let rendered = format!("{profile:?}");

        assert!(rendered.contains("rest-s3"));
        assert!(rendered.contains("<configured>"));
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("user:password"));
        assert!(!rendered.contains("access-key-value"));
        assert!(!rendered.contains("secret-key-value"));
        assert!(!rendered.contains("session-token-value"));
    }

    #[test]
    fn source_object_store_registry_rejects_duplicate_profile_ids() {
        let profile = SourceStorageProfile::s3("same", S3SourceStorageProfile::default_chain());
        let error = SourceObjectStoreRegistry::with_profiles([profile.clone(), profile]).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reason"), Some("duplicate_profile_id"));
        assert_eq!(error.fact_value("storage_profile_id"), Some("same"));
    }

    #[test]
    fn source_object_store_registry_rejects_empty_profile_ids() {
        let profile = SourceStorageProfile::s3("  ", S3SourceStorageProfile::default_chain());
        let error = SourceObjectStoreRegistry::with_profiles([profile]).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reason"), Some("empty_profile_id"));
    }

    #[test]
    fn source_object_store_registry_requires_registered_profile_id() {
        let error = SourceObjectStoreRegistry::new()
            .s3_store(S3ObjectStoreKey {
                bucket: "bucket".to_string(),
                region: "us-east-1".to_string(),
                storage_profile_id: Some("missing".to_string()),
            })
            .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("storage_profile_id"), Some("missing"));
    }
}
