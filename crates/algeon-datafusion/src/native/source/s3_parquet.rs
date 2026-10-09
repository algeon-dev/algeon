use std::{
    num::NonZeroU64,
    path::{Path, PathBuf},
};

use cudf::io::parquet::{RemoteParquetBackendBuildStatus, remote_parquet_backend_build_status};
use query_engine::exec::sanitize_cache_component;
use query_planning::{
    source::{
        CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, ResolvedParquetSource,
    },
    util::fnv1a64,
};

use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::{LakehouseByteCache, LakehouseByteCacheFill, S3SourceStorageProfile},
};

/// The id under which the configured S3 storage is registered as a runtime
/// storage profile and referenced by every resolved `s3://` source.
pub(crate) const S3_STORAGE_PROFILE_ID: &str = "s3";

/// Standard AWS SDK S3 endpoint override env var.
const AWS_ENDPOINT_URL_S3_ENV: &str = "AWS_ENDPOINT_URL_S3";
/// Standard AWS SDK endpoint override env var.
const AWS_ENDPOINT_URL_ENV: &str = "AWS_ENDPOINT_URL";

/// The endpoint the ambient AWS configuration selects, which the default
/// credential chain and process-wide KvikIO reads use.
pub(crate) fn ambient_s3_endpoint() -> Option<String> {
    [AWS_ENDPOINT_URL_S3_ENV, AWS_ENDPOINT_URL_ENV].into_iter().find_map(|name| {
        std::env::var(name).ok().map(|value| value.trim().to_string()).filter(|value| !value.is_empty())
    })
}

/// Foreground lakehouse cache read behavior for complete local cache entries.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
enum LakehouseCacheReadPolicy {
    /// Use complete local cache hits for foreground reads.
    #[default]
    Enabled,
    /// Ignore complete local cache hits for benchmark isolation.
    Disabled,
}

impl LakehouseCacheReadPolicy {
    /// Whether cache hits should be considered for foreground reads.
    const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Remote Parquet read-strategy resolver configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParquetReadStrategyConfig {
    /// Optional lakehouse byte-cache root.
    lakehouse_cache_root: Option<PathBuf>,
    /// Whether complete cache hits may be used for foreground reads.
    lakehouse_cache_read_policy: LakehouseCacheReadPolicy,
    /// Cache-miss policy for local lakehouse byte-cache entries.
    lakehouse_cache_policy: LakehouseByteCacheFill,
    /// Best-effort retained byte target for the lakehouse byte-cache root.
    lakehouse_cache_max_bytes: Option<u64>,
    /// Optional S3 endpoint identity used to scope local-cache entries.
    ///
    /// This is endpoint metadata only. Userinfo, query strings, and fragments
    /// are stripped before the value is stored or exposed in diagnostics.
    storage_endpoint: Option<String>,
    /// Execution-time native source storage profile id.
    ///
    /// This id references a source storage profile registered on the native
    /// execution config. It is not a credential and is safe for diagnostics.
    storage_profile_id: Option<String>,
    /// Credential mode used for remote object-store reads.
    credential_mode: CredentialMode,
}

impl Default for ParquetReadStrategyConfig {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

impl ParquetReadStrategyConfig {
    /// The read strategy for one execution config: the optional byte cache,
    /// the optional S3 storage, and the ambient endpoint that scopes cache
    /// entries when the storage names none.
    pub(crate) fn new(
        byte_cache: Option<&LakehouseByteCache>,
        s3_storage: Option<&S3SourceStorageProfile>,
        ambient_endpoint: Option<&str>,
    ) -> Self {
        let storage_endpoint = s3_storage
            .and_then(|profile| profile.endpoint.as_deref())
            .filter(|endpoint| !endpoint.trim().is_empty())
            .or(ambient_endpoint)
            .map(credential_free_endpoint_identity);
        let credential_mode = if s3_storage.is_some_and(S3SourceStorageProfile::requires_scoped_remote_direct_profile) {
            CredentialMode::ScopedResolved
        } else {
            CredentialMode::SharedProcessEnv
        };
        Self {
            lakehouse_cache_root: byte_cache.map(|cache| cache.root().to_path_buf()),
            lakehouse_cache_read_policy: if byte_cache.is_none_or(LakehouseByteCache::serves_hits) {
                LakehouseCacheReadPolicy::Enabled
            } else {
                LakehouseCacheReadPolicy::Disabled
            },
            lakehouse_cache_policy: byte_cache.map_or(LakehouseByteCacheFill::Disabled, LakehouseByteCache::fill),
            lakehouse_cache_max_bytes: byte_cache.and_then(LakehouseByteCache::max_retained_bytes).map(NonZeroU64::get),
            storage_endpoint,
            storage_profile_id: s3_storage.map(|_| S3_STORAGE_PROFILE_ID.to_string()),
            credential_mode,
        }
    }
}

/// Immutable object metadata used when resolving a remote Parquet source.
///
/// The resolver uses these facts to scope lakehouse byte-cache entries. The
/// identity is a diagnostic/cache key only and must not contain secrets,
/// signed query strings, bearer tokens, or credential material.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ParquetSourceMetadata {
    /// Object size in bytes, when known.
    pub(super) size_bytes: Option<u64>,
    /// Optional caller-provided content identity such as an object ETag or version.
    ///
    /// The lakehouse byte cache only serves or fills entries for sources with a
    /// non-empty identity here. Size alone never identifies an entry: an object
    /// rewritten in place with the same size must not read back stale bytes.
    pub(super) content_identity: Option<String>,
}

/// Resolves an `s3://` Parquet object location into the native reader contract.
pub(super) fn resolve_parquet_source_with_metadata(
    location: &str,
    config: &ParquetReadStrategyConfig,
    metadata: &ParquetSourceMetadata,
) -> Result<ResolvedParquetSource> {
    let resolved = resolve_s3_parquet_source(location, config, metadata)?;
    Ok(attach_remote_source_identity(resolved, location, config, metadata))
}

fn resolve_s3_parquet_source(
    location: &str,
    config: &ParquetReadStrategyConfig,
    metadata: &ParquetSourceMetadata,
) -> Result<ResolvedParquetSource> {
    if let Some(cache_root) = config.lakehouse_cache_root.as_ref() {
        return resolve_s3_parquet_source_with_cache_root(location, config, metadata, cache_root);
    }

    remote_kvikio_source(location, config)
}

fn resolve_s3_parquet_source_with_cache_root(
    location: &str,
    config: &ParquetReadStrategyConfig,
    metadata: &ParquetSourceMetadata,
    cache_root: &Path,
) -> Result<ResolvedParquetSource> {
    let cache_path = cache_root.join(cache_relative_path(location, config, metadata));
    // Without a caller-provided content identity (object ETag/version) the cached
    // bytes cannot be attributed to the current object, so the cache is neither
    // served nor filled: the foreground read goes remote.
    let cache_identity_allowed = strong_content_identity(metadata).is_some();
    let fill_mode_for_unidentified = LakehouseCacheFillMode::Disabled;
    if !config.lakehouse_cache_read_policy.is_enabled() {
        let resolved = remote_kvikio_source(location, config)?;
        return Ok(resolved
            .with_cache_status(LakehouseCacheStatus::Bypassed)
            .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Bypassed)
            .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
            .with_lakehouse_cache_path(cache_path.display().to_string())
            .with_lakehouse_cache_fill_mode(if cache_identity_allowed {
                background_fill_mode(config)
            } else {
                fill_mode_for_unidentified
            }));
    }
    if complete_cache_hit_matches_metadata(&cache_path, metadata) {
        return Ok(ResolvedParquetSource::local_cache_kvikio(
            location,
            cache_path.display().to_string(),
            config.credential_mode,
            LakehouseCacheStatus::Hit,
        )
        .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
        .with_lakehouse_cache_path(cache_path.display().to_string()));
    }
    if cache_identity_allowed && config.lakehouse_cache_policy == LakehouseByteCacheFill::FillBeforeRead {
        return Ok(ResolvedParquetSource::local_cache_kvikio(
            location,
            cache_path.display().to_string(),
            config.credential_mode,
            LakehouseCacheStatus::FillBeforeRead,
        )
        .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
        .with_lakehouse_cache_path(cache_path.display().to_string())
        .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Blocking));
    }
    let fill_mode = if cache_identity_allowed { background_fill_mode(config) } else { fill_mode_for_unidentified };
    let resolved = remote_kvikio_source(location, config)?
        .with_cache_status(LakehouseCacheStatus::Miss)
        .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Enabled)
        .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
        .with_lakehouse_cache_path(cache_path.display().to_string())
        .with_lakehouse_cache_fill_mode(fill_mode);
    Ok(resolved)
}

fn remote_kvikio_source(location: &str, config: &ParquetReadStrategyConfig) -> Result<ResolvedParquetSource> {
    match remote_parquet_backend_build_status() {
        RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled => {},
        RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled => {
            return reject_remote_kvikio_source("remote_kvikio_backend_disabled");
        },
        RemoteParquetBackendBuildStatus::Unknown => {
            return reject_remote_kvikio_source("remote_kvikio_backend_unknown");
        },
    }
    Ok(ResolvedParquetSource::remote_kvikio(location, config.credential_mode))
}

fn reject_remote_kvikio_source(unavailable_reason: &'static str) -> Result<ResolvedParquetSource> {
    Err(Error::new(
        ErrorCode::InvalidExecutionPlan,
        "algeon_datafusion.native.resolve_parquet_source",
        "native GPU queries require the remote KvikIO source contract",
    )
    .with_fact("reason", "remote_kvikio_unavailable")
    .with_fact("remote_kvikio_unavailable_reason", unavailable_reason)
    .with_fact("remote_read_mode", "require_kvikio"))
}

fn background_fill_mode(config: &ParquetReadStrategyConfig) -> LakehouseCacheFillMode {
    match config.lakehouse_cache_policy {
        LakehouseByteCacheFill::ReadThrough => LakehouseCacheFillMode::Background,
        LakehouseByteCacheFill::Disabled | LakehouseByteCacheFill::FillBeforeRead => LakehouseCacheFillMode::Disabled,
    }
}

fn attach_remote_source_identity(
    resolved: ResolvedParquetSource,
    location: &str,
    config: &ParquetReadStrategyConfig,
    metadata: &ParquetSourceMetadata,
) -> ResolvedParquetSource {
    let resolved = resolved
        .with_storage_domain(storage_domain_identity(location, config))
        .with_optional_storage_profile_id(config.storage_profile_id.clone());
    if let Some(content_identity) = content_identity(metadata) {
        return resolved.with_content_identity(content_identity);
    }
    resolved
}

fn cache_relative_path(
    location: &str,
    config: &ParquetReadStrategyConfig,
    metadata: &ParquetSourceMetadata,
) -> PathBuf {
    let scheme = location.split_once("://").map(|(scheme, _)| scheme).unwrap_or("unknown");
    PathBuf::from("v1")
        .join(sanitize_cache_component(scheme))
        .join(sanitize_cache_component(&storage_domain_identity(location, config)))
        .join(sanitize_cache_component(content_identity(metadata).as_deref().unwrap_or("content_unidentified")))
        .join(location_fingerprint(location))
}

fn storage_domain_identity(location: &str, config: &ParquetReadStrategyConfig) -> String {
    if let Some(rest) = location.strip_prefix("s3://") {
        let bucket = rest.split('/').next().filter(|bucket| !bucket.is_empty()).unwrap_or("unknown_bucket");
        return config.storage_endpoint.as_ref().filter(|value| !value.trim().is_empty()).map_or_else(
            || bucket.to_string(),
            |endpoint| format!("{bucket}@{}", credential_free_endpoint_identity(endpoint)),
        );
    }
    "unknown_storage_domain".to_string()
}

/// Returns a credential-free endpoint identity for cache scoping and diagnostics.
///
/// Userinfo, query strings, and fragments are stripped so signed URLs and
/// embedded credentials cannot leak through source facts.
fn credential_free_endpoint_identity(endpoint: &str) -> String {
    let trimmed = endpoint.trim();
    let without_query = trimmed.find(['?', '#']).map_or(trimmed, |index| &trimmed[..index]).trim_end_matches('/');
    if let Some((scheme, rest)) = without_query.split_once("://") {
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        let authority = authority.rsplit_once('@').map_or(authority, |(_, authority)| authority);
        if path.is_empty() { format!("{scheme}://{authority}") } else { format!("{scheme}://{authority}/{path}") }
    } else {
        let (authority, path) = without_query.split_once('/').unwrap_or((without_query, ""));
        let authority = authority.rsplit_once('@').map_or(authority, |(_, authority)| authority);
        if path.is_empty() { authority.to_string() } else { format!("{authority}/{path}") }
    }
}

/// Caller-provided content identity (object ETag/version) used for cache scoping.
///
/// Returns `None` when the caller supplied no usable identity. Size is
/// deliberately not a fallback. A same-size rewrite must map to no servable
/// entry, and a stale hit must not be served.
fn content_identity(metadata: &ParquetSourceMetadata) -> Option<String> {
    strong_content_identity(metadata).map(str::to_string)
}

fn strong_content_identity(metadata: &ParquetSourceMetadata) -> Option<&str> {
    metadata.content_identity.as_deref().filter(|value| !value.is_empty())
}

fn complete_cache_hit_matches_metadata(cache_path: &Path, metadata: &ParquetSourceMetadata) -> bool {
    if strong_content_identity(metadata).is_none() {
        return false;
    }
    let Ok(file_metadata) = cache_path.metadata() else {
        return false;
    };
    if !file_metadata.is_file() {
        return false;
    }
    let Some(expected_size) = metadata.size_bytes else {
        return false;
    };
    file_metadata.len() == expected_size
}

fn location_fingerprint(location: &str) -> String {
    let safe_location = credential_free_endpoint_identity(location);
    let basename = safe_location.rsplit('/').next().filter(|value| !value.is_empty()).unwrap_or("object");
    format!("{}__h{:016x}", sanitize_cache_component(basename), fnv1a64(safe_location.as_bytes()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use query_planning::source::ResolvedParquetInput;

    use super::*;

    #[test]
    fn s3_storage_endpoint_scopes_the_cache_without_credentials() {
        let storage = S3SourceStorageProfile::default_chain()
            .with_endpoint("http://user:secret@rustfs.local:9000/minio?sig=secret");
        let config = ParquetReadStrategyConfig::new(None, Some(&storage), Some("https://aws.example.invalid"));

        assert_eq!(config.storage_endpoint.as_deref(), Some("http://rustfs.local:9000/minio"));
        assert_eq!(config.storage_profile_id.as_deref(), Some(S3_STORAGE_PROFILE_ID));
        assert_eq!(config.credential_mode, CredentialMode::ScopedResolved);
    }

    #[test]
    fn s3_storage_credentials_and_region_are_scoped_to_the_profile() {
        let credentials = S3SourceStorageProfile::default_chain().with_static_credentials("access", "secret", None);
        let region = S3SourceStorageProfile::default_chain().with_region("us-west-2");
        for storage in [credentials, region] {
            let config = ParquetReadStrategyConfig::new(None, Some(&storage), None);

            assert_eq!(config.storage_profile_id.as_deref(), Some(S3_STORAGE_PROFILE_ID));
            assert_eq!(config.credential_mode, CredentialMode::ScopedResolved);
            let result = resolve_parquet_source_with_metadata(
                "s3://bucket/table/part-0.parquet",
                &config,
                &ParquetSourceMetadata::default(),
            );
            if let Some(source) = assert_expected_remote_foreground_source(result) {
                assert_eq!(source.storage_profile_id(), Some(S3_STORAGE_PROFILE_ID));
            }
        }
    }

    #[test]
    fn default_chain_storage_keeps_the_ambient_endpoint_and_process_credentials() {
        let storage = S3SourceStorageProfile::default_chain();
        let config = ParquetReadStrategyConfig::new(None, Some(&storage), Some("https://aws.example.invalid/"));

        assert_eq!(config.storage_endpoint.as_deref(), Some("https://aws.example.invalid"));
        assert_eq!(config.storage_profile_id.as_deref(), Some(S3_STORAGE_PROFILE_ID));
        assert_eq!(config.credential_mode, CredentialMode::SharedProcessEnv);
        assert_eq!(ParquetReadStrategyConfig::new(None, None, None).storage_profile_id, None);
    }

    #[test]
    fn content_identity_requires_caller_provided_identity() {
        assert_eq!(content_identity(&ParquetSourceMetadata { size_bytes: Some(4096), content_identity: None }), None);
        assert_eq!(content_identity(&ParquetSourceMetadata::default()), None);
        assert_eq!(
            content_identity(&ParquetSourceMetadata { size_bytes: None, content_identity: Some(String::new()) }),
            None,
        );
        assert_eq!(content_identity(&metadata_with_identity("etag:abc123")), Some("etag:abc123".to_string()),);
    }

    #[test]
    fn location_fingerprint_uses_credential_free_identity() {
        let fingerprint = location_fingerprint(
            "s3://secret-bucket/table/data.parquet?X-Amz-Signature=do-not-leak&X-Amz-Credential=also-secret",
        );

        assert!(fingerprint.starts_with("data.parquet__h"));
        assert!(!fingerprint.contains("secret-bucket"));
        assert!(!fingerprint.contains("X-Amz-Signature"));
        assert!(!fingerprint.contains("X-Amz-Credential"));
        assert!(!fingerprint.contains("do-not-leak"));
        assert!(!fingerprint.contains("also-secret"));
    }

    fn remote_direct_backend_disabled() -> bool {
        remote_parquet_backend_build_status() == RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled
    }

    fn assert_expected_remote_foreground_source(
        result: Result<ResolvedParquetSource>,
    ) -> Option<ResolvedParquetSource> {
        if remote_direct_backend_disabled() {
            let error = result.unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
            assert_eq!(error.fact_value("reason"), Some("remote_kvikio_unavailable"));
            assert_eq!(error.fact_value("remote_kvikio_unavailable_reason"), Some("remote_kvikio_backend_disabled"));
            None
        } else {
            let source = result.unwrap();
            assert_eq!(source.read_strategy().as_str(), "remote_direct");
            assert!(matches!(&source.input, ResolvedParquetInput::RemoteKvikio { .. }));
            Some(source)
        }
    }

    #[test]
    fn s3_auto_uses_remote_direct_on_cache_miss() {
        let root = tempfile::tempdir().unwrap();
        let config =
            ParquetReadStrategyConfig { lakehouse_cache_root: Some(root.path().to_path_buf()), ..Default::default() };

        let result = resolve_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &config,
            &ParquetSourceMetadata { size_bytes: Some(4096), content_identity: None },
        );
        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Disabled);
        assert!(source.lakehouse_cache_path.is_some());
    }

    #[test]
    fn scoped_profile_remote_direct_reaches_direct_path_when_backend_available() {
        let config =
            ParquetReadStrategyConfig { credential_mode: CredentialMode::ScopedResolved, ..Default::default() };

        let result = resolve_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &config,
            &ParquetSourceMetadata::default(),
        );
        if remote_direct_backend_disabled() {
            let error = result.unwrap_err();
            assert_eq!(error.fact_value("reason"), Some("remote_kvikio_unavailable"));
            assert_eq!(error.fact_value("remote_kvikio_unavailable_reason"), Some("remote_kvikio_backend_disabled"));
        } else {
            let source = result.unwrap();
            assert_eq!(source.read_strategy().as_str(), "remote_direct");
            assert!(matches!(
                &source.input,
                ResolvedParquetInput::RemoteKvikio { credential_mode: CredentialMode::ScopedResolved, .. }
            ));
        }
    }

    #[test]
    fn s3_auto_prefers_complete_local_cache_hit() {
        let root = tempfile::tempdir().unwrap();
        let config =
            ParquetReadStrategyConfig { lakehouse_cache_root: Some(root.path().to_path_buf()), ..Default::default() };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let source =
            resolve_parquet_source_with_metadata("s3://bucket/table/part-0.parquet", &config, &metadata).unwrap();

        assert_eq!(source.read_strategy().as_str(), "local_cache_kvikio");
        assert!(matches!(&source.input, ResolvedParquetInput::LocalKvikio { .. }));
        assert_eq!(source.cache_status, LakehouseCacheStatus::Hit);
        assert_eq!(source.reader_location(), cache_path.display().to_string());
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn s3_auto_ignores_complete_local_cache_file_with_wrong_size() {
        let root = tempfile::tempdir().unwrap();
        let config =
            ParquetReadStrategyConfig { lakehouse_cache_root: Some(root.path().to_path_buf()), ..Default::default() };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, b"incomplete").unwrap();

        let result = resolve_parquet_source_with_metadata("s3://bucket/table/part-0.parquet", &config, &metadata);
        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn s3_auto_ignores_local_cache_file_when_size_metadata_is_missing() {
        let root = tempfile::tempdir().unwrap();
        let config =
            ParquetReadStrategyConfig { lakehouse_cache_root: Some(root.path().to_path_buf()), ..Default::default() };
        let metadata = ParquetSourceMetadata { size_bytes: None, content_identity: Some("etag:abc123".to_string()) };
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let result = resolve_parquet_source_with_metadata("s3://bucket/table/part-0.parquet", &config, &metadata);
        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn explicit_local_cache_read_bypass_skips_existing_local_cache_hit() {
        let root = tempfile::tempdir().unwrap();
        let config = ParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            lakehouse_cache_read_policy: LakehouseCacheReadPolicy::Disabled,
            lakehouse_cache_policy: LakehouseByteCacheFill::ReadThrough,
            ..Default::default()
        };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let result = resolve_parquet_source_with_metadata("s3://bucket/table/part-0.parquet", &config, &metadata);

        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Bypassed);
        assert_eq!(source.reader_location(), "s3://bucket/table/part-0.parquet");
        assert_eq!(source.lakehouse_cache_read_mode, LakehouseCacheReadMode::Bypassed);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Background);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn read_through_cache_miss_selects_background_fill_without_blocking_foreground_read() {
        let root = tempfile::tempdir().unwrap();
        let config = ParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            lakehouse_cache_policy: LakehouseByteCacheFill::ReadThrough,
            ..Default::default()
        };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));

        let result = resolve_parquet_source_with_metadata("s3://bucket/table/part-0.parquet", &config, &metadata);

        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Background);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    fn metadata_with_identity(identity: &str) -> ParquetSourceMetadata {
        ParquetSourceMetadata { size_bytes: Some(4096), content_identity: Some(identity.to_string()) }
    }

    #[test]
    fn s3_same_size_rewrite_with_new_identity_is_not_a_hit() {
        let root = tempfile::tempdir().unwrap();
        let config =
            ParquetReadStrategyConfig { lakehouse_cache_root: Some(root.path().to_path_buf()), ..Default::default() };
        let old_metadata = metadata_with_identity("etag:old");
        let old_cache_path =
            root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &old_metadata));
        fs::create_dir_all(old_cache_path.parent().unwrap()).unwrap();
        fs::write(&old_cache_path, vec![0_u8; 4096]).unwrap();

        // Same size, new ETag after an in-place rewrite: the old bytes must not be served.
        let new_metadata = metadata_with_identity("etag:new");
        let new_cache_path =
            root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &new_metadata));
        assert_ne!(new_cache_path, old_cache_path);
        let result = resolve_parquet_source_with_metadata("s3://bucket/table/part-0.parquet", &config, &new_metadata);
        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(new_cache_path.to_str().unwrap()));
    }

    #[test]
    fn s3_without_content_identity_neither_serves_nor_fills() {
        for policy in [LakehouseByteCacheFill::ReadThrough, LakehouseByteCacheFill::FillBeforeRead] {
            let root = tempfile::tempdir().unwrap();
            let config = ParquetReadStrategyConfig {
                lakehouse_cache_root: Some(root.path().to_path_buf()),
                lakehouse_cache_policy: policy,
                ..Default::default()
            };
            let metadata = ParquetSourceMetadata { size_bytes: Some(4096), content_identity: None };
            // Even with bytes planted at the unidentified path, they must not be served.
            let cache_path =
                root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
            assert!(cache_path.display().to_string().contains("content_unidentified"));
            fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
            fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

            let result = resolve_parquet_source_with_metadata("s3://bucket/table/part-0.parquet", &config, &metadata);
            let Some(source) = assert_expected_remote_foreground_source(result) else {
                return;
            };
            assert_eq!(source.cache_status, LakehouseCacheStatus::Miss, "policy={policy:?}");
            assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Disabled, "policy={policy:?}");
            assert_eq!(source.content_identity, None, "policy={policy:?}");
        }
    }
}
