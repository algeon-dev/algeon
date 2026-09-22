#[cfg(test)]
use std::sync::Arc;
use std::{
    hash::Hasher,
    path::{Path, PathBuf},
};

use cudf::io::parquet::{RemoteParquetBackendBuildStatus, remote_parquet_backend_build_status};
use fnv::FnvHasher;

#[cfg(test)]
use super::{DataType, Schema};
use super::{
    LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, ResolvedParquetSource, Result,
    TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyConfig,
};
use crate::error::{Error, ErrorCode};

/// Read capability required for a table-format data file location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatReadCapability {
    /// File is readable through the local filesystem.
    LocalFileSystem,
    /// File requires an object-store reader for the given scheme.
    ObjectStore {
        /// Object-store scheme such as `s3`.
        scheme: String,
    },
    /// File location cannot be represented by native source reads.
    Unsupported {
        /// Stable unsupported reason detail.
        reason: String,
    },
}

/// Immutable object metadata used when resolving a table-format Parquet source.
///
/// The resolver uses these facts to scope lakehouse byte-cache entries. The
/// identity fields are diagnostic/cache keys only and must not contain secrets,
/// signed query strings, bearer tokens, or credential material.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableFormatParquetSourceMetadata {
    /// File size in bytes from table-format metadata, when known.
    pub size_bytes: Option<u64>,
    /// File row count from table-format metadata, when known.
    pub row_count: Option<u64>,
    /// Optional caller-provided content identity such as an object ETag or version.
    ///
    /// The lakehouse byte cache only serves or fills entries for sources with a
    /// non-empty identity here. Size and row count alone never identify an entry:
    /// an object rewritten in place with the same size must not read back stale bytes.
    pub content_identity: Option<String>,
    /// Optional caller-provided storage domain such as an endpoint/bucket scope.
    pub storage_domain: Option<String>,
}

impl TableFormatParquetSourceMetadata {
    /// Builds metadata from optional size and row-count facts.
    pub fn from_size_and_row_count(size_bytes: Option<u64>, row_count: Option<u64>) -> Self {
        Self { size_bytes, row_count, ..Default::default() }
    }

    /// Records file size in bytes.
    pub fn with_size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    /// Records file row count.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    /// Records a caller-provided content identity such as an object ETag.
    pub fn with_content_identity(mut self, content_identity: impl Into<String>) -> Self {
        self.content_identity = Some(content_identity.into());
        self
    }

    /// Records a caller-provided storage domain such as an endpoint/bucket scope.
    pub fn with_storage_domain(mut self, storage_domain: impl Into<String>) -> Self {
        self.storage_domain = Some(storage_domain.into());
        self
    }
}

impl TableFormatReadCapability {
    /// Local filesystem read capability.
    pub fn local_file_system() -> Self {
        Self::LocalFileSystem
    }

    /// Object-store read capability for a scheme such as `s3`.
    pub fn object_store(scheme: impl Into<String>) -> Self {
        Self::ObjectStore { scheme: scheme.into() }
    }

    /// Unsupported read capability with a stable reason detail.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported { reason: reason.into() }
    }

    /// Stable capability code for diagnostics.
    pub fn code(&self) -> &str {
        match self {
            Self::LocalFileSystem => "local_file_system",
            Self::ObjectStore { scheme } if scheme == "s3" => "object_store:s3",
            Self::ObjectStore { .. } => "object_store",
            Self::Unsupported { .. } => "unsupported",
        }
    }

    /// Human-readable capability detail.
    pub fn detail(&self) -> &str {
        match self {
            Self::LocalFileSystem => "local_file_system",
            Self::ObjectStore { scheme } => scheme.as_str(),
            Self::Unsupported { reason } => reason.as_str(),
        }
    }

    /// Whether this capability can be represented as a native local Parquet source.
    pub fn is_native_local_parquet_capable(&self) -> bool {
        matches!(self, Self::LocalFileSystem)
    }

    /// Whether this capability can be represented by native Parquet source reads.
    ///
    /// Object-store S3 reads require the fail-closed remote KvikIO reader.
    pub fn is_native_parquet_capable(&self) -> bool {
        match self {
            Self::LocalFileSystem => true,
            Self::ObjectStore { scheme } => scheme == "s3",
            Self::Unsupported { .. } => false,
        }
    }

    /// Whether this capability can be represented by local direct-device Parquet reads.
    ///
    /// Remote S3 KvikIO reads are tracked as a separate source strategy and
    /// must not be confused with local direct-device eligibility.
    pub fn is_direct_native_parquet_capable(&self) -> bool {
        matches!(self, Self::LocalFileSystem)
    }
}

/// Resolves a table-format Parquet data-file location into the native reader contract.
pub fn resolve_table_format_parquet_source(
    location: &str,
    read_capability: &TableFormatReadCapability,
    config: &TableFormatParquetReadStrategyConfig,
) -> Result<ResolvedParquetSource> {
    resolve_table_format_parquet_source_with_metadata(
        location,
        read_capability,
        config,
        &TableFormatParquetSourceMetadata::default(),
    )
}

/// Resolves a table-format Parquet data-file location with object metadata.
pub fn resolve_table_format_parquet_source_with_metadata(
    location: &str,
    read_capability: &TableFormatReadCapability,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> Result<ResolvedParquetSource> {
    match read_capability {
        TableFormatReadCapability::LocalFileSystem => {
            Ok(ResolvedParquetSource::local_kvikio(table_format_local_path_text(location)))
        },
        TableFormatReadCapability::ObjectStore { scheme } if scheme == "s3" => {
            resolve_s3_table_format_parquet_source(location, config, metadata)
        },
        TableFormatReadCapability::ObjectStore { scheme } => Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.table_format.resolve_parquet_source",
            "unsupported table-format object-store parquet source scheme",
        )
        .with_fact("source_location_scheme", scheme.as_str())),
        TableFormatReadCapability::Unsupported { reason } => Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.table_format.resolve_parquet_source",
            "unsupported table-format parquet source location",
        )
        .with_fact("reason", reason.as_str())),
    }
}

fn resolve_s3_table_format_parquet_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> Result<ResolvedParquetSource> {
    let resolved = resolve_auto_s3_table_format_parquet_source(location, config, metadata)?;
    Ok(attach_remote_source_identity(resolved, location, config, metadata))
}

fn resolve_auto_s3_table_format_parquet_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> Result<ResolvedParquetSource> {
    if let Some(cache_root) = config.lakehouse_cache_root.as_ref() {
        return resolve_s3_table_format_parquet_source_with_cache_root(location, config, metadata, cache_root);
    }

    remote_kvikio_source(location, config)
}

fn remote_kvikio_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
) -> Result<ResolvedParquetSource> {
    remote_kvikio_or_capability_error(location, config)
}

fn resolve_s3_table_format_parquet_source_with_cache_root(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
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
    if cache_identity_allowed && config.lakehouse_cache_policy.fills_before_read() {
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

fn resolved_remote_kvikio_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
) -> ResolvedParquetSource {
    ResolvedParquetSource::remote_kvikio(location, config.credential_mode)
}

fn remote_kvikio_or_capability_error(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
) -> Result<ResolvedParquetSource> {
    if let Some(reason) = config.remote_direct_profile_unsupported_reason {
        return reject_remote_kvikio_source(reason);
    }
    match remote_parquet_backend_build_status() {
        RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled => {},
        RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled => {
            return reject_remote_kvikio_source("remote_kvikio_backend_disabled");
        },
        RemoteParquetBackendBuildStatus::Unknown => {
            return reject_remote_kvikio_source("remote_kvikio_backend_unknown");
        },
    }
    Ok(resolved_remote_kvikio_source(location, config))
}

#[cfg(test)]
fn remote_direct_backend_disabled() -> bool {
    remote_parquet_backend_build_status() == RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled
}

fn reject_remote_kvikio_source(unavailable_reason: &'static str) -> Result<ResolvedParquetSource> {
    Err(Error::new(
        ErrorCode::InvalidExecutionPlan,
        "algeon_datafusion.table_format.resolve_parquet_source",
        "native GPU queries require the remote KvikIO source contract",
    )
    .with_fact("reason", "remote_kvikio_unavailable")
    .with_fact("remote_kvikio_unavailable_reason", unavailable_reason)
    .with_fact("remote_read_mode", "require_kvikio"))
}

fn background_fill_mode(config: &TableFormatParquetReadStrategyConfig) -> LakehouseCacheFillMode {
    match config.lakehouse_cache_policy {
        TableFormatLakehouseCachePolicy::ReadThrough => LakehouseCacheFillMode::Background,
        TableFormatLakehouseCachePolicy::Disabled | TableFormatLakehouseCachePolicy::FillBeforeRead => {
            LakehouseCacheFillMode::Disabled
        },
    }
}

fn table_format_local_path_text(location: &str) -> String {
    if let Some(path) = location.strip_prefix("file://localhost/") {
        return format!("/{path}");
    }
    if let Some(path) = location.strip_prefix("file://") {
        return path.to_string();
    }
    if let Some(path) = location.strip_prefix("file:") {
        return path.to_string();
    }
    location.to_string()
}

fn attach_remote_source_identity(
    resolved: ResolvedParquetSource,
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> ResolvedParquetSource {
    let resolved = resolved
        .with_storage_domain(storage_domain_identity(location, config, metadata))
        .with_optional_storage_profile_id(config.storage_profile_id.clone());
    if let Some(content_identity) = content_identity(metadata) {
        return resolved.with_content_identity(content_identity);
    }
    resolved
}

fn cache_relative_path(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> PathBuf {
    let scheme = location.split_once("://").map(|(scheme, _)| scheme).unwrap_or("unknown");
    PathBuf::from("v1")
        .join(sanitize_cache_component(scheme))
        .join(sanitize_cache_component(&storage_domain_identity(location, config, metadata)))
        .join(sanitize_cache_component(content_identity(metadata).as_deref().unwrap_or("content_unidentified")))
        .join(location_fingerprint(location))
}

fn storage_domain_identity(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> String {
    if let Some(storage_domain) = metadata.storage_domain.as_ref().filter(|value| !value.is_empty()) {
        return storage_domain.clone();
    }
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
/// embedded credentials cannot leak through table-format source facts.
pub fn credential_free_endpoint_identity(endpoint: &str) -> String {
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
/// Returns `None` when the caller supplied no usable identity. Size and row
/// count are deliberately not a fallback: same-size rewrites must map to no
/// servable entry rather than to a stale hit.
fn content_identity(metadata: &TableFormatParquetSourceMetadata) -> Option<String> {
    strong_content_identity(metadata).map(str::to_string)
}

fn strong_content_identity(metadata: &TableFormatParquetSourceMetadata) -> Option<&str> {
    metadata.content_identity.as_deref().filter(|value| !value.is_empty())
}

fn complete_cache_hit_matches_metadata(cache_path: &Path, metadata: &TableFormatParquetSourceMetadata) -> bool {
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

pub(super) fn redacted_scan_location_debug_value(location: &str) -> String {
    let safe_location = credential_free_endpoint_identity(location);
    redacted_scan_debug_value("location", &safe_location)
}

pub(super) fn redacted_scan_debug_value(kind: &str, value: &str) -> String {
    format!("{kind}:fnv64:{:016x}:len:{}", fnv1a64(value.as_bytes()), value.len())
}

fn sanitize_cache_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') { ch } else { '_' })
        .collect::<String>();
    if sanitized.is_empty() { "unknown".to_string() } else { sanitized }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hasher = FnvHasher::default();
    hasher.write(bytes);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs};

    use arrow::datatypes::{Field, Fields};

    use super::*;
    use crate::table_format::{
        CredentialMode, ResolvedParquetInput, S3SourceStorageProfile, SourceStorageProfile, TableFormat,
        TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
        TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent,
        TableFormatLakehouseCacheReadPolicy, TableFormatScanPlan,
        scan::{
            CapabilityReason, DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
            table_format_parquet_read_strategy_config_from_env, table_format_source_read_contract,
        },
    };

    #[test]
    fn table_format_parquet_read_strategy_config_from_env_uses_catalog_endpoint_identity() {
        let env = BTreeMap::from([
            ("ALGEON_ICEBERG_S3_ENDPOINT", "http://user:secret@rustfs.local:9000/minio?sig=secret"),
            ("AWS_ENDPOINT_URL_S3", "https://aws.example.invalid"),
        ]);
        let config =
            table_format_parquet_read_strategy_config_from_env(|name| env.get(name).map(|value| (*value).to_string()));

        assert_eq!(config.storage_endpoint.as_deref(), Some("http://rustfs.local:9000/minio"));
        assert_eq!(config.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(config.credential_mode, CredentialMode::ScopedResolved);
    }

    #[test]
    fn table_format_parquet_read_strategy_config_marks_explicit_s3_credentials_as_scoped() {
        let env = BTreeMap::from([
            ("ALGEON_ICEBERG_S3_ACCESS_KEY_ID", "access"),
            ("ALGEON_ICEBERG_S3_SECRET_ACCESS_KEY", "secret"),
        ]);
        let config =
            table_format_parquet_read_strategy_config_from_env(|name| env.get(name).map(|value| (*value).to_string()));

        assert_eq!(config.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(config.credential_mode, CredentialMode::ScopedResolved);
    }

    #[test]
    fn table_format_parquet_read_strategy_config_marks_s3_region_as_scoped() {
        let env = BTreeMap::from([("ALGEON_ICEBERG_S3_REGION", "us-west-2")]);
        let config =
            table_format_parquet_read_strategy_config_from_env(|name| env.get(name).map(|value| (*value).to_string()));

        assert_eq!(config.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(config.credential_mode, CredentialMode::ScopedResolved);

        let result = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        );
        let _ = assert_expected_remote_foreground_source(result);
    }

    #[test]
    fn table_format_parquet_read_strategy_config_uses_storage_profile_capability() {
        let default_profile = SourceStorageProfile::s3(
            DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
            S3SourceStorageProfile::default_chain(),
        );
        let default_config = TableFormatParquetReadStrategyConfig::default()
            .with_source_storage_profile_remote_direct_capability(&default_profile);
        assert_eq!(default_config.credential_mode, CredentialMode::SharedProcessEnv);

        let region_profile = SourceStorageProfile::s3(
            DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
            S3SourceStorageProfile::default_chain().with_region("us-west-2"),
        );
        let region_config = TableFormatParquetReadStrategyConfig::default()
            .with_source_storage_profile_remote_direct_capability(&region_profile);
        assert_eq!(region_config.credential_mode, CredentialMode::ScopedResolved);

        let scoped_profile = SourceStorageProfile::s3(
            DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
            S3SourceStorageProfile::default_chain()
                .with_endpoint("http://localhost:9000")
                .with_static_credentials("access", "secret", None),
        );
        let scoped_config = TableFormatParquetReadStrategyConfig::default()
            .with_source_storage_profile_remote_direct_capability(&scoped_profile);
        assert_eq!(scoped_config.credential_mode, CredentialMode::ScopedResolved);
    }

    #[test]
    fn content_identity_requires_caller_provided_identity() {
        assert_eq!(
            content_identity(&TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10))),
            None,
        );
        assert_eq!(content_identity(&TableFormatParquetSourceMetadata::default().with_size_bytes(4096)), None,);
        assert_eq!(content_identity(&TableFormatParquetSourceMetadata::default().with_row_count(10)), None,);
        assert_eq!(content_identity(&TableFormatParquetSourceMetadata::default()), None);
        assert_eq!(content_identity(&TableFormatParquetSourceMetadata::default().with_content_identity("")), None);
        assert_eq!(
            content_identity(&TableFormatParquetSourceMetadata::default().with_content_identity("etag:abc123")),
            Some("etag:abc123".to_string()),
        );
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

    #[test]
    fn table_format_scan_debug_redacts_credential_bearing_file_locations() {
        let data_location =
            "s3://secret-bucket/table/data.parquet?X-Amz-Signature=do-not-leak&X-Amz-Credential=also-secret";
        let delete_location = "s3://secret-bucket/table/delete.parquet?token=delete-secret";
        let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_data_file(TableFormatDataFile::new(
                data_location,
                data_location,
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            ))
            .with_delete_file(TableFormatDeleteFile::for_data_file(
                delete_location,
                data_location,
                delete_location,
                TableFormatDataFileFormat::Parquet,
                TableFormatDeleteFileContent::PositionDeletes,
            ));

        let rendered = format!("{plan:?}");
        for sensitive in
            ["secret-bucket", "X-Amz-Signature", "X-Amz-Credential", "do-not-leak", "also-secret", "delete-secret"]
        {
            assert!(!rendered.contains(sensitive), "debug output leaked {sensitive}: {rendered}");
        }
    }

    #[test]
    fn first_not_supported_reason_ignores_unprojected_unsupported_schema_fields() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("allRiskEvents", map_utf8_int32_type(), true),
        ]));
        let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(test_table_format_parquet_file())
            .with_projected_column(TableFormatColumnMapping::new(
                None,
                "id",
                Some("id"),
                DataType::Int64,
                false,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                TableFormatColumnRole::Projected,
            ));

        assert_eq!(plan.first_not_supported_reason(), None);
        assert_eq!(plan.first_unsupported_column(), None);
    }

    #[test]
    fn first_not_supported_reason_reports_projected_unsupported_schema_fields() {
        let map_type = map_utf8_int32_type();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("allRiskEvents", map_type.clone(), true),
        ]));
        let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
            .with_data_file(test_table_format_parquet_file())
            .with_projected_column(TableFormatColumnMapping::new(
                None,
                "allRiskEvents",
                Some("allRiskEvents"),
                map_type,
                true,
                TableFormatColumnSource::DirectTopLevelFileColumn,
                TableFormatColumnRole::Projected,
            ));

        assert_eq!(plan.first_not_supported_reason(), Some(CapabilityReason::UnsupportedType));
        let unsupported = plan.first_unsupported_column().expect("projected map column must be named");
        assert_eq!(unsupported.logical_name, "allRiskEvents");
        assert_eq!(unsupported.role_code(), "projected");
        assert!(unsupported.data_type_summary().starts_with("Map("));
    }

    #[test]
    fn local_file_uri_reader_paths_are_normalized_for_local_cudf_reads() {
        assert_eq!(table_format_local_path_text("/tmp/table.parquet"), "/tmp/table.parquet");
        assert_eq!(table_format_local_path_text("file:/tmp/table.parquet"), "/tmp/table.parquet");
        assert_eq!(table_format_local_path_text("file:///tmp/table.parquet"), "/tmp/table.parquet");
        assert_eq!(table_format_local_path_text("file://localhost/tmp/table.parquet"), "/tmp/table.parquet");
    }

    fn test_table_format_parquet_file() -> TableFormatDataFile {
        TableFormatDataFile::new(
            "data-0001",
            "/tmp/iceberg/data-0001.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::local_file_system(),
        )
    }

    fn map_utf8_int32_type() -> DataType {
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("key", DataType::Utf8, false)),
                    Arc::new(Field::new("value", DataType::Int32, true)),
                ])),
                false,
            )),
            false,
        )
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
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            ..Default::default()
        };

        let result = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10)),
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
        let config = TableFormatParquetReadStrategyConfig {
            credential_mode: CredentialMode::ScopedResolved,
            ..Default::default()
        };

        let result = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
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
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            ..Default::default()
        };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let source = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        )
        .unwrap();

        assert_eq!(source.read_strategy().as_str(), "local_cache_kvikio");
        assert!(matches!(&source.input, ResolvedParquetInput::LocalKvikio { .. }));
        assert_eq!(source.cache_status, LakehouseCacheStatus::Hit);
        assert_eq!(source.reader_location(), cache_path.display().to_string());
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
        assert_eq!(table_format_source_read_contract(&source), "local_cache_hit");
    }

    #[test]
    fn s3_auto_ignores_complete_local_cache_file_with_wrong_size() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            ..Default::default()
        };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, b"incomplete").unwrap();

        let result = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        );
        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn s3_auto_ignores_local_cache_file_when_size_metadata_is_missing() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            ..Default::default()
        };
        let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(None, Some(10))
            .with_content_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let result = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        );
        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn s3_auto_cache_read_bypass_skips_existing_local_cache_hit() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            lakehouse_cache_read_policy: TableFormatLakehouseCacheReadPolicy::Disabled,
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::ReadThrough,
            ..Default::default()
        };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let result = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        );

        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Bypassed);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Background);
        assert_eq!(source.reader_location(), "s3://bucket/table/part-0.parquet");
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn explicit_local_cache_read_bypass_skips_existing_local_cache_hit() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            lakehouse_cache_read_policy: TableFormatLakehouseCacheReadPolicy::Disabled,
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::ReadThrough,
            ..Default::default()
        };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let result = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        );

        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Bypassed);
        assert_eq!(source.lakehouse_cache_read_mode, LakehouseCacheReadMode::Bypassed);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Background);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn read_through_cache_miss_selects_background_fill_without_blocking_foreground_read() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::ReadThrough,
            ..Default::default()
        };
        let metadata = metadata_with_identity("etag:abc123");
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));

        let result = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        );

        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Background);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    fn metadata_with_identity(identity: &str) -> TableFormatParquetSourceMetadata {
        TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10)).with_content_identity(identity)
    }

    #[test]
    fn s3_same_size_rewrite_with_new_identity_is_not_a_hit() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            ..Default::default()
        };
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
        let result = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &new_metadata,
        );
        let Some(source) = assert_expected_remote_foreground_source(result) else {
            return;
        };
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(new_cache_path.to_str().unwrap()));
    }

    #[test]
    fn s3_without_content_identity_neither_serves_nor_fills() {
        for policy in [TableFormatLakehouseCachePolicy::ReadThrough, TableFormatLakehouseCachePolicy::FillBeforeRead] {
            let root = tempfile::tempdir().unwrap();
            let config = TableFormatParquetReadStrategyConfig {
                lakehouse_cache_root: Some(root.path().to_path_buf()),
                lakehouse_cache_policy: policy,
                ..Default::default()
            };
            let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10));
            // Even with bytes planted at the unidentified path, they must not be served.
            let cache_path =
                root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
            assert!(cache_path.display().to_string().contains("content_unidentified"));
            fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
            fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

            let result = resolve_table_format_parquet_source_with_metadata(
                "s3://bucket/table/part-0.parquet",
                &TableFormatReadCapability::object_store("s3"),
                &config,
                &metadata,
            );
            let Some(source) = assert_expected_remote_foreground_source(result) else {
                return;
            };
            assert_eq!(source.cache_status, LakehouseCacheStatus::Miss, "policy={policy:?}");
            assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Disabled, "policy={policy:?}");
            assert_eq!(source.content_identity, None, "policy={policy:?}");
        }
    }
}
