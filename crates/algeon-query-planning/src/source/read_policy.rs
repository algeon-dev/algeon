//! Parquet reader strategy, credentials, and cache-policy contracts.

use std::fmt;

use super::diagnostics::{redacted_debug_option, redacted_debug_value};

/// High-level Parquet read strategy selected before source execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParquetReadStrategy {
    /// Path-based cuDF/libcudf read for a local filesystem path.
    LocalKvikio,
    /// Read a previously materialized local byte-cache entry through the path-based reader.
    LocalCacheKvikio,
    /// Path-based cuDF/libcudf remote KvikIO read for an object-store URI such as `s3://...`.
    RemoteDirect,
}

impl ParquetReadStrategy {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalKvikio => "local_kvikio",
            Self::LocalCacheKvikio => "local_cache_kvikio",
            Self::RemoteDirect => "remote_direct",
        }
    }
}

impl fmt::Display for ParquetReadStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tagged cuDF/KvikIO data-plane input.
///
/// Host-memory Parquet payloads are intentionally not representable.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ResolvedParquetInput {
    /// Local filesystem path, including a completed lakehouse-cache entry.
    LocalKvikio {
        /// Local path passed to the native reader, including a completed cache path.
        reader_path: String,
    },
    /// Remote object-store URI read by cuDF's explicit KvikIO datasource.
    RemoteKvikio {
        /// Object-store URI passed to the native datasource.
        uri: String,
        /// Credential acquisition policy selected during source resolution.
        credential_mode: CredentialMode,
        /// Named storage profile, or `None` when no explicit profile was selected.
        storage_profile_id: Option<String>,
    },
}

impl fmt::Debug for ResolvedParquetInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalKvikio { reader_path } => f
                .debug_struct("LocalKvikio")
                .field("reader_path", &redacted_debug_value("reader_path", reader_path))
                .finish(),
            Self::RemoteKvikio { uri, credential_mode, storage_profile_id } => f
                .debug_struct("RemoteKvikio")
                .field("uri", &redacted_debug_value("uri", uri))
                .field("credential_mode", credential_mode)
                .field("storage_profile_id", storage_profile_id)
                .finish(),
        }
    }
}

impl ResolvedParquetInput {
    pub(crate) fn reader_location(&self) -> &str {
        match self {
            Self::LocalKvikio { reader_path } => reader_path,
            Self::RemoteKvikio { uri, .. } => uri,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ObjectStoreAccess {
    credential_mode: CredentialMode,
    storage_profile_id: Option<String>,
}

/// Credential scope used by a resolved source read.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialMode {
    /// No external credentials are needed.
    #[default]
    None,
    /// Reader uses credentials already configured for the running process.
    SharedProcessEnv,
    /// Engine resolves per-profile credentials and scopes them to the reader.
    ScopedResolved,
}

impl CredentialMode {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SharedProcessEnv => "shared_process_env",
            Self::ScopedResolved => "scoped_resolved",
        }
    }
}

impl fmt::Display for CredentialMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Local byte-cache status for a resolved Parquet source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LakehouseCacheStatus {
    /// Cache strategy is not enabled for this source.
    Disabled,
    /// A complete cache entry is used as the reader input.
    Hit,
    /// A cache entry was absent.
    Miss,
    /// Cache fill was selected before reading.
    FillBeforeRead,
    /// Cache lookup was explicitly bypassed for benchmark isolation.
    Bypassed,
    /// Cache was considered but not used.
    Skipped,
}

impl LakehouseCacheStatus {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::FillBeforeRead => "fill_before_read",
            Self::Bypassed => "bypassed",
            Self::Skipped => "skipped",
        }
    }
}

impl fmt::Display for LakehouseCacheStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Foreground lakehouse cache read behavior selected for a resolved source.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LakehouseCacheReadMode {
    /// No complete-object cache lookup participates in the foreground read.
    #[default]
    Disabled,
    /// Complete-object cache hits are eligible for foreground reads.
    Enabled,
    /// Complete-object cache hits were explicitly bypassed.
    Bypassed,
}

impl LakehouseCacheReadMode {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
            Self::Bypassed => "bypassed",
        }
    }
}

impl fmt::Display for LakehouseCacheReadMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Full-object lakehouse cache fill behavior selected for a resolved source.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LakehouseCacheFillMode {
    /// Do not fill a missing complete cache entry.
    #[default]
    Disabled,
    /// Fill the complete object asynchronously after the foreground read is scheduled.
    Background,
    /// Fill the complete cache entry before opening it as a local reader path.
    Blocking,
}

impl LakehouseCacheFillMode {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Background => "background",
            Self::Blocking => "blocking",
        }
    }
}

/// Resolved Parquet reader contract carried into native source execution.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ResolvedParquetSource {
    /// Frontend-supplied file location before read-strategy resolution.
    pub original_location: String,
    /// The only data-plane input supplied to cuDF.
    pub input: ResolvedParquetInput,
    /// Credential-free storage domain used for cache scoping, when known.
    pub storage_domain: Option<String>,
    /// Credential-free content identity used for cache scoping, when known.
    pub content_identity: Option<String>,
    /// Object-store identity retained only for cache fill and metadata maintenance
    /// when the data-plane input is already a local path.
    maintenance_access: Option<ObjectStoreAccess>,
    /// Local byte-cache status for this source.
    pub cache_status: LakehouseCacheStatus,
    /// Foreground cache-read behavior for this source.
    pub lakehouse_cache_read_mode: LakehouseCacheReadMode,
    /// Lakehouse byte-cache root used by cache maintenance, when configured.
    pub lakehouse_cache_root: Option<String>,
    /// Complete-object cache path used for hit reads or background fills.
    pub lakehouse_cache_path: Option<String>,
    /// Best-effort retained byte target for the lakehouse byte-cache root.
    pub lakehouse_cache_max_bytes: Option<u64>,
    /// Cache-fill mode for complete-object lakehouse cache maintenance.
    pub lakehouse_cache_fill_mode: LakehouseCacheFillMode,
}

impl fmt::Debug for ResolvedParquetSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedParquetSource")
            .field("original_location", &redacted_debug_value("location", &self.original_location))
            .field("reader_location", &redacted_debug_value("reader_location", self.reader_location()))
            .field("storage_domain", &self.storage_domain)
            .field("content_identity", &self.content_identity)
            .field("storage_profile_id", &self.storage_profile_id())
            .field("input", &self.input)
            .field("read_strategy", &self.read_strategy())
            .field("credential_mode", &self.credential_mode())
            .field("cache_status", &self.cache_status)
            .field("lakehouse_cache_read_mode", &self.lakehouse_cache_read_mode)
            .field("lakehouse_cache_root", &redacted_debug_option("cache_root", self.lakehouse_cache_root.as_deref()))
            .field("lakehouse_cache_path", &redacted_debug_option("cache_path", self.lakehouse_cache_path.as_deref()))
            .field("lakehouse_cache_max_bytes", &self.lakehouse_cache_max_bytes)
            .field("lakehouse_cache_fill_mode", &self.lakehouse_cache_fill_mode)
            .finish()
    }
}

impl ResolvedParquetSource {
    /// Builds a local path-based read contract.
    pub fn local_kvikio(location: impl Into<String>) -> Self {
        let location = location.into();
        Self {
            original_location: location.clone(),
            input: ResolvedParquetInput::LocalKvikio { reader_path: location },
            storage_domain: None,
            content_identity: None,
            maintenance_access: None,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_read_mode: LakehouseCacheReadMode::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_path: None,
            lakehouse_cache_max_bytes: None,
            lakehouse_cache_fill_mode: LakehouseCacheFillMode::Disabled,
        }
    }

    /// Builds a read contract for a local byte-cache entry, with the given cache status.
    pub fn local_cache_kvikio(
        original_location: impl Into<String>,
        reader_location: impl Into<String>,
        credential_mode: CredentialMode,
        cache_status: LakehouseCacheStatus,
    ) -> Self {
        Self {
            original_location: original_location.into(),
            input: ResolvedParquetInput::LocalKvikio { reader_path: reader_location.into() },
            storage_domain: None,
            content_identity: None,
            maintenance_access: Some(ObjectStoreAccess { credential_mode, storage_profile_id: None }),
            cache_status,
            lakehouse_cache_read_mode: LakehouseCacheReadMode::Enabled,
            lakehouse_cache_root: None,
            lakehouse_cache_path: None,
            lakehouse_cache_max_bytes: None,
            lakehouse_cache_fill_mode: LakehouseCacheFillMode::Disabled,
        }
    }

    /// Builds a remote path-based read contract where cuDF/libcudf owns the
    /// object-store read. Host-memory Parquet payloads are not representable.
    pub fn remote_kvikio(original_location: impl Into<String>, credential_mode: CredentialMode) -> Self {
        let original_location = original_location.into();
        Self {
            input: ResolvedParquetInput::RemoteKvikio {
                uri: original_location.clone(),
                credential_mode,
                storage_profile_id: None,
            },
            original_location,
            storage_domain: None,
            content_identity: None,
            maintenance_access: None,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_read_mode: LakehouseCacheReadMode::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_path: None,
            lakehouse_cache_max_bytes: None,
            lakehouse_cache_fill_mode: LakehouseCacheFillMode::Disabled,
        }
    }

    /// Replaces the local byte-cache status after strategy fallback.
    pub fn with_cache_status(mut self, cache_status: LakehouseCacheStatus) -> Self {
        self.cache_status = cache_status;
        self
    }

    /// Selects foreground complete-object lakehouse cache read behavior.
    pub fn with_lakehouse_cache_read_mode(mut self, read_mode: LakehouseCacheReadMode) -> Self {
        self.lakehouse_cache_read_mode = read_mode;
        self
    }

    /// Attaches a credential-free storage-domain identity for cache scoping.
    pub fn with_storage_domain(mut self, storage_domain: impl Into<String>) -> Self {
        self.storage_domain = Some(storage_domain.into());
        self
    }

    /// Attaches a credential-free content identity for cache scoping.
    pub fn with_content_identity(mut self, content_identity: impl Into<String>) -> Self {
        self.content_identity = Some(content_identity.into());
        self
    }

    /// Attaches the execution-time source storage profile id.
    pub fn with_storage_profile_id(mut self, profile_id: impl Into<String>) -> Self {
        self.set_storage_profile_id(Some(profile_id.into()));
        self
    }

    /// Attaches an execution-time source storage profile id when present.
    pub fn with_optional_storage_profile_id(mut self, profile_id: Option<String>) -> Self {
        self.set_storage_profile_id(profile_id);
        self
    }

    fn set_storage_profile_id(&mut self, profile_id: Option<String>) {
        match &mut self.input {
            ResolvedParquetInput::RemoteKvikio { storage_profile_id, .. } => *storage_profile_id = profile_id,
            ResolvedParquetInput::LocalKvikio { .. } => {
                self.maintenance_access
                    .get_or_insert(ObjectStoreAccess {
                        credential_mode: CredentialMode::None,
                        storage_profile_id: None,
                    })
                    .storage_profile_id = profile_id;
            },
        }
    }

    /// Attaches lakehouse byte-cache maintenance settings.
    pub fn with_lakehouse_cache_config(
        mut self,
        cache_root: impl Into<String>,
        max_retained_bytes: Option<u64>,
    ) -> Self {
        self.lakehouse_cache_root = Some(cache_root.into());
        self.lakehouse_cache_max_bytes = max_retained_bytes;
        self
    }

    /// Attaches a complete-object lakehouse cache path.
    pub fn with_lakehouse_cache_path(mut self, cache_path: impl Into<String>) -> Self {
        self.lakehouse_cache_path = Some(cache_path.into());
        self
    }

    /// Selects complete-object lakehouse cache fill behavior.
    pub fn with_lakehouse_cache_fill_mode(mut self, fill_mode: LakehouseCacheFillMode) -> Self {
        self.lakehouse_cache_fill_mode = fill_mode;
        self
    }

    /// Path or URI passed to cuDF.
    pub fn reader_location(&self) -> &str {
        self.input.reader_location()
    }

    /// Credential scope for the active remote read or local-cache maintenance.
    pub fn credential_mode(&self) -> CredentialMode {
        match &self.input {
            ResolvedParquetInput::RemoteKvikio { credential_mode, .. } => *credential_mode,
            ResolvedParquetInput::LocalKvikio { .. } => {
                self.maintenance_access.as_ref().map_or(CredentialMode::None, |access| access.credential_mode)
            },
        }
    }

    /// Storage profile for the active remote read or local-cache maintenance.
    pub fn storage_profile_id(&self) -> Option<&str> {
        match &self.input {
            ResolvedParquetInput::RemoteKvikio { storage_profile_id, .. } => storage_profile_id.as_deref(),
            ResolvedParquetInput::LocalKvikio { .. } => {
                self.maintenance_access.as_ref().and_then(|access| access.storage_profile_id.as_deref())
            },
        }
    }

    /// Report-facing strategy derived from the tagged input and cache facts.
    pub fn read_strategy(&self) -> ParquetReadStrategy {
        match self.input {
            ResolvedParquetInput::RemoteKvikio { .. } => ParquetReadStrategy::RemoteDirect,
            ResolvedParquetInput::LocalKvikio { .. }
                if matches!(self.cache_status, LakehouseCacheStatus::Hit | LakehouseCacheStatus::FillBeforeRead) =>
            {
                ParquetReadStrategy::LocalCacheKvikio
            },
            ResolvedParquetInput::LocalKvikio { .. } => ParquetReadStrategy::LocalKvikio,
        }
    }
}
