//! Parquet reader strategy, credentials, and cache-policy contracts.

use std::fmt;

use super::diagnostics::{redacted_debug_option, redacted_debug_value};

/// High-level Parquet read strategy selected before source execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParquetReadStrategy {
    /// Path-based cuDF/libcudf read for a local filesystem path.
    LocalKvikio,
    /// Fetch remote object bytes into host memory, then decode with cuDF.
    RemoteHostBuffer,
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
            Self::RemoteHostBuffer => "remote_host_buffer",
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

/// Mechanical input form supplied to cuDF's Parquet decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceDecodeInput {
    /// cuDF receives a path or URI string.
    Filepath,
    /// cuDF receives an already fetched host-memory Parquet payload.
    HostBuffer,
}

impl SourceDecodeInput {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Filepath => "filepath",
            Self::HostBuffer => "host_buffer",
        }
    }
}

impl fmt::Display for SourceDecodeInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Remote object-store Parquet read fallback policy requested from lower layers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteParquetReadMode {
    /// Fail unless lower layers can prove direct/range-aware remote IO.
    RequireDirect,
    /// Prefer direct/range-aware remote IO but allow the reader implementation to choose.
    #[default]
    PreferDirect,
    /// Explicitly allow host-buffer fallback when lower layers choose it.
    AllowHostFallback,
}

impl RemoteParquetReadMode {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RequireDirect => "require_direct",
            Self::PreferDirect => "prefer_direct",
            Self::AllowHostFallback => "allow_host_fallback",
        }
    }
}

impl fmt::Display for RemoteParquetReadMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
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
    /// Catalog or table-format location before read-strategy resolution.
    pub original_location: String,
    /// Path or URI string passed to the selected cuDF reader branch.
    pub reader_location: String,
    /// Credential-free storage domain used for cache scoping, when known.
    pub storage_domain: Option<String>,
    /// Credential-free content identity used for cache scoping, when known.
    pub content_identity: Option<String>,
    /// Execution-time source storage profile used for object-store fetches.
    ///
    /// The referenced profile lives on [`crate::runtime::ExecutionContext`]. Keeping only
    /// the id here lets plans and diagnostics stay credential-free.
    pub storage_profile_id: Option<String>,
    /// Selected high-level read strategy.
    pub read_strategy: ParquetReadStrategy,
    /// Mechanical decoder input used by the executor.
    pub decode_input: SourceDecodeInput,
    /// Remote object-store read fallback policy requested from lower layers.
    pub remote_read_mode: RemoteParquetReadMode,
    /// Credential scope used by the selected strategy.
    pub credential_mode: CredentialMode,
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
    /// Whether object-store host-buffer reads may use ranged mini-Parquet fallback.
    pub remote_host_buffer_ranged_enabled: bool,
    /// Stable fallback reason when the preferred strategy was unavailable.
    pub fallback_reason: Option<String>,
}

impl fmt::Debug for ResolvedParquetSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedParquetSource")
            .field("original_location", &redacted_debug_value("location", &self.original_location))
            .field("reader_location", &redacted_debug_value("reader_location", &self.reader_location))
            .field("storage_domain", &self.storage_domain)
            .field("content_identity", &self.content_identity)
            .field("storage_profile_id", &self.storage_profile_id)
            .field("read_strategy", &self.read_strategy)
            .field("decode_input", &self.decode_input)
            .field("remote_read_mode", &self.remote_read_mode)
            .field("credential_mode", &self.credential_mode)
            .field("cache_status", &self.cache_status)
            .field("lakehouse_cache_read_mode", &self.lakehouse_cache_read_mode)
            .field("lakehouse_cache_root", &redacted_debug_option("cache_root", self.lakehouse_cache_root.as_deref()))
            .field("lakehouse_cache_path", &redacted_debug_option("cache_path", self.lakehouse_cache_path.as_deref()))
            .field("lakehouse_cache_max_bytes", &self.lakehouse_cache_max_bytes)
            .field("lakehouse_cache_fill_mode", &self.lakehouse_cache_fill_mode)
            .field("remote_host_buffer_ranged_enabled", &self.remote_host_buffer_ranged_enabled)
            .field("fallback_reason", &self.fallback_reason)
            .finish()
    }
}

impl ResolvedParquetSource {
    /// Builds a local path-based read contract.
    pub fn local_kvikio(location: impl Into<String>) -> Self {
        let location = location.into();
        Self {
            original_location: location.clone(),
            reader_location: location,
            storage_domain: None,
            content_identity: None,
            storage_profile_id: None,
            read_strategy: ParquetReadStrategy::LocalKvikio,
            decode_input: SourceDecodeInput::Filepath,
            remote_read_mode: RemoteParquetReadMode::PreferDirect,
            credential_mode: CredentialMode::None,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_read_mode: LakehouseCacheReadMode::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_path: None,
            lakehouse_cache_max_bytes: None,
            lakehouse_cache_fill_mode: LakehouseCacheFillMode::Disabled,
            remote_host_buffer_ranged_enabled: false,
            fallback_reason: None,
        }
    }

    /// Builds a remote host-buffer read contract.
    pub fn remote_host_buffer(
        original_location: impl Into<String>,
        reader_location: impl Into<String>,
        credential_mode: CredentialMode,
    ) -> Self {
        Self {
            original_location: original_location.into(),
            reader_location: reader_location.into(),
            storage_domain: None,
            content_identity: None,
            storage_profile_id: None,
            read_strategy: ParquetReadStrategy::RemoteHostBuffer,
            decode_input: SourceDecodeInput::HostBuffer,
            remote_read_mode: RemoteParquetReadMode::AllowHostFallback,
            credential_mode,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_read_mode: LakehouseCacheReadMode::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_path: None,
            lakehouse_cache_max_bytes: None,
            lakehouse_cache_fill_mode: LakehouseCacheFillMode::Disabled,
            remote_host_buffer_ranged_enabled: false,
            fallback_reason: None,
        }
    }

    /// Builds a local cache hit read contract.
    pub fn local_cache_kvikio(
        original_location: impl Into<String>,
        reader_location: impl Into<String>,
        credential_mode: CredentialMode,
        cache_status: LakehouseCacheStatus,
    ) -> Self {
        Self {
            original_location: original_location.into(),
            reader_location: reader_location.into(),
            storage_domain: None,
            content_identity: None,
            storage_profile_id: None,
            read_strategy: ParquetReadStrategy::LocalCacheKvikio,
            decode_input: SourceDecodeInput::Filepath,
            remote_read_mode: RemoteParquetReadMode::PreferDirect,
            credential_mode,
            cache_status,
            lakehouse_cache_read_mode: LakehouseCacheReadMode::Enabled,
            lakehouse_cache_root: None,
            lakehouse_cache_path: None,
            lakehouse_cache_max_bytes: None,
            lakehouse_cache_fill_mode: LakehouseCacheFillMode::Disabled,
            remote_host_buffer_ranged_enabled: false,
            fallback_reason: None,
        }
    }

    /// Builds a remote path-based read contract where cuDF/libcudf owns the
    /// object-store read instead of receiving a host-memory Parquet payload.
    pub fn remote_direct(original_location: impl Into<String>, credential_mode: CredentialMode) -> Self {
        let original_location = original_location.into();
        Self {
            reader_location: original_location.clone(),
            original_location,
            storage_domain: None,
            content_identity: None,
            storage_profile_id: None,
            read_strategy: ParquetReadStrategy::RemoteDirect,
            decode_input: SourceDecodeInput::Filepath,
            remote_read_mode: RemoteParquetReadMode::PreferDirect,
            credential_mode,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_read_mode: LakehouseCacheReadMode::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_path: None,
            lakehouse_cache_max_bytes: None,
            lakehouse_cache_fill_mode: LakehouseCacheFillMode::Disabled,
            remote_host_buffer_ranged_enabled: false,
            fallback_reason: None,
        }
    }

    /// Attaches a stable fallback reason.
    pub fn with_fallback_reason(mut self, reason: impl Into<String>) -> Self {
        self.fallback_reason = Some(reason.into());
        self
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

    /// Selects remote object-store read fallback policy.
    pub fn with_remote_read_mode(mut self, read_mode: RemoteParquetReadMode) -> Self {
        self.remote_read_mode = read_mode;
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
        self.storage_profile_id = Some(profile_id.into());
        self
    }

    /// Attaches an execution-time source storage profile id when present.
    pub fn with_optional_storage_profile_id(mut self, profile_id: Option<String>) -> Self {
        self.storage_profile_id = profile_id;
        self
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

    /// Selects whether host-buffer object-store reads may use ranged mini-Parquet fallback.
    pub fn with_remote_host_buffer_ranged_enabled(mut self, enabled: bool) -> Self {
        self.remote_host_buffer_ranged_enabled = enabled;
        self
    }

    /// Whether execution must use the object-store host-buffer branch.
    pub fn uses_host_buffer(&self) -> bool {
        self.decode_input == SourceDecodeInput::HostBuffer
    }
}
