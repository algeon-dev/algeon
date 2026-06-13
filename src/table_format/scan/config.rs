use super::*;

pub const DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID: &str = "table_format_s3";

/// Env var selecting the table-format Parquet source strategy. Accepts `auto`,
/// `host_buffer`, `local_cache`, or `remote_kvikio`; defaults to `auto`.
pub const NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV: &str = "NEXUS_ICEBERG_PARQUET_READ_STRATEGY";
/// Boolean env var enabling foreground remote-direct object-store Parquet
/// reads. Defaults to enabled.
pub const NEXUS_ICEBERG_REMOTE_DIRECT_ENV: &str = "NEXUS_ICEBERG_REMOTE_DIRECT";
/// Env var selecting the lower-layer fallback policy for remote-direct Parquet
/// reads. Accepts `prefer_direct`, `require_direct`, or
/// `allow_host_fallback`; defaults to `prefer_direct`.
pub const NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV: &str = "NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE";
/// Boolean env var enabling ranged mini-Parquet fallback for remote host-buffer
/// reads. Defaults to disabled.
pub const NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV: &str = "NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED";
/// Env var selecting whether complete local lakehouse cache hits may be used
/// for foreground reads. Accepts `enabled`, `disabled`, or `bypass`; defaults
/// to `enabled`.
pub const NEXUS_LAKEHOUSE_CACHE_READ_ENV: &str = "NEXUS_LAKEHOUSE_CACHE_READ";
/// Env var pointing at the local lakehouse byte-cache root.
pub const NEXUS_LAKEHOUSE_CACHE_ROOT_ENV: &str = "NEXUS_LAKEHOUSE_CACHE_ROOT";
/// Env var selecting lakehouse byte-cache miss behavior. Accepts `disabled`,
/// `read_through`, or `fill_before_read`; defaults to `disabled`.
pub const NEXUS_LAKEHOUSE_CACHE_POLICY_ENV: &str = "NEXUS_LAKEHOUSE_CACHE_POLICY";
/// Env var selecting the best-effort retained byte target for the lakehouse
/// cache. Values are bytes; zero or invalid values are ignored by the
/// format-neutral resolver and rejected by server config validation.
pub const NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV: &str = "NEXUS_LAKEHOUSE_CACHE_MAX_BYTES";
/// Standard AWS SDK S3 endpoint override env var.
pub const AWS_ENDPOINT_URL_S3_ENV: &str = "AWS_ENDPOINT_URL_S3";
/// Standard AWS SDK endpoint override env var.
pub const AWS_ENDPOINT_URL_ENV: &str = "AWS_ENDPOINT_URL";

/// User-selected table-format Parquet strategy preference.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatParquetReadStrategyPreference {
    /// Choose the safe default for the source location.
    #[default]
    Auto,
    /// Force remote object-store host-buffer fetch/decode.
    HostBuffer,
    /// Prefer an existing local byte-cache entry when available.
    LocalCache,
    /// Prefer remote KvikIO through the remote-direct reader path.
    RemoteKvikio,
}

impl TableFormatParquetReadStrategyPreference {
    /// Stable config/report string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::HostBuffer => "host_buffer",
            Self::LocalCache => "local_cache",
            Self::RemoteKvikio => "remote_kvikio",
        }
    }
}

/// Lakehouse byte-cache behavior for remote table-format Parquet files.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatLakehouseCachePolicy {
    /// Do not fill missing local cache entries.
    #[default]
    Disabled,
    /// Read the foreground source now and fill missing complete-object entries in the background.
    ReadThrough,
    /// Require a completed local cache file before the path-based cuDF read starts.
    FillBeforeRead,
}

impl TableFormatLakehouseCachePolicy {
    /// Stable config/report string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ReadThrough => "read_through",
            Self::FillBeforeRead => "fill_before_read",
        }
    }

    /// Whether a cache miss should be materialized before execution opens the cache path.
    pub const fn fills_before_read(self) -> bool {
        matches!(self, Self::FillBeforeRead)
    }
}

/// Foreground lakehouse cache read behavior for complete local cache entries.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatLakehouseCacheReadPolicy {
    /// Use complete local cache hits for foreground reads.
    #[default]
    Enabled,
    /// Ignore complete local cache hits for benchmark isolation.
    Disabled,
}

impl TableFormatLakehouseCacheReadPolicy {
    /// Stable config/report string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }

    /// Whether cache hits should be considered for foreground reads.
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Table-format Parquet read-strategy resolver configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatParquetReadStrategyConfig {
    /// Preferred strategy for remote Parquet data files.
    pub preference: TableFormatParquetReadStrategyPreference,
    /// Whether remote-direct object-store reads are available to the foreground query policy.
    pub remote_direct_enabled: bool,
    /// Lower-layer fallback policy requested for remote-direct object-store reads.
    pub remote_direct_read_mode: RemoteParquetReadMode,
    /// Whether remote host-buffer fallback may use ranged mini-Parquet reads.
    pub remote_host_buffer_ranged_enabled: bool,
    /// Optional lakehouse byte-cache root.
    pub lakehouse_cache_root: Option<PathBuf>,
    /// Whether complete cache hits may be used for foreground reads.
    pub lakehouse_cache_read_policy: TableFormatLakehouseCacheReadPolicy,
    /// Cache-miss policy for local lakehouse byte-cache entries.
    pub lakehouse_cache_policy: TableFormatLakehouseCachePolicy,
    /// Best-effort retained byte target for the lakehouse byte-cache root.
    pub lakehouse_cache_max_bytes: Option<u64>,
    /// Optional catalog-native S3 endpoint identity used to scope local-cache entries.
    ///
    /// This is endpoint metadata only. It must not include credentials; when
    /// resolved from environment variables, userinfo, query strings, and
    /// fragments are stripped before the value is exposed in diagnostics.
    pub storage_endpoint: Option<String>,
    /// Execution-time native source storage profile id.
    ///
    /// This id references a `nexus-query-engine` profile registered on the native
    /// execution config. It is not a credential and is safe for diagnostics.
    pub storage_profile_id: Option<String>,
    /// Credential mode used for remote object-store reads.
    pub credential_mode: CredentialMode,
    /// Stable reason when the configured remote-direct profile is unsupported.
    pub remote_direct_profile_unsupported_reason: Option<&'static str>,
}

impl Default for TableFormatParquetReadStrategyConfig {
    fn default() -> Self {
        Self {
            preference: TableFormatParquetReadStrategyPreference::Auto,
            remote_direct_enabled: true,
            remote_direct_read_mode: RemoteParquetReadMode::PreferDirect,
            remote_host_buffer_ranged_enabled: false,
            lakehouse_cache_root: None,
            lakehouse_cache_read_policy: TableFormatLakehouseCacheReadPolicy::Enabled,
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::Disabled,
            lakehouse_cache_max_bytes: None,
            storage_endpoint: None,
            storage_profile_id: None,
            credential_mode: CredentialMode::SharedProcessEnv,
            remote_direct_profile_unsupported_reason: None,
        }
    }
}

impl TableFormatParquetReadStrategyConfig {
    /// Builds a resolver config from the Iceberg-scoped v1 environment knobs.
    pub fn from_env() -> Self {
        table_format_parquet_read_strategy_config_from_env(|name| std::env::var(name).ok())
    }

    /// Applies the non-secret remote-direct credential capability implied by a
    /// registered execution-time source storage profile.
    pub fn with_source_storage_profile_remote_direct_capability(mut self, profile: &SourceStorageProfile) -> Self {
        if self.storage_profile_id.is_none() {
            self.storage_profile_id = Some(profile.id().to_string());
        }
        if profile.requires_scoped_remote_direct_profile() {
            self.credential_mode = CredentialMode::ScopedResolved;
        }
        self.remote_direct_profile_unsupported_reason = profile.remote_direct_profile_unsupported_reason();
        self
    }
}

pub(super) fn table_format_parquet_read_strategy_config_from_env(
    mut get: impl FnMut(&str) -> Option<String>,
) -> TableFormatParquetReadStrategyConfig {
    let env_nonempty = |get: &mut dyn FnMut(&str) -> Option<String>, name: &str| {
        get(name).map(|value| value.trim().to_string()).filter(|value| !value.is_empty())
    };
    let preference = match get(NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV)
        .unwrap_or_else(|| "auto".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "host_buffer" => TableFormatParquetReadStrategyPreference::HostBuffer,
        "local_cache" => TableFormatParquetReadStrategyPreference::LocalCache,
        "remote_kvikio" => TableFormatParquetReadStrategyPreference::RemoteKvikio,
        _ => TableFormatParquetReadStrategyPreference::Auto,
    };
    let remote_direct_enabled = get(NEXUS_ICEBERG_REMOTE_DIRECT_ENV)
        .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"));
    let remote_direct_read_mode = get(NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV)
        .as_deref()
        .map(parse_remote_parquet_read_mode_lossy)
        .unwrap_or_default();
    let remote_host_buffer_ranged_enabled = get(NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV)
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
    let lakehouse_cache_root = env_nonempty(&mut get, NEXUS_LAKEHOUSE_CACHE_ROOT_ENV).map(PathBuf::from);
    let lakehouse_cache_read_policy = match get(NEXUS_LAKEHOUSE_CACHE_READ_ENV)
        .unwrap_or_else(|| "enabled".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "disabled" | "bypass" | "bypassed" => TableFormatLakehouseCacheReadPolicy::Disabled,
        _ => TableFormatLakehouseCacheReadPolicy::Enabled,
    };
    let lakehouse_cache_policy = match get(NEXUS_LAKEHOUSE_CACHE_POLICY_ENV)
        .unwrap_or_else(|| "disabled".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "read_through" => TableFormatLakehouseCachePolicy::ReadThrough,
        "fill_before_read" => TableFormatLakehouseCachePolicy::FillBeforeRead,
        _ => TableFormatLakehouseCachePolicy::Disabled,
    };
    let lakehouse_cache_max_bytes =
        get(NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV).and_then(|value| value.parse::<u64>().ok()).filter(|value| *value > 0);
    let storage_endpoint = env_nonempty(&mut get, "NEXUS_ICEBERG_S3_ENDPOINT")
        .or_else(|| env_nonempty(&mut get, AWS_ENDPOINT_URL_S3_ENV))
        .or_else(|| env_nonempty(&mut get, AWS_ENDPOINT_URL_ENV))
        .map(|endpoint| credential_free_endpoint_identity(&endpoint));
    let has_storage_region = env_nonempty(&mut get, "NEXUS_ICEBERG_S3_REGION").is_some();
    let has_explicit_s3_credentials = env_nonempty(&mut get, "NEXUS_ICEBERG_S3_ACCESS_KEY_ID").is_some()
        || env_nonempty(&mut get, "NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY").is_some()
        || env_nonempty(&mut get, "NEXUS_ICEBERG_S3_SESSION_TOKEN").is_some();
    let credential_mode = if storage_endpoint.is_some() || has_storage_region || has_explicit_s3_credentials {
        CredentialMode::ScopedResolved
    } else {
        CredentialMode::SharedProcessEnv
    };
    let storage_profile_id = if storage_endpoint.is_some() || has_storage_region || has_explicit_s3_credentials {
        Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID.to_string())
    } else {
        None
    };
    TableFormatParquetReadStrategyConfig {
        preference,
        remote_direct_enabled,
        remote_direct_read_mode,
        remote_host_buffer_ranged_enabled,
        lakehouse_cache_root,
        lakehouse_cache_read_policy,
        lakehouse_cache_policy,
        lakehouse_cache_max_bytes,
        storage_endpoint,
        storage_profile_id,
        credential_mode,
        remote_direct_profile_unsupported_reason: None,
    }
}

fn parse_remote_parquet_read_mode_lossy(raw: &str) -> RemoteParquetReadMode {
    match raw.trim().to_ascii_lowercase().as_str() {
        "require" | "require_direct" => RemoteParquetReadMode::RequireDirect,
        "allow_host_fallback" | "host_fallback" => RemoteParquetReadMode::AllowHostFallback,
        "prefer" | "prefer_direct" => RemoteParquetReadMode::PreferDirect,
        _ => RemoteParquetReadMode::PreferDirect,
    }
}
