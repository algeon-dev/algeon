use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::datatypes::{DataType, Schema, SchemaRef};
use futures::future::BoxFuture;
use nexus_query_engine::{
    CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, RemoteParquetReadMode,
    ResolvedParquetSource, SourceStorageProfile,
};

use crate::{Result, policy::FallbackReason};

/// Default profile id used when table-format S3 env/config supplies explicit
/// source object-store settings.
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

/// Table format that produced a resolved scan plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormat {
    /// Apache Iceberg table metadata.
    Iceberg,
    /// Delta Lake table metadata.
    Delta,
    /// Another table format identified by a stable caller-provided code.
    Other(String),
}

/// Format-neutral request passed to table-format metadata planners.
///
/// `None` projection means all logical columns. `None` snapshot/catalog values
/// leave selection to the table-format implementation. `None` scan limit means
/// no source-level row bound was supplied by DataFusion; `Some(n)` is an
/// order-insensitive upper bound the native source may apply before any
/// inexact parent filter. The filtering flags describe planner intent only;
/// unsupported row-group pruning or row selection must be reported as stable
/// unsupported features instead of executing data reads.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatScanRequest {
    /// Logical projected column names. `None` means all columns; `Some([])` means zero columns.
    pub projected_column_names: Option<Vec<String>>,
    /// Optional table snapshot id in the format's native textual form.
    pub snapshot_id: Option<String>,
    /// Optional stable catalog label, such as `glue`.
    pub catalog_type: Option<String>,
    /// Optional source-level row upper bound supplied by DataFusion.
    pub scan_limit: Option<usize>,
    /// Whether column resolution should be case-sensitive.
    pub case_sensitive: bool,
    /// Planner intent for row-group filtering. Unsupported semantics become scan facts.
    pub row_group_filtering_enabled: bool,
    /// Planner intent for row selection/page-index pruning. Unsupported semantics become scan facts.
    pub row_selection_enabled: bool,
    /// Format-neutral predicate converted from frontend filters for metadata planning.
    pub predicate: Option<TableFormatPredicate>,
}

impl TableFormatScanRequest {
    /// Request all logical columns using default scan-planning options.
    pub fn all_columns() -> Self {
        Self::default()
    }

    /// Sets the exact logical column names to project.
    pub fn with_projected_column_names(mut self, column_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.projected_column_names = Some(column_names.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the table snapshot id.
    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    /// Sets a stable catalog label.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Sets an order-insensitive source-level scan limit.
    pub fn with_scan_limit(mut self, scan_limit: Option<usize>) -> Self {
        self.scan_limit = scan_limit;
        self
    }

    /// Sets case sensitivity for table-format column resolution.
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Sets whether the format planner should request row-group filtering.
    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    /// Sets whether the format planner should request row selection/page-index pruning.
    pub fn with_row_selection_enabled(mut self, row_selection_enabled: bool) -> Self {
        self.row_selection_enabled = row_selection_enabled;
        self
    }

    /// Sets a format-neutral metadata predicate converted from frontend filters.
    pub fn with_predicate(mut self, predicate: Option<TableFormatPredicate>) -> Self {
        self.predicate = predicate;
        self
    }
}

impl Default for TableFormatScanRequest {
    fn default() -> Self {
        Self {
            projected_column_names: None,
            snapshot_id: None,
            catalog_type: None,
            scan_limit: None,
            case_sensitive: true,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
            predicate: None,
        }
    }
}

/// Adapter-owned predicate tree for table-format metadata planning.
///
/// This intentionally avoids DataFusion, Iceberg, and native-engine expression
/// types. Format-specific planners translate this tree into their own metadata
/// predicate APIs when supported.
#[derive(Debug, Clone, PartialEq)]
pub enum TableFormatPredicate {
    /// Boolean conjunction. A planner may use both sides together.
    And(Box<TableFormatPredicate>, Box<TableFormatPredicate>),
    /// Boolean disjunction. A planner should use this only when both sides are supported.
    Or(Box<TableFormatPredicate>, Box<TableFormatPredicate>),
    /// Boolean negation.
    Not(Box<TableFormatPredicate>),
    /// Column equals true.
    BooleanColumn(String),
    /// Null predicate on a column.
    IsNull(String),
    /// Non-null predicate on a column.
    IsNotNull(String),
    /// Binary comparison between a column and a literal.
    Comparison {
        /// Logical column name.
        column: String,
        /// Comparison operator.
        op: TableFormatPredicateOp,
        /// Literal value.
        literal: TableFormatLiteral,
    },
}

/// Format-neutral comparison operators for table-format metadata predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatPredicateOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

/// Literal values supported by format-neutral table-format metadata predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum TableFormatLiteral {
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
    Date32(i32),
}

/// Adapter-owned table-format metadata planning boundary.
///
/// Implementations resolve metadata into scan facts and diagnostics; they do
/// not execute data reads. Diagnostics must not contain credentials or signed
/// URLs, and unsupported features should map to stable fallback reasons.
pub trait TableFormatPlanner {
    /// Resolves metadata into scan facts without reading table data.
    fn plan_scan<'a>(&'a self, request: TableFormatScanRequest) -> BoxFuture<'a, Result<TableFormatScanPlan>>;

    /// Resolves metadata with a caller-owned wall-clock deadline.
    ///
    /// The default implementation preserves existing planner behavior. Format
    /// planners with long synchronous or ready-loop sections should override
    /// this method and check the deadline at natural cancellation points.
    fn plan_scan_with_deadline<'a>(
        &'a self,
        request: TableFormatScanRequest,
        _deadline: TableFormatPlanningDeadline,
    ) -> BoxFuture<'a, Result<TableFormatScanPlan>> {
        self.plan_scan(request)
    }

    /// Credential-free metadata facts to attach when planning fails before a
    /// [`TableFormatScanPlan`] can be produced.
    fn planning_failure_facts(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Caller-owned wall-clock deadline for table-format metadata planning.
#[derive(Debug, Clone, Copy)]
pub struct TableFormatPlanningDeadline {
    deadline: Instant,
    timeout: Duration,
}

impl TableFormatPlanningDeadline {
    /// Builds a deadline from the current instant plus `timeout`.
    pub fn from_timeout(timeout: Duration) -> Self {
        Self { deadline: Instant::now() + timeout, timeout }
    }

    /// Configured timeout duration used to create the deadline.
    pub fn timeout(self) -> Duration {
        self.timeout
    }

    /// Millisecond budget for diagnostics, clamped to one millisecond.
    pub fn timeout_ms(self) -> u128 {
        self.timeout.as_millis().max(1)
    }

    /// Remaining wall-clock duration, or `None` when the deadline has elapsed.
    pub fn remaining(self) -> Option<Duration> {
        self.deadline.checked_duration_since(Instant::now()).filter(|remaining| !remaining.is_zero())
    }

    /// Returns true when the wall-clock deadline has elapsed.
    pub fn is_elapsed(self) -> bool {
        self.remaining().is_none()
    }
}

impl TableFormat {
    /// Stable report/fallback code for this table format.
    pub fn code(&self) -> &str {
        match self {
            Self::Iceberg => "iceberg",
            Self::Delta => "delta",
            Self::Other(format) => format.as_str(),
        }
    }
}

/// File format advertised by table-format metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatDataFileFormat {
    /// Parquet data file.
    Parquet,
    /// ORC data file.
    Orc,
    /// Avro data file.
    Avro,
    /// Another file format identified by a stable caller-provided code.
    Unknown(String),
}

impl TableFormatDataFileFormat {
    /// Stable report/fallback code for this data-file format.
    pub fn code(&self) -> &str {
        match self {
            Self::Parquet => "parquet",
            Self::Orc => "orc",
            Self::Avro => "avro",
            Self::Unknown(format) => format.as_str(),
        }
    }
}

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

fn table_format_parquet_read_strategy_config_from_env(
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
    /// Optional caller-provided content identity such as an object ETag.
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
    /// Object-store S3 reads may use either the remote-direct path-based reader
    /// or the legacy object-store host-buffer fallback.
    pub fn is_native_parquet_capable(&self) -> bool {
        match self {
            Self::LocalFileSystem => true,
            Self::ObjectStore { scheme } => scheme == "s3",
            Self::Unsupported { .. } => false,
        }
    }

    /// Whether this capability can be represented by local direct-device Parquet reads.
    ///
    /// Remote S3 direct/range-aware reads are tracked as a separate source
    /// strategy and must not be confused with local GDS/NVMe eligibility.
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
        TableFormatReadCapability::ObjectStore { scheme } => Err(crate::Error::new(
            crate::ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.table_format.resolve_parquet_source",
            "unsupported table-format object-store parquet source scheme",
        )
        .with_fact("source_location_scheme", scheme.as_str())),
        TableFormatReadCapability::Unsupported { reason } => Err(crate::Error::new(
            crate::ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.table_format.resolve_parquet_source",
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
    let resolved = match config.preference {
        TableFormatParquetReadStrategyPreference::Auto => {
            resolve_auto_s3_table_format_parquet_source(location, config, metadata)
        },
        TableFormatParquetReadStrategyPreference::HostBuffer => {
            remote_host_buffer_fallback_source(location, config, "explicit_remote_host_buffer_full", false)
        },
        TableFormatParquetReadStrategyPreference::RemoteKvikio => {
            if config.remote_direct_enabled {
                remote_direct_or_capability_fallback_source(location, config, Some("remote_kvikio_compat_alias"))
            } else {
                remote_host_buffer_fallback_source(location, config, "remote_direct_disabled", true)
            }
        },
        TableFormatParquetReadStrategyPreference::LocalCache => {
            resolve_local_cache_table_format_parquet_source(location, config, metadata)
        },
    }?;
    Ok(attach_remote_source_identity(resolved, location, config, metadata))
}

fn resolve_auto_s3_table_format_parquet_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> Result<ResolvedParquetSource> {
    if let Some(cache_root) = config.lakehouse_cache_root.as_ref() {
        let cache_path = cache_root.join(cache_relative_path(location, config, metadata));
        if !config.lakehouse_cache_read_policy.is_enabled() {
            let resolved = remote_foreground_source(location, config)?
                .with_cache_status(LakehouseCacheStatus::Bypassed)
                .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Bypassed)
                .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
                .with_lakehouse_cache_path(cache_path.display().to_string())
                .with_lakehouse_cache_fill_mode(background_fill_mode(config));
            return Ok(resolved);
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
        if config.lakehouse_cache_policy.fills_before_read() {
            reject_if_remote_direct_required(config, "lakehouse_cache_fill_before_read")?;
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
        let fill_mode = background_fill_mode(config);
        return Ok(remote_foreground_source(location, config)?
            .with_cache_status(LakehouseCacheStatus::Miss)
            .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Enabled)
            .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
            .with_lakehouse_cache_path(cache_path.display().to_string())
            .with_lakehouse_cache_fill_mode(fill_mode));
    }

    remote_foreground_source(location, config)
}

fn remote_foreground_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
) -> Result<ResolvedParquetSource> {
    if config.remote_direct_enabled {
        remote_direct_or_capability_fallback_source(location, config, None)
    } else {
        remote_host_buffer_fallback_source(location, config, "remote_direct_disabled", true)
    }
}

fn resolve_local_cache_table_format_parquet_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> Result<ResolvedParquetSource> {
    let Some(cache_root) = config.lakehouse_cache_root.as_ref() else {
        let resolved = if config.remote_direct_enabled {
            remote_direct_or_capability_fallback_source(location, config, None)?
        } else {
            remote_host_buffer_fallback_source(location, config, "remote_direct_disabled", true)?
        };
        return Ok(resolved
            .with_cache_status(LakehouseCacheStatus::Skipped)
            .with_fallback_reason("lakehouse_cache_root_unset"));
    };
    let cache_path = cache_root.join(cache_relative_path(location, config, metadata));
    if !config.lakehouse_cache_read_policy.is_enabled() {
        let resolved = if config.remote_direct_enabled {
            remote_direct_or_capability_fallback_source(location, config, None)?
        } else {
            remote_host_buffer_fallback_source(location, config, "remote_direct_disabled", true)?
        };
        return Ok(resolved
            .with_cache_status(LakehouseCacheStatus::Bypassed)
            .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Bypassed)
            .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
            .with_lakehouse_cache_path(cache_path.display().to_string())
            .with_lakehouse_cache_fill_mode(background_fill_mode(config)));
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
    if config.lakehouse_cache_policy.fills_before_read() {
        reject_if_remote_direct_required(config, "lakehouse_cache_fill_before_read")?;
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
    let fill_mode = background_fill_mode(config);
    let mut resolved = if config.remote_direct_enabled {
        remote_direct_or_capability_fallback_source(location, config, None)?
    } else {
        remote_host_buffer_fallback_source(location, config, "remote_direct_disabled", true)?
    }
    .with_cache_status(LakehouseCacheStatus::Miss)
    .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Enabled)
    .with_lakehouse_cache_config(cache_root.display().to_string(), config.lakehouse_cache_max_bytes)
    .with_lakehouse_cache_path(cache_path.display().to_string())
    .with_lakehouse_cache_fill_mode(fill_mode);
    if fill_mode == LakehouseCacheFillMode::Disabled {
        resolved = resolved.with_fallback_reason("lakehouse_cache_miss_fill_disabled");
    }
    Ok(resolved)
}

fn remote_direct_source(location: &str, config: &TableFormatParquetReadStrategyConfig) -> ResolvedParquetSource {
    ResolvedParquetSource::remote_direct(location, config.credential_mode)
        .with_remote_read_mode(config.remote_direct_read_mode)
}

fn remote_direct_or_capability_fallback_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    compatibility_reason: Option<&'static str>,
) -> Result<ResolvedParquetSource> {
    if let Some(reason) = config.remote_direct_profile_unsupported_reason {
        return remote_host_buffer_fallback_source(location, config, reason, true);
    }
    match cudf::io::parquet::remote_parquet_backend_build_status() {
        cudf::io::parquet::RemoteParquetBackendBuildStatus::KvikioRemoteIoEnabled => {},
        cudf::io::parquet::RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled => {
            return remote_host_buffer_fallback_source(location, config, "remote_direct_backend_disabled", true);
        },
        cudf::io::parquet::RemoteParquetBackendBuildStatus::Unknown => {
            return remote_host_buffer_fallback_source(location, config, "remote_direct_backend_unknown", true);
        },
    }
    let resolved = remote_direct_source(location, config);
    Ok(match compatibility_reason {
        Some(reason) => resolved.with_fallback_reason(reason),
        None => resolved,
    })
}

#[cfg(test)]
fn remote_direct_backend_disabled() -> bool {
    cudf::io::parquet::remote_parquet_backend_build_status()
        == cudf::io::parquet::RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled
}

fn remote_host_buffer_fallback_source(
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    fallback_reason: &'static str,
    ranged_enabled: bool,
) -> Result<ResolvedParquetSource> {
    reject_if_remote_direct_required(config, fallback_reason)?;
    Ok(ResolvedParquetSource::remote_host_buffer(location, object_store_reader_path(location), config.credential_mode)
        .with_remote_host_buffer_ranged_enabled(ranged_enabled && config.remote_host_buffer_ranged_enabled)
        .with_fallback_reason(fallback_reason))
}

fn reject_if_remote_direct_required(
    config: &TableFormatParquetReadStrategyConfig,
    fallback_reason: &'static str,
) -> Result<()> {
    if config.remote_direct_read_mode != RemoteParquetReadMode::RequireDirect {
        return Ok(());
    }
    Err(crate::Error::new(
        crate::ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.table_format.resolve_parquet_source",
        "remote direct read mode is required, but source resolution selected a fallback path",
    )
    .with_fact("remote_read_mode", config.remote_direct_read_mode.as_str())
    .with_fact("fallback_reason", fallback_reason))
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

fn object_store_reader_path(uri: &str) -> String {
    uri.to_string()
}

fn attach_remote_source_identity(
    resolved: ResolvedParquetSource,
    location: &str,
    config: &TableFormatParquetReadStrategyConfig,
    metadata: &TableFormatParquetSourceMetadata,
) -> ResolvedParquetSource {
    resolved
        .with_storage_domain(storage_domain_identity(location, config, metadata))
        .with_content_identity(content_identity(metadata))
        .with_optional_storage_profile_id(config.storage_profile_id.clone())
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
        .join(sanitize_cache_component(&content_identity(metadata)))
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
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        if path.is_empty() { format!("{scheme}://{authority}") } else { format!("{scheme}://{authority}/{path}") }
    } else {
        let (authority, path) = without_query.split_once('/').unwrap_or((without_query, ""));
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        if path.is_empty() { authority.to_string() } else { format!("{authority}/{path}") }
    }
}

fn content_identity(metadata: &TableFormatParquetSourceMetadata) -> String {
    if let Some(content_identity) = metadata.content_identity.as_ref().filter(|value| !value.is_empty()) {
        return content_identity.clone();
    }
    match (metadata.size_bytes, metadata.row_count) {
        (Some(size_bytes), Some(row_count)) => format!("weak_size_{size_bytes}_rows_{row_count}"),
        (Some(size_bytes), None) => format!("weak_size_{size_bytes}"),
        (None, Some(row_count)) => format!("weak_rows_{row_count}"),
        (None, None) => "weak_content_unknown".to_string(),
    }
}

fn complete_cache_hit_matches_metadata(cache_path: &Path, metadata: &TableFormatParquetSourceMetadata) -> bool {
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

fn redacted_scan_location_debug_value(location: &str) -> String {
    let safe_location = credential_free_endpoint_identity(location);
    redacted_scan_debug_value("location", &safe_location)
}

fn redacted_scan_debug_value(kind: &str, value: &str) -> String {
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
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs};

    use super::*;

    #[test]
    fn table_format_parquet_read_strategy_config_from_env_uses_catalog_endpoint_identity() {
        let env = BTreeMap::from([
            ("NEXUS_ICEBERG_S3_ENDPOINT", "http://user:secret@rustfs.local:9000/minio?sig=secret"),
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
            ("NEXUS_ICEBERG_S3_ACCESS_KEY_ID", "access"),
            ("NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY", "secret"),
        ]);
        let config =
            table_format_parquet_read_strategy_config_from_env(|name| env.get(name).map(|value| (*value).to_string()));

        assert_eq!(config.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(config.credential_mode, CredentialMode::ScopedResolved);
    }

    #[test]
    fn table_format_parquet_read_strategy_config_marks_s3_region_as_scoped() {
        let env = BTreeMap::from([("NEXUS_ICEBERG_S3_REGION", "us-west-2")]);
        let config =
            table_format_parquet_read_strategy_config_from_env(|name| env.get(name).map(|value| (*value).to_string()));

        assert_eq!(config.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(config.credential_mode, CredentialMode::ScopedResolved);

        let source = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap();
        assert_expected_remote_foreground_source(&source);
    }

    #[test]
    fn table_format_parquet_read_strategy_config_uses_storage_profile_capability() {
        let default_profile = SourceStorageProfile::s3(
            DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
            nexus_query_engine::S3SourceStorageProfile::default_chain(),
        );
        let default_config = TableFormatParquetReadStrategyConfig::default()
            .with_source_storage_profile_remote_direct_capability(&default_profile);
        assert_eq!(default_config.credential_mode, CredentialMode::SharedProcessEnv);

        let region_profile = SourceStorageProfile::s3(
            DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
            nexus_query_engine::S3SourceStorageProfile::default_chain().with_region("us-west-2"),
        );
        let region_config = TableFormatParquetReadStrategyConfig::default()
            .with_source_storage_profile_remote_direct_capability(&region_profile);
        assert_eq!(region_config.credential_mode, CredentialMode::ScopedResolved);

        let scoped_profile = SourceStorageProfile::s3(
            DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
            nexus_query_engine::S3SourceStorageProfile::default_chain()
                .with_endpoint("http://localhost:9000")
                .with_static_credentials("access", "secret", None),
        );
        let scoped_config = TableFormatParquetReadStrategyConfig::default()
            .with_source_storage_profile_remote_direct_capability(&scoped_profile);
        assert_eq!(scoped_config.credential_mode, CredentialMode::ScopedResolved);
    }

    #[test]
    fn table_format_parquet_read_strategy_config_from_env_requires_opt_in_for_ranged_host_buffer() {
        let empty = BTreeMap::<&str, &str>::new();
        let default_config = table_format_parquet_read_strategy_config_from_env(|name| {
            empty.get(name).map(|value| (*value).to_string())
        });
        assert!(!default_config.remote_host_buffer_ranged_enabled);

        let env = BTreeMap::from([(NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV, "1")]);
        let enabled_config =
            table_format_parquet_read_strategy_config_from_env(|name| env.get(name).map(|value| (*value).to_string()));
        assert!(enabled_config.remote_host_buffer_ranged_enabled);
    }

    #[test]
    fn table_format_parquet_read_strategy_config_from_env_parses_remote_direct_read_mode() {
        let env = BTreeMap::from([(NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV, "require_direct")]);
        let config =
            table_format_parquet_read_strategy_config_from_env(|name| env.get(name).map(|value| (*value).to_string()));

        assert_eq!(config.remote_direct_read_mode, RemoteParquetReadMode::RequireDirect);
    }

    #[test]
    fn fallback_content_identity_is_marked_weak() {
        assert_eq!(
            content_identity(&TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10))),
            "weak_size_4096_rows_10",
        );
        assert_eq!(
            content_identity(&TableFormatParquetSourceMetadata::default().with_size_bytes(4096)),
            "weak_size_4096",
        );
        assert_eq!(content_identity(&TableFormatParquetSourceMetadata::default().with_row_count(10)), "weak_rows_10",);
        assert_eq!(content_identity(&TableFormatParquetSourceMetadata::default()), "weak_content_unknown");
        assert_eq!(
            content_identity(&TableFormatParquetSourceMetadata::default().with_content_identity("etag:abc123")),
            "etag:abc123",
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
            .with_delete_file(TableFormatDeleteFile::new(
                delete_location,
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
    fn local_file_uri_reader_paths_are_normalized_for_local_cudf_reads() {
        assert_eq!(table_format_local_path_text("/tmp/table.parquet"), "/tmp/table.parquet");
        assert_eq!(table_format_local_path_text("file:/tmp/table.parquet"), "/tmp/table.parquet");
        assert_eq!(table_format_local_path_text("file:///tmp/table.parquet"), "/tmp/table.parquet");
        assert_eq!(table_format_local_path_text("file://localhost/tmp/table.parquet"), "/tmp/table.parquet");
    }

    fn assert_expected_remote_foreground_source(source: &ResolvedParquetSource) {
        if remote_direct_backend_disabled() {
            assert_eq!(source.read_strategy.as_str(), "remote_host_buffer");
            assert_eq!(source.decode_input.as_str(), "host_buffer");
            assert_eq!(source.fallback_reason.as_deref(), Some("remote_direct_backend_disabled"));
        } else {
            assert_eq!(source.read_strategy.as_str(), "remote_direct");
            assert_eq!(source.decode_input.as_str(), "filepath");
            assert_eq!(source.remote_read_mode, RemoteParquetReadMode::PreferDirect);
        }
    }

    #[test]
    fn s3_auto_uses_remote_direct_on_cache_miss() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            ..Default::default()
        };

        let source = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10)),
        )
        .unwrap();

        assert_expected_remote_foreground_source(&source);
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Disabled);
        assert!(source.lakehouse_cache_path.is_some());
    }

    #[test]
    fn s3_auto_require_direct_uses_remote_direct_when_backend_available() {
        let config = TableFormatParquetReadStrategyConfig {
            remote_direct_read_mode: RemoteParquetReadMode::RequireDirect,
            ..Default::default()
        };

        let result = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        );

        if remote_direct_backend_disabled() {
            let error = result.unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::InvalidExecutionPlan);
            assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
            assert_eq!(error.fact_value("fallback_reason"), Some("remote_direct_backend_disabled"));
        } else {
            let source = result.unwrap();
            assert_eq!(source.read_strategy.as_str(), "remote_direct");
            assert_eq!(source.decode_input.as_str(), "filepath");
            assert_eq!(source.remote_read_mode, RemoteParquetReadMode::RequireDirect);
        }
    }

    #[test]
    fn require_direct_rejects_remote_direct_disabled_host_buffer_fallback() {
        let config = TableFormatParquetReadStrategyConfig {
            remote_direct_enabled: false,
            remote_direct_read_mode: RemoteParquetReadMode::RequireDirect,
            ..Default::default()
        };

        let error = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap_err();

        assert_eq!(error.code(), crate::ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
        assert_eq!(error.fact_value("fallback_reason"), Some("remote_direct_disabled"));
    }

    #[test]
    fn require_direct_rejects_explicit_host_buffer_fallback() {
        let config = TableFormatParquetReadStrategyConfig {
            preference: TableFormatParquetReadStrategyPreference::HostBuffer,
            remote_direct_read_mode: RemoteParquetReadMode::RequireDirect,
            ..Default::default()
        };

        let error = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap_err();

        assert_eq!(error.code(), crate::ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("fallback_reason"), Some("explicit_remote_host_buffer_full"));
    }

    #[test]
    fn scoped_profile_remote_direct_reaches_direct_path_when_backend_available() {
        let config = TableFormatParquetReadStrategyConfig {
            credential_mode: CredentialMode::ScopedResolved,
            remote_host_buffer_ranged_enabled: true,
            ..Default::default()
        };

        let source = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap();

        if remote_direct_backend_disabled() {
            assert_eq!(source.read_strategy.as_str(), "remote_host_buffer");
            assert_eq!(source.decode_input.as_str(), "host_buffer");
            assert_eq!(source.fallback_reason.as_deref(), Some("remote_direct_backend_disabled"));
            assert!(source.remote_host_buffer_ranged_enabled);
        } else {
            assert_eq!(source.read_strategy.as_str(), "remote_direct");
            assert_eq!(source.decode_input.as_str(), "filepath");
            assert_eq!(source.credential_mode, CredentialMode::ScopedResolved);
        }
    }

    #[test]
    fn require_direct_scoped_profile_reaches_direct_path_when_backend_available() {
        let config = TableFormatParquetReadStrategyConfig {
            credential_mode: CredentialMode::ScopedResolved,
            remote_direct_read_mode: RemoteParquetReadMode::RequireDirect,
            ..Default::default()
        };

        let result = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        );

        if remote_direct_backend_disabled() {
            let error = result.unwrap_err();
            assert_eq!(error.code(), crate::ErrorCode::InvalidExecutionPlan);
            assert_eq!(error.fact_value("remote_read_mode"), Some("require_direct"));
            assert_eq!(error.fact_value("fallback_reason"), Some("remote_direct_backend_disabled"));
        } else {
            let source = result.unwrap();
            assert_eq!(source.read_strategy.as_str(), "remote_direct");
            assert_eq!(source.remote_read_mode, RemoteParquetReadMode::RequireDirect);
            assert_eq!(source.credential_mode, CredentialMode::ScopedResolved);
        }
    }

    #[test]
    fn s3_auto_prefers_complete_local_cache_hit() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            ..Default::default()
        };
        let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10));
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

        assert_eq!(source.read_strategy.as_str(), "local_cache_kvikio");
        assert_eq!(source.decode_input.as_str(), "filepath");
        assert_eq!(source.cache_status, LakehouseCacheStatus::Hit);
        assert_eq!(source.reader_location, cache_path.display().to_string());
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
        let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10));
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, b"incomplete").unwrap();

        let source = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        )
        .unwrap();

        assert_expected_remote_foreground_source(&source);
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
        let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(None, Some(10));
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

        assert_expected_remote_foreground_source(&source);
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
        let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10));
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

        assert_expected_remote_foreground_source(&source);
        assert_eq!(source.cache_status, LakehouseCacheStatus::Bypassed);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Background);
        assert_eq!(source.reader_location, "s3://bucket/table/part-0.parquet");
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn read_through_cache_miss_selects_background_fill_without_blocking_foreground_read() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            preference: TableFormatParquetReadStrategyPreference::LocalCache,
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::ReadThrough,
            ..Default::default()
        };
        let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10));
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));

        let source = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        )
        .unwrap();

        assert_expected_remote_foreground_source(&source);
        assert_eq!(source.cache_status, LakehouseCacheStatus::Miss);
        assert_eq!(source.lakehouse_cache_fill_mode, LakehouseCacheFillMode::Background);
        assert_eq!(source.lakehouse_cache_path.as_deref(), Some(cache_path.to_str().unwrap()));
    }

    #[test]
    fn explicit_host_buffer_remains_legacy_full_object_fallback() {
        let config = TableFormatParquetReadStrategyConfig {
            preference: TableFormatParquetReadStrategyPreference::HostBuffer,
            ..Default::default()
        };

        let source = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap();

        assert_eq!(source.read_strategy.as_str(), "remote_host_buffer");
        assert_eq!(source.decode_input.as_str(), "host_buffer");
        assert_eq!(source.fallback_reason.as_deref(), Some("explicit_remote_host_buffer_full"));
        assert_eq!(table_format_source_read_contract(&source), "remote_host_buffer_full");
    }

    #[test]
    fn remote_direct_gate_falls_back_after_complete_cache_hit_probe() {
        let root = tempfile::tempdir().unwrap();
        let config = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(root.path().to_path_buf()),
            remote_direct_enabled: false,
            ..Default::default()
        };
        let metadata = TableFormatParquetSourceMetadata::from_size_and_row_count(Some(4096), Some(10));
        let cache_path = root.path().join(cache_relative_path("s3://bucket/table/part-0.parquet", &config, &metadata));
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, vec![0_u8; 4096]).unwrap();

        let hit = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        )
        .unwrap();
        assert_eq!(hit.read_strategy.as_str(), "local_cache_kvikio");

        fs::remove_file(&cache_path).unwrap();
        let miss = resolve_table_format_parquet_source_with_metadata(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
            &metadata,
        )
        .unwrap();
        assert_eq!(miss.read_strategy.as_str(), "remote_host_buffer");
        assert_eq!(miss.decode_input.as_str(), "host_buffer");
        assert_eq!(miss.fallback_reason.as_deref(), Some("remote_direct_disabled"));
    }

    #[test]
    fn remote_host_buffer_ranged_gate_is_carried_on_remote_direct_fallback() {
        let config = TableFormatParquetReadStrategyConfig {
            remote_direct_enabled: false,
            remote_host_buffer_ranged_enabled: false,
            ..Default::default()
        };

        let source = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap();

        assert_eq!(source.read_strategy.as_str(), "remote_host_buffer");
        assert_eq!(source.fallback_reason.as_deref(), Some("remote_direct_disabled"));
        assert!(!source.remote_host_buffer_ranged_enabled);
        assert_eq!(table_format_source_read_contract(&source), "remote_host_buffer_full");
    }

    #[test]
    fn remote_host_buffer_ranged_contract_is_visible_in_scan_diagnostics() {
        let config = TableFormatParquetReadStrategyConfig {
            remote_direct_enabled: false,
            remote_host_buffer_ranged_enabled: true,
            ..Default::default()
        };
        let source = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap();
        let plan = TableFormatScanPlan::new(TableFormat::Iceberg).with_data_file(
            TableFormatDataFile::new(
                "part-0",
                "s3://bucket/table/part-0.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_resolved_parquet_source(source),
        );

        let diagnostics = plan.diagnostics();

        assert_eq!(diagnostics.source_read_strategy, "remote_host_buffer_ranged");
    }

    #[test]
    fn explicit_host_buffer_disables_ranged_fallback() {
        let config = TableFormatParquetReadStrategyConfig {
            preference: TableFormatParquetReadStrategyPreference::HostBuffer,
            remote_host_buffer_ranged_enabled: true,
            ..Default::default()
        };

        let source = resolve_table_format_parquet_source(
            "s3://bucket/table/part-0.parquet",
            &TableFormatReadCapability::object_store("s3"),
            &config,
        )
        .unwrap();

        assert_eq!(source.fallback_reason.as_deref(), Some("explicit_remote_host_buffer_full"));
        assert!(!source.remote_host_buffer_ranged_enabled);
    }
}

/// Scalar value available from table-format metadata, usually partition facts.
#[derive(Debug, Clone, PartialEq)]
pub enum TableFormatScalarValue {
    /// Null metadata value.
    Null,
    /// Boolean metadata value.
    Boolean(bool),
    /// Signed 32-bit integer metadata value.
    Int32(i32),
    /// Signed 64-bit integer metadata value.
    Int64(i64),
    /// Signed 128-bit integer metadata value.
    Int128(i128),
    /// Unsigned 32-bit integer metadata value.
    UInt32(u32),
    /// Unsigned 64-bit integer metadata value.
    UInt64(u64),
    /// 64-bit floating-point metadata value.
    Float64(f64),
    /// UTF-8 metadata value.
    Utf8(String),
    /// Arrow Date32 day value.
    Date32(i32),
}

impl fmt::Display for TableFormatScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("null"),
            Self::Boolean(value) => write!(f, "{value}"),
            Self::Int32(value) => write!(f, "{value}"),
            Self::Int64(value) => write!(f, "{value}"),
            Self::Int128(value) => write!(f, "{value}"),
            Self::UInt32(value) => write!(f, "{value}"),
            Self::UInt64(value) => write!(f, "{value}"),
            Self::Float64(value) => write!(f, "{value}"),
            Self::Utf8(value) => write!(f, "{value:?}"),
            Self::Date32(value) => write!(f, "{value}"),
        }
    }
}

/// Partition transform represented by table metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatPartitionTransform {
    /// Identity partition transform.
    Identity,
    /// Bucket transform with bucket count.
    Bucket {
        /// Number of buckets.
        buckets: u32,
    },
    /// Truncate transform with width.
    Truncate {
        /// Truncation width.
        width: u32,
    },
    /// Year transform.
    Year,
    /// Month transform.
    Month,
    /// Day transform.
    Day,
    /// Hour transform.
    Hour,
    /// Unknown transform identified by a stable caller-provided code.
    Unknown(String),
}

impl TableFormatPartitionTransform {
    /// Stable transform code for diagnostics.
    pub fn code(&self) -> &str {
        match self {
            Self::Identity => "identity",
            Self::Bucket { .. } => "bucket",
            Self::Truncate { .. } => "truncate",
            Self::Year => "year",
            Self::Month => "month",
            Self::Day => "day",
            Self::Hour => "hour",
            Self::Unknown(transform) => transform.as_str(),
        }
    }

    /// Whether native lowering can materialize this partition value today.
    pub fn is_native_materializable(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

/// Partition value attached to a planned data file.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatPartitionValue {
    /// Logical partition column name.
    ///
    /// Identity transforms use the source column name. Non-identity transforms
    /// use the partition field name and are retained as metadata facts only.
    pub column_name: String,
    /// Arrow data type of the partition value.
    pub data_type: DataType,
    /// Whether the logical partition column is nullable.
    pub nullable: bool,
    /// Table-format partition transform.
    pub transform: TableFormatPartitionTransform,
    /// Scalar partition value from metadata.
    pub value: TableFormatScalarValue,
}

impl TableFormatPartitionValue {
    /// Builds one partition value fact.
    pub fn new(
        column_name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
        transform: TableFormatPartitionTransform,
        value: TableFormatScalarValue,
    ) -> Self {
        Self { column_name: column_name.into(), data_type, nullable, transform, value }
    }

    /// Column name that may be materialized natively from this partition value.
    pub(crate) fn native_materialized_column_name(&self) -> &str {
        self.column_name.as_str()
    }
}

/// How a logical table column is represented by the native source boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatColumnSource {
    /// Logical column maps directly to a top-level file column.
    DirectTopLevelFileColumn,
    /// Logical column is supplied from an identity partition value.
    IdentityPartitionValue,
    /// Nullable logical column is absent from the file and materialized as nulls.
    MissingNullableColumn,
    /// Logical column requires nested projection not represented by native source facts.
    UnsupportedNestedProjection,
}

impl TableFormatColumnSource {
    /// Stable source code for diagnostics.
    pub const fn code(self) -> &'static str {
        match self {
            Self::DirectTopLevelFileColumn => "direct_top_level_file_column",
            Self::IdentityPartitionValue => "identity_partition_value",
            Self::MissingNullableColumn => "missing_nullable_column",
            Self::UnsupportedNestedProjection => "unsupported_nested_projection",
        }
    }
}

/// Logical role of a column in the resolved scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatColumnRole {
    /// Column is projected into query output.
    Projected,
    /// Column is needed only for filtering.
    PureFilter,
}

impl TableFormatColumnRole {
    /// Stable role code for diagnostics.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Projected => "projected",
            Self::PureFilter => "pure_filter",
        }
    }
}

/// Mapping from table-format field facts to a native source column.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatColumnMapping {
    /// Stable table-format field id, when available.
    pub field_id: Option<i32>,
    /// Logical column name visible to query planning.
    pub logical_name: String,
    /// Physical source column name, when the value comes from a file column.
    pub source_name: Option<String>,
    /// Arrow data type exposed at the native source boundary.
    pub data_type: DataType,
    /// Whether the logical column is nullable.
    pub nullable: bool,
    /// How the logical column is represented by native source facts.
    pub source: TableFormatColumnSource,
    /// Whether the column is projected or filter-only.
    pub role: TableFormatColumnRole,
}

impl TableFormatColumnMapping {
    /// Builds one logical-to-source column mapping.
    pub fn new(
        field_id: Option<i32>,
        logical_name: impl Into<String>,
        source_name: Option<impl Into<String>>,
        data_type: DataType,
        nullable: bool,
        source: TableFormatColumnSource,
        role: TableFormatColumnRole,
    ) -> Self {
        Self {
            field_id,
            logical_name: logical_name.into(),
            source_name: source_name.map(Into::into),
            data_type,
            nullable,
            source,
            role,
        }
    }
}

/// Resolved data file from a table-format scan task.
#[derive(Clone, PartialEq)]
pub struct TableFormatDataFile {
    /// Stable file id within the resolved scan plan.
    pub id: String,
    /// Data file location. Diagnostics should avoid exposing credential-bearing URLs.
    pub location: String,
    /// Advertised data file format.
    pub format: TableFormatDataFileFormat,
    /// Optional file byte size from metadata.
    pub size_bytes: Option<u64>,
    /// Optional file row count from metadata.
    pub row_count: Option<u64>,
    /// Optional selected row-group indexes.
    pub row_groups: Option<Vec<usize>>,
    /// Required read capability for this location.
    pub read_capability: TableFormatReadCapability,
    /// Resolved Parquet read strategy for native source lowering.
    pub resolved_parquet_source: Option<ResolvedParquetSource>,
    /// Partition values attached to this data file.
    pub partition_values: Vec<TableFormatPartitionValue>,
}

impl fmt::Debug for TableFormatDataFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableFormatDataFile")
            .field("id", &redacted_scan_debug_value("data_file_id", &self.id))
            .field("location", &redacted_scan_location_debug_value(&self.location))
            .field("format", &self.format)
            .field("size_bytes", &self.size_bytes)
            .field("row_count", &self.row_count)
            .field("row_groups", &self.row_groups)
            .field("read_capability", &self.read_capability)
            .field("resolved_parquet_source", &self.resolved_parquet_source)
            .field("partition_values", &self.partition_values)
            .finish()
    }
}

impl TableFormatDataFile {
    /// Builds a data-file fact without optional size/count/partition metadata.
    pub fn new(
        id: impl Into<String>,
        location: impl Into<String>,
        format: TableFormatDataFileFormat,
        read_capability: TableFormatReadCapability,
    ) -> Self {
        Self {
            id: id.into(),
            location: location.into(),
            format,
            size_bytes: None,
            row_count: None,
            row_groups: None,
            read_capability,
            resolved_parquet_source: None,
            partition_values: Vec::new(),
        }
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

    /// Records selected row-group indexes.
    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.row_groups = Some(row_groups);
        self
    }

    /// Records the resolved Parquet source strategy for this data file.
    pub fn with_resolved_parquet_source(mut self, resolved_source: ResolvedParquetSource) -> Self {
        self.resolved_parquet_source = Some(resolved_source);
        self
    }

    /// Records partition values attached to this file.
    pub fn with_partition_values(mut self, partition_values: Vec<TableFormatPartitionValue>) -> Self {
        self.partition_values = partition_values;
        self
    }

    /// Whether the file can be represented by direct native Parquet reads.
    pub fn is_direct_native_parquet_representable(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet && self.read_capability.is_direct_native_parquet_capable()
    }

    /// Whether the file can be represented by native Parquet source reads.
    pub fn is_native_parquet_representable(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet && self.read_capability.is_native_parquet_capable()
    }

    /// Whether the file can be represented as a local native Parquet source.
    pub fn is_native_local_parquet_representable(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet && self.read_capability.is_native_local_parquet_capable()
    }
}

/// Row-level delete semantics advertised by a table-format delete file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatDeleteFileContent {
    /// Iceberg-style position deletes.
    PositionDeletes,
    /// Iceberg-style equality deletes.
    EqualityDeletes,
    /// Delete-vector representation.
    DeleteVector,
    /// Another delete content type identified by a stable caller-provided code.
    Unknown(String),
}

impl TableFormatDeleteFileContent {
    /// Stable delete-content code for diagnostics.
    pub fn code(&self) -> &str {
        match self {
            Self::PositionDeletes => "position_deletes",
            Self::EqualityDeletes => "equality_deletes",
            Self::DeleteVector => "delete_vector",
            Self::Unknown(content) => content.as_str(),
        }
    }
}

/// Delete file fact preserved by table-format planning.
#[derive(Clone, PartialEq)]
pub struct TableFormatDeleteFile {
    /// Stable delete-file id within the resolved scan plan.
    pub id: String,
    /// Delete file location. Diagnostics should avoid exposing credential-bearing URLs.
    pub location: String,
    /// Delete file format.
    pub format: TableFormatDataFileFormat,
    /// Row-level delete semantics represented by this file.
    pub content: TableFormatDeleteFileContent,
    /// Optional delete file byte size from metadata.
    pub size_bytes: Option<u64>,
    /// Optional delete row count from metadata.
    pub row_count: Option<u64>,
}

impl fmt::Debug for TableFormatDeleteFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableFormatDeleteFile")
            .field("id", &redacted_scan_debug_value("delete_file_id", &self.id))
            .field("location", &redacted_scan_location_debug_value(&self.location))
            .field("format", &self.format)
            .field("content", &self.content)
            .field("size_bytes", &self.size_bytes)
            .field("row_count", &self.row_count)
            .finish()
    }
}

impl TableFormatDeleteFile {
    /// Builds a delete-file fact without optional size/count metadata.
    pub fn new(
        id: impl Into<String>,
        location: impl Into<String>,
        format: TableFormatDataFileFormat,
        content: TableFormatDeleteFileContent,
    ) -> Self {
        Self { id: id.into(), location: location.into(), format, content, size_bytes: None, row_count: None }
    }

    /// Records delete-file size in bytes.
    pub fn with_size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    /// Records delete-file row count.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }
}

/// Unsupported scan fact category with a stable fallback reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatUnsupportedFeatureCode {
    /// Generic unsupported table-format scan.
    TableFormatUnsupported,
    /// Table-format metadata planning failed.
    TableFormatPlanningFailed,
    /// Iceberg data file is not Parquet.
    IcebergNonParquetDataFile,
    /// Iceberg delete files are present.
    IcebergDeleteFilesUnsupported,
    /// Iceberg field-id mapping is required but unsupported.
    IcebergFieldIdMappingUnsupported,
    /// Iceberg name mapping is required but unsupported.
    IcebergNameMappingUnsupported,
    /// Iceberg nested projection is required but unsupported.
    IcebergNestedProjectionUnsupported,
    /// Iceberg partition transform cannot be materialized natively.
    IcebergPartitionTransformUnsupported,
    /// Iceberg residual predicate is not represented by native scan facts.
    IcebergResidualPredicateUnsupported,
    /// Iceberg row-selection/page-index semantics are unsupported.
    IcebergRowSelectionUnsupported,
    /// Iceberg object-store read cannot be represented by native source facts.
    IcebergObjectStoreReadUnsupported,
    /// Iceberg snapshot selection is unsupported or invalid.
    IcebergSnapshotUnsupported,
}

impl TableFormatUnsupportedFeatureCode {
    /// Stable fallback reason associated with this unsupported feature.
    pub const fn fallback_reason(self) -> FallbackReason {
        match self {
            Self::TableFormatUnsupported => FallbackReason::TableFormatUnsupported,
            Self::TableFormatPlanningFailed => FallbackReason::TableFormatPlanningFailed,
            Self::IcebergNonParquetDataFile => FallbackReason::IcebergNonParquetDataFile,
            Self::IcebergDeleteFilesUnsupported => FallbackReason::IcebergDeleteFilesUnsupported,
            Self::IcebergFieldIdMappingUnsupported => FallbackReason::IcebergFieldIdMappingUnsupported,
            Self::IcebergNameMappingUnsupported => FallbackReason::IcebergNameMappingUnsupported,
            Self::IcebergNestedProjectionUnsupported => FallbackReason::IcebergNestedProjectionUnsupported,
            Self::IcebergPartitionTransformUnsupported => FallbackReason::IcebergPartitionTransformUnsupported,
            Self::IcebergResidualPredicateUnsupported => FallbackReason::IcebergResidualPredicateUnsupported,
            Self::IcebergRowSelectionUnsupported => FallbackReason::IcebergRowSelectionUnsupported,
            Self::IcebergObjectStoreReadUnsupported => FallbackReason::IcebergObjectStoreReadUnsupported,
            Self::IcebergSnapshotUnsupported => FallbackReason::IcebergSnapshotUnsupported,
        }
    }

    /// Stable fallback reason code.
    pub const fn code(self) -> &'static str {
        self.fallback_reason().code()
    }
}

/// One unsupported table-format fact retained for reporting and rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatUnsupportedFeature {
    /// Unsupported feature category.
    pub code: TableFormatUnsupportedFeatureCode,
    /// Human-readable unsupported feature detail.
    pub message: String,
    /// Optional file id when the unsupported feature is file-local.
    pub file_id: Option<String>,
}

impl TableFormatUnsupportedFeature {
    /// Builds a scan-level unsupported feature.
    pub fn new(code: TableFormatUnsupportedFeatureCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), file_id: None }
    }

    /// Builds a file-local unsupported feature.
    pub fn for_file(
        code: TableFormatUnsupportedFeatureCode,
        file_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self { code, message: message.into(), file_id: Some(file_id.into()) }
    }

    /// Stable fallback reason associated with this feature.
    pub fn fallback_reason(&self) -> FallbackReason {
        self.code.fallback_reason()
    }

    /// Stable fallback reason code.
    pub fn code(&self) -> &'static str {
        self.code.code()
    }
}

/// Native direct-device-read disposition for a resolved table-format scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatDirectDeviceReadDisposition {
    /// Scan facts are eligible for direct native device reads.
    Eligible,
    /// Scan facts must be rejected before direct native device reads.
    Rejected,
}

impl TableFormatDirectDeviceReadDisposition {
    /// Stable disposition code for diagnostics.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::Rejected => "rejected",
        }
    }
}

/// Credential-free scan diagnostics intended for tests and report rows.
///
/// `planned_bytes = None` means the table metadata did not expose a reliable
/// byte total, not that the plan is empty. String fields are stable report
/// vocabulary and should avoid paths, tokens, or other sensitive values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatScanDiagnostics {
    /// Stable table-format code.
    pub table_format: String,
    /// Snapshot id selected by planning, if known.
    pub snapshot_id: Option<String>,
    /// Stable catalog label, if known.
    pub catalog_type: Option<String>,
    /// Number of planned data files.
    pub planned_data_file_count: usize,
    /// Number of planned delete files.
    pub planned_delete_file_count: usize,
    /// Total planned data-file bytes, or `None` if any file size is unknown.
    pub planned_bytes: Option<u64>,
    /// Number of files admitted to native source lowering.
    pub admitted_native_source_file_count: usize,
    /// Stable unsupported-feature codes retained by planning.
    pub rejected_feature_list: Vec<String>,
    /// Comma-separated stable read-capability codes.
    pub native_source_read_capability: String,
    /// Comma-separated high-level source read strategies.
    pub source_read_strategy: String,
    /// Comma-separated cuDF decode input kinds.
    pub source_decode_input: String,
    /// Comma-separated credential modes.
    pub credential_mode: String,
    /// Comma-separated local byte-cache statuses.
    pub lakehouse_cache_status: String,
    /// Comma-separated foreground lakehouse cache-read modes.
    pub lakehouse_cache_read_mode: String,
    /// Comma-separated complete-object lakehouse cache fill modes.
    pub lakehouse_cache_fill_mode: String,
    /// Comma-separated stable source fallback reasons.
    pub source_fallback_reason: String,
    /// Direct device-read eligibility.
    pub direct_device_read_disposition: TableFormatDirectDeviceReadDisposition,
}

impl TableFormatScanDiagnostics {
    /// Credential-free facts suitable for structured errors and tests.
    pub fn facts(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("table_format", self.table_format.clone()),
            ("snapshot_id", self.snapshot_id.clone().unwrap_or_default()),
            ("catalog_type", self.catalog_type.clone().unwrap_or_default()),
            ("planned_data_file_count", self.planned_data_file_count.to_string()),
            ("planned_delete_file_count", self.planned_delete_file_count.to_string()),
            ("planned_bytes", self.planned_bytes.map(|bytes| bytes.to_string()).unwrap_or_default()),
            ("admitted_native_source_file_count", self.admitted_native_source_file_count.to_string()),
            ("rejected_feature_list", self.rejected_feature_list.join(",")),
            ("native_source_read_capability", self.native_source_read_capability.clone()),
            ("source_read_strategy", self.source_read_strategy.clone()),
            ("source_decode_input", self.source_decode_input.clone()),
            ("credential_mode", self.credential_mode.clone()),
            ("lakehouse_cache_status", self.lakehouse_cache_status.clone()),
            ("lakehouse_cache_read_mode", self.lakehouse_cache_read_mode.clone()),
            ("lakehouse_cache_fill_mode", self.lakehouse_cache_fill_mode.clone()),
            ("source_fallback_reason", self.source_fallback_reason.clone()),
            ("direct_device_read_disposition", self.direct_device_read_disposition.code().to_string()),
        ])
    }
}

impl fmt::Display for TableFormatScanDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, (key, value)) in self.facts().iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{key}={value}")?;
        }
        Ok(())
    }
}

/// Resolved table-format scan facts consumed by later native lowering stages.
///
/// The plan is metadata-only: data files, delete files, projected columns, and
/// pure-filter columns are facts for later source lowering. Unsupported
/// features remain attached so the adapter can produce structured fallback
/// reasons instead of silently dropping table-format semantics. `scan_limit`
/// is an optional source-level upper bound, not a final SQL `LIMIT`; when a
/// parent inexact filter remains, final row limiting is still represented by a
/// separate native or DataFusion limit operator.
#[derive(Clone, PartialEq)]
pub struct TableFormatScanPlan {
    /// Table format that produced these scan facts.
    pub format: TableFormat,
    /// Snapshot id selected by planning, if known.
    pub snapshot_id: Option<String>,
    /// Stable catalog label, if known.
    pub catalog_type: Option<String>,
    /// Optional source-level row upper bound supplied during table-format planning.
    pub scan_limit: Option<usize>,
    /// Logical table schema after projection.
    pub logical_schema: SchemaRef,
    /// Schema expected by native source lowering.
    pub native_source_schema: SchemaRef,
    /// Planned data files.
    pub data_files: Vec<TableFormatDataFile>,
    /// Planned delete files that native admission must currently reject.
    pub delete_files: Vec<TableFormatDeleteFile>,
    /// Output column mappings.
    pub projected_columns: Vec<TableFormatColumnMapping>,
    /// Filter-only column mappings.
    pub pure_filter_columns: Vec<TableFormatColumnMapping>,
    /// Unsupported features retained for structured fallback.
    pub unsupported_features: Vec<TableFormatUnsupportedFeature>,
}

impl fmt::Debug for TableFormatScanPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableFormatScanPlan")
            .field("format", &self.format)
            .field("snapshot_id", &self.snapshot_id)
            .field("catalog_type", &self.catalog_type)
            .field("scan_limit", &self.scan_limit)
            .field("logical_schema", &self.logical_schema)
            .field("native_source_schema", &self.native_source_schema)
            .field("data_files", &self.data_files)
            .field("delete_files", &self.delete_files)
            .field("projected_columns", &self.projected_columns)
            .field("pure_filter_columns", &self.pure_filter_columns)
            .field("unsupported_features", &self.unsupported_features)
            .finish()
    }
}

impl TableFormatScanPlan {
    /// Creates an empty metadata-only scan plan for one table format.
    pub fn new(format: TableFormat) -> Self {
        let empty_schema = Arc::new(Schema::empty());
        Self {
            format,
            snapshot_id: None,
            catalog_type: None,
            scan_limit: None,
            logical_schema: Arc::clone(&empty_schema),
            native_source_schema: empty_schema,
            data_files: Vec::new(),
            delete_files: Vec::new(),
            projected_columns: Vec::new(),
            pure_filter_columns: Vec::new(),
            unsupported_features: Vec::new(),
        }
    }

    /// Records the selected table snapshot id when the format exposes one.
    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    /// Records a stable catalog label such as `glue` for diagnostics.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Records an optional source-level row upper bound.
    pub fn with_scan_limit(mut self, scan_limit: Option<usize>) -> Self {
        self.scan_limit = scan_limit;
        self
    }

    /// Records the logical table schema after table-format projection.
    pub fn with_logical_schema(mut self, schema: SchemaRef) -> Self {
        self.logical_schema = schema;
        self
    }

    /// Records the physical/native source schema expected by native lowering.
    pub fn with_native_source_schema(mut self, schema: SchemaRef) -> Self {
        self.native_source_schema = schema;
        self
    }

    /// Adds one planned data file fact.
    pub fn with_data_file(mut self, file: TableFormatDataFile) -> Self {
        self.data_files.push(file);
        self
    }

    /// Adds one delete-file fact that native admission must currently reject.
    pub fn with_delete_file(mut self, file: TableFormatDeleteFile) -> Self {
        self.delete_files.push(file);
        self
    }

    /// Adds one projected output column mapping.
    pub fn with_projected_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.projected_columns.push(column);
        self
    }

    /// Adds one column needed only for source filtering.
    pub fn with_pure_filter_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.pure_filter_columns.push(column);
        self
    }

    /// Adds one unsupported table-format feature retained for structured fallback.
    pub fn with_unsupported_feature(mut self, feature: TableFormatUnsupportedFeature) -> Self {
        self.unsupported_features.push(feature);
        self
    }

    /// Number of planned data files.
    pub fn planned_data_file_count(&self) -> usize {
        self.data_files.len()
    }

    /// Number of planned delete files.
    pub fn planned_delete_file_count(&self) -> usize {
        self.delete_files.len()
    }

    /// Total planned data-file bytes when every file size is known.
    ///
    /// Returns `None` if any data file lacks a byte size; `None` means
    /// "unknown total", not "zero bytes".
    pub fn planned_bytes(&self) -> Option<u64> {
        self.data_files
            .iter()
            .map(|file| file.size_bytes)
            .try_fold(0u64, |total, size| size.map(|size| total.saturating_add(size)))
    }

    /// All-or-nothing native-admission predicate for the current scan plan.
    ///
    /// A plan is admissible only when it has at least one native-readable
    /// Parquet data file, no delete files, and no retained unsupported
    /// features. Object-store S3 files are native-readable through the
    /// host-buffer source pipeline, not through direct device reads.
    pub fn is_native_admissible(&self) -> bool {
        self.delete_files.is_empty()
            && self.unsupported_features.is_empty()
            && !self.data_files.is_empty()
            && self.data_files.iter().all(TableFormatDataFile::is_native_parquet_representable)
    }

    /// Number of source files native lowering may consume.
    ///
    /// This returns zero unless the whole scan is native-admissible.
    pub fn admitted_native_source_file_count(&self) -> usize {
        if self.is_native_admissible() { self.data_files.len() } else { 0 }
    }

    /// Stable fallback reasons for retained unsupported features.
    pub fn unsupported_fallback_reasons(&self) -> Vec<FallbackReason> {
        self.unsupported_features.iter().map(TableFormatUnsupportedFeature::fallback_reason).collect()
    }

    /// Stable unsupported-feature codes for diagnostics/report rows.
    pub fn rejected_feature_codes(&self) -> Vec<String> {
        self.unsupported_features.iter().map(|feature| feature.code().to_string()).collect()
    }

    /// Direct device-read disposition derived from native admissibility.
    pub fn direct_device_read_disposition(&self) -> TableFormatDirectDeviceReadDisposition {
        if self.is_native_admissible()
            && self.data_files.iter().all(TableFormatDataFile::is_direct_native_parquet_representable)
        {
            TableFormatDirectDeviceReadDisposition::Eligible
        } else {
            TableFormatDirectDeviceReadDisposition::Rejected
        }
    }

    /// Comma-separated stable read-capability codes across data files.
    pub fn native_source_read_capability_summary(&self) -> String {
        let mut capabilities =
            self.data_files.iter().map(|file| file.read_capability.code().to_string()).collect::<Vec<_>>();
        capabilities.sort();
        capabilities.dedup();
        capabilities.join(",")
    }

    /// Comma-separated high-level source read strategies across data files.
    pub fn source_read_strategy_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(table_format_source_read_contract)),
        )
    }

    /// Comma-separated decode input kinds across data files.
    pub fn source_decode_input_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(|source| source.decode_input.as_str())),
        )
    }

    /// Comma-separated credential modes across data files.
    pub fn credential_mode_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(|source| source.credential_mode.as_str())),
        )
    }

    /// Comma-separated lakehouse byte-cache statuses across data files.
    pub fn lakehouse_cache_status_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(|source| source.cache_status.as_str())),
        )
    }

    /// Comma-separated foreground lakehouse cache-read modes across data files.
    pub fn lakehouse_cache_read_mode_summary(&self) -> String {
        summarize_resolved_source_value(self.data_files.iter().filter_map(|file| {
            file.resolved_parquet_source.as_ref().map(|source| source.lakehouse_cache_read_mode.as_str())
        }))
    }

    /// Comma-separated lakehouse complete-object cache fill modes across data files.
    pub fn lakehouse_cache_fill_mode_summary(&self) -> String {
        summarize_resolved_source_value(self.data_files.iter().filter_map(|file| {
            file.resolved_parquet_source.as_ref().map(|source| source.lakehouse_cache_fill_mode.as_str())
        }))
    }

    /// Comma-separated stable source fallback reasons across data files.
    pub fn source_fallback_reason_summary(&self) -> String {
        summarize_resolved_source_fallback_reason(self.data_files.iter().filter_map(|file| {
            file.resolved_parquet_source.as_ref().map(|source| source.fallback_reason.as_deref().unwrap_or("none"))
        }))
    }

    /// Builds credential-free scan diagnostics for tests and report rows.
    pub fn diagnostics(&self) -> TableFormatScanDiagnostics {
        TableFormatScanDiagnostics {
            table_format: self.format.code().to_string(),
            snapshot_id: self.snapshot_id.clone(),
            catalog_type: self.catalog_type.clone(),
            planned_data_file_count: self.planned_data_file_count(),
            planned_delete_file_count: self.planned_delete_file_count(),
            planned_bytes: self.planned_bytes(),
            admitted_native_source_file_count: self.admitted_native_source_file_count(),
            rejected_feature_list: self.rejected_feature_codes(),
            native_source_read_capability: self.native_source_read_capability_summary(),
            source_read_strategy: self.source_read_strategy_summary(),
            source_decode_input: self.source_decode_input_summary(),
            credential_mode: self.credential_mode_summary(),
            lakehouse_cache_status: self.lakehouse_cache_status_summary(),
            lakehouse_cache_read_mode: self.lakehouse_cache_read_mode_summary(),
            lakehouse_cache_fill_mode: self.lakehouse_cache_fill_mode_summary(),
            source_fallback_reason: self.source_fallback_reason_summary(),
            direct_device_read_disposition: self.direct_device_read_disposition(),
        }
    }
}

fn summarize_resolved_source_value<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    if values.is_empty() { "unknown".to_string() } else { values.join(",") }
}

fn summarize_resolved_source_fallback_reason<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    if values.is_empty() { "none".to_string() } else { values.join(",") }
}

fn table_format_source_read_contract(source: &ResolvedParquetSource) -> &'static str {
    match source.read_strategy {
        nexus_query_engine::ParquetReadStrategy::LocalCacheKvikio
            if source.cache_status == LakehouseCacheStatus::Hit =>
        {
            "local_cache_hit"
        },
        nexus_query_engine::ParquetReadStrategy::RemoteHostBuffer if source.remote_host_buffer_ranged_enabled => {
            "remote_host_buffer_ranged"
        },
        nexus_query_engine::ParquetReadStrategy::RemoteHostBuffer => "remote_host_buffer_full",
        _ => source.read_strategy.as_str(),
    }
}
