//! DataFusion-free native source model and reader-predicate contracts.

use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::new_empty_array,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};

#[cfg(test)]
use super::diagnostics::redacted_object_store_path_diagnostic;
use super::{
    diagnostics::{
        LocalParquetDiagnosticColumns, PredicateDisposition, RejectedSourceFeature, SourceDiagnostics,
        ZeroColumnCountDisposition, files_have_proven_direct_device_read, object_store_diagnostic_id,
        redacted_debug_option, redacted_debug_value, redacted_stable_id_diagnostic, stable_fnv1a64,
    },
    reader_predicate_decimal_raw_for_integer_value, reader_predicate_expr_references_decimal_column,
    reader_predicate_integer_literal_raw, reader_predicate_rescale_decimal_raw, reader_predicate_unsigned_literal_raw,
};
use crate::{
    datatypes::{
        arrow_decimal_scale_to_cudf_scale, cudf_dtype_from_arrow_type, is_supported_arrow_type,
        is_supported_list_carry_type, is_timestamp_with_timezone,
    },
    expr::{BinaryOp, Expr, LiteralValue},
};

/// Synthetic output column used for Parquet plans that only need a row count.
///
/// A zero-column count source reads no physical columns and produces row counts
/// from file or row-group metadata instead of projecting normal source columns.
pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__cudf_zero_column_row_count";

/// Explicit location/read-capability model for native Parquet sources.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ParquetSourceLocation {
    /// Local filesystem path readable by cuDF local Parquet readers.
    LocalPath(PathBuf),
    /// Object-store URI plus parsed scheme.
    ObjectStoreUri {
        /// Lowercase object-store scheme such as `s3`.
        scheme: String,
        /// Original object-store URI.
        uri: String,
    },
}

impl fmt::Debug for ParquetSourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalPath(path) => f
                .debug_tuple("LocalPath")
                .field(&redacted_debug_value("path", path.display().to_string().as_str()))
                .finish(),
            Self::ObjectStoreUri { scheme, uri } => f
                .debug_struct("ObjectStoreUri")
                .field("scheme", scheme)
                .field("uri", &redacted_debug_value("uri", uri))
                .finish(),
        }
    }
}

impl ParquetSourceLocation {
    /// Build a local filesystem source location.
    pub fn local_path(path: impl Into<PathBuf>) -> Self {
        Self::LocalPath(path.into())
    }

    /// Build a supported object-store URI location.
    ///
    /// Currently only `s3://...` is accepted. Errors are stable reason strings
    /// such as `missing_uri_scheme` or `unsupported_uri_scheme:<scheme>`.
    pub fn object_store_uri(uri: impl Into<String>) -> Result<Self, String> {
        let uri = uri.into();
        let Some((scheme, _)) = uri.split_once("://") else {
            return Err("missing_uri_scheme".to_string());
        };
        if scheme.is_empty() {
            return Err("empty_uri_scheme".to_string());
        }
        let scheme = scheme.to_ascii_lowercase();
        if scheme != "s3" {
            return Err(format!("unsupported_uri_scheme:{scheme}"));
        }
        Ok(Self::ObjectStoreUri { scheme, uri })
    }

    /// Path or URI string passed to lower-level readers.
    pub fn reader_path(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::ObjectStoreUri { uri, .. } => uri.clone(),
        }
    }

    /// Stable read-capability diagnostic string.
    pub fn read_capability(&self) -> String {
        match self {
            Self::LocalPath(_) => "local_file_system".to_string(),
            Self::ObjectStoreUri { scheme, .. } => format!("object_store:{scheme}"),
        }
    }

    /// Whether this location is a local filesystem path.
    pub fn is_local_path(&self) -> bool {
        matches!(self, Self::LocalPath(_))
    }

    /// Borrow the local path when this location is local.
    pub fn as_local_path(&self) -> Option<&Path> {
        match self {
            Self::LocalPath(path) => Some(path.as_path()),
            Self::ObjectStoreUri { .. } => None,
        }
    }

    /// Display wrapper for path/URI diagnostics.
    pub fn display(&self) -> impl fmt::Display + '_ {
        self
    }

    /// Redacted diagnostic location. Local paths are preserved; object-store
    /// URIs are represented by scheme plus stable hash only.
    pub fn diagnostic_location(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::ObjectStoreUri { scheme, uri } => object_store_diagnostic_id(scheme, uri),
        }
    }

    pub(crate) fn diagnostic_stable_id(&self, stable_id: &str) -> String {
        match self {
            Self::LocalPath(_) => redacted_stable_id_diagnostic(stable_id),
            Self::ObjectStoreUri { scheme, .. } => {
                format!("object_store:{scheme}:stable_id_fnv64:{:016x}", stable_fnv1a64(stable_id.as_bytes()))
            },
        }
    }
}

impl fmt::Display for ParquetSourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalPath(path) => write!(f, "{}", path.display()),
            Self::ObjectStoreUri { uri, .. } => f.write_str(uri),
        }
    }
}

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
    /// The referenced profile lives on [`crate::ExecutionContext`]. Keeping only
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

/// Native query-engine source description.
///
/// Frontends lower scan nodes into this enum after they have resolved file
/// locations, emitted source schema, schema mapping, projected column roles,
/// filter-only column roles, and source diagnostics. The executor should not
/// need frontend-specific scan metadata.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum SourceNode {
    /// Local filesystem Parquet input.
    LocalParquet(LocalParquetSource),
    /// Already-materialized GPU input identified by a frontend-provided name.
    GpuResident(GpuResidentSource),
    /// Host Arrow literal batches imported into GPU memory at execution time.
    LiteralBatch(LiteralBatchSource),
}

impl SourceNode {
    /// Returns the native output schema exposed by this source node.
    pub fn output_schema(&self) -> &SchemaRef {
        match self {
            Self::LocalParquet(source) => source.output_schema(),
            Self::GpuResident(source) => &source.schema,
            Self::LiteralBatch(source) => &source.schema,
        }
    }
}

/// Literal host Arrow batches used as a native source.
///
/// This source is for SQL roots such as `VALUES`, `SELECT 1`, and typed empty
/// inputs. Execution imports the batches into the query runtime explicitly; it
/// does not read files or call back into a frontend executor.
#[derive(Debug, Clone, PartialEq)]
pub struct LiteralBatchSource {
    /// Schema exposed by this source.
    pub schema: SchemaRef,
    /// Host Arrow batches in source order.
    pub batches: Vec<RecordBatch>,
}

impl LiteralBatchSource {
    /// Creates a literal source from already-typed host Arrow batches.
    pub fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self { schema, batches }
    }

    /// Creates a one-row, zero-column placeholder source.
    pub fn placeholder_row() -> Self {
        let schema = Arc::new(Schema::empty());
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&schema),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .expect("one-row empty RecordBatch is valid");
        Self::new(schema, vec![batch])
    }

    /// Creates a zero-row source with the provided schema.
    pub fn empty(schema: SchemaRef) -> Self {
        let columns = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect::<Vec<_>>();
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(0)),
        )
        .expect("zero-row RecordBatch must match its schema");
        Self::new(schema, vec![batch])
    }

    /// Total row count across all batches.
    pub fn row_count(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

/// Source for data that already resides on the GPU.
///
/// `input_name` is resolved by the embedding runtime. `schema` is the native
/// output schema that downstream query IR nodes consume.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuResidentSource {
    /// Runtime input handle supplied by the frontend or embedding application.
    pub input_name: String,
    /// Schema of the GPU-resident input as seen by the native plan.
    pub schema: SchemaRef,
}

impl GpuResidentSource {
    /// Creates a GPU-resident source with the provided runtime name and schema.
    pub fn new(input_name: impl Into<String>, schema: SchemaRef) -> Self {
        Self { input_name: input_name.into(), schema }
    }
}

fn schema_column_indices(schema: &SchemaRef, columns: &[String]) -> Vec<usize> {
    columns.iter().filter_map(|column| schema.index_of(column).ok()).collect()
}

/// Partition column materialized from frontend file metadata.
///
/// Partition columns are synthetic source columns: they are not present in the
/// Parquet file schema, but they may be read, projected, or used by filters as
/// if they came from the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionColumn {
    /// Column name exposed to native expressions.
    pub name: String,
    /// Arrow type of the partition value.
    pub data_type: DataType,
    /// Whether the partition value may be null.
    pub nullable: bool,
}

impl PartitionColumn {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self { name: name.into(), data_type, nullable }
    }
}

/// Nullable logical column absent from the physical Parquet file schema.
///
/// Missing columns are explicit source facts. They are materialized as typed
/// null GPU columns after the file/chunk row count is known and before
/// source-local filters or projections run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingColumn {
    /// Column name exposed to native expressions.
    pub name: String,
    /// Arrow type of the typed null column.
    pub data_type: DataType,
    /// Whether the logical table column is nullable.
    pub nullable: bool,
}

impl MissingColumn {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self { name: name.into(), data_type, nullable }
    }
}

/// Per-file Iceberg field-id proof for one physical Parquet read column.
///
/// The logical output name can differ from the physical Parquet column name
/// when schema evolution renamed a top-level field. Execution still reads a
/// normal Parquet projection by physical name, while source output columns map
/// the decoded read column back to the logical name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParquetFieldIdReadBinding {
    /// Iceberg field id that was matched against Parquet footer metadata.
    pub logical_field_id: i32,
    /// Logical column name exposed by the table schema.
    pub logical_name: String,
    /// Physical top-level Parquet column name in this file.
    pub physical_name: String,
    /// Physical top-level Parquet root index in this file.
    pub physical_root_index: usize,
    /// Nullability recorded in the physical Parquet/Arrow footer schema.
    pub nullable: bool,
    /// Arrow data type recorded in the physical Parquet/Arrow footer schema.
    pub data_type: DataType,
}

impl ParquetFieldIdReadBinding {
    /// Builds one footer-proven field-id read binding.
    pub fn new(
        logical_field_id: i32,
        logical_name: impl Into<String>,
        physical_name: impl Into<String>,
        physical_root_index: usize,
        nullable: bool,
        data_type: DataType,
    ) -> Self {
        Self {
            logical_field_id,
            logical_name: logical_name.into(),
            physical_name: physical_name.into(),
            physical_root_index,
            nullable,
            data_type,
        }
    }
}

/// Iceberg position-delete Parquet file associated with one data file.
///
/// The delete file is read as a normal native Parquet source and filtered by
/// Iceberg's `file_path` column before anti-joining data rows by file-global
/// `pos`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParquetPositionDeleteFile {
    /// Stable delete-file id for diagnostics.
    pub id: String,
    /// Explicit source location for the delete Parquet file.
    pub location: ParquetSourceLocation,
    /// Path-shaped reader string retained for diagnostics and cuDF reader APIs.
    pub path: PathBuf,
    /// Resolved reader strategy and decode contract for this delete file.
    pub resolved_source: ResolvedParquetSource,
    /// Stable identifier for diagnostics and executor bookkeeping.
    pub stable_id: String,
    /// Known delete-file size in bytes, when available.
    pub file_size_bytes: Option<u64>,
    /// Known delete row count, when available.
    pub row_count: Option<u64>,
    /// Referenced data-file path from Iceberg position-delete metadata, when present.
    pub referenced_data_file: Option<String>,
    /// Inclusive lower bound for the Iceberg `pos` column, when manifest
    /// metrics prove it.
    pub position_lower_bound: Option<u64>,
    /// Inclusive upper bound for the Iceberg `pos` column, when manifest
    /// metrics prove it.
    pub position_upper_bound: Option<u64>,
}

impl fmt::Debug for ParquetPositionDeleteFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetPositionDeleteFile")
            .field("id", &self.id)
            .field("location", &self.location)
            .field("path", &redacted_debug_value("path", self.path.display().to_string().as_str()))
            .field("resolved_source", &self.resolved_source)
            .field("stable_id", &self.location.diagnostic_stable_id(self.stable_id.as_str()))
            .field("file_size_bytes", &self.file_size_bytes)
            .field("row_count", &self.row_count)
            .field(
                "referenced_data_file",
                &redacted_debug_option("referenced_data_file", self.referenced_data_file.as_deref()),
            )
            .field("position_lower_bound", &self.position_lower_bound)
            .field("position_upper_bound", &self.position_upper_bound)
            .finish()
    }
}

impl ParquetPositionDeleteFile {
    /// Creates a delete-file descriptor from an already resolved reader contract.
    pub fn from_resolved_parquet_source(
        id: impl Into<String>,
        resolved_source: ResolvedParquetSource,
    ) -> Result<Self, String> {
        let id = id.into();
        let location = parquet_source_location_for_original(&resolved_source.original_location)?;
        let stable_id = format!("position-delete:{id}:{}", location.diagnostic_location());
        Ok(Self {
            id,
            location,
            path: PathBuf::from(&resolved_source.reader_location),
            resolved_source,
            stable_id,
            file_size_bytes: None,
            row_count: None,
            referenced_data_file: None,
            position_lower_bound: None,
            position_upper_bound: None,
        })
    }

    /// Sets known delete-file size in bytes.
    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    /// Sets known delete row count.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    /// Sets the referenced data-file path for manifest-proven position-delete pruning.
    pub fn with_referenced_data_file(mut self, referenced_data_file: impl Into<String>) -> Self {
        self.referenced_data_file = Some(referenced_data_file.into());
        self
    }

    /// Sets inclusive Iceberg `pos` bounds from delete-file manifest metrics.
    pub fn with_position_bounds(mut self, lower_bound: Option<u64>, upper_bound: Option<u64>) -> Self {
        self.position_lower_bound = lower_bound;
        self.position_upper_bound = upper_bound;
        self
    }

    /// Builds a normal local Parquet file descriptor for reading the delete file.
    pub fn to_local_parquet_file(&self) -> LocalParquetFile {
        let mut file = LocalParquetFile::new(self.path.clone())
            .with_resolved_parquet_source(self.resolved_source.clone())
            .expect("validated position delete source remains representable")
            .with_stable_id(self.stable_id.clone());
        if let Some(file_size_bytes) = self.file_size_bytes {
            file = file.with_file_size_bytes(file_size_bytes);
        }
        if let Some(row_count) = self.row_count {
            file = file.with_row_group_rows(vec![row_count]);
        }
        file
    }
}

/// Iceberg equality-delete Parquet file associated with one data file.
///
/// The delete file is read as a normal native Parquet source and anti-joined
/// against the decoded data rows using Iceberg equality semantics: NULL keys
/// match NULL keys.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParquetEqualityDeleteFile {
    /// Stable delete-file id for diagnostics.
    pub id: String,
    /// Explicit source location for the delete Parquet file.
    pub location: ParquetSourceLocation,
    /// Path-shaped reader string retained for diagnostics and cuDF reader APIs.
    pub path: PathBuf,
    /// Resolved reader strategy and decode contract for this delete file.
    pub resolved_source: ResolvedParquetSource,
    /// Stable identifier for diagnostics and executor bookkeeping.
    pub stable_id: String,
    /// Iceberg field ids used as equality keys.
    pub equality_ids: Vec<i32>,
    /// Column names to read from the data file and delete file.
    pub equality_columns: Vec<String>,
    /// Known delete-file size in bytes, when available.
    pub file_size_bytes: Option<u64>,
    /// Known delete row count, when available.
    pub row_count: Option<u64>,
}

impl fmt::Debug for ParquetEqualityDeleteFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetEqualityDeleteFile")
            .field("id", &self.id)
            .field("location", &self.location)
            .field("path", &redacted_debug_value("path", self.path.display().to_string().as_str()))
            .field("resolved_source", &self.resolved_source)
            .field("stable_id", &self.location.diagnostic_stable_id(self.stable_id.as_str()))
            .field("equality_ids", &self.equality_ids)
            .field("equality_columns", &self.equality_columns)
            .field("file_size_bytes", &self.file_size_bytes)
            .field("row_count", &self.row_count)
            .finish()
    }
}

impl ParquetEqualityDeleteFile {
    /// Creates a delete-file descriptor from an already resolved reader contract.
    pub fn from_resolved_parquet_source(
        id: impl Into<String>,
        resolved_source: ResolvedParquetSource,
        equality_ids: Vec<i32>,
        equality_columns: Vec<String>,
    ) -> Result<Self, String> {
        if equality_ids.is_empty() || equality_ids.len() != equality_columns.len() {
            return Err("equality delete files require matching non-empty equality ids and columns".to_string());
        }
        let id = id.into();
        let location = parquet_source_location_for_original(&resolved_source.original_location)?;
        let stable_id = format!("equality-delete:{id}:{}", location.diagnostic_location());
        Ok(Self {
            id,
            location,
            path: PathBuf::from(&resolved_source.reader_location),
            resolved_source,
            stable_id,
            equality_ids,
            equality_columns,
            file_size_bytes: None,
            row_count: None,
        })
    }

    /// Sets known delete-file size in bytes.
    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    /// Sets known delete row count.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    /// Builds a normal local Parquet file descriptor for reading the delete file.
    pub fn to_local_parquet_file(&self) -> LocalParquetFile {
        let mut file = LocalParquetFile::new(self.path.clone())
            .with_resolved_parquet_source(self.resolved_source.clone())
            .expect("validated equality delete source remains representable")
            .with_stable_id(self.stable_id.clone());
        if let Some(file_size_bytes) = self.file_size_bytes {
            file = file.with_file_size_bytes(file_size_bytes);
        }
        if let Some(row_count) = self.row_count {
            file = file.with_row_group_rows(vec![row_count]);
        }
        file
    }
}

/// cuDF reader-side decimal storage type proven from Parquet metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParquetReaderDecimalTypeId {
    /// 32-bit fixed-point decimal.
    Decimal32,
    /// 64-bit fixed-point decimal.
    Decimal64,
    /// 128-bit fixed-point decimal.
    Decimal128,
}

impl ParquetReaderDecimalTypeId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decimal32 => "decimal32",
            Self::Decimal64 => "decimal64",
            Self::Decimal128 => "decimal128",
        }
    }
}

/// Per-file Parquet decimal reader type for one decoded read column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParquetReaderDecimalType {
    /// Column name as it appears in the source read projection.
    pub column_name: String,
    /// cuDF storage width used by the Parquet reader.
    pub type_id: ParquetReaderDecimalTypeId,
    /// cuDF fixed-point scale. Parquet/Arrow scale 2 is represented as -2.
    pub cudf_scale: i32,
}

impl ParquetReaderDecimalType {
    pub fn new(column_name: impl Into<String>, type_id: ParquetReaderDecimalTypeId, cudf_scale: i32) -> Self {
        Self { column_name: column_name.into(), type_id, cudf_scale }
    }
}

/// Reference to the source location for a logical column.
///
/// File columns are indexed against the Parquet file schema. Partition columns
/// are indexed against `LocalParquetSource::partition_columns`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceColumnRef {
    /// Physical column in the Parquet file schema.
    File(usize),
    /// Synthetic column from partition metadata.
    Partition(usize),
    /// Nullable logical column missing from the physical file schema.
    MissingNull(usize),
}

/// Frontend-neutral local Parquet scan contract.
///
/// Carries files, emitted schema, column roles, reader predicate, and IO
/// diagnostics for native source execution.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetSource {
    /// Local Parquet files to scan, including per-file metadata and pruning.
    pub files: Vec<LocalParquetFile>,
    /// Emitted source schema: read file columns followed by partition columns.
    pub schema: SchemaRef,
    /// Optional schema exposed to the query plan when execution transforms the
    /// physical read frame before yielding source output.
    pub output_schema_override: Option<SchemaRef>,
    /// Physical Parquet columns that the engine must read.
    ///
    /// These columns are emitted by the source. They include projected file
    /// columns and file columns needed only by source-adjacent filters.
    /// Partition columns are excluded because partition values come from file
    /// metadata and are emitted after `read_columns`.
    pub read_columns: Vec<String>,
    /// File-schema indices for `read_columns`.
    pub read_column_file_indices: Vec<usize>,
    /// Total physical column count in the source Parquet file schema, when known.
    pub file_column_count: Option<usize>,
    /// Source columns expected to survive source-adjacent filters/projections.
    pub projected_output_columns: Vec<String>,
    /// Source references for `projected_output_columns`.
    pub projected_output_column_source_refs: Vec<SourceColumnRef>,
    /// Columns emitted only so source-adjacent native filters can evaluate.
    ///
    /// These should be removed by a later native projection when they are not
    /// needed downstream.
    pub pure_filter_columns: Vec<String>,
    /// Source references for `pure_filter_columns`.
    pub pure_filter_column_source_refs: Vec<SourceColumnRef>,
    /// Synthetic source columns populated from file partition values.
    pub partition_columns: Vec<PartitionColumn>,
    /// Nullable logical source columns materialized as typed null columns.
    pub missing_columns: Vec<MissingColumn>,
    /// Source-level scan limit applied before source-adjacent native filters.
    ///
    /// For files with Iceberg position deletes this is a visible-row stop
    /// bound: the executor applies deletes for each read chunk before deciding
    /// whether enough rows have been emitted.
    pub scan_limit: Option<usize>,
    /// Reader-level Parquet predicate applied by the cuDF reader before source-local native filters.
    ///
    /// This predicate is expressed over `read_columns`; every referenced column
    /// must resolve to a physical file column that is present in the read
    /// projection. Partition and missing-null columns are not reader-visible.
    pub reader_predicate: Option<Expr>,
    /// Trusted source cardinality upper bound after source-level scan limit and
    /// reader predicate have been applied.
    pub max_rows: Option<usize>,
    /// Source capability and rejection details captured during lowering.
    pub diagnostics: SourceDiagnostics,
    /// Read one physical row-presence column, then emit row-count weights.
    pub count_star_probe: bool,
}

impl LocalParquetSource {
    /// Builds a source from fully described local Parquet files.
    ///
    /// `schema` must describe the source node's emitted schema. By default,
    /// every read file column is also marked as projected output metadata and no
    /// partition columns or pure filter columns are represented.
    pub fn from_files(files: Vec<LocalParquetFile>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        let read_column_file_indices = schema_column_indices(&schema, &read_columns);
        let file_column_count = Some(schema.fields().len());
        let projected_output_columns = read_columns.clone();
        let projected_output_column_source_refs =
            read_column_file_indices.iter().copied().map(SourceColumnRef::File).collect::<Vec<_>>();
        let pure_filter_columns = Vec::new();
        let pure_filter_column_source_refs = Vec::new();
        let partition_columns = Vec::new();
        let missing_columns = Vec::new();
        let diagnostics = SourceDiagnostics::for_local_parquet_files(
            &files,
            LocalParquetDiagnosticColumns {
                read_columns: &read_columns,
                read_column_file_indices: &read_column_file_indices,
                projected_output_columns: &projected_output_columns,
                projected_output_column_source_refs: &projected_output_column_source_refs,
                pure_filter_columns: &pure_filter_columns,
                pure_filter_column_source_refs: &pure_filter_column_source_refs,
                partition_columns: &partition_columns,
                missing_columns: &missing_columns,
                scan_limit: None,
            },
        );
        Self {
            files,
            schema,
            output_schema_override: None,
            read_columns,
            read_column_file_indices,
            file_column_count,
            projected_output_columns,
            projected_output_column_source_refs,
            pure_filter_columns,
            pure_filter_column_source_refs,
            partition_columns,
            missing_columns,
            scan_limit: None,
            reader_predicate: None,
            max_rows: None,
            diagnostics,
            count_star_probe: false,
        }
    }

    pub fn new(path: impl Into<PathBuf>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        Self::from_files(vec![LocalParquetFile::new(path)], schema, read_columns)
    }

    /// Builds the metadata-only representation for count-only Parquet scans.
    ///
    /// Zero-column count sources read no physical or partition columns. They use
    /// `ZERO_COLUMN_COUNT_COLUMN` as the synthetic output field and mark the
    /// diagnostics as represented only when the frontend has proven the shape is
    /// safe to answer from file or row-group metadata.
    pub fn zero_column_count(mut files: Vec<LocalParquetFile>) -> Self {
        for file in &mut files {
            file.partition_values.clear();
        }
        let schema = Arc::new(Schema::new(vec![Field::new(ZERO_COLUMN_COUNT_COLUMN, DataType::Int64, false)]));
        let mut source = Self::from_files(files, schema, Vec::new());
        source.mark_zero_column_count_represented();
        source
    }

    /// Builds a COUNT(*) fallback source that reads a physical row-presence
    /// column but emits one Int64 row-count weight per read unit.
    pub fn count_star_probe(files: Vec<LocalParquetFile>, read_schema: SchemaRef, read_columns: Vec<String>) -> Self {
        let output_schema = Arc::new(Schema::new(vec![Field::new(ZERO_COLUMN_COUNT_COLUMN, DataType::Int64, false)]));
        let mut source = Self::from_files(files, read_schema, read_columns);
        source.output_schema_override = Some(output_schema);
        source.count_star_probe = true;
        source
    }

    /// Sets projected and filter-only logical column roles.
    ///
    /// This does not change `schema`; it only records which emitted source
    /// columns are expected to survive source-adjacent filtering/projection and
    /// which emitted columns are only filter inputs.
    ///
    /// Source references are inferred from `read_columns` and
    /// `partition_columns`.
    pub fn with_column_roles(
        mut self,
        projected_output_columns: Vec<String>,
        pure_filter_columns: Vec<String>,
    ) -> Self {
        let projected_output_column_source_refs = self.source_column_refs(&projected_output_columns);
        let pure_filter_column_source_refs = self.source_column_refs(&pure_filter_columns);
        self.projected_output_columns = projected_output_columns;
        self.projected_output_column_source_refs = projected_output_column_source_refs;
        self.pure_filter_columns = pure_filter_columns;
        self.pure_filter_column_source_refs = pure_filter_column_source_refs;
        self.refresh_diagnostics();
        self
    }

    /// Sets explicit source references for read, projected, and filter-only columns.
    ///
    /// Use this when frontend scan metadata already resolved logical columns to
    /// file-schema positions or synthetic partition columns.
    pub fn with_column_source_refs(
        mut self,
        read_column_file_indices: Vec<usize>,
        projected_output_column_source_refs: Vec<SourceColumnRef>,
        pure_filter_column_source_refs: Vec<SourceColumnRef>,
    ) -> Self {
        self.read_column_file_indices = read_column_file_indices;
        self.projected_output_column_source_refs = projected_output_column_source_refs;
        self.pure_filter_column_source_refs = pure_filter_column_source_refs;
        self.refresh_diagnostics();
        self
    }

    /// Sets explicit file-schema indices for column roles with no partition columns.
    pub fn with_column_file_indices(
        self,
        read_column_file_indices: Vec<usize>,
        projected_output_column_file_indices: Vec<usize>,
        pure_filter_column_file_indices: Vec<usize>,
    ) -> Self {
        self.with_column_source_refs(
            read_column_file_indices,
            projected_output_column_file_indices.into_iter().map(SourceColumnRef::File).collect(),
            pure_filter_column_file_indices.into_iter().map(SourceColumnRef::File).collect(),
        )
    }

    /// Sets the schema exposed to downstream query-plan nodes.
    ///
    /// This is used when the physical Parquet read schema contains file-local
    /// names, but the source output has already been mapped back to logical
    /// table names by explicit output-column refs. Non-count-probe callers must
    /// still include every output override field in `schema`, because execution
    /// schema coercion and source-contract lowering operate on that execution
    /// source schema.
    pub fn with_output_schema_override(mut self, output_schema: SchemaRef) -> Self {
        self.output_schema_override = Some(output_schema);
        self
    }

    /// Sets the total physical column count in the source Parquet file schema.
    pub fn with_file_column_count(mut self, file_column_count: usize) -> Self {
        self.file_column_count = Some(file_column_count);
        self
    }

    /// Adds partition columns represented as synthetic source columns.
    pub fn with_partition_columns(mut self, partition_columns: Vec<PartitionColumn>) -> Self {
        self.partition_columns = partition_columns;
        self.refresh_diagnostics();
        self
    }

    /// Adds nullable logical columns represented as typed null source columns.
    pub fn with_missing_columns(mut self, missing_columns: Vec<MissingColumn>) -> Self {
        self.missing_columns = missing_columns;
        self.refresh_diagnostics();
        self
    }

    /// Sets an order-insensitive source-level scan limit.
    pub fn with_scan_limit(mut self, scan_limit: Option<usize>) -> Self {
        self.scan_limit = scan_limit;
        self.refresh_diagnostics();
        self
    }

    /// Sets a trusted source cardinality upper bound.
    pub fn with_max_rows(mut self, max_rows: Option<usize>) -> Self {
        self.max_rows = max_rows;
        self
    }

    /// Sets a reader-level Parquet predicate.
    ///
    /// Callers should only set predicates for expressions accepted by
    /// [`Self::reader_predicate_is_supported`]. Validation also checks this
    /// before execution.
    pub fn with_reader_predicate(mut self, predicate: Expr) -> Self {
        self.reader_predicate = Some(predicate);
        self.refresh_diagnostics();
        self
    }

    /// Returns true if `predicate` can be lowered to the cuDF Parquet reader AST
    /// using this source's read-column projection.
    pub fn reader_predicate_is_supported(&self, predicate: &Expr) -> bool {
        reader_predicate_root_is_supported(predicate)
            && reader_predicate_expr_is_supported(self, predicate)
            && (!reader_predicate_references_decimal_column(self, predicate)
                || reader_predicate_source_contract_shape_is_supported(self, predicate))
    }

    /// Returns a reader-level predicate expression accepted by the cuDF Parquet
    /// reader contract.
    ///
    /// Most accepted predicates are returned unchanged. Fixed ASCII prefix
    /// predicates of the form `SUBSTRING(column FROM 1 FOR n) IN (...)` are
    /// rewritten to equivalent lexicographic string ranges because the cuDF
    /// reader AST can express string comparisons but not string slicing.
    pub fn reader_predicate_supported_expression(&self, predicate: &Expr) -> Option<Expr> {
        if self.reader_predicate_is_supported(predicate) {
            return Some(predicate.clone());
        }
        let rewritten = rewrite_fixed_prefix_substring_reader_predicate(predicate)?;
        self.reader_predicate_is_supported(&rewritten).then_some(rewritten)
    }

    /// Returns true when the reader predicate needs per-source-chunk metadata
    /// to lower correctly.
    pub fn reader_predicate_requires_source_contract(&self, predicate: &Expr) -> bool {
        reader_predicate_references_decimal_column(self, predicate)
    }

    pub fn refresh_diagnostics(&mut self) {
        let zero_column_count_represented = self.is_zero_column_count_source();
        self.diagnostics = SourceDiagnostics::for_local_parquet_files(&self.files, self.diagnostic_columns());
        if self.reader_predicate.is_some() {
            self.diagnostics.predicate_disposition = PredicateDisposition::ReaderPushdown;
        }
        if zero_column_count_represented {
            self.mark_zero_column_count_represented();
        }
    }

    /// Returns the schema this source exposes to the native plan graph.
    pub fn output_schema(&self) -> &SchemaRef {
        self.output_schema_override.as_ref().unwrap_or(&self.schema)
    }

    /// Returns true when this source is the metadata-only zero-column count form.
    pub fn is_zero_column_count_source(&self) -> bool {
        self.diagnostics.zero_column_count_disposition == ZeroColumnCountDisposition::Represented
    }

    /// Returns true when this source reads a carrier column only to emit count weights.
    pub fn is_count_star_probe_source(&self) -> bool {
        self.count_star_probe
    }

    /// Returns true when any data file in this source carries Iceberg position deletes.
    pub fn has_position_delete_files(&self) -> bool {
        self.files.iter().any(LocalParquetFile::has_position_delete_files)
    }

    /// Returns true when any data file in this source carries Iceberg equality deletes.
    pub fn has_equality_delete_files(&self) -> bool {
        self.files.iter().any(LocalParquetFile::has_equality_delete_files)
    }

    /// Returns true when any data file in this source carries Iceberg row-level deletes.
    pub fn has_row_delete_files(&self) -> bool {
        self.files.iter().any(LocalParquetFile::has_row_delete_files)
    }

    /// Returns true when source scan-limit accounting must happen after
    /// per-chunk row deletes have been applied.
    pub fn has_position_delete_visible_scan_limit(&self) -> bool {
        self.scan_limit.is_some() && self.has_row_delete_files() && !self.is_count_star_probe_source()
    }

    fn mark_zero_column_count_represented(&mut self) {
        self.read_columns.clear();
        self.read_column_file_indices.clear();
        self.projected_output_columns.clear();
        self.projected_output_column_source_refs.clear();
        self.pure_filter_columns.clear();
        self.pure_filter_column_source_refs.clear();
        self.partition_columns.clear();
        self.missing_columns.clear();
        self.diagnostics = SourceDiagnostics::for_local_parquet_files(&self.files, self.diagnostic_columns());
        if self.reader_predicate.is_some() {
            self.diagnostics.predicate_disposition = PredicateDisposition::ReaderPushdown;
        }
        self.diagnostics.zero_column_count_disposition = ZeroColumnCountDisposition::Represented;
        self.diagnostics.rejected_features.retain(|feature| *feature != RejectedSourceFeature::ZeroColumnCount);
        self.diagnostics.direct_device_read =
            self.diagnostics.rejected_features.is_empty() && files_have_proven_direct_device_read(&self.files);
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn read_column_count(&self) -> usize {
        self.read_columns.len()
    }

    /// Number of source columns expected to remain after source-adjacent native
    /// filtering/projection.
    pub fn projected_output_column_count(&self) -> usize {
        self.projected_output_columns.len()
    }

    /// Number of emitted source columns used only by source filter evaluation.
    pub fn pure_filter_column_count(&self) -> usize {
        self.pure_filter_columns.len()
    }

    /// Total selected row groups across files, or zero when unknown.
    pub fn selected_row_group_count(&self) -> usize {
        self.files.iter().map(LocalParquetFile::selected_row_group_count).sum()
    }

    /// Number of file entries represented as byte-ranged in diagnostics.
    pub fn ranged_file_count(&self) -> usize {
        self.diagnostics.ranged_file_count
    }

    /// Whether diagnostics indicate that direct device read is supported.
    pub fn is_direct_device_read(&self) -> bool {
        self.diagnostics.direct_device_read
    }

    pub fn partition_column_count(&self) -> usize {
        self.partition_columns.len()
    }

    /// Returns true when every file represented by this source proves a
    /// physical file column has no nulls in its selected row groups.
    pub fn selected_file_column_has_no_nulls(&self, file_column_index: usize) -> bool {
        !self.files.is_empty()
            && self.files.iter().all(|file| file.selected_row_groups_have_no_nulls(file_column_index))
    }

    fn source_column_refs(&self, columns: &[String]) -> Vec<SourceColumnRef> {
        columns.iter().filter_map(|column| self.source_column_ref(column)).collect()
    }

    fn diagnostic_columns(&self) -> LocalParquetDiagnosticColumns<'_> {
        LocalParquetDiagnosticColumns {
            read_columns: &self.read_columns,
            read_column_file_indices: &self.read_column_file_indices,
            projected_output_columns: &self.projected_output_columns,
            projected_output_column_source_refs: &self.projected_output_column_source_refs,
            pure_filter_columns: &self.pure_filter_columns,
            pure_filter_column_source_refs: &self.pure_filter_column_source_refs,
            partition_columns: &self.partition_columns,
            missing_columns: &self.missing_columns,
            scan_limit: self.scan_limit,
        }
    }

    /// Resolves a logical source column to a physical file or partition column.
    pub fn source_column_ref(&self, column: &str) -> Option<SourceColumnRef> {
        if let Some(index) = self.projected_output_columns.iter().position(|output_column| output_column == column) {
            return self.projected_output_column_source_refs.get(index).copied();
        }
        if let Some(index) = self.pure_filter_columns.iter().position(|filter_column| filter_column == column) {
            return self.pure_filter_column_source_refs.get(index).copied();
        }
        self.read_columns
            .iter()
            .position(|read_column| read_column == column)
            .map(|index| SourceColumnRef::File(self.read_column_file_indices[index]))
            .or_else(|| {
                self.partition_columns
                    .iter()
                    .position(|partition_column| partition_column.name == column)
                    .map(SourceColumnRef::Partition)
                    .or_else(|| {
                        self.missing_columns
                            .iter()
                            .position(|missing_column| missing_column.name == column)
                            .map(SourceColumnRef::MissingNull)
                    })
            })
    }
}

fn object_store_reader_path(uri: &str) -> String {
    uri.to_string()
}

fn reader_predicate_root_is_supported(expr: &Expr) -> bool {
    match expr {
        Expr::Alias { expr, .. } => reader_predicate_root_is_supported(expr),
        Expr::Column(_) | Expr::Literal(_) => false,
        Expr::Not(_)
        | Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::Cast { .. }
        | Expr::TryCast { .. }
        | Expr::Binary { .. }
        | Expr::InList { .. }
        | Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => true,
    }
}

fn reader_predicate_expr_is_supported(source: &LocalParquetSource, expr: &Expr) -> bool {
    match expr {
        Expr::Column(column) => {
            let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column.name()) else {
                return false;
            };
            source.read_column_file_indices.contains(&file_index)
                && source
                    .schema
                    .field_with_name(column.name())
                    .ok()
                    .is_some_and(|field| reader_predicate_column_type_is_supported(field.data_type()))
        },
        Expr::Literal(value) => reader_predicate_literal_is_supported(value),
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) | Expr::Alias { expr: inner, .. } => {
            reader_predicate_expr_is_supported(source, inner)
        },
        Expr::Cast { expr, target_type } => {
            matches!(target_type, DataType::Int64 | DataType::UInt64 | DataType::Float64)
                && reader_predicate_expr_is_supported(source, expr)
        },
        Expr::Binary { left, op, right, .. } => {
            reader_predicate_binary_operator_is_supported(*op)
                && reader_predicate_expr_is_supported(source, left)
                && reader_predicate_expr_is_supported(source, right)
                && reader_predicate_binary_operand_types_are_supported(source, left, *op, right)
        },
        Expr::InList { expr, list, .. } => {
            !list.is_empty()
                && reader_predicate_expr_is_supported(source, expr)
                && reader_predicate_in_list_literal_types_are_supported(source, expr, list)
                && list
                    .iter()
                    .all(|value| matches!(value, Expr::Literal(_)) && reader_predicate_expr_is_supported(source, value))
        },
        // TRY_CAST is rejected from reader-predicate pushdown: null-on-failure
        // semantics interact differently with filter logic than strict CAST.
        Expr::TryCast { .. }
        | Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => false,
    }
}

fn rewrite_fixed_prefix_substring_reader_predicate(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Binary { left, op, right, output_type } => {
            let left = rewrite_fixed_prefix_substring_reader_predicate(left).unwrap_or_else(|| (**left).clone());
            let right = rewrite_fixed_prefix_substring_reader_predicate(right).unwrap_or_else(|| (**right).clone());
            let rewritten = Expr::Binary {
                left: Box::new(left),
                op: *op,
                right: Box::new(right),
                output_type: output_type.clone(),
            };
            (&rewritten != expr).then_some(rewritten)
        },
        Expr::Not(input) => rewrite_fixed_prefix_substring_reader_predicate(input).map(Expr::logical_not),
        Expr::Alias { expr: input, alias } => rewrite_fixed_prefix_substring_reader_predicate(input)
            .map(|expr| Expr::Alias { expr: Box::new(expr), alias: alias.clone() }),
        Expr::InList { expr: input, list, negated } => rewrite_fixed_prefix_substring_in_list(input, list, *negated),
        _ => None,
    }
}

fn rewrite_fixed_prefix_substring_in_list(expr: &Expr, list: &[Expr], negated: bool) -> Option<Expr> {
    let Expr::Substring { expr: input, start: 1, length: Some(length) } = expr else {
        return None;
    };
    let length = usize::try_from(*length).ok()?;
    if length == 0 || list.is_empty() {
        return None;
    }

    let mut ranges = Vec::with_capacity(list.len());
    for value in list {
        let prefix = fixed_ascii_prefix_literal(value, length)?;
        ranges.push(fixed_ascii_prefix_range((**input).clone(), prefix)?);
    }

    let mut ranges = ranges.into_iter();
    let first = ranges.next()?;
    let combined =
        ranges.fold(first, |left, right| Expr::binary_with_output_type(left, BinaryOp::Or, right, DataType::Boolean));
    Some(if negated { Expr::logical_not(combined) } else { combined })
}

fn fixed_ascii_prefix_literal(expr: &Expr, length: usize) -> Option<&str> {
    let Expr::Literal(LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value)) = expr else {
        return None;
    };
    (value.len() == length && value.is_ascii()).then_some(value.as_str())
}

fn fixed_ascii_prefix_range(column: Expr, prefix: &str) -> Option<Expr> {
    let upper = ascii_prefix_successor(prefix)?;
    let lower_bound = Expr::binary_with_output_type(
        column.clone(),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Utf8(prefix.to_string())),
        DataType::Boolean,
    );
    let upper_bound = Expr::binary_with_output_type(
        column,
        BinaryOp::Lt,
        Expr::literal(LiteralValue::Utf8(upper)),
        DataType::Boolean,
    );
    Some(Expr::binary_with_output_type(lower_bound, BinaryOp::And, upper_bound, DataType::Boolean))
}

fn ascii_prefix_successor(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
    let last = bytes.last_mut()?;
    if !last.is_ascii() || *last == 0x7f {
        return None;
    }
    *last += 1;
    String::from_utf8(bytes).ok()
}

fn reader_predicate_binary_operator_is_supported(op: crate::expr::BinaryOp) -> bool {
    use crate::expr::BinaryOp;
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo
            | BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor
            | BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::And
            | BinaryOp::Or
    )
}

fn reader_predicate_operator_outputs_boolean(op: crate::expr::BinaryOp) -> bool {
    use crate::expr::BinaryOp;
    matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::And
            | BinaryOp::Or
    )
}

fn reader_predicate_column_type_is_supported(data_type: &DataType) -> bool {
    is_supported_arrow_type(data_type)
        && !is_timestamp_with_timezone(data_type)
        && !is_supported_list_carry_type(data_type)
}

fn reader_predicate_literal_is_supported(value: &LiteralValue) -> bool {
    !matches!(value, LiteralValue::Null(_) | LiteralValue::IntervalMonthDayNano { .. })
}

fn reader_predicate_binary_operand_types_are_supported(
    source: &LocalParquetSource,
    left: &Expr,
    op: crate::expr::BinaryOp,
    right: &Expr,
) -> bool {
    use crate::expr::BinaryOp;
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return true;
    }
    reader_predicate_literal_matches_expr_type(source, left, right)
        && reader_predicate_literal_matches_expr_type(source, right, left)
}

fn reader_predicate_in_list_literal_types_are_supported(
    source: &LocalParquetSource,
    expr: &Expr,
    list: &[Expr],
) -> bool {
    list.iter().all(|value| reader_predicate_literal_matches_expr_type(source, value, expr))
}

fn reader_predicate_literal_matches_expr_type(source: &LocalParquetSource, literal: &Expr, typed_expr: &Expr) -> bool {
    let Expr::Literal(value) = literal else {
        return true;
    };
    let Some(data_type) = reader_predicate_expr_type(source, typed_expr) else {
        return true;
    };
    reader_predicate_literal_is_supported_for_type(value, &data_type)
}

fn reader_predicate_expr_type(source: &LocalParquetSource, expr: &Expr) -> Option<DataType> {
    match expr {
        Expr::Column(column) => {
            source.schema.field_with_name(column.name()).ok().map(|field| field.data_type().clone())
        },
        Expr::Literal(value) => Some(value.data_type()),
        Expr::Alias { expr, .. } => reader_predicate_expr_type(source, expr),
        Expr::Cast { target_type, .. }
        | Expr::TryCast { target_type, .. }
        | Expr::Round { output_type: target_type, .. } => Some(target_type.clone()),
        Expr::Not(_) | Expr::IsNull(_) | Expr::IsNotNull(_) | Expr::InList { .. } => Some(DataType::Boolean),
        Expr::StringContains { .. } | Expr::StartsWith { .. } | Expr::EndsWith { .. } => Some(DataType::Boolean),
        Expr::IsDistinctFrom { .. } | Expr::IsNotDistinctFrom { .. } => Some(DataType::Boolean),
        Expr::Binary { output_type: Some(output_type), .. } => Some(output_type.clone()),
        Expr::Binary { op, .. } if reader_predicate_operator_outputs_boolean(*op) => Some(DataType::Boolean),
        Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::Concat(_) => Some(DataType::Utf8),
        Expr::StringLength { output_type, .. } => Some(output_type.clone()),
        Expr::RegexpReplace { output_type, .. } => Some(output_type.clone()),
        Expr::DateTrunc { output_type, .. } => Some(output_type.clone()),
        Expr::Substring { .. }
        | Expr::Like { .. }
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::Binary { .. } => None,
    }
}

fn reader_predicate_literal_is_supported_for_type(value: &LiteralValue, data_type: &DataType) -> bool {
    if !reader_predicate_literal_is_supported(value) || cudf_dtype_from_arrow_type(data_type).is_none() {
        return false;
    }
    match data_type {
        DataType::Boolean => matches!(value, LiteralValue::Boolean(_)),
        DataType::Int8 => signed_literal_fits(value, i128::from(i8::MIN), i128::from(i8::MAX)),
        DataType::Int16 => signed_literal_fits(value, i128::from(i16::MIN), i128::from(i16::MAX)),
        DataType::Int32 => signed_literal_fits(value, i128::from(i32::MIN), i128::from(i32::MAX)),
        DataType::Int64 => signed_literal_fits(value, i128::from(i64::MIN), i128::from(i64::MAX)),
        DataType::UInt8 => unsigned_literal_fits(value, u128::from(u8::MAX)),
        DataType::UInt16 => unsigned_literal_fits(value, u128::from(u16::MAX)),
        DataType::UInt32 => unsigned_literal_fits(value, u128::from(u32::MAX)),
        DataType::UInt64 => unsigned_literal_fits(value, u128::from(u64::MAX)),
        DataType::Float32 => matches!(
            value,
            LiteralValue::Float32(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
        ),
        DataType::Float64 => matches!(
            value,
            LiteralValue::Float32(_)
                | LiteralValue::Float64(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
                | LiteralValue::UInt32(_)
        ),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            matches!(value, LiteralValue::Utf8(_) | LiteralValue::LargeUtf8(_))
        },
        DataType::Decimal128(_precision, scale) => decimal_literal_can_target_scale(value, *scale),
        DataType::Date32 => matches!(value, LiteralValue::Date32(_)),
        DataType::Date64 => matches!(value, LiteralValue::Date64(_)),
        DataType::Time32(unit) => {
            matches!(value, LiteralValue::Time32 { unit: literal_unit, .. } if literal_unit == unit)
        },
        DataType::Time64(unit) => {
            matches!(value, LiteralValue::Time64 { unit: literal_unit, .. } if literal_unit == unit)
        },
        DataType::Timestamp(unit, None) => {
            matches!(value, LiteralValue::Timestamp { unit: literal_unit, .. } if literal_unit == unit)
        },
        DataType::Duration(unit) => {
            matches!(value, LiteralValue::Duration { unit: literal_unit, .. } if literal_unit == unit)
        },
        _ => false,
    }
}

fn signed_literal_fits(value: &LiteralValue, min: i128, max: i128) -> bool {
    reader_predicate_integer_literal_raw(value).is_some_and(|raw| raw >= min && raw <= max)
}

fn unsigned_literal_fits(value: &LiteralValue, max: u128) -> bool {
    reader_predicate_unsigned_literal_raw(value).is_some_and(|raw| raw <= max)
}

fn decimal_literal_can_target_scale(value: &LiteralValue, target_arrow_scale: i8) -> bool {
    let target_cudf_scale = arrow_decimal_scale_to_cudf_scale(target_arrow_scale);
    if let Some(raw) = reader_predicate_integer_literal_raw(value) {
        return reader_predicate_decimal_raw_for_integer_value(raw, target_cudf_scale).is_ok();
    }
    if let LiteralValue::Decimal128 { raw, scale, .. } = value {
        return reader_predicate_rescale_decimal_raw(
            *raw,
            arrow_decimal_scale_to_cudf_scale(*scale),
            target_cudf_scale,
        )
        .is_ok();
    }
    false
}

fn reader_predicate_references_decimal_column(source: &LocalParquetSource, expr: &Expr) -> bool {
    reader_predicate_expr_references_decimal_column(expr, |column_name| {
        source
            .schema
            .field_with_name(column_name)
            .ok()
            .is_some_and(|field| matches!(field.data_type(), DataType::Decimal128(_, _)))
    })
}

fn reader_predicate_source_contract_shape_is_supported(source: &LocalParquetSource, expr: &Expr) -> bool {
    match expr {
        Expr::Binary { left, op: crate::expr::BinaryOp::And | crate::expr::BinaryOp::Or, right, .. } => {
            reader_predicate_source_contract_shape_is_supported(source, left)
                && reader_predicate_source_contract_shape_is_supported(source, right)
        },
        Expr::Binary { left, op, right, .. } => {
            reader_predicate_source_contract_comparison_is_supported(source, left, *op, right)
                || reader_predicate_source_contract_comparison_is_supported(source, right, invert_reader_op(*op), left)
        },
        Expr::InList { expr, list, .. } => {
            !list.is_empty()
                && reader_predicate_source_contract_column_type(source, expr).is_some()
                && list.iter().all(reader_predicate_source_contract_literal_is_supported)
        },
        Expr::Not(inner) | Expr::Alias { expr: inner, .. } => {
            reader_predicate_source_contract_shape_is_supported(source, inner)
        },
        Expr::Column(_)
        | Expr::Literal(_)
        | Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::Cast { .. }
        | Expr::TryCast { .. }
        | Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => false,
    }
}

fn reader_predicate_source_contract_comparison_is_supported(
    source: &LocalParquetSource,
    column: &Expr,
    op: crate::expr::BinaryOp,
    literal: &Expr,
) -> bool {
    matches!(
        op,
        crate::expr::BinaryOp::Eq
            | crate::expr::BinaryOp::NotEq
            | crate::expr::BinaryOp::Lt
            | crate::expr::BinaryOp::LtEq
            | crate::expr::BinaryOp::Gt
            | crate::expr::BinaryOp::GtEq
    ) && reader_predicate_source_contract_column_type(source, column).is_some()
        && reader_predicate_source_contract_literal_is_supported(literal)
}

fn reader_predicate_source_contract_column_type(source: &LocalParquetSource, expr: &Expr) -> Option<DataType> {
    let column = match expr {
        Expr::Column(column) => column,
        Expr::Alias { expr, .. } => return reader_predicate_source_contract_column_type(source, expr),
        _ => return None,
    };
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column.name()) else {
        return None;
    };
    if !source.read_column_file_indices.contains(&file_index) {
        return None;
    }
    let data_type = source.schema.field_with_name(column.name()).ok()?.data_type().clone();
    match data_type {
        DataType::Boolean
        | DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Decimal128(_, _)
        | DataType::Date32 => Some(data_type),
        _ => None,
    }
}

fn reader_predicate_source_contract_literal_is_supported(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal(
            LiteralValue::Boolean(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::Int64(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
                | LiteralValue::UInt32(_)
                | LiteralValue::UInt64(_)
                | LiteralValue::Float32(_)
                | LiteralValue::Float64(_)
                | LiteralValue::Utf8(_)
                | LiteralValue::LargeUtf8(_)
                | LiteralValue::Decimal128 { .. }
                | LiteralValue::Date32(_)
        )
    )
}

fn invert_reader_op(op: crate::expr::BinaryOp) -> crate::expr::BinaryOp {
    match op {
        crate::expr::BinaryOp::Lt => crate::expr::BinaryOp::Gt,
        crate::expr::BinaryOp::LtEq => crate::expr::BinaryOp::GtEq,
        crate::expr::BinaryOp::Gt => crate::expr::BinaryOp::Lt,
        crate::expr::BinaryOp::GtEq => crate::expr::BinaryOp::LtEq,
        _ => op,
    }
}

/// One local or object-store Parquet file plus optional row-group metadata.
///
/// Byte vectors are aligned with the selected row groups when `row_groups` is
/// present, otherwise with the frontend's represented row-group order. `None`
/// means the metadata is unavailable, not zero.
#[derive(Clone, PartialEq)]
pub struct LocalParquetFile {
    /// Explicit source location for the Parquet file.
    pub location: ParquetSourceLocation,
    /// Path-shaped reader string retained for existing diagnostics and cuDF
    /// reader APIs. Object-store URIs are represented as path-shaped strings
    /// but must be created through `object_store_uri`.
    pub path: PathBuf,
    /// Resolved reader strategy and decode contract for this file.
    pub resolved_source: ResolvedParquetSource,
    /// Stable identifier for diagnostics, pruning, and executor bookkeeping.
    ///
    /// Frontends should choose a value that remains stable across equivalent
    /// plans, commonly derived from path plus object metadata.
    pub stable_id: String,
    /// Known file size in bytes, when available from frontend metadata.
    pub file_size_bytes: Option<u64>,
    /// Selected Parquet row-group indices.
    ///
    /// `None` means no explicit row-group selection was represented. `Some`
    /// limits the scan to the listed row groups.
    pub row_groups: Option<Vec<usize>>,
    /// Estimated or known byte sizes for selected row groups.
    pub row_group_bytes: Option<Vec<u64>>,
    /// Estimated or known compressed byte sizes for projected output file columns.
    pub projected_row_group_compressed_bytes: Option<Vec<u64>>,
    /// Estimated or known uncompressed byte sizes for projected output file columns.
    ///
    /// Bounded source chunk planning uses this as its primary chunk-size input
    /// when available. Columns used only by source-adjacent filters are tracked
    /// separately so they can contribute to read/decode cost without forcing
    /// retained output sizing.
    pub projected_row_group_uncompressed_bytes: Option<Vec<u64>>,
    /// Estimated or known compressed byte sizes for pure-filter file columns.
    pub pure_filter_row_group_compressed_bytes: Option<Vec<u64>>,
    /// Estimated or known uncompressed byte sizes for pure-filter file columns.
    pub pure_filter_row_group_uncompressed_bytes: Option<Vec<u64>>,
    /// Estimated or known row counts for selected row groups.
    pub row_group_rows: Option<Vec<u64>>,
    /// Per-file-column null counts by row group, keyed by physical root column index.
    ///
    /// Inner vectors are indexed by physical row-group number, not selected
    /// row-group position. `None` means the footer did not prove a null count
    /// for that column chunk.
    pub row_group_null_counts: BTreeMap<usize, Vec<Option<u64>>>,
    /// Per-read-column decimal storage types proven from Parquet metadata.
    pub reader_decimal_types: Vec<ParquetReaderDecimalType>,
    /// Per-file field-id bindings proven from table-format and Parquet footer metadata.
    pub field_id_read_bindings: Vec<ParquetFieldIdReadBinding>,
    /// Position-delete files associated with this data file.
    pub position_delete_files: Vec<ParquetPositionDeleteFile>,
    /// Equality-delete files associated with this data file.
    pub equality_delete_files: Vec<ParquetEqualityDeleteFile>,
    /// Values for `LocalParquetSource::partition_columns` in source order.
    pub partition_values: Vec<LiteralValue>,
}

impl fmt::Debug for LocalParquetFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalParquetFile")
            .field("location", &self.location.diagnostic_location())
            .field("path", &redacted_debug_value("path", self.path.display().to_string().as_str()))
            .field("resolved_source", &self.resolved_source)
            .field("stable_id", &self.location.diagnostic_stable_id(self.stable_id.as_str()))
            .field("file_size_bytes", &self.file_size_bytes)
            .field("row_groups", &self.row_groups)
            .field("row_group_bytes", &self.row_group_bytes)
            .field("projected_row_group_compressed_bytes", &self.projected_row_group_compressed_bytes)
            .field("projected_row_group_uncompressed_bytes", &self.projected_row_group_uncompressed_bytes)
            .field("pure_filter_row_group_compressed_bytes", &self.pure_filter_row_group_compressed_bytes)
            .field("pure_filter_row_group_uncompressed_bytes", &self.pure_filter_row_group_uncompressed_bytes)
            .field("row_group_rows", &self.row_group_rows)
            .field("row_group_null_counts", &self.row_group_null_counts)
            .field("reader_decimal_types", &self.reader_decimal_types)
            .field("field_id_read_bindings", &self.field_id_read_bindings)
            .field("position_delete_files", &self.position_delete_files)
            .field("equality_delete_files", &self.equality_delete_files)
            .field("partition_values", &self.partition_values)
            .finish()
    }
}

impl LocalParquetFile {
    /// Creates a file descriptor with no optional metadata.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let stable_id = path.display().to_string();
        let reader_location = path.display().to_string();
        Self {
            location: ParquetSourceLocation::local_path(path.clone()),
            path,
            resolved_source: ResolvedParquetSource::local_kvikio(reader_location),
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            projected_row_group_compressed_bytes: None,
            projected_row_group_uncompressed_bytes: None,
            pure_filter_row_group_compressed_bytes: None,
            pure_filter_row_group_uncompressed_bytes: None,
            row_group_rows: None,
            row_group_null_counts: BTreeMap::new(),
            reader_decimal_types: Vec::new(),
            field_id_read_bindings: Vec::new(),
            position_delete_files: Vec::new(),
            equality_delete_files: Vec::new(),
            partition_values: Vec::new(),
        }
    }

    /// Creates an explicit object-store file descriptor.
    pub fn object_store_uri(uri: impl Into<String>) -> Result<Self, String> {
        let uri = uri.into();
        let location = ParquetSourceLocation::object_store_uri(uri.clone())?;
        let stable_id = location.diagnostic_location();
        let reader_path = object_store_reader_path(&uri);
        Ok(Self {
            location,
            path: PathBuf::from(&reader_path),
            resolved_source: ResolvedParquetSource::remote_host_buffer(
                uri,
                reader_path,
                CredentialMode::SharedProcessEnv,
            ),
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            projected_row_group_compressed_bytes: None,
            projected_row_group_uncompressed_bytes: None,
            pure_filter_row_group_compressed_bytes: None,
            pure_filter_row_group_uncompressed_bytes: None,
            row_group_rows: None,
            row_group_null_counts: BTreeMap::new(),
            reader_decimal_types: Vec::new(),
            field_id_read_bindings: Vec::new(),
            position_delete_files: Vec::new(),
            equality_delete_files: Vec::new(),
            partition_values: Vec::new(),
        })
    }

    /// Creates a file descriptor from an already resolved reader contract.
    pub fn from_resolved_parquet_source(resolved_source: ResolvedParquetSource) -> Result<Self, String> {
        let location = parquet_source_location_for_original(&resolved_source.original_location)?;
        let stable_id = location.diagnostic_location();
        Ok(Self {
            location,
            path: PathBuf::from(&resolved_source.reader_location),
            resolved_source,
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            projected_row_group_compressed_bytes: None,
            projected_row_group_uncompressed_bytes: None,
            pure_filter_row_group_compressed_bytes: None,
            pure_filter_row_group_uncompressed_bytes: None,
            row_group_rows: None,
            row_group_null_counts: BTreeMap::new(),
            reader_decimal_types: Vec::new(),
            field_id_read_bindings: Vec::new(),
            position_delete_files: Vec::new(),
            equality_delete_files: Vec::new(),
            partition_values: Vec::new(),
        })
    }

    /// Replaces the resolved reader contract and updates the reader path.
    pub fn with_resolved_parquet_source(mut self, resolved_source: ResolvedParquetSource) -> Result<Self, String> {
        self.location = parquet_source_location_for_original(&resolved_source.original_location)?;
        self.path = PathBuf::from(&resolved_source.reader_location);
        self.resolved_source = resolved_source;
        Ok(self)
    }

    /// Sets the stable file identifier used by native diagnostics/execution.
    pub fn with_stable_id(mut self, stable_id: impl Into<String>) -> Self {
        self.stable_id = stable_id.into();
        self
    }

    /// Sets known file size in bytes.
    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    /// Sets the selected row groups for this file.
    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.row_groups = Some(row_groups);
        self
    }

    /// Sets byte-size metadata for selected row groups.
    pub fn with_row_group_bytes(mut self, row_group_bytes: Vec<u64>) -> Self {
        self.row_group_bytes = Some(row_group_bytes);
        self
    }

    /// Sets projected output column byte-size metadata for row groups.
    pub fn with_projected_row_group_bytes(mut self, compressed_bytes: Vec<u64>, uncompressed_bytes: Vec<u64>) -> Self {
        self.projected_row_group_compressed_bytes = Some(compressed_bytes);
        self.projected_row_group_uncompressed_bytes = Some(uncompressed_bytes);
        self
    }

    /// Sets pure-filter column byte-size metadata for row groups.
    pub fn with_pure_filter_row_group_bytes(
        mut self,
        compressed_bytes: Vec<u64>,
        uncompressed_bytes: Vec<u64>,
    ) -> Self {
        self.pure_filter_row_group_compressed_bytes = Some(compressed_bytes);
        self.pure_filter_row_group_uncompressed_bytes = Some(uncompressed_bytes);
        self
    }

    /// Sets row-count metadata for selected row groups.
    pub fn with_row_group_rows(mut self, row_group_rows: Vec<u64>) -> Self {
        self.row_group_rows = Some(row_group_rows);
        self
    }

    /// Sets row-group null-count metadata for one physical file column.
    pub fn with_row_group_null_counts(mut self, file_column_index: usize, null_counts: Vec<Option<u64>>) -> Self {
        self.row_group_null_counts.insert(file_column_index, null_counts);
        self
    }

    /// Returns true when footer stats prove this file column has no nulls in
    /// every selected row group represented by this file descriptor.
    pub fn selected_row_groups_have_no_nulls(&self, file_column_index: usize) -> bool {
        let Some(null_counts) = self.row_group_null_counts.get(&file_column_index) else {
            return false;
        };
        let selected = self.row_groups.clone().unwrap_or_else(|| (0..null_counts.len()).collect::<Vec<_>>());
        selected.iter().all(|row_group| null_counts.get(*row_group).and_then(|count| *count) == Some(0))
    }

    /// Sets per-read-column decimal storage types proven from Parquet metadata.
    pub fn with_reader_decimal_types(mut self, reader_decimal_types: Vec<ParquetReaderDecimalType>) -> Self {
        self.reader_decimal_types = reader_decimal_types;
        self
    }

    /// Sets per-file field-id bindings proven from footer metadata.
    pub fn with_field_id_read_bindings(mut self, bindings: Vec<ParquetFieldIdReadBinding>) -> Self {
        self.field_id_read_bindings = bindings;
        self
    }

    /// Sets position-delete files associated with this data file.
    pub fn with_position_delete_files(mut self, delete_files: Vec<ParquetPositionDeleteFile>) -> Self {
        self.position_delete_files = delete_files;
        self
    }

    /// Sets equality-delete files associated with this data file.
    pub fn with_equality_delete_files(mut self, delete_files: Vec<ParquetEqualityDeleteFile>) -> Self {
        self.equality_delete_files = delete_files;
        self
    }

    /// Returns true when this data file requires Iceberg position deletes.
    pub fn has_position_delete_files(&self) -> bool {
        !self.position_delete_files.is_empty()
    }

    /// Returns true when this data file requires Iceberg equality deletes.
    pub fn has_equality_delete_files(&self) -> bool {
        !self.equality_delete_files.is_empty()
    }

    /// Returns true when this data file requires Iceberg row-level deletes.
    pub fn has_row_delete_files(&self) -> bool {
        self.has_position_delete_files() || self.has_equality_delete_files()
    }

    /// Sets values for represented partition columns.
    pub fn with_partition_values(mut self, partition_values: Vec<LiteralValue>) -> Self {
        self.partition_values = partition_values;
        self
    }

    /// Returns the represented selected row-group count, or zero if unknown.
    pub fn selected_row_group_count(&self) -> usize {
        self.row_groups.as_ref().map_or_else(
            || {
                self.row_group_bytes
                    .as_ref()
                    .or(self.projected_row_group_uncompressed_bytes.as_ref())
                    .or(self.row_group_rows.as_ref())
                    .map_or(0, Vec::len)
            },
            Vec::len,
        )
    }
}

fn parquet_source_location_for_original(location: &str) -> Result<ParquetSourceLocation, String> {
    if let Some(path) = location.strip_prefix("file://localhost/") {
        return Ok(ParquetSourceLocation::local_path(format!("/{path}")));
    }
    if let Some(path) = location.strip_prefix("file://") {
        return Ok(ParquetSourceLocation::local_path(path));
    }
    if let Some(path) = location.strip_prefix("file:") {
        return Ok(ParquetSourceLocation::local_path(path));
    }
    if location.contains("://") {
        return ParquetSourceLocation::object_store_uri(location);
    }
    Ok(ParquetSourceLocation::local_path(location))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_local_file_uri_uses_filesystem_path_for_location_and_reader_path() {
        let source = ResolvedParquetSource::local_kvikio("/tmp/table.parquet");
        let file = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource {
            original_location: "file:/tmp/table.parquet".to_string(),
            ..source
        })
        .unwrap();

        assert_eq!(file.location.as_local_path(), Some(Path::new("/tmp/table.parquet")));
        assert_eq!(file.path, PathBuf::from("/tmp/table.parquet"));
    }

    #[test]
    fn resolved_local_file_uri_with_localhost_authority_is_local_path() {
        let source = ResolvedParquetSource::local_kvikio("/tmp/table.parquet");
        let file = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource {
            original_location: "file://localhost/tmp/table.parquet".to_string(),
            ..source
        })
        .unwrap();

        assert_eq!(file.location.as_local_path(), Some(Path::new("/tmp/table.parquet")));
    }

    #[test]
    fn remote_direct_source_diagnostics_are_not_proven_direct_device_reads() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let file = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .unwrap();
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["id".to_string()]);

        assert_eq!(source.diagnostics.source_read_strategy, "remote_direct");
        assert_eq!(source.diagnostics.source_decode_input, "filepath");
        assert!(!source.diagnostics.direct_device_read);
        assert!(source.diagnostics.rejected_features.is_empty());
    }

    #[test]
    fn object_store_default_stable_ids_are_redacted_hashes() {
        let uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let file = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            uri,
            CredentialMode::SharedProcessEnv,
        ))
        .unwrap();

        assert!(file.stable_id.starts_with("object_store:s3:fnv64:"));
        assert!(!file.stable_id.contains("secret-bucket"));
        assert!(!file.stable_id.contains("X-Amz-Signature"));
        assert_eq!(redacted_stable_id_diagnostic(uri), file.stable_id);
    }

    #[test]
    fn source_debug_redacts_object_store_locations_and_stable_ids() {
        let uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let source = ResolvedParquetSource::remote_direct(uri, CredentialMode::SharedProcessEnv)
            .with_lakehouse_cache_config("/tmp/cache/secret-bucket/private/key.parquet?token=abc", None)
            .with_lakehouse_cache_path("/tmp/cache/secret-bucket/private/key.parquet?token=abc");
        let file = LocalParquetFile::from_resolved_parquet_source(source)
            .unwrap()
            .with_stable_id("s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc");

        let rendered = format!("{file:?}");
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "token=abc"] {
            assert!(!rendered.contains(sensitive), "debug output leaked {sensitive}: {rendered}");
        }

        let location = ParquetSourceLocation::object_store_uri(uri).unwrap();
        let rendered_location = format!("{location:?}");
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(
                !rendered_location.contains(sensitive),
                "location debug output leaked {sensitive}: {rendered_location}"
            );
        }
    }

    #[test]
    fn position_delete_debug_redacts_referenced_data_file() {
        let referenced = "s3://secret-bucket/private/data.parquet?X-Amz-Signature=abc";
        let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
            "delete-1",
            ResolvedParquetSource::remote_direct(
                "s3://delete-bucket/private/delete.parquet?X-Amz-Signature=def",
                CredentialMode::SharedProcessEnv,
            ),
        )
        .unwrap()
        .with_referenced_data_file(referenced);
        let file = LocalParquetFile::new("/tmp/data.parquet").with_position_delete_files(vec![delete_file]);

        let rendered = format!("{file:?}");
        for sensitive in ["secret-bucket", "private/data", "X-Amz-Signature", "delete-bucket"] {
            assert!(!rendered.contains(sensitive), "debug output leaked {sensitive}: {rendered}");
        }
        assert!(rendered.contains("referenced_data_file"));
        assert!(rendered.contains("referenced_data_file:fnv64:"));
    }

    #[test]
    fn object_store_path_diagnostics_are_redacted_hashes() {
        let path = "secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let diagnostic = redacted_object_store_path_diagnostic(path);

        assert!(diagnostic.starts_with("object_store_path:fnv64:"));
        for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
            assert!(!diagnostic.contains(sensitive), "diagnostic leaked {sensitive}: {diagnostic}");
        }
    }
}
