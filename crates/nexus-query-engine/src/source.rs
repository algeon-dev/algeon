//! Native query source contracts and source diagnostics.
//!
//! This module contains frontend-neutral source metadata. File locations,
//! schema mapping, partition values, missing columns, row-group metadata, and
//! reader predicate support are represented explicitly so native execution does
//! not depend on DataFusion scan internals.

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::new_empty_array,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};

use crate::{
    datatypes::{cudf_dtype_from_arrow_type, is_supported_arrow_type},
    expr::{Expr, LiteralValue},
};

mod diagnostics;

use diagnostics::LocalParquetDiagnosticColumns;
pub use diagnostics::{
    ByteRangeDisposition, PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature,
    ScanLimitDisposition, SchemaMappingDisposition, SelectedRowGroupStats, SourceDiagnostics, SourceFormat,
    ZeroColumnCountDisposition,
};

/// Synthetic output column used for Parquet plans that only need a row count.
///
/// A zero-column count source reads no physical columns and produces row counts
/// from file or row-group metadata instead of projecting normal source columns.
pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__cudf_zero_column_row_count";

/// Explicit location/read-capability model for native Parquet sources.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// Experimental path-based cuDF/libcudf read for a remote URI such as `s3://...`.
    RemoteKvikioExperimental,
}

impl ParquetReadStrategy {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalKvikio => "local_kvikio",
            Self::RemoteHostBuffer => "remote_host_buffer",
            Self::LocalCacheKvikio => "local_cache_kvikio",
            Self::RemoteKvikioExperimental => "remote_kvikio_experimental",
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

/// Credential scope used by a resolved source read.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialMode {
    /// No external credentials are needed.
    #[default]
    None,
    /// Reader uses credentials already configured for the running process.
    SharedProcessEnv,
    /// Placeholder for future per-tenant credential plumbing.
    FutureTenantScoped,
}

impl CredentialMode {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SharedProcessEnv => "shared_process_env",
            Self::FutureTenantScoped => "future_tenant_scoped",
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
            Self::Skipped => "skipped",
        }
    }
}

impl fmt::Display for LakehouseCacheStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Resolved Parquet reader contract carried into native source execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// Credential scope used by the selected strategy.
    pub credential_mode: CredentialMode,
    /// Local byte-cache status for this source.
    pub cache_status: LakehouseCacheStatus,
    /// Lakehouse byte-cache root used by cache maintenance, when configured.
    pub lakehouse_cache_root: Option<String>,
    /// Best-effort retained byte target for the lakehouse byte-cache root.
    pub lakehouse_cache_max_bytes: Option<u64>,
    /// Stable fallback reason when the preferred strategy was unavailable.
    pub fallback_reason: Option<String>,
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
            credential_mode: CredentialMode::None,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_max_bytes: None,
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
            credential_mode,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_max_bytes: None,
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
            credential_mode,
            cache_status,
            lakehouse_cache_root: None,
            lakehouse_cache_max_bytes: None,
            fallback_reason: None,
        }
    }

    /// Builds an experimental remote path-based read contract.
    pub fn remote_kvikio_experimental(original_location: impl Into<String>, credential_mode: CredentialMode) -> Self {
        let original_location = original_location.into();
        Self {
            reader_location: original_location.clone(),
            original_location,
            storage_domain: None,
            content_identity: None,
            storage_profile_id: None,
            read_strategy: ParquetReadStrategy::RemoteKvikioExperimental,
            decode_input: SourceDecodeInput::Filepath,
            credential_mode,
            cache_status: LakehouseCacheStatus::Disabled,
            lakehouse_cache_root: None,
            lakehouse_cache_max_bytes: None,
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
            Self::LocalParquet(source) => &source.schema,
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

/// Local Parquet source accepted by the native query engine.
///
/// This struct is the frontend-neutral scan contract. Its `schema` is the
/// schema emitted by the native source node: all read file columns followed by
/// represented partition columns. Column-role fields separately describe which
/// source columns are expected to remain after source-adjacent native filter or
/// projection nodes, and which were emitted only so those filters can evaluate.
/// Diagnostics explain whether the source can use direct device read.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetSource {
    /// Local Parquet files to scan, including per-file metadata and pruning.
    pub files: Vec<LocalParquetFile>,
    /// Native output schema emitted by the source node.
    ///
    /// For local Parquet this is every read file column, including columns used
    /// only by source-adjacent filters, followed by represented partition
    /// columns. `projected_output_columns` is role/diagnostic metadata and does
    /// not by itself define this schema.
    pub schema: SchemaRef,
    /// Physical Parquet columns that the engine must read.
    ///
    /// These columns are emitted by the source. They include projected file
    /// columns and file columns needed only by source-adjacent filters.
    /// Partition columns are excluded because partition values come from file
    /// metadata and are emitted after `read_columns`.
    pub read_columns: Vec<String>,
    /// File-schema indices for `read_columns`.
    pub read_column_file_indices: Vec<usize>,
    /// Source columns expected to remain after source-adjacent filtering/projection.
    ///
    /// These are role and diagnostic metadata, not the full emitted source
    /// schema. They are distinct from `read_columns`: a column used only by a
    /// native filter is still emitted by the source so the filter can consume
    /// it, but it should be removed by a later native projection if it is not
    /// needed downstream.
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
}

impl LocalParquetSource {
    /// Builds a source from fully described local Parquet files.
    ///
    /// `schema` must describe the source node's emitted schema. By default,
    /// every read file column is also marked as projected output metadata and no
    /// partition columns or pure filter columns are represented.
    pub fn from_files(files: Vec<LocalParquetFile>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        let read_column_file_indices = schema_column_indices(&schema, &read_columns);
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
            read_columns,
            read_column_file_indices,
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
        reader_predicate_expr_is_supported(self, predicate)
            && (!reader_predicate_references_decimal_column(self, predicate)
                || reader_predicate_source_contract_shape_is_supported(self, predicate))
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

    /// Returns true when this source is the metadata-only zero-column count form.
    pub fn is_zero_column_count_source(&self) -> bool {
        self.diagnostics.zero_column_count_disposition == ZeroColumnCountDisposition::Represented
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
        self.diagnostics.direct_device_read = self.diagnostics.rejected_features.is_empty();
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
        Expr::IsDistinctFrom { .. } | Expr::IsNotDistinctFrom { .. } => Some(DataType::Boolean),
        Expr::Binary { output_type: Some(output_type), .. } => Some(output_type.clone()),
        Expr::Binary { op, .. } if reader_predicate_operator_outputs_boolean(*op) => Some(DataType::Boolean),
        Expr::Lower(_) | Expr::Upper(_) | Expr::Concat(_) => Some(DataType::Utf8),
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
    integer_literal_raw(value).is_some_and(|raw| raw >= min && raw <= max)
}

fn unsigned_literal_fits(value: &LiteralValue, max: u128) -> bool {
    unsigned_literal_raw(value).is_some_and(|raw| raw <= max)
}

fn unsigned_literal_raw(value: &LiteralValue) -> Option<u128> {
    match value {
        LiteralValue::UInt8(value) => Some(u128::from(*value)),
        LiteralValue::UInt16(value) => Some(u128::from(*value)),
        LiteralValue::UInt32(value) => Some(u128::from(*value)),
        LiteralValue::UInt64(value) => Some(u128::from(*value)),
        LiteralValue::Int8(value) => u128::try_from(*value).ok(),
        LiteralValue::Int16(value) => u128::try_from(*value).ok(),
        LiteralValue::Int32(value) => u128::try_from(*value).ok(),
        LiteralValue::Int64(value) => u128::try_from(*value).ok(),
        _ => None,
    }
}

fn integer_literal_raw(value: &LiteralValue) -> Option<i128> {
    match value {
        LiteralValue::Int8(value) => Some(i128::from(*value)),
        LiteralValue::Int16(value) => Some(i128::from(*value)),
        LiteralValue::Int32(value) => Some(i128::from(*value)),
        LiteralValue::Int64(value) => Some(i128::from(*value)),
        LiteralValue::UInt8(value) => Some(i128::from(*value)),
        LiteralValue::UInt16(value) => Some(i128::from(*value)),
        LiteralValue::UInt32(value) => Some(i128::from(*value)),
        LiteralValue::UInt64(value) => Some(i128::from(*value)),
        _ => None,
    }
}

fn decimal_literal_can_target_scale(value: &LiteralValue, target_arrow_scale: i8) -> bool {
    let target_cudf_scale = -i32::from(target_arrow_scale);
    if let Some(raw) = integer_literal_raw(value) {
        return decimal_raw_for_integer_value(raw, target_cudf_scale).is_some();
    }
    if let LiteralValue::Decimal128 { raw, scale, .. } = value {
        return rescale_decimal_raw_for_reader_support(*raw, -i32::from(*scale), target_cudf_scale).is_some();
    }
    false
}

fn decimal_raw_for_integer_value(value: i128, target_scale: i32) -> Option<i128> {
    if target_scale <= 0 {
        checked_pow10(target_scale.unsigned_abs()).and_then(|multiplier| value.checked_mul(multiplier))
    } else {
        let divisor = checked_pow10(target_scale as u32)?;
        (value % divisor == 0).then_some(value / divisor)
    }
}

fn rescale_decimal_raw_for_reader_support(raw: i128, source_scale: i32, target_scale: i32) -> Option<i128> {
    let scale_delta = source_scale.checked_sub(target_scale)?;
    if scale_delta >= 0 {
        let multiplier = checked_pow10(scale_delta as u32)?;
        raw.checked_mul(multiplier)
    } else {
        let divisor = checked_pow10(scale_delta.unsigned_abs())?;
        (raw % divisor == 0).then_some(raw / divisor)
    }
}

fn checked_pow10(exponent: u32) -> Option<i128> {
    let mut value = 1i128;
    for _ in 0..exponent {
        value = value.checked_mul(10)?;
    }
    Some(value)
}

fn reader_predicate_references_decimal_column(source: &LocalParquetSource, expr: &Expr) -> bool {
    match expr {
        Expr::Column(column) => source
            .schema
            .field_with_name(column.name())
            .ok()
            .is_some_and(|field| matches!(field.data_type(), DataType::Decimal128(_, _))),
        Expr::Literal(_) => false,
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Alias { expr: inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. } => reader_predicate_references_decimal_column(source, inner),
        Expr::Binary { left, right, .. } => {
            reader_predicate_references_decimal_column(source, left)
                || reader_predicate_references_decimal_column(source, right)
        },
        Expr::InList { expr, list, .. } => {
            reader_predicate_references_decimal_column(source, expr)
                || list.iter().any(|value| reader_predicate_references_decimal_column(source, value))
        },
        Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. } => reader_predicate_references_decimal_column(source, expr),
        Expr::Concat(exprs) => exprs.iter().any(|expr| reader_predicate_references_decimal_column(source, expr)),
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            reader_predicate_references_decimal_column(source, left)
                || reader_predicate_references_decimal_column(source, right)
        },
        Expr::Case { branches, else_expr } => {
            branches.iter().any(|branch| {
                reader_predicate_references_decimal_column(source, &branch.when)
                    || reader_predicate_references_decimal_column(source, &branch.then)
            }) || reader_predicate_references_decimal_column(source, else_expr)
        },
    }
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
#[derive(Debug, Clone, PartialEq)]
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
    /// Values for `LocalParquetSource::partition_columns` in source order.
    pub partition_values: Vec<LiteralValue>,
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
            partition_values: Vec::new(),
        }
    }

    /// Creates an explicit object-store file descriptor.
    pub fn object_store_uri(uri: impl Into<String>) -> Result<Self, String> {
        let uri = uri.into();
        let location = ParquetSourceLocation::object_store_uri(uri.clone())?;
        let stable_id = uri.clone();
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
            partition_values: Vec::new(),
        })
    }

    /// Creates a file descriptor from an already resolved reader contract.
    pub fn from_resolved_parquet_source(resolved_source: ResolvedParquetSource) -> Result<Self, String> {
        let location = parquet_source_location_for_original(&resolved_source.original_location)?;
        let stable_id = resolved_source.original_location.clone();
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
    if let Some(path) = location.strip_prefix("file://") {
        return Ok(ParquetSourceLocation::local_path(path));
    }
    if location.contains("://") {
        return ParquetSourceLocation::object_store_uri(location);
    }
    Ok(ParquetSourceLocation::local_path(location))
}
