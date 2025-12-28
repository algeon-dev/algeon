use std::{path::PathBuf, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::expr::LiteralValue;

/// Synthetic output column used for Parquet plans that only need a row count.
///
/// A zero-column count source reads no physical columns and produces row counts
/// from file or row-group metadata instead of projecting normal source columns.
pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__cudf_zero_column_row_count";

/// Native query-engine source description.
///
/// Frontends lower scan nodes into this enum after they have resolved file
/// locations, emitted source schema, schema mapping, projected column roles,
/// filter-only column roles, and source diagnostics. The executor should not
/// need frontend-specific scan metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceNode {
    /// Local filesystem Parquet input.
    LocalParquet(LocalParquetSource),
    /// Already-materialized GPU input identified by a frontend-provided name.
    GpuResident(GpuResidentSource),
}

impl SourceNode {
    /// Returns the native output schema exposed by this source node.
    pub fn output_schema(&self) -> &SchemaRef {
        match self {
            Self::LocalParquet(source) => &source.schema,
            Self::GpuResident(source) => &source.schema,
        }
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
    /// Creates a partition column descriptor.
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
            diagnostics,
        }
    }

    /// Builds a single-file local Parquet source.
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

    /// Recomputes diagnostics from the current source metadata.
    pub fn refresh_diagnostics(&mut self) {
        let zero_column_count_represented = self.is_zero_column_count_source();
        self.diagnostics = SourceDiagnostics::for_local_parquet_files(&self.files, self.diagnostic_columns());
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
        self.diagnostics = SourceDiagnostics::for_local_parquet_files(&self.files, self.diagnostic_columns());
        self.diagnostics.zero_column_count_disposition = ZeroColumnCountDisposition::Represented;
        self.diagnostics.rejected_features.retain(|feature| *feature != RejectedSourceFeature::ZeroColumnCount);
        self.diagnostics.direct_device_read = self.diagnostics.rejected_features.is_empty();
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Number of physical Parquet columns that will be read.
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

    pub fn selected_row_group_count(&self) -> usize {
        self.files.iter().map(LocalParquetFile::selected_row_group_count).sum()
    }

    pub fn ranged_file_count(&self) -> usize {
        self.diagnostics.ranged_file_count
    }

    /// Whether diagnostics indicate that direct device read is supported.
    pub fn is_direct_device_read(&self) -> bool {
        self.diagnostics.direct_device_read
    }

    /// Number of synthetic partition columns represented by this source.
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
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetFile {
    /// Local filesystem path to the Parquet file.
    pub path: PathBuf,
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
        Self {
            path,
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            row_group_rows: None,
            partition_values: Vec::new(),
        }
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
            || self.row_group_bytes.as_ref().or(self.row_group_rows.as_ref()).map_or(0, Vec::len),
            Vec::len,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct LocalParquetDiagnosticColumns<'a> {
    read_columns: &'a [String],
    read_column_file_indices: &'a [usize],
    projected_output_columns: &'a [String],
    projected_output_column_source_refs: &'a [SourceColumnRef],
    pure_filter_columns: &'a [String],
    pure_filter_column_source_refs: &'a [SourceColumnRef],
    partition_columns: &'a [PartitionColumn],
}

/// Frontend-visible source capability and lowering diagnostics.
///
/// Diagnostics explain why a source can or cannot use the native direct device
/// path. They are intentionally structural so adapters and tests can assert on
/// stable fields instead of parsing displayed plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDiagnostics {
    /// Source file/data format.
    pub format: SourceFormat,
    /// True when no rejected features prevent native direct device read.
    pub direct_device_read: bool,
    /// Number of files represented after any frontend coalescing.
    pub file_count: usize,
    /// Number of source file entries that carried byte ranges.
    pub ranged_file_count: usize,
    /// Total known bytes across represented files, or `None` when incomplete.
    pub known_bytes: Option<u64>,
    /// Total selected row groups, or `None` when row-group selection is unknown.
    pub selected_row_groups: Option<usize>,
    /// How pushed-down source predicates were represented.
    pub predicate_disposition: PredicateDisposition,
    /// How byte-range file entries were represented.
    pub byte_range_disposition: ByteRangeDisposition,
    /// How partition columns were represented.
    pub partition_column_disposition: PartitionColumnDisposition,
    /// Whether metadata-only zero-column count was required and represented.
    pub zero_column_count_disposition: ZeroColumnCountDisposition,
    /// How logical source columns map to physical or synthetic source columns.
    pub schema_mapping_disposition: SchemaMappingDisposition,
    /// Number of physical Parquet columns read by the native source.
    pub read_column_count: usize,
    /// Number of logical source columns expected to remain after
    /// source-adjacent native filtering/projection.
    pub projected_output_column_count: usize,
    /// Number of emitted source columns used only for source filtering.
    pub pure_filter_column_count: usize,
    /// Stable feature codes explaining why direct device read was rejected.
    pub rejected_features: Vec<RejectedSourceFeature>,
}

impl SourceDiagnostics {
    fn for_local_parquet_files(files: &[LocalParquetFile], columns: LocalParquetDiagnosticColumns<'_>) -> Self {
        let known_bytes = files
            .iter()
            .map(|file| file.file_size_bytes)
            .try_fold(0u64, |total, bytes| bytes.map(|bytes| total.saturating_add(bytes)));
        let selected_row_groups = files
            .iter()
            .any(|file| file.row_groups.is_some())
            .then(|| files.iter().map(LocalParquetFile::selected_row_group_count).sum());
        let zero_column_count_disposition = if columns.read_columns.is_empty() {
            ZeroColumnCountDisposition::Rejected
        } else {
            ZeroColumnCountDisposition::NotRequired
        };
        let mut rejected_features = Vec::new();
        if zero_column_count_disposition == ZeroColumnCountDisposition::Rejected {
            rejected_features.push(RejectedSourceFeature::ZeroColumnCount);
        }
        let schema_mapping_disposition = if columns.read_columns.len() == columns.read_column_file_indices.len()
            && columns.projected_output_columns.len() == columns.projected_output_column_source_refs.len()
            && columns.pure_filter_columns.len() == columns.pure_filter_column_source_refs.len()
        {
            SchemaMappingDisposition::DirectByName
        } else {
            rejected_features.push(RejectedSourceFeature::SchemaMapping);
            SchemaMappingDisposition::Rejected
        };
        let partition_column_disposition = if columns.partition_columns.is_empty() {
            PartitionColumnDisposition::Absent
        } else {
            PartitionColumnDisposition::Represented
        };
        Self {
            format: SourceFormat::Parquet,
            direct_device_read: rejected_features.is_empty(),
            file_count: files.len(),
            ranged_file_count: 0,
            known_bytes,
            selected_row_groups,
            predicate_disposition: PredicateDisposition::Absent,
            byte_range_disposition: ByteRangeDisposition::FullFiles,
            partition_column_disposition,
            zero_column_count_disposition,
            schema_mapping_disposition,
            read_column_count: columns.read_columns.len(),
            projected_output_column_count: columns.projected_output_columns.len(),
            pure_filter_column_count: columns.pure_filter_columns.len(),
            rejected_features,
        }
    }
}

/// Source format represented by native source metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    /// Apache Parquet files.
    Parquet,
}

/// Disposition of a source-level predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredicateDisposition {
    /// No predicate was attached to the source.
    Absent,
    /// Predicate is represented only for file or row-group pruning.
    PruningOnly,
    /// Predicate exists but cannot be represented by the native source.
    Rejected,
}

impl PredicateDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::PruningOnly => "pruning_only",
            Self::Rejected => "rejected",
        }
    }
}

/// Disposition of byte-range file entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteRangeDisposition {
    /// Source entries describe whole files.
    FullFiles,
    /// Byte ranges were safely coalesced back to whole-file reads.
    CoalescedFullFiles,
    /// Byte ranges remain and are not supported by the native source.
    Rejected,
}

impl ByteRangeDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullFiles => "full_files",
            Self::CoalescedFullFiles => "coalesced_full_files",
            Self::Rejected => "rejected",
        }
    }
}

/// Disposition of partition columns used by the native source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionColumnDisposition {
    /// No partition columns are required.
    Absent,
    /// Required partition columns or values could not be represented.
    Rejected,
    /// Required partition columns are represented as synthetic source columns.
    Represented,
}

impl PartitionColumnDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Rejected => "rejected",
            Self::Represented => "represented",
        }
    }
}

/// Disposition of metadata-only row-count sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeroColumnCountDisposition {
    /// The plan reads at least one source column, so zero-column count is unused.
    NotRequired,
    /// The plan required zero-column count, but it was not proven safe.
    Rejected,
    /// The source represents row counts without reading physical columns.
    Represented,
}

impl ZeroColumnCountDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::Rejected => "rejected",
            Self::Represented => "represented",
        }
    }
}

/// Disposition of logical-to-source column mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaMappingDisposition {
    /// Logical columns map directly by name to file or partition columns.
    DirectByName,
    /// Required column mapping could not be represented.
    Rejected,
}

impl SchemaMappingDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectByName => "direct_by_name",
            Self::Rejected => "rejected",
        }
    }
}

/// Stable rejection feature codes for native source diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectedSourceFeature {
    /// Source is not backed by a local filesystem object store.
    NonLocalObjectStore,
    /// Required partition columns could not be represented.
    PartitionColumns,
    /// DataFusion scan expression adapter is present.
    ExprAdapter,
    /// DataFusion scan limit is present.
    ScanLimit,
    /// Source requires preserved input order.
    PreserveOrder,
    /// Source declares output ordering.
    OutputOrdering,
    /// Pushed-down Parquet predicate cannot be represented safely.
    PushedDownParquetPredicate,
    /// Source has no files.
    EmptySource,
    /// Required partition values are missing or unsupported.
    PartitionValues,
    /// Byte-range file entries are not supported as represented.
    ByteRange,
    /// Per-file ordering metadata is present.
    FileOrdering,
    /// Row-level selection is present.
    RowSelection,
    /// Zero-column count was required but not represented.
    ZeroColumnCount,
    /// Logical-to-source column mapping is invalid.
    SchemaMapping,
}

impl RejectedSourceFeature {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonLocalObjectStore => "non_local_object_store",
            Self::PartitionColumns => "partition_columns",
            Self::ExprAdapter => "expr_adapter",
            Self::ScanLimit => "scan_limit",
            Self::PreserveOrder => "preserve_order",
            Self::OutputOrdering => "output_ordering",
            Self::PushedDownParquetPredicate => "pushed_down_parquet_predicate",
            Self::EmptySource => "empty_source",
            Self::PartitionValues => "partition_values",
            Self::ByteRange => "byte_range",
            Self::FileOrdering => "file_ordering",
            Self::RowSelection => "row_selection",
            Self::ZeroColumnCount => "zero_column_count",
            Self::SchemaMapping => "schema_mapping",
        }
    }
}
