use std::{path::PathBuf, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::expr::{Expr, LiteralValue};

mod diagnostics;

use diagnostics::LocalParquetDiagnosticColumns;
pub use diagnostics::{
    ByteRangeDisposition, PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature,
    ScanLimitDisposition, SchemaMappingDisposition, SourceDiagnostics, SourceFormat, ZeroColumnCountDisposition,
};

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
#[allow(clippy::large_enum_variant)]
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
    /// Creates a missing-column descriptor.
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
    }

    /// Recomputes diagnostics from the current source metadata.
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
        Expr::Binary { left, right, .. } => {
            reader_predicate_expr_is_supported(source, left) && reader_predicate_expr_is_supported(source, right)
        },
        Expr::InList { expr, list, .. } => {
            !list.is_empty()
                && reader_predicate_expr_is_supported(source, expr)
                && list
                    .iter()
                    .all(|value| matches!(value, Expr::Literal(_)) && reader_predicate_expr_is_supported(source, value))
        },
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Case { .. } | Expr::Extract { .. } => false,
    }
}

fn reader_predicate_column_type_is_supported(data_type: &DataType) -> bool {
    !matches!(data_type, DataType::Decimal128(_, _))
}

fn reader_predicate_literal_is_supported(value: &LiteralValue) -> bool {
    !matches!(
        value,
        LiteralValue::Null(_) | LiteralValue::IntervalMonthDayNano { .. } | LiteralValue::Decimal128 { .. }
    )
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
        Self {
            path,
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
