//! DataFusion-free native Parquet source contracts.

#[cfg(test)]
use std::path::Path;
use std::{collections::BTreeMap, fmt, path::PathBuf, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

#[cfg(test)]
use super::diagnostics::{redacted_object_store_path_diagnostic, redacted_stable_id_diagnostic};
use super::{
    deletes::{ParquetEqualityDeleteFile, ParquetPositionDeleteFile},
    diagnostics::{
        LocalParquetDiagnosticColumns, PredicateDisposition, RejectedSourceFeature, SourceDiagnostics,
        ZeroColumnCountDisposition, files_have_proven_direct_device_read, redacted_debug_value,
    },
    location::{ParquetSourceLocation, object_store_reader_path, parquet_source_location_for_original},
    predicate::{
        reader_predicate_expr_is_supported, reader_predicate_references_decimal_column,
        reader_predicate_root_is_supported, reader_predicate_source_contract_shape_is_supported,
        rewrite_fixed_prefix_substring_reader_predicate,
    },
    read_policy::{CredentialMode, ResolvedParquetSource},
};
use crate::expr::{Expr, LiteralValue};

/// Synthetic output column used for Parquet plans that only need a row count.
///
/// A zero-column count source reads no physical columns and produces row counts
/// from file or row-group metadata instead of projecting normal source columns.
pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__cudf_zero_column_row_count";

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
        Ok(Self {
            location,
            path: PathBuf::from(object_store_reader_path(&uri)),
            resolved_source: ResolvedParquetSource::remote_kvikio(uri, CredentialMode::SharedProcessEnv),
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
            path: PathBuf::from(resolved_source.reader_location()),
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
        self.path = PathBuf::from(resolved_source.reader_location());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_local_file_uri_uses_filesystem_path_for_location_and_reader_path() {
        let mut source = ResolvedParquetSource::local_kvikio("/tmp/table.parquet");
        source.original_location = "file:/tmp/table.parquet".to_string();
        let file = LocalParquetFile::from_resolved_parquet_source(source).unwrap();

        assert_eq!(file.location.as_local_path(), Some(Path::new("/tmp/table.parquet")));
        assert_eq!(file.path, PathBuf::from("/tmp/table.parquet"));
    }

    #[test]
    fn resolved_local_file_uri_with_localhost_authority_is_local_path() {
        let mut source = ResolvedParquetSource::local_kvikio("/tmp/table.parquet");
        source.original_location = "file://localhost/tmp/table.parquet".to_string();
        let file = LocalParquetFile::from_resolved_parquet_source(source).unwrap();

        assert_eq!(file.location.as_local_path(), Some(Path::new("/tmp/table.parquet")));
    }

    #[test]
    fn remote_direct_source_diagnostics_are_not_proven_direct_device_reads() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let file = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        ))
        .unwrap();
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["id".to_string()]);

        assert_eq!(source.diagnostics.source_read_strategy, "remote_direct");
        assert!(!source.diagnostics.direct_device_read);
        assert!(source.diagnostics.rejected_features.is_empty());
    }

    #[test]
    fn object_store_uri_defaults_to_tagged_remote_kvikio_input() {
        let uri = "s3://bucket/table/part-0.parquet";
        let file = LocalParquetFile::object_store_uri(uri).unwrap();

        assert!(matches!(
            &file.resolved_source.input,
            crate::source::ResolvedParquetInput::RemoteKvikio {
                uri: resolved_uri,
                credential_mode: CredentialMode::SharedProcessEnv,
                storage_profile_id: None,
            } if resolved_uri == uri
        ));
        assert_eq!(file.resolved_source.reader_location(), uri);
        assert_eq!(file.resolved_source.read_strategy(), crate::source::ParquetReadStrategy::RemoteDirect);
    }

    #[test]
    fn remote_and_local_cache_object_store_identity_stays_with_its_tagged_access_path() {
        let remote =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::ScopedResolved)
                .with_storage_profile_id("remote-profile");
        assert!(matches!(
            &remote.input,
            crate::source::ResolvedParquetInput::RemoteKvikio {
                credential_mode: CredentialMode::ScopedResolved,
                storage_profile_id: Some(profile_id),
                ..
            } if profile_id == "remote-profile"
        ));

        let cached = ResolvedParquetSource::local_cache_kvikio(
            "s3://bucket/table/part-0.parquet",
            "/tmp/cache/part-0.parquet",
            CredentialMode::ScopedResolved,
            crate::source::LakehouseCacheStatus::Hit,
        )
        .with_storage_profile_id("maintenance-profile");
        assert!(matches!(
            &cached.input,
            crate::source::ResolvedParquetInput::LocalKvikio { reader_path }
                if reader_path == "/tmp/cache/part-0.parquet"
        ));
        assert_eq!(cached.credential_mode(), CredentialMode::ScopedResolved);
        assert_eq!(cached.storage_profile_id(), Some("maintenance-profile"));
    }

    #[test]
    fn object_store_default_stable_ids_are_redacted_hashes() {
        let uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let file = LocalParquetFile::from_resolved_parquet_source(ResolvedParquetSource::remote_kvikio(
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
        let source = ResolvedParquetSource::remote_kvikio(uri, CredentialMode::SharedProcessEnv)
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
            ResolvedParquetSource::remote_kvikio(
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
