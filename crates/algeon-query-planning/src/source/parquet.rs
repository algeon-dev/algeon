//! DataFusion-free native Parquet source contracts.

#[cfg(test)]
use std::path::Path;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::PathBuf,
    sync::Arc,
};

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
use crate::{
    backend::AnalysisContext,
    expr::{Expr, LiteralValue},
    parquet_stats::ParquetVariableWidthPayloadEvidence,
};

/// Synthetic output column used for Parquet plans that only need a row count.
///
/// A zero-column count source reads no physical columns and produces row counts
/// from file or row-group metadata instead of projecting normal source columns.
pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__algeon_native_zero_column_row_count";

/// Parquet-footer key for a graph vertex upper bound tied to the complete edge
/// rowset in one file.
///
/// The producer must derive the value from the same edge artifact it writes.
/// Consumers only use it when the native source still represents the complete,
/// unfiltered file rowset; it is deliberately not a query/session override.
pub const GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY: &str = "algeon.graph.vertex_upper_bound.v1";

/// Parquet-footer key for a complete, simple directed edge rowset.
///
/// Its JSON value declares the physical root-column indices for the source and
/// destination endpoints. The producer attests that both endpoints are
/// non-null and each ordered endpoint pair appears exactly once. Consumers use
/// it only for one complete, unfiltered file; a per-file proof cannot establish
/// uniqueness across a multi-file source.
pub const GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY: &str =
    "algeon.graph.simple_directed_edge_topology.v1";

/// Footer-proven topology needed to skip Python-compatible edge normalization.
///
/// This is deliberately a narrow source contract. It says nothing about graph
/// algorithms or their workspace: it only proves that filtering null endpoints
/// and deduplicating a directed edge list are semantic no-ops for these two
/// physical Parquet columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphSimpleDirectedEdgeTopology {
    /// Zero-based physical Parquet root-column index for the source endpoint.
    pub source_column_index: usize,
    /// Zero-based physical Parquet root-column index for the destination endpoint.
    pub destination_column_index: usize,
}

impl GraphSimpleDirectedEdgeTopology {
    /// Creates a simple-directed topology proof for two distinct endpoint columns.
    pub const fn new(source_column_index: usize, destination_column_index: usize) -> Option<Self> {
        if source_column_index == destination_column_index {
            return None;
        }
        Some(Self { source_column_index, destination_column_index })
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
    /// Source-owner unique keys expressed in emitted physical column names.
    pub unique_key_columns: Vec<BTreeSet<String>>,
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
            unique_key_columns: Vec::new(),
            diagnostics,
            count_star_probe: false,
        }
    }

    pub fn new(path: impl Into<PathBuf>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        Self::from_files(vec![LocalParquetFile::new(path)], schema, read_columns)
    }

    pub fn with_unique_key_columns(mut self, unique_key_columns: Vec<BTreeSet<String>>) -> Self {
        self.unique_key_columns = unique_key_columns;
        self
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
        if self.output_schema_override.is_none() {
            let mut seen = BTreeSet::new();
            let fields = self
                .projected_output_columns
                .iter()
                .chain(&self.pure_filter_columns)
                .filter(|column| seen.insert((*column).clone()))
                .map(|column| self.schema.field_with_name(column).cloned())
                .collect::<std::result::Result<Vec<_>, _>>();
            if let Ok(fields) = fields {
                self.output_schema_override = Some(Arc::new(Schema::new(fields)));
            }
        }
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
    pub fn reader_predicate_is_supported(&self, predicate: &Expr, ctx: AnalysisContext<'_>) -> bool {
        reader_predicate_root_is_supported(predicate)
            && reader_predicate_expr_is_supported(self, predicate, ctx)
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
    pub fn reader_predicate_supported_expression(&self, predicate: &Expr, ctx: AnalysisContext<'_>) -> Option<Expr> {
        if self.reader_predicate_is_supported(predicate, ctx) {
            return Some(predicate.clone());
        }
        let rewritten = rewrite_fixed_prefix_substring_reader_predicate(predicate)?;
        self.reader_predicate_is_supported(&rewritten, ctx).then_some(rewritten)
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

    #[doc(hidden)]
    pub fn exact_selected_row_group_rows(&self) -> Option<Vec<u64>> {
        self.files.iter().try_fold(Vec::with_capacity(self.selected_row_group_count()), |mut selected_rows, file| {
            let row_group_rows = file.row_group_rows.as_ref()?;
            match file.row_groups.as_ref() {
                Some(row_groups) => {
                    for row_group in row_groups {
                        selected_rows.push(*row_group_rows.get(*row_group)?);
                    }
                },
                None => selected_rows.extend(row_group_rows.iter().copied()),
            }
            Some(selected_rows)
        })
    }

    /// Returns the exact number of rows represented by the selected Parquet
    /// row groups, when every selected group has footer-backed row metadata.
    ///
    /// This is deliberately narrower than the source's descriptive estimates:
    /// callers that need an exact graph-input cardinality cannot substitute
    /// file sizes, optimizer statistics, or an unselected row-group total.
    pub fn exact_selected_row_count(&self) -> Option<u64> {
        self.exact_selected_row_group_rows()?.into_iter().try_fold(0_u64, |total, rows| total.checked_add(rows))
    }

    /// Returns a footer-proven graph vertex upper bound when this source still
    /// denotes complete, unfiltered edge-file rowsets.
    ///
    /// A graph vertex count belongs to the exact source rowset that produced
    /// it. A scan limit, reader predicate, row-level delete, or selected
    /// row-group subset changes that rowset, so this deliberately refuses to
    /// carry the file-level value across such a transformation. Multiple whole
    /// files are safe to sum: duplicate vertices only make the result more
    /// conservative.
    pub fn graph_vertex_upper_bound(&self) -> Option<u64> {
        if !self.has_complete_untransformed_graph_rowset() {
            return None;
        }

        self.files.iter().try_fold(0_u64, |total, file| total.checked_add(file.graph_vertex_upper_bound?))
    }

    /// Returns the footer-proven raw edge-row count when this source still
    /// denotes complete, unfiltered edge-file rowsets.
    ///
    /// Edge normalization retains this raw source frame while its normalized
    /// graph frame is live. The count therefore belongs to the source rowset,
    /// not to a transformed graph operator or a caller-provided estimate.
    pub fn graph_raw_edge_row_count(&self) -> Option<u64> {
        self.has_complete_untransformed_graph_rowset().then(|| self.exact_selected_row_count()).flatten()
    }

    /// Returns whether this source proves a simple directed edge relation for
    /// the two requested physical file columns.
    ///
    /// A file footer cannot prove that endpoint pairs are unique *across*
    /// several files, so this deliberately accepts exactly one complete local
    /// Parquet file. Column roles must resolve directly to physical file
    /// columns, which prevents aliases, partitions, or materialized nulls from
    /// borrowing the proof.
    pub fn has_simple_directed_edge_topology(&self, source_column: &str, destination_column: &str) -> bool {
        if !self.has_complete_untransformed_graph_rowset() || self.files.len() != 1 {
            return false;
        }
        let (Some(SourceColumnRef::File(source_column_index)), Some(SourceColumnRef::File(destination_column_index))) =
            (self.source_column_ref(source_column), self.source_column_ref(destination_column))
        else {
            return false;
        };
        self.files[0].graph_simple_directed_edge_topology
            == GraphSimpleDirectedEdgeTopology::new(source_column_index, destination_column_index)
    }

    fn has_complete_untransformed_graph_rowset(&self) -> bool {
        self.scan_limit.is_none()
            && self.reader_predicate.is_none()
            && !self.has_row_delete_files()
            && self.files.iter().all(|file| file.row_groups.is_none())
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
    /// Estimated or known row counts in physical Parquet row-group order.
    ///
    /// The vector covers every physical row group; `row_groups`, when present,
    /// contains the physical IDs selected for this source.
    pub row_group_rows: Option<Vec<u64>>,
    /// Footer-proven upper bound on distinct graph vertices in this complete
    /// edge file, when the fixture/data producer attached one.
    ///
    /// This is not applied to a partial row-group selection or a transformed
    /// source. [`LocalParquetSource::graph_vertex_upper_bound`] enforces that
    /// rowset-identity boundary before admission consumes it.
    pub graph_vertex_upper_bound: Option<u64>,
    /// Footer-proven simple directed-edge topology for this complete file.
    ///
    /// The proof is intentionally stored per file and becomes usable only
    /// through [`LocalParquetSource::has_simple_directed_edge_topology`], which
    /// refuses partial or multi-file sources.
    pub graph_simple_directed_edge_topology: Option<GraphSimpleDirectedEdgeTopology>,
    /// Per-file-column null counts by row group, keyed by physical root column index.
    ///
    /// Inner vectors are indexed by physical row-group number, not selected
    /// row-group position. `None` means the footer did not prove a null count
    /// for that column chunk.
    pub row_group_null_counts: BTreeMap<usize, Vec<Option<u64>>>,
    /// Footer-proven variable-width payload bounds keyed by physical root-column index.
    pub variable_width_payload_evidence: BTreeMap<usize, ParquetVariableWidthPayloadEvidence>,
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
            .field("graph_vertex_upper_bound", &self.graph_vertex_upper_bound)
            .field("graph_simple_directed_edge_topology", &self.graph_simple_directed_edge_topology)
            .field("row_group_null_counts", &self.row_group_null_counts)
            .field("variable_width_payload_evidence", &self.variable_width_payload_evidence)
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
            graph_vertex_upper_bound: None,
            graph_simple_directed_edge_topology: None,
            row_group_null_counts: BTreeMap::new(),
            variable_width_payload_evidence: BTreeMap::new(),
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
            graph_vertex_upper_bound: None,
            graph_simple_directed_edge_topology: None,
            row_group_null_counts: BTreeMap::new(),
            variable_width_payload_evidence: BTreeMap::new(),
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
            graph_vertex_upper_bound: None,
            graph_simple_directed_edge_topology: None,
            row_group_null_counts: BTreeMap::new(),
            variable_width_payload_evidence: BTreeMap::new(),
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

    /// Sets row-count metadata in physical Parquet row-group order.
    pub fn with_row_group_rows(mut self, row_group_rows: Vec<u64>) -> Self {
        self.row_group_rows = Some(row_group_rows);
        self
    }

    /// Attaches a footer-proven graph vertex upper bound for this complete
    /// edge-file rowset.
    ///
    /// Frontend code should populate this only by reading
    /// [`GRAPH_VERTEX_UPPER_BOUND_PARQUET_METADATA_KEY`] from this file's
    /// Parquet footer. It stays on the file descriptor so source identity and
    /// graph admission cannot accidentally borrow a detached query setting.
    pub fn with_graph_vertex_upper_bound(mut self, graph_vertex_upper_bound: u64) -> Self {
        self.graph_vertex_upper_bound = Some(graph_vertex_upper_bound);
        self
    }

    /// Attaches a footer-proven simple-directed edge topology for this complete
    /// Parquet file.
    ///
    /// Frontends must populate this only from
    /// [`GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_PARQUET_METADATA_KEY`]. The source
    /// contract checks complete-rowset and single-file identity before graph
    /// lowering can use it.
    pub fn with_graph_simple_directed_edge_topology(
        mut self,
        graph_simple_directed_edge_topology: GraphSimpleDirectedEdgeTopology,
    ) -> Self {
        self.graph_simple_directed_edge_topology = Some(graph_simple_directed_edge_topology);
        self
    }

    /// Sets row-group null-count metadata for one physical file column.
    pub fn with_row_group_null_counts(mut self, file_column_index: usize, null_counts: Vec<Option<u64>>) -> Self {
        self.row_group_null_counts.insert(file_column_index, null_counts);
        self
    }

    /// Sets footer-proven variable-width payload evidence keyed by physical root-column index.
    pub fn with_variable_width_payload_evidence(
        mut self,
        evidence: BTreeMap<usize, ParquetVariableWidthPayloadEvidence>,
    ) -> Self {
        self.variable_width_payload_evidence = evidence;
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

    /// Footer-proven variable-width payload bounds keyed by physical root-column index.
    pub fn variable_width_payload_evidence(&self) -> &BTreeMap<usize, ParquetVariableWidthPayloadEvidence> {
        &self.variable_width_payload_evidence
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
    fn column_roles_expose_projected_then_filter_only_fields_without_duplicates() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("filter", DataType::Boolean, false),
            Field::new("shared", DataType::Utf8, true),
            Field::new("projected", DataType::Int64, false),
        ]));
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("roles.parquet")],
            schema,
            vec!["filter".to_owned(), "shared".to_owned(), "projected".to_owned()],
        )
        .with_column_roles(
            vec!["projected".to_owned(), "shared".to_owned()],
            vec!["filter".to_owned(), "shared".to_owned()],
        );

        assert_eq!(
            source.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
            ["projected", "shared", "filter"]
        );
        assert_eq!(source.output_schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(source.output_schema().field(1).data_type(), &DataType::Utf8);
        assert_eq!(source.output_schema().field(2).data_type(), &DataType::Boolean);
    }

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

    #[test]
    fn exact_selected_row_count_uses_only_footer_proven_selected_groups() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("first.parquet").with_row_groups(vec![0, 2]).with_row_group_rows(vec![3, 5, 7]),
                LocalParquetFile::new("second.parquet").with_row_group_rows(vec![11]),
            ],
            schema,
            vec!["id".to_string()],
        );

        assert_eq!(source.exact_selected_row_count(), Some(21));
    }

    #[test]
    fn graph_vertex_upper_bound_requires_complete_untransformed_file_rowsets() {
        let schema = Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false)]));
        let complete = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("first.parquet").with_graph_vertex_upper_bound(7),
                LocalParquetFile::new("second.parquet").with_graph_vertex_upper_bound(11),
            ],
            Arc::clone(&schema),
            vec!["src".to_string()],
        );

        assert_eq!(complete.graph_vertex_upper_bound(), Some(18));
        assert_eq!(complete.graph_raw_edge_row_count(), None);
        assert_eq!(complete.clone().with_scan_limit(Some(1)).graph_vertex_upper_bound(), None);
        assert_eq!(complete.with_scan_limit(Some(1)).graph_raw_edge_row_count(), None);

        let selected_row_groups = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("first.parquet").with_row_groups(vec![0]).with_graph_vertex_upper_bound(7)],
            schema,
            vec!["src".to_string()],
        );
        assert_eq!(selected_row_groups.graph_vertex_upper_bound(), None);
        assert_eq!(selected_row_groups.graph_raw_edge_row_count(), None);
    }

    #[test]
    fn graph_raw_edge_row_count_requires_complete_footer_proven_rowsets() {
        let schema = Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false)]));
        let complete = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("first.parquet").with_row_group_rows(vec![3, 5]),
                LocalParquetFile::new("second.parquet").with_row_group_rows(vec![7]),
            ],
            Arc::clone(&schema),
            vec!["src".to_string()],
        );
        assert_eq!(complete.graph_raw_edge_row_count(), Some(15));

        let partial = LocalParquetSource::from_files(
            vec![LocalParquetFile::new("first.parquet").with_row_group_rows(vec![3, 5]).with_row_groups(vec![0])],
            schema,
            vec!["src".to_string()],
        );
        assert_eq!(partial.graph_raw_edge_row_count(), None);
    }

    #[test]
    fn simple_directed_edge_topology_requires_one_complete_matching_file() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]));
        let topology = GraphSimpleDirectedEdgeTopology::new(0, 1).expect("distinct endpoint columns");
        let complete = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("edges.parquet")
                    .with_row_group_rows(vec![3])
                    .with_graph_simple_directed_edge_topology(topology),
            ],
            Arc::clone(&schema),
            vec!["src".to_string(), "dst".to_string()],
        );

        assert!(complete.has_simple_directed_edge_topology("src", "dst"));
        assert!(!complete.has_simple_directed_edge_topology("dst", "src"));
        assert!(!complete.with_scan_limit(Some(1)).has_simple_directed_edge_topology("src", "dst"));

        let multi_file = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("first.parquet").with_graph_simple_directed_edge_topology(topology),
                LocalParquetFile::new("second.parquet").with_graph_simple_directed_edge_topology(topology),
            ],
            schema,
            vec!["src".to_string(), "dst".to_string()],
        );
        assert!(!multi_file.has_simple_directed_edge_topology("src", "dst"));
    }
}
