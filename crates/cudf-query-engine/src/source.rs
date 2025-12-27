use std::{path::PathBuf, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::expr::LiteralValue;

pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__cudf_zero_column_row_count";

#[derive(Debug, Clone, PartialEq)]
pub enum SourceNode {
    LocalParquet(LocalParquetSource),
    GpuResident(GpuResidentSource),
}

impl SourceNode {
    pub fn output_schema(&self) -> &SchemaRef {
        match self {
            Self::LocalParquet(source) => &source.schema,
            Self::GpuResident(source) => &source.schema,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuResidentSource {
    pub input_name: String,
    pub schema: SchemaRef,
}

impl GpuResidentSource {
    pub fn new(input_name: impl Into<String>, schema: SchemaRef) -> Self {
        Self { input_name: input_name.into(), schema }
    }
}

fn schema_column_indices(schema: &SchemaRef, columns: &[String]) -> Vec<usize> {
    columns.iter().filter_map(|column| schema.index_of(column).ok()).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

impl PartitionColumn {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self { name: name.into(), data_type, nullable }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceColumnRef {
    File(usize),
    Partition(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetSource {
    pub files: Vec<LocalParquetFile>,
    pub schema: SchemaRef,
    pub read_columns: Vec<String>,
    pub read_column_file_indices: Vec<usize>,
    pub projected_output_columns: Vec<String>,
    pub projected_output_column_source_refs: Vec<SourceColumnRef>,
    pub pure_filter_columns: Vec<String>,
    pub pure_filter_column_source_refs: Vec<SourceColumnRef>,
    pub partition_columns: Vec<PartitionColumn>,
    pub diagnostics: SourceDiagnostics,
}

impl LocalParquetSource {
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

    pub fn new(path: impl Into<PathBuf>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        Self::from_files(vec![LocalParquetFile::new(path)], schema, read_columns)
    }

    pub fn zero_column_count(mut files: Vec<LocalParquetFile>) -> Self {
        for file in &mut files {
            file.partition_values.clear();
        }
        let schema = Arc::new(Schema::new(vec![Field::new(ZERO_COLUMN_COUNT_COLUMN, DataType::Int64, false)]));
        let mut source = Self::from_files(files, schema, Vec::new());
        source.mark_zero_column_count_represented();
        source
    }

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

    pub fn with_partition_columns(mut self, partition_columns: Vec<PartitionColumn>) -> Self {
        self.partition_columns = partition_columns;
        self.refresh_diagnostics();
        self
    }

    pub fn refresh_diagnostics(&mut self) {
        let zero_column_count_represented = self.is_zero_column_count_source();
        self.diagnostics = SourceDiagnostics::for_local_parquet_files(&self.files, self.diagnostic_columns());
        if zero_column_count_represented {
            self.mark_zero_column_count_represented();
        }
    }

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

    pub fn read_column_count(&self) -> usize {
        self.read_columns.len()
    }

    pub fn projected_output_column_count(&self) -> usize {
        self.projected_output_columns.len()
    }

    pub fn pure_filter_column_count(&self) -> usize {
        self.pure_filter_columns.len()
    }

    pub fn selected_row_group_count(&self) -> usize {
        self.files.iter().map(LocalParquetFile::selected_row_group_count).sum()
    }

    pub fn ranged_file_count(&self) -> usize {
        self.diagnostics.ranged_file_count
    }

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
        }
    }

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
    pub path: PathBuf,
    pub stable_id: String,
    pub file_size_bytes: Option<u64>,
    pub row_groups: Option<Vec<usize>>,
    pub row_group_bytes: Option<Vec<u64>>,
    pub row_group_rows: Option<Vec<u64>>,
    pub partition_values: Vec<LiteralValue>,
}

impl LocalParquetFile {
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

    pub fn with_stable_id(mut self, stable_id: impl Into<String>) -> Self {
        self.stable_id = stable_id.into();
        self
    }

    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.row_groups = Some(row_groups);
        self
    }

    pub fn with_row_group_bytes(mut self, row_group_bytes: Vec<u64>) -> Self {
        self.row_group_bytes = Some(row_group_bytes);
        self
    }

    pub fn with_row_group_rows(mut self, row_group_rows: Vec<u64>) -> Self {
        self.row_group_rows = Some(row_group_rows);
        self
    }

    pub fn with_partition_values(mut self, partition_values: Vec<LiteralValue>) -> Self {
        self.partition_values = partition_values;
        self
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDiagnostics {
    pub format: SourceFormat,
    pub direct_device_read: bool,
    pub file_count: usize,
    pub ranged_file_count: usize,
    pub known_bytes: Option<u64>,
    pub selected_row_groups: Option<usize>,
    pub predicate_disposition: PredicateDisposition,
    pub byte_range_disposition: ByteRangeDisposition,
    pub partition_column_disposition: PartitionColumnDisposition,
    pub zero_column_count_disposition: ZeroColumnCountDisposition,
    pub schema_mapping_disposition: SchemaMappingDisposition,
    pub read_column_count: usize,
    pub projected_output_column_count: usize,
    pub pure_filter_column_count: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    Parquet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredicateDisposition {
    Absent,
    PruningOnly,
    Rejected,
}

impl PredicateDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::PruningOnly => "pruning_only",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteRangeDisposition {
    FullFiles,
    CoalescedFullFiles,
    Rejected,
}

impl ByteRangeDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullFiles => "full_files",
            Self::CoalescedFullFiles => "coalesced_full_files",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionColumnDisposition {
    Absent,
    Rejected,
    Represented,
}

impl PartitionColumnDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Rejected => "rejected",
            Self::Represented => "represented",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeroColumnCountDisposition {
    NotRequired,
    Rejected,
    Represented,
}

impl ZeroColumnCountDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::Rejected => "rejected",
            Self::Represented => "represented",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaMappingDisposition {
    DirectByName,
    Rejected,
}

impl SchemaMappingDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectByName => "direct_by_name",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectedSourceFeature {
    NonLocalObjectStore,
    PartitionColumns,
    ExprAdapter,
    ScanLimit,
    PreserveOrder,
    OutputOrdering,
    PushedDownParquetPredicate,
    EmptySource,
    PartitionValues,
    ByteRange,
    FileOrdering,
    RowSelection,
    ZeroColumnCount,
    SchemaMapping,
}

impl RejectedSourceFeature {
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
