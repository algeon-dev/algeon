use std::path::PathBuf;

use arrow::datatypes::SchemaRef;

#[derive(Debug, Clone, PartialEq)]
pub enum SourceNode {
    LocalParquet(LocalParquetSource),
}

impl SourceNode {
    pub fn output_schema(&self) -> &SchemaRef {
        match self {
            Self::LocalParquet(source) => &source.schema,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetSource {
    pub files: Vec<LocalParquetFile>,
    pub schema: SchemaRef,
    pub read_columns: Vec<String>,
    pub diagnostics: SourceDiagnostics,
}

impl LocalParquetSource {
    pub fn from_files(files: Vec<LocalParquetFile>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        let diagnostics = SourceDiagnostics::for_local_parquet_files(&files);
        Self { files, schema, read_columns, diagnostics }
    }

    pub fn new(path: impl Into<PathBuf>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        Self::from_files(vec![LocalParquetFile::new(path)], schema, read_columns)
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn read_column_count(&self) -> usize {
        self.read_columns.len()
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalParquetFile {
    pub path: PathBuf,
    pub file_size_bytes: Option<u64>,
    pub row_groups: Option<Vec<usize>>,
}

impl LocalParquetFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), file_size_bytes: None, row_groups: None }
    }

    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.row_groups = Some(row_groups);
        self
    }

    pub fn selected_row_group_count(&self) -> usize {
        self.row_groups.as_ref().map_or(0, Vec::len)
    }
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
    pub rejected_features: Vec<RejectedSourceFeature>,
}

impl SourceDiagnostics {
    fn for_local_parquet_files(files: &[LocalParquetFile]) -> Self {
        let known_bytes = files
            .iter()
            .map(|file| file.file_size_bytes)
            .try_fold(0u64, |total, bytes| bytes.map(|bytes| total.saturating_add(bytes)));
        let selected_row_groups = files
            .iter()
            .any(|file| file.row_groups.is_some())
            .then(|| files.iter().map(LocalParquetFile::selected_row_group_count).sum());
        Self {
            format: SourceFormat::Parquet,
            direct_device_read: true,
            file_count: files.len(),
            ranged_file_count: 0,
            known_bytes,
            selected_row_groups,
            predicate_disposition: PredicateDisposition::Absent,
            rejected_features: Vec::new(),
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
        }
    }
}
