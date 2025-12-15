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
    pub path: PathBuf,
    pub file_size_bytes: Option<u64>,
    pub schema: SchemaRef,
    pub read_columns: Vec<String>,
    pub row_groups: Option<Vec<usize>>,
    pub diagnostics: SourceDiagnostics,
}

impl LocalParquetSource {
    pub fn new(path: impl Into<PathBuf>, schema: SchemaRef, read_columns: Vec<String>) -> Self {
        Self {
            path: path.into(),
            file_size_bytes: None,
            schema,
            read_columns,
            row_groups: None,
            diagnostics: SourceDiagnostics::local_parquet(),
        }
    }

    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self.diagnostics.known_bytes = Some(file_size_bytes);
        self
    }

    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.diagnostics.selected_row_groups = Some(row_groups.len());
        self.row_groups = Some(row_groups);
        self
    }

    pub fn read_column_count(&self) -> usize {
        self.read_columns.len()
    }

    pub fn selected_row_group_count(&self) -> usize {
        self.row_groups.as_ref().map_or(0, Vec::len)
    }

    pub fn is_direct_device_read(&self) -> bool {
        self.diagnostics.direct_device_read
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDiagnostics {
    pub format: SourceFormat,
    pub direct_device_read: bool,
    pub known_bytes: Option<u64>,
    pub selected_row_groups: Option<usize>,
}

impl SourceDiagnostics {
    pub fn local_parquet() -> Self {
        Self { format: SourceFormat::Parquet, direct_device_read: true, known_bytes: None, selected_row_groups: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    Parquet,
}
