use super::*;

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
    /// Immutable content identity retained for native source inspection.
    pub source_content_identity: Option<String>,
    /// Required read capability for this location.
    pub read_capability: TableFormatReadCapability,
    /// Resolved Parquet read strategy for native source lowering.
    pub resolved_parquet_source: Option<ResolvedParquetSource>,
    /// Footer Arrow schema with Parquet field-id metadata, when the table-format
    /// planner fetched it during metadata planning.
    pub footer_schema: Option<SchemaRef>,
    /// Footer-proven row-group null counts by physical root column index.
    ///
    /// Inner vectors are indexed by physical row-group number. `None` entries
    /// mean the footer did not prove a null count for that column chunk.
    pub row_group_null_counts: BTreeMap<usize, Vec<Option<u64>>>,
    /// Partition values attached to this data file.
    pub partition_values: Vec<TableFormatPartitionValue>,
    /// Supported position-delete files associated with this data file.
    pub position_delete_files: Vec<TableFormatDeleteFile>,
    /// Supported equality-delete files associated with this data file.
    pub equality_delete_files: Vec<TableFormatDeleteFile>,
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
            .field("source_content_identity_available", &self.source_content_identity.is_some())
            .field("read_capability", &self.read_capability)
            .field("resolved_parquet_source", &self.resolved_parquet_source)
            .field("footer_schema_available", &self.footer_schema.is_some())
            .field("row_group_null_counts", &self.row_group_null_counts)
            .field("partition_values", &self.partition_values)
            .field("position_delete_files", &self.position_delete_files)
            .field("equality_delete_files", &self.equality_delete_files)
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
            source_content_identity: None,
            read_capability,
            resolved_parquet_source: None,
            footer_schema: None,
            row_group_null_counts: BTreeMap::new(),
            partition_values: Vec::new(),
            position_delete_files: Vec::new(),
            equality_delete_files: Vec::new(),
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

    /// Records an immutable source content identity for later native inspection.
    pub fn with_source_content_identity(mut self, content_identity: impl Into<String>) -> Self {
        self.source_content_identity = Some(content_identity.into());
        self
    }

    /// Records the resolved Parquet source strategy for this data file.
    pub fn with_resolved_parquet_source(mut self, resolved_source: ResolvedParquetSource) -> Self {
        self.resolved_parquet_source = Some(resolved_source);
        self
    }

    /// Records the footer Arrow schema used for Iceberg field-id proof.
    pub fn with_footer_schema(mut self, schema: SchemaRef) -> Self {
        self.footer_schema = Some(schema);
        self
    }

    /// Records footer-proven row-group null counts by physical root column index.
    pub fn with_row_group_null_counts(mut self, null_counts: BTreeMap<usize, Vec<Option<u64>>>) -> Self {
        self.row_group_null_counts = null_counts;
        self
    }

    /// Records partition values attached to this file.
    pub fn with_partition_values(mut self, partition_values: Vec<TableFormatPartitionValue>) -> Self {
        self.partition_values = partition_values;
        self
    }

    /// Records supported position-delete files attached to this data file.
    pub fn with_position_delete_files(mut self, position_delete_files: Vec<TableFormatDeleteFile>) -> Self {
        self.position_delete_files = position_delete_files;
        self
    }

    /// Records supported equality-delete files attached to this data file.
    pub fn with_equality_delete_files(mut self, equality_delete_files: Vec<TableFormatDeleteFile>) -> Self {
        self.equality_delete_files = equality_delete_files;
        self
    }

    /// Adds one supported position-delete file attached to this data file.
    pub fn push_position_delete_file(&mut self, delete_file: TableFormatDeleteFile) {
        self.position_delete_files.push(delete_file);
    }

    /// Adds one supported equality-delete file attached to this data file.
    pub fn push_equality_delete_file(&mut self, delete_file: TableFormatDeleteFile) {
        self.equality_delete_files.push(delete_file);
    }

    /// Whether this data file has supported position deletes attached.
    pub fn has_position_delete_files(&self) -> bool {
        !self.position_delete_files.is_empty()
    }

    /// Whether this data file has supported equality deletes attached.
    pub fn has_equality_delete_files(&self) -> bool {
        !self.equality_delete_files.is_empty()
    }

    /// Whether this data file has row-level delete files attached.
    pub fn has_row_delete_files(&self) -> bool {
        self.has_position_delete_files() || self.has_equality_delete_files()
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
    /// Stable data-file id this delete file applies to.
    pub data_file_id: String,
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
    /// Referenced data-file path from Iceberg position-delete metadata, when present.
    pub referenced_data_file: Option<String>,
    /// Inclusive lower bound for the Iceberg position-delete `pos` column.
    pub position_lower_bound: Option<u64>,
    /// Inclusive upper bound for the Iceberg position-delete `pos` column.
    pub position_upper_bound: Option<u64>,
    /// Required read capability for this delete file.
    pub read_capability: TableFormatReadCapability,
    /// Resolved Parquet read strategy for native source lowering.
    pub resolved_parquet_source: Option<ResolvedParquetSource>,
    /// Iceberg equality-delete field ids for equality-delete files.
    pub equality_ids: Option<Vec<i32>>,
}

impl fmt::Debug for TableFormatDeleteFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableFormatDeleteFile")
            .field("id", &redacted_scan_debug_value("delete_file_id", &self.id))
            .field("data_file_id", &redacted_scan_debug_value("data_file_id", &self.data_file_id))
            .field("location", &redacted_scan_location_debug_value(&self.location))
            .field("format", &self.format)
            .field("content", &self.content)
            .field("size_bytes", &self.size_bytes)
            .field("row_count", &self.row_count)
            .field(
                "referenced_data_file",
                &self.referenced_data_file.as_deref().map(redacted_scan_location_debug_value),
            )
            .field("position_lower_bound", &self.position_lower_bound)
            .field("position_upper_bound", &self.position_upper_bound)
            .field("read_capability", &self.read_capability)
            .field("resolved_parquet_source", &self.resolved_parquet_source)
            .field("equality_ids", &self.equality_ids)
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
        Self::for_data_file(id, "", location, format, content)
    }

    /// Builds a delete-file fact associated with one planned data file.
    pub fn for_data_file(
        id: impl Into<String>,
        data_file_id: impl Into<String>,
        location: impl Into<String>,
        format: TableFormatDataFileFormat,
        content: TableFormatDeleteFileContent,
    ) -> Self {
        Self {
            id: id.into(),
            data_file_id: data_file_id.into(),
            location: location.into(),
            format,
            content,
            size_bytes: None,
            row_count: None,
            referenced_data_file: None,
            position_lower_bound: None,
            position_upper_bound: None,
            read_capability: TableFormatReadCapability::unsupported("unresolved"),
            resolved_parquet_source: None,
            equality_ids: None,
        }
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

    /// Records the referenced data-file path for manifest-proven position-delete pruning.
    pub fn with_referenced_data_file(mut self, referenced_data_file: impl Into<String>) -> Self {
        self.referenced_data_file = Some(referenced_data_file.into());
        self
    }

    /// Records inclusive Iceberg position-delete `pos` bounds.
    pub fn with_position_bounds(mut self, lower_bound: Option<u64>, upper_bound: Option<u64>) -> Self {
        self.position_lower_bound = lower_bound;
        self.position_upper_bound = upper_bound;
        self
    }

    /// Records required read capability for this delete file.
    pub fn with_read_capability(mut self, read_capability: TableFormatReadCapability) -> Self {
        self.read_capability = read_capability;
        self
    }

    /// Records the resolved Parquet source strategy for this delete file.
    pub fn with_resolved_parquet_source(mut self, resolved_source: ResolvedParquetSource) -> Self {
        self.resolved_parquet_source = Some(resolved_source);
        self
    }

    /// Records Iceberg equality-delete field ids.
    pub fn with_equality_ids(mut self, equality_ids: impl IntoIterator<Item = i32>) -> Self {
        self.equality_ids = Some(equality_ids.into_iter().collect());
        self
    }

    /// Whether this delete file can be represented by native position-delete handling.
    pub fn is_supported_position_delete(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet
            && self.content == TableFormatDeleteFileContent::PositionDeletes
            && self.read_capability.is_native_parquet_capable()
            && self.resolved_parquet_source.is_some()
    }

    /// Whether this delete file can be represented by native equality-delete handling.
    pub fn is_supported_equality_delete(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet
            && self.content == TableFormatDeleteFileContent::EqualityDeletes
            && self.read_capability.is_native_parquet_capable()
            && self.resolved_parquet_source.is_some()
            && self.equality_ids.as_ref().is_some_and(|ids| !ids.is_empty())
    }

    /// Whether this delete file can be represented by native row-level delete handling.
    pub fn is_supported_row_delete(&self) -> bool {
        self.is_supported_position_delete() || self.is_supported_equality_delete()
    }
}
