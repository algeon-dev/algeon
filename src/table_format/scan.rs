use std::{collections::BTreeMap, fmt, sync::Arc};

use arrow::datatypes::{DataType, Schema, SchemaRef};
use futures::future::BoxFuture;

use crate::{Result, policy::FallbackReason};

/// Table format that produced a resolved scan plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormat {
    Iceberg,
    Delta,
    Other(String),
}

/// Format-neutral request passed to table-format metadata planners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatScanRequest {
    pub projected_column_names: Option<Vec<String>>,
    pub snapshot_id: Option<String>,
    pub catalog_type: Option<String>,
    pub case_sensitive: bool,
    pub row_group_filtering_enabled: bool,
    pub row_selection_enabled: bool,
}

impl TableFormatScanRequest {
    pub fn all_columns() -> Self {
        Self::default()
    }

    pub fn with_projected_column_names(mut self, column_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.projected_column_names = Some(column_names.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    pub fn with_row_selection_enabled(mut self, row_selection_enabled: bool) -> Self {
        self.row_selection_enabled = row_selection_enabled;
        self
    }
}

impl Default for TableFormatScanRequest {
    fn default() -> Self {
        Self {
            projected_column_names: None,
            snapshot_id: None,
            catalog_type: None,
            case_sensitive: true,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
        }
    }
}

/// Adapter-owned metadata planner boundary. Implementations resolve table-format
/// metadata into scan facts; they do not execute data reads.
pub trait TableFormatPlanner {
    fn plan_scan<'a>(&'a self, request: TableFormatScanRequest) -> BoxFuture<'a, Result<TableFormatScanPlan>>;
}

impl TableFormat {
    pub fn code(&self) -> &str {
        match self {
            Self::Iceberg => "iceberg",
            Self::Delta => "delta",
            Self::Other(format) => format.as_str(),
        }
    }
}

/// File format advertised by table-format metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatDataFileFormat {
    Parquet,
    Orc,
    Avro,
    Unknown(String),
}

impl TableFormatDataFileFormat {
    pub fn code(&self) -> &str {
        match self {
            Self::Parquet => "parquet",
            Self::Orc => "orc",
            Self::Avro => "avro",
            Self::Unknown(format) => format.as_str(),
        }
    }
}

/// Read capability required for a table-format data file location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatReadCapability {
    LocalFileSystem,
    ObjectStore { scheme: String },
    Unsupported { reason: String },
}

impl TableFormatReadCapability {
    pub fn local_file_system() -> Self {
        Self::LocalFileSystem
    }

    pub fn object_store(scheme: impl Into<String>) -> Self {
        Self::ObjectStore { scheme: scheme.into() }
    }

    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported { reason: reason.into() }
    }

    pub fn code(&self) -> &str {
        match self {
            Self::LocalFileSystem => "local_file_system",
            Self::ObjectStore { scheme } if scheme == "s3" => "object_store:s3",
            Self::ObjectStore { .. } => "object_store",
            Self::Unsupported { .. } => "unsupported",
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            Self::LocalFileSystem => "local_file_system",
            Self::ObjectStore { scheme } => scheme.as_str(),
            Self::Unsupported { reason } => reason.as_str(),
        }
    }

    pub fn is_native_local_parquet_capable(&self) -> bool {
        matches!(self, Self::LocalFileSystem)
    }

    pub fn is_direct_native_parquet_capable(&self) -> bool {
        match self {
            Self::LocalFileSystem => true,
            Self::ObjectStore { scheme } => scheme.eq_ignore_ascii_case("s3"),
            Self::Unsupported { .. } => false,
        }
    }
}

/// Scalar value available from table-format metadata, usually partition facts.
#[derive(Debug, Clone, PartialEq)]
pub enum TableFormatScalarValue {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Float64(f64),
    Utf8(String),
    Date32(i32),
}

impl fmt::Display for TableFormatScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("null"),
            Self::Boolean(value) => write!(f, "{value}"),
            Self::Int32(value) => write!(f, "{value}"),
            Self::Int64(value) => write!(f, "{value}"),
            Self::UInt32(value) => write!(f, "{value}"),
            Self::UInt64(value) => write!(f, "{value}"),
            Self::Float64(value) => write!(f, "{value}"),
            Self::Utf8(value) => write!(f, "{value:?}"),
            Self::Date32(value) => write!(f, "{value}"),
        }
    }
}

/// Partition transform represented by table metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatPartitionTransform {
    Identity,
    Bucket { buckets: u32 },
    Truncate { width: u32 },
    Year,
    Month,
    Day,
    Hour,
    Unknown(String),
}

impl TableFormatPartitionTransform {
    pub fn code(&self) -> &str {
        match self {
            Self::Identity => "identity",
            Self::Bucket { .. } => "bucket",
            Self::Truncate { .. } => "truncate",
            Self::Year => "year",
            Self::Month => "month",
            Self::Day => "day",
            Self::Hour => "hour",
            Self::Unknown(transform) => transform.as_str(),
        }
    }

    pub fn is_native_materializable(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

/// Partition value attached to a planned data file.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatPartitionValue {
    pub column_name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub transform: TableFormatPartitionTransform,
    pub value: TableFormatScalarValue,
}

impl TableFormatPartitionValue {
    pub fn new(
        column_name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
        transform: TableFormatPartitionTransform,
        value: TableFormatScalarValue,
    ) -> Self {
        Self { column_name: column_name.into(), data_type, nullable, transform, value }
    }
}

/// How a logical table column is represented by the native source boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatColumnSource {
    DirectTopLevelFileColumn,
    IdentityPartitionValue,
    MissingNullableColumn,
    UnsupportedNestedProjection,
}

impl TableFormatColumnSource {
    pub const fn code(self) -> &'static str {
        match self {
            Self::DirectTopLevelFileColumn => "direct_top_level_file_column",
            Self::IdentityPartitionValue => "identity_partition_value",
            Self::MissingNullableColumn => "missing_nullable_column",
            Self::UnsupportedNestedProjection => "unsupported_nested_projection",
        }
    }
}

/// Logical role of a column in the resolved scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatColumnRole {
    Projected,
    PureFilter,
}

impl TableFormatColumnRole {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Projected => "projected",
            Self::PureFilter => "pure_filter",
        }
    }
}

/// Mapping from table-format field facts to a native source column.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatColumnMapping {
    pub field_id: Option<i32>,
    pub logical_name: String,
    pub source_name: Option<String>,
    pub data_type: DataType,
    pub nullable: bool,
    pub source: TableFormatColumnSource,
    pub role: TableFormatColumnRole,
}

impl TableFormatColumnMapping {
    pub fn new(
        field_id: Option<i32>,
        logical_name: impl Into<String>,
        source_name: Option<impl Into<String>>,
        data_type: DataType,
        nullable: bool,
        source: TableFormatColumnSource,
        role: TableFormatColumnRole,
    ) -> Self {
        Self {
            field_id,
            logical_name: logical_name.into(),
            source_name: source_name.map(Into::into),
            data_type,
            nullable,
            source,
            role,
        }
    }
}

/// Resolved data file from a table-format scan task.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatDataFile {
    pub id: String,
    pub location: String,
    pub format: TableFormatDataFileFormat,
    pub size_bytes: Option<u64>,
    pub row_count: Option<u64>,
    pub row_groups: Option<Vec<usize>>,
    pub read_capability: TableFormatReadCapability,
    pub partition_values: Vec<TableFormatPartitionValue>,
}

impl TableFormatDataFile {
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
            read_capability,
            partition_values: Vec::new(),
        }
    }

    pub fn with_size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.row_groups = Some(row_groups);
        self
    }

    pub fn with_partition_values(mut self, partition_values: Vec<TableFormatPartitionValue>) -> Self {
        self.partition_values = partition_values;
        self
    }

    pub fn is_direct_native_parquet_representable(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet && self.read_capability.is_direct_native_parquet_capable()
    }

    pub fn is_native_local_parquet_representable(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet && self.read_capability.is_native_local_parquet_capable()
    }
}

/// Row-level delete semantics advertised by a table-format delete file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatDeleteFileContent {
    PositionDeletes,
    EqualityDeletes,
    DeleteVector,
    Unknown(String),
}

impl TableFormatDeleteFileContent {
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
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatDeleteFile {
    pub id: String,
    pub location: String,
    pub format: TableFormatDataFileFormat,
    pub content: TableFormatDeleteFileContent,
    pub size_bytes: Option<u64>,
    pub row_count: Option<u64>,
}

impl TableFormatDeleteFile {
    pub fn new(
        id: impl Into<String>,
        location: impl Into<String>,
        format: TableFormatDataFileFormat,
        content: TableFormatDeleteFileContent,
    ) -> Self {
        Self { id: id.into(), location: location.into(), format, content, size_bytes: None, row_count: None }
    }

    pub fn with_size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }
}

/// Unsupported scan fact category with a stable fallback reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatUnsupportedFeatureCode {
    TableFormatUnsupported,
    TableFormatPlanningFailed,
    IcebergNonParquetDataFile,
    IcebergDeleteFilesUnsupported,
    IcebergFieldIdMappingUnsupported,
    IcebergNameMappingUnsupported,
    IcebergNestedProjectionUnsupported,
    IcebergPartitionTransformUnsupported,
    IcebergResidualPredicateUnsupported,
    IcebergRowSelectionUnsupported,
    IcebergObjectStoreReadUnsupported,
    IcebergSnapshotUnsupported,
}

impl TableFormatUnsupportedFeatureCode {
    pub const fn fallback_reason(self) -> FallbackReason {
        match self {
            Self::TableFormatUnsupported => FallbackReason::TableFormatUnsupported,
            Self::TableFormatPlanningFailed => FallbackReason::TableFormatPlanningFailed,
            Self::IcebergNonParquetDataFile => FallbackReason::IcebergNonParquetDataFile,
            Self::IcebergDeleteFilesUnsupported => FallbackReason::IcebergDeleteFilesUnsupported,
            Self::IcebergFieldIdMappingUnsupported => FallbackReason::IcebergFieldIdMappingUnsupported,
            Self::IcebergNameMappingUnsupported => FallbackReason::IcebergNameMappingUnsupported,
            Self::IcebergNestedProjectionUnsupported => FallbackReason::IcebergNestedProjectionUnsupported,
            Self::IcebergPartitionTransformUnsupported => FallbackReason::IcebergPartitionTransformUnsupported,
            Self::IcebergResidualPredicateUnsupported => FallbackReason::IcebergResidualPredicateUnsupported,
            Self::IcebergRowSelectionUnsupported => FallbackReason::IcebergRowSelectionUnsupported,
            Self::IcebergObjectStoreReadUnsupported => FallbackReason::IcebergObjectStoreReadUnsupported,
            Self::IcebergSnapshotUnsupported => FallbackReason::IcebergSnapshotUnsupported,
        }
    }

    pub const fn code(self) -> &'static str {
        self.fallback_reason().code()
    }
}

/// One unsupported table-format fact retained for reporting and rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatUnsupportedFeature {
    pub code: TableFormatUnsupportedFeatureCode,
    pub message: String,
    pub file_id: Option<String>,
}

impl TableFormatUnsupportedFeature {
    pub fn new(code: TableFormatUnsupportedFeatureCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), file_id: None }
    }

    pub fn for_file(
        code: TableFormatUnsupportedFeatureCode,
        file_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self { code, message: message.into(), file_id: Some(file_id.into()) }
    }

    pub fn fallback_reason(&self) -> FallbackReason {
        self.code.fallback_reason()
    }

    pub fn code(&self) -> &'static str {
        self.code.code()
    }
}

/// Native direct-device-read disposition for a resolved table-format scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatDirectDeviceReadDisposition {
    Eligible,
    Rejected,
}

impl TableFormatDirectDeviceReadDisposition {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::Rejected => "rejected",
        }
    }
}

/// Credential-free scan diagnostics intended for tests and report rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatScanDiagnostics {
    pub table_format: String,
    pub snapshot_id: Option<String>,
    pub catalog_type: Option<String>,
    pub planned_data_file_count: usize,
    pub planned_delete_file_count: usize,
    pub planned_bytes: Option<u64>,
    pub admitted_native_source_file_count: usize,
    pub rejected_feature_list: Vec<String>,
    pub native_source_read_capability: String,
    pub direct_device_read_disposition: TableFormatDirectDeviceReadDisposition,
}

impl TableFormatScanDiagnostics {
    pub fn facts(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("table_format", self.table_format.clone()),
            ("snapshot_id", self.snapshot_id.clone().unwrap_or_default()),
            ("catalog_type", self.catalog_type.clone().unwrap_or_default()),
            ("planned_data_file_count", self.planned_data_file_count.to_string()),
            ("planned_delete_file_count", self.planned_delete_file_count.to_string()),
            ("planned_bytes", self.planned_bytes.map(|bytes| bytes.to_string()).unwrap_or_default()),
            ("admitted_native_source_file_count", self.admitted_native_source_file_count.to_string()),
            ("rejected_feature_list", self.rejected_feature_list.join(",")),
            ("native_source_read_capability", self.native_source_read_capability.clone()),
            ("direct_device_read_disposition", self.direct_device_read_disposition.code().to_string()),
        ])
    }
}

impl fmt::Display for TableFormatScanDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, (key, value)) in self.facts().iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{key}={value}")?;
        }
        Ok(())
    }
}

/// Resolved table-format scan facts consumed by later native lowering phases.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatScanPlan {
    pub format: TableFormat,
    pub snapshot_id: Option<String>,
    pub catalog_type: Option<String>,
    pub logical_schema: SchemaRef,
    pub native_source_schema: SchemaRef,
    pub data_files: Vec<TableFormatDataFile>,
    pub delete_files: Vec<TableFormatDeleteFile>,
    pub projected_columns: Vec<TableFormatColumnMapping>,
    pub pure_filter_columns: Vec<TableFormatColumnMapping>,
    pub unsupported_features: Vec<TableFormatUnsupportedFeature>,
}

impl TableFormatScanPlan {
    pub fn new(format: TableFormat) -> Self {
        let empty_schema = Arc::new(Schema::empty());
        Self {
            format,
            snapshot_id: None,
            catalog_type: None,
            logical_schema: Arc::clone(&empty_schema),
            native_source_schema: empty_schema,
            data_files: Vec::new(),
            delete_files: Vec::new(),
            projected_columns: Vec::new(),
            pure_filter_columns: Vec::new(),
            unsupported_features: Vec::new(),
        }
    }

    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    pub fn with_logical_schema(mut self, schema: SchemaRef) -> Self {
        self.logical_schema = schema;
        self
    }

    pub fn with_native_source_schema(mut self, schema: SchemaRef) -> Self {
        self.native_source_schema = schema;
        self
    }

    pub fn with_data_file(mut self, file: TableFormatDataFile) -> Self {
        self.data_files.push(file);
        self
    }

    pub fn with_delete_file(mut self, file: TableFormatDeleteFile) -> Self {
        self.delete_files.push(file);
        self
    }

    pub fn with_projected_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.projected_columns.push(column);
        self
    }

    pub fn with_pure_filter_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.pure_filter_columns.push(column);
        self
    }

    pub fn with_unsupported_feature(mut self, feature: TableFormatUnsupportedFeature) -> Self {
        self.unsupported_features.push(feature);
        self
    }

    pub fn planned_data_file_count(&self) -> usize {
        self.data_files.len()
    }

    pub fn planned_delete_file_count(&self) -> usize {
        self.delete_files.len()
    }

    pub fn planned_bytes(&self) -> Option<u64> {
        self.data_files
            .iter()
            .map(|file| file.size_bytes)
            .try_fold(0u64, |total, size| size.map(|size| total.saturating_add(size)))
    }

    pub fn is_native_admissible(&self) -> bool {
        self.delete_files.is_empty()
            && self.unsupported_features.is_empty()
            && !self.data_files.is_empty()
            && self.data_files.iter().all(TableFormatDataFile::is_direct_native_parquet_representable)
    }

    pub fn admitted_native_source_file_count(&self) -> usize {
        if self.is_native_admissible() { self.data_files.len() } else { 0 }
    }

    pub fn unsupported_fallback_reasons(&self) -> Vec<FallbackReason> {
        self.unsupported_features.iter().map(TableFormatUnsupportedFeature::fallback_reason).collect()
    }

    pub fn rejected_feature_codes(&self) -> Vec<String> {
        self.unsupported_features.iter().map(|feature| feature.code().to_string()).collect()
    }

    pub fn direct_device_read_disposition(&self) -> TableFormatDirectDeviceReadDisposition {
        if self.is_native_admissible() {
            TableFormatDirectDeviceReadDisposition::Eligible
        } else {
            TableFormatDirectDeviceReadDisposition::Rejected
        }
    }

    pub fn native_source_read_capability_summary(&self) -> String {
        let mut capabilities =
            self.data_files.iter().map(|file| file.read_capability.code().to_string()).collect::<Vec<_>>();
        capabilities.sort();
        capabilities.dedup();
        capabilities.join(",")
    }

    pub fn diagnostics(&self) -> TableFormatScanDiagnostics {
        TableFormatScanDiagnostics {
            table_format: self.format.code().to_string(),
            snapshot_id: self.snapshot_id.clone(),
            catalog_type: self.catalog_type.clone(),
            planned_data_file_count: self.planned_data_file_count(),
            planned_delete_file_count: self.planned_delete_file_count(),
            planned_bytes: self.planned_bytes(),
            admitted_native_source_file_count: self.admitted_native_source_file_count(),
            rejected_feature_list: self.rejected_feature_codes(),
            native_source_read_capability: self.native_source_read_capability_summary(),
            direct_device_read_disposition: self.direct_device_read_disposition(),
        }
    }
}
