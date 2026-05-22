#![deny(missing_docs)]

use std::{collections::BTreeMap, fmt, sync::Arc};

use arrow::datatypes::{DataType, Schema, SchemaRef};
use futures::future::BoxFuture;

use crate::{Result, policy::FallbackReason};

/// Table format that produced a resolved scan plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormat {
    /// Apache Iceberg table metadata.
    Iceberg,
    /// Delta Lake table metadata.
    Delta,
    /// Another table format identified by a stable caller-provided code.
    Other(String),
}

/// Format-neutral request passed to table-format metadata planners.
///
/// `None` projection means all logical columns. `None` snapshot/catalog values
/// leave selection to the table-format implementation. `None` scan limit means
/// no source-level row bound was supplied by DataFusion; `Some(n)` is an
/// order-insensitive upper bound the native source may apply before any
/// inexact parent filter. The filtering flags describe planner intent only;
/// unsupported row-group pruning or row selection must be reported as stable
/// unsupported features instead of executing data reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatScanRequest {
    /// Logical projected column names. `None` means all columns; `Some([])` means zero columns.
    pub projected_column_names: Option<Vec<String>>,
    /// Optional table snapshot id in the format's native textual form.
    pub snapshot_id: Option<String>,
    /// Optional stable catalog label, such as `glue`.
    pub catalog_type: Option<String>,
    /// Optional source-level row upper bound supplied by DataFusion.
    pub scan_limit: Option<usize>,
    /// Whether column resolution should be case-sensitive.
    pub case_sensitive: bool,
    /// Planner intent for row-group filtering. Unsupported semantics become scan facts.
    pub row_group_filtering_enabled: bool,
    /// Planner intent for row selection/page-index pruning. Unsupported semantics become scan facts.
    pub row_selection_enabled: bool,
}

impl TableFormatScanRequest {
    /// Request all logical columns using default scan-planning options.
    pub fn all_columns() -> Self {
        Self::default()
    }

    /// Sets the exact logical column names to project.
    pub fn with_projected_column_names(mut self, column_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.projected_column_names = Some(column_names.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the table snapshot id.
    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    /// Sets a stable catalog label.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Sets an order-insensitive source-level scan limit.
    pub fn with_scan_limit(mut self, scan_limit: Option<usize>) -> Self {
        self.scan_limit = scan_limit;
        self
    }

    /// Sets case sensitivity for table-format column resolution.
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Sets whether the format planner should request row-group filtering.
    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    /// Sets whether the format planner should request row selection/page-index pruning.
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
            scan_limit: None,
            case_sensitive: true,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
        }
    }
}

/// Adapter-owned table-format metadata planning boundary.
///
/// Implementations resolve metadata into scan facts and diagnostics; they do
/// not execute data reads. Diagnostics must not contain credentials or signed
/// URLs, and unsupported features should map to stable fallback reasons.
pub trait TableFormatPlanner {
    /// Resolves metadata into scan facts without reading table data.
    fn plan_scan<'a>(&'a self, request: TableFormatScanRequest) -> BoxFuture<'a, Result<TableFormatScanPlan>>;
}

impl TableFormat {
    /// Stable report/fallback code for this table format.
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

/// Read capability required for a table-format data file location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFormatReadCapability {
    /// File is readable through the local filesystem.
    LocalFileSystem,
    /// File requires an object-store reader for the given scheme.
    ObjectStore {
        /// Object-store scheme such as `s3`.
        scheme: String,
    },
    /// File location cannot be represented by native source reads.
    Unsupported {
        /// Stable unsupported reason detail.
        reason: String,
    },
}

impl TableFormatReadCapability {
    /// Local filesystem read capability.
    pub fn local_file_system() -> Self {
        Self::LocalFileSystem
    }

    /// Object-store read capability for a scheme such as `s3`.
    pub fn object_store(scheme: impl Into<String>) -> Self {
        Self::ObjectStore { scheme: scheme.into() }
    }

    /// Unsupported read capability with a stable reason detail.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported { reason: reason.into() }
    }

    /// Stable capability code for diagnostics.
    pub fn code(&self) -> &str {
        match self {
            Self::LocalFileSystem => "local_file_system",
            Self::ObjectStore { scheme } if scheme == "s3" => "object_store:s3",
            Self::ObjectStore { .. } => "object_store",
            Self::Unsupported { .. } => "unsupported",
        }
    }

    /// Human-readable capability detail.
    pub fn detail(&self) -> &str {
        match self {
            Self::LocalFileSystem => "local_file_system",
            Self::ObjectStore { scheme } => scheme.as_str(),
            Self::Unsupported { reason } => reason.as_str(),
        }
    }

    /// Whether this capability can be represented as a native local Parquet source.
    pub fn is_native_local_parquet_capable(&self) -> bool {
        matches!(self, Self::LocalFileSystem)
    }

    /// Whether this capability can be represented by direct native Parquet reads.
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
    /// Null metadata value.
    Null,
    /// Boolean metadata value.
    Boolean(bool),
    /// Signed 32-bit integer metadata value.
    Int32(i32),
    /// Signed 64-bit integer metadata value.
    Int64(i64),
    /// Unsigned 32-bit integer metadata value.
    UInt32(u32),
    /// Unsigned 64-bit integer metadata value.
    UInt64(u64),
    /// 64-bit floating-point metadata value.
    Float64(f64),
    /// UTF-8 metadata value.
    Utf8(String),
    /// Arrow Date32 day value.
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
    /// Identity partition transform.
    Identity,
    /// Bucket transform with bucket count.
    Bucket {
        /// Number of buckets.
        buckets: u32,
    },
    /// Truncate transform with width.
    Truncate {
        /// Truncation width.
        width: u32,
    },
    /// Year transform.
    Year,
    /// Month transform.
    Month,
    /// Day transform.
    Day,
    /// Hour transform.
    Hour,
    /// Unknown transform identified by a stable caller-provided code.
    Unknown(String),
}

impl TableFormatPartitionTransform {
    /// Stable transform code for diagnostics.
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

    /// Whether native lowering can materialize this partition value today.
    pub fn is_native_materializable(&self) -> bool {
        matches!(self, Self::Identity)
    }
}

/// Partition value attached to a planned data file.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatPartitionValue {
    /// Logical partition column name.
    pub column_name: String,
    /// Arrow data type of the partition value.
    pub data_type: DataType,
    /// Whether the logical partition column is nullable.
    pub nullable: bool,
    /// Table-format partition transform.
    pub transform: TableFormatPartitionTransform,
    /// Scalar partition value from metadata.
    pub value: TableFormatScalarValue,
}

impl TableFormatPartitionValue {
    /// Builds one partition value fact.
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
    /// Logical column maps directly to a top-level file column.
    DirectTopLevelFileColumn,
    /// Logical column is supplied from an identity partition value.
    IdentityPartitionValue,
    /// Nullable logical column is absent from the file and materialized as nulls.
    MissingNullableColumn,
    /// Logical column requires nested projection not represented by native source facts.
    UnsupportedNestedProjection,
}

impl TableFormatColumnSource {
    /// Stable source code for diagnostics.
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
    /// Column is projected into query output.
    Projected,
    /// Column is needed only for filtering.
    PureFilter,
}

impl TableFormatColumnRole {
    /// Stable role code for diagnostics.
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
    /// Stable table-format field id, when available.
    pub field_id: Option<i32>,
    /// Logical column name visible to query planning.
    pub logical_name: String,
    /// Physical source column name, when the value comes from a file column.
    pub source_name: Option<String>,
    /// Arrow data type exposed at the native source boundary.
    pub data_type: DataType,
    /// Whether the logical column is nullable.
    pub nullable: bool,
    /// How the logical column is represented by native source facts.
    pub source: TableFormatColumnSource,
    /// Whether the column is projected or filter-only.
    pub role: TableFormatColumnRole,
}

impl TableFormatColumnMapping {
    /// Builds one logical-to-source column mapping.
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
    /// Required read capability for this location.
    pub read_capability: TableFormatReadCapability,
    /// Partition values attached to this data file.
    pub partition_values: Vec<TableFormatPartitionValue>,
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
            read_capability,
            partition_values: Vec::new(),
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

    /// Records partition values attached to this file.
    pub fn with_partition_values(mut self, partition_values: Vec<TableFormatPartitionValue>) -> Self {
        self.partition_values = partition_values;
        self
    }

    /// Whether the file can be represented by direct native Parquet reads.
    pub fn is_direct_native_parquet_representable(&self) -> bool {
        self.format == TableFormatDataFileFormat::Parquet && self.read_capability.is_direct_native_parquet_capable()
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
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatDeleteFile {
    /// Stable delete-file id within the resolved scan plan.
    pub id: String,
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
}

impl TableFormatDeleteFile {
    /// Builds a delete-file fact without optional size/count metadata.
    pub fn new(
        id: impl Into<String>,
        location: impl Into<String>,
        format: TableFormatDataFileFormat,
        content: TableFormatDeleteFileContent,
    ) -> Self {
        Self { id: id.into(), location: location.into(), format, content, size_bytes: None, row_count: None }
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
}

/// Unsupported scan fact category with a stable fallback reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatUnsupportedFeatureCode {
    /// Generic unsupported table-format scan.
    TableFormatUnsupported,
    /// Table-format metadata planning failed.
    TableFormatPlanningFailed,
    /// Iceberg data file is not Parquet.
    IcebergNonParquetDataFile,
    /// Iceberg delete files are present.
    IcebergDeleteFilesUnsupported,
    /// Iceberg field-id mapping is required but unsupported.
    IcebergFieldIdMappingUnsupported,
    /// Iceberg name mapping is required but unsupported.
    IcebergNameMappingUnsupported,
    /// Iceberg nested projection is required but unsupported.
    IcebergNestedProjectionUnsupported,
    /// Iceberg partition transform cannot be materialized natively.
    IcebergPartitionTransformUnsupported,
    /// Iceberg residual predicate is not represented by native scan facts.
    IcebergResidualPredicateUnsupported,
    /// Iceberg row-selection/page-index semantics are unsupported.
    IcebergRowSelectionUnsupported,
    /// Iceberg object-store read cannot be represented by native source facts.
    IcebergObjectStoreReadUnsupported,
    /// Iceberg snapshot selection is unsupported or invalid.
    IcebergSnapshotUnsupported,
}

impl TableFormatUnsupportedFeatureCode {
    /// Stable fallback reason associated with this unsupported feature.
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

    /// Stable fallback reason code.
    pub const fn code(self) -> &'static str {
        self.fallback_reason().code()
    }
}

/// One unsupported table-format fact retained for reporting and rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatUnsupportedFeature {
    /// Unsupported feature category.
    pub code: TableFormatUnsupportedFeatureCode,
    /// Human-readable unsupported feature detail.
    pub message: String,
    /// Optional file id when the unsupported feature is file-local.
    pub file_id: Option<String>,
}

impl TableFormatUnsupportedFeature {
    /// Builds a scan-level unsupported feature.
    pub fn new(code: TableFormatUnsupportedFeatureCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), file_id: None }
    }

    /// Builds a file-local unsupported feature.
    pub fn for_file(
        code: TableFormatUnsupportedFeatureCode,
        file_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self { code, message: message.into(), file_id: Some(file_id.into()) }
    }

    /// Stable fallback reason associated with this feature.
    pub fn fallback_reason(&self) -> FallbackReason {
        self.code.fallback_reason()
    }

    /// Stable fallback reason code.
    pub fn code(&self) -> &'static str {
        self.code.code()
    }
}

/// Native direct-device-read disposition for a resolved table-format scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatDirectDeviceReadDisposition {
    /// Scan facts are eligible for direct native device reads.
    Eligible,
    /// Scan facts must be rejected before direct native device reads.
    Rejected,
}

impl TableFormatDirectDeviceReadDisposition {
    /// Stable disposition code for diagnostics.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::Rejected => "rejected",
        }
    }
}

/// Credential-free scan diagnostics intended for tests and report rows.
///
/// `planned_bytes = None` means the table metadata did not expose a reliable
/// byte total, not that the plan is empty. String fields are stable report
/// vocabulary and should avoid paths, tokens, or other sensitive values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatScanDiagnostics {
    /// Stable table-format code.
    pub table_format: String,
    /// Snapshot id selected by planning, if known.
    pub snapshot_id: Option<String>,
    /// Stable catalog label, if known.
    pub catalog_type: Option<String>,
    /// Number of planned data files.
    pub planned_data_file_count: usize,
    /// Number of planned delete files.
    pub planned_delete_file_count: usize,
    /// Total planned data-file bytes, or `None` if any file size is unknown.
    pub planned_bytes: Option<u64>,
    /// Number of files admitted to native source lowering.
    pub admitted_native_source_file_count: usize,
    /// Stable unsupported-feature codes retained by planning.
    pub rejected_feature_list: Vec<String>,
    /// Comma-separated stable read-capability codes.
    pub native_source_read_capability: String,
    /// Direct device-read eligibility.
    pub direct_device_read_disposition: TableFormatDirectDeviceReadDisposition,
}

impl TableFormatScanDiagnostics {
    /// Credential-free facts suitable for structured errors and tests.
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
///
/// The plan is metadata-only: data files, delete files, projected columns, and
/// pure-filter columns are facts for later source lowering. Unsupported
/// features remain attached so the adapter can produce structured fallback
/// reasons instead of silently dropping table-format semantics. `scan_limit`
/// is an optional source-level upper bound, not a final SQL `LIMIT`; when a
/// parent inexact filter remains, final row limiting is still represented by a
/// separate native or DataFusion limit operator.
#[derive(Debug, Clone, PartialEq)]
pub struct TableFormatScanPlan {
    /// Table format that produced these scan facts.
    pub format: TableFormat,
    /// Snapshot id selected by planning, if known.
    pub snapshot_id: Option<String>,
    /// Stable catalog label, if known.
    pub catalog_type: Option<String>,
    /// Optional source-level row upper bound supplied during table-format planning.
    pub scan_limit: Option<usize>,
    /// Logical table schema after projection.
    pub logical_schema: SchemaRef,
    /// Schema expected by native source lowering.
    pub native_source_schema: SchemaRef,
    /// Planned data files.
    pub data_files: Vec<TableFormatDataFile>,
    /// Planned delete files that native admission must currently reject.
    pub delete_files: Vec<TableFormatDeleteFile>,
    /// Output column mappings.
    pub projected_columns: Vec<TableFormatColumnMapping>,
    /// Filter-only column mappings.
    pub pure_filter_columns: Vec<TableFormatColumnMapping>,
    /// Unsupported features retained for structured fallback.
    pub unsupported_features: Vec<TableFormatUnsupportedFeature>,
}

impl TableFormatScanPlan {
    /// Creates an empty metadata-only scan plan for one table format.
    pub fn new(format: TableFormat) -> Self {
        let empty_schema = Arc::new(Schema::empty());
        Self {
            format,
            snapshot_id: None,
            catalog_type: None,
            scan_limit: None,
            logical_schema: Arc::clone(&empty_schema),
            native_source_schema: empty_schema,
            data_files: Vec::new(),
            delete_files: Vec::new(),
            projected_columns: Vec::new(),
            pure_filter_columns: Vec::new(),
            unsupported_features: Vec::new(),
        }
    }

    /// Records the selected table snapshot id when the format exposes one.
    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    /// Records a stable catalog label such as `glue` for diagnostics.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Records an optional source-level row upper bound.
    pub fn with_scan_limit(mut self, scan_limit: Option<usize>) -> Self {
        self.scan_limit = scan_limit;
        self
    }

    /// Records the logical table schema after table-format projection.
    pub fn with_logical_schema(mut self, schema: SchemaRef) -> Self {
        self.logical_schema = schema;
        self
    }

    /// Records the physical/native source schema expected by native lowering.
    pub fn with_native_source_schema(mut self, schema: SchemaRef) -> Self {
        self.native_source_schema = schema;
        self
    }

    /// Adds one planned data file fact.
    pub fn with_data_file(mut self, file: TableFormatDataFile) -> Self {
        self.data_files.push(file);
        self
    }

    /// Adds one delete-file fact that native admission must currently reject.
    pub fn with_delete_file(mut self, file: TableFormatDeleteFile) -> Self {
        self.delete_files.push(file);
        self
    }

    /// Adds one projected output column mapping.
    pub fn with_projected_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.projected_columns.push(column);
        self
    }

    /// Adds one column needed only for source filtering.
    pub fn with_pure_filter_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.pure_filter_columns.push(column);
        self
    }

    /// Adds one unsupported table-format feature retained for structured fallback.
    pub fn with_unsupported_feature(mut self, feature: TableFormatUnsupportedFeature) -> Self {
        self.unsupported_features.push(feature);
        self
    }

    /// Number of planned data files.
    pub fn planned_data_file_count(&self) -> usize {
        self.data_files.len()
    }

    /// Number of planned delete files.
    pub fn planned_delete_file_count(&self) -> usize {
        self.delete_files.len()
    }

    /// Total planned data-file bytes when every file size is known.
    ///
    /// Returns `None` if any data file lacks a byte size; `None` means
    /// "unknown total", not "zero bytes".
    pub fn planned_bytes(&self) -> Option<u64> {
        self.data_files
            .iter()
            .map(|file| file.size_bytes)
            .try_fold(0u64, |total, size| size.map(|size| total.saturating_add(size)))
    }

    /// All-or-nothing native-admission predicate for the current scan plan.
    ///
    /// A plan is admissible only when it has at least one direct native Parquet
    /// data file, no delete files, and no retained unsupported features.
    pub fn is_native_admissible(&self) -> bool {
        self.delete_files.is_empty()
            && self.unsupported_features.is_empty()
            && !self.data_files.is_empty()
            && self.data_files.iter().all(TableFormatDataFile::is_direct_native_parquet_representable)
    }

    /// Number of source files native lowering may consume.
    ///
    /// This returns zero unless the whole scan is native-admissible.
    pub fn admitted_native_source_file_count(&self) -> usize {
        if self.is_native_admissible() { self.data_files.len() } else { 0 }
    }

    /// Stable fallback reasons for retained unsupported features.
    pub fn unsupported_fallback_reasons(&self) -> Vec<FallbackReason> {
        self.unsupported_features.iter().map(TableFormatUnsupportedFeature::fallback_reason).collect()
    }

    /// Stable unsupported-feature codes for diagnostics/report rows.
    pub fn rejected_feature_codes(&self) -> Vec<String> {
        self.unsupported_features.iter().map(|feature| feature.code().to_string()).collect()
    }

    /// Direct device-read disposition derived from native admissibility.
    pub fn direct_device_read_disposition(&self) -> TableFormatDirectDeviceReadDisposition {
        if self.is_native_admissible() {
            TableFormatDirectDeviceReadDisposition::Eligible
        } else {
            TableFormatDirectDeviceReadDisposition::Rejected
        }
    }

    /// Comma-separated stable read-capability codes across data files.
    pub fn native_source_read_capability_summary(&self) -> String {
        let mut capabilities =
            self.data_files.iter().map(|file| file.read_capability.code().to_string()).collect::<Vec<_>>();
        capabilities.sort();
        capabilities.dedup();
        capabilities.join(",")
    }

    /// Builds credential-free scan diagnostics for tests and report rows.
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
