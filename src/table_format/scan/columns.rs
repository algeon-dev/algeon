use super::*;

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
    /// Signed 128-bit integer metadata value.
    Int128(i128),
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
            Self::Int128(value) => write!(f, "{value}"),
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
    ///
    /// Identity transforms use the source column name. Non-identity transforms
    /// use the partition field name and are retained as metadata facts only.
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

    /// Column name that may be materialized natively from this partition value.
    pub(crate) fn native_materialized_column_name(&self) -> &str {
        self.column_name.as_str()
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
    /// Additional physical column names allowed by table-format field-id
    /// fallback metadata, such as Iceberg name mapping entries.
    pub field_id_source_names: Vec<String>,
    /// Arrow data type exposed at the native source boundary.
    pub data_type: DataType,
    /// Whether the logical column is nullable.
    pub nullable: bool,
    /// How the logical column is represented by native source facts.
    pub source: TableFormatColumnSource,
    /// Whether the column is projected or filter-only.
    pub role: TableFormatColumnRole,
    /// Literal value used when this logical column is absent from an older data file.
    pub missing_default_value: Option<TableFormatScalarValue>,
}

/// First unsupported scan column selected by native table-format admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableFormatUnsupportedColumn<'a> {
    /// Logical column name visible to SQL/DataFusion.
    pub(crate) logical_name: &'a str,
    /// Arrow type exposed by the table-format scan.
    pub(crate) data_type: &'a DataType,
    /// Column role in the native source boundary.
    pub(crate) role: TableFormatColumnRole,
}

impl TableFormatUnsupportedColumn<'_> {
    /// Stable role code for diagnostics.
    pub(crate) const fn role_code(&self) -> &'static str {
        self.role.code()
    }

    /// Human-readable Arrow type for client-facing diagnostics.
    pub(crate) fn data_type_summary(&self) -> String {
        nexus_query_engine::diagnostic_arrow_type(self.data_type)
    }
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
            field_id_source_names: Vec::new(),
            data_type,
            nullable,
            source,
            role,
            missing_default_value: None,
        }
    }

    /// Records a table-format default value for missing older-file columns.
    pub fn with_missing_default_value(mut self, value: TableFormatScalarValue) -> Self {
        self.missing_default_value = Some(value);
        self
    }

    /// Adds fallback physical names used only when binding by table-format
    /// field-id metadata and the file footer has no embedded field ids.
    pub fn with_field_id_source_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in names {
            let name = name.into();
            if !self.field_id_source_names.contains(&name) {
                self.field_id_source_names.push(name);
            }
        }
        self
    }

    pub(crate) fn field_id_physical_name_candidates(&self) -> impl Iterator<Item = &str> {
        let source_name = if self.field_id_source_names.is_empty() {
            Some(self.source_name.as_deref().unwrap_or(self.logical_name.as_str()))
        } else {
            None
        };
        source_name.into_iter().chain(self.field_id_source_names.iter().map(String::as_str))
    }
}
