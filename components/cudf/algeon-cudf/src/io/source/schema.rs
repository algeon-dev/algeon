use super::{
    ArrowDataType, DataType, Error, ErrorCode, FACT_TYPE_ID, OP_COLUMN_NEW, OP_OUTPUT_NEW, OP_REQUIRED_NEW,
    OP_SCHEMA_NEW, Result, TypeId, checked_name, validate_unique_named,
};
use crate::interop::arrow_host::cudf_dtype_from_arrow_type;

/// Source-level schema used for validation and typed empty outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSchema {
    pub(super) columns: Vec<SourceColumn>,
}

impl SourceSchema {
    /// Create a schema with unique, non-empty column names.
    pub fn new<I>(columns: I) -> Result<Self>
    where I: IntoIterator<Item = SourceColumn> {
        let columns: Vec<_> = columns.into_iter().collect();
        if columns.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, OP_SCHEMA_NEW, "source schema must not be empty")
                .with_detail("reason", "empty_schema"));
        }
        validate_unique_named(
            columns.iter().map(|column| column.name.as_str()),
            OP_SCHEMA_NEW,
            "duplicate_schema_column",
        )?;
        Ok(Self { columns })
    }

    /// Schema columns in caller-provided order.
    pub fn columns(&self) -> &[SourceColumn] {
        &self.columns
    }

    /// Find a schema column by exact name.
    pub fn column(&self, name: &str) -> Option<&SourceColumn> {
        self.columns.iter().find(|column| column.name == name)
    }
}

/// One source-visible column and its logical nullability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceColumn {
    pub(super) name: String,
    pub(super) data_type: DataType,
    pub(super) arrow_data_type: Option<ArrowDataType>,
    pub(super) nullable: bool,
}

impl SourceColumn {
    /// Create a source schema column.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Result<Self> {
        let name = checked_name(name.into(), OP_COLUMN_NEW, "source column")?;
        Ok(Self { name, data_type, arrow_data_type: None, nullable })
    }

    /// Create a list source column with its complete Arrow child schema.
    pub fn new_list(name: impl Into<String>, arrow_data_type: ArrowDataType, nullable: bool) -> Result<Self> {
        let name = checked_name(name.into(), OP_COLUMN_NEW, "source column")?;
        let data_type = cudf_dtype_from_arrow_type(&arrow_data_type)
            .map_err(|error| error.reanchor(OP_COLUMN_NEW).with_column(name.clone()))?;
        if data_type.id() != TypeId::List {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                OP_COLUMN_NEW,
                "list source column Arrow type must map to cudf LIST",
            )
            .with_column(name)
            .with_fact(FACT_TYPE_ID, data_type.id())
            .with_detail("arrow_type", format!("{arrow_data_type:?}"))
            .with_detail("reason", "source_list_arrow_type_mismatch"));
        }
        Ok(Self { name, data_type, arrow_data_type: Some(arrow_data_type), nullable })
    }

    /// Column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Logical cuDF data type.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Complete Arrow list type retained for typed source materialization.
    pub fn arrow_data_type(&self) -> Option<&ArrowDataType> {
        self.arrow_data_type.as_ref()
    }

    /// Source-level nullability truth.
    pub fn nullable(&self) -> bool {
        self.nullable
    }
}

/// A final source output column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceOutputColumn {
    pub(super) name: String,
    pub(super) source: SourceColumnRef,
}

impl SourceOutputColumn {
    /// Create an output column mapping.
    pub fn new(name: impl Into<String>, source: SourceColumnRef) -> Result<Self> {
        let name = checked_name(name.into(), OP_OUTPUT_NEW, "source output column")?;
        Ok(Self { name, source })
    }

    /// Final output column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Source location used to materialize the output column.
    pub fn source(&self) -> SourceColumnRef {
        self.source
    }
}

/// A column required by source-local execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRequiredColumn {
    pub(super) name: String,
    pub(super) source: SourceColumnRef,
    pub(super) role: SourceColumnRole,
}

impl SourceRequiredColumn {
    /// Create a required execution-frame column mapping.
    pub fn new(name: impl Into<String>, source: SourceColumnRef, role: SourceColumnRole) -> Result<Self> {
        let name = checked_name(name.into(), OP_REQUIRED_NEW, "source required column")?;
        Ok(Self { name, source, role })
    }

    /// Required column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Source location used to materialize the required column.
    pub fn source(&self) -> SourceColumnRef {
        self.source
    }

    /// Why this column is required.
    pub fn role(&self) -> SourceColumnRole {
        self.role
    }
}

/// Role for a required source column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceColumnRole {
    /// The column participates in final output.
    Output,
    /// The column is needed only for source-local filter/projection work.
    PureFilter,
}

/// A reference to a decoded read column, partition column, synthetic count, or
/// nullable missing source column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceColumnRef {
    /// Decoded read projection ordinal.
    ///
    /// This index points into [`LocalParquetSource::read_columns`]. It does not
    /// refer to the original Parquet file schema. Callers that start from
    /// file-schema ordinals must translate them into decoded read-projection
    /// ordinals.
    ReadColumn(usize),
    /// Partition-column ordinal in [`LocalParquetSource::partition_columns`].
    Partition(usize),
    /// Synthetic metadata row-count column. Valid only for
    /// [`LocalParquetSourceKind::MetadataRowCount`].
    ZeroColumnCount,
    /// Nullable logical column absent from the physical file schema. Valid only
    /// for data sources whose target schema column is nullable and whose dtype
    /// can be materialized as a typed null column.
    MissingNull,
}
