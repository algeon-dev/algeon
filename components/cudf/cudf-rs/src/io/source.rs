//! Local Parquet source-planning contracts.
//!
//! This module describes checked, schema-aware local Parquet sources that can
//! be lowered from query planners before any GPU execution happens. The types
//! here are intentionally concrete: they describe local Parquet files,
//! projected decoded read columns, explicit partition constants, final output
//! columns, required execution-frame columns, and derived diagnostics.
//!
//! `LocalParquetSource` is not a generic file-source trait and is not an
//! Iceberg table-scan contract. Richer planners should lower into this type
//! only when the resulting scan is truthfully representable as local Parquet
//! files with explicit read-column mapping, nullable missing-column
//! materialization, and explicit metadata.
//!
//! Future source capabilities should be additive and explicitly named.
//! Predicate pruning, object stores, byte ranges, row-level selection, Iceberg
//! delete files, Iceberg field-id or name mapping, partition transforms, and
//! GDS/KvikIO direct-read policy all need their own source facts before callers
//! may rely on them. Local sources therefore reject or keep those planner facts
//! downstream instead of silently treating them as ordinary local full-file
//! reads.

use std::{
    fmt,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use parquet::{
    basic::{ConvertedType, LogicalType, Type as ParquetPhysicalType},
    file::reader::{FileReader, SerializedFileReader},
    schema::types::ColumnDescriptor,
};

use crate::{
    core::{
        error::{Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_COLUMN_INDEX, FACT_EXPECTED_COUNT, FACT_TYPE_ID, Result},
        types::{DataType, TimestampResolution, TypeId},
    },
    internal::hash::{FastHashSet, fast_hash_set_with_capacity},
    ops::ast::{AstExpression, AstOperator},
    runtime::Runtime,
};

/// Synthetic column name used by metadata-only zero-column count sources.
pub const ZERO_COLUMN_COUNT_COLUMN: &str = "__cudf_zero_column_row_count";

const OP_FILE_NEW: &str = "io::source::LocalParquetFile::new";
const OP_FILE_METADATA: &str = "io::source::LocalParquetFile::with_row_group_metadata";
const OP_SOURCE_DATA: &str = "io::source::LocalParquetSource::data";
const OP_SOURCE_COUNT: &str = "io::source::LocalParquetSource::zero_column_count";
const OP_SCHEMA_NEW: &str = "io::source::SourceSchema::new";
const OP_COLUMN_NEW: &str = "io::source::SourceColumn::new";
const OP_OUTPUT_NEW: &str = "io::source::SourceOutputColumn::new";
const OP_REQUIRED_NEW: &str = "io::source::SourceRequiredColumn::new";
const OP_PARTITION_NEW: &str = "io::source::PartitionColumn::new";
const OP_CHUNK_POLICY_NEW: &str = "io::source::SourceChunkPlanningPolicy::new";
const OP_PLAN_CHUNKS: &str = "io::source::plan_local_parquet_source_chunks";
const OP_ROW_GROUP_TO_I32: &str = "io::source::source_row_group_index_to_i32";
const OP_ROW_COUNT_TO_I64: &str = "io::source::source_row_count_to_i64";
const OP_SOURCE_READER_PREDICATE: &str = "io::source::build_local_parquet_reader_predicate";

/// Explicit Parquet source location represented by the checked source
/// contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParquetSourceLocation {
    LocalPath(PathBuf),
    ObjectStoreUri(String),
}

impl ParquetSourceLocation {
    pub fn local_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        validate_local_path(&path)?;
        Ok(Self::LocalPath(path))
    }

    pub fn object_store_uri(uri: impl Into<String>) -> Result<Self> {
        let uri = uri.into();
        validate_object_store_uri(&uri)?;
        Ok(Self::ObjectStoreUri(uri))
    }

    pub fn reader_path(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::ObjectStoreUri(uri) => uri.clone(),
        }
    }

    pub fn read_capability(&self) -> &'static str {
        match self {
            Self::LocalPath(_) => "local_file_system",
            Self::ObjectStoreUri(uri) if uri_scheme(uri).is_some_and(|scheme| scheme.eq_ignore_ascii_case("s3")) => {
                "object_store:s3"
            },
            Self::ObjectStoreUri(_) => "object_store:unsupported",
        }
    }

    pub fn display(&self) -> impl fmt::Display + '_ {
        self
    }
}

impl fmt::Display for ParquetSourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalPath(path) => write!(f, "{}", path.display()),
            Self::ObjectStoreUri(uri) => f.write_str(uri),
        }
    }
}

/// A complete checked local Parquet source contract.
///
/// Fields are private so callers cannot mutate the plan into an inconsistent
/// shape after construction. Use [`LocalParquetSource::data`] for normal data
/// reads and [`LocalParquetSource::zero_column_count`] for metadata-only count
/// sources.
#[derive(Debug, Clone)]
pub struct LocalParquetSource {
    kind: LocalParquetSourceKind,
    files: Vec<LocalParquetFile>,
    schema: SourceSchema,
    read_columns: Vec<String>,
    output_columns: Vec<SourceOutputColumn>,
    required_columns: Vec<SourceRequiredColumn>,
    partition_columns: Vec<PartitionColumn>,
    reader_predicate: Option<Arc<AstExpression>>,
}

impl PartialEq for LocalParquetSource {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.files == other.files
            && self.schema == other.schema
            && self.read_columns == other.read_columns
            && self.output_columns == other.output_columns
            && self.required_columns == other.required_columns
            && self.partition_columns == other.partition_columns
            && self.reader_predicate.is_some() == other.reader_predicate.is_some()
    }
}

impl LocalParquetSource {
    /// Build a normal local Parquet data source.
    ///
    /// `SourceColumnRef::ReadColumn(index)` values point into `read_columns`
    /// after projection, not into the original file schema. Planner adapters
    /// that start from file-schema ordinals must translate them before calling
    /// this constructor.
    pub fn data<I, R, O, Q, P>(
        files: I,
        schema: SourceSchema,
        read_columns: R,
        output_columns: O,
        required_columns: Q,
        partition_columns: P,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = LocalParquetFile>,
        R: IntoIterator,
        R::Item: Into<String>,
        O: IntoIterator<Item = SourceOutputColumn>,
        Q: IntoIterator<Item = SourceRequiredColumn>,
        P: IntoIterator<Item = PartitionColumn>,
    {
        let source = Self {
            kind: LocalParquetSourceKind::Data,
            files: files.into_iter().collect(),
            schema,
            read_columns: read_columns.into_iter().map(Into::into).collect(),
            output_columns: output_columns.into_iter().collect(),
            required_columns: required_columns.into_iter().collect(),
            partition_columns: partition_columns.into_iter().collect(),
            reader_predicate: None,
        };
        source.validate(OP_SOURCE_DATA)?;
        Ok(source)
    }

    /// Build a metadata-only count source using the synthetic count column.
    ///
    /// This constructor does not infer count mode from an empty data
    /// projection. Count sources always have no read columns, no partition
    /// columns, and exactly one non-null `Int64` output named
    /// [`ZERO_COLUMN_COUNT_COLUMN`].
    pub fn zero_column_count<I>(files: I) -> Result<Self>
    where I: IntoIterator<Item = LocalParquetFile> {
        let files: Vec<_> = files.into_iter().collect();
        for (file_index, file) in files.iter().enumerate() {
            if !file.partition_values.is_empty() {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedOperation,
                    OP_SOURCE_COUNT,
                    "metadata row-count sources cannot carry partition values",
                )
                .with_detail("reason", "metadata_count_partition_values")
                .with_detail("file_index", file_index.to_string())
                .with_detail("stable_id", file.stable_id.clone())
                .with_detail("path", file.path.display().to_string()));
            }
        }

        let schema =
            SourceSchema::new([SourceColumn::new(ZERO_COLUMN_COUNT_COLUMN, DataType::new(TypeId::Int64), false)?])?;
        let source = Self {
            kind: LocalParquetSourceKind::MetadataRowCount,
            files,
            schema,
            read_columns: Vec::new(),
            output_columns: vec![SourceOutputColumn::new(ZERO_COLUMN_COUNT_COLUMN, SourceColumnRef::ZeroColumnCount)?],
            required_columns: Vec::new(),
            partition_columns: Vec::new(),
            reader_predicate: None,
        };
        source.validate(OP_SOURCE_COUNT)?;
        Ok(source)
    }

    /// Source kind.
    pub fn kind(&self) -> LocalParquetSourceKind {
        self.kind
    }

    /// Local Parquet files included in this source.
    pub fn files(&self) -> &[LocalParquetFile] {
        &self.files
    }

    /// Source schema used for typed empty outputs and validation.
    pub fn schema(&self) -> &SourceSchema {
        &self.schema
    }

    /// Decoded Parquet read projection by name.
    pub fn read_columns(&self) -> &[String] {
        &self.read_columns
    }

    /// Final source output columns in output order.
    pub fn output_columns(&self) -> &[SourceOutputColumn] {
        &self.output_columns
    }

    /// Columns required by source-local execution.
    pub fn required_columns(&self) -> &[SourceRequiredColumn] {
        &self.required_columns
    }

    /// Explicit caller-provided partition columns.
    pub fn partition_columns(&self) -> &[PartitionColumn] {
        &self.partition_columns
    }

    /// Optional reader-level Parquet predicate.
    pub fn reader_predicate(&self) -> Option<&Arc<AstExpression>> {
        self.reader_predicate.as_ref()
    }

    /// Attach a reader-level Parquet predicate to this data source.
    pub fn with_reader_predicate(mut self, predicate: Arc<AstExpression>) -> Result<Self> {
        if self.kind != LocalParquetSourceKind::Data {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                OP_SOURCE_DATA,
                "reader predicates are only valid for local Parquet data sources",
            )
            .with_detail("reason", "reader_predicate_on_non_data_source"));
        }
        self.reader_predicate = Some(predicate);
        self.validate(OP_SOURCE_DATA)?;
        Ok(self)
    }

    /// Compute diagnostics from checked source facts.
    ///
    /// Diagnostics are derived on demand and are not stored as mutable caller
    /// state in the source.
    pub fn diagnostics(&self) -> SourceDiagnostics {
        SourceDiagnostics::from_source(self)
    }

    fn validate(&self, operation: &'static str) -> Result<()> {
        if self.files.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, operation, "local Parquet source has no files")
                .with_detail("reason", "empty_source_files")
                .with_detail("rejected_features", RejectedSourceFeature::EmptySource.as_str()));
        }

        validate_unique_strings(self.read_columns.iter().map(String::as_str), operation, "read_column")?;
        validate_unique_named(
            self.output_columns.iter().map(|column| column.name.as_str()),
            operation,
            "duplicate_output_column",
        )?;
        validate_unique_named(
            self.required_columns.iter().map(|column| column.name.as_str()),
            operation,
            "duplicate_required_column",
        )?;
        validate_unique_named(
            self.partition_columns.iter().map(|column| column.name.as_str()),
            operation,
            "duplicate_partition_column",
        )?;

        validate_read_partition_name_collisions(&self.read_columns, &self.partition_columns, operation)?;
        self.validate_kind_shape(operation)?;
        self.validate_partition_values(operation)?;
        self.validate_output_and_required_refs(operation)?;
        Ok(())
    }

    fn validate_kind_shape(&self, operation: &'static str) -> Result<()> {
        match self.kind {
            LocalParquetSourceKind::Data => {
                for output in &self.output_columns {
                    if output.source == SourceColumnRef::ZeroColumnCount {
                        return Err(Error::new_with(
                            ErrorCode::UnsupportedOperation,
                            operation,
                            "zero-column count refs are only valid for metadata row-count sources",
                        )
                        .with_column(output.name.clone())
                        .with_detail("reason", "zero_count_ref_in_data_source"));
                    }
                }
                for required in &self.required_columns {
                    if required.source == SourceColumnRef::ZeroColumnCount {
                        return Err(Error::new_with(
                            ErrorCode::UnsupportedOperation,
                            operation,
                            "zero-column count refs are only valid for metadata row-count sources",
                        )
                        .with_column(required.name.clone())
                        .with_detail("reason", "zero_count_ref_in_data_source"));
                    }
                }
            },
            LocalParquetSourceKind::MetadataRowCount => {
                if !self.read_columns.is_empty()
                    || !self.required_columns.is_empty()
                    || !self.partition_columns.is_empty()
                    || self.output_columns.len() != 1
                    || self.output_columns[0].name != ZERO_COLUMN_COUNT_COLUMN
                    || self.output_columns[0].source != SourceColumnRef::ZeroColumnCount
                {
                    return Err(Error::new_with(
                        ErrorCode::UnsupportedOperation,
                        operation,
                        "metadata row-count sources must use only the synthetic count output",
                    )
                    .with_detail("reason", "invalid_metadata_count_shape"));
                }
            },
        }
        Ok(())
    }

    fn validate_partition_values(&self, operation: &'static str) -> Result<()> {
        for (partition_index, column) in self.partition_columns.iter().enumerate() {
            if !is_supported_partition_dtype(column.data_type) {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    operation,
                    "partition column dtype is not supported by the local Parquet source contract",
                )
                .with_column(column.name.clone())
                .with_detail("reason", "unsupported_partition_dtype")
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name.clone())
                .with_detail("partition_dtype", dtype_label(column.data_type)));
            }
        }

        for (file_index, file) in self.files.iter().enumerate() {
            if file.partition_values.len() != self.partition_columns.len() {
                return Err(Error::new_with(
                    ErrorCode::ColumnLengthMismatch,
                    operation,
                    "file partition value count does not match partition column count",
                )
                .with_fact(FACT_ACTUAL_COUNT, file.partition_values.len())
                .with_fact(FACT_EXPECTED_COUNT, self.partition_columns.len())
                .with_detail("reason", "partition_value_count_mismatch")
                .with_detail("file_index", file_index.to_string())
                .with_detail("stable_id", file.stable_id.clone())
                .with_detail("path", file.path.display().to_string()));
            }

            for (partition_index, (column, value)) in
                self.partition_columns.iter().zip(file.partition_values.iter()).enumerate()
            {
                if !is_supported_partition_dtype(value.data_type()) {
                    return Err(Error::new_with(
                        ErrorCode::UnsupportedDtype,
                        operation,
                        "partition value dtype is not supported by the local Parquet source contract",
                    )
                    .with_column(column.name.clone())
                    .with_detail("reason", "unsupported_partition_value_dtype")
                    .with_detail("file_index", file_index.to_string())
                    .with_detail("stable_id", file.stable_id.clone())
                    .with_detail("path", file.path.display().to_string())
                    .with_detail("partition_index", partition_index.to_string())
                    .with_detail("partition_column", column.name.clone())
                    .with_detail("actual_partition_dtype", dtype_label(value.data_type())));
                }
                if !partition_value_matches_dtype(value, column.data_type) {
                    return Err(Error::new_with(
                        ErrorCode::TypeMismatch,
                        operation,
                        "partition value type does not match partition column dtype",
                    )
                    .with_column(column.name.clone())
                    .with_detail("reason", "partition_value_type_mismatch")
                    .with_detail("file_index", file_index.to_string())
                    .with_detail("stable_id", file.stable_id.clone())
                    .with_detail("path", file.path.display().to_string())
                    .with_detail("partition_index", partition_index.to_string())
                    .with_detail("partition_column", column.name.clone())
                    .with_detail("expected_partition_dtype", dtype_label(column.data_type))
                    .with_detail("actual_partition_dtype", dtype_label(value.data_type())));
                }
                if value.is_null() && !column.nullable {
                    return Err(Error::new_with(
                        ErrorCode::NullUnexpected,
                        operation,
                        "null partition value requires a nullable partition column",
                    )
                    .with_column(column.name.clone())
                    .with_detail("reason", "partition_null_for_non_nullable_column")
                    .with_detail("file_index", file_index.to_string())
                    .with_detail("stable_id", file.stable_id.clone())
                    .with_detail("path", file.path.display().to_string())
                    .with_detail("partition_index", partition_index.to_string())
                    .with_detail("partition_column", column.name.clone())
                    .with_detail("partition_dtype", dtype_label(column.data_type)));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_chunk_partition_values(&self, chunk: &SourceChunk, operation: &'static str) -> Result<()> {
        let source_file = self.files.get(chunk.file_index).ok_or_else(|| {
            Error::new_with(
                ErrorCode::IndexOutOfBounds,
                operation,
                "source chunk file index is out of bounds for supplied source",
            )
            .with_fact(FACT_COLUMN_INDEX, chunk.file_index)
            .with_detail("reason", "source_chunk_file_index_out_of_bounds")
            .with_detail("chunk_index", chunk.chunk_index.to_string())
            .with_detail("file_index", chunk.file_index.to_string())
            .with_detail("source_file_count", self.files.len().to_string())
            .with_detail("stable_id", chunk.stable_id.clone())
            .with_detail("path", chunk.path.display().to_string())
        })?;

        if chunk.partition_values.len() != self.partition_columns.len() {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                operation,
                "chunk partition value count does not match source partition column count",
            )
            .with_fact(FACT_ACTUAL_COUNT, chunk.partition_values.len())
            .with_fact(FACT_EXPECTED_COUNT, self.partition_columns.len())
            .with_detail("reason", "partition_value_count_mismatch")
            .with_detail("chunk_index", chunk.chunk_index.to_string())
            .with_detail("file_index", chunk.file_index.to_string())
            .with_detail("stable_id", chunk.stable_id.clone())
            .with_detail("path", chunk.path.display().to_string()));
        }

        for (partition_index, (column, value)) in
            self.partition_columns.iter().zip(chunk.partition_values.iter()).enumerate()
        {
            if !is_supported_partition_dtype(value.data_type()) {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    operation,
                    "chunk partition value dtype is not supported by the local Parquet source contract",
                )
                .with_column(column.name.clone())
                .with_detail("reason", "unsupported_partition_value_dtype")
                .with_detail("chunk_index", chunk.chunk_index.to_string())
                .with_detail("file_index", chunk.file_index.to_string())
                .with_detail("stable_id", chunk.stable_id.clone())
                .with_detail("path", chunk.path.display().to_string())
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name.clone())
                .with_detail("actual_partition_dtype", dtype_label(value.data_type())));
            }
            if !partition_value_matches_dtype(value, column.data_type) {
                return Err(Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "chunk partition value type does not match source partition column dtype",
                )
                .with_column(column.name.clone())
                .with_detail("reason", "partition_value_type_mismatch")
                .with_detail("chunk_index", chunk.chunk_index.to_string())
                .with_detail("file_index", chunk.file_index.to_string())
                .with_detail("stable_id", chunk.stable_id.clone())
                .with_detail("path", chunk.path.display().to_string())
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name.clone())
                .with_detail("expected_partition_dtype", dtype_label(column.data_type))
                .with_detail("actual_partition_dtype", dtype_label(value.data_type())));
            }
            if value.is_null() && !column.nullable {
                return Err(Error::new_with(
                    ErrorCode::NullUnexpected,
                    operation,
                    "null chunk partition value requires a nullable source partition column",
                )
                .with_column(column.name.clone())
                .with_detail("reason", "partition_null_for_non_nullable_column")
                .with_detail("chunk_index", chunk.chunk_index.to_string())
                .with_detail("file_index", chunk.file_index.to_string())
                .with_detail("stable_id", chunk.stable_id.clone())
                .with_detail("path", chunk.path.display().to_string())
                .with_detail("partition_index", partition_index.to_string())
                .with_detail("partition_column", column.name.clone())
                .with_detail("partition_dtype", dtype_label(column.data_type)));
            }
        }

        self.validate_chunk_source_membership(chunk, source_file, operation)?;
        Ok(())
    }

    fn validate_chunk_source_membership(
        &self,
        chunk: &SourceChunk,
        source_file: &LocalParquetFile,
        operation: &'static str,
    ) -> Result<()> {
        if chunk.path != source_file.path {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk path does not match supplied source file",
                "source_chunk_path_mismatch",
            )
            .with_detail("expected_path", source_file.path.display().to_string())
            .with_detail("actual_path", chunk.path.display().to_string()));
        }

        if chunk.stable_id != source_file.stable_id {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk stable id does not match supplied source file",
                "source_chunk_stable_id_mismatch",
            )
            .with_detail("expected_stable_id", source_file.stable_id.clone())
            .with_detail("actual_stable_id", chunk.stable_id.clone()));
        }

        let expected_read_columns: &[String] = match self.kind {
            LocalParquetSourceKind::Data => self.read_columns.as_slice(),
            LocalParquetSourceKind::MetadataRowCount => &[],
        };
        if chunk.read_columns != expected_read_columns {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk read columns do not match supplied source",
                "source_chunk_read_columns_mismatch",
            )
            .with_detail("expected_read_columns", format!("{:?}", expected_read_columns))
            .with_detail("actual_read_columns", format!("{:?}", chunk.read_columns)));
        }

        if chunk.partition_values != source_file.partition_values {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk partition values do not match supplied source file",
                "source_chunk_partition_values_mismatch",
            )
            .with_detail("expected_partition_values", format!("{:?}", source_file.partition_values))
            .with_detail("actual_partition_values", format!("{:?}", chunk.partition_values)));
        }

        let expected_zero_column_count = self.kind == LocalParquetSourceKind::MetadataRowCount;
        if chunk.zero_column_count != expected_zero_column_count {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk zero-column-count mode does not match supplied source kind",
                "source_chunk_zero_column_count_mismatch",
            )
            .with_detail("source_kind", self.kind.as_str())
            .with_detail("expected_zero_column_count", expected_zero_column_count.to_string())
            .with_detail("actual_zero_column_count", chunk.zero_column_count.to_string()));
        }

        validate_chunk_row_groups_for_source_file(chunk, source_file, operation)?;
        validate_chunk_source_file_contract(chunk, source_file, operation)
    }

    fn validate_output_and_required_refs(&self, operation: &'static str) -> Result<()> {
        for output in &self.output_columns {
            self.validate_named_ref(output.name.as_str(), output.source, operation)?;
        }
        for required in &self.required_columns {
            self.validate_named_ref(required.name.as_str(), required.source, operation)?;
        }
        Ok(())
    }

    fn validate_named_ref(&self, name: &str, source_ref: SourceColumnRef, operation: &'static str) -> Result<()> {
        let target = self.schema.column(name).ok_or_else(|| schema_missing_error(operation, name))?;
        if source_ref == SourceColumnRef::MissingNull {
            if !target.nullable {
                return Err(Error::new_with(
                    ErrorCode::NullUnexpected,
                    operation,
                    "missing source columns can only materialize nullable target columns",
                )
                .with_column(name.to_string())
                .with_detail("reason", "missing_null_for_non_nullable_column")
                .with_detail("target_dtype", dtype_label(target.data_type)));
            }
            if !is_supported_missing_null_dtype(target.data_type) {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    operation,
                    "missing source column dtype is not supported by local Parquet source materialization",
                )
                .with_column(name.to_string())
                .with_detail("reason", "unsupported_missing_null_dtype")
                .with_detail("target_dtype", dtype_label(target.data_type)));
            }
            return Ok(());
        }
        let source_dtype = self.source_ref_dtype(source_ref, operation)?;
        if target.data_type != source_dtype {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                operation,
                "source column reference dtype does not match target schema column",
            )
            .with_column(name.to_string())
            .with_detail("reason", "source_ref_type_mismatch")
            .with_detail("target_dtype", dtype_label(target.data_type))
            .with_detail("source_dtype", dtype_label(source_dtype)));
        }
        if let SourceColumnRef::Partition(index) = source_ref {
            let partition = self.partition_columns.get(index).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::IndexOutOfBounds,
                    operation,
                    "partition column reference index is out of bounds",
                )
                .with_fact(FACT_COLUMN_INDEX, index)
                .with_detail("reason", "partition_ref_out_of_bounds")
                .with_detail("partition_column_count", self.partition_columns.len().to_string())
            })?;
            if target.nullable != partition.nullable {
                return Err(Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "partition source column nullability does not match target schema column",
                )
                .with_column(name.to_string())
                .with_detail("reason", "source_ref_nullability_mismatch")
                .with_detail("source_ref", "partition")
                .with_detail("partition_index", index.to_string())
                .with_detail("partition_column", partition.name.clone())
                .with_detail("target_nullable", target.nullable.to_string())
                .with_detail("source_nullable", partition.nullable.to_string()));
            }
        }
        Ok(())
    }

    fn source_ref_dtype(&self, source_ref: SourceColumnRef, operation: &'static str) -> Result<DataType> {
        match source_ref {
            SourceColumnRef::ReadColumn(index) => {
                let read_name = self.read_columns.get(index).ok_or_else(|| {
                    Error::new_with(
                        ErrorCode::IndexOutOfBounds,
                        operation,
                        "read column reference index is out of bounds",
                    )
                    .with_fact(FACT_COLUMN_INDEX, index)
                    .with_detail("reason", "read_column_ref_out_of_bounds")
                    .with_detail("read_column_count", self.read_columns.len().to_string())
                })?;
                let column = self.schema.column(read_name).ok_or_else(|| schema_missing_error(operation, read_name))?;
                Ok(column.data_type)
            },
            SourceColumnRef::Partition(index) => {
                let partition = self.partition_columns.get(index).ok_or_else(|| {
                    Error::new_with(
                        ErrorCode::IndexOutOfBounds,
                        operation,
                        "partition column reference index is out of bounds",
                    )
                    .with_fact(FACT_COLUMN_INDEX, index)
                    .with_detail("reason", "partition_ref_out_of_bounds")
                    .with_detail("partition_column_count", self.partition_columns.len().to_string())
                })?;
                Ok(partition.data_type)
            },
            SourceColumnRef::ZeroColumnCount => {
                if self.kind != LocalParquetSourceKind::MetadataRowCount {
                    return Err(Error::new_with(
                        ErrorCode::UnsupportedOperation,
                        operation,
                        "zero-column count source ref is not valid for this source kind",
                    )
                    .with_detail("reason", "zero_count_ref_in_data_source"));
                }
                Ok(DataType::new(TypeId::Int64))
            },
            SourceColumnRef::MissingNull => Err(Error::new_with(
                ErrorCode::InvariantViolation,
                operation,
                "missing null source references are validated against their target schema column",
            )
            .with_detail("reason", "missing_null_ref_without_target")),
        }
    }
}

/// Local Parquet source kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalParquetSourceKind {
    /// Normal data-producing local Parquet read.
    Data,
    /// Metadata-only row-count source using [`ZERO_COLUMN_COUNT_COLUMN`].
    MetadataRowCount,
}

impl LocalParquetSourceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::MetadataRowCount => "metadata_row_count",
        }
    }
}

/// Output shape requested from source execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceReadMode {
    /// Return the final source output columns and aliases.
    FinalOutput,
    /// Return columns needed for source-local query execution, including
    /// pure-filter columns.
    ExecutionFrame,
}

/// Policy for splitting local Parquet sources into file-local row-group chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceChunkPlanningPolicy {
    max_source_chunk_bytes: Option<u64>,
    max_row_groups_per_chunk: usize,
}

impl SourceChunkPlanningPolicy {
    /// Create a source chunk planning policy.
    ///
    /// `max_source_chunk_bytes` is applied only when row-group byte metadata is
    /// available. `max_row_groups_per_chunk` must be non-zero.
    pub fn new(max_source_chunk_bytes: Option<u64>, max_row_groups_per_chunk: usize) -> Result<Self> {
        if matches!(max_source_chunk_bytes, Some(0)) {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                OP_CHUNK_POLICY_NEW,
                "max source chunk bytes must be greater than zero",
            )
            .with_detail("reason", "max_source_chunk_bytes_zero"));
        }
        if max_row_groups_per_chunk == 0 {
            return Err(Error::new_with(
                ErrorCode::ForeignInvalidArgument,
                OP_CHUNK_POLICY_NEW,
                "max row groups per chunk must be greater than zero",
            )
            .with_detail("reason", "max_row_groups_per_chunk_zero"));
        }

        Ok(Self { max_source_chunk_bytes, max_row_groups_per_chunk })
    }

    /// Maximum planned chunk bytes when row-group byte metadata is available.
    pub fn max_source_chunk_bytes(&self) -> Option<u64> {
        self.max_source_chunk_bytes
    }

    /// Maximum row groups per planned chunk.
    pub fn max_row_groups_per_chunk(&self) -> usize {
        self.max_row_groups_per_chunk
    }
}

impl Default for SourceChunkPlanningPolicy {
    fn default() -> Self {
        Self { max_source_chunk_bytes: None, max_row_groups_per_chunk: usize::MAX }
    }
}

/// Planned chunks for a local Parquet source.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceChunkPlan {
    chunks: Vec<SourceChunk>,
    row_group_metadata_available: bool,
    size_estimate_based: bool,
    metadata_unavailable_files: usize,
    known_total_row_count: Option<u64>,
}

impl SourceChunkPlan {
    /// Planned file-local chunks.
    pub fn chunks(&self) -> &[SourceChunk] {
        &self.chunks
    }

    /// Whether row-group byte and row-count metadata was available for every
    /// non-empty planned file.
    pub fn row_group_metadata_available(&self) -> bool {
        self.row_group_metadata_available
    }

    /// Whether any chunk byte size is based on a file-level estimate rather
    /// than row-group byte metadata.
    pub fn size_estimate_based(&self) -> bool {
        self.size_estimate_based
    }

    /// Number of non-empty planned files without row-group metadata.
    pub fn metadata_unavailable_files(&self) -> usize {
        self.metadata_unavailable_files
    }

    /// Known total row count when every chunk has row-count metadata.
    pub fn known_total_row_count(&self) -> Option<u64> {
        self.known_total_row_count
    }

    /// Known total row count converted for later `i64` materialization.
    pub fn known_total_row_count_i64(&self) -> Result<Option<i64>> {
        self.known_total_row_count.map(source_row_count_to_i64).transpose()
    }
}

/// One file-local source chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceChunk {
    chunk_index: usize,
    file_index: usize,
    location: ParquetSourceLocation,
    path: PathBuf,
    stable_id: String,
    row_group_indexes: Vec<usize>,
    reads_all_row_groups: bool,
    known_file_bytes: Option<u64>,
    known_row_group_bytes: Option<u64>,
    known_projected_compressed_bytes: Option<u64>,
    known_projected_uncompressed_bytes: Option<u64>,
    known_pure_filter_compressed_bytes: Option<u64>,
    known_pure_filter_uncompressed_bytes: Option<u64>,
    estimated_chunk_bytes: Option<u64>,
    estimated_source_read_bytes: Option<u64>,
    source_file_row_groups: Option<Vec<usize>>,
    source_file_row_group_bytes: Option<Vec<u64>>,
    source_file_projected_compressed_bytes: Option<Vec<u64>>,
    source_file_projected_uncompressed_bytes: Option<Vec<u64>>,
    source_file_pure_filter_compressed_bytes: Option<Vec<u64>>,
    source_file_pure_filter_uncompressed_bytes: Option<Vec<u64>>,
    source_file_row_group_rows: Option<Vec<u64>>,
    read_columns: Vec<String>,
    partition_values: Vec<PartitionValue>,
    zero_column_count: bool,
    known_row_count: Option<u64>,
    size_estimate_based: bool,
}

impl SourceChunk {
    /// Build one planned chunk descriptor from an already-validated source file
    /// and caller-selected row groups.
    ///
    /// This is the direct descriptor path for higher-level planners that have
    /// already decided split boundaries. It validates source membership and
    /// derives byte/row-count facts from metadata already attached to the
    /// [`LocalParquetSource`]; it does not run policy-based chunk planning.
    pub fn from_source_file_selection(
        source: &LocalParquetSource,
        file_index: usize,
        chunk_index: usize,
        row_group_indexes: Option<Vec<usize>>,
    ) -> Result<Self> {
        source.validate(OP_PLAN_CHUNKS)?;
        let source_file = source.files.get(file_index).ok_or_else(|| {
            Error::new_with(
                ErrorCode::IndexOutOfBounds,
                OP_PLAN_CHUNKS,
                "source chunk file index is out of bounds for supplied source",
            )
            .with_fact(FACT_COLUMN_INDEX, file_index)
            .with_detail("reason", "source_chunk_file_index_out_of_bounds")
            .with_detail("file_index", file_index.to_string())
            .with_detail("source_file_count", source.files.len().to_string())
        })?;
        validate_source_file_metadata_lengths(source_file, file_index)?;

        let metadata_available = source_file.row_group_bytes.is_some() && source_file.row_group_rows.is_some();
        if source.kind == LocalParquetSourceKind::MetadataRowCount && !metadata_available {
            return Err(missing_row_group_row_count_metadata_error(source_file, file_index));
        }

        let reads_all_row_groups = row_group_indexes.is_none();
        let selected_row_groups = match row_group_indexes {
            Some(row_groups) => row_groups,
            None => {
                source_file.row_groups.clone().unwrap_or_else(|| chunk_metadata_row_groups_for_source_file(source_file))
            },
        };

        if let Some(row_group_rows) = source_file.row_group_rows.as_deref() {
            validate_selected_row_groups(source_file, file_index, &selected_row_groups, row_group_rows.len())?;
        }

        let known_row_group_bytes =
            optional_checked_row_group_sum(source_file.row_group_bytes.as_deref(), &selected_row_groups);
        let known_projected_compressed_bytes = optional_checked_row_group_sum(
            source_file.projected_row_group_compressed_bytes.as_deref(),
            &selected_row_groups,
        );
        let known_projected_uncompressed_bytes = optional_checked_row_group_sum(
            source_file.projected_row_group_uncompressed_bytes.as_deref(),
            &selected_row_groups,
        );
        let known_pure_filter_compressed_bytes = optional_checked_row_group_sum(
            source_file.pure_filter_row_group_compressed_bytes.as_deref(),
            &selected_row_groups,
        );
        let known_pure_filter_uncompressed_bytes = optional_checked_row_group_sum(
            source_file.pure_filter_row_group_uncompressed_bytes.as_deref(),
            &selected_row_groups,
        );
        let estimated_chunk_bytes = if metadata_available {
            known_projected_uncompressed_bytes.or(known_row_group_bytes)
        } else {
            source_file.file_size_bytes
        };
        let estimated_source_read_bytes = if metadata_available {
            known_projected_uncompressed_bytes
                .zip(known_pure_filter_uncompressed_bytes)
                .and_then(|(projected, filter)| projected.checked_add(filter))
                .or(known_row_group_bytes)
                .or(estimated_chunk_bytes)
        } else {
            source_file.file_size_bytes
        };
        let known_row_count = if let Some(row_group_rows) = source_file.row_group_rows.as_deref() {
            Some(checked_row_group_row_count(
                source_file,
                file_index,
                chunk_index,
                row_group_rows,
                &selected_row_groups,
            )?)
        } else {
            None
        };
        let size_estimate_based = !metadata_available && source_file.file_size_bytes.is_some();

        let chunk = Self {
            chunk_index,
            file_index,
            location: source_file.location.clone(),
            path: source_file.path.clone(),
            stable_id: source_file.stable_id.clone(),
            row_group_indexes: if reads_all_row_groups { Vec::new() } else { selected_row_groups },
            reads_all_row_groups,
            known_file_bytes: source_file.file_size_bytes,
            known_row_group_bytes,
            known_projected_compressed_bytes,
            known_projected_uncompressed_bytes,
            known_pure_filter_compressed_bytes,
            known_pure_filter_uncompressed_bytes,
            estimated_chunk_bytes,
            estimated_source_read_bytes,
            source_file_row_groups: source_file.row_groups.clone(),
            source_file_row_group_bytes: source_file.row_group_bytes.clone(),
            source_file_projected_compressed_bytes: source_file.projected_row_group_compressed_bytes.clone(),
            source_file_projected_uncompressed_bytes: source_file.projected_row_group_uncompressed_bytes.clone(),
            source_file_pure_filter_compressed_bytes: source_file.pure_filter_row_group_compressed_bytes.clone(),
            source_file_pure_filter_uncompressed_bytes: source_file.pure_filter_row_group_uncompressed_bytes.clone(),
            source_file_row_group_rows: source_file.row_group_rows.clone(),
            read_columns: if source.kind == LocalParquetSourceKind::MetadataRowCount {
                Vec::new()
            } else {
                source.read_columns.clone()
            },
            partition_values: source_file.partition_values.clone(),
            zero_column_count: source.kind == LocalParquetSourceKind::MetadataRowCount,
            known_row_count,
            size_estimate_based,
        };
        source.validate_chunk_partition_values(&chunk, OP_PLAN_CHUNKS)?;
        Ok(chunk)
    }

    /// Plan-local chunk index.
    pub fn chunk_index(&self) -> usize {
        self.chunk_index
    }

    /// Source file index.
    pub fn file_index(&self) -> usize {
        self.file_index
    }

    /// Local filesystem path for this chunk's file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Explicit source location for this chunk's file.
    pub fn location(&self) -> &ParquetSourceLocation {
        &self.location
    }

    /// Stable caller-provided file identity.
    pub fn stable_id(&self) -> &str {
        &self.stable_id
    }

    /// Explicit row-group indexes for this chunk.
    ///
    /// An empty slice with [`reads_all_row_groups`](Self::reads_all_row_groups)
    /// set means the later execution boundary should read the whole file.
    pub fn row_group_indexes(&self) -> &[usize] {
        &self.row_group_indexes
    }

    /// Row-group indexes converted to the libcudf C-shim `i32` range.
    pub fn row_group_indexes_i32(&self) -> Result<Vec<i32>> {
        self.row_group_indexes
            .iter()
            .copied()
            .enumerate()
            .map(|(position, row_group)| {
                source_row_group_index_to_i32_with_context(row_group, Some(self), Some(position))
            })
            .collect()
    }

    /// Whether this chunk reads all row groups from its file.
    pub fn reads_all_row_groups(&self) -> bool {
        self.reads_all_row_groups
    }

    /// Known full-file byte size, if supplied.
    pub fn known_file_bytes(&self) -> Option<u64> {
        self.known_file_bytes
    }

    /// Known sum of selected row-group bytes, if available.
    pub fn known_row_group_bytes(&self) -> Option<u64> {
        self.known_row_group_bytes
    }

    /// Known sum of projected output compressed bytes, if available.
    pub fn known_projected_compressed_bytes(&self) -> Option<u64> {
        self.known_projected_compressed_bytes
    }

    /// Known sum of projected output uncompressed bytes, if available.
    pub fn known_projected_uncompressed_bytes(&self) -> Option<u64> {
        self.known_projected_uncompressed_bytes
    }

    /// Known sum of pure-filter compressed bytes, if available.
    pub fn known_pure_filter_compressed_bytes(&self) -> Option<u64> {
        self.known_pure_filter_compressed_bytes
    }

    /// Known sum of pure-filter uncompressed bytes, if available.
    pub fn known_pure_filter_uncompressed_bytes(&self) -> Option<u64> {
        self.known_pure_filter_uncompressed_bytes
    }

    /// Estimated chunk bytes used for source scheduling.
    pub fn estimated_chunk_bytes(&self) -> Option<u64> {
        self.estimated_chunk_bytes
    }

    /// Estimated bytes that may be materialized during source read/decode.
    pub fn estimated_source_read_bytes(&self) -> Option<u64> {
        self.estimated_source_read_bytes
    }

    /// Decoded Parquet read columns for this chunk.
    pub fn read_columns(&self) -> &[String] {
        &self.read_columns
    }

    /// Partition values carried from the source file.
    pub fn partition_values(&self) -> &[PartitionValue] {
        &self.partition_values
    }

    /// Whether this is a synthetic metadata row-count chunk.
    pub fn zero_column_count(&self) -> bool {
        self.zero_column_count
    }

    /// Known row count for this chunk, if row-count metadata was available.
    pub fn known_row_count(&self) -> Option<u64> {
        self.known_row_count
    }

    /// Known row count converted for later `i64` materialization.
    pub fn known_row_count_i64(&self) -> Result<Option<i64>> {
        self.known_row_count.map(source_row_count_to_i64).transpose()
    }

    /// Whether this chunk byte estimate is based on file-level information.
    pub fn size_estimate_based(&self) -> bool {
        self.size_estimate_based
    }
}

/// Convert a source row-group index to the libcudf C-shim `i32` range.
pub fn source_row_group_index_to_i32(row_group: usize) -> Result<i32> {
    source_row_group_index_to_i32_with_context(row_group, None, None)
}

/// Convert a source row count to the synthetic count-column `i64` range.
pub fn source_row_count_to_i64(row_count: u64) -> Result<i64> {
    source_row_count_to_i64_with_context(row_count, None)
}

/// Plan file-local row-group chunks for a checked local Parquet source.
pub fn plan_local_parquet_source_chunks(
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
) -> Result<SourceChunkPlan> {
    source.validate(OP_PLAN_CHUNKS)?;

    let mut chunks = Vec::new();
    let mut metadata_unavailable_files = 0usize;

    for (file_index, file) in source.files.iter().enumerate() {
        if matches!(file.row_groups.as_deref(), Some([])) {
            continue;
        }
        validate_source_file_metadata_lengths(file, file_index)?;

        let row_group_bytes = file.row_group_bytes.as_deref();
        let row_group_rows = file.row_group_rows.as_deref();
        let metadata_available = row_group_bytes.is_some() && row_group_rows.is_some();

        if !metadata_available {
            metadata_unavailable_files += 1;
            if source.kind == LocalParquetSourceKind::MetadataRowCount {
                return Err(missing_row_group_row_count_metadata_error(file, file_index));
            }

            let size_estimate_based = file.file_size_bytes.is_some();
            chunks.push(SourceChunk {
                chunk_index: chunks.len(),
                file_index,
                location: file.location.clone(),
                path: file.path.clone(),
                stable_id: file.stable_id.clone(),
                row_group_indexes: file.row_groups.clone().unwrap_or_default(),
                reads_all_row_groups: file.row_groups.is_none(),
                known_file_bytes: file.file_size_bytes,
                known_row_group_bytes: None,
                known_projected_compressed_bytes: None,
                known_projected_uncompressed_bytes: None,
                known_pure_filter_compressed_bytes: None,
                known_pure_filter_uncompressed_bytes: None,
                estimated_chunk_bytes: file.file_size_bytes,
                estimated_source_read_bytes: file.file_size_bytes,
                source_file_row_groups: file.row_groups.clone(),
                source_file_row_group_bytes: file.row_group_bytes.clone(),
                source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
                source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
                source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
                source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
                source_file_row_group_rows: file.row_group_rows.clone(),
                read_columns: source.read_columns.clone(),
                partition_values: file.partition_values.clone(),
                zero_column_count: false,
                known_row_count: None,
                size_estimate_based,
            });
            continue;
        }

        let row_group_bytes = row_group_bytes.expect("metadata_available checked row_group_bytes");
        let row_group_rows = row_group_rows.expect("metadata_available checked row_group_rows");
        let reads_all_from_source = file.row_groups.is_none();
        let selected_row_groups = match &file.row_groups {
            Some(row_groups) => {
                validate_selected_row_groups(file, file_index, row_groups, row_group_rows.len())?;
                row_groups.clone()
            },
            None => (0..row_group_rows.len()).collect(),
        };

        if selected_row_groups.is_empty() {
            continue;
        }

        let chunk_planning_bytes = file.projected_row_group_uncompressed_bytes.as_deref().unwrap_or(row_group_bytes);
        let chunk_groups = split_row_groups_for_policy(&selected_row_groups, chunk_planning_bytes, policy);
        let single_all_row_group_chunk = reads_all_from_source && chunk_groups.len() == 1;
        for row_groups in chunk_groups {
            let known_row_group_bytes = checked_row_group_sum(row_group_bytes, &row_groups);
            let known_projected_compressed_bytes =
                optional_checked_row_group_sum(file.projected_row_group_compressed_bytes.as_deref(), &row_groups);
            let known_projected_uncompressed_bytes =
                optional_checked_row_group_sum(file.projected_row_group_uncompressed_bytes.as_deref(), &row_groups);
            let known_pure_filter_compressed_bytes =
                optional_checked_row_group_sum(file.pure_filter_row_group_compressed_bytes.as_deref(), &row_groups);
            let known_pure_filter_uncompressed_bytes =
                optional_checked_row_group_sum(file.pure_filter_row_group_uncompressed_bytes.as_deref(), &row_groups);
            let estimated_chunk_bytes = known_projected_uncompressed_bytes.or(known_row_group_bytes);
            let estimated_source_read_bytes = known_projected_uncompressed_bytes
                .zip(known_pure_filter_uncompressed_bytes)
                .and_then(|(projected, filter)| projected.checked_add(filter))
                .or(known_row_group_bytes)
                .or(estimated_chunk_bytes);
            let known_row_count =
                checked_row_group_row_count(file, file_index, chunks.len(), row_group_rows, &row_groups)?;

            let reads_all_row_groups = single_all_row_group_chunk && row_groups.len() == row_group_rows.len();
            chunks.push(SourceChunk {
                chunk_index: chunks.len(),
                file_index,
                location: file.location.clone(),
                path: file.path.clone(),
                stable_id: file.stable_id.clone(),
                row_group_indexes: if reads_all_row_groups { Vec::new() } else { row_groups.clone() },
                reads_all_row_groups,
                known_file_bytes: file.file_size_bytes,
                known_row_group_bytes,
                known_projected_compressed_bytes,
                known_projected_uncompressed_bytes,
                known_pure_filter_compressed_bytes,
                known_pure_filter_uncompressed_bytes,
                estimated_chunk_bytes,
                estimated_source_read_bytes,
                source_file_row_groups: file.row_groups.clone(),
                source_file_row_group_bytes: Some(row_group_bytes.to_vec()),
                source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
                source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
                source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
                source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
                source_file_row_group_rows: Some(row_group_rows.to_vec()),
                read_columns: if source.kind == LocalParquetSourceKind::MetadataRowCount {
                    Vec::new()
                } else {
                    source.read_columns.clone()
                },
                partition_values: file.partition_values.clone(),
                zero_column_count: source.kind == LocalParquetSourceKind::MetadataRowCount,
                known_row_count: Some(known_row_count),
                size_estimate_based: false,
            });
        }
    }

    let row_group_metadata_available = metadata_unavailable_files == 0;
    let size_estimate_based = chunks.iter().any(|chunk| chunk.size_estimate_based);
    let known_total_row_count = known_total_row_count(&chunks)?;

    Ok(SourceChunkPlan {
        chunks,
        row_group_metadata_available,
        size_estimate_based,
        metadata_unavailable_files,
        known_total_row_count,
    })
}

/// File-local facts for a local Parquet source.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetFile {
    location: ParquetSourceLocation,
    path: PathBuf,
    stable_id: String,
    file_size_bytes: Option<u64>,
    row_groups: Option<Vec<usize>>,
    row_group_bytes: Option<Vec<u64>>,
    projected_row_group_compressed_bytes: Option<Vec<u64>>,
    projected_row_group_uncompressed_bytes: Option<Vec<u64>>,
    pure_filter_row_group_compressed_bytes: Option<Vec<u64>>,
    pure_filter_row_group_uncompressed_bytes: Option<Vec<u64>>,
    row_group_rows: Option<Vec<u64>>,
    partition_values: Vec<PartitionValue>,
}

impl LocalParquetFile {
    /// Create a local Parquet file descriptor.
    ///
    /// `row_groups == None` means all row groups. Use
    /// [`with_row_groups`](Self::with_row_groups) with `Some(vec![])` to
    /// preserve an explicit empty row-group selection.
    pub fn new<P, S, V>(path: P, stable_id: S, partition_values: V) -> Result<Self>
    where
        P: AsRef<Path>,
        S: Into<String>,
        V: IntoIterator<Item = PartitionValue>,
    {
        let location = ParquetSourceLocation::local_path(path.as_ref())?;
        let path = path.as_ref().to_path_buf();
        let stable_id = stable_id.into();
        if stable_id.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "file stable_id must not be empty")
                .with_detail("reason", "empty_stable_id")
                .with_detail("path", path.display().to_string()));
        }

        Ok(Self {
            location,
            path,
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            projected_row_group_compressed_bytes: None,
            projected_row_group_uncompressed_bytes: None,
            pure_filter_row_group_compressed_bytes: None,
            pure_filter_row_group_uncompressed_bytes: None,
            row_group_rows: None,
            partition_values: partition_values.into_iter().collect(),
        })
    }

    /// Create an object-store Parquet file descriptor.
    ///
    /// Currently this admits `s3://...` URIs because libcudf/kvikIO owns the
    /// lower remote read implementation. Unsupported schemes are rejected
    /// here instead of being represented as local paths.
    pub fn object_store_uri<U, S, V>(uri: U, stable_id: S, partition_values: V) -> Result<Self>
    where
        U: Into<String>,
        S: Into<String>,
        V: IntoIterator<Item = PartitionValue>,
    {
        let uri = uri.into();
        let location = ParquetSourceLocation::object_store_uri(uri.clone())?;
        let reader_path = object_store_reader_path(&uri);
        let stable_id = stable_id.into();
        if stable_id.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "file stable_id must not be empty")
                .with_detail("reason", "empty_stable_id")
                .with_detail("path", uri));
        }

        Ok(Self {
            location,
            path: PathBuf::from(reader_path),
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            projected_row_group_compressed_bytes: None,
            projected_row_group_uncompressed_bytes: None,
            pure_filter_row_group_compressed_bytes: None,
            pure_filter_row_group_uncompressed_bytes: None,
            row_group_rows: None,
            partition_values: partition_values.into_iter().collect(),
        })
    }

    /// Explicit source location.
    pub fn location(&self) -> &ParquetSourceLocation {
        &self.location
    }

    /// Path string passed to libcudf's Parquet reader.
    pub fn reader_path(&self) -> String {
        self.location.reader_path()
    }

    /// Attach a known full-file size in bytes.
    pub fn with_file_size_bytes(mut self, file_size_bytes: u64) -> Self {
        self.file_size_bytes = Some(file_size_bytes);
        self
    }

    /// Attach selected row groups.
    ///
    /// `None` means read all row groups. `Some(vec![])` means an explicit
    /// empty selection and is preserved by the source contract.
    pub fn with_row_groups(mut self, row_groups: Option<Vec<usize>>) -> Self {
        self.row_groups = row_groups;
        self
    }

    /// Attach row-group byte and row-count metadata.
    ///
    /// The two vectors must have the same length because both are indexed by
    /// file-local row-group ordinal.
    pub fn with_row_group_metadata(mut self, row_group_bytes: Vec<u64>, row_group_rows: Vec<u64>) -> Result<Self> {
        validate_matching_row_group_metadata_lengths(
            OP_FILE_METADATA,
            self.stable_id.as_str(),
            &self.path,
            "row_group_metadata_count_mismatch",
            row_group_bytes.len(),
            row_group_rows.len(),
        )?;
        self.row_group_bytes = Some(row_group_bytes);
        self.row_group_rows = Some(row_group_rows);
        Ok(self)
    }

    /// Attach projected output row-group byte metadata.
    pub fn with_projected_row_group_bytes(
        mut self,
        compressed_bytes: Vec<u64>,
        uncompressed_bytes: Vec<u64>,
    ) -> Result<Self> {
        validate_matching_row_group_metadata_lengths(
            OP_FILE_METADATA,
            self.stable_id.as_str(),
            &self.path,
            "projected_row_group_metadata_count_mismatch",
            compressed_bytes.len(),
            uncompressed_bytes.len(),
        )?;
        self.projected_row_group_compressed_bytes = Some(compressed_bytes);
        self.projected_row_group_uncompressed_bytes = Some(uncompressed_bytes);
        Ok(self)
    }

    /// Attach pure-filter row-group byte metadata.
    pub fn with_pure_filter_row_group_bytes(
        mut self,
        compressed_bytes: Vec<u64>,
        uncompressed_bytes: Vec<u64>,
    ) -> Result<Self> {
        validate_matching_row_group_metadata_lengths(
            OP_FILE_METADATA,
            self.stable_id.as_str(),
            &self.path,
            "pure_filter_row_group_metadata_count_mismatch",
            compressed_bytes.len(),
            uncompressed_bytes.len(),
        )?;
        self.pure_filter_row_group_compressed_bytes = Some(compressed_bytes);
        self.pure_filter_row_group_uncompressed_bytes = Some(uncompressed_bytes);
        Ok(self)
    }

    /// Local filesystem path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stable caller-provided file identity.
    pub fn stable_id(&self) -> &str {
        &self.stable_id
    }

    /// Known full-file size in bytes, if supplied by the caller.
    pub fn file_size_bytes(&self) -> Option<u64> {
        self.file_size_bytes
    }

    /// Selected row groups, or `None` for all row groups.
    pub fn row_groups(&self) -> Option<&[usize]> {
        self.row_groups.as_deref()
    }

    /// Row-group byte metadata, if supplied.
    pub fn row_group_bytes(&self) -> Option<&[u64]> {
        self.row_group_bytes.as_deref()
    }

    /// Projected output compressed byte metadata, if supplied.
    pub fn projected_row_group_compressed_bytes(&self) -> Option<&[u64]> {
        self.projected_row_group_compressed_bytes.as_deref()
    }

    /// Projected output uncompressed byte metadata, if supplied.
    pub fn projected_row_group_uncompressed_bytes(&self) -> Option<&[u64]> {
        self.projected_row_group_uncompressed_bytes.as_deref()
    }

    /// Pure-filter compressed byte metadata, if supplied.
    pub fn pure_filter_row_group_compressed_bytes(&self) -> Option<&[u64]> {
        self.pure_filter_row_group_compressed_bytes.as_deref()
    }

    /// Pure-filter uncompressed byte metadata, if supplied.
    pub fn pure_filter_row_group_uncompressed_bytes(&self) -> Option<&[u64]> {
        self.pure_filter_row_group_uncompressed_bytes.as_deref()
    }

    /// Row-group row-count metadata, if supplied.
    pub fn row_group_rows(&self) -> Option<&[u64]> {
        self.row_group_rows.as_deref()
    }

    /// Per-file partition values in [`LocalParquetSource::partition_columns`]
    /// order.
    pub fn partition_values(&self) -> &[PartitionValue] {
        &self.partition_values
    }
}

/// A source-schema column referenced by a local Parquet reader predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceReaderPredicateColumn {
    name: String,
    source: SourceColumnRef,
}

impl SourceReaderPredicateColumn {
    /// Create a predicate column reference backed by one decoded read column.
    pub fn read_column(name: impl Into<String>, read_index: usize) -> Result<Self> {
        let name = checked_name(name.into(), OP_SOURCE_READER_PREDICATE, "reader predicate column")?;
        Ok(Self { name, source: SourceColumnRef::ReadColumn(read_index) })
    }

    /// Logical source-schema column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Source column reference.
    pub fn source(&self) -> SourceColumnRef {
        self.source
    }
}

/// Comparison operators supported by source-aware reader predicate lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceReaderPredicateOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl SourceReaderPredicateOp {
    fn ast_operator(self) -> AstOperator {
        match self {
            Self::Equal => AstOperator::Equal,
            Self::NotEqual => AstOperator::NotEqual,
            Self::Less => AstOperator::Less,
            Self::LessEqual => AstOperator::LessEqual,
            Self::Greater => AstOperator::Greater,
            Self::GreaterEqual => AstOperator::GreaterEqual,
        }
    }
}

/// Literal value for a source-aware reader predicate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SourceReaderPredicateLiteral {
    Boolean(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    /// Decimal128 raw fixed-point value plus cudf scale.
    Decimal128 {
        raw: i128,
        scale: i32,
    },
    Date32(i32),
    String(String),
}

/// Source-aware local Parquet reader predicate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SourceReaderPredicate {
    /// Binary comparison between a decoded read column and a literal.
    Comparison {
        column: SourceReaderPredicateColumn,
        op: SourceReaderPredicateOp,
        literal: SourceReaderPredicateLiteral,
    },
    /// Literal-list membership for one decoded read column.
    InList { column: SourceReaderPredicateColumn, literals: Vec<SourceReaderPredicateLiteral>, negated: bool },
    /// Null-aware boolean AND of two source reader predicates.
    And(Box<SourceReaderPredicate>, Box<SourceReaderPredicate>),
    /// Null-aware boolean OR of two source reader predicates.
    Or(Box<SourceReaderPredicate>, Box<SourceReaderPredicate>),
    /// Boolean NOT of one source reader predicate.
    Not(Box<SourceReaderPredicate>),
}

impl SourceReaderPredicate {
    /// Build a column/literal comparison.
    pub fn comparison(
        column: SourceReaderPredicateColumn,
        op: SourceReaderPredicateOp,
        literal: SourceReaderPredicateLiteral,
    ) -> Self {
        Self::Comparison { column, op, literal }
    }

    /// Build `column = literal`.
    pub fn equal(column: SourceReaderPredicateColumn, literal: SourceReaderPredicateLiteral) -> Self {
        Self::comparison(column, SourceReaderPredicateOp::Equal, literal)
    }

    /// Build `column >= literal`.
    pub fn greater_equal(column: SourceReaderPredicateColumn, literal: SourceReaderPredicateLiteral) -> Self {
        Self::comparison(column, SourceReaderPredicateOp::GreaterEqual, literal)
    }

    /// Build literal-list membership for one decoded read column.
    pub fn in_list(
        column: SourceReaderPredicateColumn,
        literals: Vec<SourceReaderPredicateLiteral>,
        negated: bool,
    ) -> Result<Self> {
        if literals.is_empty() {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicate IN list must not be empty",
            )
            .with_detail("reason", "empty_in_list"));
        }
        Ok(Self::InList { column, literals, negated })
    }

    /// Build a null-aware boolean AND.
    pub fn and(left: SourceReaderPredicate, right: SourceReaderPredicate) -> Self {
        Self::And(Box::new(left), Box::new(right))
    }

    /// Build a null-aware boolean OR.
    pub fn or(left: SourceReaderPredicate, right: SourceReaderPredicate) -> Self {
        Self::Or(Box::new(left), Box::new(right))
    }
}

/// Build a libcudf reader predicate in the same dtype domain used by the local
/// Parquet reader.
///
/// Decimal predicates are typed from the Parquet footer for the selected
/// source chunk. This keeps logical `Decimal128` source schemas compatible with
/// Parquet columns that libcudf evaluates as Decimal32, Decimal64, or
/// Decimal128.
pub fn build_local_parquet_reader_predicate(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    predicate: &SourceReaderPredicate,
) -> Result<AstExpression> {
    source.validate_chunk_partition_values(chunk, OP_SOURCE_READER_PREDICATE)?;
    lower_source_reader_predicate(runtime, source, chunk, predicate)
}

fn lower_source_reader_predicate(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    predicate: &SourceReaderPredicate,
) -> Result<AstExpression> {
    match predicate {
        SourceReaderPredicate::Comparison { column, op, literal } => {
            let (read_index, target_type) = resolve_source_reader_column(source, chunk, column)?;
            let left = source_reader_column_ast(read_index)?;
            let right = source_reader_literal_ast(runtime, literal, target_type)?;
            AstExpression::binary(op.ast_operator(), left, right).map_err(|error| {
                error
                    .reanchor(OP_SOURCE_READER_PREDICATE)
                    .with_detail("reason", "ast_binary_create_failed")
                    .with_detail("operator", format!("{op:?}"))
            })
        },
        SourceReaderPredicate::InList { column, literals, negated } => {
            if literals.is_empty() {
                return Err(Error::new_with(
                    ErrorCode::EmptyInput,
                    OP_SOURCE_READER_PREDICATE,
                    "source reader predicate IN list must not be empty",
                )
                .with_detail("reason", "empty_in_list"));
            }
            let (read_index, target_type) = resolve_source_reader_column(source, chunk, column)?;
            let input = source_reader_column_ast(read_index)?;
            let values = literals
                .iter()
                .map(|literal| source_reader_literal_ast(runtime, literal, target_type))
                .collect::<Result<Vec<_>>>()?;
            if *negated {
                AstExpression::is_not_in_list(input, values)
            } else {
                AstExpression::is_in_list(input, values)
            }
            .map_err(|error| {
                error
                    .reanchor(OP_SOURCE_READER_PREDICATE)
                    .with_detail("reason", "ast_in_list_create_failed")
                    .with_detail("negated", negated.to_string())
                    .with_detail("value_count", literals.len().to_string())
            })
        },
        SourceReaderPredicate::And(left, right) => {
            let left = lower_source_reader_predicate(runtime, source, chunk, left)?;
            let right = lower_source_reader_predicate(runtime, source, chunk, right)?;
            AstExpression::binary(AstOperator::NullLogicalAnd, left, right)
                .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_and_failed"))
        },
        SourceReaderPredicate::Or(left, right) => {
            let left = lower_source_reader_predicate(runtime, source, chunk, left)?;
            let right = lower_source_reader_predicate(runtime, source, chunk, right)?;
            AstExpression::binary(AstOperator::NullLogicalOr, left, right)
                .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_or_failed"))
        },
        SourceReaderPredicate::Not(input) => {
            let input = lower_source_reader_predicate(runtime, source, chunk, input)?;
            AstExpression::unary(AstOperator::Not, input)
                .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_not_failed"))
        },
    }
}

fn source_reader_column_ast(read_index: usize) -> Result<AstExpression> {
    let index = i32::try_from(read_index).map_err(|_| {
        Error::new_with(
            ErrorCode::IndexOutOfBounds,
            OP_SOURCE_READER_PREDICATE,
            "source reader predicate read column index exceeds i32 range",
        )
        .with_fact(FACT_COLUMN_INDEX, read_index)
        .with_detail("reason", "read_column_index_exceeds_i32")
    })?;
    AstExpression::left_column(index)
        .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_column_failed"))
}

fn source_reader_literal_ast(
    runtime: &Runtime,
    literal: &SourceReaderPredicateLiteral,
    target_type: DataType,
) -> Result<AstExpression> {
    let scalar = source_reader_literal_scalar(runtime, literal, target_type)?;
    AstExpression::literal(scalar)
        .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_literal_failed"))
}

fn source_reader_literal_scalar(
    runtime: &Runtime,
    literal: &SourceReaderPredicateLiteral,
    target_type: DataType,
) -> Result<crate::core::Scalar> {
    match target_type.id() {
        TypeId::Bool8 => match literal {
            SourceReaderPredicateLiteral::Boolean(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::Int64 => match literal {
            SourceReaderPredicateLiteral::Int64(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::UInt64 => match literal {
            SourceReaderPredicateLiteral::UInt64(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::Float64 => match literal {
            SourceReaderPredicateLiteral::Float64(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::String => match literal {
            SourceReaderPredicateLiteral::String(value) => runtime.string_scalar(value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::TimestampDays => match literal {
            SourceReaderPredicateLiteral::Date32(value) => {
                runtime.timestamp_raw_scalar(i64::from(*value), TimestampResolution::Days)
            },
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => {
            let raw = source_reader_decimal_literal_raw(literal, target_type.scale())?;
            runtime.decimal_raw_scalar(raw, target_type)
        },
        other => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            OP_SOURCE_READER_PREDICATE,
            "source reader predicate literal target dtype is not supported",
        )
        .with_fact(FACT_TYPE_ID, other)
        .with_detail("reason", "unsupported_literal_target_dtype")),
    }
    .map_err(|error| {
        error
            .reanchor(OP_SOURCE_READER_PREDICATE)
            .with_detail_if_absent("target_type_id", target_type.id().to_string())
            .with_detail_if_absent("target_type_scale", target_type.scale().to_string())
    })
}

fn source_reader_literal_type_mismatch<T>(literal: &SourceReaderPredicateLiteral, target_type: DataType) -> Result<T> {
    Err(Error::new_with(
        ErrorCode::TypeMismatch,
        OP_SOURCE_READER_PREDICATE,
        "source reader predicate literal type does not match target column dtype",
    )
    .with_detail("reason", "literal_type_mismatch")
    .with_detail("literal", format!("{literal:?}"))
    .with_detail("target_type_id", target_type.id().to_string())
    .with_detail("target_type_scale", target_type.scale().to_string()))
}

fn source_reader_decimal_literal_raw(literal: &SourceReaderPredicateLiteral, target_scale: i32) -> Result<i128> {
    match literal {
        SourceReaderPredicateLiteral::Int64(value) => decimal_raw_for_integer_value(i128::from(*value), target_scale),
        SourceReaderPredicateLiteral::UInt64(value) => decimal_raw_for_integer_value(i128::from(*value), target_scale),
        SourceReaderPredicateLiteral::Decimal128 { raw, scale } => rescale_decimal_raw(*raw, *scale, target_scale),
        _ => source_reader_literal_type_mismatch(literal, DataType::fixed_point(TypeId::Decimal128, target_scale)),
    }
}

fn resolve_source_reader_column(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    column: &SourceReaderPredicateColumn,
) -> Result<(usize, DataType)> {
    let read_index = match column.source {
        SourceColumnRef::ReadColumn(index) => index,
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicates can only reference decoded read columns",
            )
            .with_column(column.name.clone())
            .with_detail("reason", "predicate_column_not_read_column"));
        },
    };
    let read_column = source.read_columns.get(read_index).ok_or_else(|| {
        Error::new_with(
            ErrorCode::MissingColumn,
            OP_SOURCE_READER_PREDICATE,
            "source reader predicate column is missing from read projection",
        )
        .with_column(column.name.clone())
        .with_fact(FACT_COLUMN_INDEX, read_index)
        .with_detail("reason", "read_projection_column_missing")
    })?;
    let schema_column = source.schema.column(column.name.as_str()).ok_or_else(|| {
        schema_missing_error(OP_SOURCE_READER_PREDICATE, column.name.as_str())
            .with_detail("reason", "predicate_schema_column_missing")
    })?;
    let logical_type = schema_column.data_type();
    let target_type = if logical_type.id().is_decimal() {
        let reader_type = parquet_reader_decimal_dtype(chunk.path(), read_column, read_index)?;
        if logical_type.scale() != reader_type.scale() {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicate decimal scale does not match Parquet footer scale",
            )
            .with_column(column.name.clone())
            .with_detail("reason", "decimal_scale_mismatch")
            .with_detail("logical_scale", logical_type.scale().to_string())
            .with_detail("reader_scale", reader_type.scale().to_string())
            .with_detail("read_column", read_column.clone()));
        }
        reader_type
    } else {
        logical_type
    };
    Ok((read_index, target_type))
}

fn parquet_reader_decimal_dtype(path: &Path, read_column: &str, read_index: usize) -> Result<DataType> {
    let file = File::open(path).map_err(|source| {
        Error::new_with(
            ErrorCode::IoFailure,
            OP_SOURCE_READER_PREDICATE,
            "failed to open Parquet file for reader predicate footer typing",
        )
        .with_detail("reason", "parquet_footer_open_failed")
        .with_detail("path", path.display().to_string())
        .with_source(source)
    })?;
    let reader = SerializedFileReader::new(file).map_err(|source| {
        Error::new_with(
            ErrorCode::IoFailure,
            OP_SOURCE_READER_PREDICATE,
            "failed to read Parquet footer for reader predicate typing",
        )
        .with_detail("reason", "parquet_footer_read_failed")
        .with_detail("path", path.display().to_string())
        .with_source(source)
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let descriptor = schema
        .columns()
        .get(read_index)
        .filter(|descriptor| parquet_descriptor_matches_column(descriptor, read_column))
        .or_else(|| {
            schema.columns().iter().find(|descriptor| parquet_descriptor_matches_column(descriptor, read_column))
        })
        .ok_or_else(|| {
            Error::new_with(
                ErrorCode::MissingColumn,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicate column was not found in Parquet footer",
            )
            .with_column(read_column.to_string())
            .with_fact(FACT_COLUMN_INDEX, read_index)
            .with_detail("reason", "parquet_footer_column_missing")
            .with_detail("path", path.display().to_string())
        })?;
    let (precision, scale) = parquet_decimal_precision_scale(descriptor).ok_or_else(|| {
        Error::new_with(
            ErrorCode::TypeMismatch,
            OP_SOURCE_READER_PREDICATE,
            "source schema declares a decimal predicate column but Parquet footer column is not decimal",
        )
        .with_column(read_column.to_string())
        .with_fact(FACT_COLUMN_INDEX, read_index)
        .with_detail("reason", "parquet_footer_column_not_decimal")
        .with_detail("path", path.display().to_string())
        .with_detail("physical_type", format!("{:?}", descriptor.physical_type()))
    })?;
    let type_id = parquet_reader_decimal_type_id(descriptor, precision)?;
    Ok(DataType::fixed_point(type_id, -scale))
}

fn parquet_descriptor_matches_column(descriptor: &ColumnDescriptor, column_name: &str) -> bool {
    descriptor.name() == column_name || descriptor.path().parts().last().is_some_and(|part| part == column_name)
}

fn parquet_decimal_precision_scale(descriptor: &ColumnDescriptor) -> Option<(i32, i32)> {
    if let Some(LogicalType::Decimal { precision, scale }) = descriptor.logical_type_ref() {
        return Some((*precision, *scale));
    }
    (descriptor.converted_type() == ConvertedType::DECIMAL)
        .then(|| (descriptor.type_precision(), descriptor.type_scale()))
}

fn parquet_reader_decimal_type_id(descriptor: &ColumnDescriptor, precision: i32) -> Result<TypeId> {
    match descriptor.physical_type() {
        ParquetPhysicalType::INT32 => Ok(TypeId::Decimal32),
        ParquetPhysicalType::INT64 => Ok(TypeId::Decimal64),
        ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY => match descriptor.type_length() {
            0..=4 => Ok(TypeId::Decimal32),
            5..=8 => Ok(TypeId::Decimal64),
            9..=16 => Ok(TypeId::Decimal128),
            type_length => Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OP_SOURCE_READER_PREDICATE,
                "Parquet decimal fixed byte width exceeds Decimal128 storage",
            )
            .with_detail("reason", "decimal_fixed_len_byte_width_unsupported")
            .with_detail("type_length", type_length.to_string())),
        },
        ParquetPhysicalType::BYTE_ARRAY => match precision {
            1..=9 => Ok(TypeId::Decimal32),
            10..=18 => Ok(TypeId::Decimal64),
            19..=38 => Ok(TypeId::Decimal128),
            precision => Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OP_SOURCE_READER_PREDICATE,
                "Parquet decimal precision exceeds Decimal128 storage",
            )
            .with_detail("reason", "decimal_precision_unsupported")
            .with_detail("precision", precision.to_string())),
        },
        physical_type => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            OP_SOURCE_READER_PREDICATE,
            "Parquet decimal physical type is not supported by reader predicates",
        )
        .with_detail("reason", "unsupported_decimal_physical_type")
        .with_detail("physical_type", format!("{physical_type:?}"))),
    }
}

fn decimal_raw_for_integer_value(value: i128, target_scale: i32) -> Result<i128> {
    if target_scale <= 0 {
        let multiplier = checked_pow10(target_scale.unsigned_abs())?;
        value.checked_mul(multiplier).ok_or_else(|| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "integer source reader predicate literal raw value overflows target decimal scale",
            )
            .with_detail("reason", "integer_decimal_literal_overflow")
            .with_detail("target_scale", target_scale.to_string())
        })
    } else {
        let divisor = checked_pow10(target_scale as u32)?;
        if value % divisor != 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "integer source reader predicate literal cannot be represented exactly at target decimal scale",
            )
            .with_detail("reason", "integer_decimal_literal_not_exact")
            .with_detail("target_scale", target_scale.to_string()));
        }
        Ok(value / divisor)
    }
}

fn rescale_decimal_raw(raw: i128, source_scale: i32, target_scale: i32) -> Result<i128> {
    let scale_delta = source_scale.checked_sub(target_scale).ok_or_else(|| {
        Error::new_with(
            ErrorCode::ValueOutOfRange,
            OP_SOURCE_READER_PREDICATE,
            "decimal source reader predicate literal scale alignment overflowed",
        )
        .with_detail("reason", "decimal_scale_alignment_overflow")
        .with_detail("source_scale", source_scale.to_string())
        .with_detail("target_scale", target_scale.to_string())
    })?;
    if scale_delta >= 0 {
        let multiplier = checked_pow10(scale_delta as u32)?;
        raw.checked_mul(multiplier).ok_or_else(|| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "decimal source reader predicate literal raw value overflows target scale",
            )
            .with_detail("reason", "decimal_literal_rescale_overflow")
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string())
        })
    } else {
        let divisor = checked_pow10(scale_delta.unsigned_abs())?;
        if raw % divisor != 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "decimal source reader predicate literal cannot be represented exactly at target scale",
            )
            .with_detail("reason", "decimal_literal_rescale_not_exact")
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string()));
        }
        Ok(raw / divisor)
    }
}

fn checked_pow10(exponent: u32) -> Result<i128> {
    let mut value = 1i128;
    for _ in 0..exponent {
        value = value.checked_mul(10).ok_or_else(|| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "decimal source reader predicate scale factor overflows i128",
            )
            .with_detail("reason", "decimal_scale_factor_overflow")
            .with_detail("exponent", exponent.to_string())
        })?;
    }
    Ok(value)
}

/// Source-level schema used for validation and typed empty outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSchema {
    columns: Vec<SourceColumn>,
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
    name: String,
    data_type: DataType,
    nullable: bool,
}

impl SourceColumn {
    /// Create a source schema column.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Result<Self> {
        let name = checked_name(name.into(), OP_COLUMN_NEW, "source column")?;
        Ok(Self { name, data_type, nullable })
    }

    /// Column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Logical cuDF data type.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Source-level nullability truth.
    pub fn nullable(&self) -> bool {
        self.nullable
    }
}

/// A final source output column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceOutputColumn {
    name: String,
    source: SourceColumnRef,
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
    name: String,
    source: SourceColumnRef,
    role: SourceColumnRole,
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
    /// This index points into [`LocalParquetSource::read_columns`], not into
    /// the original Parquet file schema. Callers that start from file-schema
    /// ordinals must translate them into decoded read-projection ordinals.
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

/// An explicit partition column to be materialized after reading a file/chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionColumn {
    name: String,
    data_type: DataType,
    nullable: bool,
}

impl PartitionColumn {
    /// Create partition column metadata.
    ///
    /// `nullable` records source schema truth. Typed null partition values
    /// are accepted only when this flag is true.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Result<Self> {
        let name = checked_name(name.into(), OP_PARTITION_NEW, "partition column")?;
        Ok(Self { name, data_type, nullable })
    }

    /// Partition column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Partition column dtype.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Source-level nullability truth.
    pub fn nullable(&self) -> bool {
        self.nullable
    }
}

/// Partition value supported by the local Parquet source contract.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PartitionValue {
    /// Typed null value. The dtype must match the partition column dtype and
    /// the partition column must be nullable.
    Null(DataType),
    /// Boolean value.
    Boolean(bool),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Signed 32-bit integer.
    Int32(i32),
    /// Signed 64-bit integer.
    Int64(i64),
    /// Unsigned 8-bit integer.
    UInt8(u8),
    /// Unsigned 16-bit integer.
    UInt16(u16),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// 32-bit floating point value.
    Float32(f32),
    /// 64-bit floating point value.
    Float64(f64),
    /// Decimal128 raw fixed-point value plus cudf scale.
    Decimal128 { raw: i128, scale: i32 },
    /// Arrow Date32-compatible day value, represented as cudf timestamp days.
    Date32(i32),
    /// Timezone-less timestamp raw value with explicit resolution.
    Timestamp { raw: i64, resolution: TimestampResolution },
    /// Duration raw value with explicit resolution.
    Duration { raw: i64, resolution: TimestampResolution },
    /// UTF-8 string value.
    String(String),
}

impl PartitionValue {
    /// cuDF dtype represented by this partition value.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Null(data_type) => *data_type,
            Self::Boolean(_) => DataType::new(TypeId::Bool8),
            Self::Int8(_) => DataType::new(TypeId::Int8),
            Self::Int16(_) => DataType::new(TypeId::Int16),
            Self::Int32(_) => DataType::new(TypeId::Int32),
            Self::Int64(_) => DataType::new(TypeId::Int64),
            Self::UInt8(_) => DataType::new(TypeId::UInt8),
            Self::UInt16(_) => DataType::new(TypeId::UInt16),
            Self::UInt32(_) => DataType::new(TypeId::UInt32),
            Self::UInt64(_) => DataType::new(TypeId::UInt64),
            Self::Float32(_) => DataType::new(TypeId::Float32),
            Self::Float64(_) => DataType::new(TypeId::Float64),
            Self::Decimal128 { scale, .. } => DataType::fixed_point(TypeId::Decimal128, *scale),
            Self::Date32(_) => DataType::new(TypeId::TimestampDays),
            Self::Timestamp { resolution, .. } => DataType::new(resolution.type_id()),
            Self::Duration { resolution, .. } => DataType::new(resolution.duration_type_id()),
            Self::String(_) => DataType::new(TypeId::String),
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, Self::Null(_))
    }
}

/// Derived source diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDiagnostics {
    format: SourceFormat,
    direct_device_read: bool,
    file_count: usize,
    ranged_file_count: usize,
    known_bytes: Option<u64>,
    selected_row_groups: Option<usize>,
    predicate_disposition: PredicateDisposition,
    byte_range_disposition: ByteRangeDisposition,
    partition_column_disposition: PartitionColumnDisposition,
    zero_column_count_disposition: ZeroColumnCountDisposition,
    schema_mapping_disposition: SchemaMappingDisposition,
    read_column_count: usize,
    output_column_count: usize,
    pure_filter_column_count: usize,
    rejected_features: Vec<RejectedSourceFeature>,
}

impl SourceDiagnostics {
    fn from_source(source: &LocalParquetSource) -> Self {
        let known_bytes = source.files.iter().try_fold(0u64, |acc, file| {
            let file_size = file.file_size_bytes?;
            acc.checked_add(file_size)
        });
        let selected_row_groups = if source.files.iter().any(|file| file.row_groups.is_some()) {
            Some(source.files.iter().filter_map(|file| file.row_groups.as_ref()).map(Vec::len).sum())
        } else {
            None
        };
        let pure_filter_column_count =
            source.required_columns.iter().filter(|column| column.role == SourceColumnRole::PureFilter).count();
        let zero_column_count_disposition = match source.kind {
            LocalParquetSourceKind::Data => ZeroColumnCountDisposition::NotRequired,
            LocalParquetSourceKind::MetadataRowCount => ZeroColumnCountDisposition::Represented,
        };

        let schema_mapping_disposition =
            if source.output_columns.iter().any(|column| column.source == SourceColumnRef::MissingNull)
                || source.required_columns.iter().any(|column| column.source == SourceColumnRef::MissingNull)
            {
                SchemaMappingDisposition::MissingNullableColumns
            } else {
                SchemaMappingDisposition::DirectByName
            };

        Self {
            format: SourceFormat::Parquet,
            direct_device_read: true,
            file_count: source.files.len(),
            ranged_file_count: 0,
            known_bytes,
            selected_row_groups,
            predicate_disposition: if source.reader_predicate.is_some() {
                PredicateDisposition::ReaderPushdown
            } else {
                PredicateDisposition::Absent
            },
            byte_range_disposition: ByteRangeDisposition::FullFiles,
            partition_column_disposition: if source.partition_columns.is_empty() {
                PartitionColumnDisposition::Absent
            } else {
                PartitionColumnDisposition::Represented
            },
            zero_column_count_disposition,
            schema_mapping_disposition,
            read_column_count: source.read_columns.len(),
            output_column_count: source.output_columns.len(),
            pure_filter_column_count,
            rejected_features: Vec::new(),
        }
    }

    /// Source file format.
    pub fn format(&self) -> SourceFormat {
        self.format
    }

    /// Whether the current local Parquet execution path can honor all
    /// represented source facts.
    pub fn direct_device_read(&self) -> bool {
        self.direct_device_read
    }

    /// Number of local files.
    pub fn file_count(&self) -> usize {
        self.file_count
    }

    /// Number of ranged files. Phase 1 accepts only full local files.
    pub fn ranged_file_count(&self) -> usize {
        self.ranged_file_count
    }

    /// Known total bytes when all file sizes are known and addition fits.
    pub fn known_bytes(&self) -> Option<u64> {
        self.known_bytes
    }

    /// Total selected row groups if any file has explicit row-group selection.
    pub fn selected_row_groups(&self) -> Option<usize> {
        self.selected_row_groups
    }

    /// Predicate disposition.
    pub fn predicate_disposition(&self) -> PredicateDisposition {
        self.predicate_disposition
    }

    /// Byte-range disposition.
    pub fn byte_range_disposition(&self) -> ByteRangeDisposition {
        self.byte_range_disposition
    }

    /// Partition-column disposition.
    pub fn partition_column_disposition(&self) -> PartitionColumnDisposition {
        self.partition_column_disposition
    }

    /// Zero-column count disposition.
    pub fn zero_column_count_disposition(&self) -> ZeroColumnCountDisposition {
        self.zero_column_count_disposition
    }

    /// Schema mapping disposition.
    pub fn schema_mapping_disposition(&self) -> SchemaMappingDisposition {
        self.schema_mapping_disposition
    }

    /// Number of decoded read columns.
    pub fn read_column_count(&self) -> usize {
        self.read_column_count
    }

    /// Number of final output columns.
    pub fn output_column_count(&self) -> usize {
        self.output_column_count
    }

    /// Number of pure-filter required columns.
    pub fn pure_filter_column_count(&self) -> usize {
        self.pure_filter_column_count
    }

    /// Neutral rejected source features. Valid constructed sources have no
    /// rejected features.
    pub fn rejected_features(&self) -> &[RejectedSourceFeature] {
        &self.rejected_features
    }
}

/// Source format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    /// Parquet.
    Parquet,
}

/// Predicate support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredicateDisposition {
    /// No source predicate was requested.
    Absent,
    /// Predicate was used for pruning only.
    PruningOnly,
    /// Predicate was represented as a reader-level Parquet predicate.
    ReaderPushdown,
    /// Predicate remains a native post-scan filter.
    NativePostScanOnly,
    /// Predicate execution could not be represented.
    Rejected,
}

/// Byte-range support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteRangeDisposition {
    /// Full local files only.
    FullFiles,
    /// Caller already coalesced ranges into full local files.
    CoalescedFullFiles,
    /// Byte ranges could not be represented.
    Rejected,
}

/// Partition-column support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionColumnDisposition {
    /// No partition columns.
    Absent,
    /// Partition columns are represented explicitly.
    Represented,
    /// Partition columns could not be represented.
    Rejected,
}

/// Metadata-only zero-column count support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeroColumnCountDisposition {
    /// Count mode is not requested.
    NotRequired,
    /// Count mode is represented by the synthetic count column.
    Represented,
    /// Count mode could not be represented.
    Rejected,
}

/// Schema mapping support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaMappingDisposition {
    /// Direct exact by-name mapping.
    DirectByName,
    /// One or more nullable logical columns are absent from the physical file
    /// schema and represented as typed null columns.
    MissingNullableColumns,
    /// Schema mapping could not be represented.
    Rejected,
}

/// Neutral source feature names for rejected planner features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectedSourceFeature {
    /// The input was not a local filesystem source.
    NonLocalSource,
    /// Partition columns or values could not be represented.
    UnsupportedPartitionColumns,
    /// Predicate execution could not be represented.
    UnsupportedPredicateExecution,
    /// Scan limit could not be represented.
    UnsupportedScanLimit,
    /// Ordering requirement could not be represented.
    UnsupportedOrderingRequirement,
    /// The source contained no files.
    EmptySource,
    /// Partition values could not be represented.
    UnsupportedPartitionValues,
    /// Byte ranges could not be represented.
    UnsupportedByteRange,
    /// Row-level selection could not be represented.
    UnsupportedRowSelection,
    /// Metadata-only count could not be represented.
    UnsupportedZeroColumnCount,
    /// Schema mapping could not be represented.
    UnsupportedSchemaMapping,
}

impl RejectedSourceFeature {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonLocalSource => "non_local_source",
            Self::UnsupportedPartitionColumns => "unsupported_partition_columns",
            Self::UnsupportedPredicateExecution => "unsupported_predicate_execution",
            Self::UnsupportedScanLimit => "unsupported_scan_limit",
            Self::UnsupportedOrderingRequirement => "unsupported_ordering_requirement",
            Self::EmptySource => "empty_source",
            Self::UnsupportedPartitionValues => "unsupported_partition_values",
            Self::UnsupportedByteRange => "unsupported_byte_range",
            Self::UnsupportedRowSelection => "unsupported_row_selection",
            Self::UnsupportedZeroColumnCount => "unsupported_zero_column_count",
            Self::UnsupportedSchemaMapping => "unsupported_schema_mapping",
        }
    }
}

macro_rules! stable_display {
    ($type:ty, {$($variant:pat => $name:expr),+ $(,)?}) => {
        impl $type {
            /// Stable diagnostic string.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($variant => $name,)+
                }
            }
        }

        impl fmt::Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

stable_display!(SourceFormat, {
    SourceFormat::Parquet => "parquet",
});
stable_display!(PredicateDisposition, {
    PredicateDisposition::Absent => "absent",
    PredicateDisposition::PruningOnly => "pruning_only",
    PredicateDisposition::ReaderPushdown => "reader_pushdown",
    PredicateDisposition::NativePostScanOnly => "native_post_scan_only",
    PredicateDisposition::Rejected => "rejected",
});
stable_display!(ByteRangeDisposition, {
    ByteRangeDisposition::FullFiles => "full_files",
    ByteRangeDisposition::CoalescedFullFiles => "coalesced_full_files",
    ByteRangeDisposition::Rejected => "rejected",
});
stable_display!(PartitionColumnDisposition, {
    PartitionColumnDisposition::Absent => "absent",
    PartitionColumnDisposition::Represented => "represented",
    PartitionColumnDisposition::Rejected => "rejected",
});
stable_display!(ZeroColumnCountDisposition, {
    ZeroColumnCountDisposition::NotRequired => "not_required",
    ZeroColumnCountDisposition::Represented => "represented",
    ZeroColumnCountDisposition::Rejected => "rejected",
});
stable_display!(SchemaMappingDisposition, {
    SchemaMappingDisposition::DirectByName => "direct_by_name",
    SchemaMappingDisposition::MissingNullableColumns => "missing_nullable_columns",
    SchemaMappingDisposition::Rejected => "rejected",
});

impl fmt::Display for RejectedSourceFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

fn checked_name(name: String, operation: &'static str, resource: &'static str) -> Result<String> {
    if name.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "name must not be empty")
            .with_detail("reason", "empty_name")
            .with_detail("resource", resource));
    }
    Ok(name)
}

fn validate_local_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "file path must not be empty")
            .with_detail("reason", "empty_path")
            .with_detail("resource", "path"));
    }

    let path_string = path.to_string_lossy();
    if path_string.contains("://") {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_FILE_NEW,
            "local Parquet sources require local filesystem paths",
        )
        .with_detail("reason", "non_local_source")
        .with_detail("path", path_string.into_owned())
        .with_detail("rejected_features", RejectedSourceFeature::NonLocalSource.as_str()));
    }
    Ok(())
}

fn validate_object_store_uri(uri: &str) -> Result<()> {
    if uri.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "object-store URI must not be empty")
            .with_detail("reason", "empty_object_store_uri")
            .with_detail("resource", "location"));
    }

    let Some(scheme) = uri_scheme(uri) else {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_FILE_NEW,
            "object-store Parquet sources require an explicit URI scheme",
        )
        .with_detail("reason", "missing_object_store_scheme")
        .with_detail("path", uri)
        .with_detail("rejected_features", RejectedSourceFeature::NonLocalSource.as_str()));
    };

    if !scheme.eq_ignore_ascii_case("s3") {
        return Err(Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_FILE_NEW,
            "object-store Parquet source scheme is not supported",
        )
        .with_detail("reason", "unsupported_object_store_scheme")
        .with_detail("scheme", scheme.to_ascii_lowercase())
        .with_detail("path", uri)
        .with_detail("rejected_features", RejectedSourceFeature::NonLocalSource.as_str()));
    }

    Ok(())
}

fn uri_scheme(value: &str) -> Option<&str> {
    value.split_once("://").map(|(scheme, _)| scheme).filter(|scheme| !scheme.is_empty())
}

fn object_store_reader_path(uri: &str) -> String {
    if let Some(path) = s3_https_reader_path(uri) {
        return path;
    }
    uri.to_string()
}

fn s3_https_reader_path(uri: &str) -> Option<String> {
    let rest = uri.strip_prefix("s3://")?;
    let (bucket, key) = rest.split_once('/')?;
    if bucket.is_empty() || key.is_empty() {
        return None;
    }
    let region = std::env::var("AWS_REGION")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "us-east-1".to_string());
    Some(format!("https://{bucket}.s3.{region}.amazonaws.com/{key}"))
}

fn validate_unique_strings<'a>(
    names: impl IntoIterator<Item = &'a str>,
    operation: &'static str,
    resource: &'static str,
) -> Result<()> {
    let names = names.into_iter();
    let mut seen = fast_hash_set_with_capacity(names.size_hint().0);
    for (index, name) in names.enumerate() {
        if name.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, operation, "column name must not be empty")
                .with_detail("reason", "empty_name")
                .with_detail("resource", resource)
                .with_detail("position", index.to_string()));
        }
        if !seen.insert(name) {
            return Err(Error::new_with(ErrorCode::DuplicateColumnName, operation, "duplicate read column name")
                .with_column(name.to_string())
                .with_detail("reason", "duplicate_read_column")
                .with_detail("resource", resource)
                .with_detail("position", index.to_string()));
        }
    }
    Ok(())
}

fn validate_unique_named<'a>(
    names: impl IntoIterator<Item = &'a str>,
    operation: &'static str,
    reason: &'static str,
) -> Result<()> {
    let names = names.into_iter();
    let mut seen = fast_hash_set_with_capacity(names.size_hint().0);
    for (index, name) in names.enumerate() {
        if name.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, operation, "column name must not be empty")
                .with_detail("reason", "empty_name")
                .with_detail("position", index.to_string()));
        }
        if !seen.insert(name) {
            return Err(Error::new_with(ErrorCode::DuplicateColumnName, operation, "duplicate column name")
                .with_column(name.to_string())
                .with_detail("reason", reason)
                .with_detail("position", index.to_string()));
        }
    }
    Ok(())
}

fn validate_read_partition_name_collisions(
    read_columns: &[String],
    partition_columns: &[PartitionColumn],
    operation: &'static str,
) -> Result<()> {
    let read_names: FastHashSet<&str> = read_columns.iter().map(String::as_str).collect();
    for (partition_index, column) in partition_columns.iter().enumerate() {
        if read_names.contains(column.name.as_str()) {
            return Err(Error::new_with(
                ErrorCode::DuplicateColumnName,
                operation,
                "partition column name also appears in read columns",
            )
            .with_column(column.name.clone())
            .with_detail("reason", "read_partition_name_collision")
            .with_detail("partition_index", partition_index.to_string())
            .with_detail("partition_column", column.name.clone()));
        }
    }
    Ok(())
}

fn schema_missing_error(operation: &'static str, name: &str) -> Error {
    Error::new_with(ErrorCode::MissingColumn, operation, "source schema does not contain required column")
        .with_column(name.to_string())
        .with_detail("reason", "schema_column_missing")
}

fn validate_matching_row_group_metadata_lengths(
    operation: &'static str,
    stable_id: &str,
    path: &Path,
    reason: &'static str,
    expected_count: usize,
    actual_count: usize,
) -> Result<()> {
    if expected_count != actual_count {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            "row-group metadata vectors must have matching lengths",
        )
        .with_fact(FACT_ACTUAL_COUNT, actual_count)
        .with_fact(FACT_EXPECTED_COUNT, expected_count)
        .with_detail("reason", reason)
        .with_detail("stable_id", stable_id.to_string())
        .with_detail("path", path.display().to_string()));
    }
    Ok(())
}

fn source_chunk_mismatch_error(
    operation: &'static str,
    chunk: &SourceChunk,
    message: &'static str,
    reason: &'static str,
) -> Error {
    Error::new_with(ErrorCode::TypeMismatch, operation, message)
        .with_detail("reason", reason)
        .with_detail("chunk_index", chunk.chunk_index.to_string())
        .with_detail("file_index", chunk.file_index.to_string())
        .with_detail("stable_id", chunk.stable_id.clone())
        .with_detail("path", chunk.path.display().to_string())
}

fn is_supported_partition_dtype(data_type: DataType) -> bool {
    let type_id = data_type.id();
    if !type_id.is_decimal() && data_type.scale() != 0 {
        return false;
    }
    matches!(
        type_id,
        TypeId::Bool8
            | TypeId::Int8
            | TypeId::Int16
            | TypeId::Int32
            | TypeId::Int64
            | TypeId::UInt8
            | TypeId::UInt16
            | TypeId::UInt32
            | TypeId::UInt64
            | TypeId::Float32
            | TypeId::Float64
            | TypeId::String
            | TypeId::Decimal128
            | TypeId::DurationSeconds
            | TypeId::DurationMilliseconds
            | TypeId::DurationMicroseconds
            | TypeId::DurationNanoseconds
    ) || type_id.is_timestamp()
}

fn is_supported_missing_null_dtype(data_type: DataType) -> bool {
    is_supported_partition_dtype(data_type)
}

fn partition_value_matches_dtype(value: &PartitionValue, data_type: DataType) -> bool {
    value.data_type() == data_type
}

fn split_row_groups_for_policy(
    selected_row_groups: &[usize],
    row_group_bytes: &[u64],
    policy: SourceChunkPlanningPolicy,
) -> Vec<Vec<usize>> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0u64;

    for &row_group in selected_row_groups {
        let group_bytes = row_group_bytes[row_group];
        let would_exceed_count = current.len() >= policy.max_row_groups_per_chunk;
        let would_exceed_bytes = policy
            .max_source_chunk_bytes
            .is_some_and(|max_bytes| !current.is_empty() && current_bytes.saturating_add(group_bytes) > max_bytes);

        if would_exceed_count || would_exceed_bytes {
            chunks.push(std::mem::take(&mut current));
            current_bytes = 0;
        }

        current.push(row_group);
        current_bytes = current_bytes.saturating_add(group_bytes);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn validate_selected_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    row_group_count: usize,
) -> Result<()> {
    for (row_group_position, &row_group) in row_groups.iter().enumerate() {
        if row_group >= row_group_count {
            let mut err = Error::new_with(
                ErrorCode::IndexOutOfBounds,
                OP_PLAN_CHUNKS,
                "selected row group index is out of bounds for source metadata",
            )
            .with_fact(FACT_COLUMN_INDEX, row_group)
            .with_detail("reason", "row_group_index_out_of_bounds")
            .with_detail("file_index", file_index.to_string())
            .with_detail("stable_id", file.stable_id.clone())
            .with_detail("path", file.path.display().to_string())
            .with_detail("row_group", row_group.to_string())
            .with_detail("row_group_position", row_group_position.to_string());
            if row_group_count > 0 {
                err = err.with_detail("max_row_group", (row_group_count - 1).to_string());
            }
            return Err(err);
        }
    }
    Ok(())
}

fn validate_source_file_metadata_lengths(file: &LocalParquetFile, file_index: usize) -> Result<()> {
    let expected_count = [
        file.row_group_bytes.as_ref(),
        file.projected_row_group_compressed_bytes.as_ref(),
        file.projected_row_group_uncompressed_bytes.as_ref(),
        file.pure_filter_row_group_compressed_bytes.as_ref(),
        file.pure_filter_row_group_uncompressed_bytes.as_ref(),
        file.row_group_rows.as_ref(),
    ]
    .into_iter()
    .flatten()
    .next()
    .map(Vec::len);
    let Some(expected_count) = expected_count else {
        return Ok(());
    };

    for (field, values) in [
        ("row_group_bytes", file.row_group_bytes.as_ref()),
        ("projected_row_group_compressed_bytes", file.projected_row_group_compressed_bytes.as_ref()),
        ("projected_row_group_uncompressed_bytes", file.projected_row_group_uncompressed_bytes.as_ref()),
        ("pure_filter_row_group_compressed_bytes", file.pure_filter_row_group_compressed_bytes.as_ref()),
        ("pure_filter_row_group_uncompressed_bytes", file.pure_filter_row_group_uncompressed_bytes.as_ref()),
        ("row_group_rows", file.row_group_rows.as_ref()),
    ] {
        if let Some(values) = values {
            if values.len() != expected_count {
                return Err(Error::new_with(
                    ErrorCode::ColumnLengthMismatch,
                    OP_PLAN_CHUNKS,
                    "row-group metadata vectors must have matching lengths",
                )
                .with_fact(FACT_ACTUAL_COUNT, values.len())
                .with_fact(FACT_EXPECTED_COUNT, expected_count)
                .with_detail("reason", "row_group_metadata_count_mismatch")
                .with_detail("field", field.to_string())
                .with_detail("file_index", file_index.to_string())
                .with_detail("stable_id", file.stable_id.clone())
                .with_detail("path", file.path.display().to_string()));
            }
        }
    }

    Ok(())
}

fn validate_chunk_row_groups_for_source_file(
    chunk: &SourceChunk,
    source_file: &LocalParquetFile,
    operation: &'static str,
) -> Result<()> {
    if chunk.reads_all_row_groups && !chunk.row_group_indexes.is_empty() {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk cannot both read all row groups and carry explicit row groups",
            "source_chunk_reads_all_with_explicit_row_groups",
        ));
    }

    if let Some(selected_row_groups) = source_file.row_groups.as_deref() {
        if chunk.reads_all_row_groups {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk reads all row groups but supplied source has an explicit row-group selection",
                "source_chunk_reads_all_for_selected_row_groups",
            )
            .with_detail("source_row_groups", format!("{:?}", selected_row_groups)));
        }

        for (row_group_position, row_group) in chunk.row_group_indexes.iter().copied().enumerate() {
            if !selected_row_groups.contains(&row_group) {
                return Err(source_chunk_mismatch_error(
                    operation,
                    chunk,
                    "source chunk row group is not selected by supplied source file",
                    "source_chunk_row_group_not_selected",
                )
                .with_fact(FACT_COLUMN_INDEX, row_group)
                .with_detail("row_group", row_group.to_string())
                .with_detail("row_group_position", row_group_position.to_string())
                .with_detail("source_row_groups", format!("{:?}", selected_row_groups)));
            }
        }
    }

    let row_group_count = source_file
        .row_group_rows
        .as_ref()
        .or(source_file.row_group_bytes.as_ref())
        .or(source_file.projected_row_group_uncompressed_bytes.as_ref())
        .or(source_file.projected_row_group_compressed_bytes.as_ref())
        .or(source_file.pure_filter_row_group_uncompressed_bytes.as_ref())
        .or(source_file.pure_filter_row_group_compressed_bytes.as_ref())
        .map(Vec::len);
    if let Some(row_group_count) = row_group_count {
        for (row_group_position, row_group) in chunk.row_group_indexes.iter().copied().enumerate() {
            if row_group >= row_group_count {
                let mut err = Error::new_with(
                    ErrorCode::IndexOutOfBounds,
                    operation,
                    "source chunk row group index is out of bounds for supplied source file metadata",
                )
                .with_fact(FACT_COLUMN_INDEX, row_group)
                .with_detail("reason", "source_chunk_row_group_out_of_bounds")
                .with_detail("chunk_index", chunk.chunk_index.to_string())
                .with_detail("file_index", chunk.file_index.to_string())
                .with_detail("stable_id", chunk.stable_id.clone())
                .with_detail("path", chunk.path.display().to_string())
                .with_detail("row_group", row_group.to_string())
                .with_detail("row_group_position", row_group_position.to_string());
                if row_group_count > 0 {
                    err = err.with_detail("max_row_group", (row_group_count - 1).to_string());
                }
                return Err(err);
            }
        }
    }

    Ok(())
}

fn validate_chunk_source_file_contract(
    chunk: &SourceChunk,
    source_file: &LocalParquetFile,
    operation: &'static str,
) -> Result<()> {
    if chunk.source_file_row_groups != source_file.row_groups {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from a different source file row-group selection",
            "source_chunk_source_row_groups_mismatch",
        )
        .with_detail("expected_source_row_groups", format!("{:?}", source_file.row_groups))
        .with_detail("actual_source_row_groups", format!("{:?}", chunk.source_file_row_groups)));
    }

    if chunk.source_file_row_group_bytes != source_file.row_group_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different source file row-group byte metadata",
            "source_chunk_row_group_bytes_metadata_mismatch",
        )
        .with_detail("expected_row_group_bytes", format!("{:?}", source_file.row_group_bytes))
        .with_detail("actual_row_group_bytes", format!("{:?}", chunk.source_file_row_group_bytes)));
    }

    if chunk.source_file_projected_compressed_bytes != source_file.projected_row_group_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different projected compressed byte metadata",
            "source_chunk_projected_compressed_metadata_mismatch",
        )
        .with_detail(
            "expected_projected_compressed_bytes",
            format!("{:?}", source_file.projected_row_group_compressed_bytes),
        )
        .with_detail(
            "actual_projected_compressed_bytes",
            format!("{:?}", chunk.source_file_projected_compressed_bytes),
        ));
    }

    if chunk.source_file_projected_uncompressed_bytes != source_file.projected_row_group_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different projected uncompressed byte metadata",
            "source_chunk_projected_uncompressed_metadata_mismatch",
        )
        .with_detail(
            "expected_projected_uncompressed_bytes",
            format!("{:?}", source_file.projected_row_group_uncompressed_bytes),
        )
        .with_detail(
            "actual_projected_uncompressed_bytes",
            format!("{:?}", chunk.source_file_projected_uncompressed_bytes),
        ));
    }

    if chunk.source_file_pure_filter_compressed_bytes != source_file.pure_filter_row_group_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different pure-filter compressed byte metadata",
            "source_chunk_pure_filter_compressed_metadata_mismatch",
        )
        .with_detail(
            "expected_pure_filter_compressed_bytes",
            format!("{:?}", source_file.pure_filter_row_group_compressed_bytes),
        )
        .with_detail(
            "actual_pure_filter_compressed_bytes",
            format!("{:?}", chunk.source_file_pure_filter_compressed_bytes),
        ));
    }

    if chunk.source_file_pure_filter_uncompressed_bytes != source_file.pure_filter_row_group_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different pure-filter uncompressed byte metadata",
            "source_chunk_pure_filter_uncompressed_metadata_mismatch",
        )
        .with_detail(
            "expected_pure_filter_uncompressed_bytes",
            format!("{:?}", source_file.pure_filter_row_group_uncompressed_bytes),
        )
        .with_detail(
            "actual_pure_filter_uncompressed_bytes",
            format!("{:?}", chunk.source_file_pure_filter_uncompressed_bytes),
        ));
    }

    if chunk.source_file_row_group_rows != source_file.row_group_rows {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk was planned from different source file row-count metadata",
            "source_chunk_row_group_rows_metadata_mismatch",
        )
        .with_detail("expected_row_group_rows", format!("{:?}", source_file.row_group_rows))
        .with_detail("actual_row_group_rows", format!("{:?}", chunk.source_file_row_group_rows)));
    }

    if chunk.known_file_bytes != source_file.file_size_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk file byte metadata does not match supplied source file",
            "source_chunk_file_bytes_metadata_mismatch",
        )
        .with_detail("expected_file_size_bytes", format!("{:?}", source_file.file_size_bytes))
        .with_detail("actual_file_size_bytes", format!("{:?}", chunk.known_file_bytes)));
    }

    let metadata_available = source_file.row_group_bytes.is_some() && source_file.row_group_rows.is_some();
    let selected_row_groups = chunk_metadata_row_groups(chunk, source_file);
    let expected_known_row_group_bytes = if metadata_available {
        checked_row_group_sum(
            source_file.row_group_bytes.as_deref().expect("metadata_available checked row_group_bytes"),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_row_group_bytes != expected_known_row_group_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk row-group byte metadata does not match supplied source file",
            "source_chunk_known_row_group_bytes_mismatch",
        )
        .with_detail("expected_known_row_group_bytes", format!("{expected_known_row_group_bytes:?}"))
        .with_detail("actual_known_row_group_bytes", format!("{:?}", chunk.known_row_group_bytes)));
    }

    let expected_known_projected_compressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.projected_row_group_compressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_projected_compressed_bytes != expected_known_projected_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk projected compressed byte metadata does not match supplied source file",
            "source_chunk_known_projected_compressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_projected_compressed_bytes",
            format!("{expected_known_projected_compressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_projected_compressed_bytes",
            format!("{:?}", chunk.known_projected_compressed_bytes),
        ));
    }

    let expected_known_projected_uncompressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.projected_row_group_uncompressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_projected_uncompressed_bytes != expected_known_projected_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk projected uncompressed byte metadata does not match supplied source file",
            "source_chunk_known_projected_uncompressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_projected_uncompressed_bytes",
            format!("{expected_known_projected_uncompressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_projected_uncompressed_bytes",
            format!("{:?}", chunk.known_projected_uncompressed_bytes),
        ));
    }

    let expected_known_pure_filter_compressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.pure_filter_row_group_compressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_pure_filter_compressed_bytes != expected_known_pure_filter_compressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk pure-filter compressed byte metadata does not match supplied source file",
            "source_chunk_known_pure_filter_compressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_pure_filter_compressed_bytes",
            format!("{expected_known_pure_filter_compressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_pure_filter_compressed_bytes",
            format!("{:?}", chunk.known_pure_filter_compressed_bytes),
        ));
    }

    let expected_known_pure_filter_uncompressed_bytes = if metadata_available {
        optional_checked_row_group_sum(
            source_file.pure_filter_row_group_uncompressed_bytes.as_deref(),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_pure_filter_uncompressed_bytes != expected_known_pure_filter_uncompressed_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk pure-filter uncompressed byte metadata does not match supplied source file",
            "source_chunk_known_pure_filter_uncompressed_bytes_mismatch",
        )
        .with_detail(
            "expected_known_pure_filter_uncompressed_bytes",
            format!("{expected_known_pure_filter_uncompressed_bytes:?}"),
        )
        .with_detail(
            "actual_known_pure_filter_uncompressed_bytes",
            format!("{:?}", chunk.known_pure_filter_uncompressed_bytes),
        ));
    }

    let expected_estimated_chunk_bytes = if metadata_available {
        expected_known_projected_uncompressed_bytes.or(expected_known_row_group_bytes)
    } else {
        source_file.file_size_bytes
    };
    if chunk.estimated_chunk_bytes != expected_estimated_chunk_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk byte estimate does not match supplied source file metadata",
            "source_chunk_estimated_bytes_mismatch",
        )
        .with_detail("expected_estimated_chunk_bytes", format!("{expected_estimated_chunk_bytes:?}"))
        .with_detail("actual_estimated_chunk_bytes", format!("{:?}", chunk.estimated_chunk_bytes)));
    }

    let expected_estimated_source_read_bytes = if metadata_available {
        expected_known_projected_uncompressed_bytes
            .zip(expected_known_pure_filter_uncompressed_bytes)
            .and_then(|(projected, filter)| projected.checked_add(filter))
            .or(expected_known_row_group_bytes)
            .or(expected_estimated_chunk_bytes)
    } else {
        source_file.file_size_bytes
    };
    if chunk.estimated_source_read_bytes != expected_estimated_source_read_bytes {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk source-read byte estimate does not match supplied source file metadata",
            "source_chunk_estimated_source_read_bytes_mismatch",
        )
        .with_detail("expected_estimated_source_read_bytes", format!("{expected_estimated_source_read_bytes:?}"))
        .with_detail("actual_estimated_source_read_bytes", format!("{:?}", chunk.estimated_source_read_bytes)));
    }

    let expected_size_estimate_based = !metadata_available && source_file.file_size_bytes.is_some();
    if chunk.size_estimate_based != expected_size_estimate_based {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk size-estimate mode does not match supplied source file metadata",
            "source_chunk_size_estimate_based_mismatch",
        )
        .with_detail("expected_size_estimate_based", expected_size_estimate_based.to_string())
        .with_detail("actual_size_estimate_based", chunk.size_estimate_based.to_string()));
    }

    let expected_known_row_count = if metadata_available {
        checked_row_group_sum(
            source_file.row_group_rows.as_deref().expect("metadata_available checked row_group_rows"),
            &selected_row_groups,
        )
    } else {
        None
    };
    if chunk.known_row_count != expected_known_row_count {
        return Err(source_chunk_mismatch_error(
            operation,
            chunk,
            "source chunk row-count metadata does not match supplied source file",
            "source_chunk_known_row_count_mismatch",
        )
        .with_detail("expected_known_row_count", format!("{expected_known_row_count:?}"))
        .with_detail("actual_known_row_count", format!("{:?}", chunk.known_row_count)));
    }

    Ok(())
}

fn chunk_metadata_row_groups(chunk: &SourceChunk, source_file: &LocalParquetFile) -> Vec<usize> {
    if !chunk.reads_all_row_groups {
        return chunk.row_group_indexes.clone();
    }

    chunk_metadata_row_groups_for_source_file(source_file)
}

fn chunk_metadata_row_groups_for_source_file(source_file: &LocalParquetFile) -> Vec<usize> {
    let row_group_count = source_file
        .row_group_rows
        .as_ref()
        .or(source_file.row_group_bytes.as_ref())
        .or(source_file.projected_row_group_uncompressed_bytes.as_ref())
        .or(source_file.projected_row_group_compressed_bytes.as_ref())
        .or(source_file.pure_filter_row_group_uncompressed_bytes.as_ref())
        .or(source_file.pure_filter_row_group_compressed_bytes.as_ref())
        .map(Vec::len);
    row_group_count.map_or_else(Vec::new, |count| (0..count).collect())
}

fn checked_row_group_sum(values: &[u64], row_groups: &[usize]) -> Option<u64> {
    row_groups.iter().try_fold(0u64, |acc, &row_group| acc.checked_add(values[row_group]))
}

fn optional_checked_row_group_sum(values: Option<&[u64]>, row_groups: &[usize]) -> Option<u64> {
    checked_row_group_sum(values?, row_groups)
}

fn checked_row_group_row_count(
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    row_group_rows: &[u64],
    row_groups: &[usize],
) -> Result<u64> {
    let mut total = 0u64;
    for &row_group in row_groups {
        total = total.checked_add(row_group_rows[row_group]).ok_or_else(|| {
            Error::new_with(ErrorCode::UnsupportedOperation, OP_ROW_COUNT_TO_I64, "source row count exceeds i64 range")
                .with_detail("reason", "row_count_total_overflow")
                .with_detail("chunk_index", chunk_index.to_string())
                .with_detail("file_index", file_index.to_string())
                .with_detail("stable_id", file.stable_id.clone())
                .with_detail("path", file.path.display().to_string())
        })?;
    }
    source_row_count_to_i64_with_context(total, Some((file, file_index, chunk_index)))?;
    Ok(total)
}

fn known_total_row_count(chunks: &[SourceChunk]) -> Result<Option<u64>> {
    let mut total = 0u64;
    for chunk in chunks {
        let Some(row_count) = chunk.known_row_count else {
            return Ok(None);
        };
        total = total.checked_add(row_count).ok_or_else(|| {
            Error::new_with(
                ErrorCode::UnsupportedOperation,
                OP_PLAN_CHUNKS,
                "planned source row count exceeds the supported i64 range",
            )
            .with_detail("reason", "row_count_exceeds_i64")
        })?;
    }
    source_row_count_to_i64_with_context(total, None)?;
    Ok(Some(total))
}

fn missing_row_group_row_count_metadata_error(file: &LocalParquetFile, file_index: usize) -> Error {
    Error::new_with(
        ErrorCode::ForeignInvalidArgument,
        OP_PLAN_CHUNKS,
        "metadata row-count source requires row-group row-count metadata",
    )
    .with_detail("reason", "missing_row_group_row_count_metadata")
    .with_detail("file_index", file_index.to_string())
    .with_detail("stable_id", file.stable_id.clone())
    .with_detail("path", file.path.display().to_string())
}

fn source_row_group_index_to_i32_with_context(
    row_group: usize,
    chunk: Option<&SourceChunk>,
    position: Option<usize>,
) -> Result<i32> {
    i32::try_from(row_group).map_err(|_| {
        let mut err = Error::new_with(
            ErrorCode::UnsupportedOperation,
            OP_ROW_GROUP_TO_I32,
            "row group index exceeds libcudf i32 range",
        )
        .with_detail("reason", "row_group_index_exceeds_i32")
        .with_detail("row_group", row_group.to_string());
        if let Some(position) = position {
            err = err.with_detail("row_group_position", position.to_string());
        }
        if let Some(chunk) = chunk {
            err = err
                .with_detail("chunk_index", chunk.chunk_index.to_string())
                .with_detail("file_index", chunk.file_index.to_string())
                .with_detail("stable_id", chunk.stable_id.clone())
                .with_detail("path", chunk.path.display().to_string());
        }
        err
    })
}

fn source_row_count_to_i64_with_context(
    row_count: u64,
    context: Option<(&LocalParquetFile, usize, usize)>,
) -> Result<i64> {
    i64::try_from(row_count).map_err(|_| {
        let mut err =
            Error::new_with(ErrorCode::UnsupportedOperation, OP_ROW_COUNT_TO_I64, "source row count exceeds i64 range")
                .with_detail("reason", "row_count_exceeds_i64")
                .with_detail("row_count", row_count.to_string());
        if let Some((file, file_index, chunk_index)) = context {
            err = err
                .with_detail("chunk_index", chunk_index.to_string())
                .with_detail("file_index", file_index.to_string())
                .with_detail("stable_id", file.stable_id.clone())
                .with_detail("path", file.path.display().to_string());
        }
        err
    })
}

fn dtype_label(data_type: DataType) -> String {
    if data_type.id().is_decimal() || data_type.scale() != 0 {
        format!("{}(scale={})", data_type.id(), data_type.scale())
    } else {
        data_type.id().to_string()
    }
}
