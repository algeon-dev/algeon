use super::*;

/// Explicit Parquet source location represented by the checked source
/// contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParquetSourceLocation {
    LocalPath(PathBuf),
    ObjectStoreUri(crate::io::parquet::RemoteParquetSource),
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
        Ok(Self::ObjectStoreUri(crate::io::parquet::RemoteParquetSource::s3(uri)?))
    }

    pub fn reader_path(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::ObjectStoreUri(source) => source.reader_path().to_string(),
        }
    }

    pub fn remote_source(&self) -> Option<&crate::io::parquet::RemoteParquetSource> {
        match self {
            Self::LocalPath(_) => None,
            Self::ObjectStoreUri(source) => Some(source),
        }
    }

    pub fn read_capability(&self) -> &'static str {
        match self {
            Self::LocalPath(_) => "local_file_system",
            Self::ObjectStoreUri(source)
                if uri_scheme(source.uri()).is_some_and(|scheme| scheme.eq_ignore_ascii_case("s3")) =>
            {
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
            Self::ObjectStoreUri(source) => f.write_str(source.uri()),
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
    pub(super) kind: LocalParquetSourceKind,
    pub(super) files: Vec<LocalParquetFile>,
    pub(super) schema: SourceSchema,
    pub(super) read_columns: Vec<String>,
    pub(super) output_columns: Vec<SourceOutputColumn>,
    pub(super) required_columns: Vec<SourceRequiredColumn>,
    pub(super) partition_columns: Vec<PartitionColumn>,
    pub(super) reader_predicate: Option<Arc<AstExpression>>,
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

    pub(super) fn validate(&self, operation: &'static str) -> Result<()> {
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

        if chunk.location != source_file.location {
            return Err(source_chunk_mismatch_error(
                operation,
                chunk,
                "source chunk location contract does not match supplied source file",
                "source_chunk_location_mismatch",
            )
            .with_detail("expected_location", format!("{:?}", source_file.location))
            .with_detail("actual_location", format!("{:?}", chunk.location)));
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

/// File-local facts for a local Parquet source.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetFile {
    pub(super) location: ParquetSourceLocation,
    pub(super) path: PathBuf,
    pub(super) stable_id: String,
    pub(super) file_size_bytes: Option<u64>,
    pub(super) row_groups: Option<Vec<usize>>,
    pub(super) row_group_bytes: Option<Vec<u64>>,
    pub(super) projected_row_group_compressed_bytes: Option<Vec<u64>>,
    pub(super) projected_row_group_uncompressed_bytes: Option<Vec<u64>>,
    pub(super) pure_filter_row_group_compressed_bytes: Option<Vec<u64>>,
    pub(super) pure_filter_row_group_uncompressed_bytes: Option<Vec<u64>>,
    pub(super) row_group_rows: Option<Vec<u64>>,
    pub(super) reader_decimal_types: Vec<ParquetReaderDecimalType>,
    pub(super) partition_values: Vec<PartitionValue>,
}

/// Per-file Parquet decimal reader type for one decoded read column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParquetReaderDecimalType {
    column_name: String,
    data_type: DataType,
}

impl ParquetReaderDecimalType {
    pub fn new(column_name: impl Into<String>, data_type: DataType) -> Result<Self> {
        let column_name = checked_name(column_name.into(), OP_FILE_METADATA, "reader decimal column")?;
        if !data_type.id().is_decimal() {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OP_FILE_METADATA,
                "reader decimal type metadata requires a decimal dtype",
            )
            .with_column(column_name)
            .with_detail("reason", "reader_decimal_type_not_decimal")
            .with_detail("data_type_id", data_type.id().to_string()));
        }
        Ok(Self { column_name, data_type })
    }

    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }
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
            reader_decimal_types: Vec::new(),
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
        let stable_id = stable_id.into();
        if stable_id.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "file stable_id must not be empty")
                .with_detail("reason", "empty_stable_id")
                .with_detail("path", uri));
        }

        Ok(Self {
            location,
            path: PathBuf::from(&uri),
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            projected_row_group_compressed_bytes: None,
            projected_row_group_uncompressed_bytes: None,
            pure_filter_row_group_compressed_bytes: None,
            pure_filter_row_group_uncompressed_bytes: None,
            row_group_rows: None,
            reader_decimal_types: Vec::new(),
            partition_values: partition_values.into_iter().collect(),
        })
    }

    /// Explicit source location.
    pub fn location(&self) -> &ParquetSourceLocation {
        &self.location
    }

    /// Typed remote Parquet source when this file is an object-store URI.
    pub fn remote_source(&self) -> Option<&crate::io::parquet::RemoteParquetSource> {
        self.location.remote_source()
    }

    /// Path string passed to libcudf's Parquet reader.
    pub fn reader_path(&self) -> String {
        self.location.reader_path()
    }

    /// Attach a typed remote Parquet profile and read policy.
    pub fn with_remote_parquet_contract(
        mut self,
        profile: crate::io::parquet::RemoteParquetProfile,
        read_mode: crate::io::parquet::RemoteReadMode,
    ) -> Result<Self> {
        let ParquetSourceLocation::ObjectStoreUri(source) = &mut self.location else {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                OP_FILE_REMOTE_CONTRACT,
                "remote Parquet contracts can only be attached to object-store sources",
            )
            .with_detail("reason", "non_remote_source")
            .with_detail("path", self.path.display().to_string()));
        };
        let updated = source.clone().with_profile(profile).with_read_mode(read_mode);
        updated.validate_contract_for_current_reader(OP_FILE_REMOTE_CONTRACT)?;
        *source = updated;
        Ok(self)
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

    /// Attach per-read-column decimal storage types proven from Parquet metadata.
    pub fn with_reader_decimal_types<I>(mut self, reader_decimal_types: I) -> Result<Self>
    where I: IntoIterator<Item = ParquetReaderDecimalType> {
        self.reader_decimal_types = reader_decimal_types.into_iter().collect();
        validate_unique_strings(
            self.reader_decimal_types.iter().map(|dtype| dtype.column_name.as_str()),
            OP_FILE_METADATA,
            "reader_decimal_column",
        )?;
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

    /// Per-read-column decimal storage types proven from Parquet metadata.
    pub fn reader_decimal_types(&self) -> &[ParquetReaderDecimalType] {
        &self.reader_decimal_types
    }

    /// Per-file partition values in [`LocalParquetSource::partition_columns`]
    /// order.
    pub fn partition_values(&self) -> &[PartitionValue] {
        &self.partition_values
    }
}
