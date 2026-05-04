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
//! files with direct by-name read-column mapping and explicit metadata.
//!
//! Future source capabilities should be additive and explicitly named.
//! Predicate pruning, object stores, byte ranges, row-level selection, Iceberg
//! delete files, Iceberg field-id or name mapping, partition transforms, and
//! GDS/KvikIO direct-read policy all need their own source facts before callers
//! may rely on them. Phase 1 local sources therefore reject or keep those
//! planner facts downstream instead of silently treating them as ordinary local
//! full-file reads.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use crate::{
    core::{
        error::{Error, ErrorCode, FACT_ACTUAL_COUNT, FACT_COLUMN_INDEX, FACT_EXPECTED_COUNT, Result},
        types::{DataType, TypeId},
    },
    internal::hash::{FastHashSet, fast_hash_set_with_capacity},
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

/// A complete checked local Parquet source contract.
///
/// Fields are private so callers cannot mutate the plan into an inconsistent
/// shape after construction. Use [`LocalParquetSource::data`] for normal data
/// reads and [`LocalParquetSource::zero_column_count`] for metadata-only count
/// sources.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalParquetSource {
    kind: LocalParquetSourceKind,
    files: Vec<LocalParquetFile>,
    schema: SourceSchema,
    read_columns: Vec<String>,
    output_columns: Vec<SourceOutputColumn>,
    required_columns: Vec<SourceRequiredColumn>,
    partition_columns: Vec<PartitionColumn>,
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
            }
        }
        Ok(())
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
    path: PathBuf,
    stable_id: String,
    row_group_indexes: Vec<usize>,
    reads_all_row_groups: bool,
    known_file_bytes: Option<u64>,
    known_row_group_bytes: Option<u64>,
    estimated_chunk_bytes: Option<u64>,
    read_columns: Vec<String>,
    partition_values: Vec<PartitionValue>,
    zero_column_count: bool,
    known_row_count: Option<u64>,
    size_estimate_based: bool,
}

impl SourceChunk {
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

    /// Estimated chunk bytes used for source scheduling.
    pub fn estimated_chunk_bytes(&self) -> Option<u64> {
        self.estimated_chunk_bytes
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
                path: file.path.clone(),
                stable_id: file.stable_id.clone(),
                row_group_indexes: file.row_groups.clone().unwrap_or_default(),
                reads_all_row_groups: file.row_groups.is_none(),
                known_file_bytes: file.file_size_bytes,
                known_row_group_bytes: None,
                estimated_chunk_bytes: file.file_size_bytes,
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

        let chunk_groups = split_row_groups_for_policy(&selected_row_groups, row_group_bytes, policy);
        let single_all_row_group_chunk = reads_all_from_source && chunk_groups.len() == 1;
        for row_groups in chunk_groups {
            let known_row_group_bytes = checked_row_group_sum(row_group_bytes, &row_groups);
            let known_row_count =
                checked_row_group_row_count(file, file_index, chunks.len(), row_group_rows, &row_groups)?;

            let reads_all_row_groups = single_all_row_group_chunk && row_groups.len() == row_group_rows.len();
            chunks.push(SourceChunk {
                chunk_index: chunks.len(),
                file_index,
                path: file.path.clone(),
                stable_id: file.stable_id.clone(),
                row_group_indexes: if reads_all_row_groups { Vec::new() } else { row_groups.clone() },
                reads_all_row_groups,
                known_file_bytes: file.file_size_bytes,
                known_row_group_bytes,
                estimated_chunk_bytes: known_row_group_bytes,
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
    path: PathBuf,
    stable_id: String,
    file_size_bytes: Option<u64>,
    row_groups: Option<Vec<usize>>,
    row_group_bytes: Option<Vec<u64>>,
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
        let path = path.as_ref().to_path_buf();
        validate_local_path(&path)?;
        let stable_id = stable_id.into();
        if stable_id.is_empty() {
            return Err(Error::new_with(ErrorCode::EmptyInput, OP_FILE_NEW, "file stable_id must not be empty")
                .with_detail("reason", "empty_stable_id")
                .with_detail("path", path.display().to_string()));
        }

        Ok(Self {
            path,
            stable_id,
            file_size_bytes: None,
            row_groups: None,
            row_group_bytes: None,
            row_group_rows: None,
            partition_values: partition_values.into_iter().collect(),
        })
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
        if row_group_bytes.len() != row_group_rows.len() {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                OP_FILE_METADATA,
                "row-group metadata vectors must have matching lengths",
            )
            .with_fact(FACT_ACTUAL_COUNT, row_group_rows.len())
            .with_fact(FACT_EXPECTED_COUNT, row_group_bytes.len())
            .with_detail("reason", "row_group_metadata_count_mismatch")
            .with_detail("stable_id", self.stable_id.clone())
            .with_detail("path", self.path.display().to_string()));
        }
        self.row_group_bytes = Some(row_group_bytes);
        self.row_group_rows = Some(row_group_rows);
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

/// A reference to a decoded read column, partition column, or synthetic count.
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
    /// `nullable` records source schema truth. Phase 1 partition values are
    /// still non-null, so a nullable partition column may receive non-null
    /// values but null partition values are not yet represented.
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

/// Non-null partition value supported by the Phase 1 source contract.
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionValue {
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
    /// UTF-8 string value.
    String(String),
}

impl PartitionValue {
    /// cuDF dtype represented by this non-null partition value.
    pub fn data_type(&self) -> DataType {
        DataType::new(match self {
            Self::Boolean(_) => TypeId::Bool8,
            Self::Int8(_) => TypeId::Int8,
            Self::Int16(_) => TypeId::Int16,
            Self::Int32(_) => TypeId::Int32,
            Self::Int64(_) => TypeId::Int64,
            Self::UInt8(_) => TypeId::UInt8,
            Self::UInt16(_) => TypeId::UInt16,
            Self::UInt32(_) => TypeId::UInt32,
            Self::UInt64(_) => TypeId::UInt64,
            Self::Float32(_) => TypeId::Float32,
            Self::Float64(_) => TypeId::Float64,
            Self::String(_) => TypeId::String,
        })
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

        Self {
            format: SourceFormat::Parquet,
            direct_device_read: true,
            file_count: source.files.len(),
            ranged_file_count: 0,
            known_bytes,
            selected_row_groups,
            predicate_disposition: PredicateDisposition::Absent,
            byte_range_disposition: ByteRangeDisposition::FullFiles,
            partition_column_disposition: if source.partition_columns.is_empty() {
                PartitionColumnDisposition::Absent
            } else {
                PartitionColumnDisposition::Represented
            },
            zero_column_count_disposition,
            schema_mapping_disposition: SchemaMappingDisposition::DirectByName,
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

fn is_supported_partition_dtype(data_type: DataType) -> bool {
    matches!(
        data_type.id(),
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
    )
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

fn checked_row_group_sum(values: &[u64], row_groups: &[usize]) -> Option<u64> {
    row_groups.iter().try_fold(0u64, |acc, &row_group| acc.checked_add(values[row_group]))
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
    if data_type.id().is_decimal() {
        format!("{}(scale={})", data_type.id(), data_type.scale())
    } else {
        data_type.id().to_string()
    }
}
