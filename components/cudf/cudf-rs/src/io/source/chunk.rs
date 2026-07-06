use super::*;

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
    pub(super) max_source_chunk_bytes: Option<u64>,
    pub(super) max_row_groups_per_chunk: usize,
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
    pub(super) chunks: Vec<SourceChunk>,
    pub(super) row_group_metadata_available: bool,
    pub(super) size_estimate_based: bool,
    pub(super) metadata_unavailable_files: usize,
    pub(super) known_total_row_count: Option<u64>,
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
    pub(super) chunk_index: usize,
    pub(super) file_index: usize,
    pub(super) location: ParquetSourceLocation,
    pub(super) path: PathBuf,
    pub(super) stable_id: String,
    pub(super) row_group_indexes: Vec<usize>,
    pub(super) reads_all_row_groups: bool,
    pub(super) known_file_bytes: Option<u64>,
    pub(super) known_row_group_bytes: Option<u64>,
    pub(super) known_projected_compressed_bytes: Option<u64>,
    pub(super) known_projected_uncompressed_bytes: Option<u64>,
    pub(super) known_pure_filter_compressed_bytes: Option<u64>,
    pub(super) known_pure_filter_uncompressed_bytes: Option<u64>,
    pub(super) estimated_chunk_bytes: Option<u64>,
    pub(super) estimated_source_read_bytes: Option<u64>,
    pub(super) source_file_row_groups: Option<Vec<usize>>,
    pub(super) source_file_row_group_bytes: Option<Vec<u64>>,
    pub(super) source_file_projected_compressed_bytes: Option<Vec<u64>>,
    pub(super) source_file_projected_uncompressed_bytes: Option<Vec<u64>>,
    pub(super) source_file_pure_filter_compressed_bytes: Option<Vec<u64>>,
    pub(super) source_file_pure_filter_uncompressed_bytes: Option<Vec<u64>>,
    pub(super) source_file_row_group_rows: Option<Vec<u64>>,
    pub(super) read_columns: Vec<String>,
    pub(super) partition_values: Vec<PartitionValue>,
    pub(super) zero_column_count: bool,
    pub(super) known_row_count: Option<u64>,
    pub(super) size_estimate_based: bool,
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

    /// Path-shaped reader string for this chunk's file.
    ///
    /// Local sources contain filesystem paths. Object-store sources preserve the
    /// caller-provided URI string so libcudf/KvikIO owns remote path handling.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Path or URI string passed to libcudf's Parquet reader.
    pub fn reader_path(&self) -> String {
        self.location.reader_path()
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
