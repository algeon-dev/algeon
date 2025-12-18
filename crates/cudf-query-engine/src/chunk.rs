use std::path::PathBuf;

use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::NodeId,
    runtime::MemoryControlPolicy,
    source::{LocalParquetFile, LocalParquetSource},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceChunkPlanningPolicy {
    pub max_source_chunk_bytes: Option<u64>,
    pub max_row_groups_per_chunk: usize,
}

impl SourceChunkPlanningPolicy {
    pub const fn new(max_source_chunk_bytes: Option<u64>, max_row_groups_per_chunk: usize) -> Self {
        Self { max_source_chunk_bytes, max_row_groups_per_chunk }
    }

    fn effective_max_row_groups_per_chunk(self) -> usize {
        self.max_row_groups_per_chunk.max(1)
    }
}

impl Default for SourceChunkPlanningPolicy {
    fn default() -> Self {
        Self { max_source_chunk_bytes: None, max_row_groups_per_chunk: usize::MAX }
    }
}

impl From<MemoryControlPolicy> for SourceChunkPlanningPolicy {
    fn from(policy: MemoryControlPolicy) -> Self {
        match policy {
            MemoryControlPolicy::Unbounded => Self::default(),
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, max_row_groups_per_chunk, .. } => {
                Self::new(Some(max_source_chunk_bytes), max_row_groups_per_chunk)
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChunkPlan {
    pub chunks: Vec<SourceChunk>,
    pub source_node_id: NodeId,
    pub row_group_metadata_available: bool,
    pub size_estimate_based: bool,
    pub metadata_unavailable_files: usize,
}

impl SourceChunkPlan {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn row_group_count(&self) -> usize {
        self.chunks.iter().map(SourceChunk::row_group_count).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChunk {
    pub source_node_id: NodeId,
    pub chunk_index: usize,
    pub file_index: usize,
    pub path: PathBuf,
    pub row_group_indexes: Vec<usize>,
    pub reads_all_row_groups: bool,
    pub known_file_bytes: Option<u64>,
    pub known_row_group_bytes: Option<u64>,
    pub estimated_chunk_bytes: Option<u64>,
    pub read_columns: Vec<String>,
    pub size_estimate_based: bool,
}

impl SourceChunk {
    pub fn row_group_count(&self) -> usize {
        self.row_group_indexes.len()
    }
}

pub fn plan_local_parquet_source_chunks(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
) -> Result<SourceChunkPlan> {
    validate_chunk_planning_source(source)?;
    validate_chunk_planning_policy(policy)?;

    let mut chunks = Vec::new();
    let mut saw_metadata = false;
    let mut saw_missing_metadata = false;
    let mut metadata_unavailable_files = 0;

    for (file_index, file) in source.files.iter().enumerate() {
        let metadata = read_parquet_metadata(file);
        saw_metadata |= metadata.is_some();
        if metadata.is_none() {
            saw_missing_metadata = true;
            metadata_unavailable_files += 1;
        }

        let row_groups = planned_row_groups(file, file_index, metadata.as_ref())?;
        let Some(row_groups) = row_groups else {
            chunks.push(single_all_row_groups_chunk(source_node_id, source, file, file_index, chunks.len()));
            continue;
        };

        append_row_group_chunks(
            source_node_id,
            source,
            file,
            file_index,
            &row_groups,
            metadata.as_ref(),
            policy,
            &mut chunks,
        );
    }

    Ok(SourceChunkPlan {
        chunks,
        source_node_id,
        row_group_metadata_available: saw_metadata && !saw_missing_metadata,
        size_estimate_based: saw_missing_metadata,
        metadata_unavailable_files,
    })
}

fn validate_chunk_planning_policy(policy: SourceChunkPlanningPolicy) -> Result<()> {
    if policy.max_row_groups_per_chunk == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_chunk.policy",
            "source chunk planning requires max_row_groups_per_chunk to be greater than zero",
        )
        .with_fact("field", "max_row_groups_per_chunk")
        .with_fact("reason", "zero"));
    }

    Ok(())
}

fn validate_chunk_planning_source(source: &LocalParquetSource) -> Result<()> {
    if source.files.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_chunk.plan",
            "local parquet source must contain at least one file for chunk planning",
        )
        .with_fact("reason", "empty_files"));
    }

    if source.read_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_chunk.plan",
            "local parquet source must name at least one required read column for chunk planning",
        )
        .with_fact("reason", "empty_read_columns"));
    }

    for (file_index, file) in source.files.iter().enumerate() {
        if file.path.as_os_str().is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_chunk.plan",
                "local parquet source path must not be empty for chunk planning",
            )
            .with_fact("file_index", file_index));
        }
    }

    Ok(())
}

fn read_parquet_metadata(file: &LocalParquetFile) -> Option<RowGroupMetadata> {
    let reader = SerializedFileReader::try_from(file.path.as_path()).ok()?;
    let metadata = reader.metadata();
    let row_group_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.compressed_size()).ok())
        .collect::<Option<Vec<_>>>()?;
    Some(RowGroupMetadata { row_group_bytes })
}

fn planned_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    metadata: Option<&RowGroupMetadata>,
) -> Result<Option<Vec<usize>>> {
    if let Some(row_groups) = &file.row_groups
        && !row_groups.is_empty()
    {
        validate_selected_row_groups(file, file_index, row_groups, metadata)?;
        return Ok(Some(row_groups.clone()));
    }

    let Some(metadata) = metadata else {
        return Ok(None);
    };

    Ok(Some((0..metadata.row_group_bytes.len()).collect()))
}

fn validate_selected_row_groups(
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
) -> Result<()> {
    let Some(metadata) = metadata else {
        return Ok(());
    };

    for (row_group_position, row_group) in row_groups.iter().enumerate() {
        if *row_group >= metadata.row_group_bytes.len() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_chunk.plan",
                "selected parquet row group is outside file metadata",
            )
            .with_fact("path", file.path.display())
            .with_fact("file_index", file_index)
            .with_fact("row_group_position", row_group_position)
            .with_fact("row_group", row_group)
            .with_fact("max_row_group", metadata.row_group_bytes.len().saturating_sub(1)));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_row_group_chunks(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    row_groups: &[usize],
    metadata: Option<&RowGroupMetadata>,
    policy: SourceChunkPlanningPolicy,
    chunks: &mut Vec<SourceChunk>,
) {
    let max_row_groups = policy.effective_max_row_groups_per_chunk();
    let mut current = PendingChunk::default();

    for row_group in row_groups {
        let row_group_bytes = metadata.and_then(|metadata| metadata.row_group_bytes.get(*row_group).copied());
        if current.should_flush_before(row_group_bytes, max_row_groups, policy.max_source_chunk_bytes) {
            push_pending_chunk(source_node_id, source, file, file_index, row_groups.len(), &mut current, chunks);
        }
        current.push(*row_group, row_group_bytes);
    }

    push_pending_chunk(source_node_id, source, file, file_index, row_groups.len(), &mut current, chunks);
}

fn push_pending_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    planned_row_group_count: usize,
    pending: &mut PendingChunk,
    chunks: &mut Vec<SourceChunk>,
) {
    if pending.row_groups.is_empty() {
        return;
    }

    let known_row_group_bytes = pending.known_bytes();
    let estimated_chunk_bytes = known_row_group_bytes
        .or_else(|| proportional_bytes(file.file_size_bytes, pending.row_groups.len(), planned_row_group_count));
    chunks.push(SourceChunk {
        source_node_id,
        chunk_index: chunks.len(),
        file_index,
        path: file.path.clone(),
        row_group_indexes: std::mem::take(&mut pending.row_groups),
        reads_all_row_groups: false,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes,
        estimated_chunk_bytes,
        read_columns: source.read_columns.clone(),
        size_estimate_based: known_row_group_bytes.is_none(),
    });
    pending.clear();
}

fn single_all_row_groups_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
) -> SourceChunk {
    SourceChunk {
        source_node_id,
        chunk_index,
        file_index,
        path: file.path.clone(),
        row_group_indexes: Vec::new(),
        reads_all_row_groups: true,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes: None,
        estimated_chunk_bytes: file.file_size_bytes,
        read_columns: source.read_columns.clone(),
        size_estimate_based: true,
    }
}

fn proportional_bytes(total_bytes: Option<u64>, child_row_groups: usize, parent_row_groups: usize) -> Option<u64> {
    let total_bytes = total_bytes?;
    if parent_row_groups == 0 {
        return Some(total_bytes);
    }

    let numerator = u128::from(total_bytes) * child_row_groups as u128;
    let denominator = parent_row_groups as u128;
    Some(numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64)
}

#[derive(Debug)]
struct RowGroupMetadata {
    row_group_bytes: Vec<u64>,
}

#[derive(Debug, Default)]
struct PendingChunk {
    row_groups: Vec<usize>,
    known_bytes: u64,
    has_unknown_bytes: bool,
}

impl PendingChunk {
    fn push(&mut self, row_group: usize, row_group_bytes: Option<u64>) {
        self.row_groups.push(row_group);
        if let Some(row_group_bytes) = row_group_bytes {
            self.known_bytes = self.known_bytes.saturating_add(row_group_bytes);
        } else {
            self.has_unknown_bytes = true;
        }
    }

    fn should_flush_before(
        &self,
        next_row_group_bytes: Option<u64>,
        max_row_groups: usize,
        max_source_chunk_bytes: Option<u64>,
    ) -> bool {
        if self.row_groups.is_empty() {
            return false;
        }
        if self.row_groups.len() >= max_row_groups {
            return true;
        }

        let Some(max_source_chunk_bytes) = max_source_chunk_bytes else {
            return false;
        };
        let Some(next_row_group_bytes) = next_row_group_bytes else {
            return false;
        };

        self.known_bytes.saturating_add(next_row_group_bytes) > max_source_chunk_bytes
    }

    fn known_bytes(&self) -> Option<u64> {
        (!self.has_unknown_bytes).then_some(self.known_bytes)
    }

    fn clear(&mut self) {
        self.row_groups.clear();
        self.known_bytes = 0;
        self.has_unknown_bytes = false;
    }
}
