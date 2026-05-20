use std::{
    collections::{BTreeSet, HashMap},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use parquet::file::{
    metadata::ParquetMetaData,
    reader::{FileReader, SerializedFileReader},
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::LiteralValue,
    plan::NodeId,
    runtime::MemoryControlPolicy,
    source::{LocalParquetFile, LocalParquetSource, ParquetSourceLocation},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceChunkPlanningPolicy {
    pub max_source_chunk_bytes: Option<u64>,
    pub max_row_groups_per_chunk: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceTaskMemoryModel {
    source_read_slots: u64,
    source_adjacent_operator_slots: u64,
    downstream_operator_slots: u64,
    retained_output_slots: u64,
    source_read_preflight_slack_slots: u64,
}

impl SourceTaskMemoryModel {
    /// Source -> optional source-local unary stages -> aggregate.
    pub const fn aggregate_pipeline(has_source_adjacent_stages: bool) -> Self {
        Self {
            source_read_slots: 1,
            // Source-adjacent filter/projection chains can retain a large input,
            // an operator work region, and the transformed output before the
            // downstream aggregate takes ownership.
            source_adjacent_operator_slots: if has_source_adjacent_stages { 3 } else { 0 },
            downstream_operator_slots: 1,
            retained_output_slots: 1,
            source_read_preflight_slack_slots: 1,
        }
    }

    /// Source -> optional source-local unary stages -> join build/probe.
    pub const fn join_leaf_stream(has_source_adjacent_stages: bool) -> Self {
        Self {
            source_read_slots: 1,
            source_adjacent_operator_slots: if has_source_adjacent_stages { 1 } else { 0 },
            downstream_operator_slots: 1,
            retained_output_slots: 2,
            source_read_preflight_slack_slots: 1,
        }
    }

    pub const fn source_read_slots(self) -> u64 {
        self.source_read_slots
    }

    pub const fn source_adjacent_operator_slots(self) -> u64 {
        self.source_adjacent_operator_slots
    }

    pub const fn downstream_operator_slots(self) -> u64 {
        self.downstream_operator_slots
    }

    pub const fn retained_output_slots(self) -> u64 {
        self.retained_output_slots
    }

    pub const fn source_read_preflight_slack_slots(self) -> u64 {
        self.source_read_preflight_slack_slots
    }

    pub const fn live_slot_count(self) -> u64 {
        self.source_read_slots
            + self.source_adjacent_operator_slots
            + self.downstream_operator_slots
            + self.retained_output_slots
            + self.source_read_preflight_slack_slots
    }
}

impl SourceChunkPlanningPolicy {
    pub const fn new(max_source_chunk_bytes: Option<u64>, max_row_groups_per_chunk: usize) -> Self {
        Self { max_source_chunk_bytes, max_row_groups_per_chunk }
    }

    pub fn for_source_task(memory_policy: MemoryControlPolicy, model: SourceTaskMemoryModel) -> Self {
        let base = Self::from(memory_policy);
        let MemoryControlPolicy::Bounded {
            device_budget_bytes, max_source_chunk_bytes, max_row_groups_per_chunk, ..
        } = memory_policy
        else {
            return base;
        };
        if max_source_chunk_bytes > device_budget_bytes {
            return base;
        }
        let budget_target = device_budget_bytes / model.live_slot_count().max(1);
        let task_target = max_source_chunk_bytes.min(budget_target.max(1));
        Self::new(Some(task_target), max_row_groups_per_chunk)
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

#[derive(Debug, Clone, PartialEq)]
pub struct SourceChunkPlan {
    pub chunks: Vec<SourceChunk>,
    pub source_node_id: NodeId,
    pub row_group_metadata_available: bool,
    pub size_estimate_based: bool,
    pub metadata_unavailable_files: usize,
    pub metadata_load_time: Duration,
    pub metadata_load_count: u64,
}

impl SourceChunkPlan {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn row_group_count(&self) -> usize {
        self.chunks.iter().map(SourceChunk::row_group_count).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceChunk {
    pub source_node_id: NodeId,
    pub chunk_index: usize,
    pub file_index: usize,
    pub location: ParquetSourceLocation,
    pub path: PathBuf,
    pub stable_id: String,
    pub row_group_indexes: Vec<usize>,
    pub reads_all_row_groups: bool,
    pub known_file_bytes: Option<u64>,
    pub known_row_group_bytes: Option<u64>,
    pub known_projected_compressed_bytes: Option<u64>,
    pub known_projected_uncompressed_bytes: Option<u64>,
    pub known_pure_filter_compressed_bytes: Option<u64>,
    pub known_pure_filter_uncompressed_bytes: Option<u64>,
    pub estimated_chunk_bytes: Option<u64>,
    pub estimated_source_read_bytes: Option<u64>,
    pub exact_source_read_preflight_bytes: Option<u64>,
    pub source_file_row_groups: Option<Vec<usize>>,
    pub source_file_row_group_bytes: Option<Vec<u64>>,
    pub source_file_projected_compressed_bytes: Option<Vec<u64>>,
    pub source_file_projected_uncompressed_bytes: Option<Vec<u64>>,
    pub source_file_pure_filter_compressed_bytes: Option<Vec<u64>>,
    pub source_file_pure_filter_uncompressed_bytes: Option<Vec<u64>>,
    pub source_file_row_group_rows: Option<Vec<u64>>,
    pub read_columns: Vec<String>,
    pub partition_values: Vec<LiteralValue>,
    pub zero_column_count: bool,
    pub known_row_count: Option<u64>,
    pub scan_limit_rows: Option<usize>,
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
    let mut metadata_cache = SourceMetadataCache::default();
    plan_local_parquet_source_chunks_with_metadata_cache(source_node_id, source, policy, &mut metadata_cache)
}

pub fn plan_local_parquet_source_chunks_with_metadata_cache(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
    metadata_cache: &mut SourceMetadataCache,
) -> Result<SourceChunkPlan> {
    validate_chunk_planning_source(source)?;
    validate_chunk_planning_policy(policy)?;

    let mut chunks = Vec::new();
    let mut saw_metadata = false;
    let mut saw_missing_metadata = false;
    let mut metadata_unavailable_files = 0;
    let mut metadata_load_time = Duration::ZERO;
    let mut metadata_load_count = 0u64;

    for (file_index, file) in source.files.iter().enumerate() {
        let metadata_load = read_parquet_metadata(source, file, metadata_cache);
        metadata_load_time += metadata_load.load_time;
        metadata_load_count = metadata_load_count.saturating_add(metadata_load.load_count);
        let metadata = metadata_load.metadata;
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

    chunks = apply_source_scan_limit_to_chunks(source, chunks)?;

    if source.is_zero_column_count_source()
        && let Some(chunk) = chunks.iter().find(|chunk| chunk.known_row_count.is_none())
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_chunk.plan",
            "zero-column count source requires parquet row-count metadata for each planned chunk",
        )
        .with_fact("reason", "row_count_metadata_unavailable")
        .with_fact("path", chunk.path.display())
        .with_fact("file_index", chunk.file_index)
        .with_fact("chunk_index", chunk.chunk_index));
    }

    Ok(SourceChunkPlan {
        chunks,
        source_node_id,
        row_group_metadata_available: saw_metadata && !saw_missing_metadata,
        size_estimate_based: saw_missing_metadata,
        metadata_unavailable_files,
        metadata_load_time,
        metadata_load_count,
    })
}

pub fn local_parquet_source_has_exact_read_preflight(source: &LocalParquetSource) -> bool {
    source.files.iter().all(source_file_has_row_count_metadata)
        && exact_source_read_preflight_bytes(source, Some(1), source.is_zero_column_count_source()).is_some()
}

fn apply_source_scan_limit_to_chunks(
    source: &LocalParquetSource,
    chunks: Vec<SourceChunk>,
) -> Result<Vec<SourceChunk>> {
    let Some(scan_limit) = source.scan_limit else {
        return Ok(chunks);
    };
    if chunks.is_empty() {
        return Ok(chunks);
    }

    let mut remaining = u64::try_from(scan_limit).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.source_chunk.scan_limit",
            "source scan limit exceeds native row-count range",
        )
        .with_fact("scan_limit", scan_limit)
    })?;
    let mut limited = Vec::new();
    for mut chunk in chunks {
        let known_row_count = chunk.known_row_count.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_chunk.scan_limit",
                "source scan limit requires row-count metadata for planned chunks",
            )
            .with_fact("path", chunk.path.display())
            .with_fact("file_index", chunk.file_index)
            .with_fact("chunk_index", chunk.chunk_index)
        })?;

        if remaining == 0 {
            if limited.is_empty() {
                chunk.scan_limit_rows = Some(0);
                chunk.known_row_count = Some(0);
                chunk.exact_source_read_preflight_bytes =
                    exact_source_read_preflight_bytes(source, chunk.known_row_count, chunk.zero_column_count);
                limited.push(chunk);
            }
            break;
        }

        if known_row_count <= remaining {
            remaining = remaining.saturating_sub(known_row_count);
            limited.push(chunk);
            continue;
        }

        let keep_rows = usize::try_from(remaining).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_chunk.scan_limit",
                "source scan limit chunk slice exceeds native row-count range",
            )
            .with_fact("scan_limit", scan_limit)
            .with_fact("remaining_rows", remaining)
        })?;
        chunk.scan_limit_rows = Some(keep_rows);
        chunk.known_row_count = Some(remaining);
        chunk.exact_source_read_preflight_bytes =
            exact_source_read_preflight_bytes(source, chunk.known_row_count, chunk.zero_column_count);
        limited.push(chunk);
        break;
    }
    Ok(limited)
}

#[derive(Debug, Default)]
pub struct SourceMetadataCache {
    footers: HashMap<SourceMetadataCacheKey, Option<Arc<ParquetMetaData>>>,
}

impl SourceMetadataCache {
    pub fn cached_footer_count(&self) -> usize {
        self.footers.len()
    }

    fn footer_metadata(&mut self, file: &LocalParquetFile) -> FooterMetadataLoad {
        let key = SourceMetadataCacheKey::from_file(file);
        if let Some(metadata) = self.footers.get(&key) {
            return FooterMetadataLoad { metadata: metadata.clone(), load_time: Duration::ZERO, load_count: 0 };
        }

        let timer = Instant::now();
        let metadata = file.location.as_local_path().and_then(|path| {
            SerializedFileReader::try_from(path).ok().map(|reader| Arc::new(reader.metadata().clone()))
        });
        let load_time = timer.elapsed();
        self.footers.insert(key, metadata.clone());
        FooterMetadataLoad { metadata, load_time, load_count: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceMetadataCacheKey {
    path: PathBuf,
    stable_id: String,
    file_size_bytes: Option<u64>,
}

impl SourceMetadataCacheKey {
    fn from_file(file: &LocalParquetFile) -> Self {
        Self { path: file.path.clone(), stable_id: file.stable_id.clone(), file_size_bytes: file.file_size_bytes }
    }
}

struct FooterMetadataLoad {
    metadata: Option<Arc<ParquetMetaData>>,
    load_time: Duration,
    load_count: u64,
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

    if source.read_columns.is_empty() && !source.is_zero_column_count_source() {
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
        validate_file_row_group_metadata_lengths(file, file_index)?;
    }

    Ok(())
}

fn validate_file_row_group_metadata_lengths(file: &LocalParquetFile, file_index: usize) -> Result<()> {
    let expected_len = [
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
    let Some(expected_len) = expected_len else {
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
        if let Some(values) = values
            && values.len() != expected_len
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_chunk.plan",
                "local parquet source row-group metadata vectors must have matching lengths",
            )
            .with_fact("reason", "row_group_metadata_count_mismatch")
            .with_fact("file_index", file_index)
            .with_fact("field", field)
            .with_fact("expected_count", expected_len)
            .with_fact("actual_count", values.len()));
        }
    }
    Ok(())
}

struct SourceMetadataLoad {
    metadata: Option<RowGroupMetadata>,
    load_time: Duration,
    load_count: u64,
}

fn read_parquet_metadata(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    metadata_cache: &mut SourceMetadataCache,
) -> SourceMetadataLoad {
    let needs_footer_metadata = file.row_group_bytes.is_none()
        || file.projected_row_group_compressed_bytes.is_none()
        || file.projected_row_group_uncompressed_bytes.is_none()
        || file.pure_filter_row_group_compressed_bytes.is_none()
        || file.pure_filter_row_group_uncompressed_bytes.is_none()
        || file.row_group_rows.is_none();
    let footer_load = if needs_footer_metadata {
        metadata_cache.footer_metadata(file)
    } else {
        FooterMetadataLoad { metadata: None, load_time: Duration::ZERO, load_count: 0 }
    };
    let file_metadata = footer_load.metadata.as_ref().map(|metadata| {
        let read_column_indexes = source.read_column_file_indices.iter().copied().collect::<BTreeSet<_>>();
        let projected_column_indexes = source
            .projected_output_column_source_refs
            .iter()
            .filter_map(|source_ref| match source_ref {
                crate::source::SourceColumnRef::File(index) => Some(*index),
                crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => None,
            })
            .collect::<BTreeSet<_>>();
        let pure_filter_column_indexes = source
            .pure_filter_column_source_refs
            .iter()
            .filter_map(|source_ref| match source_ref {
                crate::source::SourceColumnRef::File(index) if !projected_column_indexes.contains(index) => {
                    Some(*index)
                },
                crate::source::SourceColumnRef::File(_)
                | crate::source::SourceColumnRef::Partition(_)
                | crate::source::SourceColumnRef::MissingNull(_) => None,
            })
            .collect::<BTreeSet<_>>();
        let row_group_bytes = metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &read_column_indexes, ColumnByteKind::Compressed))
            .collect::<Option<Vec<_>>>();
        let projected_compressed_bytes = metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Compressed))
            .collect::<Option<Vec<_>>>();
        let projected_uncompressed_bytes = metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Uncompressed))
            .collect::<Option<Vec<_>>>();
        let pure_filter_compressed_bytes = metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Compressed))
            .collect::<Option<Vec<_>>>();
        let pure_filter_uncompressed_bytes = metadata
            .row_groups()
            .iter()
            .map(|row_group| {
                row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Uncompressed)
            })
            .collect::<Option<Vec<_>>>();
        let row_group_rows = metadata
            .row_groups()
            .iter()
            .map(|row_group| u64::try_from(row_group.num_rows()).ok())
            .collect::<Option<Vec<_>>>();
        (
            row_group_bytes,
            projected_compressed_bytes,
            projected_uncompressed_bytes,
            pure_filter_compressed_bytes,
            pure_filter_uncompressed_bytes,
            row_group_rows,
        )
    });
    let row_group_bytes =
        file.row_group_bytes.clone().or_else(|| file_metadata.as_ref().and_then(|(bytes, ..)| bytes.clone()));
    let projected_compressed_bytes = file
        .projected_row_group_compressed_bytes
        .clone()
        .or_else(|| file_metadata.as_ref().and_then(|(_, bytes, ..)| bytes.clone()));
    let projected_uncompressed_bytes = file
        .projected_row_group_uncompressed_bytes
        .clone()
        .or_else(|| file_metadata.as_ref().and_then(|(_, _, bytes, ..)| bytes.clone()));
    let pure_filter_compressed_bytes = file
        .pure_filter_row_group_compressed_bytes
        .clone()
        .or_else(|| file_metadata.as_ref().and_then(|(_, _, _, bytes, ..)| bytes.clone()));
    let pure_filter_uncompressed_bytes = file
        .pure_filter_row_group_uncompressed_bytes
        .clone()
        .or_else(|| file_metadata.as_ref().and_then(|(_, _, _, _, bytes, _rows)| bytes.clone()));
    let row_group_rows =
        file.row_group_rows.clone().or_else(|| file_metadata.as_ref().and_then(|(_, _, _, _, _, rows)| rows.clone()));
    if row_group_bytes.is_some()
        || projected_compressed_bytes.is_some()
        || projected_uncompressed_bytes.is_some()
        || pure_filter_compressed_bytes.is_some()
        || pure_filter_uncompressed_bytes.is_some()
        || row_group_rows.is_some()
    {
        SourceMetadataLoad {
            metadata: Some(RowGroupMetadata {
                row_group_bytes,
                projected_compressed_bytes,
                projected_uncompressed_bytes,
                pure_filter_compressed_bytes,
                pure_filter_uncompressed_bytes,
                row_group_rows,
            }),
            load_time: footer_load.load_time,
            load_count: footer_load.load_count,
        }
    } else {
        SourceMetadataLoad { metadata: None, load_time: footer_load.load_time, load_count: footer_load.load_count }
    }
}

#[derive(Debug, Clone, Copy)]
enum ColumnByteKind {
    Compressed,
    Uncompressed,
}

fn row_group_column_bytes(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    column_indexes: &BTreeSet<usize>,
    kind: ColumnByteKind,
) -> Option<u64> {
    column_indexes.iter().try_fold(0u64, |total, column_index| {
        let column = row_group.columns().get(*column_index)?;
        let bytes = match kind {
            ColumnByteKind::Compressed => column.compressed_size(),
            ColumnByteKind::Uncompressed => column.uncompressed_size(),
        };
        let bytes = u64::try_from(bytes).ok()?;
        total.checked_add(bytes)
    })
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

    Ok(Some((0..metadata.row_group_count()).collect()))
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
        if *row_group >= metadata.row_group_count() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.source_chunk.plan",
                "selected parquet row group is outside file metadata",
            )
            .with_fact("path", file.path.display())
            .with_fact("file_index", file_index)
            .with_fact("row_group_position", row_group_position)
            .with_fact("row_group", row_group)
            .with_fact("max_row_group", metadata.row_group_count().saturating_sub(1)));
        }
    }

    Ok(())
}
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
    let row_group_count_basis = metadata.map_or(row_groups.len(), RowGroupMetadata::row_group_count).max(1);
    let mut current = PendingChunk::default();

    for row_group in row_groups {
        let row_group_bytes = metadata.and_then(|metadata| metadata.row_group_bytes(*row_group));
        let estimated_row_group_bytes =
            row_group_bytes.or_else(|| proportional_bytes(file.file_size_bytes, 1, row_group_count_basis));
        let projected_compressed_bytes = metadata.and_then(|metadata| metadata.projected_compressed_bytes(*row_group));
        let projected_uncompressed_bytes =
            metadata.and_then(|metadata| metadata.projected_uncompressed_bytes(*row_group));
        let pure_filter_compressed_bytes =
            metadata.and_then(|metadata| metadata.pure_filter_compressed_bytes(*row_group));
        let pure_filter_uncompressed_bytes =
            metadata.and_then(|metadata| metadata.pure_filter_uncompressed_bytes(*row_group));
        let row_group_rows = metadata.and_then(|metadata| metadata.row_group_rows(*row_group));
        let retained_output_bytes = projected_uncompressed_bytes.or(estimated_row_group_bytes);
        let source_read_planning_bytes =
            source_read_uncompressed_bytes(source, projected_uncompressed_bytes, pure_filter_uncompressed_bytes)
                .or(retained_output_bytes);
        if current.should_flush_before(source_read_planning_bytes, max_row_groups, policy.max_source_chunk_bytes) {
            push_pending_chunk(
                source_node_id,
                source,
                file,
                file_index,
                row_groups.len(),
                metadata,
                &mut current,
                chunks,
            );
        }
        current.push(
            *row_group,
            row_group_bytes,
            source_read_planning_bytes,
            projected_compressed_bytes,
            projected_uncompressed_bytes,
            pure_filter_compressed_bytes,
            pure_filter_uncompressed_bytes,
            row_group_rows,
        );
    }

    push_pending_chunk(source_node_id, source, file, file_index, row_groups.len(), metadata, &mut current, chunks);
}
fn push_pending_chunk(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    file_index: usize,
    planned_row_group_count: usize,
    metadata: Option<&RowGroupMetadata>,
    pending: &mut PendingChunk,
    chunks: &mut Vec<SourceChunk>,
) {
    if pending.row_groups.is_empty() {
        return;
    }

    let known_row_group_bytes = pending.known_bytes();
    let known_projected_compressed_bytes = pending.known_projected_compressed_bytes();
    let known_projected_uncompressed_bytes = pending.known_projected_uncompressed_bytes();
    let known_pure_filter_compressed_bytes = pending.known_pure_filter_compressed_bytes();
    let known_pure_filter_uncompressed_bytes = pending.known_pure_filter_uncompressed_bytes();
    let known_row_count = pending.known_rows();
    let estimated_chunk_bytes = known_projected_uncompressed_bytes
        .or(known_row_group_bytes)
        .or_else(|| proportional_bytes(file.file_size_bytes, pending.row_groups.len(), planned_row_group_count));
    let estimated_source_read_bytes = source_read_uncompressed_bytes(
        source,
        known_projected_uncompressed_bytes,
        known_pure_filter_uncompressed_bytes,
    )
    .or(known_row_group_bytes)
    .or(estimated_chunk_bytes);
    let exact_source_read_preflight_bytes =
        exact_source_read_preflight_bytes(source, known_row_count, source.is_zero_column_count_source());
    chunks.push(SourceChunk {
        source_node_id,
        chunk_index: chunks.len(),
        file_index,
        location: file.location.clone(),
        path: file.path.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: std::mem::take(&mut pending.row_groups),
        reads_all_row_groups: false,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes,
        known_projected_compressed_bytes,
        known_projected_uncompressed_bytes,
        known_pure_filter_compressed_bytes,
        known_pure_filter_uncompressed_bytes,
        estimated_chunk_bytes,
        estimated_source_read_bytes,
        exact_source_read_preflight_bytes,
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: metadata.and_then(RowGroupMetadata::row_group_bytes_vec),
        source_file_projected_compressed_bytes: metadata.and_then(RowGroupMetadata::projected_compressed_bytes_vec),
        source_file_projected_uncompressed_bytes: metadata.and_then(RowGroupMetadata::projected_uncompressed_bytes_vec),
        source_file_pure_filter_compressed_bytes: metadata.and_then(RowGroupMetadata::pure_filter_compressed_bytes_vec),
        source_file_pure_filter_uncompressed_bytes: metadata
            .and_then(RowGroupMetadata::pure_filter_uncompressed_bytes_vec),
        source_file_row_group_rows: metadata.and_then(RowGroupMetadata::row_group_rows_vec),
        read_columns: source.read_columns.clone(),
        partition_values: file.partition_values.clone(),
        zero_column_count: source.is_zero_column_count_source(),
        known_row_count,
        scan_limit_rows: None,
        size_estimate_based: known_row_group_bytes.is_none(),
    });
    pending.clear();
}

fn source_read_uncompressed_bytes(
    source: &LocalParquetSource,
    projected_uncompressed_bytes: Option<u64>,
    pure_filter_uncompressed_bytes: Option<u64>,
) -> Option<u64> {
    match (projected_uncompressed_bytes, pure_filter_uncompressed_bytes) {
        (Some(projected), Some(filter)) if projected_and_pure_filter_file_columns_overlap(source) => {
            Some(projected.max(filter))
        },
        (Some(projected), Some(filter)) => projected.checked_add(filter),
        (Some(projected), None) => Some(projected),
        (None, Some(filter)) => Some(filter),
        (None, None) => None,
    }
}

fn exact_source_read_preflight_bytes(
    source: &LocalParquetSource,
    known_row_count: Option<u64>,
    zero_column_count: bool,
) -> Option<u64> {
    let row_count = if zero_column_count { 1 } else { known_row_count? };
    if zero_column_count {
        return exact_source_read_required_budget_bytes(exact_fixed_width_column_upper_bound_bytes(
            &arrow::datatypes::DataType::Int64,
            false,
            row_count,
        )?);
    }
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for column in source.projected_output_columns.iter().chain(&source.pure_filter_columns) {
        if !seen.insert(column.as_str()) {
            continue;
        }
        let field = source.schema.field_with_name(column).ok()?;
        let column_bytes =
            exact_fixed_width_column_upper_bound_bytes(field.data_type(), field.is_nullable(), row_count)?;
        total = total.checked_add(column_bytes)?;
    }
    exact_source_read_required_budget_bytes(total)
}

fn exact_source_read_required_budget_bytes(retained_output_bytes: u64) -> Option<u64> {
    let minimum_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).ok()?;
    Some(retained_output_bytes.max(minimum_budget))
}

fn exact_fixed_width_column_upper_bound_bytes(
    data_type: &arrow::datatypes::DataType,
    nullable: bool,
    row_count: u64,
) -> Option<u64> {
    let element_size = match data_type {
        arrow::datatypes::DataType::Boolean | arrow::datatypes::DataType::Int8 | arrow::datatypes::DataType::UInt8 => 1,
        arrow::datatypes::DataType::Int16 | arrow::datatypes::DataType::UInt16 => 2,
        arrow::datatypes::DataType::Int32
        | arrow::datatypes::DataType::UInt32
        | arrow::datatypes::DataType::Float32
        | arrow::datatypes::DataType::Date32 => 4,
        arrow::datatypes::DataType::Int64
        | arrow::datatypes::DataType::UInt64
        | arrow::datatypes::DataType::Float64
        | arrow::datatypes::DataType::Date64
        | arrow::datatypes::DataType::Timestamp(_, _)
        | arrow::datatypes::DataType::Duration(_) => 8,
        arrow::datatypes::DataType::Decimal128(_, _) => 16,
        _ => return None,
    };
    let data_bytes = row_count.checked_mul(element_size)?;
    let mask_bytes = if nullable { null_mask_allocation_size_bytes(row_count) } else { 0 };
    data_bytes.checked_add(mask_bytes)
}

fn null_mask_allocation_size_bytes(row_count: u64) -> u64 {
    if row_count == 0 { 0 } else { row_count.div_ceil(8).div_ceil(64) * 64 }
}

fn source_file_has_row_count_metadata(file: &LocalParquetFile) -> bool {
    let Some(rows) = &file.row_group_rows else {
        return false;
    };
    if let Some(selected) = &file.row_groups {
        selected.iter().all(|row_group| *row_group < rows.len())
    } else {
        !rows.is_empty()
    }
}

fn projected_and_pure_filter_file_columns_overlap(source: &LocalParquetSource) -> bool {
    let projected = source
        .projected_output_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            crate::source::SourceColumnRef::File(index) => Some(*index),
            crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => None,
        })
        .collect::<BTreeSet<_>>();
    source.pure_filter_column_source_refs.iter().any(|source_ref| match source_ref {
        crate::source::SourceColumnRef::File(index) => projected.contains(index),
        crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => false,
    })
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
        location: file.location.clone(),
        path: file.path.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: Vec::new(),
        reads_all_row_groups: true,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes: None,
        known_projected_compressed_bytes: None,
        known_projected_uncompressed_bytes: None,
        known_pure_filter_compressed_bytes: None,
        known_pure_filter_uncompressed_bytes: None,
        estimated_chunk_bytes: file.file_size_bytes,
        estimated_source_read_bytes: file.file_size_bytes,
        exact_source_read_preflight_bytes: None,
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: None,
        source_file_projected_compressed_bytes: None,
        source_file_projected_uncompressed_bytes: None,
        source_file_pure_filter_compressed_bytes: None,
        source_file_pure_filter_uncompressed_bytes: None,
        source_file_row_group_rows: None,
        read_columns: source.read_columns.clone(),
        partition_values: file.partition_values.clone(),
        zero_column_count: source.is_zero_column_count_source(),
        known_row_count: None,
        scan_limit_rows: None,
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

#[derive(Debug, Clone)]
struct RowGroupMetadata {
    row_group_bytes: Option<Vec<u64>>,
    projected_compressed_bytes: Option<Vec<u64>>,
    projected_uncompressed_bytes: Option<Vec<u64>>,
    pure_filter_compressed_bytes: Option<Vec<u64>>,
    pure_filter_uncompressed_bytes: Option<Vec<u64>>,
    row_group_rows: Option<Vec<u64>>,
}

impl RowGroupMetadata {
    fn row_group_count(&self) -> usize {
        self.row_group_bytes
            .as_ref()
            .or(self.projected_uncompressed_bytes.as_ref())
            .or(self.projected_compressed_bytes.as_ref())
            .or(self.pure_filter_uncompressed_bytes.as_ref())
            .or(self.pure_filter_compressed_bytes.as_ref())
            .or(self.row_group_rows.as_ref())
            .map_or(0, Vec::len)
    }

    fn row_group_bytes(&self, row_group: usize) -> Option<u64> {
        self.row_group_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    fn projected_compressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.projected_compressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    fn projected_uncompressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.projected_uncompressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    fn pure_filter_compressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.pure_filter_compressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    fn pure_filter_uncompressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.pure_filter_uncompressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    fn row_group_rows(&self, row_group: usize) -> Option<u64> {
        self.row_group_rows.as_ref().and_then(|rows| rows.get(row_group).copied())
    }

    fn row_group_bytes_vec(&self) -> Option<Vec<u64>> {
        self.row_group_bytes.clone()
    }

    fn projected_compressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.projected_compressed_bytes.clone()
    }

    fn projected_uncompressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.projected_uncompressed_bytes.clone()
    }

    fn pure_filter_compressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.pure_filter_compressed_bytes.clone()
    }

    fn pure_filter_uncompressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.pure_filter_uncompressed_bytes.clone()
    }

    fn row_group_rows_vec(&self) -> Option<Vec<u64>> {
        self.row_group_rows.clone()
    }
}

#[derive(Debug, Default)]
struct PendingChunk {
    row_groups: Vec<usize>,
    known_bytes: u64,
    known_planning_bytes: u64,
    known_projected_compressed_bytes: u64,
    known_projected_uncompressed_bytes: u64,
    known_pure_filter_compressed_bytes: u64,
    known_pure_filter_uncompressed_bytes: u64,
    known_rows: u64,
    has_unknown_bytes: bool,
    has_unknown_projected_compressed_bytes: bool,
    has_unknown_projected_uncompressed_bytes: bool,
    has_unknown_pure_filter_compressed_bytes: bool,
    has_unknown_pure_filter_uncompressed_bytes: bool,
    has_unknown_rows: bool,
}

impl PendingChunk {
    fn push(
        &mut self,
        row_group: usize,
        row_group_bytes: Option<u64>,
        planning_bytes: Option<u64>,
        projected_compressed_bytes: Option<u64>,
        projected_uncompressed_bytes: Option<u64>,
        pure_filter_compressed_bytes: Option<u64>,
        pure_filter_uncompressed_bytes: Option<u64>,
        row_group_rows: Option<u64>,
    ) {
        self.row_groups.push(row_group);
        if let Some(row_group_bytes) = row_group_bytes {
            self.known_bytes = self.known_bytes.saturating_add(row_group_bytes);
        } else {
            self.has_unknown_bytes = true;
        }
        if let Some(planning_bytes) = planning_bytes {
            self.known_planning_bytes = self.known_planning_bytes.saturating_add(planning_bytes);
        }
        if let Some(projected_compressed_bytes) = projected_compressed_bytes {
            self.known_projected_compressed_bytes =
                self.known_projected_compressed_bytes.saturating_add(projected_compressed_bytes);
        } else {
            self.has_unknown_projected_compressed_bytes = true;
        }
        if let Some(projected_uncompressed_bytes) = projected_uncompressed_bytes {
            self.known_projected_uncompressed_bytes =
                self.known_projected_uncompressed_bytes.saturating_add(projected_uncompressed_bytes);
        } else {
            self.has_unknown_projected_uncompressed_bytes = true;
        }
        if let Some(pure_filter_compressed_bytes) = pure_filter_compressed_bytes {
            self.known_pure_filter_compressed_bytes =
                self.known_pure_filter_compressed_bytes.saturating_add(pure_filter_compressed_bytes);
        } else {
            self.has_unknown_pure_filter_compressed_bytes = true;
        }
        if let Some(pure_filter_uncompressed_bytes) = pure_filter_uncompressed_bytes {
            self.known_pure_filter_uncompressed_bytes =
                self.known_pure_filter_uncompressed_bytes.saturating_add(pure_filter_uncompressed_bytes);
        } else {
            self.has_unknown_pure_filter_uncompressed_bytes = true;
        }
        if let Some(row_group_rows) = row_group_rows {
            self.known_rows = self.known_rows.saturating_add(row_group_rows);
        } else {
            self.has_unknown_rows = true;
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

        self.known_planning_bytes.saturating_add(next_row_group_bytes) > max_source_chunk_bytes
    }

    fn known_bytes(&self) -> Option<u64> {
        (!self.has_unknown_bytes).then_some(self.known_bytes)
    }

    fn known_projected_compressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_projected_compressed_bytes).then_some(self.known_projected_compressed_bytes)
    }

    fn known_projected_uncompressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_projected_uncompressed_bytes).then_some(self.known_projected_uncompressed_bytes)
    }

    fn known_pure_filter_compressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_pure_filter_compressed_bytes).then_some(self.known_pure_filter_compressed_bytes)
    }

    fn known_pure_filter_uncompressed_bytes(&self) -> Option<u64> {
        (!self.has_unknown_pure_filter_uncompressed_bytes).then_some(self.known_pure_filter_uncompressed_bytes)
    }

    fn known_rows(&self) -> Option<u64> {
        (!self.has_unknown_rows).then_some(self.known_rows)
    }

    fn clear(&mut self) {
        self.row_groups.clear();
        self.known_bytes = 0;
        self.known_planning_bytes = 0;
        self.known_projected_compressed_bytes = 0;
        self.known_projected_uncompressed_bytes = 0;
        self.known_pure_filter_compressed_bytes = 0;
        self.known_pure_filter_uncompressed_bytes = 0;
        self.known_rows = 0;
        self.has_unknown_bytes = false;
        self.has_unknown_projected_compressed_bytes = false;
        self.has_unknown_projected_uncompressed_bytes = false;
        self.has_unknown_pure_filter_compressed_bytes = false;
        self.has_unknown_pure_filter_uncompressed_bytes = false;
        self.has_unknown_rows = false;
    }
}
