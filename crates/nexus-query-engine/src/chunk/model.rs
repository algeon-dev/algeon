use std::{path::PathBuf, time::Duration};

use crate::{
    expr::LiteralValue,
    plan::NodeId,
    runtime::SourceChunkPlanningMode,
    source::{ParquetReaderDecimalType, ParquetSourceLocation},
};

/// Planned chunks and metadata diagnostics for one source node.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceChunkPlan {
    /// Ordered chunks produced for one source node.
    pub chunks: Vec<SourceChunk>,
    /// Native source node these chunks belong to.
    pub source_node_id: NodeId,
    /// True when every planned file had row-group metadata available.
    pub row_group_metadata_available: bool,
    /// True when at least one chunk used file-size proportional estimates.
    pub size_estimate_based: bool,
    /// Number of files whose row-group metadata could not be loaded or supplied.
    pub metadata_unavailable_files: usize,
    /// Wall-clock time spent loading Parquet footer metadata.
    pub metadata_load_time: Duration,
    /// Number of footer metadata reads performed after cache misses.
    pub metadata_load_count: u64,
}

impl SourceChunkPlan {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the total number of explicitly planned row groups.
    ///
    /// File-level chunks whose row groups are unknown contribute zero here.
    pub fn row_group_count(&self) -> usize {
        self.chunks.iter().map(SourceChunk::row_group_count).sum()
    }
}

/// Adjacent source chunks that may be executed as one physical Parquet read.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceChunkBatch {
    /// Zero-based index within the batch plan.
    pub batch_index: usize,
    /// Child chunks retained for source identity, metrics, and retry fallback.
    pub chunks: Vec<SourceChunk>,
    /// Best available source-read byte estimate for the batch.
    pub estimated_source_read_bytes: Option<u64>,
    /// Stable diagnostic reason for this batching decision.
    ///
    /// `batched` means the read contains multiple source chunks. Singleton
    /// values explain why the chunk could not be grouped with its neighbor or
    /// why batching was disabled before chunk compatibility was evaluated.
    pub source_batching_reason: String,
}

impl SourceChunkBatch {
    pub fn is_singleton(&self) -> bool {
        self.chunks.len() == 1
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the number of explicitly selected row groups across children.
    pub fn row_group_count(&self) -> usize {
        self.chunks.iter().map(SourceChunk::row_group_count).sum()
    }
}

/// One executable read range for a local Parquet source.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceChunk {
    /// Native source node this chunk belongs to.
    pub source_node_id: NodeId,
    /// Zero-based index within the containing [`SourceChunkPlan`].
    pub chunk_index: usize,
    /// Zero-based index into [`LocalParquetSource::files`](crate::source::LocalParquetSource::files).
    pub file_index: usize,
    /// Source location for the Parquet file.
    pub location: ParquetSourceLocation,
    /// Local path used for diagnostics and footer reads.
    pub path: PathBuf,
    /// Resolved reader strategy and decode contract for this chunk.
    pub resolved_source: crate::source::ResolvedParquetSource,
    /// Stable file identifier copied from the source definition.
    pub stable_id: String,
    /// Row groups selected for this chunk when row-group metadata is known.
    pub row_group_indexes: Vec<usize>,
    /// True when the chunk must read the whole file because row groups are unknown.
    pub reads_all_row_groups: bool,
    /// Known file size in bytes, if supplied by the source.
    pub known_file_bytes: Option<u64>,
    /// Sum of selected row-group bytes across all read columns, if known.
    pub known_row_group_bytes: Option<u64>,
    /// Sum of selected projected-output column compressed bytes, if known.
    pub known_projected_compressed_bytes: Option<u64>,
    /// Sum of selected projected-output column uncompressed bytes, if known.
    pub known_projected_uncompressed_bytes: Option<u64>,
    /// Sum of pure-filter column compressed bytes not emitted downstream, if known.
    pub known_pure_filter_compressed_bytes: Option<u64>,
    /// Sum of pure-filter column uncompressed bytes not emitted downstream, if known.
    pub known_pure_filter_uncompressed_bytes: Option<u64>,
    /// Best available retained-output size estimate in bytes.
    pub estimated_chunk_bytes: Option<u64>,
    /// Best available source-read budget estimate in bytes.
    pub estimated_source_read_bytes: Option<u64>,
    /// Exact cudf source-read budget in bytes for fixed-width projected columns.
    pub exact_source_read_preflight_bytes: Option<u64>,
    /// Source-level selected row groups, before chunking.
    pub source_file_row_groups: Option<Vec<usize>>,
    /// Per-row-group read-column bytes for the full file, if available.
    pub source_file_row_group_bytes: Option<Vec<u64>>,
    /// Per-row-group projected-output compressed bytes for the full file, if available.
    pub source_file_projected_compressed_bytes: Option<Vec<u64>>,
    /// Per-row-group projected-output uncompressed bytes for the full file, if available.
    pub source_file_projected_uncompressed_bytes: Option<Vec<u64>>,
    /// Per-row-group pure-filter compressed bytes for the full file, if available.
    pub source_file_pure_filter_compressed_bytes: Option<Vec<u64>>,
    /// Per-row-group pure-filter uncompressed bytes for the full file, if available.
    pub source_file_pure_filter_uncompressed_bytes: Option<Vec<u64>>,
    /// Per-row-group row counts for the full file, if available.
    pub source_file_row_group_rows: Option<Vec<u64>>,
    /// Per-read-column decimal storage types proven from Parquet metadata.
    pub reader_decimal_types: Vec<ParquetReaderDecimalType>,
    /// True when `reader_decimal_types` came from, or was checked against, the
    /// footer for this exact object-store file during source chunk planning.
    pub reader_decimal_types_footer_verified: bool,
    /// Physical file columns read by the source.
    pub read_columns: Vec<String>,
    /// Partition values attached to the source file.
    pub partition_values: Vec<LiteralValue>,
    /// True when this source is a zero-column count scan.
    pub zero_column_count: bool,
    /// Known row count for this chunk after scan-limit trimming, if available.
    pub known_row_count: Option<u64>,
    /// Per-chunk row limit applied when a source scan limit cuts through this chunk.
    pub scan_limit_rows: Option<usize>,
    /// True when chunk sizing relied on file-size estimates instead of row-group bytes.
    pub size_estimate_based: bool,
    /// Stable diagnostic reason for the source chunk batching decision that
    /// applies to this chunk.
    pub source_batching_reason: Option<String>,
    /// Source chunk planning mode requested for the policy that emitted this chunk.
    pub source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Stable diagnostic reason for the row-group and byte policy that emitted this chunk.
    pub source_chunk_planning_reason: String,
}

impl SourceChunk {
    pub fn row_group_count(&self) -> usize {
        self.row_group_indexes.len()
    }

    /// Returns the represented row-group count for this chunk when known.
    ///
    /// `None` means the chunk is a correctness-preserving read-all chunk whose
    /// file row-group count was unknown at planning time.
    pub fn known_read_row_group_count(&self) -> Option<usize> {
        if !self.reads_all_row_groups {
            return Some(self.row_group_indexes.len());
        }
        self.known_source_file_row_group_count()
    }

    /// Returns true when this chunk explicitly selects every known row group
    /// in its file.
    pub fn selects_all_known_file_row_groups(&self) -> bool {
        let selected = self.row_group_indexes.iter().copied().collect::<std::collections::BTreeSet<_>>();
        if let Some(file_row_groups) = &self.source_file_row_groups {
            if file_row_groups.is_empty() || selected.len() != file_row_groups.len() {
                return false;
            }
            let known = file_row_groups.iter().copied().collect::<std::collections::BTreeSet<_>>();
            return selected == known;
        }

        let Some(row_group_count) = self.known_source_file_row_group_count() else {
            return false;
        };
        if row_group_count == 0 || selected.len() != row_group_count {
            return false;
        }
        selected == (0..row_group_count).collect::<std::collections::BTreeSet<_>>()
    }

    fn known_source_file_row_group_count(&self) -> Option<usize> {
        [
            self.source_file_row_group_rows.as_ref().map(Vec::len),
            self.source_file_row_group_bytes.as_ref().map(Vec::len),
            self.source_file_projected_compressed_bytes.as_ref().map(Vec::len),
            self.source_file_projected_uncompressed_bytes.as_ref().map(Vec::len),
            self.source_file_pure_filter_compressed_bytes.as_ref().map(Vec::len),
            self.source_file_pure_filter_uncompressed_bytes.as_ref().map(Vec::len),
        ]
        .into_iter()
        .flatten()
        .reduce(|left, right| if left == right { left } else { usize::MAX })
        .filter(|count| *count != usize::MAX)
    }
}
