//! Stable local-Parquet source identities.

use query_planning::{
    chunk::{SourceChunk, SourceChunkBatch},
    source::LocalParquetSource,
};

use super::fingerprint::*;

/// Stable identity for one GPU-local Parquet source scan or chunk.
///
/// The key fingerprints the complete source scan contract: schema, projection,
/// predicates, resolved read contract, file content identity, and chunk
/// boundaries. It is local-process data, never a persistence format.
#[derive(Clone, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub struct GpuLocalParquetSourceCacheKey {
    fingerprint: String,
    diagnostic_fingerprint: String,
}

impl std::fmt::Debug for GpuLocalParquetSourceCacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLocalParquetSourceCacheKey").field("fingerprint", &self.diagnostic_fingerprint).finish()
    }
}

impl GpuLocalParquetSourceCacheKey {
    /// Complete canonical identity used only as input to a one-way cache
    /// observation fingerprint.  Callers must not surface this string: it
    /// contains the private source contract that the cache key protects.
    pub(crate) fn event_fingerprint_source(&self) -> &str {
        &self.fingerprint
    }

    /// Build a key for an unbounded local-Parquet source scan.
    #[doc(hidden)]
    pub fn from_source(source: &LocalParquetSource) -> Self {
        let mut fingerprint = String::new();
        push_token(&mut fingerprint, "mode", "unbounded_local_parquet_source");
        push_complete_source_contract(&mut fingerprint, source);
        let mut diagnostic_fingerprint = fingerprint.clone();
        redact_source_cache_fingerprint(&mut diagnostic_fingerprint, source);
        Self { fingerprint, diagnostic_fingerprint }
    }

    /// Build a key for one bounded local-Parquet chunk.
    ///
    /// Bounded execution cannot reuse the unbounded scan key: a single source
    /// is split into independently scheduled row-group ranges, each of which
    /// can carry a different visible row limit.  The full source contract plus
    /// the executed chunk boundary is therefore the cache identity.
    #[doc(hidden)]
    pub fn from_chunk(source: &LocalParquetSource, chunk: &SourceChunk) -> Self {
        let mut fingerprint = String::new();
        push_token(&mut fingerprint, "mode", "bounded_local_parquet_chunk");
        push_complete_source_contract(&mut fingerprint, source);
        push_chunk_contract(&mut fingerprint, "chunk", chunk);
        let mut diagnostic_fingerprint = fingerprint.clone();
        redact_source_cache_fingerprint(&mut diagnostic_fingerprint, source);
        Self { fingerprint, diagnostic_fingerprint }
    }

    /// Build a key for one bounded physical read containing adjacent chunks.
    ///
    /// A batch has one decoded output frame, so it must retain its ordered
    /// child identities as one value.  Treating the children as independent
    /// entries would let a later batched read consume a frame with the wrong
    /// concat/row-order contract.
    #[doc(hidden)]
    pub fn from_chunk_batch(source: &LocalParquetSource, batch: &SourceChunkBatch) -> Self {
        let mut fingerprint = String::new();
        push_token(&mut fingerprint, "mode", "bounded_local_parquet_chunk_batch");
        push_complete_source_contract(&mut fingerprint, source);
        push_token(&mut fingerprint, "batch.index", batch.batch_index);
        push_token(&mut fingerprint, "batch.chunk_count", batch.chunks.len());
        for (index, chunk) in batch.chunks.iter().enumerate() {
            push_chunk_contract(&mut fingerprint, format!("batch.chunk.{index}"), chunk);
        }
        let mut diagnostic_fingerprint = fingerprint.clone();
        redact_source_cache_fingerprint(&mut diagnostic_fingerprint, source);
        Self { fingerprint, diagnostic_fingerprint }
    }
}

fn push_complete_source_contract(output: &mut String, source: &LocalParquetSource) {
    push_source_scan_contract(output, source);
    push_token(output, "scan_limit", source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()));
    push_token(output, "file_count", source.files.len().to_string());
    for (index, file) in source.files.iter().enumerate() {
        push_token(output, format!("file.{index}.location"), file.location.diagnostic_location());
        push_token(output, format!("file.{index}.stable_id"), &file.stable_id);
        push_resolved_source_contract(output, format!("file.{index}.resolved"), &file.resolved_source);
        push_token(
            output,
            format!("file.{index}.size"),
            file.file_size_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
        );
        push_optional_usize_vec(output, format!("file.{index}.row_groups"), file.row_groups.as_deref());
        push_optional_u64_vec(output, format!("file.{index}.row_group_bytes"), file.row_group_bytes.as_deref());
        push_optional_u64_vec(
            output,
            format!("file.{index}.projected_compressed"),
            file.projected_row_group_compressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            output,
            format!("file.{index}.projected_uncompressed"),
            file.projected_row_group_uncompressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            output,
            format!("file.{index}.pure_filter_compressed"),
            file.pure_filter_row_group_compressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            output,
            format!("file.{index}.pure_filter_uncompressed"),
            file.pure_filter_row_group_uncompressed_bytes.as_deref(),
        );
        push_optional_u64_vec(output, format!("file.{index}.row_group_rows"), file.row_group_rows.as_deref());
        // This footer fact is part of the complete source contract.  It does
        // not change decoded rows, but it may change a graph admission
        // decision for this exact edge rowset, so reusing a cache identity
        // across different values would detach pricing from its source.
        push_optional_u64(output, format!("file.{index}.graph_vertex_upper_bound"), file.graph_vertex_upper_bound);
        push_reader_decimal_types(output, format!("file.{index}.reader_decimal_types"), &file.reader_decimal_types);
        let partition_values = file.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
        push_string_vec(output, format!("file.{index}.partition_values"), &partition_values);
    }
}

fn push_chunk_contract(output: &mut String, label: impl AsRef<str>, chunk: &SourceChunk) {
    let label = label.as_ref();
    push_token(output, format!("{label}.source_node"), chunk.source_node_id.index());
    push_token(output, format!("{label}.index"), chunk.chunk_index);
    push_token(output, format!("{label}.file_index"), chunk.file_index);
    push_usize_vec(output, format!("{label}.row_groups"), &chunk.row_group_indexes);
    push_token(output, format!("{label}.reads_all_row_groups"), chunk.reads_all_row_groups);
    push_token(output, format!("{label}.zero_column_count"), chunk.zero_column_count);
    push_optional_u64(output, format!("{label}.known_row_count"), chunk.known_row_count);
    match chunk.scan_limit_rows {
        Some(limit) => push_token(output, format!("{label}.scan_limit_rows"), limit),
        None => push_token(output, format!("{label}.scan_limit_rows"), "none"),
    }
    push_reader_decimal_types(output, format!("{label}.reader_decimal_types"), &chunk.reader_decimal_types);
    push_token(
        output,
        format!("{label}.reader_decimal_types_footer_verified"),
        chunk.reader_decimal_types_footer_verified,
    );
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};
    use query_planning::{
        chunk::SourceChunk,
        execution_contracts::SourceChunkPlanningMode,
        plan::NodeId,
        source::{LocalParquetFile, LocalParquetSource, ParquetSourceLocation, ResolvedParquetSource},
    };

    use super::GpuLocalParquetSourceCacheKey;

    #[test]
    fn equivalent_source_contracts_share_an_identity_and_limits_do_not() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));
        let source =
            LocalParquetSource::new("/tmp/algeon-source-identity.parquet", Arc::clone(&schema), vec!["value".into()]);
        let equivalent = LocalParquetSource::new("/tmp/algeon-source-identity.parquet", schema, vec!["value".into()]);

        let key = GpuLocalParquetSourceCacheKey::from_source(&source);
        assert_eq!(key, GpuLocalParquetSourceCacheKey::from_source(&equivalent));
        assert_ne!(key, GpuLocalParquetSourceCacheKey::from_source(&source.with_scan_limit(Some(7))));
    }

    #[test]
    fn graph_vertex_footer_bound_is_part_of_the_source_identity() {
        let schema = Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false)]));
        let source = |vertex_upper_bound| {
            LocalParquetSource::from_files(
                vec![
                    LocalParquetFile::new("/tmp/algeon-graph-source.parquet")
                        .with_graph_vertex_upper_bound(vertex_upper_bound),
                ],
                Arc::clone(&schema),
                vec!["src".to_string()],
            )
        };

        assert_ne!(
            GpuLocalParquetSourceCacheKey::from_source(&source(10)),
            GpuLocalParquetSourceCacheKey::from_source(&source(11)),
            "a footer-bound graph cardinality changes admission semantics and must not share a source identity"
        );
    }

    #[test]
    fn bounded_chunk_identity_keeps_row_group_and_limit_boundaries_distinct() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));
        let source = LocalParquetSource::new("/tmp/algeon-source-identity.parquet", schema, vec!["value".into()]);
        let first = test_chunk(0, vec![0], None);
        let same = test_chunk(0, vec![0], None);
        let different_row_group = test_chunk(0, vec![1], None);
        let different_limit = test_chunk(0, vec![0], Some(7));

        assert_eq!(
            GpuLocalParquetSourceCacheKey::from_chunk(&source, &first),
            GpuLocalParquetSourceCacheKey::from_chunk(&source, &same),
        );
        assert_ne!(
            GpuLocalParquetSourceCacheKey::from_chunk(&source, &first),
            GpuLocalParquetSourceCacheKey::from_chunk(&source, &different_row_group),
        );
        assert_ne!(
            GpuLocalParquetSourceCacheKey::from_chunk(&source, &first),
            GpuLocalParquetSourceCacheKey::from_chunk(&source, &different_limit),
        );
    }

    fn test_chunk(index: usize, row_groups: Vec<usize>, scan_limit_rows: Option<usize>) -> SourceChunk {
        let path = PathBuf::from("/tmp/algeon-source-identity.parquet");
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(0),
            chunk_index: index,
            file_index: 0,
            location: ParquetSourceLocation::LocalPath(path.clone()),
            path: path.clone(),
            resolved_source: ResolvedParquetSource::local_kvikio(path.display().to_string()),
            stable_id: "identity-source".to_string(),
            row_group_indexes: row_groups,
            reads_all_row_groups: false,
            known_file_bytes: None,
            known_row_group_bytes: None,
            known_projected_compressed_bytes: None,
            known_projected_uncompressed_bytes: None,
            known_pure_filter_compressed_bytes: None,
            known_pure_filter_uncompressed_bytes: None,
            estimated_chunk_bytes: None,
            estimated_source_read_bytes: None,
            exact_source_read_preflight_bytes: None,
            source_file_row_groups: None,
            source_file_row_group_bytes: None,
            source_file_projected_compressed_bytes: None,
            source_file_projected_uncompressed_bytes: None,
            source_file_pure_filter_compressed_bytes: None,
            source_file_pure_filter_uncompressed_bytes: None,
            source_file_row_group_rows: None,
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["value".to_string()],
            partition_values: Vec::new(),
            zero_column_count: false,
            known_row_count: None,
            scan_limit_rows,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test".to_string(),
        }
    }
}
