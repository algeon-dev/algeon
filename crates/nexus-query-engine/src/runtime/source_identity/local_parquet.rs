//! Stable local-Parquet source identities.

use super::fingerprint::*;
use crate::{chunk::SourceChunk, source::LocalParquetSource};

/// Stable identity for one GPU-local Parquet source scan or chunk.
///
/// The key fingerprints the source scan contract: schema, projected/read
/// columns, predicates, file identities, row groups, scan limit, and chunk
/// boundaries. It is stable for equivalent local source scans, but is not a
/// cross-version or cross-process persistence format.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct GpuLocalParquetSourceCacheKey {
    fingerprint: String,
    diagnostic_fingerprint: String,
}

impl std::fmt::Debug for GpuLocalParquetSourceCacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLocalParquetSourceCacheKey").field("fingerprint", &self.diagnostic_fingerprint).finish()
    }
}

impl GpuLocalParquetSourceCacheKey {
    /// Build a key for an unbounded local Parquet source scan.
    pub(crate) fn from_source(source: &LocalParquetSource) -> Self {
        let mut fingerprint = String::new();
        push_token(&mut fingerprint, "mode", "unbounded_local_parquet_source");
        push_source_scan_contract(&mut fingerprint, source);
        push_token(
            &mut fingerprint,
            "scan_limit",
            source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        );
        push_token(&mut fingerprint, "file_count", source.files.len().to_string());
        for (index, file) in source.files.iter().enumerate() {
            push_token(&mut fingerprint, format!("file.{index}.location"), file.location.diagnostic_location());
            push_token(&mut fingerprint, format!("file.{index}.stable_id"), &file.stable_id);
            push_resolved_source_contract(&mut fingerprint, format!("file.{index}.resolved"), &file.resolved_source);
            push_token(
                &mut fingerprint,
                format!("file.{index}.size"),
                file.file_size_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
            );
            push_optional_usize_vec(&mut fingerprint, format!("file.{index}.row_groups"), file.row_groups.as_deref());
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.row_group_bytes"),
                file.row_group_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.projected_compressed"),
                file.projected_row_group_compressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.projected_uncompressed"),
                file.projected_row_group_uncompressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.pure_filter_compressed"),
                file.pure_filter_row_group_compressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.pure_filter_uncompressed"),
                file.pure_filter_row_group_uncompressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.row_group_rows"),
                file.row_group_rows.as_deref(),
            );
            push_reader_decimal_types(
                &mut fingerprint,
                format!("file.{index}.reader_decimal_types"),
                &file.reader_decimal_types,
            );
            push_position_delete_files(
                &mut fingerprint,
                format!("file.{index}.position_delete_files"),
                &file.position_delete_files,
            );
            push_equality_delete_files(
                &mut fingerprint,
                format!("file.{index}.equality_delete_files"),
                &file.equality_delete_files,
            );
            let partition_values = file.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
            push_string_vec(&mut fingerprint, format!("file.{index}.partition_values"), &partition_values);
        }
        let mut diagnostic_fingerprint = fingerprint.clone();
        redact_source_cache_fingerprint(&mut diagnostic_fingerprint, source);
        Self { fingerprint, diagnostic_fingerprint }
    }

    /// Build a key for one bounded local Parquet source chunk.
    pub(crate) fn from_source_chunk(source: &LocalParquetSource, chunk: &SourceChunk) -> Self {
        let mut fingerprint = String::new();
        push_token(&mut fingerprint, "mode", "bounded_local_parquet_source_chunk");
        push_source_scan_contract(&mut fingerprint, source);
        push_token(
            &mut fingerprint,
            "source_scan_limit",
            source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        );
        push_token(
            &mut fingerprint,
            "chunk_scan_limit",
            chunk.scan_limit_rows.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        );
        push_token(&mut fingerprint, "chunk.file_index", chunk.file_index.to_string());
        push_token(&mut fingerprint, "chunk.location", chunk.location.diagnostic_location());
        push_token(&mut fingerprint, "chunk.stable_id", &chunk.stable_id);
        push_resolved_source_contract(&mut fingerprint, "chunk.resolved", &chunk.resolved_source);
        push_token(&mut fingerprint, "chunk.reads_all_row_groups", chunk.reads_all_row_groups.to_string());
        push_usize_vec(&mut fingerprint, "chunk.row_groups", &chunk.row_group_indexes);
        push_optional_u64(&mut fingerprint, "chunk.known_file_bytes", chunk.known_file_bytes);
        push_optional_u64(&mut fingerprint, "chunk.known_row_group_bytes", chunk.known_row_group_bytes);
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_projected_compressed_bytes",
            chunk.known_projected_compressed_bytes,
        );
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_projected_uncompressed_bytes",
            chunk.known_projected_uncompressed_bytes,
        );
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_pure_filter_compressed_bytes",
            chunk.known_pure_filter_compressed_bytes,
        );
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_pure_filter_uncompressed_bytes",
            chunk.known_pure_filter_uncompressed_bytes,
        );
        push_optional_usize_vec(
            &mut fingerprint,
            "chunk.source_file_row_groups",
            chunk.source_file_row_groups.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_row_group_bytes",
            chunk.source_file_row_group_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_projected_compressed_bytes",
            chunk.source_file_projected_compressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_projected_uncompressed_bytes",
            chunk.source_file_projected_uncompressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_pure_filter_compressed_bytes",
            chunk.source_file_pure_filter_compressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_pure_filter_uncompressed_bytes",
            chunk.source_file_pure_filter_uncompressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_row_group_rows",
            chunk.source_file_row_group_rows.as_deref(),
        );
        push_string_vec(&mut fingerprint, "chunk.read_columns", &chunk.read_columns);
        push_reader_decimal_types(&mut fingerprint, "chunk.reader_decimal_types", &chunk.reader_decimal_types);
        if let Some(file) = source.files.get(chunk.file_index) {
            push_position_delete_files(&mut fingerprint, "chunk.position_delete_files", &file.position_delete_files);
            push_equality_delete_files(&mut fingerprint, "chunk.equality_delete_files", &file.equality_delete_files);
        } else {
            push_token(&mut fingerprint, "chunk.position_delete_files.len", "missing");
            push_token(&mut fingerprint, "chunk.equality_delete_files.len", "missing");
        }
        push_token(
            &mut fingerprint,
            "chunk.reader_decimal_types_footer_verified",
            chunk.reader_decimal_types_footer_verified.to_string(),
        );
        let partition_values = chunk.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
        push_string_vec(&mut fingerprint, "chunk.partition_values", &partition_values);
        let mut diagnostic_fingerprint = fingerprint.clone();
        redact_source_chunk_cache_fingerprint(&mut diagnostic_fingerprint, source, chunk);
        Self { fingerprint, diagnostic_fingerprint }
    }

    /// Escaped textual fingerprint for diagnostics and tests.
    pub(crate) fn as_str(&self) -> &str {
        &self.diagnostic_fingerprint
    }
}
