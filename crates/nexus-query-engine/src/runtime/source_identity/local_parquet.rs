//! Stable local-Parquet source identities.

use super::fingerprint::*;
use crate::source::LocalParquetSource;

/// Stable identity for one GPU-local Parquet source scan or chunk.
///
/// The key fingerprints the complete source scan contract: schema, projection,
/// predicates, resolved read contract, file content identity, deletes, and
/// chunk boundaries. It is local-process data, never a persistence format.
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
    /// Build a key for an unbounded local-Parquet source scan.
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::GpuLocalParquetSourceCacheKey;
    use crate::source::LocalParquetSource;

    #[test]
    fn equivalent_source_contracts_share_an_identity_and_limits_do_not() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));
        let source =
            LocalParquetSource::new("/tmp/nexus-source-identity.parquet", Arc::clone(&schema), vec!["value".into()]);
        let equivalent = LocalParquetSource::new("/tmp/nexus-source-identity.parquet", schema, vec!["value".into()]);

        let key = GpuLocalParquetSourceCacheKey::from_source(&source);
        assert_eq!(key, GpuLocalParquetSourceCacheKey::from_source(&equivalent));
        assert_ne!(key, GpuLocalParquetSourceCacheKey::from_source(&source.with_scan_limit(Some(7))));
    }
}
