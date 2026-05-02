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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::GpuLocalParquetSourceCacheKey;
    use crate::{
        chunk::{SourceChunkPlanningPolicy, plan_local_parquet_source_chunks},
        plan::QueryPlanBuilder,
        source::{
            CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, LocalParquetFile,
            LocalParquetSource, ParquetPositionDeleteFile, ParquetReaderDecimalType, ParquetReaderDecimalTypeId,
            ResolvedParquetSource, SourceNode,
        },
    };

    #[test]
    fn keys_isolate_resolved_contracts_and_redact_diagnostic_values() {
        let direct = source_with_contract(remote_direct_contract());
        let cache_policy = source_with_contract(cache_policy_contract());
        let decimal = source_with_contract(remote_direct_contract()).with_reader_decimal_contract();

        let direct_key = GpuLocalParquetSourceCacheKey::from_source(&direct);
        let cache_policy_key = GpuLocalParquetSourceCacheKey::from_source(&cache_policy);
        let decimal_key = GpuLocalParquetSourceCacheKey::from_source(&decimal);
        assert_ne!(direct_key, cache_policy_key, "source keys include resolved read/cache/credential settings");
        assert_ne!(direct_key, decimal_key, "source keys include reader decimal settings");
        assert!(direct_key.as_str().contains("file.0.resolved.read_strategy=remote_direct"));
        assert!(cache_policy_key.as_str().contains("file.0.resolved.lakehouse_cache_status=miss"));
        assert!(decimal_key.as_str().contains("file.0.reader_decimal_types.0.type_id=decimal64"));
        assert!(!direct_key.as_str().contains("query_variant=a"));
        assert!(!cache_policy_key.as_str().contains("/tmp/nexus-lakehouse-cache"));

        let mut builder = QueryPlanBuilder::new();
        let direct_id = builder.add_source(SourceNode::LocalParquet(direct.clone()));
        let cache_policy_id = builder.add_source(SourceNode::LocalParquet(cache_policy.clone()));
        let decimal_id = builder.add_source(SourceNode::LocalParquet(decimal.clone()));
        let direct_chunk =
            plan_local_parquet_source_chunks(direct_id, &direct, SourceChunkPlanningPolicy::new(None, 1))
                .expect("direct source chunk planning succeeds");
        let cache_policy_chunk =
            plan_local_parquet_source_chunks(cache_policy_id, &cache_policy, SourceChunkPlanningPolicy::new(None, 1))
                .expect("cache-policy source chunk planning succeeds");
        let decimal_chunk =
            plan_local_parquet_source_chunks(decimal_id, &decimal, SourceChunkPlanningPolicy::new(None, 1))
                .expect("decimal source chunk planning succeeds");
        let direct_chunk_key = GpuLocalParquetSourceCacheKey::from_source_chunk(&direct, &direct_chunk.chunks[0]);
        let cache_policy_chunk_key =
            GpuLocalParquetSourceCacheKey::from_source_chunk(&cache_policy, &cache_policy_chunk.chunks[0]);
        let decimal_chunk_key = GpuLocalParquetSourceCacheKey::from_source_chunk(&decimal, &decimal_chunk.chunks[0]);
        assert_ne!(
            direct_chunk_key, cache_policy_chunk_key,
            "chunk keys include resolved read/cache/credential settings"
        );
        assert_ne!(direct_chunk_key, decimal_chunk_key, "chunk keys include reader decimal settings");
        assert!(decimal_chunk_key.as_str().contains("chunk.reader_decimal_types.0.type_id=decimal64"));
    }

    #[test]
    fn keys_isolate_position_delete_files_without_disclosing_locations() {
        let base = source_with_contract(remote_direct_contract());
        let delete_a = source_with_contract(remote_direct_contract())
            .with_position_delete_file("delete-a", "s3://warehouse/customer_address-delete-a.parquet");
        let delete_b = source_with_contract(remote_direct_contract())
            .with_position_delete_file("delete-b", "s3://warehouse/customer_address-delete-b.parquet");

        let base_key = GpuLocalParquetSourceCacheKey::from_source(&base);
        let delete_a_key = GpuLocalParquetSourceCacheKey::from_source(&delete_a);
        let delete_b_key = GpuLocalParquetSourceCacheKey::from_source(&delete_b);
        assert_ne!(base_key, delete_a_key, "source keys include position delete files");
        assert_ne!(delete_a_key, delete_b_key, "source keys distinguish delete file identities");
        assert!(delete_a_key.as_str().contains("file.0.position_delete_files.0.id=delete-a"));
        assert!(!delete_a_key.as_str().contains("customer_address-delete-a.parquet"));

        let mut builder = QueryPlanBuilder::new();
        let base_id = builder.add_source(SourceNode::LocalParquet(base.clone()));
        let delete_id = builder.add_source(SourceNode::LocalParquet(delete_a.clone()));
        let base_chunk = plan_local_parquet_source_chunks(base_id, &base, SourceChunkPlanningPolicy::new(None, 1))
            .expect("base source chunk planning succeeds");
        let delete_chunk =
            plan_local_parquet_source_chunks(delete_id, &delete_a, SourceChunkPlanningPolicy::new(None, 1))
                .expect("delete source chunk planning succeeds");
        let base_chunk_key = GpuLocalParquetSourceCacheKey::from_source_chunk(&base, &base_chunk.chunks[0]);
        let delete_chunk_key = GpuLocalParquetSourceCacheKey::from_source_chunk(&delete_a, &delete_chunk.chunks[0]);
        assert_ne!(base_chunk_key, delete_chunk_key, "chunk keys include position delete files");
        assert!(delete_chunk_key.as_str().contains("chunk.position_delete_files.len=1"));
        assert!(!delete_chunk_key.as_str().contains("customer_address-delete-a.parquet"));
    }

    fn source_with_contract(resolved_source: ResolvedParquetSource) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
            .expect("resolved source is valid")
            .with_stable_id("customer_address")
            .with_row_group_bytes(vec![2 * 1024 * 1024])
            .with_row_group_rows(vec![100_000]);
        LocalParquetSource::from_files(vec![file], schema, vec!["id".to_string()])
    }

    trait LocalParquetSourceTestExt {
        fn with_reader_decimal_contract(self) -> Self;
        fn with_position_delete_file(self, id: &str, location: &str) -> Self;
    }

    impl LocalParquetSourceTestExt for LocalParquetSource {
        fn with_reader_decimal_contract(mut self) -> Self {
            self.files[0].reader_decimal_types =
                vec![ParquetReaderDecimalType::new("id", ParquetReaderDecimalTypeId::Decimal64, -2)];
            self.refresh_diagnostics();
            self
        }

        fn with_position_delete_file(mut self, id: &str, location: &str) -> Self {
            let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
                id,
                ResolvedParquetSource::remote_kvikio(location, CredentialMode::SharedProcessEnv),
            )
            .expect("resolved delete source is valid");
            self.files[0].position_delete_files = vec![delete_file];
            self.refresh_diagnostics();
            self
        }
    }

    fn remote_direct_contract() -> ResolvedParquetSource {
        ResolvedParquetSource::remote_kvikio(
            "s3://warehouse/customer_address.parquet?query_variant=a",
            CredentialMode::SharedProcessEnv,
        )
        .with_storage_profile_id("remote-profile-a")
        .with_storage_domain("warehouse@s3")
        .with_content_identity("snapshot=42,file=customer_address,etag=abc")
    }

    fn cache_policy_contract() -> ResolvedParquetSource {
        remote_direct_contract()
            .with_lakehouse_cache_config("/tmp/nexus-lakehouse-cache", Some(128 * 1024 * 1024))
            .with_lakehouse_cache_path("/tmp/nexus-lakehouse-cache/customer_address.parquet")
            .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Enabled)
            .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background)
            .with_cache_status(LakehouseCacheStatus::Miss)
    }
}
