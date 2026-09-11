use super::{DataFrame, ExecutionContext, LocalParquetFile, LocalParquetSource, SourceChunk, cudf_query_error};

mod cache;
mod maintenance;
mod policy;
mod s3;

#[cfg(test)]
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::Ordering,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub(super) use cache::{
    BackgroundLakehouseCacheFillEnqueue, LakehouseOptimizedBatchCacheEntry, acquire_lakehouse_cache_chunk_read_pin,
    acquire_lakehouse_cache_file_read_pin, acquire_lakehouse_optimized_batch_read_pin,
    enqueue_background_lakehouse_cache_chunk_fill, enqueue_background_lakehouse_cache_file_fill,
    ensure_local_cache_parquet_chunk, ensure_local_cache_parquet_file, local_cache_optimized_batch_entry,
    publish_lakehouse_optimized_batch_frame,
};
#[cfg(test)]
use cache::{
    LAKEHOUSE_CACHE_READ_PIN_COUNTER, LakehouseCacheReadPin, collect_lakehouse_cache_entries,
    complete_cache_file_matches_expected_size, evict_lakehouse_cache, lakehouse_cache_entry_has_read_pin,
    lakehouse_cache_io_error, lakehouse_cache_lock_path, lakehouse_cache_read_pin_path,
    local_cache_optimized_batch_signature, optimized_batch_uses_identity_source_frame, stable_log_hash,
};
#[cfg(test)]
use policy::LakehouseCachePolicy;
#[cfg(test)]
use s3::object_store_bucket_fact;
pub(in crate::exec::source) use s3::parse_s3_source_uri;
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

    use super::*;
    use crate::{
        chunk::SourceChunkBatch,
        plan::NodeId,
        runtime::SourceChunkPlanningMode,
        source::{
            CredentialMode, LakehouseCacheFillMode, LakehouseCacheStatus, LocalParquetFile, MissingColumn,
            ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, ResolvedParquetSource,
            SourceColumnRef, parquet_file_path_fact, parquet_file_stable_id_fact,
        },
    };

    #[test]
    fn complete_cache_file_hit_requires_expected_size() {
        let unique = LAKEHOUSE_CACHE_READ_PIN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("algeon-cache-size-test-{}-{unique}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("part-0.parquet");
        fs::write(&path, [0_u8; 8]).unwrap();

        assert!(complete_cache_file_matches_expected_size(&path, Some(8)));
        assert!(!complete_cache_file_matches_expected_size(&path, Some(9)));
        assert!(!complete_cache_file_matches_expected_size(&path, None));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn lakehouse_cache_hit_without_content_identity_is_rejected() {
        let root = unique_temp_cache_root("cache-hit-without-identity");
        let cache_path = write_cache_entry(&root, "v1/s3/bucket/etagless/part-0.parquet", 16);
        let context = ExecutionContext::builder().build().unwrap();
        let unidentified = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(
                ResolvedParquetSource::local_cache_kvikio(
                    "s3://bucket/table/part-0.parquet",
                    cache_path.to_str().unwrap(),
                    CredentialMode::SharedProcessEnv,
                    LakehouseCacheStatus::Hit,
                )
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap()),
            )
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(16);

        // Size matches, but without a content identity the bytes cannot be
        // attributed to the current object: the hit must not be served.
        let error = ensure_local_cache_parquet_file(&context, &unidentified).unwrap_err();
        assert_eq!(error.operation(), "query_engine.lakehouse_cache.content_identity");

        // Control: the same entry with a caller-provided identity is served.
        let identified = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(
                ResolvedParquetSource::local_cache_kvikio(
                    "s3://bucket/table/part-0.parquet",
                    cache_path.to_str().unwrap(),
                    CredentialMode::SharedProcessEnv,
                    LakehouseCacheStatus::Hit,
                )
                .with_content_identity("etag:abc123")
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap()),
            )
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(16);
        ensure_local_cache_parquet_file(&context, &identified).unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_fill_without_content_identity_is_skipped() {
        let root = unique_temp_cache_root("background-fill-without-identity");
        let cache_path = root.join("v1/s3/bucket/etagless/part-0.parquet");
        let context = ExecutionContext::builder().build().unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap())
                .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background);
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(1024);

        let outcome = enqueue_background_lakehouse_cache_file_fill(&context, &file).unwrap();

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("missing_content_identity"));
        assert!(!lakehouse_cache_lock_path(&cache_path).exists());
        let _ = fs::remove_dir_all(root);
    }

    fn optimized_cache_source() -> LocalParquetSource {
        LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/source.parquet")],
            Arc::new(Schema::new(vec![
                Field::new("c0", ArrowDataType::Int32, false),
                Field::new("c1", ArrowDataType::Int64, false),
            ])),
            vec!["c0".to_string(), "c1".to_string()],
        )
    }

    fn optimized_cache_batch() -> SourceChunkBatch {
        let chunks = (0..8).map(optimized_cache_chunk).collect::<Vec<_>>();
        SourceChunkBatch {
            batch_index: 0,
            chunks,
            estimated_source_read_bytes: Some(8 * 1024),
            source_batching_reason: "batched".to_string(),
        }
    }

    fn optimized_cache_chunk(index: usize) -> SourceChunk {
        let reader_path = format!("/tmp/cache/file-{index}.parquet");
        let resolved_source = ResolvedParquetSource::local_cache_kvikio(
            format!("s3://bucket/file-{index}.parquet"),
            reader_path.clone(),
            CredentialMode::SharedProcessEnv,
            LakehouseCacheStatus::Hit,
        )
        .with_storage_domain("bucket@endpoint")
        .with_content_identity(format!("size=1024,row_count=10,file={index}"))
        .with_lakehouse_cache_config("/tmp/lakehouse-cache", None);
        SourceChunk {
            source_node_id: NodeId::from_index_for_internal_use(1),
            chunk_index: index,
            file_index: index,
            location: ParquetSourceLocation::object_store_uri(format!("s3://bucket/file-{index}.parquet")).unwrap(),
            path: PathBuf::from(reader_path),
            resolved_source,
            stable_id: format!("file-{index}"),
            row_group_indexes: vec![0],
            reads_all_row_groups: false,
            known_file_bytes: Some(1024),
            known_row_group_bytes: Some(1024),
            known_projected_compressed_bytes: Some(512),
            known_projected_uncompressed_bytes: Some(1024),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(1024),
            estimated_source_read_bytes: Some(1024),
            exact_source_read_preflight_bytes: Some(1024),
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![1024]),
            source_file_projected_compressed_bytes: Some(vec![512]),
            source_file_projected_uncompressed_bytes: Some(vec![1024]),
            source_file_pure_filter_compressed_bytes: Some(vec![0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0]),
            source_file_row_group_rows: Some(vec![10]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["c0".to_string(), "c1".to_string()],
            partition_values: vec![],
            zero_column_count: false,
            known_row_count: Some(10),
            scan_limit_rows: None,
            size_estimate_based: false,
            source_batching_reason: None,
            source_chunk_planning_mode: SourceChunkPlanningMode::Conservative,
            source_chunk_planning_reason: "test_policy".to_string(),
        }
    }

    #[test]
    fn optimized_batch_entry_requires_identity_source_frame() {
        let source = optimized_cache_source();
        let batch = optimized_cache_batch();
        assert!(local_cache_optimized_batch_entry(&source, &batch).is_none());
        assert!(optimized_batch_uses_identity_source_frame(&source));

        let mut alias_source = source.clone();
        alias_source.projected_output_columns = vec!["c1".to_string()];
        alias_source.projected_output_column_source_refs = vec![SourceColumnRef::File(1)];
        assert!(!optimized_batch_uses_identity_source_frame(&alias_source));
        assert!(local_cache_optimized_batch_entry(&alias_source, &batch).is_none());

        let pure_filter_source = source.clone().with_column_roles(vec!["c0".to_string()], vec!["c1".to_string()]);
        assert!(!optimized_batch_uses_identity_source_frame(&pure_filter_source));
        assert!(local_cache_optimized_batch_entry(&pure_filter_source, &batch).is_none());

        let mut missing_source = source;
        missing_source.missing_columns.push(MissingColumn {
            name: "missing".to_string(),
            data_type: ArrowDataType::Int32,
            nullable: true,
        });
        assert!(!optimized_batch_uses_identity_source_frame(&missing_source));
        assert!(local_cache_optimized_batch_entry(&missing_source, &batch).is_none());
    }

    #[test]
    fn optimized_batch_entry_is_scoped_by_resolved_source_contract() {
        let source = optimized_cache_source();
        let batch = optimized_cache_batch();
        let first_signature = local_cache_optimized_batch_signature(&source, &batch);

        let mut different_profile_batch = batch.clone();
        for chunk in &mut different_profile_batch.chunks {
            chunk.resolved_source = chunk
                .resolved_source
                .clone()
                .with_storage_profile_id("different-profile")
                .with_lakehouse_cache_path(format!("/tmp/other-cache/file-{}.parquet", chunk.file_index));
        }
        let second_signature = local_cache_optimized_batch_signature(&source, &different_profile_batch);

        assert_ne!(first_signature, second_signature);

        let mut different_decimal_batch = batch.clone();
        for chunk in &mut different_decimal_batch.chunks {
            chunk.reader_decimal_types =
                vec![ParquetReaderDecimalType::new("c0", ParquetReaderDecimalTypeId::Decimal64, -2)];
            chunk.reader_decimal_types_footer_verified = true;
        }
        let decimal_signature = local_cache_optimized_batch_signature(&source, &different_decimal_batch);

        assert_ne!(first_signature, decimal_signature);
    }

    #[test]
    fn parse_s3_source_uri_preserves_storage_profile_id() {
        let location = parse_s3_source_uri("s3://bucket/prefix/file.parquet", Some("rest-s3".to_string())).unwrap();

        assert_eq!(location.bucket, "bucket");
        assert_eq!(location.key.as_ref(), "prefix/file.parquet");
        assert_eq!(location.storage_profile_id.as_deref(), Some("rest-s3"));
    }

    fn unique_temp_cache_root(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("query-engine-{test_name}-{nonce}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_cache_entry(root: &Path, relative: &str, bytes: usize) -> PathBuf {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, vec![b'x'; bytes]).unwrap();
        path
    }

    #[test]
    fn background_lakehouse_cache_fill_enqueue_reports_saturated_gate_without_blocking() {
        let root = unique_temp_cache_root("background-fill-saturated");
        let cache_path = root.join("v1/s3/bucket/content/part-0.parquet");
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap())
                .with_content_identity("test-content-identity")
                .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background);
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(1024);

        let outcome = enqueue_background_lakehouse_cache_file_fill(&context, &file).unwrap();

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("fill_concurrency_saturated"));
        assert!(!lakehouse_cache_lock_path(&cache_path).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_lakehouse_cache_fill_removes_stale_lock_before_enqueue() {
        let root = unique_temp_cache_root("background-fill-stale-lock");
        let cache_path = root.join("v1/s3/bucket/content/part-0.parquet");
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        let lock_path = lakehouse_cache_lock_path(&cache_path);
        fs::write(&lock_path, b"pid=4294967295\n").unwrap();
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap())
                .with_content_identity("test-content-identity")
                .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background);
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(1024);

        let outcome = enqueue_background_lakehouse_cache_file_fill(&context, &file).unwrap();

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("fill_concurrency_saturated"));
        assert!(!lock_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_lakehouse_cache_fill_removes_stale_malformed_lock_before_enqueue() {
        let root = unique_temp_cache_root("background-fill-malformed-stale-lock");
        let cache_path = root.join("v1/s3/bucket/content/part-0.parquet");
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        let lock_path = lakehouse_cache_lock_path(&cache_path);
        fs::write(&lock_path, b"").unwrap();
        thread::sleep(LakehouseCachePolicy::DEFAULT_MALFORMED_CACHE_LOCK_STALE_AGE + Duration::from_millis(25));
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap())
                .with_content_identity("test-content-identity")
                .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background);
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(1024);

        let outcome = enqueue_background_lakehouse_cache_file_fill(&context, &file).unwrap();

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("fill_concurrency_saturated"));
        assert!(!lock_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_lakehouse_cache_fill_enqueue_reports_saturated_byte_budget() {
        let root = unique_temp_cache_root("background-fill-byte-budget");
        let cache_path = root.join("v1/s3/bucket/content/part-0.parquet");
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_budget_bytes(512).build().unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap())
                .with_content_identity("test-content-identity")
                .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background);
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(1024);

        let outcome = enqueue_background_lakehouse_cache_file_fill(&context, &file).unwrap();

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("fill_byte_budget_saturated"));
        assert!(!lakehouse_cache_lock_path(&cache_path).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_lakehouse_cache_fill_accepts_remote_kvikio_sources() {
        let root = unique_temp_cache_root("background-fill-remote-kvikio");
        let cache_path = root.join("v1/s3/bucket/content/part-0.parquet");
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_kvikio("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
                .with_lakehouse_cache_config(root.to_str().unwrap(), None)
                .with_lakehouse_cache_path(cache_path.to_str().unwrap())
                .with_content_identity("test-content-identity")
                .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background);
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(resolved_source)
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(1024);

        let outcome = enqueue_background_lakehouse_cache_file_fill(&context, &file).unwrap();

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("fill_concurrency_saturated"));
        assert!(!lakehouse_cache_lock_path(&cache_path).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_lakehouse_cache_fill_log_ids_hash_sensitive_values() {
        let sensitive = "s3://access:secret@bucket/path/file.parquet?token=do-not-leak";
        let first = stable_log_hash(sensitive);
        let second = stable_log_hash(sensitive);

        assert_eq!(first, second);
        assert_eq!(first.len(), 16);
        assert!(!first.contains("s3://"));
        assert!(!first.contains("secret"));
        assert!(!first.contains("do-not-leak"));
    }

    #[test]
    fn cache_maintenance_object_store_error_facts_redact_source_identity() {
        let sensitive = "s3://secret-bucket/private/file.parquet?X-Amz-Signature=do-not-leak";
        let file = LocalParquetFile::object_store_uri(sensitive).unwrap().with_stable_id(sensitive);

        let path = parquet_file_path_fact(&file);
        let stable_id = parquet_file_stable_id_fact(&file);

        assert!(path.starts_with("object_store:s3:fnv64:"));
        assert!(stable_id.starts_with("object_store:s3:stable_id_fnv64:"));
        for diagnostic in [&path, &stable_id] {
            for sensitive_part in ["secret-bucket", "private/file", "X-Amz-Signature", "do-not-leak"] {
                assert!(!diagnostic.contains(sensitive_part), "diagnostic leaked {sensitive_part}: {diagnostic}");
            }
        }
    }

    #[test]
    fn lakehouse_cache_error_facts_hash_source_derived_values() {
        let sensitive_path =
            PathBuf::from("/tmp/cache/s3/secret-bucket/private/file.parquet?X-Amz-Signature=do-not-leak");
        let sensitive_stable_id = "s3://secret-bucket/private/file.parquet?X-Amz-Signature=do-not-leak";
        let error = lakehouse_cache_io_error(
            std::io::Error::other("permission denied"),
            "query_engine.test.cache_error",
            "failed cache operation",
            &sensitive_path,
            sensitive_stable_id,
            None,
            None,
        );

        let rendered = error.to_string();
        assert!(error.fact_value("path").is_some_and(|value| value.starts_with("lakehouse_cache_path:fnv64:")));
        assert!(
            error.fact_value("stable_id").is_some_and(|value| value.starts_with("lakehouse_cache_stable_id:fnv64:"))
        );
        assert!(object_store_bucket_fact("secret-bucket").starts_with("object_store_bucket:fnv64:"));
        for sensitive in ["secret-bucket", "private/file", "X-Amz-Signature", "do-not-leak"] {
            assert!(!rendered.contains(sensitive), "error leaked {sensitive}: {rendered}");
        }
    }

    #[test]
    fn lakehouse_cache_eviction_keeps_protected_entry_and_removes_old_data_files() {
        let root = unique_temp_cache_root("eviction");
        let old_a = write_cache_entry(&root, "v1/s3/bucket/content/a.parquet__h1", 4);
        let old_b = write_cache_entry(&root, "v1/s3/bucket/content/b.parquet__h2", 4);
        let protected = write_cache_entry(&root, "v1/s3/bucket/content/c.parquet__h3", 4);
        let old_a_sidecar = crate::parquet_metadata::parquet_footer_sidecar_path(&old_a);
        fs::write(&old_a_sidecar, b"sidecar").unwrap();
        let lock_path = lakehouse_cache_lock_path(&old_b);
        fs::write(&lock_path, b"locked").unwrap();

        evict_lakehouse_cache(Some(root.to_str().unwrap()), &protected, Some(4));

        assert!(protected.exists());
        assert!(!old_a.exists());
        assert!(!old_a_sidecar.exists());
        assert!(old_b.exists());
        assert!(lock_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn lakehouse_cache_eviction_skips_active_read_pins() {
        let root = unique_temp_cache_root("read-pin-eviction");
        let old = write_cache_entry(&root, "v1/s3/bucket/content/old.parquet__h1", 8);
        let active = write_cache_entry(&root, "v1/s3/bucket/content/active.parquet__h2", 4);
        let protected = write_cache_entry(&root, "v1/s3/bucket/content/protected.parquet__h3", 4);
        let pin_path = lakehouse_cache_read_pin_path(&active);
        fs::write(&pin_path, b"pid=123").unwrap();

        let entry_paths = collect_lakehouse_cache_entries(&root, &protected)
            .unwrap()
            .into_iter()
            .map(|entry| entry.path)
            .collect::<Vec<_>>();
        assert!(entry_paths.contains(&old));
        assert!(entry_paths.contains(&active));
        assert!(entry_paths.contains(&protected));
        assert!(!entry_paths.contains(&pin_path));

        evict_lakehouse_cache(Some(root.to_str().unwrap()), &protected, Some(8));

        assert!(!old.exists());
        assert!(active.exists());
        assert!(protected.exists());
        assert!(pin_path.exists());

        let pin = LakehouseCacheReadPin { path: Some(pin_path.clone()) };
        drop(pin);
        assert!(!pin_path.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn lakehouse_cache_read_pin_check_removes_stale_pid_pins_when_proc_is_available() {
        if !Path::new("/proc").is_dir() {
            return;
        }
        let root = unique_temp_cache_root("stale-read-pin");
        let entry = write_cache_entry(&root, "v1/s3/bucket/content/active.parquet__h2", 4);
        let file_name = entry.file_name().unwrap().to_str().unwrap();
        let stale_pin = entry.with_file_name(format!(".{file_name}.read.4294967295.1.0.pin"));
        fs::write(&stale_pin, b"pid=4294967295").unwrap();

        assert!(!lakehouse_cache_entry_has_read_pin(&entry));
        assert!(!stale_pin.exists());
        let _ = fs::remove_dir_all(root);
    }
}
