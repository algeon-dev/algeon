use std::time::Instant;

use super::{
    cudf_contract::{
        MiniParquetHostBufferDescriptor, cudf_local_parquet_source_mini_host_buffer_chunk,
        mini_host_buffer_reader_decimal_types_for_chunk, mini_host_buffer_reader_decimal_types_for_file,
    },
    *,
};
use crate::error::{QueryError, QueryErrorCode, Result};

mod lakehouse_cache;
mod metrics;
mod object_store_fetch;
mod object_store_safety;
mod ranged_parquet;

#[cfg(test)]
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::Ordering,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub(super) use lakehouse_cache::{
    BackgroundLakehouseCacheFillEnqueue, LakehouseOptimizedBatchCacheEntry, acquire_lakehouse_cache_chunk_read_pin,
    acquire_lakehouse_cache_file_read_pin, acquire_lakehouse_optimized_batch_read_pin,
    enqueue_background_lakehouse_cache_chunk_fill, enqueue_background_lakehouse_cache_file_fill,
    ensure_local_cache_parquet_chunk, ensure_local_cache_parquet_file, local_cache_optimized_batch_entry,
    publish_lakehouse_optimized_batch_frame,
};
#[cfg(test)]
use lakehouse_cache::{
    LAKEHOUSE_CACHE_READ_PIN_COUNTER, LakehouseCacheReadPin, MALFORMED_LAKEHOUSE_CACHE_LOCK_STALE_AGE,
    collect_lakehouse_cache_entries, complete_cache_file_matches_expected_size, evict_lakehouse_cache,
    lakehouse_cache_entry_has_read_pin, lakehouse_cache_io_error, lakehouse_cache_lock_path,
    lakehouse_cache_read_pin_path, local_cache_optimized_batch_signature, optimized_batch_uses_identity_source_frame,
    stable_log_hash,
};
pub(super) use metrics::HostBufferFetchRuntimeMetrics;
use metrics::{
    parquet_chunk_path_fact, parquet_chunk_stable_id_fact, parquet_file_path_fact, parquet_file_stable_id_fact,
    record_host_buffer_chunk_read_metrics, record_host_buffer_file_read_metrics,
};
pub(super) use object_store_fetch::ScheduledHostParquetFetch;
use object_store_fetch::{
    HostParquetBuffer, start_host_parquet_buffer_fetch, start_host_parquet_buffer_pipeline_fetch,
    try_start_host_parquet_buffer_pipeline_fetch,
};
#[cfg(test)]
use object_store_safety::object_store_bucket_fact;
pub(super) use object_store_safety::parse_s3_source_uri;
#[cfg(test)]
use ranged_parquet::RangedParquetFetchDiagnostics;
use ranged_parquet::{
    attach_fetch_facts as attach_ranged_fetch_facts, attach_file_fetch_facts as attach_ranged_fetch_file_facts,
    chunk_request as ranged_parquet_chunk_request, file_request as ranged_parquet_file_request,
    start_host_parquet_buffer_ranged_fetch,
};

pub(super) fn read_object_store_parquet_file(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    source_node_id: NodeId,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let fetch = start_object_store_parquet_file_fetch(context, source, file)?;
    finish_object_store_parquet_file_fetch(
        context,
        source,
        source_node_id,
        file,
        file_index,
        chunk_index,
        fetch,
        metrics,
    )
}
pub(super) fn start_object_store_parquet_file_fetch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> Result<ScheduledHostParquetFetch> {
    if let Some(request) = ranged_parquet_file_request(source, file) {
        return start_host_parquet_buffer_ranged_fetch(context, &file.location, &file.resolved_source, request);
    }
    start_host_parquet_buffer_fetch(context, &file.location, &file.resolved_source, file.file_size_bytes)
}

pub(super) fn start_object_store_parquet_file_pipeline_fetch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> Result<ScheduledHostParquetFetch> {
    if ranged_parquet_file_request(source, file).is_some() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.pipeline",
            "object-store fetch pipeline cannot start ranged file fetches",
        ));
    }
    let file_size_bytes = pipeline_file_size_bytes(file)?;
    start_host_parquet_buffer_pipeline_fetch(context, &file.location, &file.resolved_source, file_size_bytes)
}

pub(super) fn try_start_object_store_parquet_file_pipeline_fetch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> Result<Option<ScheduledHostParquetFetch>> {
    if ranged_parquet_file_request(source, file).is_some() {
        return Ok(None);
    }
    let Some(file_size_bytes) = file.file_size_bytes else {
        return Ok(None);
    };
    try_start_host_parquet_buffer_pipeline_fetch(context, &file.location, &file.resolved_source, file_size_bytes)
}

pub(super) fn object_store_file_fetch_pipeline_compatible(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> bool {
    file.file_size_bytes.is_some() && ranged_parquet_file_request(source, file).is_none()
}

fn pipeline_file_size_bytes(file: &LocalParquetFile) -> Result<u64> {
    file.file_size_bytes.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_fetch.pipeline",
            "object-store fetch pipeline requires known file sizes",
        )
    })
}

pub(super) fn finish_object_store_parquet_file_fetch(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    source_node_id: NodeId,
    file: &LocalParquetFile,
    file_index: usize,
    chunk_index: usize,
    fetch: ScheduledHostParquetFetch,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let mut fetch =
        finish_with_full_fetch_fallback(context, &file.location, &file.resolved_source, file.file_size_bytes, fetch)?;
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    let decode_timer = Instant::now();
    let output = if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
        let reader_decimal_types = mini_host_buffer_reader_decimal_types_for_file(context, source, file)?;
        let descriptor = MiniParquetHostBufferDescriptor::from_file(
            file,
            source,
            file_index,
            chunk_index,
            fetch.actual_bytes,
            &ranged_fetch.mini_row_group_bytes,
            &ranged_fetch.mini_row_group_rows,
            reader_decimal_types,
        );
        let (mini_source, mini_chunk) =
            cudf_local_parquet_source_mini_host_buffer_chunk(context, source, descriptor)
                .map_err(|error| attach_ranged_fetch_file_facts(error, ranged_fetch, file, file_index, chunk_index))?;
        DataFrame::read_parquet_source_chunk_from_host_buffer_with_evidence(
            context.runtime(),
            &mini_source,
            &mini_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
        )
    } else {
        let (cudf_source, cudf_chunk) =
            cudf_local_parquet_source_file_selection(context, source, file, file_index, chunk_index)?;
        DataFrame::read_parquet_source_chunk_from_host_buffer_with_evidence(
            context.runtime(),
            &cudf_source,
            &cudf_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
        )
    }
    .map_err(|error| {
        let error = QueryError::from_cudf(
            "nexus_query_engine.source_decode.host_buffer_file",
            "failed to decode object-store Parquet source file from host buffer",
            error,
        )
        .with_fact("path", parquet_file_path_fact(file))
        .with_fact("stable_id", parquet_file_stable_id_fact(file))
        .with_fact("file_index", file_index)
        .with_fact("read_columns", source.read_columns.len());
        if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
            attach_ranged_fetch_file_facts(error, ranged_fetch, file, file_index, chunk_index)
        } else {
            error
        }
    })?;
    let decode_elapsed = decode_timer.elapsed();
    let read_evidence = output.read_evidence().cloned();
    let frame = output.into_frame();
    record_host_buffer_file_read_metrics(
        source,
        source_node_id,
        file,
        file_index,
        chunk_index,
        &fetch,
        frame.num_rows() as u64,
        decode_elapsed,
        read_evidence.as_ref(),
        metrics,
    );
    fetch.release_after_decode();
    Ok(frame)
}

pub(super) fn read_object_store_parquet_chunk(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    cudf_source: &cudf::io::source::LocalParquetSource,
    cudf_chunk: &cudf::io::source::SourceChunk,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<ObjectStoreParquetChunkRead> {
    let _source_file = crate::chunk::source_file_for_chunk(
        source,
        chunk,
        "nexus_query_engine.source_decode.host_buffer_chunk.source_file",
    )?;
    let estimated_fetch_bytes = chunk.known_file_bytes.or(chunk.estimated_source_read_bytes);
    let fetch = if let Some(request) = ranged_parquet_chunk_request(source, chunk) {
        start_host_parquet_buffer_ranged_fetch(context, &chunk.location, &chunk.resolved_source, request)?
    } else {
        start_host_parquet_buffer_fetch(context, &chunk.location, &chunk.resolved_source, estimated_fetch_bytes)?
    };
    let mut fetch = finish_with_full_fetch_fallback(
        context,
        &chunk.location,
        &chunk.resolved_source,
        estimated_fetch_bytes,
        fetch,
    )?;
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    let decode_timer = Instant::now();
    let outcome = if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
        let reader_decimal_types = mini_host_buffer_reader_decimal_types_for_chunk(context, source, chunk)?;
        let descriptor = MiniParquetHostBufferDescriptor::from_chunk(
            chunk,
            fetch.actual_bytes,
            &ranged_fetch.mini_row_group_bytes,
            &ranged_fetch.mini_row_group_rows,
            reader_decimal_types,
        );
        let (mini_source, mini_chunk) = cudf_local_parquet_source_mini_host_buffer_chunk(context, source, descriptor)
            .map_err(|error| attach_ranged_fetch_facts(error, ranged_fetch, chunk))?;
        DataFrame::read_parquet_source_chunk_with_budget_from_host_buffer_with_evidence(
            context.runtime(),
            &mini_source,
            &mini_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
            token,
        )
    } else {
        DataFrame::read_parquet_source_chunk_with_budget_from_host_buffer_with_evidence(
            context.runtime(),
            cudf_source,
            cudf_chunk,
            SourceReadMode::ExecutionFrame,
            fetch.bytes(),
            token,
        )
    }
    .map_err(|error| {
        let error = QueryError::from_cudf_with_budget_details(
            "nexus_query_engine.source_decode.host_buffer_chunk",
            "failed to decode bounded object-store Parquet source chunk from host buffer",
            error,
        )
        .with_fact("path", parquet_chunk_path_fact(chunk))
        .with_fact("stable_id", parquet_chunk_stable_id_fact(chunk))
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", chunk.read_columns.len())
        .with_row_group_facts(chunk);
        if let Some(ranged_fetch) = fetch.ranged_fetch.as_ref() {
            attach_ranged_fetch_facts(error, ranged_fetch, chunk)
        } else {
            error
        }
    })?;
    let observed_row_group_count =
        record_host_buffer_chunk_read_metrics(chunk, &fetch, decode_timer.elapsed(), metrics);
    let host_buffer_fetch = fetch.runtime_metrics();
    fetch.release_after_decode();
    Ok(ObjectStoreParquetChunkRead { outcome, observed_row_group_count, host_buffer_fetch })
}

// Decoded object-store chunk result with the fetch evidence needed for metrics.
pub(super) struct ObjectStoreParquetChunkRead {
    pub(super) outcome: BudgetedExecutionOutcome<cudf::frame::dataframe::ParquetSourceChunkReadOutput>,
    pub(super) observed_row_group_count: Option<u64>,
    pub(super) host_buffer_fetch: HostBufferFetchRuntimeMetrics,
}

fn represented_file_row_group_count(file: &LocalParquetFile) -> Option<usize> {
    file.row_group_bytes
        .as_ref()
        .or(file.projected_row_group_uncompressed_bytes.as_ref())
        .or(file.projected_row_group_compressed_bytes.as_ref())
        .or(file.pure_filter_row_group_uncompressed_bytes.as_ref())
        .or(file.pure_filter_row_group_compressed_bytes.as_ref())
        .or(file.row_group_rows.as_ref())
        .map(Vec::len)
}

fn finish_with_full_fetch_fallback(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: Option<u64>,
    fetch: ScheduledHostParquetFetch,
) -> Result<HostParquetBuffer> {
    match fetch.finish() {
        Ok(fetch) => Ok(fetch),
        Err(error) if error.code() == QueryErrorCode::UnsupportedPlanShape => {
            start_host_parquet_buffer_fetch(context, location, resolved_source, estimated_fetch_bytes)?.finish()
        },
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};

    use super::*;
    use crate::{
        chunk::SourceChunkBatch,
        plan::QueryPlanBuilder,
        runtime::SourceChunkPlanningMode,
        source::{
            CredentialMode, LakehouseCacheFillMode, LakehouseCacheStatus, LocalParquetFile, MissingColumn,
            ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, ResolvedParquetSource,
            SourceColumnRef, SourceNode,
        },
    };

    fn scan_limited_source() -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("payload", ArrowDataType::Int32, true)]));
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet")
            .unwrap()
            .with_resolved_parquet_source(
                ResolvedParquetSource::remote_host_buffer(
                    "s3://bucket/table/part-0.parquet",
                    "s3://bucket/table/part-0.parquet",
                    CredentialMode::SharedProcessEnv,
                )
                .with_remote_host_buffer_ranged_enabled(true),
            )
            .unwrap()
            .with_stable_id("part-0")
            .with_file_size_bytes(4096)
            .with_row_group_bytes(vec![1024, 1024, 2048])
            .with_row_group_rows(vec![10, 10, 10]);
        LocalParquetSource::from_files(vec![file], schema, vec!["payload".to_string()]).with_scan_limit(Some(15))
    }

    #[test]
    fn complete_cache_file_hit_requires_expected_size() {
        let unique = LAKEHOUSE_CACHE_READ_PIN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("nexus-cache-size-test-{}-{unique}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("part-0.parquet");
        fs::write(&path, [0_u8; 8]).unwrap();

        assert!(complete_cache_file_matches_expected_size(&path, Some(8)));
        assert!(!complete_cache_file_matches_expected_size(&path, Some(9)));
        assert!(!complete_cache_file_matches_expected_size(&path, None));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    fn source_chunk(source: &LocalParquetSource, row_group_indexes: Vec<usize>) -> SourceChunk {
        let mut builder = QueryPlanBuilder::new();
        let source_node_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
        SourceChunk {
            source_node_id,
            chunk_index: 0,
            file_index: 0,
            location: source.files[0].location.clone(),
            path: source.files[0].path.clone(),
            resolved_source: source.files[0].resolved_source.clone(),
            stable_id: source.files[0].stable_id.clone(),
            row_group_indexes,
            reads_all_row_groups: false,
            known_file_bytes: Some(4096),
            known_row_group_bytes: Some(1024),
            known_projected_compressed_bytes: Some(1024),
            known_projected_uncompressed_bytes: Some(1024),
            known_pure_filter_compressed_bytes: Some(0),
            known_pure_filter_uncompressed_bytes: Some(0),
            estimated_chunk_bytes: Some(1024),
            estimated_source_read_bytes: Some(1024),
            exact_source_read_preflight_bytes: None,
            source_file_row_groups: None,
            source_file_row_group_bytes: Some(vec![1024, 1024, 2048]),
            source_file_projected_compressed_bytes: Some(vec![1024, 1024, 2048]),
            source_file_projected_uncompressed_bytes: Some(vec![1024, 1024, 2048]),
            source_file_pure_filter_compressed_bytes: Some(vec![0, 0, 0]),
            source_file_pure_filter_uncompressed_bytes: Some(vec![0, 0, 0]),
            source_file_row_group_rows: Some(vec![10, 10, 10]),
            reader_decimal_types: Vec::new(),
            reader_decimal_types_footer_verified: false,
            read_columns: vec!["payload".to_string()],
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
        let root = std::env::temp_dir().join(format!("nexus-query-engine-{test_name}-{nonce}"));
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
                "nexus_query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
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

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("fill_concurrency_saturated"));
        assert!(!lakehouse_cache_lock_path(&cache_path).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_lakehouse_cache_fill_does_not_skip_wrong_size_cache_file() {
        let root = unique_temp_cache_root("background-fill-wrong-size");
        let cache_path = root.join("v1/s3/bucket/content/part-0.parquet");
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, b"incomplete").unwrap();
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "nexus_query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
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
                "nexus_query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
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
        thread::sleep(MALFORMED_LAKEHOUSE_CACHE_LOCK_STALE_AGE + Duration::from_millis(25));
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "nexus_query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source =
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
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
            ResolvedParquetSource::remote_direct("s3://bucket/table/part-0.parquet", CredentialMode::SharedProcessEnv)
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

        assert_eq!(outcome, BackgroundLakehouseCacheFillEnqueue::Skipped("fill_byte_budget_saturated"));
        assert!(!lakehouse_cache_lock_path(&cache_path).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn background_lakehouse_cache_fill_accepts_host_buffer_fallback_sources() {
        let root = unique_temp_cache_root("background-fill-host-buffer");
        let cache_path = root.join("v1/s3/bucket/content/part-0.parquet");
        let context = ExecutionContext::builder().with_lakehouse_cache_fill_max_concurrent_ops(1).build().unwrap();
        let _held_permit = context
            .acquire_lakehouse_cache_fill_operation_with_timeout(
                Duration::ZERO,
                "nexus_query_engine.test.background_fill_gate",
            )
            .unwrap()
            .unwrap();
        let resolved_source = ResolvedParquetSource::remote_host_buffer(
            "s3://bucket/table/part-0.parquet",
            "s3://bucket/table/part-0.parquet",
            CredentialMode::SharedProcessEnv,
        )
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
    fn host_buffer_object_store_error_facts_redact_source_identity() {
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
            "nexus_query_engine.test.cache_error",
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
    fn ranged_fetch_facts_report_general_ranged_parquet_mode() {
        let source = scan_limited_source();
        let chunk = source_chunk(&source, vec![0]);
        let diagnostics = RangedParquetFetchDiagnostics {
            mini_payload_bytes: 128,
            mini_row_group_bytes: vec![64],
            mini_row_group_rows: vec![10],
            footer_bytes: 32,
            original_file_bytes: 4096,
            selected_original_row_groups: vec![0],
            selected_file_column_indices: vec![0],
            requested_range_count: 3,
            requested_range_bytes: 256,
        };
        let error = QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.test.ranged_fetch_facts",
            "test ranged fetch diagnostics",
        );

        let error = attach_ranged_fetch_facts(error, &diagnostics, &chunk);

        assert_eq!(error.fact_value("source_fetch_mode"), Some("ranged_parquet"));
        assert_eq!(error.fact_value("requested_range_count"), Some("3"));
        assert_eq!(error.fact_value("requested_range_bytes"), Some("256"));
        assert_eq!(error.fact_value("selected_file_column_indices"), Some("0"));
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

    #[test]
    fn ranged_scan_limit_trigger_covers_fully_kept_limited_chunks() {
        let source = scan_limited_source();
        let chunk = source_chunk(&source, vec![0]);

        let request = ranged_parquet_chunk_request(&source, &chunk);
        assert!(matches!(
            request.map(|request| request.row_group_selector),
            Some(ranged_parquet::RangedRowGroupSelector::Explicit(row_groups)) if row_groups == vec![0]
        ));
    }

    #[test]
    fn ranged_chunk_request_uses_explicit_selection_without_source_limit() {
        let source = scan_limited_source();
        let chunk = source_chunk(&source, vec![0]);
        let request = ranged_parquet_chunk_request(&source, &chunk);
        assert!(matches!(
            request.map(|request| request.row_group_selector),
            Some(ranged_parquet::RangedRowGroupSelector::Explicit(row_groups)) if row_groups == vec![0]
        ));

        let source_without_limit = source.clone().with_scan_limit(None);
        let limited_chunk = source_chunk(&source_without_limit, vec![0]);
        let request = ranged_parquet_chunk_request(&source_without_limit, &limited_chunk);
        assert!(matches!(
            request.map(|request| request.row_group_selector),
            Some(ranged_parquet::RangedRowGroupSelector::Explicit(row_groups)) if row_groups == vec![0]
        ));

        let zero_limit_source = source.with_scan_limit(Some(0));
        let zero_limit_chunk = source_chunk(&zero_limit_source, vec![0]);
        let request = ranged_parquet_chunk_request(&zero_limit_source, &zero_limit_chunk);
        assert!(matches!(
            request.map(|request| request.row_group_selector),
            Some(ranged_parquet::RangedRowGroupSelector::Explicit(row_groups)) if row_groups == vec![0]
        ));
    }

    #[test]
    fn ranged_chunk_request_respects_host_buffer_ranged_gate() {
        let mut source = scan_limited_source();
        source.files[0].resolved_source =
            source.files[0].resolved_source.clone().with_remote_host_buffer_ranged_enabled(false);
        let chunk = source_chunk(&source, vec![0]);

        assert!(ranged_parquet_chunk_request(&source, &chunk).is_none());
    }

    #[test]
    fn ranged_chunk_request_includes_all_read_columns_including_filter_only_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("projected", ArrowDataType::Int32, true),
            Field::new("filter_only", ArrowDataType::Int32, true),
            Field::new("unused", ArrowDataType::Int32, true),
        ]));
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-filter.parquet")
            .unwrap()
            .with_resolved_parquet_source(
                ResolvedParquetSource::remote_host_buffer(
                    "s3://bucket/table/part-filter.parquet",
                    "s3://bucket/table/part-filter.parquet",
                    CredentialMode::SharedProcessEnv,
                )
                .with_remote_host_buffer_ranged_enabled(true),
            )
            .unwrap()
            .with_stable_id("part-filter")
            .with_file_size_bytes(4096)
            .with_row_group_bytes(vec![2048, 2048])
            .with_projected_row_group_bytes(vec![512, 512], vec![1024, 1024])
            .with_pure_filter_row_group_bytes(vec![256, 256], vec![512, 512])
            .with_row_group_rows(vec![10, 10]);
        let source = LocalParquetSource::from_files(
            vec![file],
            schema,
            vec!["projected".to_string(), "filter_only".to_string()],
        )
        .with_column_roles(vec!["projected".to_string()], vec!["filter_only".to_string()])
        .with_column_file_indices(vec![0, 1], vec![0], vec![1]);
        let mut chunk = source_chunk(&source, vec![0, 1]);
        chunk.estimated_source_read_bytes = Some(1536);
        chunk.source_file_projected_compressed_bytes = Some(vec![512, 512]);
        chunk.source_file_pure_filter_compressed_bytes = Some(vec![256, 256]);

        let request = ranged_parquet_chunk_request(&source, &chunk).unwrap();

        assert_eq!(request.file_column_indices, Some(vec![0, 1]));
        assert!(matches!(
            request.row_group_selector,
            ranged_parquet::RangedRowGroupSelector::Explicit(row_groups) if row_groups == vec![0, 1]
        ));
    }

    #[test]
    fn ranged_chunk_request_treats_prefix_projection_as_column_pruning_when_file_column_count_is_known() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("c0", ArrowDataType::Int32, true),
            Field::new("c1", ArrowDataType::Int32, true),
            Field::new("c2", ArrowDataType::Int32, true),
        ]));
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part-prefix.parquet")
            .unwrap()
            .with_resolved_parquet_source(
                ResolvedParquetSource::remote_host_buffer(
                    "s3://bucket/table/part-prefix.parquet",
                    "s3://bucket/table/part-prefix.parquet",
                    CredentialMode::SharedProcessEnv,
                )
                .with_remote_host_buffer_ranged_enabled(true),
            )
            .unwrap()
            .with_stable_id("part-prefix")
            .with_file_size_bytes(4096)
            .with_row_group_bytes(vec![2048, 2048])
            .with_row_group_rows(vec![10, 10]);
        let source = LocalParquetSource::from_files(vec![file], schema, vec!["c0".to_string(), "c1".to_string()])
            .with_column_file_indices(vec![0, 1], vec![0, 1], vec![])
            .with_file_column_count(3);
        let chunk = source_chunk(&source, vec![0, 1]);

        let request = ranged_parquet_chunk_request(&source, &chunk).unwrap();

        assert_eq!(request.file_column_indices, Some(vec![0, 1]));
        assert!(matches!(
            request.row_group_selector,
            ranged_parquet::RangedRowGroupSelector::Explicit(row_groups) if row_groups == vec![0, 1]
        ));
    }

    #[test]
    fn ranged_file_request_uses_scan_limit_when_explicit_row_groups_cover_whole_file() {
        let mut source = scan_limited_source();
        source.files[0] = source.files[0].clone().with_row_groups(vec![0, 1, 2]);

        let request = ranged_parquet_file_request(&source, &source.files[0]).unwrap();

        assert!(matches!(request.row_group_selector, ranged_parquet::RangedRowGroupSelector::ScanLimit(15)));
    }

    #[test]
    fn ranged_file_request_prefers_scan_limit_prefix_over_column_pruned_whole_file() {
        let source = scan_limited_source().with_column_file_indices(vec![0], vec![0], vec![]).with_file_column_count(2);

        let request = ranged_parquet_file_request(&source, &source.files[0]).unwrap();

        assert_eq!(request.file_column_indices, Some(vec![0]));
        assert!(matches!(request.row_group_selector, ranged_parquet::RangedRowGroupSelector::ScanLimit(15)));
    }

    #[test]
    fn ranged_chunk_request_prefers_scan_limit_prefix_over_column_pruned_whole_file() {
        let source = scan_limited_source().with_column_file_indices(vec![0], vec![0], vec![]).with_file_column_count(2);
        let mut chunk = source_chunk(&source, vec![0, 1, 2]);
        chunk.scan_limit_rows = Some(15);

        let request = ranged_parquet_chunk_request(&source, &chunk).unwrap();

        assert_eq!(request.file_column_indices, Some(vec![0]));
        assert!(matches!(request.row_group_selector, ranged_parquet::RangedRowGroupSelector::ScanLimit(15)));
    }

    #[test]
    fn ranged_scan_limit_trigger_uses_footer_selector_when_only_file_row_count_is_known() {
        let source = scan_limited_source();
        let mut chunk = source_chunk(&source, vec![0]);
        chunk.scan_limit_rows = Some(15);
        chunk.known_row_count = Some(15);
        chunk.source_file_row_group_bytes = None;
        chunk.source_file_projected_compressed_bytes = None;
        chunk.source_file_projected_uncompressed_bytes = None;
        chunk.source_file_pure_filter_compressed_bytes = None;
        chunk.source_file_pure_filter_uncompressed_bytes = None;
        chunk.source_file_row_group_rows = Some(vec![30]);
        chunk.estimated_source_read_bytes = chunk.known_file_bytes;

        let request = ranged_parquet_chunk_request(&source, &chunk);

        assert!(matches!(
            request.map(|request| request.row_group_selector),
            Some(ranged_parquet::RangedRowGroupSelector::ScanLimit(15))
        ));
    }
}
