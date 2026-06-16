use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use object_store::ObjectStoreExt;

use super::{
    lakehouse_cache_policy::LakehouseCachePolicy,
    object_store_safety::{object_store_key_fact, source_fetch_location},
    start_host_parquet_buffer_fetch, *,
};
use crate::{
    datatypes::is_arrow_fixed_width,
    error::{QueryError, QueryErrorCode, Result},
    parquet_metadata::write_parquet_footer_sidecar_from_file,
    runtime::{LakehouseCacheFillErrorEvent, LakehouseCacheFillEvent, LakehouseCacheFillOutcome, S3ObjectStoreKey},
};

pub(super) static LAKEHOUSE_CACHE_READ_PIN_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(in crate::exec::source) fn ensure_local_cache_parquet_file(
    context: &ExecutionContext,
    file: &LocalParquetFile,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    ensure_local_cache_parquet_entry(
        context,
        &file.location,
        &file.resolved_source,
        &file.path,
        file.file_size_bytes,
        file.stable_id.as_str(),
        file.resolved_source.lakehouse_cache_root.as_deref(),
        file.resolved_source.lakehouse_cache_max_bytes,
        None,
        None,
        metrics,
    )
}

pub(in crate::exec::source) fn ensure_local_cache_parquet_chunk(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    ensure_local_cache_parquet_entry(
        context,
        &chunk.location,
        &chunk.resolved_source,
        &chunk.path,
        chunk.known_file_bytes,
        chunk.stable_id.as_str(),
        chunk.resolved_source.lakehouse_cache_root.as_deref(),
        chunk.resolved_source.lakehouse_cache_max_bytes,
        Some(chunk.file_index),
        Some(chunk.chunk_index),
        metrics,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::source) enum BackgroundLakehouseCacheFillEnqueue {
    Enqueued,
    Skipped(&'static str),
}

pub(in crate::exec::source) fn enqueue_background_lakehouse_cache_file_fill(
    context: &ExecutionContext,
    file: &LocalParquetFile,
) -> Result<BackgroundLakehouseCacheFillEnqueue> {
    enqueue_background_lakehouse_cache_entry_fill(
        context,
        &file.location,
        &file.resolved_source,
        file.file_size_bytes,
        file.stable_id.as_str(),
        None,
        None,
    )
}

pub(in crate::exec::source) fn enqueue_background_lakehouse_cache_chunk_fill(
    context: &ExecutionContext,
    chunk: &SourceChunk,
) -> Result<BackgroundLakehouseCacheFillEnqueue> {
    enqueue_background_lakehouse_cache_entry_fill(
        context,
        &chunk.location,
        &chunk.resolved_source,
        chunk.known_file_bytes,
        chunk.stable_id.as_str(),
        Some(chunk.file_index),
        Some(chunk.chunk_index),
    )
}

pub(in crate::exec::source) struct LakehouseCacheReadPin {
    pub(super) path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(in crate::exec::source) struct LakehouseOptimizedBatchCacheEntry {
    pub(in crate::exec::source) path: PathBuf,
    pub(in crate::exec::source) stable_id: String,
    cache_root: Option<String>,
    cache_max_bytes: Option<u64>,
}

impl Drop for LakehouseCacheReadPin {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

pub(in crate::exec::source) fn acquire_lakehouse_cache_file_read_pin(
    context: &ExecutionContext,
    file: &LocalParquetFile,
) -> Result<LakehouseCacheReadPin> {
    acquire_lakehouse_cache_read_pin(context, &file.path, file.stable_id.as_str(), None, None)
}

pub(in crate::exec::source) fn acquire_lakehouse_cache_chunk_read_pin(
    context: &ExecutionContext,
    chunk: &SourceChunk,
) -> Result<LakehouseCacheReadPin> {
    acquire_lakehouse_cache_read_pin(
        context,
        &chunk.path,
        chunk.stable_id.as_str(),
        Some(chunk.file_index),
        Some(chunk.chunk_index),
    )
}

pub(in crate::exec::source) fn acquire_lakehouse_optimized_batch_read_pin(
    context: &ExecutionContext,
    entry: &LakehouseOptimizedBatchCacheEntry,
) -> Result<LakehouseCacheReadPin> {
    acquire_lakehouse_cache_read_pin(context, &entry.path, entry.stable_id.as_str(), None, None)
}

pub(in crate::exec::source) fn local_cache_optimized_batch_entry(
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
) -> Option<LakehouseOptimizedBatchCacheEntry> {
    if !local_cache_optimized_batch_eligible(source, batch) {
        return None;
    }
    let first = batch.chunks.first()?;
    let cache_root = first.resolved_source.lakehouse_cache_root.clone()?;
    let source_node = first.source_node_id.to_string();
    let signature = local_cache_optimized_batch_signature(source, batch);
    let hash = stable_fnv1a64(signature.as_bytes());
    let file_name = format!("{}__batch_v4_{hash:016x}.parquet", sanitize_cache_component(&source_node));
    let path = Path::new(&cache_root).join("optimized-batches").join("v4").join(file_name);
    Some(LakehouseOptimizedBatchCacheEntry {
        path,
        stable_id: format!("optimized-batch-v4-fnv64-{hash:016x}"),
        cache_root: Some(cache_root),
        cache_max_bytes: first.resolved_source.lakehouse_cache_max_bytes,
    })
}

pub(in crate::exec::source) fn publish_lakehouse_optimized_batch_frame(
    context: &ExecutionContext,
    entry: &LakehouseOptimizedBatchCacheEntry,
    frame: &DataFrame,
) -> Result<bool> {
    context.check_cancellation("nexus_query_engine.lakehouse_cache.optimized_batch.publish")?;
    if entry.path.is_file() {
        return Ok(false);
    }
    if let Some(parent) = entry.path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            lakehouse_cache_io_error(
                error,
                "nexus_query_engine.lakehouse_cache.optimized_batch.mkdir",
                "failed to create optimized lakehouse cache parent directory",
                &entry.path,
                entry.stable_id.as_str(),
                None,
                None,
            )
        })?;
    }

    let lock_path = lakehouse_cache_lock_path(&entry.path);
    let _lock = acquire_lakehouse_cache_lock(context, &entry.path, &lock_path, entry.stable_id.as_str(), None, None)?;
    if entry.path.is_file() {
        return Ok(false);
    }

    let temp_path = lakehouse_cache_temp_path(&entry.path);
    let row_group_size_rows = optimized_batch_row_group_size_rows(frame.num_rows());
    let write_result = frame.write_parquet_with_options(
        cudf::frame::dataframe::ParquetWriteOptions::builder()
            .filepath(&temp_path)
            .row_group_size_rows(row_group_size_rows)
            .dictionary_policy(cudf::frame::dataframe::ParquetDictionaryPolicy::Adaptive)
            .with_runtime(context.runtime())
            .build(),
    );
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(QueryError::from_cudf(
            "nexus_query_engine.lakehouse_cache.optimized_batch.write",
            "failed to write optimized lakehouse cache batch parquet file",
            error,
        )
        .with_fact("path", lakehouse_cache_path_fact(&entry.path))
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(entry.stable_id.as_str())));
    }
    if let Err(error) = fs::rename(&temp_path, &entry.path) {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.optimized_batch.publish",
            "failed to atomically publish optimized lakehouse cache batch",
            &entry.path,
            entry.stable_id.as_str(),
            None,
            None,
        )
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path)));
    }
    evict_lakehouse_cache(entry.cache_root.as_deref(), &entry.path, entry.cache_max_bytes);
    Ok(true)
}

fn acquire_lakehouse_cache_read_pin(
    context: &ExecutionContext,
    cache_path: &Path,
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<LakehouseCacheReadPin> {
    context.check_cancellation("nexus_query_engine.lakehouse_cache.read_pin")?;
    if !cache_path.is_file() {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.lakehouse_cache.read_pin",
            "lakehouse cache entry is missing before read pin acquisition",
        )
        .with_fact("path", lakehouse_cache_path_fact(cache_path))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id)));
    }
    let pin_path = lakehouse_cache_read_pin_path(cache_path);
    let mut pin_file = OpenOptions::new().write(true).create_new(true).open(&pin_path).map_err(|error| {
        lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.read_pin",
            "failed to create lakehouse cache read pin",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("pin_path", lakehouse_cache_path_fact(&pin_path))
    })?;
    if let Err(error) = writeln!(pin_file, "pid={}", std::process::id()) {
        let _ = fs::remove_file(&pin_path);
        return Err(lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.read_pin",
            "failed to write lakehouse cache read pin",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("pin_path", lakehouse_cache_path_fact(&pin_path)));
    }
    if !cache_path.is_file() {
        let _ = fs::remove_file(&pin_path);
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.lakehouse_cache.read_pin",
            "lakehouse cache entry disappeared during read pin acquisition",
        )
        .with_fact("path", lakehouse_cache_path_fact(cache_path))
        .with_fact("pin_path", lakehouse_cache_path_fact(&pin_path))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id)));
    }
    Ok(LakehouseCacheReadPin { path: Some(pin_path) })
}

fn local_cache_optimized_batch_eligible(source: &LocalParquetSource, batch: &crate::chunk::SourceChunkBatch) -> bool {
    let policy = LakehouseCachePolicy::default();

    if !std::env::var("NEXUS_LAKEHOUSE_OPTIMIZED_BATCH_CACHE").is_ok_and(|value| value == "1")
        || !policy.optimized_batch_has_min_files(batch.chunk_count())
        || source.reader_predicate.is_some()
        || source.scan_limit.is_some()
        || !source.partition_columns.is_empty()
        || source.read_columns.is_empty()
        || !optimized_batch_uses_identity_source_frame(source)
        || !source_read_columns_are_fixed_width(source)
    {
        return false;
    }
    let Some(estimated_source_read_bytes) = batch.estimated_source_read_bytes else {
        return false;
    };
    if !policy.optimized_batch_source_read_bytes_allowed(estimated_source_read_bytes) {
        return false;
    }

    let Some(first) = batch.chunks.first() else {
        return false;
    };
    let cache_root = first.resolved_source.lakehouse_cache_root.as_deref();
    cache_root.is_some()
        && batch.chunks.iter().all(|chunk| {
            chunk.resolved_source.read_strategy == crate::source::ParquetReadStrategy::LocalCacheKvikio
                && chunk.resolved_source.lakehouse_cache_root.as_deref() == cache_root
                && !chunk.zero_column_count
                && chunk.scan_limit_rows.is_none()
                && !chunk.read_columns.is_empty()
                && chunk.partition_values.is_empty()
                && chunk.selects_all_known_file_row_groups()
        })
}

pub(super) fn local_cache_optimized_batch_signature(
    source: &LocalParquetSource,
    batch: &crate::chunk::SourceChunkBatch,
) -> String {
    let mut signature = String::from(
        "optimized-local-cache-source-batch-v4-small-fixed-width-rowgroup-normalized-adaptive-dictionary\n",
    );
    signature.push_str(&format!("schema={:?}\n", source.schema));
    signature.push_str(&format!("read_columns={:?}\n", source.read_columns));
    signature.push_str(&format!("read_column_file_indices={:?}\n", source.read_column_file_indices));
    signature.push_str(&format!("projected_output_columns={:?}\n", source.projected_output_columns));
    signature
        .push_str(&format!("projected_output_column_source_refs={:?}\n", source.projected_output_column_source_refs));
    signature.push_str(&format!("pure_filter_columns={:?}\n", source.pure_filter_columns));
    signature.push_str(&format!("pure_filter_column_source_refs={:?}\n", source.pure_filter_column_source_refs));
    signature.push_str(&format!("missing_columns={:?}\n", source.missing_columns));
    let mut chunk_signatures = batch
        .chunks
        .iter()
        .map(|chunk| {
            format!(
                "stable_id={} resolved_contract={} reader_decimal_contract={} reader_decimal_footer_verified={} known_file_bytes={:?} row_groups={:?}",
                chunk.location.diagnostic_stable_id(chunk.stable_id.as_str()),
                optimized_batch_resolved_source_contract_signature(&chunk.resolved_source),
                optimized_batch_reader_decimal_contract_signature(&chunk.reader_decimal_types),
                chunk.reader_decimal_types_footer_verified,
                chunk.known_file_bytes,
                chunk.row_group_indexes
            )
        })
        .collect::<Vec<_>>();
    chunk_signatures.sort();
    for chunk_signature in chunk_signatures {
        signature.push_str("chunk=");
        signature.push_str(&chunk_signature);
        signature.push('\n');
    }
    signature
}

fn optimized_batch_reader_decimal_contract_signature(
    reader_decimal_types: &[crate::source::ParquetReaderDecimalType],
) -> String {
    let mut output = String::new();
    push_optimized_batch_contract_value(
        &mut output,
        "reader_decimal_types.len",
        reader_decimal_types.len().to_string().as_str(),
    );
    for (index, decimal_type) in reader_decimal_types.iter().enumerate() {
        push_optimized_batch_contract_value(
            &mut output,
            format!("reader_decimal_types.{index}.column_name").as_str(),
            decimal_type.column_name.as_str(),
        );
        push_optimized_batch_contract_value(
            &mut output,
            format!("reader_decimal_types.{index}.type_id").as_str(),
            decimal_type.type_id.as_str(),
        );
        push_optimized_batch_contract_value(
            &mut output,
            format!("reader_decimal_types.{index}.cudf_scale").as_str(),
            decimal_type.cudf_scale.to_string().as_str(),
        );
    }
    output
}

fn optimized_batch_resolved_source_contract_signature(
    resolved_source: &crate::source::ResolvedParquetSource,
) -> String {
    let mut output = String::new();
    push_optimized_batch_contract_hash(
        &mut output,
        "original_location",
        Some(resolved_source.original_location.as_str()),
    );
    push_optimized_batch_contract_hash(&mut output, "reader_location", Some(resolved_source.reader_location.as_str()));
    push_optimized_batch_contract_hash(&mut output, "storage_domain", resolved_source.storage_domain.as_deref());
    push_optimized_batch_contract_hash(&mut output, "content_identity", resolved_source.content_identity.as_deref());
    push_optimized_batch_contract_value(
        &mut output,
        "storage_profile_id",
        resolved_source.storage_profile_id.as_deref().unwrap_or("none"),
    );
    push_optimized_batch_contract_value(&mut output, "read_strategy", resolved_source.read_strategy.as_str());
    push_optimized_batch_contract_value(&mut output, "decode_input", resolved_source.decode_input.as_str());
    push_optimized_batch_contract_value(&mut output, "remote_read_mode", resolved_source.remote_read_mode.as_str());
    push_optimized_batch_contract_value(&mut output, "credential_mode", resolved_source.credential_mode.as_str());
    push_optimized_batch_contract_value(&mut output, "cache_status", resolved_source.cache_status.as_str());
    push_optimized_batch_contract_value(
        &mut output,
        "lakehouse_cache_read_mode",
        resolved_source.lakehouse_cache_read_mode.as_str(),
    );
    push_optimized_batch_contract_hash(
        &mut output,
        "lakehouse_cache_root",
        resolved_source.lakehouse_cache_root.as_deref(),
    );
    push_optimized_batch_contract_hash(
        &mut output,
        "lakehouse_cache_path",
        resolved_source.lakehouse_cache_path.as_deref(),
    );
    push_optimized_batch_contract_value(
        &mut output,
        "lakehouse_cache_max_bytes",
        resolved_source
            .lakehouse_cache_max_bytes
            .map_or_else(|| "none".to_string(), |bytes| bytes.to_string())
            .as_str(),
    );
    push_optimized_batch_contract_value(
        &mut output,
        "lakehouse_cache_fill_mode",
        resolved_source.lakehouse_cache_fill_mode.as_str(),
    );
    push_optimized_batch_contract_value(
        &mut output,
        "remote_host_buffer_ranged_enabled",
        if resolved_source.remote_host_buffer_ranged_enabled { "true" } else { "false" },
    );
    push_optimized_batch_contract_value(
        &mut output,
        "fallback_reason",
        resolved_source.fallback_reason.as_deref().unwrap_or("none"),
    );
    output
}

fn push_optimized_batch_contract_hash(output: &mut String, key: &str, value: Option<&str>) {
    match value {
        Some(value) => push_optimized_batch_contract_value(
            output,
            key,
            format!("fnv64:{:016x}:len:{}", stable_fnv1a64(value.as_bytes()), value.len()).as_str(),
        ),
        None => push_optimized_batch_contract_value(output, key, "none"),
    }
}

fn push_optimized_batch_contract_value(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push('=');
    output.push_str(value);
    output.push(';');
}

pub(super) fn optimized_batch_uses_identity_source_frame(source: &LocalParquetSource) -> bool {
    source.missing_columns.is_empty()
        && source.pure_filter_columns.is_empty()
        && source.projected_output_columns == source.read_columns
        && source.projected_output_column_source_refs.len() == source.read_columns.len()
        && source
            .projected_output_column_source_refs
            .iter()
            .zip(&source.read_column_file_indices)
            .all(|(source_ref, read_index)| {
                matches!(source_ref, crate::source::SourceColumnRef::File(index) if index == read_index)
            })
}

fn source_read_columns_are_fixed_width(source: &LocalParquetSource) -> bool {
    source.read_columns.iter().all(|column| {
        source.schema.field_with_name(column).is_ok_and(|field| optimized_cache_fixed_width_type(field.data_type()))
    })
}

fn optimized_cache_fixed_width_type(data_type: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType;
    is_arrow_fixed_width(data_type) || matches!(data_type, DataType::Float16)
}

fn optimized_batch_row_group_size_rows(row_count: usize) -> i32 {
    LakehouseCachePolicy::default().optimized_batch_row_group_size_rows(row_count)
}

fn sanitize_cache_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') { ch } else { '_' })
        .collect::<String>();
    if sanitized.is_empty() { "unknown".to_string() } else { sanitized }
}

fn stable_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn ensure_local_cache_parquet_entry(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    cache_path: &Path,
    estimated_fetch_bytes: Option<u64>,
    stable_id: &str,
    cache_root: Option<&str>,
    cache_max_bytes: Option<u64>,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    context.check_cancellation("nexus_query_engine.lakehouse_cache.ensure")?;
    if complete_cache_file_matches_expected_size(cache_path, estimated_fetch_bytes) {
        return Ok(());
    }
    if resolved_source.cache_status == crate::source::LakehouseCacheStatus::Hit {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.lakehouse_cache.stale_hit",
            "planned lakehouse cache hit is missing or does not match source metadata",
        )
        .with_fact("path", lakehouse_cache_path_fact(cache_path))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id))
        .with_fact("expected_bytes", estimated_fetch_bytes.map(|bytes| bytes.to_string()).unwrap_or_default()));
    }
    if !matches!(location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.lakehouse_cache.location",
            "local lakehouse cache fill requires an object-store source location",
        )
        .with_fact("path", lakehouse_cache_path_fact(cache_path))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id)));
    }
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            lakehouse_cache_io_error(
                error,
                "nexus_query_engine.lakehouse_cache.mkdir",
                "failed to create lakehouse cache parent directory",
                cache_path,
                stable_id,
                file_index,
                chunk_index,
            )
        })?;
    }

    let lock_path = lakehouse_cache_lock_path(cache_path);
    let _lock = acquire_lakehouse_cache_lock(context, cache_path, &lock_path, stable_id, file_index, chunk_index)?;
    if complete_cache_file_matches_expected_size(cache_path, estimated_fetch_bytes) {
        return Ok(());
    }

    let mut fetch =
        start_host_parquet_buffer_fetch(context, location, resolved_source, estimated_fetch_bytes)?.finish()?;
    metrics.record_host_fetch_budget_wait(fetch.host_budget_wait_elapsed);
    metrics.record_source_fetch_concurrency_wait(fetch.concurrency_wait_elapsed);
    metrics.record_source_fetch(fetch.actual_bytes, fetch.fetch_elapsed);
    metrics.source_read_time += fetch.fetch_elapsed;
    validate_lakehouse_cache_fetch_size(
        cache_path,
        stable_id,
        estimated_fetch_bytes,
        fetch.actual_bytes,
        file_index,
        chunk_index,
    )?;
    publish_lakehouse_cache_file(
        cache_path,
        fetch.bytes(),
        stable_id,
        sidecar_content_identity(resolved_source, stable_id),
        file_index,
        chunk_index,
    )?;
    evict_lakehouse_cache(cache_root, cache_path, cache_max_bytes);
    fetch.release_after_decode();
    Ok(())
}

fn validate_lakehouse_cache_fetch_size(
    cache_path: &Path,
    stable_id: &str,
    expected_bytes: Option<u64>,
    actual_bytes: u64,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<()> {
    let Some(expected_bytes) = expected_bytes else {
        return Ok(());
    };
    if actual_bytes == expected_bytes {
        return Ok(());
    }
    let mut error = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.lakehouse_cache.fill_size",
        "lakehouse cache fill fetched a different byte count than source metadata",
    )
    .with_fact("path", lakehouse_cache_path_fact(cache_path))
    .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id))
    .with_fact("expected_bytes", expected_bytes)
    .with_fact("actual_bytes", actual_bytes);
    if let Some(file_index) = file_index {
        error = error.with_fact("file_index", file_index);
    }
    if let Some(chunk_index) = chunk_index {
        error = error.with_fact("chunk_index", chunk_index);
    }
    Err(error)
}

pub(super) fn complete_cache_file_matches_expected_size(cache_path: &Path, expected_bytes: Option<u64>) -> bool {
    let Ok(metadata) = fs::metadata(cache_path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    let Some(expected_bytes) = expected_bytes else {
        return false;
    };
    metadata.len() == expected_bytes
}

fn enqueue_background_lakehouse_cache_entry_fill(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: Option<u64>,
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<BackgroundLakehouseCacheFillEnqueue> {
    if resolved_source.lakehouse_cache_fill_mode != crate::source::LakehouseCacheFillMode::Background {
        return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("fill_disabled"));
    }
    let Some(cache_path) = resolved_source.lakehouse_cache_path.as_ref().map(PathBuf::from) else {
        return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("missing_cache_path"));
    };
    if complete_cache_file_matches_expected_size(&cache_path, estimated_fetch_bytes) {
        return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("cache_file_exists"));
    }
    if !matches!(location, crate::source::ParquetSourceLocation::ObjectStoreUri { .. }) {
        return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("not_object_store_source"));
    }
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            lakehouse_cache_io_error(
                error,
                "nexus_query_engine.lakehouse_cache.background_fill.mkdir",
                "failed to create background lakehouse cache parent directory",
                &cache_path,
                stable_id,
                file_index,
                chunk_index,
            )
        })?;
    }

    let lock_path = lakehouse_cache_lock_path(&cache_path);
    for attempt in 0..2 {
        match OpenOptions::new().write(true).create_new(true).open(&lock_path) {
            Ok(mut file) => {
                let _ = writeln!(file, "pid={}", std::process::id());
                break;
            },
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if attempt == 0 && remove_lakehouse_cache_lock_if_stale(&lock_path) {
                    continue;
                }
                return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("fill_already_running"));
            },
            Err(error) => {
                return Err(lakehouse_cache_io_error(
                    error,
                    "nexus_query_engine.lakehouse_cache.background_fill.lock",
                    "failed to create background lakehouse cache fill lock",
                    &cache_path,
                    stable_id,
                    file_index,
                    chunk_index,
                )
                .with_fact("lock_path", lakehouse_cache_path_fact(&lock_path)));
            },
        }
    }
    let lock_guard = BackgroundLakehouseCacheLock { path: lock_path.clone() };
    let Some(operation_permit) = context.acquire_lakehouse_cache_fill_operation_with_timeout(
        Duration::ZERO,
        "nexus_query_engine.lakehouse_cache.background_fill.concurrency",
    )?
    else {
        drop(lock_guard);
        return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("fill_concurrency_saturated"));
    };
    let initial_budget_bytes = estimated_fetch_bytes.filter(|bytes| *bytes > 0).unwrap_or(1);
    let budget_permit = match context.acquire_lakehouse_cache_fill_budget_with_timeout(
        initial_budget_bytes,
        Duration::ZERO,
        "nexus_query_engine.lakehouse_cache.background_fill.budget",
    ) {
        Ok(Some(permit)) => permit,
        Ok(None) => {
            drop(lock_guard);
            return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("fill_byte_budget_saturated"));
        },
        Err(error) if error.code() == QueryErrorCode::MemoryReservationFailed => {
            drop(lock_guard);
            return Ok(BackgroundLakehouseCacheFillEnqueue::Skipped("fill_byte_budget_saturated"));
        },
        Err(error) => return Err(error),
    };

    let fetch_location = source_fetch_location(location, resolved_source)?;
    let store = context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
        storage_profile_id: fetch_location.storage_profile_id.clone(),
    })?;
    let cache_root = resolved_source.lakehouse_cache_root.clone();
    let cache_max_bytes = resolved_source.lakehouse_cache_max_bytes;
    let temp_path = lakehouse_cache_temp_path(&cache_path);
    let stable_id = stable_id.to_string();
    let callback_stable_id = stable_id.clone();
    let callback_file_index = file_index;
    let callback_chunk_index = chunk_index;
    let callback_cache_path_hash = stable_log_hash(&cache_path.display().to_string());
    let callback_stable_id_hash = stable_log_hash(callback_stable_id.as_str());
    let event_sink = context.lakehouse_cache_fill_events();
    let sidecar_content_identity = sidecar_content_identity(resolved_source, stable_id.as_str()).to_string();
    let _fetch_handle = context.spawn_lakehouse_cache_fill(
        {
            let cache_path = cache_path.clone();
            move || {
                let _operation_permit = operation_permit;
                stream_lakehouse_cache_fill(
                    store,
                    fetch_location.key,
                    cache_path,
                    temp_path,
                    lock_guard,
                    stable_id,
                    sidecar_content_identity,
                    estimated_fetch_bytes,
                    budget_permit,
                )
            }
        },
        move |fill_result| {
            match fill_result.value {
                Ok(bytes) => {
                    event_sink.record(LakehouseCacheFillEvent {
                        outcome: LakehouseCacheFillOutcome::Completed,
                        bytes: Some(bytes),
                        elapsed: fill_result.elapsed,
                        cache_path_hash: callback_cache_path_hash.clone(),
                        stable_id_hash: callback_stable_id_hash.clone(),
                        file_index: callback_file_index,
                        chunk_index: callback_chunk_index,
                        error: None,
                    });
                    eprintln!(
                        "[nexus-query-engine] lakehouse_cache_background_fill outcome=completed bytes={} elapsed_ns={} cache_path_hash={} stable_id_hash={} file_index={} chunk_index={}",
                        bytes,
                        fill_result.elapsed.as_nanos().min(u128::from(u64::MAX)),
                        callback_cache_path_hash,
                        callback_stable_id_hash,
                        optional_usize_log(callback_file_index),
                        optional_usize_log(callback_chunk_index),
                    );
                    evict_lakehouse_cache(cache_root.as_deref(), &cache_path, cache_max_bytes);
                },
                Err(error) => {
                    event_sink.record(LakehouseCacheFillEvent {
                        outcome: LakehouseCacheFillOutcome::Failed,
                        bytes: None,
                        elapsed: fill_result.elapsed,
                        cache_path_hash: callback_cache_path_hash.clone(),
                        stable_id_hash: callback_stable_id_hash.clone(),
                        file_index: callback_file_index,
                        chunk_index: callback_chunk_index,
                        error: Some(LakehouseCacheFillErrorEvent {
                            code: error.code().to_string(),
                            kind: error.kind().to_string(),
                            status: error.status().to_string(),
                            operation: error.operation().to_string(),
                        }),
                    });
                    eprintln!(
                        "[nexus-query-engine] lakehouse_cache_background_fill outcome=failed error_code={} error_kind={} error_status={} error_operation={} elapsed_ns={} cache_path_hash={} stable_id_hash={} file_index={} chunk_index={}",
                        error.code(),
                        error.kind(),
                        error.status(),
                        error.operation(),
                        fill_result.elapsed.as_nanos().min(u128::from(u64::MAX)),
                        callback_cache_path_hash,
                        callback_stable_id_hash,
                        optional_usize_log(callback_file_index),
                        optional_usize_log(callback_chunk_index),
                    );
                },
            }
        },
    );
    Ok(BackgroundLakehouseCacheFillEnqueue::Enqueued)
}

struct BackgroundLakehouseCacheLock {
    path: PathBuf,
}

impl Drop for BackgroundLakehouseCacheLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct LakehouseCacheLock {
    path: Option<PathBuf>,
}

impl Drop for LakehouseCacheLock {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
    }
}

fn acquire_lakehouse_cache_lock(
    context: &ExecutionContext,
    cache_path: &Path,
    lock_path: &Path,
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<LakehouseCacheLock> {
    let policy = LakehouseCachePolicy::default();

    for attempt in 0..policy.cache_lock_max_attempts() {
        context.check_cancellation("nexus_query_engine.lakehouse_cache.lock")?;
        match OpenOptions::new().write(true).create_new(true).open(lock_path) {
            Ok(mut file) => {
                let _ = writeln!(file, "pid={}", std::process::id());
                return Ok(LakehouseCacheLock { path: Some(lock_path.to_path_buf()) });
            },
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if remove_lakehouse_cache_lock_if_stale(lock_path) {
                    continue;
                }
                if cache_path.is_file() {
                    return Ok(LakehouseCacheLock { path: None });
                }
                if attempt + 1 < policy.cache_lock_max_attempts() {
                    thread::sleep(policy.cache_lock_wait());
                    continue;
                }
                let mut error = QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.lakehouse_cache.lock_timeout",
                    "timed out waiting for a lakehouse cache fill lock",
                )
                .with_fact("path", lakehouse_cache_path_fact(cache_path))
                .with_fact("lock_path", lakehouse_cache_path_fact(lock_path))
                .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id));
                if let Some(file_index) = file_index {
                    error = error.with_fact("file_index", file_index);
                }
                if let Some(chunk_index) = chunk_index {
                    error = error.with_fact("chunk_index", chunk_index);
                }
                return Err(error);
            },
            Err(error) => {
                return Err(lakehouse_cache_io_error(
                    error,
                    "nexus_query_engine.lakehouse_cache.lock",
                    "failed to create lakehouse cache fill lock",
                    cache_path,
                    stable_id,
                    file_index,
                    chunk_index,
                )
                .with_fact("lock_path", lakehouse_cache_path_fact(lock_path)));
            },
        }
    }
    unreachable!("lakehouse cache lock loop returns on timeout")
}

fn publish_lakehouse_cache_file(
    cache_path: &Path,
    bytes: &[u8],
    stable_id: &str,
    sidecar_content_identity: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> Result<()> {
    let temp_path = lakehouse_cache_temp_path(cache_path);
    let mut temp_file = OpenOptions::new().write(true).create_new(true).open(&temp_path).map_err(|error| {
        lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.temp",
            "failed to create lakehouse cache temp file",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path))
    })?;
    if let Err(error) = temp_file.write_all(bytes).and_then(|()| temp_file.sync_all()) {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.write",
            "failed to write lakehouse cache temp file",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path)));
    }
    drop(temp_file);
    if let Err(error) = fs::rename(&temp_path, cache_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.publish",
            "failed to atomically publish lakehouse cache file",
            cache_path,
            stable_id,
            file_index,
            chunk_index,
        )
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path)));
    }
    let _ = write_parquet_footer_sidecar_from_file(cache_path, sidecar_content_identity);
    Ok(())
}

fn sidecar_content_identity<'a>(
    resolved_source: &'a crate::source::ResolvedParquetSource,
    stable_id: &'a str,
) -> &'a str {
    resolved_source.content_identity.as_deref().unwrap_or(stable_id)
}

pub(super) fn lakehouse_cache_lock_path(cache_path: &Path) -> PathBuf {
    let file_name = cache_path.file_name().and_then(|value| value.to_str()).unwrap_or("cache-entry");
    cache_path.with_file_name(format!("{file_name}.lock"))
}

fn remove_lakehouse_cache_lock_if_stale(lock_path: &Path) -> bool {
    let Ok(lock_content) = fs::read_to_string(lock_path) else {
        return false;
    };
    if !Path::new("/proc").is_dir() {
        return false;
    }
    if let Some(pid) =
        lock_content.lines().find_map(|line| line.strip_prefix("pid=")).and_then(|pid| pid.trim().parse::<u32>().ok())
    {
        if pid == std::process::id() || Path::new("/proc").join(pid.to_string()).exists() {
            return false;
        }
    } else if !malformed_lakehouse_cache_lock_is_stale(lock_path) {
        return false;
    }
    fs::remove_file(lock_path).is_ok()
}

fn malformed_lakehouse_cache_lock_is_stale(lock_path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(lock_path) else {
        return false;
    };
    let Ok(modified) = metadata.modified() else {
        return false;
    };
    modified.elapsed().is_ok_and(|elapsed| elapsed >= LakehouseCachePolicy::default().malformed_cache_lock_stale_age())
}

fn lakehouse_cache_temp_path(cache_path: &Path) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
    let file_name = cache_path.file_name().and_then(|value| value.to_str()).unwrap_or("cache-entry");
    cache_path.with_file_name(format!(".{file_name}.tmp.{}.{}", std::process::id(), nonce))
}

pub(super) fn stable_log_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn lakehouse_cache_path_fact(path: &Path) -> String {
    format!("lakehouse_cache_path:fnv64:{}", stable_log_hash(&path.display().to_string()))
}

fn lakehouse_cache_stable_id_fact(stable_id: &str) -> String {
    format!("lakehouse_cache_stable_id:fnv64:{}", stable_log_hash(stable_id))
}

fn optional_usize_log(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

pub(super) fn lakehouse_cache_read_pin_path(cache_path: &Path) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
    let sequence = LAKEHOUSE_CACHE_READ_PIN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = cache_path.file_name().and_then(|value| value.to_str()).unwrap_or("cache-entry");
    cache_path.with_file_name(format!(".{file_name}.read.{}.{}.{}.pin", std::process::id(), nonce, sequence))
}

// Filesystem cache entry metadata used to choose eviction victims.
#[derive(Debug)]
pub(super) struct LakehouseCacheEntry {
    pub(in crate::exec::source) path: PathBuf,
    bytes: u64,
    last_used: SystemTime,
}

pub(super) fn evict_lakehouse_cache(cache_root: Option<&str>, protected_path: &Path, max_bytes: Option<u64>) {
    let Some(max_bytes) = max_bytes.filter(|value| *value > 0) else {
        return;
    };
    let Some(cache_root) = cache_root else {
        return;
    };
    let root = Path::new(cache_root);
    let Ok(mut entries) = collect_lakehouse_cache_entries(root, protected_path) else {
        return;
    };
    let mut retained_bytes = entries.iter().map(|entry| entry.bytes).sum::<u64>();
    if retained_bytes <= max_bytes {
        return;
    }

    entries.sort_by(|left, right| left.last_used.cmp(&right.last_used).then_with(|| left.path.cmp(&right.path)));
    for entry in entries {
        if retained_bytes <= max_bytes {
            break;
        }
        if entry.path == protected_path || lakehouse_cache_entry_has_sidecar_lock(&entry.path) {
            continue;
        }
        if fs::remove_file(&entry.path).is_ok() {
            let _ = fs::remove_file(crate::parquet_metadata::parquet_footer_sidecar_path(&entry.path));
            retained_bytes = retained_bytes.saturating_sub(entry.bytes);
        }
    }
}

pub(super) fn collect_lakehouse_cache_entries(
    root: &Path,
    protected_path: &Path,
) -> std::io::Result<Vec<LakehouseCacheEntry>> {
    let mut entries = Vec::new();
    collect_lakehouse_cache_entries_inner(root, protected_path, &mut entries)?;
    Ok(entries)
}

fn collect_lakehouse_cache_entries_inner(
    dir: &Path,
    protected_path: &Path,
    entries: &mut Vec<LakehouseCacheEntry>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_lakehouse_cache_entries_inner(&path, protected_path, entries)?;
            continue;
        }
        if !metadata.is_file() || !is_lakehouse_cache_data_file(&path) {
            continue;
        }
        let last_used = if path == protected_path {
            SystemTime::now()
        } else {
            metadata.accessed().or_else(|_| metadata.modified()).unwrap_or(UNIX_EPOCH)
        };
        entries.push(LakehouseCacheEntry { path, bytes: metadata.len(), last_used });
    }
    Ok(())
}

fn is_lakehouse_cache_data_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    !(name.ends_with(".lock")
        || name.ends_with(".footer.v1")
        || is_lakehouse_cache_read_pin_name(name)
        || (name.starts_with('.') && name.contains(".tmp.")))
}

fn lakehouse_cache_entry_has_sidecar_lock(path: &Path) -> bool {
    lakehouse_cache_lock_path(path).exists() || lakehouse_cache_entry_has_read_pin(path)
}

pub(super) fn lakehouse_cache_entry_has_read_pin(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    let prefix = format!(".{file_name}.read.");
    let Ok(entries) = fs::read_dir(parent) else {
        return true;
    };
    entries.filter_map(std::result::Result::ok).any(|entry| {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            return false;
        };
        name.starts_with(&prefix) && name.ends_with(".pin") && lakehouse_cache_read_pin_is_active(&entry.path(), name)
    })
}

fn is_lakehouse_cache_read_pin_name(name: &str) -> bool {
    name.starts_with('.') && name.contains(".read.") && name.ends_with(".pin")
}

fn lakehouse_cache_read_pin_is_active(pin_path: &Path, pin_name: &str) -> bool {
    let Some(read_suffix) = pin_name.rsplit_once(".read.").map(|(_, suffix)| suffix) else {
        return true;
    };
    let Some(pid_text) = read_suffix.split('.').next().filter(|value| !value.is_empty()) else {
        return true;
    };
    let Ok(pid) = pid_text.parse::<u32>() else {
        return true;
    };
    if pid == std::process::id() {
        return true;
    }
    if !Path::new("/proc").is_dir() {
        return true;
    }
    let active = Path::new("/proc").join(pid.to_string()).exists();
    if !active {
        let _ = fs::remove_file(pin_path);
    }
    active
}

pub(super) fn lakehouse_cache_io_error(
    error: std::io::Error,
    operation: &'static str,
    message: &'static str,
    cache_path: &Path,
    stable_id: &str,
    file_index: Option<usize>,
    chunk_index: Option<usize>,
) -> QueryError {
    let mut error = QueryError::new(QueryErrorCode::CudfDependencyFailure, operation, message)
        .with_fact("path", lakehouse_cache_path_fact(cache_path))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id))
        .with_source(error);
    if let Some(file_index) = file_index {
        error = error.with_fact("file_index", file_index);
    }
    if let Some(chunk_index) = chunk_index {
        error = error.with_fact("chunk_index", chunk_index);
    }
    error
}

async fn stream_lakehouse_cache_fill(
    store: std::sync::Arc<dyn object_store::ObjectStore>,
    location: object_store::path::Path,
    cache_path: PathBuf,
    temp_path: PathBuf,
    lock_guard: BackgroundLakehouseCacheLock,
    stable_id: String,
    sidecar_content_identity: String,
    estimated_fetch_bytes: Option<u64>,
    mut budget_permit: crate::runtime::HostFetchPermit,
) -> Result<u64> {
    let _lock = lock_guard;
    if complete_cache_file_matches_expected_size(&cache_path, estimated_fetch_bytes) {
        return Ok(0);
    }
    let result = store.get(&location).await.map_err(|_error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.lakehouse_cache.background_fill.fetch",
            "failed to start background lakehouse cache object fetch",
        )
        .with_fact("path", lakehouse_cache_path_fact(&cache_path))
        .with_fact("object_store_path", object_store_key_fact(&location))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id.as_str()))
    })?;
    let mut temp_file = OpenOptions::new().write(true).create_new(true).open(&temp_path).map_err(|error| {
        lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.background_fill.temp",
            "failed to create background lakehouse cache temp file",
            &cache_path,
            stable_id.as_str(),
            None,
            None,
        )
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path))
    })?;

    let mut total_bytes = 0u64;
    let mut stream = result.into_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_error| {
            let _ = fs::remove_file(&temp_path);
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.lakehouse_cache.background_fill.fetch",
                "failed to stream background lakehouse cache object bytes",
            )
            .with_fact("path", lakehouse_cache_path_fact(&cache_path))
            .with_fact("object_store_path", object_store_key_fact(&location))
            .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id.as_str()))
        })?;
        total_bytes = total_bytes
            .checked_add(u64::try_from(chunk.len()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.lakehouse_cache.background_fill.size",
                    "background lakehouse cache chunk length exceeds u64 range",
                )
            })?)
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.lakehouse_cache.background_fill.size",
                    "background lakehouse cache byte count overflows u64",
                )
            })?;
        if total_bytes > budget_permit.bytes() {
            budget_permit.grow_to(total_bytes).map_err(|error| {
                let _ = fs::remove_file(&temp_path);
                error
                    .with_fact_if_absent("path", lakehouse_cache_path_fact(&cache_path))
                    .with_fact_if_absent("stable_id", lakehouse_cache_stable_id_fact(stable_id.as_str()))
                    .with_fact_if_absent("background_fill_bytes", total_bytes)
            })?;
        }
        if let Err(error) = temp_file.write_all(&chunk) {
            let _ = fs::remove_file(&temp_path);
            return Err(lakehouse_cache_io_error(
                error,
                "nexus_query_engine.lakehouse_cache.background_fill.write",
                "failed to write background lakehouse cache temp file",
                &cache_path,
                stable_id.as_str(),
                None,
                None,
            )
            .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path)));
        }
    }
    if let Some(expected_bytes) = estimated_fetch_bytes
        && total_bytes != expected_bytes
    {
        let _ = fs::remove_file(&temp_path);
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.lakehouse_cache.background_fill.size",
            "background lakehouse cache fetch byte count did not match source metadata",
        )
        .with_fact("path", lakehouse_cache_path_fact(&cache_path))
        .with_fact("stable_id", lakehouse_cache_stable_id_fact(stable_id.as_str()))
        .with_fact("expected_bytes", expected_bytes)
        .with_fact("actual_bytes", total_bytes));
    }
    if let Err(error) = temp_file.sync_all() {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.background_fill.sync",
            "failed to sync background lakehouse cache temp file",
            &cache_path,
            stable_id.as_str(),
            None,
            None,
        )
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path)));
    }
    drop(temp_file);
    if let Err(error) = fs::rename(&temp_path, &cache_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(lakehouse_cache_io_error(
            error,
            "nexus_query_engine.lakehouse_cache.background_fill.publish",
            "failed to atomically publish background lakehouse cache file",
            &cache_path,
            stable_id.as_str(),
            None,
            None,
        )
        .with_fact("temp_path", lakehouse_cache_path_fact(&temp_path)));
    }
    let _ = write_parquet_footer_sidecar_from_file(&cache_path, sidecar_content_identity.as_str());
    Ok(total_bytes)
}
