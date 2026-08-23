//! Local Parquet chunk plan entry points.

use std::time::Duration;

use nexus_query_planning::{
    chunk::{
        assembly::{
            append_row_group_chunks, planned_row_groups, single_all_row_groups_chunk, zero_column_count_identity_chunk,
        },
        scan_limit::apply_source_scan_limit_to_chunks,
    },
    error::{QueryError, QueryErrorCode, Result},
    plan::NodeId,
    source::{LocalParquetFile, LocalParquetSource},
};

use super::{
    SourceChunkPlan, SourceChunkPlanningPolicy, SourceMetadataCache,
    metadata::{SourceMetadataLoadContext, read_parquet_metadata},
};
use crate::runtime::ExecutionContext;

/// Plans chunks for a local Parquet source using a fresh footer metadata cache.
pub fn plan_local_parquet_source_chunks(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
) -> Result<SourceChunkPlan> {
    let mut metadata_cache = SourceMetadataCache::default();
    plan_local_parquet_source_chunks_with_metadata_cache_and_context(
        source_node_id,
        source,
        policy,
        &mut metadata_cache,
        None,
    )
}

/// Plans chunks using the execution context to opportunistically range-read
/// object-store Parquet footers before falling back to conservative read-all
/// chunks.
pub fn plan_local_parquet_source_chunks_with_execution_context(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
    context: &ExecutionContext,
) -> Result<SourceChunkPlan> {
    let mut metadata_cache = SourceMetadataCache::default();
    plan_local_parquet_source_chunks_with_metadata_cache_and_context(
        source_node_id,
        source,
        policy,
        &mut metadata_cache,
        Some(SourceMetadataLoadContext { execution_context: context }),
    )
}

/// Plans chunks for a local Parquet source using a caller-owned footer cache.
///
/// The cache is keyed by source path, stable id, and known file size so repeated
/// planning of the same fixture can avoid re-reading footers.
pub fn plan_local_parquet_source_chunks_with_metadata_cache(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
    metadata_cache: &mut SourceMetadataCache,
) -> Result<SourceChunkPlan> {
    plan_local_parquet_source_chunks_with_metadata_cache_and_context(
        source_node_id,
        source,
        policy,
        metadata_cache,
        None,
    )
}

fn plan_local_parquet_source_chunks_with_metadata_cache_and_context(
    source_node_id: NodeId,
    source: &LocalParquetSource,
    policy: SourceChunkPlanningPolicy,
    metadata_cache: &mut SourceMetadataCache,
    metadata_load_context: Option<SourceMetadataLoadContext<'_>>,
) -> Result<SourceChunkPlan> {
    validate_chunk_planning_source(source)?;
    validate_chunk_planning_policy(policy)?;

    let mut chunks = Vec::new();
    let mut saw_metadata = false;
    let mut saw_missing_metadata = false;
    let mut metadata_unavailable_files = 0;
    let mut metadata_load_time = Duration::ZERO;
    let mut metadata_load_count = 0u64;
    let metadata_prefetch =
        metadata_cache.prefetch_footer_metadata_for_source(source, metadata_load_context.as_ref())?;
    metadata_load_time += metadata_prefetch.load_time;
    metadata_load_count = metadata_load_count.saturating_add(metadata_prefetch.load_count);

    for (file_index, file) in source.files.iter().enumerate() {
        let metadata_load = read_parquet_metadata(source, file, metadata_cache, metadata_load_context.as_ref())?;
        metadata_load_time += metadata_load.load_time;
        metadata_load_count = metadata_load_count.saturating_add(metadata_load.load_count);
        let reader_decimal_types = metadata_load.reader_decimal_types;
        let reader_decimal_types_footer_verified = metadata_load.reader_decimal_types_footer_verified;
        let metadata = metadata_load.metadata;
        saw_metadata |= metadata.is_some();
        if metadata.is_none() {
            saw_missing_metadata = true;
            metadata_unavailable_files += 1;
        }

        let row_groups = planned_row_groups(file, file_index, metadata.as_ref())?;
        let Some(row_groups) = row_groups else {
            chunks.push(single_all_row_groups_chunk(
                source_node_id,
                source,
                file,
                file_index,
                chunks.len(),
                policy,
                &reader_decimal_types,
                reader_decimal_types_footer_verified,
            ));
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
            &reader_decimal_types,
            reader_decimal_types_footer_verified,
            &mut chunks,
        );
    }

    chunks = apply_source_scan_limit_to_chunks(source, chunks)?;

    if source.is_zero_column_count_source() && chunks.is_empty() {
        let (file_index, file) = source.files.iter().enumerate().next().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.plan",
                "represented zero-column count source requires at least one file",
            )
            .with_fact("reason", "empty_files")
        })?;
        chunks.push(zero_column_count_identity_chunk(source_node_id, source, file, file_index, 0, policy, &[], false));
    }

    if source.read_columns.is_empty()
        && let Some(chunk) = chunks.iter().find(|chunk| chunk.known_row_count.is_none())
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "zero-read-column source requires parquet row-count metadata for each planned chunk",
        )
        .with_fact("reason", "row_count_metadata_unavailable")
        .with_fact("path", nexus_query_planning::source::parquet_source_path_fact(&chunk.location, &chunk.path))
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

fn validate_chunk_planning_policy(policy: SourceChunkPlanningPolicy) -> Result<()> {
    if policy.max_row_groups_per_chunk == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.policy",
            "source chunk planning requires max_row_groups_per_chunk to be greater than zero",
        )
        .with_fact("field", "max_row_groups_per_chunk")
        .with_fact("reason", "zero"));
    }

    Ok(())
}

pub(super) fn validate_chunk_planning_source(source: &LocalParquetSource) -> Result<()> {
    if source.files.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "local parquet source must contain at least one file for chunk planning",
        )
        .with_fact("reason", "empty_files"));
    }

    if source.read_columns.is_empty()
        && !source.is_zero_column_count_source()
        && source.missing_columns.is_empty()
        && source.partition_columns.is_empty()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_chunk.plan",
            "local parquet source must name at least one required read column for chunk planning",
        )
        .with_fact("reason", "empty_read_columns"));
    }

    for (file_index, file) in source.files.iter().enumerate() {
        if file.path.as_os_str().is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_chunk.plan",
                "local parquet source path must not be empty for chunk planning",
            )
            .with_fact("file_index", file_index));
        }
        validate_file_row_group_metadata_lengths(file, file_index)?;
    }

    Ok(())
}

pub(super) fn validate_file_row_group_metadata_lengths(file: &LocalParquetFile, file_index: usize) -> Result<()> {
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
                "nexus_query_engine.source_chunk.plan",
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
