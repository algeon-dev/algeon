//! Source chunk planning and runtime metadata facade.

#[cfg(test)]
pub(crate) use nexus_query_planning::chunk::decimal;
pub use nexus_query_planning::chunk::{
    LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK, LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES, SourceChunk,
    SourceChunkBatch, SourceChunkPlan, SourceChunkPlanningOptions, SourceChunkPlanningPolicy,
    SourceChunkRowPositionRange, SourceTaskMemoryModel, annotate_source_chunk_batching_reasons,
    local_parquet_source_has_exact_read_preflight, plan_source_chunk_batches, plan_source_chunk_batches_with_row_cap,
};
pub(crate) use nexus_query_planning::chunk::{
    object_store_decimal_reader_predicate_requires_footer_contract, source_chunk_row_position_range,
    source_file_for_chunk,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use nexus_query_runtime::runtime::source_chunk::metadata;
pub(crate) use nexus_query_runtime::runtime::source_chunk::metadata::{
    footer_verified_object_store_reader_decimal_types,
    footer_verified_object_store_reader_decimal_types_with_supplied_cache,
};
pub use nexus_query_runtime::runtime::source_chunk::{
    SourceMetadataCache, plan_local_parquet_source_chunks, plan_local_parquet_source_chunks_with_execution_context,
    plan_local_parquet_source_chunks_with_metadata_cache,
};
