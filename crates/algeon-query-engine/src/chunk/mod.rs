//! Source chunk planning and runtime metadata facade.

#[cfg(test)]
pub(crate) use query_planning::chunk::{
    LOCAL_PARQUET_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK, LOCAL_PARQUET_THROUGHPUT_MIN_DEVICE_BUDGET_BYTES, decimal,
};
pub(crate) use query_planning::chunk::{
    SourceChunk, SourceChunkBatch, SourceChunkPlan, annotate_source_chunk_batching_reasons,
    decimal::object_store_decimal_reader_predicate_requires_footer_contract, plan_source_chunk_batches,
    plan_source_chunk_batches_with_row_cap, source_file_for_chunk,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use crate::source_chunk::metadata;
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub use crate::source_chunk::{
    SourceMetadataCache, plan_local_parquet_source_chunks, plan_local_parquet_source_chunks_with_metadata_cache,
};
pub(crate) use crate::source_chunk::{
    metadata::{
        footer_verified_object_store_reader_decimal_types,
        footer_verified_object_store_reader_decimal_types_with_supplied_cache,
    },
    plan_local_parquet_source_chunks_with_execution_context,
    plan_local_parquet_source_chunks_with_metadata_cache_and_context, source_footer_metadata,
};
