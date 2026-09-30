use std::time::Instant;

use super::{
    BoundedExecutionAttempt, BoundedPartitionRowPolicy, BoundedPartitionTargetPolicy, BoundedSourceBatchingPolicy,
    CommittedPartial, DataFrame, DropDuplicatesOptions, DuplicateKeepOption, ExecutionBudgetToken, ExecutionContext,
    ExecutionMetrics, ExecutionOverrides, Expr, FilterNode, GpuDataFrame, GpuFilteredJoinBuildHandle,
    GpuHashJoinBuildHandle, GpuResidentInputMap, JoinNode, JoinOutputColumn, JoinSide, JoinType, LocalParquetSource,
    MemoryControlPolicy, NodeId, NullAwareAntiProbe, ProjectionNode, QueryError, QueryErrorCode, ResidentGpuFrame,
    Result, RetryFacts, SourceAdjacentAggregateChunkCapHints, SourceChunk, SourceChunkBatch, SourceChunkPlanningPolicy,
    SourceTaskMemoryModel, annotate_source_chunk_batching_reasons, apply_source_adjacent_aggregate_chunk_caps,
    attach_budget_token_failure_facts, chunk_retry_facts, cudf_query_error, dataframe_cross_join_options,
    dataframe_mixed_join_options, dataframe_predicate_join_options, datafusion_order_columns,
    execute_filter_with_budget, execute_projection_with_budget, execution_budget_token_for_boundary,
    execution_overrides, is_retryable_chunk_error, join_index_error, join_input_columns_with_order,
    join_leaf_prefetch_admitted, join_side_projection_with_order, lower_expr, lower_join_residual_predicate,
    lower_null_equality, materialize_join_output_projection, materialize_left_only_join_output_projection,
    materialize_mark_join_output_projection, null_aware_anti_probe, null_aware_anti_probe_input,
    observe_device_memory_bytes, plan_local_parquet_source_chunks_with_execution_context, plan_source_chunk_batches,
    probe_side, projected_input_columns, projected_output_columns, read_gpu_resident_source, read_literal_batch_source,
    record_bounded_parquet_source_metrics, record_chunk_retry_error, reorder_projected_cross_join_output,
    residual_input_columns, select_join_input, side_metric, split_retryable_chunk, unsupported_bounded_shape,
    validate_executable_join_subset, validate_local_parquet_source,
};

mod common;
pub(in crate::exec) mod compiled;
mod hash;
mod leaf;
mod probe_parallel;

pub(in crate::exec::bounded) use compiled::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
    PartitionedJoinBuildCommitment, consume_bounded_join_build_input_stream,
    execute_bounded_compiled_join_build_from_frame, execute_bounded_compiled_join_build_from_resident_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_materialize_preserving_stream,
    execute_bounded_compiled_join_probe_from_frame, execute_bounded_compiled_join_probe_from_frame_stream,
    execute_bounded_compiled_join_probe_from_resident_frame, execute_bounded_compiled_join_probe_from_stream,
    partitioned_join_build_commitment, release_bounded_compiled_join_probe_state, stateless_streaming_probe_supported,
};
use hash::{
    BoundedRootHashJoinBuildState, BoundedRootHashJoinProbeState, execute_bounded_root_hash_join_build_from_frame,
    execute_bounded_root_hash_join_build_from_resident_frame, execute_bounded_root_hash_join_materialize,
    execute_bounded_root_hash_join_materialize_preserving_stream,
    execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame,
    execute_bounded_root_hash_join_probe_from_frame, execute_bounded_root_hash_join_probe_from_resident_frame,
    release_bounded_root_hash_join_build_state, release_bounded_root_hash_join_probe_state,
};
pub(super) use leaf::{
    BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream,
    bounded_join_leaf_local_parquet_source_frame_stream, execute_bounded_join_leaf_source,
    materialize_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream, stream_bounded_join_leaf_local_parquet_source,
    stream_bounded_join_leaf_local_parquet_source_chunks, stream_bounded_join_leaf_local_parquet_source_serial,
    try_start_bounded_join_leaf_local_parquet_source_prefetch,
};
pub(in crate::exec::bounded) use probe_parallel::{
    hash_parallel_probe_supported, parallel_hash_probe_frame_stream, serial_probe_reason,
};

use crate::materialization_sizing::{JoinOutputColumnUpperBoundInput, join_output_upper_bound_bytes};
