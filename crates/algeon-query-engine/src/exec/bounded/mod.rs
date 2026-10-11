use query_planning::{
    admission::{AdmissionDecision, AdmittedPlan, admit},
    mode::StatisticalAggregateExecutionMode,
    plan::ResultSink,
    policy::PlanningPolicy,
};
use query_runtime::observability::RuntimeExecutionObservationScope;

use super::{
    AggregateFunction, AggregateNode, AggregateSpec, Arc, AvgStateAliases, CudfSortOptions, DataFrame,
    DropDuplicatesOptions, DuplicateKeepOption, Exec, ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, Expr,
    Field, FilterNode, GpuDataFrame, GpuFilteredJoinBuildHandle, GpuHashJoinBuildHandle, GpuResidentInputMap,
    GpuResidentSource, Instant, JoinNode, JoinOutputColumn, JoinSide, JoinType, LimitNode, LimitSpec,
    LocalParquetSource, MemoryControlPolicy, NodeId, NullAwareAntiProbe, PartialAggregateInputEvidence, ProjectionNode,
    QueryError, QueryErrorCode, QueryOutput, QueryPlan, RecordBatch, Result, RetryFacts, SortNode, SourceChunk,
    SourceChunkBatch, SourceChunkPlanningPolicy, SourceTaskMemoryModel, aggregate_spec_has_avg,
    aggregate_spec_has_distinct, aggregate_spec_has_floating_statistical_aggregate, aggregate_spec_has_median,
    aggregate_spec_requires_whole_input, annotate_source_chunk_batching_reasons, attach_budget_token_failure_facts,
    bounded_invariant_violation, budgeted_value, cast_count_output_columns_with_budget, cudf_query_error,
    dataframe_cross_join_options, dataframe_mixed_join_options, dataframe_predicate_join_options,
    datafusion_order_columns, execute_aggregate_with_budget, execute_aggregate_with_budget_for_node,
    execute_datafusion_exact_host_aggregate_batch, execute_edge_normalize, execute_filter_with_budget,
    execute_global_aggregate_with_budget, execute_groupby_aggregate_with_budget_observed,
    execute_partial_aggregate_with_budget, execute_projection_with_budget, execute_projection_with_budget_and_prepared,
    execution_budget_token_for_boundary, final_global_aggregate_state_output_schema,
    finalize_avg_output_columns_with_budget, global_valid_count_state_alias, join_index_error,
    join_input_columns_with_order, join_side_projection_with_order, limit_keeps_all_rows, limit_slice,
    lower_aggregate_spec_keys, lower_column_expr, lower_expr, lower_final_aggregate_spec,
    lower_join_residual_predicate, lower_named_column_expr, lower_null_equality, lower_sort_spec,
    materialize_bounded_frame_to_host, materialize_datafusion_exact_host_aggregate_input,
    materialize_join_output_projection, materialize_left_only_join_output_projection,
    materialize_mark_join_output_projection, null_aware_anti_probe, null_aware_anti_probe_input,
    observe_device_memory_bytes, partial_aggregate_source_chunk_fingerprint, partial_global_aggregate_output_schema,
    plan_local_parquet_source_chunks_with_execution_context, plan_source_chunk_batches,
    plan_source_chunk_batches_with_row_cap, prepare_aggregate_input_frame,
    prepare_datafusion_exact_host_aggregate_input_frame, probe_side, projected_input_columns, projected_output_columns,
    projection_regexp_replace_prepare, read_gpu_resident_source, read_literal_batch_source, record_batch_memory_size,
    record_default_grouped_aggregate_strategy_decision, record_device_memory_observation,
    record_grouped_aggregate_strategy_decision, reorder_projected_cross_join_output, residual_input_columns,
    select_join_input, should_execute_datafusion_exact_floating_statistical_aggregate, side_metric,
    source::record_bounded_parquet_source_metrics, topk_candidate_rows, upload_datafusion_exact_host_aggregate_output,
    validate_aggregate_output_names, validate_executable_join_subset, validate_local_parquet_source,
};
#[cfg(feature = "cugraph")]
use super::{edge_null_filter, execute_graph_algorithm_sanitized};
use crate::rapids_backend::rapids_analysis_context;

// Bounded execution ownership is divided among the modules below.
// - pipeline: the bounded execution facade.
// - compiled: the shared compiled pipeline runtime.
// - remaining modules: bounded runtime/operator support; none owns a separate
//   admission or reporting mode.
mod aggregate;
mod attempt;
pub(in crate::exec) mod chunks;
mod compiled;
mod facts;
mod final_sort;
pub(in crate::exec) mod join;
mod operator;
mod partition_row_policy;
mod partition_target_policy;
pub(in crate::exec) mod pipeline;
mod resident;
mod source_batching_policy;

pub(in crate::exec::bounded) use aggregate::apply_source_adjacent_aggregate_chunk_caps;
use aggregate::{
    BoundedPlan, BoundedUnaryStage, SourceAggregateTopK, execute_compiled_bounded_aggregate_from_frame,
    execute_compiled_bounded_aggregate_from_source_stream, execute_compiled_bounded_aggregate_from_stream,
    execute_compiled_source_aggregate_chunk, is_source_adjacent_grouped_aggregate, reduce_merge_fan_in,
};
pub(in crate::exec) use attempt::BoundedExecutionAttempt;
use chunks::{
    RawSourceInput, chunk_retry_facts, concatenate_source_read_outputs, execute_bounded_raw_source_chunk,
    execute_bounded_raw_source_chunk_batch, is_retryable_chunk_error, join_leaf_prefetch_admitted,
    read_local_parquet_chunks_required_only, record_chunk_retry_error, split_retryable_chunk,
};
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::{attach_chunk_resource_facts, attach_source_batch_resource_facts};
use final_sort::{
    execute_final_sort_node_from_committed_with_retry, execute_final_sort_node_from_frame_with_retry,
    execute_final_sort_node_from_stream_with_retry, execute_final_sort_node_with_retry,
};
pub(in crate::exec::bounded) use join::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
    BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream, PartitionedJoinBuildCommitment,
    consume_bounded_join_build_input_stream, execute_bounded_compiled_join_build_from_resident_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_materialize_preserving_stream,
    execute_bounded_compiled_join_probe_from_frame_stream, execute_bounded_compiled_join_probe_from_resident_frame,
    execute_bounded_compiled_join_probe_from_stream, materialize_bounded_join_leaf_local_parquet_source_stream,
    partitioned_join_build_commitment, plan_bounded_aggregate_input_local_parquet_source_stream,
    plan_bounded_join_leaf_local_parquet_source_stream, plan_bounded_shared_join_aggregate_local_parquet_source_stream,
    plan_bounded_unary_chain_local_parquet_source_stream, release_bounded_compiled_join_probe_state,
    stateless_streaming_probe_supported, stream_bounded_join_leaf_local_parquet_source_chunks,
    try_start_bounded_join_leaf_local_parquet_source_prefetch,
};
#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) use operator::execute_bounded_graph_algorithm_from_inputs;
pub(in crate::exec::bounded) use operator::{
    execute_bounded_edge_normalize_from_input, execute_bounded_projection, execute_bounded_unary, run_budgeted,
};
use operator::{
    execute_bounded_gpu_resident_source, execute_bounded_literal_batch_source, execute_bounded_local_parquet_source,
    execute_bounded_union, union_branch_with_output_names,
};
pub(in crate::exec::bounded) use partition_row_policy::BoundedPartitionRowPolicy;
pub(in crate::exec::bounded) use partition_target_policy::BoundedPartitionTargetPolicy;
use pipeline::execute_bounded_pipeline;
pub(in crate::exec) use resident::{CommittedPartial, ResidentGpuFrame};
pub(in crate::exec::bounded) use source_batching_policy::BoundedSourceBatchingPolicy;

/// Entry from `exec` into bounded execution. The cancellation check here is
/// the last one before the compiled pipeline takes over; later checks sit at
/// task and chunk boundaries inside it.
pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    admitted: &AdmittedPlan,
    gpu_inputs: GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    context.check_cancellation("query_engine.exec.bounded.execute_query_plan")?;
    execute_bounded_pipeline(context, admitted, gpu_inputs, metrics, observation_scope)
}

/// Decides once whether `plan` runs as a compiled pipeline and returns the
/// admitted artifact execution runs.
///
/// A rejection becomes a typed error built from the same decision. A
/// validation refusal is the validator's own error; any other rejection
/// carries the caller-visible sink as a diagnostic fact that never selects a
/// second decision.
pub(super) fn admit_or_execution_error(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    policy: PlanningPolicy,
) -> Result<Arc<AdmittedPlan>> {
    match admit(plan, statistical_aggregate_execution_mode, policy, rapids_analysis_context(policy))? {
        AdmissionDecision::Admitted(admission) => Ok(admission),
        AdmissionDecision::Rejected(rejection) if rejection.validation_error().is_some() => {
            Err(rejection.execution_error(plan))
        },
        AdmissionDecision::Rejected(rejection) => {
            Err(rejection.execution_error(plan).with_fact("caller_visible_sink", caller_visible_sink_kind(plan.sink())))
        },
    }
}

fn caller_visible_sink_kind(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::HostArrow { .. } | ResultSink::DeferredHostArrow { .. } => "host_arrow",
        ResultSink::GpuResident { .. } => "gpu_resident",
    }
}
