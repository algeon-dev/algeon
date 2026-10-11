use query_planning::{
    admission::{AdmissionDecision, AdmittedPlan, admit},
    mode::StatisticalAggregateExecutionMode,
    plan::ResultSink,
    policy::PlanningPolicy,
};
use query_runtime::observability::RuntimeExecutionObservationScope;

use super::{source::record_bounded_parquet_source_metrics, *};
use crate::rapids_backend::rapids_analysis_context;

// Bounded execution ownership boundary:
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
use aggregate::*;
pub(in crate::exec) use attempt::BoundedExecutionAttempt;
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
use final_sort::*;
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
