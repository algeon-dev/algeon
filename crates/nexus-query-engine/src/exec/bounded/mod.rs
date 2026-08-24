use std::sync::Arc;

use super::{source::record_bounded_parquet_source_metrics, *};
use crate::mode::StatisticalAggregateExecutionMode;

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
mod generic;
pub(in crate::exec) mod join;
mod partition_row_policy;
mod partition_target_policy;
pub(in crate::exec) mod pipeline;
mod resident;
mod source_batching_policy;

use aggregate::*;
pub use aggregate::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};
pub(in crate::exec::bounded) use aggregate::{
    SourceAdjacentAggregateChunkCapHints, apply_source_adjacent_aggregate_chunk_caps,
};
pub(in crate::exec) use attempt::BoundedExecutionAttempt;
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
use final_sort::*;
#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) use generic::execute_bounded_graph_algorithm_from_inputs;
pub(in crate::exec::bounded) use generic::{
    execute_bounded_edge_normalize_from_input, execute_bounded_generic_projection,
    execute_bounded_generic_projection_from_frame, execute_bounded_generic_unary,
    execute_bounded_generic_unary_from_frame,
};
use generic::{
    execute_bounded_generic_gpu_resident_source, execute_bounded_generic_literal_batch_source,
    execute_bounded_generic_local_parquet_source, execute_bounded_generic_query_plan,
    execute_bounded_generic_unary_consuming_input, execute_bounded_generic_union,
};
pub(in crate::exec::bounded) use join::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream,
    execute_bounded_compiled_join_build_from_frame, execute_bounded_compiled_join_build_from_resident_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_materialize_preserving_stream,
    execute_bounded_compiled_join_probe_from_frame, execute_bounded_compiled_join_probe_from_frame_stream,
    execute_bounded_compiled_join_probe_from_resident_frame, execute_bounded_compiled_join_probe_from_stream,
    execute_bounded_join_leaf_source, materialize_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream, release_bounded_compiled_join_probe_state,
    stream_bounded_join_leaf_local_parquet_source_chunks, try_start_bounded_join_leaf_local_parquet_source_prefetch,
};
pub use join::{
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, STRING_OFFSET_BYTES_CONSERVATIVE,
    join_output_upper_bound_bytes,
};
pub(in crate::exec::bounded) use partition_row_policy::BoundedPartitionRowPolicy;
pub(in crate::exec::bounded) use partition_target_policy::BoundedPartitionTargetPolicy;
use pipeline::try_execute_bounded_pipeline;
pub(in crate::exec) use resident::{CommittedPartial, ResidentGpuFrame};
pub(in crate::exec::bounded) use source_batching_policy::BoundedSourceBatchingPolicy;

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    context.check_cancellation("nexus_query_engine.exec.bounded.execute_query_plan")?;
    if let Some(output) = try_execute_bounded_pipeline(context, plan, gpu_inputs, metrics, observation_scope)? {
        return Ok(output);
    }

    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    if !plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Aggregate(_))) {
        return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics, observation_scope);
    }

    let bounded = match BoundedPlan::analyze(plan) {
        Ok(bounded) => bounded,
        Err(_error) => {
            return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics, observation_scope);
        },
    };
    let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics, observation_scope)?;
    let frame = execute_bounded_aggregate_to_frame(&mut execution, &bounded)?;
    let frame = execute_post_aggregate_stages(&mut execution, &bounded, frame)?;
    let frame = execute_final_sort_limit_with_retry(&mut execution, &bounded, frame)?;
    let frame = execute_post_sort_projections(&mut execution, &bounded, frame)?;
    let frame = execute_standalone_limit(&mut execution, &bounded, frame)?;

    record_device_memory_observation(&frame.frame, execution.metrics_mut())?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let output_bytes = frame.frame.device_allocated_size_bytes()?;
            let resident_owner = execution.context().retain_resident_output_owner(output_bytes)?;
            let output_frame = frame
                .frame
                .with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(plan));
            let handle = match GpuResultHandle::new_with_resident_owner(
                execution.context(),
                output_frame,
                Arc::clone(plan.output_schema()),
                resident_owner,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    return Err(error);
                },
            };
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                &mut execution,
                frame,
                plan.sink().input(),
                plan.node(plan.sink().input()).expect("validated sink input exists").output_schema(),
                schema,
            )?;
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

pub(super) fn preflight_bounded_join_plan(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
) -> Result<()> {
    if plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Join(_))) {
        // A frontend-owned incremental D2H cursor uses a GPU-resident physical
        // sink, while capability diagnostics must describe the caller-visible
        // host sink.
        let rejection_contract_plan = plan.clone_with_caller_visible_sink_contract();
        let plan = &rejection_contract_plan;
        let mut error = unsupported_bounded_shape(
            "bounded native join execution requires an executable compiled pipeline",
            "bounded_join_plan_shape_unsupported",
        );
        let mut projection_error_source = None;
        let rejection_projection =
            match crate::admission::bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
                plan,
                statistical_aggregate_execution_mode,
            ) {
                Ok(Some(projection)) => Some(projection),
                Ok(None) => return Ok(()),
                Err(projection_error) => {
                    projection_error_source = Some(projection_error);
                    None
                },
            };
        let mut facts =
            crate::admission::bounded_join_unsupported_shape_facts_with_projection(plan, rejection_projection.as_ref());
        if let Some(projection_error) = projection_error_source.as_ref() {
            crate::admission::insert_bounded_join_rejection_projection_error_facts(&mut facts, projection_error);
        }
        for (key, value) in facts {
            error = error.with_fact(key, value);
        }
        if let Some(projection_error) = projection_error_source {
            error = error.with_source(projection_error);
        }
        return Err(error);
    }

    Ok(())
}

fn reject_bounded_join_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<()> {
    preflight_bounded_join_plan(plan, context.statistical_aggregate_execution_mode())
}
