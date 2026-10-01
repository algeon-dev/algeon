use std::sync::Arc;

use super::{source::record_bounded_parquet_source_metrics, *};
use crate::{
    admission::{
        BoundedPipelineAdmission, bounded_join_unsupported_shape_facts_with_projection, bounded_pipeline_decision,
        insert_bounded_join_rejection_projection_error_facts,
    },
    mode::StatisticalAggregateExecutionMode,
    observability::RuntimeExecutionObservationScope,
    plan::ResultSink,
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
};

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
mod operator;
mod partition_row_policy;
mod partition_target_policy;
pub(in crate::exec) mod pipeline;
mod resident;
mod source_batching_policy;

use aggregate::*;
pub(in crate::exec::bounded) use aggregate::{
    SourceAdjacentAggregateChunkCapHints, apply_source_adjacent_aggregate_chunk_caps,
};
pub(in crate::exec) use attempt::BoundedExecutionAttempt;
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
use final_sort::*;
use generic::execute_bounded_generic_query_plan;
pub(in crate::exec::bounded) use join::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
    BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream, PartitionedJoinBuildCommitment,
    consume_bounded_join_build_input_stream, execute_bounded_compiled_join_build_from_frame,
    execute_bounded_compiled_join_build_from_resident_frame, execute_bounded_compiled_join_materialize,
    execute_bounded_compiled_join_materialize_preserving_stream, execute_bounded_compiled_join_probe_from_frame,
    execute_bounded_compiled_join_probe_from_frame_stream, execute_bounded_compiled_join_probe_from_resident_frame,
    execute_bounded_compiled_join_probe_from_stream, execute_bounded_join_leaf_source,
    materialize_bounded_join_leaf_local_parquet_source_stream, partitioned_join_build_commitment,
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream, release_bounded_compiled_join_probe_state,
    stateless_streaming_probe_supported, stream_bounded_join_leaf_local_parquet_source_chunks,
    try_start_bounded_join_leaf_local_parquet_source_prefetch,
};
#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) use operator::execute_bounded_graph_algorithm_from_inputs;
pub(in crate::exec::bounded) use operator::{
    execute_bounded_edge_normalize_from_input, execute_bounded_projection, execute_bounded_projection_from_frame,
    execute_bounded_unary, execute_bounded_unary_from_frame,
};
use operator::{
    execute_bounded_gpu_resident_source, execute_bounded_literal_batch_source, execute_bounded_local_parquet_source,
    execute_bounded_union,
};
pub(in crate::exec::bounded) use partition_row_policy::BoundedPartitionRowPolicy;
pub(in crate::exec::bounded) use partition_target_policy::BoundedPartitionTargetPolicy;
use pipeline::try_execute_bounded_pipeline;
pub(in crate::exec) use resident::{CommittedPartial, ResidentGpuFrame};
pub(in crate::exec::bounded) use source_batching_policy::BoundedSourceBatchingPolicy;

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    admission: Option<&BoundedPipelineAdmission>,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    context.check_cancellation("query_engine.exec.bounded.execute_query_plan")?;
    if let Some(output) =
        try_execute_bounded_pipeline(context, plan, admission, gpu_inputs, metrics, observation_scope)?
    {
        return Ok(output);
    }
    let unreachable = plan.unreachable_node_ids();
    if plan
        .nodes()
        .iter()
        .any(|node| !unreachable.contains(&node.id) && matches!(node.kind, PlanNode::VectorAlgorithm(_)))
    {
        return Err(unsupported_bounded_shape(
            "vector execution requires an admitted compiled pipeline",
            "compiled_vector_pipeline_required",
        ));
    }

    if plan.nodes().iter().any(|node| !unreachable.contains(&node.id) && matches!(node.kind, PlanNode::MlAlgorithm(_)))
    {
        return Err(unsupported_bounded_shape(
            "ML execution requires an admitted compiled pipeline",
            "compiled_ml_pipeline_required",
        ));
    }

    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    if !plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Aggregate(_))) {
        return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics);
    }

    let bounded = match BoundedPlan::analyze(plan) {
        Ok(bounded) => bounded,
        Err(_error) => {
            return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics);
        },
    };
    let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics)?;
    let frame = execute_bounded_aggregate_to_frame(&mut execution, &bounded)?;
    let frame = execute_post_aggregate_stages(&mut execution, &bounded, frame)?;
    let frame = execute_final_sort_limit_with_retry(&mut execution, &bounded, frame)?;
    let frame = execute_post_sort_projections(&mut execution, &bounded, frame)?;
    let frame = execute_standalone_limit(&mut execution, &bounded, frame)?;

    record_device_memory_observation(&frame.frame, execution.metrics())?;
    match plan.sink() {
        ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => {
            let output_bytes = frame.frame.device_allocated_size_bytes()?;
            let resident_owner = execution.context().retain_resident_output_owner(output_bytes)?;
            let output_frame = frame.frame.with_producer_execution_scope(plan.execution_fingerprint());
            let handle = match GpuResultHandle::new_with_resident_owner(
                execution.context(),
                output_frame,
                Arc::clone(plan.output_schema()),
                resident_owner,
            ) {
                Ok(handle) => handle.with_producer_execution_plan(plan.clone()),
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

/// Decides the executable plan once and hands the admitted artifact to the
/// caller. Execution runs this same artifact instead of re-deriving it.
///
/// The decision runs on the plan as given. When it rejects, the
/// caller-visible sink is attached as a diagnostic fact only; it never
/// selects a second decision.
pub(super) fn preflight_bounded_join_plan(
    plan: &QueryPlan,
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    policy: PlanningPolicy,
) -> Result<Option<BoundedPipelineAdmission>> {
    if plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Join(_))) {
        let mut error = unsupported_bounded_shape(
            "bounded native join execution requires an executable compiled pipeline",
            "bounded_join_plan_shape_unsupported",
        );
        let ctx = rapids_analysis_context().with_policy(policy);
        let decision = match bounded_pipeline_decision(plan, statistical_aggregate_execution_mode, ctx) {
            Ok(decision) => decision,
            Err(projection_error) => {
                let mut facts = bounded_join_unsupported_shape_facts_with_projection(plan, None, ctx);
                insert_bounded_join_rejection_projection_error_facts(&mut facts, &projection_error);
                for (key, value) in facts {
                    error = error.with_fact(key, value);
                }
                return Err(error.with_source(projection_error));
            },
        };
        let Some(rejection_projection) = decision.rejection_projection() else {
            return Ok(decision.admission().cloned());
        };
        let facts = bounded_join_unsupported_shape_facts_with_projection(plan, Some(rejection_projection), ctx);
        for (key, value) in facts {
            error = error.with_fact(key, value);
        }
        error = error.with_fact("caller_visible_sink", caller_visible_sink_kind(plan.sink()));
        return Err(error);
    }

    Ok(None)
}

fn caller_visible_sink_kind(sink: &ResultSink) -> &'static str {
    match sink {
        ResultSink::HostArrow { .. } | ResultSink::DeferredHostArrow { .. } => "host_arrow",
        ResultSink::GpuResident { .. } => "gpu_resident",
    }
}

fn reject_bounded_join_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<()> {
    preflight_bounded_join_plan(plan, context.statistical_aggregate_execution_mode(), context.planning_policy())
        .map(|_| ())
}
