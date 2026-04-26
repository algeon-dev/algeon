use super::{source::record_bounded_parquet_source_metrics, *};

// Milestone E ownership boundary:
// - adaptive: adaptive-only execution facade.
// - strict: strict bounded execution facade.
// - compiled: shared compiled pipeline runtime.
// - remaining modules: bounded runtime/operator support shared by the current
//   bounded executors, but not owners of mode-specific admission/report policy.
mod adaptive;
mod aggregate;
pub(in crate::exec) mod chunks;
mod compiled;
mod facts;
mod final_sort;
mod generic;
mod join;
mod partition_row_policy;
mod partition_target_policy;
mod reservation;
mod source_batching_policy;
pub(in crate::exec) mod strict;

use adaptive::try_execute_adaptive_bounded_pipeline;
use aggregate::*;
pub use aggregate::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};
pub(in crate::exec::bounded) use aggregate::{
    SourceAdjacentAggregateChunkCapHints, apply_source_adjacent_aggregate_chunk_caps,
};
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
pub(in crate::exec) use final_sort::topk_candidate_rows;
use final_sort::*;
use generic::{
    execute_bounded_generic_gpu_resident_source, execute_bounded_generic_literal_batch_source,
    execute_bounded_generic_local_parquet_source, execute_bounded_generic_query_plan,
    execute_bounded_generic_unary_consuming_input_with_reservation_kind, execute_bounded_generic_union,
    release_reservations_reverse,
};
pub(in crate::exec::bounded) use generic::{
    execute_bounded_generic_projection, execute_bounded_generic_projection_from_frame, execute_bounded_generic_unary,
    execute_bounded_generic_unary_from_frame, execute_bounded_generic_unary_from_frame_with_reservation_kind,
};
pub(in crate::exec) use join::JoinMaterializationTxn;
pub(in crate::exec::bounded) use join::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedJoinLeafLocalParquetSourcePrefetch, BoundedJoinLeafLocalParquetSourceStream,
    execute_bounded_compiled_join_build_from_frame, execute_bounded_compiled_join_build_from_resident_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_materialize_preserving_stream,
    execute_bounded_compiled_join_probe_from_frame, execute_bounded_compiled_join_probe_from_frame_stream,
    execute_bounded_compiled_join_probe_from_resident_frame, execute_bounded_compiled_join_probe_from_stream,
    execute_bounded_join_leaf_source, materialize_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_aggregate_input_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    plan_bounded_shared_join_aggregate_local_parquet_source_stream, release_bounded_compiled_join_build_state,
    release_bounded_compiled_join_probe_state, reusable_hash_join_stream_target_output_bytes,
    stream_bounded_join_leaf_local_parquet_source_chunks, try_start_bounded_join_leaf_local_parquet_source_prefetch,
};
pub use join::{
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, STRING_OFFSET_BYTES_CONSERVATIVE,
    join_output_upper_bound_bytes,
};
pub(in crate::exec::bounded) use partition_row_policy::BoundedPartitionRowPolicy;
pub(in crate::exec::bounded) use partition_target_policy::BoundedPartitionTargetPolicy;
pub(in crate::exec::bounded) use reservation::reconcile_work_reservation_to_output;
pub(in crate::exec) use reservation::{ActiveReservation, CommittedPartial, ReservationTracker, ResidentGpuFrame};
pub(in crate::exec::bounded) use source_batching_policy::BoundedSourceBatchingPolicy;
use strict::try_execute_strict_bounded_pipeline;

use crate::mode::NativePipelineExecutionMode;

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    context.check_cancellation("nexus_query_engine.exec.bounded.execute_query_plan")?;
    match bounded_pipeline_execution_mode(context) {
        NativePipelineExecutionMode::StrictBounded => {
            if let Some(output) =
                try_execute_strict_bounded_pipeline(context, plan, gpu_inputs, metrics, observation_scope)?
            {
                return Ok(output);
            }
        },
        NativePipelineExecutionMode::Adaptive => {
            if let Some(output) =
                try_execute_adaptive_bounded_pipeline(context, plan, gpu_inputs, metrics, observation_scope)?
            {
                return Ok(output);
            }
        },
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
    let mut reservations =
        ReservationTracker::new_for_context(context, device_budget_bytes, metrics, observation_scope)?;
    let frame = execute_bounded_aggregate_to_frame(context, &bounded, &mut reservations, metrics)?;
    let frame = execute_post_aggregate_stages(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_final_sort_limit_with_retry(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_post_sort_projections(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_standalone_limit(context, &bounded, frame, &mut reservations, metrics)?;

    record_device_memory_observation(&frame.frame, metrics)?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let live_reservation = frame.reservation.retain_grant_live_reservation();
            let handle = match GpuResultHandle::new_with_live_reservation(
                context,
                frame.frame,
                Arc::clone(plan.output_schema()),
                live_reservation,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(frame.reservation, metrics);
                    return Err(error);
                },
            };
            reservations.release(frame.reservation, metrics);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                frame,
                plan.sink().input(),
                plan.node(plan.sink().input()).expect("validated sink input exists").output_schema(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn bounded_pipeline_execution_mode(context: &ExecutionContext) -> NativePipelineExecutionMode {
    context.native_pipeline_execution_mode()
}

fn reject_bounded_join_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<()> {
    if plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Join(_))) {
        // A frontend-owned incremental D2H cursor executes through a physical
        // GPU-resident sink, but runtime rejection facts remain a contract with
        // that frontend. Project them from the retained caller-visible host
        // sink so the explicit materialization reservation is not erased.
        let rejection_contract_plan = plan.clone_with_caller_visible_sink_contract();
        let plan = &rejection_contract_plan;
        let contract = BoundedJoinReservationContract;
        let mut error = unsupported_bounded_shape(
            bounded_join_unsupported_message(contract),
            "bounded_join_plan_shape_unsupported",
        );
        let mut projection_error_source = None;
        let facts = match crate::admission::strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
            plan,
            context.statistical_aggregate_execution_mode(),
        ) {
            Ok(Some(projection)) => crate::admission::bounded_join_unsupported_shape_facts_with_projection(
                plan,
                context.boundary_policy(),
                Some(&projection),
            ),
            Ok(None) => contract.unsupported_shape_facts(context.boundary_policy()),
            Err(projection_error) => {
                let mut facts = crate::admission::bounded_join_unsupported_shape_facts_with_projection(
                    plan,
                    context.boundary_policy(),
                    None,
                );
                crate::admission::insert_bounded_join_rejection_projection_error_facts(&mut facts, &projection_error);
                projection_error_source = Some(projection_error);
                facts
            },
        };
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

fn bounded_join_unsupported_message(contract: BoundedJoinReservationContract) -> String {
    format!(
        "bounded native join execution requires strict bounded capability admission: {}",
        contract.supported_shape_requirement()
    )
}
