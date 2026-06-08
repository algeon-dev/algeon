use super::{source::record_bounded_parquet_source_metrics, *};

// Milestone E ownership boundary:
// - adaptive: adaptive-only admission/report/execution facade.
// - strict: strict bounded admission/report/execution facade.
// - compiled: shared compiled pipeline runtime and graph-support checks.
// - remaining modules: bounded runtime/operator support shared by the current
//   bounded executors, but not owners of mode-specific admission/report policy.
mod adaptive;
mod admission;
mod aggregate;
pub(in crate::exec) mod chunks;
mod compiled;
mod facts;
mod final_sort;
mod generic;
mod join;
mod reservation;
pub(in crate::exec) mod strict;

use adaptive::try_execute_adaptive_bounded_pipeline;
pub use adaptive::{
    AdaptivePipelineAdmission, adaptive_bounded_pipeline_admission,
    adaptive_bounded_pipeline_admission_with_memory_history,
};
use aggregate::*;
pub use aggregate::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
pub(in crate::exec) use final_sort::topk_candidate_rows;
use final_sort::*;
use generic::{
    execute_bounded_generic_gpu_resident_source, execute_bounded_generic_literal_batch_source,
    execute_bounded_generic_local_parquet_source, execute_bounded_generic_query_plan, execute_bounded_generic_unary,
    execute_bounded_generic_unary_consuming_input_with_reservation_kind, execute_bounded_generic_unary_from_frame,
    execute_bounded_generic_unary_from_frame_with_reservation_kind, execute_bounded_generic_union,
    release_reservations_reverse,
};
pub(in crate::exec) use join::JoinMaterializationTxn;
pub(in crate::exec::bounded) use join::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedJoinLeafLocalParquetSourceStream, execute_bounded_compiled_join_build_from_frame,
    execute_bounded_compiled_join_build_from_resident_frame, execute_bounded_compiled_join_materialize,
    execute_bounded_compiled_join_materialize_preserving_stream, execute_bounded_compiled_join_probe_from_frame,
    execute_bounded_compiled_join_probe_from_frame_stream, execute_bounded_compiled_join_probe_from_resident_frame,
    execute_bounded_compiled_join_probe_from_stream, execute_bounded_join_leaf_source,
    materialize_bounded_join_leaf_local_parquet_source_stream, plan_bounded_join_leaf_local_parquet_source_stream,
    release_bounded_compiled_join_build_state, release_bounded_compiled_join_probe_state,
};
pub use join::{
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, STRING_OFFSET_BYTES_CONSERVATIVE,
    join_output_upper_bound_bytes,
};
pub(in crate::exec) use reservation::{ActiveReservation, CommittedPartial, ReservationTracker, ResidentGpuFrame};
use strict::try_execute_strict_bounded_pipeline;

use crate::mode::NativePipelineExecutionMode;

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    context.check_cancellation("nexus_query_engine.exec.bounded.execute_query_plan")?;
    match bounded_pipeline_execution_mode(context) {
        NativePipelineExecutionMode::StrictBounded => {
            if let Some(output) = try_execute_strict_bounded_pipeline(context, plan, gpu_inputs, metrics)? {
                return Ok(output);
            }
        },
        NativePipelineExecutionMode::Adaptive => {
            if let Some(output) = try_execute_adaptive_bounded_pipeline(context, plan, gpu_inputs, metrics)? {
                return Ok(output);
            }
        },
    }

    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    if !plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Aggregate(_))) {
        return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics);
    }

    let bounded = match BoundedPlan::analyze(plan) {
        Ok(bounded) => bounded,
        Err(_error) => return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics),
    };
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let frame = execute_bounded_aggregate_to_frame(context, &bounded, &mut reservations, metrics)?;
    let frame = execute_post_aggregate_stages(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_final_sort_limit_with_retry(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_post_sort_projections(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_standalone_limit(context, &bounded, frame, &mut reservations, metrics)?;

    record_device_memory_observation(&frame.frame, metrics)?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, frame.frame, Arc::clone(plan.output_schema())) {
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
        let contract = BoundedJoinReservationContract;
        let mut error = unsupported_bounded_shape(
            bounded_join_unsupported_message(contract),
            "bounded_join_plan_shape_unsupported",
        );
        let facts = match strict::strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
            plan,
            context.statistical_aggregate_execution_mode(),
        ) {
            Ok(Some(projection)) => {
                crate::bounded::bounded_join_unsupported_shape_facts(plan, context.boundary_policy(), Some(&projection))
            },
            Ok(None) => contract.unsupported_shape_facts(context.boundary_policy()),
            Err(_) => crate::bounded::bounded_join_unsupported_shape_facts(plan, context.boundary_policy(), None),
        };
        for (key, value) in facts {
            error = error.with_fact(key, value);
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
