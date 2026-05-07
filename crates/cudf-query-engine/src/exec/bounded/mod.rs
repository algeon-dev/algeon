use super::{source::record_bounded_parquet_source_metrics, *};

mod aggregate;
mod chunks;
mod facts;
mod final_sort;
mod join;
mod multi_join;
mod pipeline;
mod reservation;

use aggregate::*;
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
pub(in crate::exec) use final_sort::topk_candidate_rows;
use final_sort::*;
use join::*;
use multi_join::*;
use pipeline::*;
pub(in crate::exec) use reservation::{CommittedPartial, ReservationTracker};

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    if plan.contains_join() {
        if plan.has_bounded_executable_root_join() {
            return execute_bounded_root_join_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_join_pipeline() {
            return execute_bounded_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_multi_join_pipeline() {
            return execute_bounded_multi_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        reject_bounded_join_plan(context, plan)?;
    }

    if !plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Aggregate(_))) {
        return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics);
    }

    let bounded = BoundedPlan::analyze(plan)?;
    validate_local_parquet_source(bounded.source)?;
    bounded.validate_capabilities()?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let chunk_plan = plan_local_parquet_source_chunks(
        bounded.source_id,
        bounded.source,
        SourceChunkPlanningPolicy::from(context.memory_control_policy()),
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.plan",
            "bounded execution requires at least one planned source chunk",
        )
        .with_fact("source_node_id", bounded.source_id));
    }
    record_bounded_parquet_source_metrics(bounded.source, &chunk_plan, metrics);

    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut partials = Vec::with_capacity(chunk_plan.chunks.len());
    for chunk in &chunk_plan.chunks {
        partials.extend(execute_bounded_chunk_with_retry(context, &bounded, chunk, &mut reservations, metrics)?);
    }

    let frame = execute_final_aggregate_merge(context, &bounded, partials, &mut reservations, metrics)?;
    let frame = execute_post_aggregate_projection(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_final_sort_limit_with_retry(context, &bounded, frame, &mut reservations, metrics)?;

    record_device_memory_observation(&frame.frame, metrics)?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, frame.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(frame.reservation);
                    return Err(error);
                },
            };
            reservations.release(frame.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output =
                materialize_bounded_to_host(context, frame, plan.sink().input(), schema, &mut reservations, metrics)?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn reject_bounded_join_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<()> {
    if plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Join(_))) {
        let contract = BoundedJoinReservationContract;
        let mut error = unsupported_bounded_shape(
            bounded_join_unsupported_message(contract),
            "bounded_join_plan_shape_unsupported",
        );
        for (key, value) in contract.rejection_facts(context.boundary_policy()) {
            error = error.with_fact(key, value);
        }
        return Err(error);
    }

    Ok(())
}

fn bounded_join_unsupported_message(contract: BoundedJoinReservationContract) -> String {
    format!(
        "bounded native join execution currently supports these ordered-reservation families: {}",
        contract.execution_families_csv()
    )
}

fn execute_bounded_generic_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    if plan.contains_join() {
        if plan.has_bounded_executable_root_join() {
            return execute_bounded_root_join_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_join_pipeline() {
            return execute_bounded_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_multi_join_pipeline() {
            return execute_bounded_multi_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        reject_bounded_join_plan(context, plan)?;
    }

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut scratch = ExecutionScratch::new(context);
    let frame = execute_node(context, plan, plan.sink().input(), gpu_inputs, &mut scratch, metrics)?;
    let frame_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, frame_bytes, reservations.budget())
            .with_memory_boundary("operator_output")
            .with_node_id(plan.sink().input())
            .with_stage("bounded_generic")
            .with_attempt(1, 1),
        metrics,
    )?;
    let committed = CommittedPartial { frame, reservation };

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, committed.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}
