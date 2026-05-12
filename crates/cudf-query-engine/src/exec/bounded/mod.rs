use super::{source::record_bounded_parquet_source_metrics, *};

// Phase E ownership boundary:
// - adaptive: adaptive-only admission/report/execution facade.
// - strict: strict bounded admission/report/execution facade.
// - compiled: shared compiled pipeline runtime and graph-support checks.
// - remaining modules: bounded runtime/operator support shared by the current
//   bounded executors, but not owners of mode-specific admission/report policy.
mod adaptive;
mod admission;
mod aggregate;
mod chunks;
mod compiled;
mod facts;
mod final_sort;
mod join;
mod reservation;
pub(in crate::exec) mod strict;

use adaptive::try_execute_adaptive_bounded_pipeline;
pub use adaptive::{
    AdaptivePipelineAdmission, adaptive_bounded_pipeline_admission,
    adaptive_bounded_pipeline_admission_with_memory_history,
};
use aggregate::*;
use chunks::*;
pub(in crate::exec) use facts::RuntimeResourceFacts;
use facts::*;
pub(in crate::exec) use final_sort::topk_candidate_rows;
use final_sort::*;
pub(in crate::exec::bounded) use join::{
    BoundedCompiledJoinBuildState, BoundedCompiledJoinProbeState, execute_bounded_compiled_join_build_from_frame,
    execute_bounded_compiled_join_materialize, execute_bounded_compiled_join_probe_from_frame,
    execute_bounded_join_leaf_source,
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

    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let partials = execute_bounded_chunks_with_source_scheduler(
        context,
        &bounded,
        &chunk_plan.chunks,
        &mut reservations,
        metrics,
    )?;

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
        let facts = plan
            .bounded_join_unsupported_shape_facts(context.boundary_policy())
            .unwrap_or_else(|| contract.unsupported_shape_facts(context.boundary_policy()));
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

fn execute_bounded_generic_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let committed =
        execute_bounded_generic_node(context, plan, plan.sink().input(), gpu_inputs, &mut reservations, metrics)?;

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

fn execute_bounded_generic_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let node = plan
        .node(id)
        .ok_or_else(|| invalid_node_reference(id, "bounded generic query plan references a missing node"))?;
    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => {
                execute_bounded_generic_local_parquet_source(context, source, id, reservations, metrics)
            },
            crate::source::SourceNode::GpuResident(source) => {
                execute_bounded_generic_gpu_resident_source(context, source, id, gpu_inputs, reservations, metrics)
            },
        },
        PlanNode::Filter(filter) => {
            let input = execute_bounded_generic_node(context, plan, filter.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_filter",
                "bounded_generic_filter_output",
                input,
                reservations,
                metrics,
                true,
                |context, input, token, metrics| {
                    execute_filter_with_budget(
                        context,
                        filter,
                        input,
                        token.expect("budget token is created for bounded generic filter stages"),
                        metrics,
                    )
                },
            )
        },
        PlanNode::Projection(projection) => {
            let input =
                execute_bounded_generic_node(context, plan, projection.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_projection",
                "bounded_generic_projection_output",
                input,
                reservations,
                metrics,
                true,
                |context, input, token, metrics| {
                    execute_projection_with_budget(
                        context,
                        projection,
                        input,
                        token.expect("budget token is created for bounded generic projection stages"),
                        metrics,
                    )
                },
            )
        },
        PlanNode::Sort(sort) => {
            let input = execute_bounded_generic_node(context, plan, sort.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_sort",
                "bounded_generic_sort_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| execute_sort(context, sort, input, metrics),
            )
        },
        PlanNode::Limit(limit) => {
            let input = execute_bounded_generic_node(context, plan, limit.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_limit",
                "bounded_generic_limit_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| execute_limit(context, limit, input, metrics),
            )
        },
        PlanNode::Aggregate(_) | PlanNode::Join(_) => Err(unsupported_bounded_shape(
            "bounded generic execution supports only source, filter, projection, sort, and limit nodes",
            "bounded_generic_plan_shape_unsupported",
        )
        .with_fact("node_id", id)
        .with_fact("node_kind", bounded_generic_node_kind(&node.kind))),
    }?;

    Ok(frame)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source)?;
    let chunk_plan = plan_local_parquet_source_chunks(
        node_id,
        source,
        SourceChunkPlanningPolicy::from(context.memory_control_policy()),
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.generic.source",
            "bounded generic local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);

    let chunks = read_local_parquet_chunks_ordered(
        context,
        source,
        &chunk_plan.chunks,
        reservations,
        metrics,
        SourceReadSchedulerBoundaries::new("bounded_generic_source_read", "bounded_generic_source_read_prefetch"),
    )?;

    concatenate_source_read_outputs(
        context,
        node_id,
        chunks,
        reservations,
        metrics,
        "bounded_generic_source_read",
        "cudf_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_gpu_resident_source(
    context: &ExecutionContext,
    source: &GpuResidentSource,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let frame = read_gpu_resident_source(context, source, gpu_inputs, metrics)?;
    let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
            .with_memory_boundary("bounded_generic_gpu_resident_source")
            .with_node_id(node_id)
            .with_stage("bounded_generic_gpu_resident_source")
            .with_attempt(1, 1),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}

#[allow(clippy::too_many_arguments)]
fn execute_bounded_generic_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let result = execute_bounded_generic_unary_from_frame(
        context,
        node_id,
        stage,
        output_boundary,
        input.frame,
        input_reserved_bytes,
        reservations,
        metrics,
        use_budget_token,
        execute,
    );
    reservations.release(input.reservation);
    result
}

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_from_frame_with_reservation_kind(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        input_reserved_bytes,
        ReservationKind::OperatorOutput,
        reservations,
        metrics,
        use_budget_token,
        execute,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame_with_reservation_kind(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservation_kind: ReservationKind,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let work_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    )?;
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&work_reservation),
            stage,
            "cudf_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                reservations.release(work_reservation);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let result = execute(context, input, token.as_ref(), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(work_reservation);
            return Err(error);
        },
    };
    reservations.release(work_reservation);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

fn bounded_generic_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
    }
}
