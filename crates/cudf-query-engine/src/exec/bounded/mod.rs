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
pub(in crate::exec) use reservation::{ActiveReservation, CommittedPartial, ReservationTracker, ResidentGpuFrame};

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
        if plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline() {
            return execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan(
                context, plan, gpu_inputs, metrics,
            );
        }
        if plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline() {
            return execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan(
                context, plan, gpu_inputs, metrics,
            );
        }
        if plan.has_bounded_executable_multi_join_pipeline() {
            return execute_bounded_multi_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline() {
            return execute_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_query_plan(
                context, plan, gpu_inputs, metrics,
            );
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

    let mut reservations = ReservationTracker::new(context.device_memory_space(), device_budget_bytes);
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
        if plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline() {
            return execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan(
                context, plan, gpu_inputs, metrics,
            );
        }
        if plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline() {
            return execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan(
                context, plan, gpu_inputs, metrics,
            );
        }
        if plan.has_bounded_executable_multi_join_pipeline() {
            return execute_bounded_multi_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline() {
            return execute_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_query_plan(
                context, plan, gpu_inputs, metrics,
            );
        }
        reject_bounded_join_plan(context, plan)?;
    }

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(context.device_memory_space(), device_budget_bytes);
    let mut scratch = ExecutionScratch::new(context);
    let frame = execute_bounded_generic_node(context, plan, plan.sink().input(), gpu_inputs, &mut scratch, metrics)?;
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
    scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if let Some(frame) = scratch.node_results.get(&id) {
        return Ok(frame.clone());
    }

    let node = plan
        .node(id)
        .ok_or_else(|| invalid_node_reference(id, "bounded generic query plan references a missing node"))?;
    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => read_local_parquet_source(context, source, metrics),
            crate::source::SourceNode::GpuResident(source) => {
                read_gpu_resident_source(context, source, gpu_inputs, metrics)
            },
        },
        PlanNode::Filter(filter) => {
            let input = execute_bounded_generic_node(context, plan, filter.input, gpu_inputs, scratch, metrics)?;
            let token = bounded_generic_budget_token(
                context,
                id,
                "bounded_generic_filter",
                "cudf_query_engine.bounded_generic.filter_budget_token",
                "failed to create bounded generic filter budget token",
            )?;
            execute_filter_with_budget(context, filter, input, &token, metrics)
        },
        PlanNode::Projection(projection) => {
            let input = execute_bounded_generic_node(context, plan, projection.input, gpu_inputs, scratch, metrics)?;
            let token = bounded_generic_budget_token(
                context,
                id,
                "bounded_generic_projection",
                "cudf_query_engine.bounded_generic.projection_budget_token",
                "failed to create bounded generic projection budget token",
            )?;
            execute_projection_with_budget(context, projection, input, &token, metrics)
        },
        PlanNode::Sort(sort) => {
            let input = execute_bounded_generic_node(context, plan, sort.input, gpu_inputs, scratch, metrics)?;
            execute_sort(context, sort, input, metrics)
        },
        PlanNode::Limit(limit) => {
            let input = execute_bounded_generic_node(context, plan, limit.input, gpu_inputs, scratch, metrics)?;
            execute_limit(context, limit, input, metrics)
        },
        PlanNode::Aggregate(_) | PlanNode::Join(_) => Err(unsupported_bounded_shape(
            "bounded generic execution supports only source, filter, projection, sort, and limit nodes",
            "bounded_generic_plan_shape_unsupported",
        )
        .with_fact("node_id", id)
        .with_fact("node_kind", bounded_generic_node_kind(&node.kind))),
    }?;

    record_device_memory_observation(&frame, metrics)?;
    scratch.node_results.insert(id, frame.clone());
    Ok(frame)
}

fn bounded_generic_budget_token(
    context: &ExecutionContext,
    node_id: NodeId,
    budget_boundary: &'static str,
    operation: &'static str,
    message: &'static str,
) -> Result<ExecutionBudgetToken> {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("bounded generic budget tokens are created only for bounded policies");
    };
    execution_budget_token_for_boundary(context, device_budget_bytes, budget_boundary, operation, message)
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))
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
