use super::*;
use crate::bounded_support::{
    BoundedJoinPipelineRejection, BoundedJoinPipelineStage, analyze_bounded_join_pipeline,
    analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
};

mod keyless;
mod stages;

use keyless::{
    execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan_inner,
    execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan_inner,
};
pub(super) use stages::{execute_bounded_join_pipeline_stage, validate_bounded_join_pipeline_stages};

const KEYLESS_CONDITIONAL_INPUT_SUBPLAN_EXECUTION_FAMILY_FACT: &str = "bounded_join_execution_family";

fn attach_bounded_join_execution_family(
    error: QueryError,
    family: crate::runtime::BoundedJoinExecutionFamily,
) -> QueryError {
    if error.fact_value(KEYLESS_CONDITIONAL_INPUT_SUBPLAN_EXECUTION_FAMILY_FACT).is_some() {
        return error;
    }

    error.with_fact(KEYLESS_CONDITIONAL_INPUT_SUBPLAN_EXECUTION_FAMILY_FACT, family.as_str())
}

fn attach_keyless_conditional_join_input_subplan_execution_family(error: QueryError) -> QueryError {
    attach_bounded_join_execution_family(
        error,
        crate::runtime::BoundedJoinExecutionFamily::KeylessConditionalJoinInputSubplanPipeline,
    )
}

fn attach_keyless_conditional_join_aggregate_over_join_subplan_inputs_execution_family(
    error: QueryError,
) -> QueryError {
    attach_bounded_join_execution_family(
        error,
        crate::runtime::BoundedJoinExecutionFamily::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    )
}

fn attach_mixed_keyed_residual_join_resident_grouped_aggregate_input_execution_family(error: QueryError) -> QueryError {
    attach_bounded_join_execution_family(
        error,
        crate::runtime::BoundedJoinExecutionFamily::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
    )
}

pub(super) fn execute_bounded_join_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = BoundedJoinPipelinePlan::analyze(plan)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;
    validate_bounded_join_leaf_input_capabilities(plan, pipeline.join.inputs[0])?;
    validate_bounded_join_leaf_input_capabilities(plan, pipeline.join.inputs[1])?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_join_pipeline_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(context.device_memory_space(), device_budget_bytes);
    let left = execute_bounded_join_leaf_input(
        context,
        plan,
        pipeline.join.inputs[0],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let right = execute_bounded_join_leaf_input(
        context,
        plan,
        pipeline.join.inputs[1],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let mut committed =
        execute_bounded_root_join(context, pipeline.join_id, pipeline.join, left, right, &mut reservations, metrics)?;

    for stage in &pipeline.stages {
        committed = execute_bounded_join_pipeline_stage(context, stage, committed, &mut reservations, metrics)?;
    }

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

pub(super) fn execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan_inner(context, plan, gpu_inputs, metrics)
        .map_err(attach_keyless_conditional_join_input_subplan_execution_family)
}

pub(super) fn execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan_inner(
        context, plan, gpu_inputs, metrics,
    )
    .map_err(attach_keyless_conditional_join_aggregate_over_join_subplan_inputs_execution_family)
}

pub(super) fn execute_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    execute_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_query_plan_inner(
        context, plan, gpu_inputs, metrics,
    )
    .map_err(attach_mixed_keyed_residual_join_resident_grouped_aggregate_input_execution_family)
}

fn execute_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_query_plan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(plan)
        .map_err(bounded_join_pipeline_rejection_error)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;
    validate_bounded_multi_join_subtree_capabilities(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.join_subtree_side)],
    )?;
    let grouped_aggregate_input = BoundedPlan::analyze_projection_wrapped_local_aggregate_subplan(
        plan,
        pipeline.join.inputs[join_side_index(pipeline.grouped_aggregate_side)],
    )?;
    grouped_aggregate_input.validate_capabilities()?;
    validate_executable_join_subset(pipeline.join)?;
    let residual_columns = residual_input_columns(pipeline.join)?;
    lower_join_residual_predicate(context, pipeline.join, &residual_columns)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!(
            "execute_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_query_plan is called only for bounded policies"
        );
    };
    let mut reservations = ReservationTracker::new(context.device_memory_space(), device_budget_bytes);
    let join_subtree = execute_bounded_multi_join_subtree(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.join_subtree_side)],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let grouped_aggregate = execute_bounded_projection_wrapped_local_aggregate_input(
        context,
        &grouped_aggregate_input,
        &mut reservations,
        metrics,
    )?;
    let (left, right) = match pipeline.join_subtree_side {
        JoinSide::Left => (join_subtree, grouped_aggregate),
        JoinSide::Right => (grouped_aggregate, join_subtree),
    };
    let mut committed =
        execute_bounded_root_join(context, pipeline.join_id, pipeline.join, left, right, &mut reservations, metrics)?;

    for stage in &pipeline.stages {
        committed = execute_bounded_join_pipeline_stage(context, stage, committed, &mut reservations, metrics)?;
    }

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

struct BoundedJoinPipelinePlan<'a> {
    join_id: NodeId,
    join: &'a JoinNode,
    stages: Vec<BoundedJoinPipelineStage<'a>>,
}

impl<'a> BoundedJoinPipelinePlan<'a> {
    fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let shape = analyze_bounded_join_pipeline(plan).map_err(bounded_join_pipeline_rejection_error)?;
        Ok(Self { join_id: shape.join_id, join: shape.join, stages: shape.stages })
    }
}

fn execute_bounded_projection_wrapped_local_aggregate_input(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_local_parquet_source(bounded.source)?;
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
            "bounded grouped aggregate join input requires at least one planned source chunk",
        )
        .with_fact("source_node_id", bounded.source_id));
    }
    record_bounded_parquet_source_metrics(bounded.source, &chunk_plan, metrics);

    let partials =
        execute_bounded_chunks_with_source_scheduler(context, bounded, &chunk_plan.chunks, reservations, metrics)?;

    let frame = execute_final_aggregate_merge(context, bounded, partials, reservations, metrics)?;
    execute_post_aggregate_projection(context, bounded, frame, reservations, metrics)
}

fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

pub(super) fn bounded_join_pipeline_rejection_error(rejection: BoundedJoinPipelineRejection<'_>) -> QueryError {
    match rejection {
        BoundedJoinPipelineRejection::MissingNode { node_id, message } => invalid_node_reference(node_id, message),
        BoundedJoinPipelineRejection::UnsupportedJoin { join } => unsupported_bounded_shape(
            bounded_join_pipeline_supported_families_message(),
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", join.spec.join_type)
        .with_fact("build_side", join.spec.build_side)
        .with_fact("key_count", join.spec.keys.len())
        .with_fact("residual_predicate", join.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", join.spec.output_projection.len()),
        BoundedJoinPipelineRejection::UnsupportedShapeNode { node_id, node_kind } => unsupported_bounded_shape(
            "bounded multi-join pipeline execution supports only source/filter/projection leaf subtrees below joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", node_kind),
        BoundedJoinPipelineRejection::LimitNotDirectlyAboveSort => unsupported_bounded_shape(
            "bounded join pipeline execution does not support limit without a preceding global sort",
            "limit_not_directly_above_sort",
        ),
        BoundedJoinPipelineRejection::MissingRootJoin => {
            unsupported_bounded_shape(bounded_join_pipeline_supported_families_message(), "missing_root_join")
        },
        BoundedJoinPipelineRejection::MultipleReachableJoins => unsupported_bounded_shape(
            "bounded join pipeline execution requires exactly one reachable native join",
            "multiple_reachable_joins",
        ),
        BoundedJoinPipelineRejection::SingleReachableJoin => unsupported_bounded_shape(
            "bounded multi-join pipeline execution requires more than one reachable native join",
            "single_reachable_join",
        ),
    }
}

fn bounded_join_pipeline_supported_families_message() -> String {
    format!(
        "bounded join pipeline execution requires one of these ordered-reservation families below supported unary operators: {}",
        BoundedJoinReservationContract.execution_families_csv()
    )
}
