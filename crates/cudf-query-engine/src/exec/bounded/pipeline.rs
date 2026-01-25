use super::*;
use crate::bounded::{
    BoundedJoinPipelineRejection, BoundedJoinPipelineStage,
    analyze_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline,
    analyze_bounded_join_pipeline, analyze_bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
    analyze_bounded_multi_join_aggregate_over_join_subplan_topk_pipeline,
    analyze_bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline,
    analyze_keyed_hash_join_resident_subplan_pipeline,
};

mod keyless;
mod stages;

use keyless::{
    AggregateOverJoinSubplanExecutionKind,
    execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan_inner,
    execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan_inner,
    execute_bounded_projection_wrapped_aggregate_over_join_subplan_input,
    validate_bounded_projection_wrapped_aggregate_over_join_subplan_input_capabilities,
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

fn attach_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_execution_family(
    error: QueryError,
) -> QueryError {
    attach_bounded_join_execution_family(
        error,
        crate::runtime::BoundedJoinExecutionFamily::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline,
    )
}

fn attach_multi_join_resident_grouped_aggregate_subquery_filter_execution_family(error: QueryError) -> QueryError {
    attach_bounded_join_execution_family(
        error,
        crate::runtime::BoundedJoinExecutionFamily::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline,
    )
}

fn attach_multi_join_aggregate_over_join_subplan_topk_execution_family(error: QueryError) -> QueryError {
    attach_bounded_join_execution_family(
        error,
        crate::runtime::BoundedJoinExecutionFamily::MultiJoinAggregateOverJoinSubplanTopKPipeline,
    )
}

fn attach_keyed_hash_join_resident_subplan_execution_family(error: QueryError) -> QueryError {
    attach_bounded_join_execution_family(
        error,
        crate::runtime::BoundedJoinExecutionFamily::KeyedHashJoinResidentSubplanPipeline,
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
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
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

pub(super) fn execute_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    execute_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline_query_plan_inner(
        context, plan, gpu_inputs, metrics,
    )
    .map_err(attach_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_execution_family)
}

pub(super) fn execute_bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    execute_bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_query_plan_inner(
        context, plan, gpu_inputs, metrics,
    )
    .map_err(attach_multi_join_aggregate_over_join_subplan_topk_execution_family)
}

pub(super) fn execute_bounded_keyed_hash_join_resident_subplan_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    execute_bounded_keyed_hash_join_resident_subplan_pipeline_query_plan_inner(context, plan, gpu_inputs, metrics)
        .map_err(attach_keyed_hash_join_resident_subplan_execution_family)
}

pub(super) fn execute_bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    execute_bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline_query_plan_inner(
        context, plan, gpu_inputs, metrics,
    )
    .map_err(attach_multi_join_resident_grouped_aggregate_subquery_filter_execution_family)
}

fn execute_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline_query_plan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = analyze_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(plan)
        .map_err(bounded_join_pipeline_rejection_error)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;
    let [
        post_join_stages @ ..,
        BoundedJoinPipelineStage::Sort { node_id: final_sort_id, node: final_sort },
        BoundedJoinPipelineStage::Limit { node: limit, .. },
    ] = pipeline.stages.as_slice()
    else {
        return Err(unsupported_bounded_shape(
            "bounded hash join resident grouped aggregate TopK pipeline requires final sort and limit stages",
            "bounded_join_plan_shape_unsupported",
        ));
    };

    validate_bounded_multi_join_subtree_capabilities(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.join_subtree_side)],
    )?;
    let grouped_aggregate_input = BoundedPlan::analyze_projection_wrapped_local_aggregate_subplan_with_post_stages(
        plan,
        pipeline.join.inputs[join_side_index(pipeline.grouped_aggregate_side)],
    )?;
    grouped_aggregate_input.0.validate_capabilities()?;
    validate_bounded_join_pipeline_stages(&grouped_aggregate_input.1)?;
    validate_executable_join_subset(pipeline.join)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!(
            "execute_bounded_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline_query_plan is called only for bounded policies"
        );
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let join_subtree = execute_bounded_multi_join_subtree(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.join_subtree_side)],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let mut grouped_aggregate = execute_bounded_projection_wrapped_local_aggregate_input(
        context,
        &grouped_aggregate_input.0,
        &mut reservations,
        metrics,
    )?;
    for stage in &grouped_aggregate_input.1 {
        grouped_aggregate =
            execute_bounded_join_pipeline_stage(context, stage, grouped_aggregate, &mut reservations, metrics)?;
    }
    let (left, right) = match pipeline.join_subtree_side {
        JoinSide::Left => (join_subtree, grouped_aggregate),
        JoinSide::Right => (grouped_aggregate, join_subtree),
    };
    let mut committed =
        execute_bounded_root_join(context, pipeline.join_id, pipeline.join, left, right, &mut reservations, metrics)?;

    for stage in post_join_stages {
        committed = execute_bounded_join_pipeline_stage(context, stage, committed, &mut reservations, metrics)?;
    }
    let committed = execute_final_sort_node_with_retry(
        context,
        *final_sort_id,
        final_sort,
        Some(limit),
        committed,
        &mut reservations,
        metrics,
    )?;

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

fn execute_bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline_query_plan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = analyze_bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(plan)
        .map_err(bounded_join_pipeline_rejection_error)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;
    let [BoundedJoinPipelineStage::Sort { node_id: final_sort_id, node: final_sort }] = pipeline.stages.as_slice()
    else {
        return Err(unsupported_bounded_shape(
            "bounded resident grouped aggregate subquery filter pipeline requires one final sort stage",
            "bounded_join_plan_shape_unsupported",
        ));
    };

    validate_bounded_multi_join_subtree_capabilities(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.output_side)],
    )?;
    validate_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter_capabilities(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.subquery_filter_side)],
    )?;
    validate_executable_join_subset(pipeline.join)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!(
            "execute_bounded_multi_join_resident_grouped_aggregate_subquery_filter_pipeline_query_plan is called only for bounded policies"
        );
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let output = execute_bounded_multi_join_subtree(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.output_side)],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let subquery_filter = execute_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.subquery_filter_side)],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let (left, right) = match pipeline.output_side {
        JoinSide::Left => (output, subquery_filter),
        JoinSide::Right => (subquery_filter, output),
    };
    let committed =
        execute_bounded_root_join(context, pipeline.join_id, pipeline.join, left, right, &mut reservations, metrics)?;
    let committed = execute_final_sort_node_with_retry(
        context,
        *final_sort_id,
        final_sort,
        None,
        committed,
        &mut reservations,
        metrics,
    )?;

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

fn execute_bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_query_plan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = analyze_bounded_multi_join_aggregate_over_join_subplan_topk_pipeline(plan)
        .map_err(bounded_join_pipeline_rejection_error)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;
    let [
        post_join_stages @ ..,
        BoundedJoinPipelineStage::Sort { node_id: final_sort_id, node: final_sort },
        BoundedJoinPipelineStage::Limit { node: limit, .. },
    ] = pipeline.stages.as_slice()
    else {
        return Err(unsupported_bounded_shape(
            "bounded multi-join aggregate-over-join TopK pipeline requires final sort and limit stages",
            "bounded_join_plan_shape_unsupported",
        ));
    };

    validate_bounded_multi_join_subtree_capabilities(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.multi_join_side)],
    )?;
    validate_bounded_projection_wrapped_aggregate_over_join_subplan_input_capabilities(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.aggregate_over_join_side)],
        AggregateOverJoinSubplanExecutionKind::Grouped,
    )?;
    validate_executable_join_subset(pipeline.join)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!(
            "execute_bounded_multi_join_aggregate_over_join_subplan_topk_pipeline_query_plan is called only for bounded policies"
        );
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let multi_join = execute_bounded_multi_join_subtree(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.multi_join_side)],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let aggregate_over_join = execute_bounded_projection_wrapped_aggregate_over_join_subplan_input(
        context,
        plan,
        pipeline.join.inputs[join_side_index(pipeline.aggregate_over_join_side)],
        AggregateOverJoinSubplanExecutionKind::Grouped,
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let (left, right) = match pipeline.multi_join_side {
        JoinSide::Left => (multi_join, aggregate_over_join),
        JoinSide::Right => (aggregate_over_join, multi_join),
    };
    let mut committed =
        execute_bounded_root_join(context, pipeline.join_id, pipeline.join, left, right, &mut reservations, metrics)?;

    for stage in post_join_stages {
        committed = execute_bounded_join_pipeline_stage(context, stage, committed, &mut reservations, metrics)?;
    }
    let committed = execute_final_sort_node_with_retry(
        context,
        *final_sort_id,
        final_sort,
        Some(limit),
        committed,
        &mut reservations,
        metrics,
    )?;

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

fn execute_bounded_keyed_hash_join_resident_subplan_pipeline_query_plan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let shape =
        analyze_keyed_hash_join_resident_subplan_pipeline(plan).map_err(bounded_join_pipeline_rejection_error)?;
    validate_executable_join_subset(shape.root_join)?;
    validate_bounded_join_pipeline_stages(&[BoundedJoinPipelineStage::Sort {
        node_id: shape.final_sort_id,
        node: shape.final_sort,
    }])?;
    validate_bounded_join_pipeline_stages(&shape.supplier_revenue.projection_stages)?;
    validate_executable_join_subset(shape.supplier_revenue.join)?;

    let supplier_input =
        shape.supplier_revenue.join.inputs[join_side_index(shape.supplier_revenue.supplier_source_side)];
    validate_bounded_join_leaf_input_capabilities(plan, supplier_input)?;
    let supplier_grouped_input =
        shape.supplier_revenue.join.inputs[join_side_index(shape.supplier_revenue.grouped_revenue_side)];
    let supplier_grouped_aggregate =
        BoundedPlan::analyze_projection_wrapped_local_aggregate_subplan(plan, supplier_grouped_input)?;
    supplier_grouped_aggregate.validate_capabilities()?;
    let scalar_grouped_aggregate = BoundedPlan::analyze_projection_wrapped_local_aggregate_subplan(
        plan,
        shape.scalar_max.grouped_revenue_root_id,
    )?;
    scalar_grouped_aggregate.validate_capabilities()?;
    validate_bounded_join_pipeline_stages(&shape.scalar_max.projection_stages_after_grouped_revenue)?;
    validate_bounded_join_pipeline_stages(&[BoundedJoinPipelineStage::Aggregate {
        node_id: shape.scalar_max.aggregate_id,
        node: shape.scalar_max.aggregate,
        output_schema: shape.scalar_max.output_schema,
    }])?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!(
            "execute_bounded_keyed_hash_join_resident_subplan_pipeline_query_plan is called only for bounded policies"
        );
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let supplier_revenue = execute_bounded_keyed_hash_join_resident_supplier_revenue_branch(
        context,
        plan,
        &shape.supplier_revenue,
        &supplier_grouped_aggregate,
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let scalar_max = execute_bounded_keyed_hash_join_resident_scalar_max_branch(
        context,
        &shape.scalar_max,
        &scalar_grouped_aggregate,
        &mut reservations,
        metrics,
    )?;
    let (left, right) = match shape.supplier_revenue_side {
        JoinSide::Left => (supplier_revenue, scalar_max),
        JoinSide::Right => (scalar_max, supplier_revenue),
    };
    let committed = execute_bounded_root_join(
        context,
        shape.root_join_id,
        shape.root_join,
        left,
        right,
        &mut reservations,
        metrics,
    )?;
    let committed = execute_final_sort_node_with_retry(
        context,
        shape.final_sort_id,
        shape.final_sort,
        None,
        committed,
        &mut reservations,
        metrics,
    )?;

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

#[allow(clippy::too_many_arguments)]
fn execute_bounded_keyed_hash_join_resident_supplier_revenue_branch(
    context: &ExecutionContext,
    plan: &QueryPlan,
    branch: &crate::bounded::BoundedKeyedHashJoinResidentSupplierRevenueBranchShape<'_>,
    grouped_aggregate: &BoundedPlan<'_>,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let supplier_input = branch.join.inputs[join_side_index(branch.supplier_source_side)];
    let supplier = execute_bounded_join_leaf_input(context, plan, supplier_input, gpu_inputs, reservations, metrics)?;
    let grouped =
        execute_bounded_projection_wrapped_local_aggregate_input(context, grouped_aggregate, reservations, metrics)?;
    let (left, right) = match branch.supplier_source_side {
        JoinSide::Left => (supplier, grouped),
        JoinSide::Right => (grouped, supplier),
    };
    let mut committed =
        execute_bounded_root_join(context, branch.join_id, branch.join, left, right, reservations, metrics)?;
    for stage in &branch.projection_stages {
        committed = execute_bounded_join_pipeline_stage(context, stage, committed, reservations, metrics)?;
    }
    Ok(committed)
}

fn execute_bounded_keyed_hash_join_resident_scalar_max_branch(
    context: &ExecutionContext,
    branch: &crate::bounded::BoundedKeyedHashJoinResidentScalarMaxBranchShape<'_>,
    grouped_aggregate: &BoundedPlan<'_>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let grouped =
        execute_bounded_projection_wrapped_local_aggregate_input(context, grouped_aggregate, reservations, metrics)?;
    let mut grouped = grouped;
    for stage in &branch.projection_stages_after_grouped_revenue {
        grouped = execute_bounded_join_pipeline_stage(context, stage, grouped, reservations, metrics)?;
    }
    execute_bounded_join_pipeline_stage(
        context,
        &BoundedJoinPipelineStage::Aggregate {
            node_id: branch.aggregate_id,
            node: branch.aggregate,
            output_schema: branch.output_schema,
        },
        grouped,
        reservations,
        metrics,
    )
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
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
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

fn execute_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if let Ok((bounded, post_aggregate_stages)) =
        BoundedPlan::analyze_projection_wrapped_local_aggregate_subplan_with_post_stages(plan, node_id)
    {
        bounded.validate_capabilities()?;
        validate_bounded_join_pipeline_stages(&post_aggregate_stages)?;
        let mut committed =
            execute_bounded_projection_wrapped_local_aggregate_input(context, &bounded, reservations, metrics)?;
        for stage in &post_aggregate_stages {
            committed = execute_bounded_join_pipeline_stage(context, stage, committed, reservations, metrics)?;
        }
        return Ok(committed);
    }

    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(node_id, "bounded resident aggregate subquery references a missing node")
    })?;
    match &node.kind {
        PlanNode::Join(join) => {
            let left = execute_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter(
                context,
                plan,
                join.inputs[0],
                gpu_inputs,
                reservations,
                metrics,
            )?;
            let right = execute_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter(
                context,
                plan,
                join.inputs[1],
                gpu_inputs,
                reservations,
                metrics,
            )?;
            execute_bounded_root_join(context, node.id, join, left, right, reservations, metrics)
        },
        PlanNode::Filter(filter) => {
            let input = execute_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter(
                context,
                plan,
                filter.input,
                gpu_inputs,
                reservations,
                metrics,
            )?;
            execute_bounded_join_pipeline_stage(
                context,
                &BoundedJoinPipelineStage::Filter { node_id, node: filter },
                input,
                reservations,
                metrics,
            )
        },
        PlanNode::Projection(projection) => {
            let input = execute_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter(
                context,
                plan,
                projection.input,
                gpu_inputs,
                reservations,
                metrics,
            )?;
            execute_bounded_join_pipeline_stage(
                context,
                &BoundedJoinPipelineStage::Projection {
                    node_id,
                    node: projection,
                    output_schema: &node.facts.output_schema,
                },
                input,
                reservations,
                metrics,
            )
        },
        PlanNode::Source(_) => execute_bounded_join_leaf_input(context, plan, node_id, gpu_inputs, reservations, metrics),
        PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => Err(unsupported_bounded_shape(
            "bounded resident grouped aggregate subquery filter execution supports source/filter/projection/join nodes plus one projection-wrapped grouped local aggregate",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_resident_aggregate_subquery_node_kind(&node.kind))),
    }
}

fn validate_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter_capabilities(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    if let Ok((bounded, post_aggregate_stages)) =
        BoundedPlan::analyze_projection_wrapped_local_aggregate_subplan_with_post_stages(plan, node_id)
    {
        bounded.validate_capabilities()?;
        return validate_bounded_join_pipeline_stages(&post_aggregate_stages);
    }

    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(node_id, "bounded resident aggregate subquery references a missing node")
    })?;
    match &node.kind {
        PlanNode::Join(join) => {
            validate_executable_join_subset(join)?;
            if !crate::bounded::bounded_multi_join_compatible(join) {
                return Err(unsupported_bounded_shape(
                    "bounded resident grouped aggregate subquery filter execution supports reusable hash joins, filtered equality semi/anti joins, and mixed keyed-residual inner, left semi, and left anti joins",
                    "bounded_join_plan_shape_unsupported",
                )
                .with_fact("join_type", join.spec.join_type)
                .with_fact("build_side", join.spec.build_side)
                .with_fact("key_count", join.spec.keys.len())
                .with_fact("residual_predicate", join.spec.residual_predicate.is_some())
                .with_fact("output_projection_len", join.spec.output_projection.len()));
            }
            if join.spec.residual_predicate.is_some() {
                let residual_columns = residual_input_columns(join)?;
                lower_join_residual_predicate(context, join, &residual_columns)?;
            }
            validate_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter_capabilities(
                context,
                plan,
                join.inputs[0],
            )?;
            validate_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter_capabilities(
                context,
                plan,
                join.inputs[1],
            )
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => {
            lower_expr(&filter.predicate)?;
            validate_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter_capabilities(
                context,
                plan,
                filter.input,
            )
        },
        PlanNode::Projection(projection) => {
            validate_projection_capability(projection, &node.facts.output_schema)?;
            validate_bounded_multi_join_subtree_with_resident_grouped_aggregate_subquery_filter_capabilities(
                context,
                plan,
                projection.input,
            )
        },
        PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => Err(unsupported_bounded_shape(
            "bounded resident grouped aggregate subquery filter execution supports source/filter/projection/join nodes plus one projection-wrapped grouped local aggregate",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_resident_aggregate_subquery_node_kind(&node.kind))),
    }
}

fn bounded_resident_aggregate_subquery_node_kind(node: &PlanNode) -> &'static str {
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
