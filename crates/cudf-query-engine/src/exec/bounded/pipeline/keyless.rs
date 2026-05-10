use super::*;

mod unary;
mod validate;
use unary::{
    execute_bounded_keyless_conditional_join_input_subplan_aggregate,
    execute_bounded_keyless_conditional_join_input_subplan_projection,
    execute_bounded_keyless_conditional_join_input_subplan_unary,
};
use validate::{
    bounded_keyless_conditional_join_input_subplan_node_kind,
    validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_capabilities,
    validate_bounded_keyless_conditional_join_input_subplan_capabilities,
};

use crate::bounded::{
    BoundedJoinPipelineStage, analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
    analyze_bounded_keyless_conditional_join_input_subplan_pipeline,
};

pub(super) fn execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = BoundedKeylessConditionalJoinInputSubplanPipelinePlan::analyze(plan)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;
    validate_bounded_keyless_conditional_join_input_subplan_capabilities(plan, pipeline.join.inputs[0])?;
    validate_bounded_keyless_conditional_join_input_subplan_capabilities(plan, pipeline.join.inputs[1])?;
    validate_executable_join_subset(pipeline.join)?;
    let residual_columns = residual_input_columns(pipeline.join)?;
    lower_join_residual_predicate(context, pipeline.join, &residual_columns)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!(
            "execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan is called only for bounded policies"
        );
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let left = execute_bounded_keyless_conditional_join_input_subplan(
        context,
        plan,
        pipeline.join.inputs[0],
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let right = execute_bounded_keyless_conditional_join_input_subplan(
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

pub(super) fn execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = BoundedKeylessConditionalJoinAggregateOverJoinSubplanInputsPipelinePlan::analyze(plan)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;
    validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_capabilities(
        context,
        plan,
        pipeline.join.inputs[0],
        AggregateOverJoinSubplanInputKind::Grouped,
    )?;
    validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_capabilities(
        context,
        plan,
        pipeline.join.inputs[1],
        AggregateOverJoinSubplanInputKind::Scalar,
    )?;
    validate_executable_join_subset(pipeline.join)?;
    let residual_columns = residual_input_columns(pipeline.join)?;
    lower_join_residual_predicate(context, pipeline.join, &residual_columns)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!(
            "execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan is called only for bounded policies"
        );
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let left = execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs(
        context,
        plan,
        pipeline.join.inputs[0],
        AggregateOverJoinSubplanInputKind::Grouped,
        gpu_inputs,
        &mut reservations,
        metrics,
    )?;
    let right = execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs(
        context,
        plan,
        pipeline.join.inputs[1],
        AggregateOverJoinSubplanInputKind::Scalar,
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

struct BoundedKeylessConditionalJoinInputSubplanPipelinePlan<'a> {
    join_id: NodeId,
    join: &'a JoinNode,
    stages: Vec<BoundedJoinPipelineStage<'a>>,
}

struct BoundedKeylessConditionalJoinAggregateOverJoinSubplanInputsPipelinePlan<'a> {
    join_id: NodeId,
    join: &'a JoinNode,
    stages: Vec<BoundedJoinPipelineStage<'a>>,
}

impl<'a> BoundedKeylessConditionalJoinAggregateOverJoinSubplanInputsPipelinePlan<'a> {
    fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let shape = analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(plan)
            .map_err(bounded_join_pipeline_rejection_error)?;
        Ok(Self { join_id: shape.join_id, join: shape.join, stages: shape.stages })
    }
}

#[derive(Debug, Clone, Copy)]
enum AggregateOverJoinSubplanInputKind {
    /// Execute and validate a grouped aggregate over a bounded join subplan.
    Grouped,
    /// Execute and validate a scalar/global aggregate over a bounded join subplan.
    Scalar,
}

impl<'a> BoundedKeylessConditionalJoinInputSubplanPipelinePlan<'a> {
    fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let shape = analyze_bounded_keyless_conditional_join_input_subplan_pipeline(plan)
            .map_err(bounded_join_pipeline_rejection_error)?;
        Ok(Self { join_id: shape.join_id, join: shape.join, stages: shape.stages })
    }
}

fn execute_bounded_keyless_conditional_join_input_subplan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_bounded_keyless_conditional_join_input_subplan_inner(
        context,
        plan,
        node_id,
        gpu_inputs,
        reservations,
        metrics,
        false,
    )
}

fn execute_bounded_keyless_conditional_join_input_subplan_inner(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    join_reached_through_projection: bool,
) -> Result<CommittedPartial> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join input-subplan pipeline references a missing node",
        )
    })?;
    match &node.kind {
        PlanNode::Source(_) => {
            execute_bounded_join_leaf_input(context, plan, node_id, gpu_inputs, reservations, metrics)
        },
        PlanNode::Filter(filter) => {
            let input = execute_bounded_keyless_conditional_join_input_subplan_inner(
                context,
                plan,
                filter.input,
                gpu_inputs,
                reservations,
                metrics,
                join_reached_through_projection,
            )?;
            execute_bounded_keyless_conditional_join_input_subplan_unary(
                context,
                node_id,
                "join_input_subplan_filter",
                "join_input_subplan_filter_output",
                input,
                reservations,
                metrics,
                |context, input, metrics| execute_filter(context, filter, input, metrics),
            )
        },
        PlanNode::Projection(projection) => {
            let input = execute_bounded_keyless_conditional_join_input_subplan_inner(
                context,
                plan,
                projection.input,
                gpu_inputs,
                reservations,
                metrics,
                true,
            )?;
            execute_bounded_keyless_conditional_join_input_subplan_projection(
                context,
                node_id,
                "join_input_subplan_projection",
                "join_input_subplan_projection_output",
                input,
                reservations,
                metrics,
                projection,
            )
        },
        PlanNode::Aggregate(aggregate) => {
            let input = execute_bounded_keyless_conditional_join_input_subplan_inner(
                context,
                plan,
                aggregate.input,
                gpu_inputs,
                reservations,
                metrics,
                join_reached_through_projection,
            )?;
            execute_bounded_keyless_conditional_join_input_subplan_aggregate(
                context,
                node_id,
                "join_input_subplan_global_aggregate",
                "join_input_subplan_global_aggregate_output",
                input,
                reservations,
                metrics,
                aggregate,
                &node.facts.output_schema,
            )
        },
        PlanNode::Join(join) => {
            if !join_reached_through_projection {
                return Err(unsupported_bounded_shape(
                    "bounded keyless conditional join input-subplan pipeline execution requires left anti hash-join subplans to be projection-wrapped",
                    "bounded_join_plan_shape_unsupported",
                )
                .with_fact("node_id", node_id)
                .with_fact("node_kind", "bare_join")
                .with_fact("join_type", join.spec.join_type));
            }
            let left =
                execute_bounded_join_leaf_input(context, plan, join.inputs[0], gpu_inputs, reservations, metrics)?;
            let right =
                execute_bounded_join_leaf_input(context, plan, join.inputs[1], gpu_inputs, reservations, metrics)?;
            execute_bounded_root_join(context, node_id, join, left, right, reservations, metrics)
        },
        PlanNode::Sort(_) | PlanNode::Limit(_) => Err(unsupported_bounded_shape(
            "bounded keyless conditional join input-subplan pipeline execution does not support sort or limit below root join inputs",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&node.kind))),
    }
}

fn execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    kind: AggregateOverJoinSubplanInputKind,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let projection_node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline references a missing node",
        )
    })?;
    let PlanNode::Projection(projection) = &projection_node.kind else {
        return Err(unsupported_bounded_shape(
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires projection-wrapped aggregate inputs",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&projection_node.kind)));
    };

    let aggregate_node = plan.node(projection.input).ok_or_else(|| {
        invalid_node_reference(
            projection.input,
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline projection references a missing aggregate",
        )
    })?;
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        return Err(unsupported_bounded_shape(
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires projection-wrapped aggregate inputs",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", projection.input)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&aggregate_node.kind)));
    };

    let joined = execute_bounded_aggregate_over_join_subplan_inputs_join_subplan(
        context,
        plan,
        aggregate.input,
        gpu_inputs,
        reservations,
        metrics,
    )?;
    let (aggregate_stage, aggregate_output_boundary) = match kind {
        AggregateOverJoinSubplanInputKind::Grouped => {
            ("join_input_subplan_grouped_aggregate", "join_input_subplan_grouped_aggregate_output")
        },
        AggregateOverJoinSubplanInputKind::Scalar => {
            ("join_input_subplan_global_aggregate", "join_input_subplan_global_aggregate_output")
        },
    };
    let aggregated = execute_bounded_keyless_conditional_join_input_subplan_aggregate(
        context,
        projection.input,
        aggregate_stage,
        aggregate_output_boundary,
        joined,
        reservations,
        metrics,
        aggregate,
        &aggregate_node.facts.output_schema,
    )?;

    execute_bounded_keyless_conditional_join_input_subplan_projection(
        context,
        node_id,
        "join_input_subplan_projection",
        "join_input_subplan_projection_output",
        aggregated,
        reservations,
        metrics,
        projection,
    )
}

fn execute_bounded_aggregate_over_join_subplan_inputs_join_subplan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline references a missing node",
        )
    })?;
    match &node.kind {
        PlanNode::Join(_) => {
            execute_bounded_multi_join_subtree(context, plan, node_id, gpu_inputs, reservations, metrics)
        },
        PlanNode::Projection(projection) => {
            let input = execute_bounded_aggregate_over_join_subplan_inputs_join_subplan(
                context,
                plan,
                projection.input,
                gpu_inputs,
                reservations,
                metrics,
            )?;
            execute_bounded_keyless_conditional_join_input_subplan_projection(
                context,
                node_id,
                "join_input_projection",
                "join_input_projection_output",
                input,
                reservations,
                metrics,
                projection,
            )
        },
        _ => Err(unsupported_bounded_shape(
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires aggregate inputs rooted at joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&node.kind))),
    }
}
