use super::*;
use crate::bounded_support::{
    BoundedJoinPipelineStage, analyze_bounded_multi_join_pipeline, bounded_multi_join_compatible,
};

pub(super) fn execute_bounded_multi_join_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let shape = analyze_bounded_multi_join_pipeline(plan).map_err(bounded_join_pipeline_rejection_error)?;
    validate_bounded_join_pipeline_stages(&shape.stages)?;
    validate_bounded_multi_join_subtree_capabilities(context, plan, shape.join_id)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_multi_join_pipeline_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(context.device_memory_space(), device_budget_bytes);
    let mut committed =
        execute_bounded_multi_join_subtree(context, plan, shape.join_id, gpu_inputs, &mut reservations, metrics)?;

    for stage in &shape.stages {
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

pub(super) fn execute_bounded_multi_join_subtree(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_bounded_multi_join_input(context, plan, node_id, gpu_inputs, reservations, metrics)
}

fn execute_bounded_multi_join_input(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let node = plan
        .node(node_id)
        .ok_or_else(|| invalid_node_reference(node_id, "bounded multi-join references a missing node"))?;
    match &node.kind {
        PlanNode::Join(join) => {
            let left_input =
                execute_bounded_multi_join_input(context, plan, join.inputs[0], gpu_inputs, reservations, metrics)?;
            let right_input =
                execute_bounded_multi_join_input(context, plan, join.inputs[1], gpu_inputs, reservations, metrics)?;
            execute_bounded_root_join(context, node.id, join, left_input, right_input, reservations, metrics)
        },
        PlanNode::Filter(filter) => {
            let input =
                execute_bounded_multi_join_input(context, plan, filter.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_multi_join_unary_input(
                context,
                BoundedJoinPipelineStage::Filter { node_id, node: filter },
                input,
                reservations,
                metrics,
            )
        },
        PlanNode::Projection(projection) => {
            let input =
                execute_bounded_multi_join_input(context, plan, projection.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_multi_join_unary_input(
                context,
                BoundedJoinPipelineStage::Projection {
                    node_id,
                    node: projection,
                    output_schema: &node.facts.output_schema,
                },
                input,
                reservations,
                metrics,
            )
        },
        PlanNode::Source(_) => {
            execute_bounded_join_leaf_input(context, plan, node_id, gpu_inputs, reservations, metrics)
        },
        PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => Err(unsupported_bounded_shape(
            "bounded multi-join pipeline execution supports only source/filter/projection leaf subtrees below joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_multi_join_node_kind(&node.kind))),
    }
}

fn execute_bounded_multi_join_unary_input(
    context: &ExecutionContext,
    stage: BoundedJoinPipelineStage<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_bounded_join_pipeline_stage(context, &stage, input, reservations, metrics)
}

pub(super) fn validate_bounded_multi_join_subtree_capabilities(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    let node = plan
        .node(node_id)
        .ok_or_else(|| invalid_node_reference(node_id, "bounded multi-join references a missing node"))?;
    match &node.kind {
        PlanNode::Join(join) => {
            validate_executable_join_subset(join)?;
            if !bounded_multi_join_compatible(join) {
                return Err(unsupported_bounded_shape(
                    "bounded multi-join pipeline execution supports reusable hash equi joins, filtered equality semi/anti joins, and mixed keyed-residual inner, left semi, and left anti joins",
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
            validate_bounded_multi_join_subtree_capabilities(context, plan, join.inputs[0])?;
            validate_bounded_multi_join_subtree_capabilities(context, plan, join.inputs[1])
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => {
            lower_expr(&filter.predicate)?;
            validate_bounded_multi_join_subtree_capabilities(context, plan, filter.input)
        },
        PlanNode::Projection(projection) => {
            validate_projection_capability(projection, &node.facts.output_schema)?;
            validate_bounded_multi_join_subtree_capabilities(context, plan, projection.input)
        },
        PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => Err(unsupported_bounded_shape(
            "bounded multi-join pipeline execution supports only source/filter/projection leaf subtrees below joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_multi_join_node_kind(&node.kind))),
    }
}

fn bounded_multi_join_node_kind(node: &PlanNode) -> &'static str {
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
