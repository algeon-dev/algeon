use super::*;
use crate::bounded_support::{
    BoundedJoinPipelineRejection, BoundedJoinPipelineStage, analyze_bounded_join_pipeline,
    analyze_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
    analyze_bounded_keyless_conditional_join_input_subplan_pipeline,
};

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
    let mut reservations = ReservationTracker::new(device_budget_bytes);
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

fn execute_bounded_keyless_conditional_join_input_subplan_pipeline_query_plan_inner(
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
    let mut reservations = ReservationTracker::new(device_budget_bytes);
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
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn execute_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_query_plan_inner(
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
    let mut reservations = ReservationTracker::new(device_budget_bytes);
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

pub(super) fn validate_bounded_join_pipeline_stages(stages: &[BoundedJoinPipelineStage<'_>]) -> Result<()> {
    for stage in stages {
        match stage {
            BoundedJoinPipelineStage::Filter { node, .. } => {
                lower_expr(&node.predicate)?;
            },
            BoundedJoinPipelineStage::Projection { node, output_schema, .. } => {
                validate_projection_capability(node, output_schema)?;
            },
            BoundedJoinPipelineStage::Aggregate { node, output_schema, .. } => {
                validate_aggregate_capability(node, output_schema)?;
            },
            BoundedJoinPipelineStage::Sort { node, .. } => {
                lower_sort_spec(&node.spec)?;
            },
            BoundedJoinPipelineStage::Limit { node, .. } => {
                topk_candidate_rows(node.spec)?;
            },
        }
    }
    Ok(())
}

fn validate_bounded_keyless_conditional_join_input_subplan_capabilities(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    validate_bounded_keyless_conditional_join_input_subplan_capabilities_inner(plan, node_id, false)
}

fn validate_bounded_keyless_conditional_join_input_subplan_capabilities_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    join_reached_through_projection: bool,
) -> Result<()> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join input-subplan pipeline references a missing node",
        )
    })?;
    match &node.kind {
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(_) => validate_bounded_join_leaf_input_capabilities(plan, node_id),
        PlanNode::Projection(projection) => {
            validate_projection_capability(projection, &node.facts.output_schema)?;
            validate_bounded_keyless_conditional_join_input_subplan_capabilities_inner(plan, projection.input, true)
        },
        PlanNode::Aggregate(aggregate) => {
            if !aggregate.spec.group_exprs.is_empty() {
                return Err(unsupported_bounded_shape(
                    "bounded keyless conditional join input-subplan pipeline execution supports only scalar/global aggregate inputs",
                    "bounded_join_plan_shape_unsupported",
                )
                .with_fact("node_id", node_id)
                .with_fact("node_kind", "grouped_aggregate"));
            }
            validate_aggregate_capability(aggregate, &node.facts.output_schema)?;
            validate_bounded_join_leaf_input_capabilities(plan, aggregate.input)
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
            validate_executable_join_subset(join)?;
            if !(join.spec.join_type == JoinType::LeftAnti && join.is_bounded_root_filtered_join_compatible()) {
                return Err(unsupported_bounded_shape(
                    "bounded keyless conditional join input-subplan pipeline execution supports only left anti hash-join subplans",
                    "bounded_join_plan_shape_unsupported",
                )
                .with_fact("join_type", join.spec.join_type)
                .with_fact("build_side", join.spec.build_side)
                .with_fact("key_count", join.spec.keys.len())
                .with_fact("residual_predicate", join.spec.residual_predicate.is_some())
                .with_fact("output_projection_len", join.spec.output_projection.len()));
            }
            validate_bounded_join_leaf_input_capabilities(plan, join.inputs[0])?;
            validate_bounded_join_leaf_input_capabilities(plan, join.inputs[1])
        },
        PlanNode::Sort(_) | PlanNode::Limit(_) => Err(unsupported_bounded_shape(
            "bounded keyless conditional join input-subplan pipeline execution does not support sort or limit below root join inputs",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", bounded_keyless_conditional_join_input_subplan_node_kind(&node.kind))),
    }
}

fn validate_bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_capabilities(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    kind: AggregateOverJoinSubplanInputKind,
) -> Result<()> {
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
    validate_projection_capability(projection, &projection_node.facts.output_schema)?;

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
    match kind {
        AggregateOverJoinSubplanInputKind::Grouped if aggregate.spec.group_exprs.is_empty() => {
            return Err(unsupported_bounded_shape(
                "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires grouped aggregate on the left input",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", projection.input)
            .with_fact("node_kind", "scalar_aggregate"));
        },
        AggregateOverJoinSubplanInputKind::Scalar if !aggregate.spec.group_exprs.is_empty() => {
            return Err(unsupported_bounded_shape(
                "bounded keyless conditional join aggregate-over-join subplan inputs pipeline execution requires scalar/global aggregate on the right input",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", projection.input)
            .with_fact("node_kind", "grouped_aggregate"));
        },
        AggregateOverJoinSubplanInputKind::Grouped | AggregateOverJoinSubplanInputKind::Scalar => {},
    }
    validate_aggregate_capability(aggregate, &aggregate_node.facts.output_schema)?;
    validate_bounded_aggregate_over_join_subplan_inputs_join_subplan_capabilities(context, plan, aggregate.input)
}

fn validate_bounded_aggregate_over_join_subplan_inputs_join_subplan_capabilities(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    let node = plan.node(node_id).ok_or_else(|| {
        invalid_node_reference(
            node_id,
            "bounded keyless conditional join aggregate-over-join subplan inputs pipeline aggregate references a missing join subtree",
        )
    })?;
    match &node.kind {
        PlanNode::Join(_) => validate_bounded_multi_join_subtree_capabilities(context, plan, node_id),
        PlanNode::Projection(projection) => {
            validate_projection_capability(projection, &node.facts.output_schema)?;
            validate_bounded_aggregate_over_join_subplan_inputs_join_subplan_capabilities(
                context,
                plan,
                projection.input,
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
            execute_bounded_keyless_conditional_join_input_subplan_unary(
                context,
                node_id,
                "join_input_subplan_projection",
                "join_input_subplan_projection_output",
                input,
                reservations,
                metrics,
                |context, input, metrics| {
                    execute_projection(context, projection, input, &node.facts.output_schema, metrics)
                },
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

    execute_bounded_keyless_conditional_join_input_subplan_unary(
        context,
        node_id,
        "join_input_subplan_projection",
        "join_input_subplan_projection_output",
        aggregated,
        reservations,
        metrics,
        |context, input, metrics| {
            execute_projection(context, projection, input, &projection_node.facts.output_schema, metrics)
        },
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
            execute_bounded_keyless_conditional_join_input_subplan_unary(
                context,
                node_id,
                "join_input_projection",
                "join_input_projection_output",
                input,
                reservations,
                metrics,
                |context, input, metrics| {
                    execute_projection(context, projection, input, &node.facts.output_schema, metrics)
                },
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

fn execute_bounded_keyless_conditional_join_input_subplan_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &mut QueryMetrics) -> Result<GpuDataFrame>,
) -> Result<CommittedPartial> {
    execute_bounded_keyless_conditional_join_input_subplan_unary_impl(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        reservations,
        metrics,
        false,
        |context, input, _token, metrics| execute(context, input, metrics),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_bounded_keyless_conditional_join_input_subplan_aggregate(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    aggregate: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<CommittedPartial> {
    execute_bounded_keyless_conditional_join_input_subplan_unary_impl(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        reservations,
        metrics,
        true,
        |context, input, token, metrics| {
            execute_aggregate_with_budget(
                context,
                aggregate,
                input,
                output_schema,
                token.expect("budget token is created for aggregate subplan stages"),
                metrics,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_bounded_keyless_conditional_join_input_subplan_unary_impl(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<CommittedPartial> {
    let work_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            input.reservation.bytes(),
            reservations.budget(),
        )
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
            "cudf_query_engine.bounded_pipeline.budget_token",
            "failed to create bounded native pipeline operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                reservations.release(work_reservation);
                reservations.release(input.reservation);
                return Err(RuntimeResourceFacts {
                    memory_boundary: stage,
                    stage,
                    node_id,
                    attempt: 1,
                    max_attempts: 1,
                    resume_boundary: stage,
                    metrics,
                }
                .attach_to(error));
            },
        }
    } else {
        None
    };
    let result = execute(context, input.frame, token.as_ref(), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: stage,
                stage,
                node_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: stage,
                metrics,
            }
            .attach_to(error));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: output_boundary,
                stage,
                node_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: stage,
                metrics,
            }
            .attach_to(error));
        },
    };
    let output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(error);
        },
    };
    reservations.release(work_reservation);
    reservations.release(input.reservation);
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn execute_bounded_join_pipeline_stage(
    context: &ExecutionContext,
    stage: &BoundedJoinPipelineStage<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let node_id = pipeline_stage_node_id(stage);
    let stage_name = pipeline_stage_memory_boundary(stage);
    let kind = pipeline_stage_reservation_kind(stage);
    let work_request = MemoryReservationRequest::new(kind, input.reservation.bytes(), reservations.budget())
        .with_memory_boundary(stage_name)
        .with_node_id(node_id)
        .with_stage(stage_name)
        .with_attempt(1, 1);
    let work_reservation = reservations.try_reserve(work_request, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: stage_name,
            stage: stage_name,
            node_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: stage_name,
            metrics,
        }
        .attach_retry_context(error)
    })?;
    let aggregate_token = if matches!(stage, BoundedJoinPipelineStage::Aggregate { .. }) {
        match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&work_reservation),
            stage_name,
            "cudf_query_engine.bounded_pipeline.budget_token",
            "failed to create bounded native pipeline aggregate budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                reservations.release(work_reservation);
                reservations.release(input.reservation);
                return Err(RuntimeResourceFacts {
                    memory_boundary: stage_name,
                    stage: stage_name,
                    node_id,
                    attempt: 1,
                    max_attempts: 1,
                    resume_boundary: stage_name,
                    metrics,
                }
                .attach_to(error));
            },
        }
    } else {
        None
    };
    let result = match stage {
        BoundedJoinPipelineStage::Filter { node, .. } => execute_filter(context, node, input.frame, metrics),
        BoundedJoinPipelineStage::Projection { node, output_schema, .. } => {
            execute_projection(context, node, input.frame, output_schema, metrics)
        },
        BoundedJoinPipelineStage::Aggregate { node, output_schema, .. } => execute_aggregate_with_budget(
            context,
            node,
            input.frame,
            output_schema,
            aggregate_token.as_ref().expect("aggregate pipeline stages create a budget token"),
            metrics,
        ),
        BoundedJoinPipelineStage::Sort { node, .. } => execute_sort(context, node, input.frame, metrics),
        BoundedJoinPipelineStage::Limit { node, .. } => execute_limit(context, node, input.frame, metrics),
    };

    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: stage_name,
                stage: stage_name,
                node_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: stage_name,
                metrics,
            }
            .attach_to(error));
        },
    };

    let output_boundary = pipeline_stage_output_memory_boundary(stage);
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: output_boundary,
                stage: stage_name,
                node_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: stage_name,
                metrics,
            }
            .attach_to(error));
        },
    };
    let output_request = MemoryReservationRequest::new(kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage_name)
        .with_attempt(1, 1);
    let output_reservation = match reservations.try_reserve(output_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: output_boundary,
                stage: stage_name,
                node_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: stage_name,
                metrics,
            }
            .attach_retry_context(error));
        },
    };
    reservations.release(work_reservation);
    reservations.release(input.reservation);
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn bounded_keyless_conditional_join_input_subplan_node_kind(node: &PlanNode) -> &'static str {
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

fn pipeline_stage_node_id(stage: &BoundedJoinPipelineStage<'_>) -> NodeId {
    match stage {
        BoundedJoinPipelineStage::Filter { node_id, .. }
        | BoundedJoinPipelineStage::Projection { node_id, .. }
        | BoundedJoinPipelineStage::Aggregate { node_id, .. }
        | BoundedJoinPipelineStage::Sort { node_id, .. }
        | BoundedJoinPipelineStage::Limit { node_id, .. } => *node_id,
    }
}

fn pipeline_stage_reservation_kind(stage: &BoundedJoinPipelineStage<'_>) -> ReservationKind {
    match stage {
        BoundedJoinPipelineStage::Sort { .. } | BoundedJoinPipelineStage::Limit { .. } => ReservationKind::FinalSort,
        BoundedJoinPipelineStage::Filter { .. }
        | BoundedJoinPipelineStage::Projection { .. }
        | BoundedJoinPipelineStage::Aggregate { .. } => ReservationKind::OperatorOutput,
    }
}

fn pipeline_stage_memory_boundary(stage: &BoundedJoinPipelineStage<'_>) -> &'static str {
    match stage {
        BoundedJoinPipelineStage::Filter { .. } => "join_pipeline_filter",
        BoundedJoinPipelineStage::Projection { .. } => "join_pipeline_projection",
        BoundedJoinPipelineStage::Aggregate { .. } => "join_pipeline_aggregate",
        BoundedJoinPipelineStage::Sort { .. } => "join_pipeline_sort",
        BoundedJoinPipelineStage::Limit { .. } => "join_pipeline_limit",
    }
}

fn pipeline_stage_output_memory_boundary(stage: &BoundedJoinPipelineStage<'_>) -> &'static str {
    match stage {
        BoundedJoinPipelineStage::Filter { .. } => "join_pipeline_filter_output",
        BoundedJoinPipelineStage::Projection { .. } => "join_pipeline_projection_output",
        BoundedJoinPipelineStage::Aggregate { .. } => "join_pipeline_aggregate_output",
        BoundedJoinPipelineStage::Sort { .. } => "join_pipeline_sort_output",
        BoundedJoinPipelineStage::Limit { .. } => "join_pipeline_limit_output",
    }
}
