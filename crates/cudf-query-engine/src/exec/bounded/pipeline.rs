use super::*;
use crate::bounded_support::{BoundedJoinPipelineRejection, BoundedJoinPipelineStage, analyze_bounded_join_pipeline};

pub(super) fn execute_bounded_join_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = BoundedJoinPipelinePlan::analyze(plan)?;
    validate_bounded_join_pipeline_stages(&pipeline.stages)?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_join_pipeline_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut scratch = ExecutionScratch::new(context);
    let left = execute_node(context, plan, pipeline.join.inputs[0], gpu_inputs, &mut scratch, metrics)?;
    let right = execute_node(context, plan, pipeline.join.inputs[1], gpu_inputs, &mut scratch, metrics)?;
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
    let result = match stage {
        BoundedJoinPipelineStage::Filter { node, .. } => execute_filter(context, node, input.frame, metrics),
        BoundedJoinPipelineStage::Projection { node, output_schema, .. } => {
            execute_projection(context, node, input.frame, output_schema, metrics)
        },
        BoundedJoinPipelineStage::Aggregate { node, output_schema, .. } => {
            execute_aggregate(context, node, input.frame, output_schema, metrics)
        },
        BoundedJoinPipelineStage::Sort { node, .. } => execute_sort(context, node, input.frame, metrics),
        BoundedJoinPipelineStage::Limit { node, .. } => execute_limit(context, node, input.frame, metrics),
    };
    reservations.release(work_reservation);

    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
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

    reservations.release(input.reservation);
    let output_boundary = pipeline_stage_output_memory_boundary(stage);
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: output_boundary,
            stage: stage_name,
            node_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: stage_name,
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request = MemoryReservationRequest::new(kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage_name)
        .with_attempt(1, 1);
    let output_reservation = reservations.try_reserve(output_request, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: output_boundary,
            stage: stage_name,
            node_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: stage_name,
            metrics,
        }
        .attach_retry_context(error)
    })?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
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
        BoundedJoinPipelineRejection::UnsupportedLeaf { node_id, node_kind } => unsupported_bounded_shape(
            "bounded multi-join pipeline execution supports only source/filter/projection leaf subtrees below joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("node_id", node_id)
        .with_fact("node_kind", node_kind),
        BoundedJoinPipelineRejection::LimitWithoutSort => unsupported_bounded_shape(
            "bounded join pipeline execution does not support limit without a preceding global sort",
            "limit_without_sort",
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
