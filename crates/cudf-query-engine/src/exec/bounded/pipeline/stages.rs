use super::*;

pub(in crate::exec::bounded) fn validate_bounded_join_pipeline_stages(
    stages: &[BoundedJoinPipelineStage<'_>],
) -> Result<()> {
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

pub(in crate::exec::bounded) fn execute_bounded_join_pipeline_stage(
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
    let operator_token =
        if matches!(stage, BoundedJoinPipelineStage::Projection { .. } | BoundedJoinPipelineStage::Aggregate { .. }) {
            match execution_budget_token_for_boundary(
                context,
                reservations.available_bytes_including(&work_reservation),
                stage_name,
                "cudf_query_engine.bounded_pipeline.budget_token",
                "failed to create bounded native pipeline operator budget token",
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
            debug_assert_eq!(node.output_schema().as_ref(), output_schema.as_ref());
            execute_projection_with_budget(
                context,
                node,
                input.frame,
                operator_token.as_ref().expect("projection pipeline stages create a budget token"),
                metrics,
            )
        },
        BoundedJoinPipelineStage::Aggregate { node, output_schema, .. } => execute_aggregate_with_budget(
            context,
            node,
            input.frame,
            output_schema,
            operator_token.as_ref().expect("aggregate pipeline stages create a budget token"),
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
