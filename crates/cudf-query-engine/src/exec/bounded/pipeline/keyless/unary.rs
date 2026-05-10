use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_bounded_keyless_conditional_join_input_subplan_unary(
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
pub(super) fn execute_bounded_keyless_conditional_join_input_subplan_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    projection: &ProjectionNode,
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
            execute_projection_with_budget(
                context,
                projection,
                input,
                token.expect("budget token is created for projection subplan stages"),
                metrics,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_bounded_keyless_conditional_join_input_subplan_aggregate(
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
