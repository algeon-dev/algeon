use super::*;

pub(super) fn apply_join_leaf_source_stream_stages(
    context: &ExecutionContext,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    mut input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    for stage in &stream.stages {
        input = match stage {
            BoundedJoinLeafSourceStreamStage::Filter { node_id, node, dynamic_filter } => {
                let rows_before = input.frame.num_rows() as u64;
                let output = execute_join_leaf_source_stream_unary(
                    context,
                    *node_id,
                    "join_source_filter",
                    "join_source_filter_output",
                    input,
                    reservations,
                    metrics,
                    |context, input, token, metrics| {
                        execute_filter_with_budget(
                            context,
                            node,
                            input,
                            token.expect("budget token is created for join source stream filter stages"),
                            metrics,
                        )
                    },
                )?;
                if let Some(dynamic_filter) = dynamic_filter {
                    metrics.record_dynamic_filter_source_post_decode_stage(
                        &dynamic_filter.route,
                        &dynamic_filter.filter,
                        Some(rows_before),
                        Some(output.frame.num_rows() as u64),
                    );
                }
                Ok(output)
            },
            BoundedJoinLeafSourceStreamStage::Projection { node_id, node } => {
                execute_join_leaf_source_stream_projection(
                    context,
                    *node_id,
                    "join_source_projection",
                    "join_source_projection_output",
                    node,
                    input,
                    reservations,
                    metrics,
                )
            },
        }?;
    }
    Ok(input)
}
pub(super) fn execute_join_leaf_source_stream_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let work_request_bytes = projection_regexp_replace_work_reservation_bytes(context, projection, &input.frame)?
        .map(|preflight| preflight.max(input.reservation.bytes()))
        .unwrap_or_else(|| input.reservation.bytes());
    execute_join_leaf_source_stream_unary_with_work_bytes(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        work_request_bytes,
        reservations,
        metrics,
        |context, input, token, metrics| {
            execute_projection_with_budget(
                context,
                projection,
                input,
                token.expect("budget token is created for join source stream projection stages"),
                metrics,
            )
        },
    )
}

pub(super) fn execute_join_leaf_source_stream_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    execute_join_leaf_source_stream_unary_with_work_bytes(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        input_reserved_bytes,
        reservations,
        metrics,
        execute,
    )
}

fn execute_join_leaf_source_stream_unary_with_work_bytes(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    work_request_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let mut work_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, work_request_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.bounded.join.source_stream_unary.budget_token",
        "failed to create bounded native join source stream operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let result = execute(context, input.frame, Some(&token), metrics);
    let frame = match result {
        Ok(frame) => {
            reservations.release(input.reservation, metrics);
            frame
        },
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}
