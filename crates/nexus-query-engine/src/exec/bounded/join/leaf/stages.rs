use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, BoundedJoinLeafSourceStreamStage,
    ExecutionBudgetToken, ExecutionContext, GpuDataFrame, MemoryReservationRequest, NodeId, ProjectionNode,
    QueryMetrics, ReservationKind, ResidentGpuFrame, Result, execute_filter_with_budget,
    execute_projection_with_budget, execution_budget_token_for_boundary, observe_device_memory_bytes,
    projection_regexp_replace_work_reservation_bytes,
};

pub(super) fn apply_join_leaf_source_stream_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    mut input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    for stage in &stream.stages {
        input = match stage {
            BoundedJoinLeafSourceStreamStage::Filter { node_id, node, dynamic_filter } => {
                let rows_before = input.frame.num_rows() as u64;
                let output = execute_join_leaf_source_stream_unary(
                    execution,
                    *node_id,
                    "join_source_filter",
                    "join_source_filter_output",
                    input,
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
                    execution.metrics_mut().record_dynamic_filter_source_post_decode_stage(
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
                    execution,
                    *node_id,
                    "join_source_projection",
                    "join_source_projection_output",
                    node,
                    input,
                )
            },
        }?;
    }
    Ok(input)
}
pub(super) fn execute_join_leaf_source_stream_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    let work_request_bytes =
        projection_regexp_replace_work_reservation_bytes(execution.context(), projection, &input.frame)?
            .map(|preflight| preflight.max(input.reservation.bytes()))
            .unwrap_or_else(|| input.reservation.bytes());
    execute_join_leaf_source_stream_unary_with_work_bytes(
        execution,
        node_id,
        stage,
        output_boundary,
        input,
        work_request_bytes,
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
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    execute_join_leaf_source_stream_unary_with_work_bytes(
        execution,
        node_id,
        stage,
        output_boundary,
        input,
        input_reserved_bytes,
        execute,
    )
}

fn execute_join_leaf_source_stream_unary_with_work_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    work_request_bytes: u64,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let mut work_reservation = match execution.try_reserve_work(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            work_request_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary(stage)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_attempt(1, 1),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(input.reservation);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.bounded.join.source_stream_unary.budget_token",
        "failed to create bounded native join source stream operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input.reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let result = execute(execution.context(), input.frame, Some(&token), execution.metrics_mut());
    let frame = match result {
        Ok(frame) => {
            execution.release(input.reservation);
            frame
        },
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input.reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(work_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, execution.reservations().budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if let Err(error) =
        execution.reconcile_work_reservation_to_output(&mut work_reservation, output_bytes, &output_request)
    {
        execution.release(work_reservation);
        return Err(error);
    }
    let output_reservation = execution.transfer(work_reservation, output_request);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}
