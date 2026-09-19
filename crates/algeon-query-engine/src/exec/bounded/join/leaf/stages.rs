use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, BoundedJoinLeafSourceStreamStage,
    ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, GpuDataFrame, NodeId, ProjectionNode, ResidentGpuFrame,
    Result, execute_filter_with_budget, execute_projection_with_budget, execution_budget_token_for_boundary,
    observe_device_memory_bytes,
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
                    execution.metrics().record_dynamic_filter_source_post_decode_stage(
                        &dynamic_filter.route,
                        &dynamic_filter.filter,
                        Some(rows_before),
                        Some(output.frame.num_rows() as u64),
                    );
                }
                Ok(output)
            },
            BoundedJoinLeafSourceStreamStage::Projection { node_id, node } => {
                execute_join_leaf_source_stream_projection(execution, *node_id, "join_source_projection", node, input)
            },
        }?;
    }
    Ok(input)
}
pub(super) fn execute_join_leaf_source_stream_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    execute_join_leaf_source_stream_unary(execution, node_id, stage, input, |context, input, token, metrics| {
        execute_projection_with_budget(
            context,
            projection,
            input,
            token.expect("budget token is created for join source stream projection stages"),
            metrics,
        )
    })
}

pub(super) fn execute_join_leaf_source_stream_unary(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    input: ResidentGpuFrame,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &ExecutionMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "query_engine.bounded.join.source_stream_unary.budget_token",
        "failed to create bounded native join source stream operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let result = execute(execution.context(), input.frame, Some(&token), execution.metrics());
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    Ok(ResidentGpuFrame::with_device_bytes(frame, output_bytes))
}
