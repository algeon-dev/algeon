use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, BoundedJoinLeafSourceStreamStage, NodeId,
    ProjectionNode, ResidentGpuFrame, Result, execute_filter_with_budget, execute_projection_with_budget,
};
use crate::exec::bounded::run_budgeted;

pub(super) fn apply_join_leaf_source_stream_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stream: &BoundedJoinLeafLocalParquetSourceStream,
    mut input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    for stage in &stream.stages {
        input = match stage {
            BoundedJoinLeafSourceStreamStage::Filter { node_id, node, dynamic_filter } => {
                let rows_before = input.frame.num_rows() as u64;
                let output = run_budgeted(
                    execution,
                    *node_id,
                    "join_source_filter",
                    input.frame,
                    |context, input, token, metrics| execute_filter_with_budget(context, node, input, token, metrics),
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
    run_budgeted(execution, node_id, stage, input.frame, |context, input, token, metrics| {
        execute_projection_with_budget(context, projection, input, token, metrics)
    })
}
