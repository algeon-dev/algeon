use query_planning::dynamic_filter::DynamicFilterConsumer;

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, BoundedJoinLeafSourceStreamStage, NodeId,
    ProjectionNode, ResidentGpuFrame, Result, execute_filter_with_budget, execute_projection_with_budget,
};
use crate::exec::bounded::run_budgeted;

/// Runs the stream's post-decode filter and projection stages on `input` in order on the
/// calling thread, each under a budget token labeled `join_source_filter` or
/// `join_source_projection`.
///
/// Consumes `input`; an error drops it with any intermediate frame. After a dynamic-filter
/// stage the chunk's rows before and after are recorded.
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
                // Both route kinds were counted as applied when the stage was
                // installed; each chunk adds only its row evidence.
                if let Some(dynamic_filter) = dynamic_filter {
                    let rows_after = Some(output.frame.num_rows() as u64);
                    let metrics = execution.metrics();
                    match dynamic_filter.route.consumer {
                        DynamicFilterConsumer::LocalParquetSource { .. } => metrics
                            .record_dynamic_filter_source_post_decode_stage(
                                &dynamic_filter.route,
                                &dynamic_filter.filter,
                                Some(rows_before),
                                rows_after,
                            ),
                        DynamicFilterConsumer::Filter { .. } | DynamicFilterConsumer::HashJoinProbe { .. } => metrics
                            .record_dynamic_filter_post_decode_rows(
                                &dynamic_filter.route,
                                &dynamic_filter.filter,
                                Some(rows_before),
                                rows_after,
                            ),
                    }
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
