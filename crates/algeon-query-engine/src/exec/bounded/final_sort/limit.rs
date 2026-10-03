use std::time::Instant;

use super::{
    CommittedPartial, DataFrame, ExecutionMetrics, GpuDataFrame, LimitNode, LimitSpec, Result, cudf_query_error,
    limit_keeps_all_rows, limit_slice,
};
use crate::exec::operators::attach_budget_token_failure_facts;

pub(super) fn execute_limit_with_token(
    node: &LimitNode,
    input: GpuDataFrame,
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let row_count = input.num_rows();
    if limit_keeps_all_rows(row_count, node.spec)? {
        return Ok(input);
    }

    execute_limit_spec_on_frame_with_token(node.spec, input.into_frame(), row_count, token, operation, metrics)
}

pub(super) fn execute_limit_on_frame_with_token(
    node: &LimitNode,
    frame: DataFrame,
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let row_count = frame.num_rows();
    execute_limit_spec_on_frame_with_token(node.spec, frame, row_count, token, operation, metrics)
}

pub(super) fn execute_limit_spec_with_token(
    spec: LimitSpec,
    input: GpuDataFrame,
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let row_count = input.num_rows();
    if limit_keeps_all_rows(row_count, spec)? {
        return Ok(input);
    }

    execute_limit_spec_on_frame_with_token(spec, input.into_frame(), row_count, token, operation, metrics)
}

pub(super) fn execute_limit_spec_on_frame_with_token(
    spec: LimitSpec,
    frame: DataFrame,
    row_count: usize,
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let (offset, length) = limit_slice(row_count, spec)?;
    if offset == 0 && length == row_count {
        return Ok(GpuDataFrame::new(frame));
    }

    let timer = Instant::now();
    let output = frame.with_execution(&token.execution_overrides()).slice(offset, length).map_err(|error| {
        attach_budget_token_failure_facts(
            cudf_query_error(operation, "failed to execute bounded native final sort limit", error),
            token,
        )
    })?;
    metrics.add_operator_time(timer.elapsed());
    Ok(GpuDataFrame::new(output))
}

pub(super) fn concat_committed_frames_with_token(
    frames: &[CommittedPartial],
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
) -> Result<GpuDataFrame> {
    let first_frame = frames[0].frame.frame().clone();
    let other_frames = frames[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let output =
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                cudf_query_error(operation, "failed to concatenate bounded native final sort candidates", error),
                token,
            )
            .with_fact("candidate_count", frames.len())
        })?;
    Ok(GpuDataFrame::new(output))
}
