use std::time::Instant;

use super::{
    CommittedPartial, DataFrame, ExecutionMetrics, GpuDataFrame, LimitNode, LimitSpec, Result, cudf_query_error,
    limit_keeps_all_rows, limit_slice,
};
use crate::exec::operators::attach_budget_token_failure_facts;

/// Applies `node`'s skip and fetch to `input`, returning it untouched when it keeps every
/// row. Otherwise slices under `token`; `operation` names the error on failure.
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

/// `execute_limit_with_token` for a plain `DataFrame`.
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

/// `execute_limit_with_token` for a bare `LimitSpec`, used for the `skip + fetch` truncation
/// of top-k candidates.
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

/// Slices `frame` to the rows `spec` keeps out of `row_count`. An offset past the end gives
/// an empty frame, and a slice that keeps every row is returned without a cuDF call. A
/// skip plus fetch that overflows usize is `InvalidPlan`. Slice failures gain the budget
/// scope of `token`.
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
    let output = frame.on(token.exec()).slice(offset, length).map_err(|error| {
        attach_budget_token_failure_facts(
            cudf_query_error(operation, "failed to execute bounded native final sort limit", error),
            token,
        )
    })?;
    metrics.add_operator_time(timer.elapsed());
    Ok(GpuDataFrame::new(output))
}

/// Concatenates candidate frames, in order, into one frame under `token`.
///
/// `frames` must hold at least one candidate. The inputs are cloned handles and stay alive
/// in the caller; the output is an additional allocation. A failure carries a
/// `candidate_count` fact.
pub(super) fn concat_committed_frames_with_token(
    frames: &[CommittedPartial],
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
) -> Result<GpuDataFrame> {
    let first_frame = frames[0].frame.frame().clone();
    let other_frames = frames[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let output = first_frame.on(token.exec()).concat_many(&other_frame_refs).map_err(|error| {
        attach_budget_token_failure_facts(
            cudf_query_error(operation, "failed to concatenate bounded native final sort candidates", error),
            token,
        )
        .with_fact("candidate_count", frames.len())
    })?;
    Ok(GpuDataFrame::new(output))
}
