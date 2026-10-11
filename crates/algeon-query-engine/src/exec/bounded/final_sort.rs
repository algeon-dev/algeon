//! Global ORDER BY and ORDER BY ... LIMIT over committed chunk results.
//!
//! A global order cannot come from independently sorted chunks. The final sort either sorts
//! the whole input at once (`full_sort`) or, when a finite limit makes it cheaper, runs a
//! hierarchical top-k (`hierarchical_topk`). A top-k sorts row ranges into candidates of
//! `skip + fetch` rows, then merges candidate groups level by level, where each merge
//! concatenates, sorts and truncates to the same bound. `sort_lane` states why the final
//! candidate equals the global top rows; only the closing limit applies the offset. Neither
//! path asks for a stable sort; rows tied on the sort key have no defined order.
//!
//! Retry shrinks the working set. A full sort with a limit that hits a query-cap rejection
//! reruns as a hierarchical top-k at attempt 2. Inside a top-k, a failed lane fan-out reruns
//! its candidates serially, and a failed merge level halves its fan-in and redoes that level
//! from the same candidates, down to a fan-in of 2. Candidate row ranges never shrink, and
//! a serial first-attempt candidate failure has nothing narrower to try. Other error codes,
//! including physical device exhaustion, return unchanged (`exec::retry`). The query
//! allocator enforces the cap in every path; the budget tokens here only label stages.

use std::time::Instant;

use cudf::{ExecutionBudgetReport, ExecutionBudgetRequest};
use query_planning::chunk::SourceChunkPlanningPolicy;

use super::{
    BoundedExecutionAttempt, CommittedPartial, CudfSortOptions, DataFrame, ExecutionBudgetToken, ExecutionMetrics,
    GpuDataFrame, LimitNode, LimitSpec, NodeId, QueryError, QueryErrorCode, ResidentGpuFrame, Result, RetryFacts,
    RuntimeResourceFacts, SortNode, compiled::ResidentFrameStreamPayload, cudf_query_error, limit_keeps_all_rows,
    limit_slice, lower_sort_spec, observe_device_memory_bytes, reduce_merge_fan_in, topk_candidate_rows,
};

mod full_sort;
mod hierarchical_topk;
mod limit;
mod retry;
mod sizing;
mod sort_lane;
mod topk_candidate;
mod topk_level;

use limit::{
    concat_committed_frames_with_token, execute_limit_on_frame_with_token, execute_limit_spec_with_token,
    execute_limit_with_token,
};
use query_runtime::observability::{
    RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
};
use retry::{
    final_sort_retry_facts, final_sort_topk_retry_facts, is_retryable_final_sort_error, record_final_sort_retry_error,
};
use sizing::{estimate_rows_bytes, estimated_bytes_per_row, final_sort_retained_bytes, final_sort_rows_per_chunk};
use sort_lane::{
    SortLaneCandidate, execute_sort_lane_candidate, sort_lane_fan_out, sort_lane_params, split_sort_candidate_ranges,
    stream_candidate_chunk_hint,
};
use topk_candidate::sort_topk_candidate_with_budget;
use topk_level::{TopKLevelItem, commit_topk_merge_level};

use super::compiled::lanes::AggregateLaneDispatch;
use crate::exec::retry::{ResourceFailureRecovery, RetryLimits, backoff_before_retry, resource_failure_recovery};

/// Sorts a committed input globally and applies `limit`, if any.
///
/// With a limit, `should_use_hierarchical_topk` chooses between a top-k at attempt 1 and a
/// full sort. A full sort that fails with a query-cap rejection reruns as a top-k at
/// attempt 2 after a backoff, when `max_attempts` allows and a limit exists. Without a limit,
/// or with a single attempt, the error returns with `retry_attempts_exhausted` retry facts.
/// A top-k failure returns as it is, since the top-k carries its own retry. `limit` must
/// carry a fetch bound for the top-k; planning does not fuse an offset-only limit into the
/// sort, and one passed here fails with `UnsupportedOperator`. Consumes `input`.
///
/// Final sort time is added to the metrics on every exit. When `dispatch` has two or more
/// lanes, the first top-k attempt fans its candidates out to them.
pub(super) fn execute_final_sort_node_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: CommittedPartial,
    dispatch: Option<&AggregateLaneDispatch>,
) -> Result<CommittedPartial> {
    let RetryLimits { max_attempts, .. } = RetryLimits::of(execution.context());
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { execution, dispatch, node_id, sort, max_attempts };
    let input_rows = input.frame.num_rows() as u64;
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(final_sort.execution.context());
    let stage_guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::FinalSort,
        RuntimeOperatorStageIdentity::new().with_node_id(node_id.to_string()).with_input_rows(input_rows),
    );

    if let Some(limit) = limit {
        match final_sort.should_use_hierarchical_topk(limit, &input) {
            Ok(true) => {
                let result = final_sort.hierarchical_topk(limit, input, 1);
                final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
                return match result {
                    Ok(output) => {
                        stage_guard.finish(Some(output.frame.num_rows() as u64), None);
                        Ok(output)
                    },
                    Err(error) => {
                        stage_guard.fail(&error);
                        Err(error)
                    },
                };
            },
            Ok(false) => {},
            Err(error) => {
                final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
                stage_guard.fail(&error);
                return Err(error);
            },
        }
    }

    let result = final_sort.full_sort_limit_attempt(limit, &input, 1);

    match result {
        Ok(output) => {
            final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
            stage_guard.finish(Some(output.frame.num_rows() as u64), None);
            Ok(output)
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.execution.metrics());
            if max_attempts > 1
                && let Some(limit) = limit
            {
                if let Err(backoff_error) = backoff_before_retry(
                    final_sort.execution.context(),
                    "query_engine.exec.bounded.final_sort_hierarchical_topk_retry",
                ) {
                    stage_guard.fail(&backoff_error);
                    return Err(backoff_error);
                }
                match final_sort.hierarchical_topk(limit, input, 2) {
                    Ok(output) => {
                        final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
                        stage_guard.finish(Some(output.frame.num_rows() as u64), None);
                        Ok(output)
                    },
                    Err(error) => {
                        final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
                        stage_guard.fail(&error);
                        Err(error)
                    },
                }
            } else {
                final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
                let error = final_sort_retry_facts("retry_attempts_exhausted", final_sort.execution.metrics())
                    .attach_to_error(error);
                stage_guard.fail(&error);
                Err(error)
            }
        },
        Err(error) => {
            final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
            stage_guard.fail(&error);
            Err(error)
        },
    }
}

/// `execute_final_sort_node_with_retry` for a resident frame that has not been committed as
/// a partial.
pub(in crate::exec::bounded) fn execute_final_sort_node_from_committed_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: ResidentGpuFrame,
    dispatch: Option<&AggregateLaneDispatch>,
) -> Result<ResidentGpuFrame> {
    execute_final_sort_node_with_retry(execution, node_id, sort, limit, input, dispatch)
}

/// Full sort of a raw frame with no limit, serially on the coordinator. Chunk-local sorts
/// cannot replace a global sort; no lane fan-out applies. There is no shrinking retry
/// without a limit: a query-cap rejection returns with `retry_attempts_exhausted` facts.
pub(in crate::exec::bounded) fn execute_final_sort_node_from_frame_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    input_frame: GpuDataFrame,
) -> Result<ResidentGpuFrame> {
    let RetryLimits { max_attempts, .. } = RetryLimits::of(execution.context());
    let timer = Instant::now();
    // Full sorts without a limit stay serial on the coordinator: chunk-local
    // sorts cannot replace a global sort, so no lane fan-out applies here.
    let mut final_sort = FinalSortExecutor { execution, dispatch: None, node_id, sort, max_attempts };
    let result = final_sort.full_sort_limit_attempt_from_frame(None, &input_frame, 1);

    match result {
        Ok(output) => {
            final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
            Ok(ResidentGpuFrame { frame: output.frame, device_bytes: output.device_bytes })
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.execution.metrics());
            final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
            Err(final_sort_retry_facts("retry_attempts_exhausted", final_sort.execution.metrics())
                .attach_to_error(error))
        },
        Err(error) => {
            final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
            Err(error)
        },
    }
}

/// Top-k over a stream of frames, without materializing the whole input.
///
/// Each pulled frame becomes one candidate. The producer is released on every exit, and
/// buffered frames not yet pulled are dropped when the top-k fails. See
/// `hierarchical_topk_stream` for the retry behavior. It is narrower than the frame path
/// because pulled frames are consumed by the attempt that pulled them.
pub(in crate::exec::bounded) fn execute_final_sort_node_from_stream_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    limit: &LimitNode,
    mut input_stream: ResidentFrameStreamPayload,
    dispatch: Option<&AggregateLaneDispatch>,
) -> Result<ResidentGpuFrame> {
    let RetryLimits { max_attempts, .. } = RetryLimits::of(execution.context());
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { execution, dispatch, node_id, sort, max_attempts };
    let result = final_sort.hierarchical_topk_stream(limit, &mut input_stream, 1);
    input_stream.release_producer(final_sort.execution);
    final_sort.execution.metrics().add_final_sort_time(timer.elapsed());
    result
}

/// State shared by the final sort strategies for one node.
///
/// `dispatch` is the compiled run's lane dispatcher. It is `None` when the sort runs outside
/// a lane-capable run, and the top-k then stays serial. `max_attempts` is the context's
/// retry limit, shared by the full-sort rerun and the merge-level retries.
struct FinalSortExecutor<'a, 'e, 'd> {
    execution: &'a mut BoundedExecutionAttempt<'e>,
    dispatch: Option<&'d AggregateLaneDispatch>,
    node_id: NodeId,
    sort: &'a SortNode,
    max_attempts: usize,
}

/// Adds the final sort boundary, attempt numbers and committed-progress counts to a
/// resource failure. Other error classes pass through, and facts already present win, so
/// an inner stage keeps its own location.
fn final_sort_runtime_facts(
    memory_boundary: &'static str,
    node_id: NodeId,
    attempt: usize,
    max_attempts: usize,
    metrics: &ExecutionMetrics,
    error: QueryError,
) -> QueryError {
    RuntimeResourceFacts {
        memory_boundary,
        stage: "final_sort",
        node_id,
        attempt,
        max_attempts,
        resume_boundary: "final_sort",
        metrics,
    }
    .attach_to(error)
}

/// Reads the device bytes of a sorted or truncated frame and wraps it as a partial. A
/// device accounting failure gains the `final_sort_output` boundary facts.
fn observe_final_sort_output(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: GpuDataFrame,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics()).map_err(|error| {
        final_sort_runtime_facts("final_sort_output", node_id, attempt, max_attempts, execution.metrics(), error)
    })?;

    Ok(CommittedPartial::with_device_bytes(frame, output_bytes))
}

impl FinalSortExecutor<'_, '_, '_> {
    /// Budget token labeled `boundary`, sized to the attempt's device budget. The token names
    /// the stage in diagnostics; it does not reserve memory.
    fn final_sort_budget_token(&self, boundary: &'static str, attempt: usize) -> Result<cudf::ExecutionBudgetToken> {
        self.final_sort_budget_token_for_bytes(boundary, attempt, self.execution.device_budget_bytes())
    }

    fn final_sort_budget_token_for_bytes(
        &self,
        boundary: &'static str,
        attempt: usize,
        budget_bytes: u64,
    ) -> Result<cudf::ExecutionBudgetToken> {
        let budget = usize::try_from(budget_bytes).unwrap_or(usize::MAX);
        self.execution
            .context()
            .execution_budget_token(ExecutionBudgetRequest::new(budget).with_boundary(boundary))
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))
    }

    fn observe_final_sort_output(&mut self, frame: GpuDataFrame, attempt: usize) -> Result<CommittedPartial> {
        observe_final_sort_output(self.execution, self.node_id, frame, attempt, self.max_attempts)
    }

    fn attach_final_sort_resource_facts(&self, error: QueryError, attempt: usize) -> QueryError {
        if resource_failure_recovery(&error) == ResourceFailureRecovery::NotResource {
            return error;
        }

        error
            .with_fact("memory_boundary", "final_sort")
            .with_fact("stage", "final_sort")
            .with_fact("node_id", self.node_id)
            .with_fact("attempt", attempt)
            .with_fact("max_attempts", self.max_attempts)
    }

    /// Whether a top-k is cheaper than sorting the whole input for this limit.
    ///
    /// True for a zero-row limit, or when the candidate bound (`skip + fetch` rows at the
    /// input's average row size) is at most half the input bytes. Also true when the input
    /// exceeds the source chunk byte target while the bound fits it. False for an empty
    /// input or a bound covering every row. An offset-only limit has no bound and fails
    /// with `UnsupportedOperator`.
    fn should_use_hierarchical_topk(&self, limit: &LimitNode, input: &CommittedPartial) -> Result<bool> {
        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            return Ok(true);
        }

        let input_rows = input.frame.num_rows();
        if input_rows == 0 || keep_rows >= input_rows {
            return Ok(false);
        }

        let input_bytes = input.device_bytes();
        let limited_output_bytes = estimate_rows_bytes(estimated_bytes_per_row(input_bytes, input_rows), keep_rows);
        let selective_topk_threshold = input_bytes.div_ceil(2).max(1);
        if limited_output_bytes <= selective_topk_threshold {
            return Ok(true);
        }

        let target = SourceChunkPlanningPolicy::from(self.execution.context().memory_control_policy())
            .max_source_chunk_bytes
            .expect("bounded source planning has a finite byte target");
        Ok(input_bytes > target && limited_output_bytes <= target)
    }
}
