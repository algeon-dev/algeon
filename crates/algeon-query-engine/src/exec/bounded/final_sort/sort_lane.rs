//! Lane-parallel local top-k candidates for final sort.
//!
//! A global top-k over one total order is the merge of per-chunk top-k
//! candidates: any row of the global top-k is in its own chunk's top-k, and
//! truncating a candidate set to its top-k never removes a global top-k row.
//! Every chunk and every merge level therefore sorts with the same comparator
//! (per-key direction and per-key null precedence), keeps `skip + fetch` rows,
//! and applies the offset only in the final limit. Concatenating locally
//! sorted chunks without the merge is never a global sort.
//!
//! Each candidate sorts and truncates on its own execution lane stream from
//! an owned input, so lanes never share coordinator state and no lane waits
//! on another lane's stream. The candidate's readiness event travels with the
//! lane output through the shared lane-work dispatcher; the coordinator
//! observes it before commit, which orders the merge after every candidate's
//! GPU work without host-synchronizing the lanes. The bounded fan-in merge
//! stays on the coordinator behind the existing retry contract: a retryable
//! failure retries serially, which strictly shrinks simultaneous live work,
//! while the merge loop keeps reducing its fan-in.

use std::sync::Arc;

use super::{
    BoundedExecutionAttempt, CommittedPartial, ExecutionBudgetRequest, GpuDataFrame, LimitSpec, NodeId,
    ResidentGpuFrame, Result, SortNode, cudf_query_error_with_budget_details, execute_limit_spec_with_token,
    final_sort_runtime_facts, observe_final_sort_output, sort_topk_candidate_with_budget,
};
use crate::{
    error::QueryError,
    exec::{
        bounded::compiled::lanes::AggregateLaneDispatch,
        retry::{ResourceFailureRecovery, resource_failure_recovery},
    },
};

/// Owned sort identity shared by every lane candidate of one fan-out.
pub(super) struct SortLaneParams {
    pub(super) sort: SortNode,
    pub(super) node_id: NodeId,
    pub(super) keep_rows: usize,
    pub(super) attempt: usize,
    pub(super) max_attempts: usize,
}

/// One lane worker's owned candidate input.
///
/// Frame-path chunks clone the shared parent frame and slice their row range
/// on the lane's own stream, so the slice copy and the sort share one stream
/// and no cross-stream edge leaves the coordinator. Stream-path candidates
/// carry an already independent frame with no slice.
pub(super) struct SortLaneCandidate {
    pub(super) input: ResidentGpuFrame,
    pub(super) slice: Option<(usize, usize)>,
    pub(super) boundary: &'static str,
}

/// Whether one top-k may fan its `chunk_count` independent candidates out to
/// lanes.
///
/// Ready work, not lane capacity, decides first: a single chunk stays serial.
/// A retryable failure retries serially (`attempt > 1`), which strictly
/// shrinks simultaneous live work while the merge loop keeps reducing its
/// fan-in. This is the small cost/batch policy for sort width: no learned
/// scheduler, no tenant policy, no dynamic grant.
pub(super) fn sort_lane_fan_out(dispatch: Option<&AggregateLaneDispatch>, chunk_count: usize, attempt: usize) -> bool {
    attempt <= 1 && chunk_count >= 2 && dispatch.is_some_and(|dispatch| dispatch.width() >= 2)
}

/// Row ranges splitting `input_rows` into `rows_per_chunk` chunks.
///
/// The last chunk takes the remainder, so no ordinal is skipped or repeated.
/// Pure arithmetic: covered by CPU unit tests, no GPU needed.
pub(super) fn split_sort_candidate_ranges(input_rows: usize, rows_per_chunk: usize) -> Vec<(usize, usize)> {
    let rows_per_chunk = rows_per_chunk.max(1);
    let mut ranges = Vec::new();
    let mut offset = 0;
    while offset < input_rows {
        let length = rows_per_chunk.min(input_rows - offset);
        ranges.push((offset, length));
        offset += length;
    }
    ranges
}

/// Sorts and truncates one owned candidate to `keep_rows` on this lane.
///
/// Mirrors the serial per-chunk candidate exactly: same memory boundary,
/// same budget token boundary, same comparator, same `fetch(keep_rows)`
/// truncation with the offset left for the final limit. An empty input passes
/// through untouched, matching the serial stream path.
pub(super) fn execute_sort_lane_candidate(
    execution: &mut BoundedExecutionAttempt<'_>,
    params: &SortLaneParams,
    candidate: SortLaneCandidate,
) -> Result<CommittedPartial> {
    const OPERATION: &str = "query_engine.exec.bounded.final_sort_lane_candidate";
    execution.context().check_cancellation(OPERATION)?;
    let input_rows = match candidate.slice {
        Some((_, length)) => length,
        None => candidate.input.frame.num_rows(),
    };
    if input_rows == 0 {
        return Ok(candidate.input);
    }
    let work_token = lane_candidate_budget_token(execution, params, candidate.boundary)?;
    let frame = lane_candidate_frame(&candidate, &work_token)?;
    let sorted = match sort_topk_candidate_with_budget(
        &params.sort,
        frame,
        &work_token,
        candidate.boundary,
        execution.metrics(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(final_sort_runtime_facts(
                candidate.boundary,
                params.node_id,
                params.attempt,
                params.max_attempts,
                execution.metrics(),
                error,
            ));
        },
    };
    let limit_token = lane_candidate_budget_token(execution, params, candidate.boundary)?;
    let frame = match execute_limit_spec_with_token(
        LimitSpec::fetch(params.keep_rows),
        sorted,
        &limit_token,
        "query_engine.final_sort.topk_limit",
        execution.metrics(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(final_sort_runtime_facts(
                candidate.boundary,
                params.node_id,
                params.attempt,
                params.max_attempts,
                execution.metrics(),
                error,
            ));
        },
    };
    observe_final_sort_output(execution, params.node_id, frame, params.attempt, params.max_attempts)
}

/// Ready-work hint for a frame stream: buffered frames are exact for eager
/// payloads, while zero buffered frames behind a live producer means the
/// count is unknown. The dispatcher narrows to one job when a single frame
/// arrives, so the unknown case assumes multi-frame optimistically.
pub(super) fn stream_candidate_chunk_hint(buffered_frames: usize) -> usize {
    if buffered_frames == 0 { 2 } else { buffered_frames }
}

/// Shares one fan-out's sort identity across lane workers.
pub(super) fn sort_lane_params(
    sort: &SortNode,
    node_id: NodeId,
    keep_rows: usize,
    attempt: usize,
    max_attempts: usize,
) -> Arc<SortLaneParams> {
    Arc::new(SortLaneParams { sort: sort.clone(), node_id, keep_rows, attempt, max_attempts })
}

fn lane_candidate_budget_token(
    execution: &BoundedExecutionAttempt<'_>,
    params: &SortLaneParams,
    boundary: &'static str,
) -> Result<cudf::ExecutionBudgetToken> {
    let budget = usize::try_from(execution.device_budget_bytes()).unwrap_or(usize::MAX);
    execution
        .context()
        .execution_budget_token(ExecutionBudgetRequest::new(budget).with_boundary(boundary))
        .map_err(|error| attach_lane_candidate_resource_facts(error, params))
}

fn lane_candidate_frame(
    candidate: &SortLaneCandidate,
    work_token: &cudf::ExecutionBudgetToken,
) -> Result<GpuDataFrame> {
    let Some((offset, length)) = candidate.slice else {
        return Ok(candidate.input.frame.clone());
    };
    candidate
        .input
        .frame
        .frame()
        .clone()
        .with_execution(&work_token.execution_overrides())
        .slice(offset, length)
        .map(GpuDataFrame::new)
        .map_err(|error| {
            cudf_query_error_with_budget_details(
                "query_engine.final_sort.topk_slice",
                "failed to slice bounded native final sort input",
                error,
            )
            .with_fact_if_absent("budget_boundary", work_token.request().boundary())
            .with_fact_if_absent("budget_bytes", work_token.request().budget_bytes())
            .with_fact_if_absent("budget_operation", "dataframe::slice")
        })
}

fn attach_lane_candidate_resource_facts(error: QueryError, params: &SortLaneParams) -> QueryError {
    if resource_failure_recovery(&error) == ResourceFailureRecovery::NotResource {
        return error;
    }
    error
        .with_fact("memory_boundary", "final_sort")
        .with_fact("stage", "final_sort")
        .with_fact("node_id", params.node_id)
        .with_fact("attempt", params.attempt)
        .with_fact("max_attempts", params.max_attempts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_ranges_cover_every_row_once() {
        assert!(split_sort_candidate_ranges(0, 8).is_empty());
        assert_eq!(split_sort_candidate_ranges(5, 8), vec![(0, 5)]);
        assert_eq!(split_sort_candidate_ranges(8, 8), vec![(0, 8)]);
        assert_eq!(split_sort_candidate_ranges(10, 8), vec![(0, 8), (8, 2)]);
        assert_eq!(split_sort_candidate_ranges(16, 8), vec![(0, 8), (8, 8)]);
        assert_eq!(split_sort_candidate_ranges(3, 0), vec![(0, 1), (1, 1), (2, 1)]);
    }

    #[test]
    fn fan_out_needs_ready_work_width_and_first_attempt() {
        assert!(!sort_lane_fan_out(None, 4, 1));
        assert!(!sort_lane_fan_out(None, 1, 1));
        assert!(!sort_lane_fan_out(None, 4, 2));
    }
}
