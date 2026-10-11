use super::{
    AggregateLaneDispatch, CommittedPartial, ExecutionBudgetToken, FinalSortExecutor, GpuDataFrame, LimitNode,
    LimitSpec, QueryError, QueryErrorCode, ResidentFrameStreamPayload, ResidentGpuFrame, Result, SortLaneCandidate,
    TopKLevelItem, commit_topk_merge_level, concat_committed_frames_with_token, cudf_query_error,
    estimated_bytes_per_row, execute_limit_on_frame_with_token, execute_limit_spec_with_token,
    execute_limit_with_token, execute_sort_lane_candidate, final_sort_retry_facts, final_sort_rows_per_chunk,
    final_sort_runtime_facts, final_sort_topk_retry_facts, is_retryable_final_sort_error,
    record_final_sort_retry_error, reduce_merge_fan_in, sort_lane_fan_out, sort_lane_params,
    sort_topk_candidate_with_budget, split_sort_candidate_ranges, stream_candidate_chunk_hint, topk_candidate_rows,
};
use crate::exec::{operators::attach_budget_token_failure_facts, retry::backoff_before_retry};

/// Candidate row range sizing for the serial first pass.
#[derive(Clone, Copy)]
struct TopKCandidatePlan {
    keep_rows: usize,
    rows_per_chunk: usize,
}

#[derive(Clone, Copy)]
struct TopKSliceAttempt {
    offset: usize,
    length: usize,
    keep_rows: usize,
    attempt: usize,
}

#[derive(Clone, Copy)]
struct TopKMergeAttempt {
    keep_rows: usize,
    merge_fan_in: usize,
    attempt: usize,
}

/// Coordinator state for a frame fan-out: the unassigned row ranges, a shared clone of the
/// parent frame, and candidates in commit order.
struct FrameCandidateState {
    ranges: std::vec::IntoIter<(usize, usize)>,
    parent: GpuDataFrame,
    parent_bytes: u64,
    candidates: Vec<CommittedPartial>,
}

/// Coordinator state for a stream fan-out. Pulls stay on the coordinator because the
/// payload has a single cursor.
struct StreamCandidateState<'p> {
    payload: &'p mut ResidentFrameStreamPayload,
    candidates: Vec<CommittedPartial>,
}

/// How a failure to create a candidate's sort token is reported: as the token error itself
/// (`Direct`) or wrapped with the final sort runtime facts (`RuntimeFacts`).
#[derive(Clone, Copy)]
enum TopKWorkTokenFailureMode {
    Direct,
    RuntimeFacts,
}

impl FinalSortExecutor<'_, '_, '_> {
    /// Global top-k of `input`: sort row ranges into candidates, merge candidates level by
    /// level, then apply the limit.
    ///
    /// Every candidate holds the first `skip + fetch` rows of its inputs under the sort
    /// order. The last candidate therefore holds the global top rows, and the closing limit
    /// applies the offset and fetch. A zero-row limit returns an empty frame without sorting,
    /// and an empty input returns itself.
    ///
    /// Candidates come from ranges of `rows_per_chunk` rows, sized from the device budget left
    /// after `input`. At `initial_attempt == 1` with two or more lanes and two or more
    /// ranges, the ranges run concurrently, each sized against a per-lane share of that
    /// remainder. If the fan-out fails with a query-cap rejection and `max_attempts` exceeds
    /// 1, one backoff precedes a serial rerun of all ranges at attempt 2. A serial failure
    /// has nothing narrower to try and returns with `hierarchical_topk_initial_failed` facts.
    ///
    /// Each merge level concatenates groups of `merge_fan_in` candidates on the coordinator,
    /// sorts and truncates each group, and carries a trailing group of one unchanged. A
    /// query-cap rejection with attempts left halves the fan-in (rounded up, floor 2) and
    /// redoes the level from the same candidates; otherwise the error returns with
    /// `final_sort_topk_retry_facts` naming the level and fan-in. A failed level leaves the
    /// candidates intact and drops the groups it had merged; a successful level drops the
    /// candidates it consumed. Merge attempts count from `initial_attempt` against
    /// `max_attempts`; the rerun after a failed full sort (`initial_attempt` 2) has fewer
    /// reductions available.
    ///
    /// The closing limit has no retry; its failures report `attempt == max_attempts`.
    /// `input` stays allocated until this function returns, through every merge level.
    pub(super) fn hierarchical_topk(
        &mut self,
        limit: &LimitNode,
        input: CommittedPartial,
        initial_attempt: usize,
    ) -> Result<CommittedPartial> {
        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            let token = match self.final_sort_budget_token("final_sort_limit", initial_attempt) {
                Ok(token) => token,
                Err(error) => {
                    return Err(error);
                },
            };
            let output = match execute_limit_on_frame_with_token(
                limit,
                input.frame.frame().clone(),
                &token,
                "query_engine.final_sort.limit",
                self.execution.metrics(),
            ) {
                Ok(output) => output,
                Err(error) => {
                    return Err(error);
                },
            };
            return self.observe_final_sort_output(output, initial_attempt);
        }

        let input_rows = input.frame.num_rows();
        if input_rows == 0 {
            return Ok(input);
        }
        let bytes_per_row = estimated_bytes_per_row(input.device_bytes(), input_rows);
        let available = self.execution.device_budget_bytes().saturating_sub(input.device_bytes());
        let rows_per_chunk = final_sort_rows_per_chunk(input_rows, bytes_per_row, available);
        let ranges = split_sort_candidate_ranges(input_rows, rows_per_chunk);
        // Size fan-out chunks against a per-lane share of the remaining
        // budget: each lane sorts its chunk concurrently, so the serial
        // single-chunk sizing would admit roughly `width` times the live
        // work. The serial path below keeps the original sizing.
        let (ranges, rows_per_chunk) = if sort_lane_fan_out(self.dispatch, ranges.len(), initial_attempt) {
            let width = self.dispatch.expect("sort lane fan-out requires a lane dispatcher").width().max(1);
            let rows_per_chunk = final_sort_rows_per_chunk(input_rows, bytes_per_row, available / width as u64);
            (split_sort_candidate_ranges(input_rows, rows_per_chunk), rows_per_chunk)
        } else {
            (ranges, rows_per_chunk)
        };
        let initial: Result<Vec<CommittedPartial>> = if sort_lane_fan_out(self.dispatch, ranges.len(), initial_attempt)
        {
            let dispatch = self.dispatch.expect("sort lane fan-out requires a lane dispatcher");
            match self.lane_frame_candidates(dispatch, &input, keep_rows, ranges, initial_attempt) {
                Ok(candidates) => Ok(candidates),
                // The frame input is intact (candidates borrowed it), so a
                // retryable fan-out failure retries serially at attempt 2,
                // which keeps one chunk live where the fan-out held `width`.
                // A serial attempt-1 failure has nothing narrower to fall
                // back to and fails directly through the shared wrapper below.
                Err(error)
                    if is_retryable_final_sort_error(&error) && initial_attempt <= 1 && self.max_attempts > 1 =>
                {
                    record_final_sort_retry_error(&error, self.execution.metrics());
                    backoff_before_retry(self.execution.context(), "query_engine.exec.bounded.final_sort_lane_retry")?;
                    self.initial_topk_candidates(&input, TopKCandidatePlan { keep_rows, rows_per_chunk }, 2)
                },
                Err(error) => Err(error),
            }
        } else {
            self.initial_topk_candidates(&input, TopKCandidatePlan { keep_rows, rows_per_chunk }, initial_attempt)
        };
        let mut candidates = match initial {
            Ok(candidates) => candidates,
            Err(error) => {
                record_final_sort_retry_error(&error, self.execution.metrics());
                return Err(final_sort_retry_facts("hierarchical_topk_initial_failed", self.execution.metrics())
                    .attach_to_error(error));
            },
        };

        let mut merge_level = 1;
        while candidates.len() > 1 {
            let mut merge_fan_in = candidates.len();
            let mut attempt = initial_attempt;

            loop {
                match self.topk_merge_level_attempt(&candidates, TopKMergeAttempt { keep_rows, merge_fan_in, attempt })
                {
                    Ok(level_output) => {
                        candidates = commit_topk_merge_level(candidates, level_output);
                        merge_level += 1;
                        break;
                    },
                    Err(error) if is_retryable_final_sort_error(&error) => {
                        record_final_sort_retry_error(&error, self.execution.metrics());
                        if attempt < self.max_attempts
                            && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                        {
                            merge_fan_in = reduced_fan_in;
                            attempt += 1;
                            backoff_before_retry(
                                self.execution.context(),
                                "query_engine.exec.bounded.hierarchical_topk_merge_retry",
                            )?;
                            continue;
                        }

                        return Err(final_sort_topk_retry_facts(merge_level, merge_fan_in, self.execution.metrics())
                            .attach_to_error(error));
                    },
                    Err(error) => {
                        return Err(error);
                    },
                }
            }
        }

        let candidate = candidates.pop().expect("hierarchical top-k leaves exactly one candidate");
        if limit.spec.skip == 0 && limit.spec.fetch.is_some_and(|fetch| candidate.frame.num_rows() <= fetch) {
            return Ok(candidate);
        }

        let token = match self.final_sort_budget_token("final_sort_limit", self.max_attempts) {
            Ok(token) => token,
            Err(error) => {
                return Err(error);
            },
        };
        let output = match execute_limit_with_token(
            limit,
            candidate.frame,
            &token,
            "query_engine.final_sort.limit",
            self.execution.metrics(),
        ) {
            Ok(output) => output,
            Err(error) => {
                return Err(final_sort_runtime_facts(
                    "final_sort_limit",
                    self.node_id,
                    self.max_attempts,
                    self.max_attempts,
                    self.execution.metrics(),
                    error,
                ));
            },
        };
        self.observe_final_sort_output(output, self.max_attempts)
    }

    /// Sorts and truncates one frame's row ranges on execution lanes.
    ///
    /// Each lane slices its range on its own stream from the shared parent
    /// clone, so no cross-stream edge leaves the coordinator. Commit order is
    /// the stable chunk order, and every uncommitted output drains through
    /// release accounting when the fan-out fails.
    fn lane_frame_candidates(
        &mut self,
        dispatch: &AggregateLaneDispatch,
        input: &CommittedPartial,
        keep_rows: usize,
        ranges: Vec<(usize, usize)>,
        attempt: usize,
    ) -> Result<Vec<CommittedPartial>> {
        let params = sort_lane_params(self.sort, self.node_id, keep_rows, attempt, self.max_attempts);
        let parent = input.frame.clone();
        let parent_bytes = input.device_bytes();
        let mut state =
            FrameCandidateState { ranges: ranges.into_iter(), parent, parent_bytes, candidates: Vec::new() };
        dispatch.execute_work(
            self.execution,
            &mut state,
            |state, _| {
                Ok(state.ranges.next().map(|(offset, length)| SortLaneCandidate {
                    input: ResidentGpuFrame::with_device_bytes(state.parent.clone(), state.parent_bytes),
                    slice: Some((offset, length)),
                    boundary: "final_sort_topk_chunk",
                }))
            },
            move |execution, candidate| execute_sort_lane_candidate(execution, &params, candidate),
            |state, _, candidate| {
                state.candidates.push(candidate);
                Ok(())
            },
        )?;
        Ok(state.candidates)
    }

    /// Sorts and truncates one stream's frames on execution lanes.
    ///
    /// Pulling stays serial on the coordinator behind the payload's single
    /// cursor contract; only the owned pulled frames move to lanes. Lane
    /// failures drain through release accounting and leave unpulled frames
    /// in the payload for the caller to release.
    fn lane_stream_candidates(
        &mut self,
        dispatch: &AggregateLaneDispatch,
        inputs: &mut ResidentFrameStreamPayload,
        keep_rows: usize,
        attempt: usize,
    ) -> Result<Vec<CommittedPartial>> {
        let params = sort_lane_params(self.sort, self.node_id, keep_rows, attempt, self.max_attempts);
        let mut state = StreamCandidateState { payload: inputs, candidates: Vec::new() };
        dispatch.execute_work(
            self.execution,
            &mut state,
            |state, execution| match state.payload.next_frame(execution) {
                Some(Ok(input)) => {
                    Ok(Some(SortLaneCandidate { input, slice: None, boundary: "final_sort_topk_stream_chunk" }))
                },
                Some(Err(error)) => Err(error),
                None => Ok(None),
            },
            move |execution, candidate| execute_sort_lane_candidate(execution, &params, candidate),
            |state, _, candidate| {
                state.candidates.push(candidate);
                Ok(())
            },
        )?;
        Ok(state.candidates)
    }

    /// Serial first pass: sorts and truncates each `rows_per_chunk` range of `input` in order
    /// on the coordinator, holding one sorted range at a time besides the finished candidates.
    /// The first error returns and the finished candidates drop.
    fn initial_topk_candidates(
        &mut self,
        input: &CommittedPartial,
        plan: TopKCandidatePlan,
        attempt: usize,
    ) -> Result<Vec<CommittedPartial>> {
        let input_rows = input.frame.num_rows();
        let mut candidates = Vec::with_capacity(input_rows.div_ceil(plan.rows_per_chunk));
        let mut offset = 0;
        while offset < input_rows {
            let length = plan.rows_per_chunk.min(input_rows - offset);
            match self
                .topk_slice_attempt(input, TopKSliceAttempt { offset, length, keep_rows: plan.keep_rows, attempt })
            {
                Ok(candidate) => candidates.push(candidate),
                Err(error) => {
                    return Err(error);
                },
            }
            offset += length;
        }
        Ok(candidates)
    }

    /// Top-k over pulled frames: every non-empty frame becomes one candidate. Candidate size
    /// follows the producer and nothing re-chunks it. Empty frames pass through unsorted.
    ///
    /// Merge levels and the closing limit match `hierarchical_topk`. The first pass has no
    /// retry: pulled frames are consumed by the pass that pulled them. A failure drops
    /// the buffered frames and returns with `hierarchical_topk_stream_initial_failed` facts.
    /// With lanes and `initial_attempt == 1` the pulled frames run on lanes, and an unknown
    /// frame count is treated as two or more. A payload with nothing to pull, or one that
    /// yields no frame, is `InvalidPlan`. The caller releases the producer.
    pub(super) fn hierarchical_topk_stream(
        &mut self,
        limit: &LimitNode,
        inputs: &mut ResidentFrameStreamPayload,
        initial_attempt: usize,
    ) -> Result<CommittedPartial> {
        if inputs.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.final_sort.topk_stream",
                "compiled final sort stream requires at least one input frame",
            )
            .with_fact("node_id", self.node_id));
        }

        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            return self.zero_limit_stream(limit, inputs, initial_attempt);
        }

        // Buffered frames are exact for eager payloads; zero buffered frames
        // behind a live producer means the count is unknown, so the gate
        // assumes multi-frame and the dispatcher narrows to one job when a
        // single frame arrives. That lone lane output merges unmerged on the
        // coordinator while the run's leases stay held, which keeps its
        // lifetime correct on every sink.
        let chunk_hint = stream_candidate_chunk_hint(inputs.len());
        let initial: Result<Vec<CommittedPartial>> = if sort_lane_fan_out(self.dispatch, chunk_hint, initial_attempt) {
            let dispatch = self.dispatch.expect("sort lane fan-out requires a lane dispatcher");
            self.lane_stream_candidates(dispatch, inputs, keep_rows, initial_attempt)
        } else {
            let mut candidates = Vec::with_capacity(inputs.len());
            while let Some(input) = inputs.next_frame(self.execution) {
                let input = match input {
                    Ok(input) => input,
                    Err(error) => {
                        inputs.release_remaining_frames();
                        return Err(error);
                    },
                };
                match self.topk_stream_frame_candidate_attempt(input, keep_rows, initial_attempt) {
                    Ok(candidate) => candidates.push(candidate),
                    Err(error) => {
                        inputs.release_remaining_frames();
                        return Err(error);
                    },
                }
            }
            Ok(candidates)
        };
        let mut candidates = match initial {
            Ok(candidates) => candidates,
            Err(error) => {
                inputs.release_remaining_frames();
                return Err(final_sort_retry_facts(
                    "hierarchical_topk_stream_initial_failed",
                    self.execution.metrics(),
                )
                .attach_to_error(error));
            },
        };

        if candidates.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.final_sort.topk_stream",
                "compiled final sort stream produced no input frames",
            )
            .with_fact("node_id", self.node_id));
        }

        let mut merge_level = 1;
        while candidates.len() > 1 {
            let mut merge_fan_in = candidates.len();
            let mut attempt = initial_attempt;

            loop {
                match self.topk_merge_level_attempt(&candidates, TopKMergeAttempt { keep_rows, merge_fan_in, attempt })
                {
                    Ok(level_output) => {
                        candidates = commit_topk_merge_level(candidates, level_output);
                        merge_level += 1;
                        break;
                    },
                    Err(error) if is_retryable_final_sort_error(&error) => {
                        record_final_sort_retry_error(&error, self.execution.metrics());
                        if attempt < self.max_attempts
                            && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                        {
                            merge_fan_in = reduced_fan_in;
                            attempt += 1;
                            backoff_before_retry(
                                self.execution.context(),
                                "query_engine.exec.bounded.hierarchical_topk_stream_merge_retry",
                            )?;
                            continue;
                        }

                        return Err(final_sort_topk_retry_facts(merge_level, merge_fan_in, self.execution.metrics())
                            .attach_to_error(error));
                    },
                    Err(error) => {
                        return Err(error);
                    },
                }
            }
        }

        let candidate = candidates.pop().expect("stream hierarchical top-k leaves exactly one candidate");
        if limit.spec.skip == 0 && limit.spec.fetch.is_some_and(|fetch| candidate.frame.num_rows() <= fetch) {
            return Ok(candidate);
        }

        self.apply_final_limit(candidate, limit, self.max_attempts)
    }

    /// A zero-row limit over a stream: limits the first frame to keep the output schema and
    /// drops the other buffered frames. The caller releases the producer.
    fn zero_limit_stream(
        &mut self,
        limit: &LimitNode,
        inputs: &mut ResidentFrameStreamPayload,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let first = match inputs.next_frame(self.execution) {
            Some(Ok(input)) => input,
            Some(Err(error)) => {
                inputs.release_remaining_frames();
                return Err(error);
            },
            None => {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.final_sort.topk_stream",
                    "compiled final sort stream produced no input frames",
                )
                .with_fact("node_id", self.node_id));
            },
        };
        let token = match self.final_sort_budget_token("final_sort_limit", attempt) {
            Ok(token) => token,
            Err(error) => {
                inputs.release_remaining_frames();
                return Err(error);
            },
        };
        let output = match execute_limit_on_frame_with_token(
            limit,
            first.frame.into_frame(),
            &token,
            "query_engine.final_sort.limit",
            self.execution.metrics(),
        ) {
            Ok(output) => output,
            Err(error) => {
                inputs.release_remaining_frames();
                return Err(error);
            },
        };
        inputs.release_remaining_frames();
        self.observe_final_sort_output(output, attempt)
    }

    /// Candidate for one row range of a retained frame, sliced on the coordinator stream under
    /// the sort's token.
    fn topk_slice_attempt(&mut self, input: &CommittedPartial, slice: TopKSliceAttempt) -> Result<CommittedPartial> {
        const BOUNDARY: &str = "final_sort_topk_chunk";
        self.sort_and_limit_topk_candidate(
            BOUNDARY,
            slice.attempt,
            slice.keep_rows,
            TopKWorkTokenFailureMode::Direct,
            |work_token| {
                input
                    .frame
                    .frame()
                    .clone()
                    .on(work_token.exec())
                    .slice(slice.offset, slice.length)
                    .map(GpuDataFrame::new)
                    .map_err(|error| {
                        attach_budget_token_failure_facts(
                            cudf_query_error(
                                "query_engine.final_sort.topk_slice",
                                "failed to slice bounded native final sort input",
                                error,
                            ),
                            work_token,
                        )
                    })
            },
        )
    }

    fn topk_stream_frame_candidate_attempt(
        &mut self,
        input: ResidentGpuFrame,
        keep_rows: usize,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let input_rows = input.frame.num_rows();
        if input_rows == 0 {
            return Ok(input);
        }

        const BOUNDARY: &str = "final_sort_topk_stream_chunk";
        let ResidentGpuFrame { frame, .. } = input;
        self.sort_and_limit_topk_candidate(
            BOUNDARY,
            attempt,
            keep_rows,
            TopKWorkTokenFailureMode::Direct,
            move |_work_token| Ok(frame),
        )
    }

    /// Sorts the frame produced by `frame_for_sort` and truncates it to `keep_rows`.
    ///
    /// `frame_for_sort` runs under the sort's budget token; a slice or concatenation is
    /// attributed to that stage. The truncation uses a second token. A sort failure gains the
    /// final sort runtime facts for `memory_boundary`; `work_token_failure_mode` decides the
    /// same for a failure to create the sort token. A limit-token failure returns as is.
    fn sort_and_limit_topk_candidate(
        &mut self,
        memory_boundary: &'static str,
        attempt: usize,
        keep_rows: usize,
        work_token_failure_mode: TopKWorkTokenFailureMode,
        frame_for_sort: impl FnOnce(&ExecutionBudgetToken) -> Result<GpuDataFrame>,
    ) -> Result<CommittedPartial> {
        let sorted = {
            let work_token = match self.final_sort_budget_token(memory_boundary, attempt) {
                Ok(token) => token,
                Err(error) => match work_token_failure_mode {
                    TopKWorkTokenFailureMode::Direct => {
                        return Err(error);
                    },
                    TopKWorkTokenFailureMode::RuntimeFacts => {
                        return Err(final_sort_runtime_facts(
                            memory_boundary,
                            self.node_id,
                            attempt,
                            self.max_attempts,
                            self.execution.metrics(),
                            error,
                        ));
                    },
                },
            };
            frame_for_sort(&work_token).and_then(|frame| {
                sort_topk_candidate_with_budget(
                    self.sort,
                    frame,
                    &work_token,
                    memory_boundary,
                    self.execution.metrics(),
                )
            })
        };
        let sorted = match sorted {
            Ok(frame) => frame,
            Err(error) => {
                return Err(final_sort_runtime_facts(
                    memory_boundary,
                    self.node_id,
                    attempt,
                    self.max_attempts,
                    self.execution.metrics(),
                    error,
                ));
            },
        };
        let limit_token = match self.final_sort_budget_token(memory_boundary, attempt) {
            Ok(token) => token,
            Err(error) => {
                return Err(error);
            },
        };
        let frame = match execute_limit_spec_with_token(
            LimitSpec::fetch(keep_rows),
            sorted,
            &limit_token,
            "query_engine.final_sort.topk_limit",
            self.execution.metrics(),
        ) {
            Ok(frame) => frame,
            Err(error) => {
                return Err(final_sort_runtime_facts(
                    memory_boundary,
                    self.node_id,
                    attempt,
                    self.max_attempts,
                    self.execution.metrics(),
                    error,
                ));
            },
        };
        self.observe_final_sort_output(frame, attempt)
    }

    /// Merges `candidates` in groups of `merge.merge_fan_in` and returns one item per group:
    /// `Merged` for two or more candidates, `Carry` naming the candidate of a group of one.
    ///
    /// The slice is not consumed. `commit_topk_merge_level` applies the result after a full
    /// level succeeds. An error returns at once, dropping the groups already merged.
    fn topk_merge_level_attempt(
        &mut self,
        candidates: &[CommittedPartial],
        merge: TopKMergeAttempt,
    ) -> Result<Vec<TopKLevelItem>> {
        let mut output = Vec::with_capacity(candidates.len().div_ceil(merge.merge_fan_in));

        for (group_index, group) in candidates.chunks(merge.merge_fan_in).enumerate() {
            if group.len() == 1 {
                output.push(TopKLevelItem::Carry { partial_index: group_index * merge.merge_fan_in });
                continue;
            }

            match self.topk_merge_group_attempt(
                group,
                TopKMergeAttempt { keep_rows: merge.keep_rows, merge_fan_in: group.len(), attempt: merge.attempt },
            ) {
                Ok(candidate) => output.push(TopKLevelItem::Merged(Box::new(candidate))),
                Err(error) => {
                    return Err(error);
                },
            }
        }

        Ok(output)
    }

    /// Concatenates one group on the coordinator stream, then sorts and truncates it. The
    /// group's candidates stay alive during the concatenation and the sort; a group of
    /// `n` candidates holds the inputs, the concatenation and the sorted copy.
    fn topk_merge_group_attempt(
        &mut self,
        candidates: &[CommittedPartial],
        merge: TopKMergeAttempt,
    ) -> Result<CommittedPartial> {
        debug_assert!(candidates.len() > 1);
        const BOUNDARY: &str = "final_sort_topk_merge";
        self.sort_and_limit_topk_candidate(
            BOUNDARY,
            merge.attempt,
            merge.keep_rows,
            TopKWorkTokenFailureMode::RuntimeFacts,
            |work_token| {
                concat_committed_frames_with_token(candidates, work_token, "query_engine.final_sort.topk_concat")
            },
        )
    }

    /// Applies the query's skip and fetch to the last candidate. No retry runs here.
    fn apply_final_limit(
        &mut self,
        candidate: CommittedPartial,
        limit: &LimitNode,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let token = match self.final_sort_budget_token("final_sort_limit", attempt) {
            Ok(token) => token,
            Err(error) => {
                return Err(error);
            },
        };
        let output = match execute_limit_with_token(
            limit,
            candidate.frame,
            &token,
            "query_engine.final_sort.limit",
            self.execution.metrics(),
        ) {
            Ok(output) => output,
            Err(error) => {
                return Err(final_sort_runtime_facts(
                    "final_sort_limit",
                    self.node_id,
                    attempt,
                    self.max_attempts,
                    self.execution.metrics(),
                    error,
                ));
            },
        };
        self.observe_final_sort_output(output, attempt)
    }
}
