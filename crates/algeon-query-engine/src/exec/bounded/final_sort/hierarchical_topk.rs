use super::{
    AggregateLaneDispatch, CommittedPartial, ExecutionBudgetToken, FinalSortExecutor, GpuDataFrame, LimitNode,
    LimitSpec, QueryError, QueryErrorCode, ResidentFrameStreamPayload, ResidentGpuFrame, Result, SortLaneCandidate,
    TopKLevelItem, commit_topk_merge_level, concat_committed_frames_with_token, cudf_query_error_with_budget_details,
    estimate_rows_bytes, estimated_bytes_per_row, estimated_limited_output_bytes, execute_limit_on_frame_with_token,
    execute_limit_spec_with_token, execute_limit_with_token, execute_sort_lane_candidate, final_sort_retry_facts,
    final_sort_rows_per_chunk, final_sort_runtime_facts, final_sort_topk_retry_facts, is_retryable_final_sort_error,
    record_final_sort_retry_error, reduce_merge_fan_in, sort_lane_fan_out, sort_lane_params,
    sort_topk_candidate_with_budget, split_sort_candidate_ranges, stream_candidate_chunk_hint, topk_candidate_rows,
};

#[derive(Clone, Copy)]
struct TopKCandidatePlan {
    keep_rows: usize,
    bytes_per_row: u64,
    rows_per_chunk: usize,
}

#[derive(Clone, Copy)]
struct TopKSliceAttempt {
    offset: usize,
    length: usize,
    keep_rows: usize,
    bytes_per_row: u64,
    attempt: usize,
}

#[derive(Clone, Copy)]
struct TopKMergeAttempt {
    keep_rows: usize,
    merge_fan_in: usize,
    attempt: usize,
}

struct FrameCandidateState {
    ranges: std::vec::IntoIter<(usize, usize)>,
    parent: GpuDataFrame,
    parent_bytes: u64,
    candidates: Vec<CommittedPartial>,
}

struct StreamCandidateState<'p> {
    payload: &'p mut ResidentFrameStreamPayload,
    candidates: Vec<CommittedPartial>,
}

#[derive(Clone, Copy)]
enum TopKWorkTokenFailureMode {
    Direct,
    RuntimeFacts,
}

impl FinalSortExecutor<'_, '_, '_> {
    pub(super) fn hierarchical_topk(
        &mut self,
        limit: &LimitNode,
        input: CommittedPartial,
        initial_attempt: usize,
    ) -> Result<CommittedPartial> {
        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            let output_reservation = match self.reserve_estimated_final_sort_output(0, initial_attempt) {
                Ok(reservation) => reservation,
                Err(error) => {
                    return Err(error);
                },
            };
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
            return self.finalize_final_sort_output_reservation(output, output_reservation, initial_attempt);
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
        // work. The serial path below keeps the original sizing exactly.
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
            match self.lane_frame_candidates(dispatch, &input, keep_rows, bytes_per_row, ranges, initial_attempt) {
                Ok(candidates) => Ok(candidates),
                // The frame input is intact (candidates borrowed it), so a
                // retryable fan-out failure retries serially at attempt 2:
                // one live chunk instead of `width`. A serial attempt-1
                // failure has nothing narrower to fall back to and fails
                // directly through the shared wrapper below.
                Err(error)
                    if is_retryable_final_sort_error(&error) && initial_attempt <= 1 && self.max_attempts > 1 =>
                {
                    record_final_sort_retry_error(&error, self.execution.metrics());
                    crate::exec::retry::backoff_before_retry(
                        self.execution.context(),
                        "query_engine.exec.bounded.final_sort_lane_retry",
                    )?;
                    self.initial_topk_candidates(
                        &input,
                        TopKCandidatePlan { keep_rows, bytes_per_row, rows_per_chunk },
                        2,
                    )
                },
                Err(error) => Err(error),
            }
        } else {
            self.initial_topk_candidates(
                &input,
                TopKCandidatePlan { keep_rows, bytes_per_row, rows_per_chunk },
                initial_attempt,
            )
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
                            crate::exec::retry::backoff_before_retry(
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

        let output_reservation = match self.reserve_estimated_final_sort_output(
            estimated_limited_output_bytes(candidate.device_bytes(), candidate.frame.num_rows(), limit.spec)?,
            self.max_attempts,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                return Err(error);
            },
        };
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
        self.finalize_final_sort_output_reservation(output, output_reservation, self.max_attempts)
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
        bytes_per_row: u64,
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
                    bytes_per_row,
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
            |state, execution| match state.payload.next_yielded_frame(execution) {
                Some(Ok(mut yielded)) => {
                    let input = yielded.take_for_release();
                    let bytes_per_row = estimated_bytes_per_row(input.device_bytes(), input.frame.num_rows());
                    Ok(Some(SortLaneCandidate {
                        input,
                        slice: None,
                        bytes_per_row,
                        boundary: "final_sort_topk_stream_chunk",
                    }))
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
            match self.topk_slice_attempt(
                input,
                TopKSliceAttempt {
                    offset,
                    length,
                    keep_rows: plan.keep_rows,
                    bytes_per_row: plan.bytes_per_row,
                    attempt,
                },
            ) {
                Ok(candidate) => candidates.push(candidate),
                Err(error) => {
                    return Err(error);
                },
            }
            offset += length;
        }
        Ok(candidates)
    }

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
            while let Some(input) = inputs.next_yielded_frame(self.execution) {
                let mut input = match input {
                    Ok(input) => input,
                    Err(error) => {
                        inputs.release_remaining_frames();
                        return Err(error);
                    },
                };
                let input = input.take_for_release();
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
                            crate::exec::retry::backoff_before_retry(
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

    fn zero_limit_stream(
        &mut self,
        limit: &LimitNode,
        inputs: &mut ResidentFrameStreamPayload,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let mut first = match inputs.next_yielded_frame(self.execution) {
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
        let first = first.take_for_release();
        let output_reservation = match self.reserve_estimated_final_sort_output(0, attempt) {
            Ok(reservation) => reservation,
            Err(error) => {
                inputs.release_remaining_frames();
                return Err(error);
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
        self.finalize_final_sort_output_reservation(output, output_reservation, attempt)
    }

    fn topk_slice_attempt(&mut self, input: &CommittedPartial, slice: TopKSliceAttempt) -> Result<CommittedPartial> {
        const BOUNDARY: &str = "final_sort_topk_chunk";
        let output_rows = slice.keep_rows.min(slice.length);
        let output_reservation = match self
            .reserve_estimated_final_sort_output(estimate_rows_bytes(slice.bytes_per_row, output_rows), slice.attempt)
        {
            Ok(reservation) => reservation,
            Err(error) => {
                return Err(error);
            },
        };
        self.sort_and_limit_topk_candidate(
            output_reservation,
            BOUNDARY,
            slice.attempt,
            slice.keep_rows,
            TopKWorkTokenFailureMode::Direct,
            |work_token| {
                input
                    .frame
                    .frame()
                    .clone()
                    .with_execution(&work_token.execution_overrides())
                    .slice(slice.offset, slice.length)
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
        let bytes_per_row = estimated_bytes_per_row(input.device_bytes(), input_rows);
        let output_rows = keep_rows.min(input_rows);
        let output_reservation =
            match self.reserve_estimated_final_sort_output(estimate_rows_bytes(bytes_per_row, output_rows), attempt) {
                Ok(reservation) => reservation,
                Err(error) => {
                    return Err(error);
                },
            };
        let ResidentGpuFrame { frame, .. } = input;
        self.sort_and_limit_topk_candidate(
            output_reservation,
            BOUNDARY,
            attempt,
            keep_rows,
            TopKWorkTokenFailureMode::Direct,
            move |_work_token| Ok(frame),
        )
    }

    fn sort_and_limit_topk_candidate(
        &mut self,
        output_reservation: u64,
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
        self.finalize_final_sort_output_reservation(frame, output_reservation, attempt)
    }

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

    fn topk_merge_group_attempt(
        &mut self,
        candidates: &[CommittedPartial],
        merge: TopKMergeAttempt,
    ) -> Result<CommittedPartial> {
        debug_assert!(candidates.len() > 1);
        const BOUNDARY: &str = "final_sort_topk_merge";
        let requested_bytes = candidates.iter().map(|candidate| candidate.device_bytes()).sum::<u64>();
        let candidate_rows = candidates.iter().map(|candidate| candidate.frame.num_rows()).sum::<usize>();
        let bytes_per_row = estimated_bytes_per_row(requested_bytes, candidate_rows);
        let output_rows = merge.keep_rows.min(candidate_rows);
        let output_reservation = match self
            .reserve_estimated_final_sort_output(estimate_rows_bytes(bytes_per_row, output_rows), merge.attempt)
        {
            Ok(reservation) => reservation,
            Err(error) => {
                return Err(error);
            },
        };
        self.sort_and_limit_topk_candidate(
            output_reservation,
            BOUNDARY,
            merge.attempt,
            merge.keep_rows,
            TopKWorkTokenFailureMode::RuntimeFacts,
            |work_token| {
                concat_committed_frames_with_token(candidates, work_token, "query_engine.final_sort.topk_concat")
            },
        )
    }

    fn apply_final_limit(
        &mut self,
        candidate: CommittedPartial,
        limit: &LimitNode,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let output_reservation = match self.reserve_estimated_final_sort_output(
            estimated_limited_output_bytes(candidate.device_bytes(), candidate.frame.num_rows(), limit.spec)?,
            attempt,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                return Err(error);
            },
        };
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
        self.finalize_final_sort_output_reservation(output, output_reservation, attempt)
    }
}
