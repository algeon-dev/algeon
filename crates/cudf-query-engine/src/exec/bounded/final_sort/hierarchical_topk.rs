use super::*;

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

impl<'a, 'plan> FinalSortExecutor<'a, 'plan> {
    pub(super) fn hierarchical_topk(&mut self, limit: &LimitNode, input: CommittedPartial) -> Result<CommittedPartial> {
        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            let output_reservation = match self.reserve_estimated_final_sort_output(0, 2) {
                Ok(reservation) => reservation,
                Err(error) => {
                    self.reservations.release(input.reservation);
                    return Err(error);
                },
            };
            let token = match self.final_sort_budget_token_including("final_sort_limit", 2, &output_reservation) {
                Ok(token) => token,
                Err(error) => {
                    self.reservations.release(output_reservation);
                    self.reservations.release(input.reservation);
                    return Err(error);
                },
            };
            let output = match execute_limit_on_frame_with_token(
                limit,
                input.frame.frame().clone(),
                &token,
                "cudf_query_engine.final_sort.limit",
                self.metrics,
            ) {
                Ok(output) => output,
                Err(error) => {
                    self.reservations.release(output_reservation);
                    self.reservations.release(input.reservation);
                    return Err(error);
                },
            };
            self.reservations.release(input.reservation);
            return self.finalize_final_sort_output_reservation(output, output_reservation, 2);
        }

        let input_rows = input.frame.num_rows();
        if input_rows == 0 {
            return Ok(input);
        }
        let bytes_per_row = estimated_bytes_per_row(input.reservation.bytes(), input_rows);
        let rows_per_chunk = final_sort_rows_per_chunk(
            input_rows,
            bytes_per_row,
            self.reservations.budget().saturating_sub(input.reservation.bytes()),
        );
        let mut candidates = match self
            .initial_topk_candidates(&input, TopKCandidatePlan { keep_rows, bytes_per_row, rows_per_chunk })
        {
            Ok(candidates) => candidates,
            Err(error) => {
                record_final_sort_retry_error(&error, self.metrics);
                self.reservations.release(input.reservation);
                return Err(
                    final_sort_retry_facts("hierarchical_topk_initial_failed", self.metrics).attach_to_error(error)
                );
            },
        };

        self.reservations.release(input.reservation);

        let mut merge_level = 1;
        while candidates.len() > 1 {
            let mut merge_fan_in = candidates.len();
            let mut attempt = 2;

            loop {
                match self.topk_merge_level_attempt(&candidates, TopKMergeAttempt { keep_rows, merge_fan_in, attempt })
                {
                    Ok(level_output) => {
                        candidates = commit_topk_merge_level(candidates, level_output, self.reservations);
                        merge_level += 1;
                        break;
                    },
                    Err(error) if is_retryable_final_sort_error(&error) => {
                        record_final_sort_retry_error(&error, self.metrics);
                        if attempt < self.max_attempts
                            && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                        {
                            merge_fan_in = reduced_fan_in;
                            attempt += 1;
                            continue;
                        }

                        release_committed_partials(candidates, self.reservations);
                        return Err(
                            final_sort_topk_retry_facts(merge_level, merge_fan_in, self.metrics).attach_to_error(error)
                        );
                    },
                    Err(error) => {
                        release_committed_partials(candidates, self.reservations);
                        return Err(error);
                    },
                }
            }
        }

        let candidate = candidates.pop().expect("hierarchical top-k leaves exactly one candidate");
        if limit.spec.skip == 0 && candidate.frame.num_rows() <= limit.spec.fetch {
            return Ok(candidate);
        }

        let output_reservation = match self.reserve_estimated_final_sort_output(
            estimated_limited_output_bytes(candidate.reservation.bytes(), candidate.frame.num_rows(), limit.spec)?,
            self.max_attempts,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                self.reservations.release(candidate.reservation);
                return Err(error);
            },
        };
        let token =
            match self.final_sort_budget_token_including("final_sort_limit", self.max_attempts, &output_reservation) {
                Ok(token) => token,
                Err(error) => {
                    self.reservations.release(output_reservation);
                    self.reservations.release(candidate.reservation);
                    return Err(error);
                },
            };
        let output = match execute_limit_with_token(
            limit,
            candidate.frame,
            &token,
            "cudf_query_engine.final_sort.limit",
            self.metrics,
        ) {
            Ok(output) => output,
            Err(error) => {
                self.reservations.release(output_reservation);
                return Err(RuntimeResourceFacts {
                    memory_boundary: "final_sort_limit",
                    stage: "final_sort",
                    node_id: self.bounded.aggregate_id,
                    attempt: self.max_attempts,
                    max_attempts: self.max_attempts,
                    resume_boundary: "final_sort",
                    metrics: self.metrics,
                }
                .attach_to(error));
            },
        };
        self.reservations.release(candidate.reservation);
        self.finalize_final_sort_output_reservation(output, output_reservation, self.max_attempts)
    }

    fn initial_topk_candidates(
        &mut self,
        input: &CommittedPartial,
        plan: TopKCandidatePlan,
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
                    attempt: 2,
                },
            ) {
                Ok(candidate) => candidates.push(candidate),
                Err(error) => {
                    release_committed_partials(candidates, self.reservations);
                    return Err(error);
                },
            }
            offset += length;
        }
        Ok(candidates)
    }

    fn topk_slice_attempt(&mut self, input: &CommittedPartial, slice: TopKSliceAttempt) -> Result<CommittedPartial> {
        let requested_bytes = estimate_rows_bytes(slice.bytes_per_row, slice.length);
        let request =
            MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, self.reservations.budget())
                .with_memory_boundary("final_sort_topk_chunk")
                .with_node_id(self.bounded.aggregate_id)
                .with_stage("final_sort")
                .with_attempt(slice.attempt, self.max_attempts);
        let work_reservation = self.reservations.try_reserve(request, self.metrics)?;

        let output_rows = slice.keep_rows.min(slice.length);
        let output_reservation = match self
            .reserve_estimated_final_sort_output(estimate_rows_bytes(slice.bytes_per_row, output_rows), slice.attempt)
        {
            Ok(reservation) => reservation,
            Err(error) => {
                self.reservations.release(work_reservation);
                return Err(error);
            },
        };
        let work_token =
            match self.final_sort_budget_token_including("final_sort_topk_chunk", slice.attempt, &work_reservation) {
                Ok(token) => token,
                Err(error) => {
                    self.reservations.release(output_reservation);
                    self.reservations.release(work_reservation);
                    return Err(error);
                },
            };
        let limit_token =
            match self.final_sort_budget_token_including("final_sort_topk_chunk", slice.attempt, &output_reservation) {
                Ok(token) => token,
                Err(error) => {
                    self.reservations.release(output_reservation);
                    self.reservations.release(work_reservation);
                    return Err(error);
                },
            };
        let result = input
            .frame
            .frame()
            .clone()
            .with_execution(&work_token.execution_overrides())
            .slice(slice.offset, slice.length)
            .map(GpuDataFrame::new)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.final_sort.topk_slice",
                    "failed to slice bounded native final sort input",
                    error,
                )
                .with_fact_if_absent("budget_boundary", work_token.request().boundary())
                .with_fact_if_absent("budget_bytes", work_token.request().budget_bytes())
                .with_fact_if_absent("budget_operation", "dataframe::slice")
            })
            .and_then(|frame| {
                execute_topk_candidate_with_budget(
                    self.sort,
                    frame,
                    slice.keep_rows,
                    &work_token,
                    &limit_token,
                    "final_sort_topk_chunk",
                    self.metrics,
                )
            });
        self.reservations.release(work_reservation);
        let frame = match result {
            Ok(frame) => frame,
            Err(error) => {
                self.reservations.release(output_reservation);
                return Err(RuntimeResourceFacts {
                    memory_boundary: "final_sort_topk_chunk",
                    stage: "final_sort",
                    node_id: self.bounded.aggregate_id,
                    attempt: slice.attempt,
                    max_attempts: self.max_attempts,
                    resume_boundary: "final_sort",
                    metrics: self.metrics,
                }
                .attach_to(error));
            },
        };
        self.finalize_final_sort_output_reservation(frame, output_reservation, slice.attempt)
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
                Ok(candidate) => output.push(TopKLevelItem::Merged(candidate)),
                Err(error) => {
                    release_topk_level_items(output, self.reservations);
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
        let requested_bytes = candidates.iter().map(|candidate| candidate.reservation.bytes()).sum::<u64>();
        let candidate_rows = candidates.iter().map(|candidate| candidate.frame.num_rows()).sum::<usize>();
        let bytes_per_row = estimated_bytes_per_row(requested_bytes, candidate_rows);
        let request =
            MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, self.reservations.budget())
                .with_memory_boundary("final_sort_topk_merge")
                .with_node_id(self.bounded.aggregate_id)
                .with_stage("final_sort")
                .with_attempt(merge.attempt, self.max_attempts);
        let merge_reservation = self.reservations.try_reserve(request, self.metrics)?;

        let output_rows = merge.keep_rows.min(candidate_rows);
        let output_reservation = match self
            .reserve_estimated_final_sort_output(estimate_rows_bytes(bytes_per_row, output_rows), merge.attempt)
        {
            Ok(reservation) => reservation,
            Err(error) => {
                self.reservations.release(merge_reservation);
                return Err(error);
            },
        };
        let result = (|| {
            let work_token =
                self.final_sort_budget_token_including("final_sort_topk_merge", merge.attempt, &merge_reservation)?;
            let limit_token =
                self.final_sort_budget_token_including("final_sort_topk_merge", merge.attempt, &output_reservation)?;
            let frame = concat_committed_frames_with_token(
                candidates,
                &work_token,
                "cudf_query_engine.final_sort.topk_concat",
            )?;
            execute_topk_candidate_with_budget(
                self.sort,
                frame,
                merge.keep_rows,
                &work_token,
                &limit_token,
                "final_sort_topk_merge",
                self.metrics,
            )
        })();
        self.reservations.release(merge_reservation);
        let frame = match result {
            Ok(frame) => frame,
            Err(error) => {
                self.reservations.release(output_reservation);
                return Err(RuntimeResourceFacts {
                    memory_boundary: "final_sort_topk_merge",
                    stage: "final_sort",
                    node_id: self.bounded.aggregate_id,
                    attempt: merge.attempt,
                    max_attempts: self.max_attempts,
                    resume_boundary: "final_sort",
                    metrics: self.metrics,
                }
                .attach_to(error));
            },
        };
        self.finalize_final_sort_output_reservation(frame, output_reservation, merge.attempt)
    }
}
