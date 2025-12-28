use cudf::{ExecutionBudgetReport, ExecutionBudgetRequest};

use super::*;

pub(super) fn execute_final_sort_limit_with_retry(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let Some(sort) = bounded.sort else {
        return Ok(input);
    };
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_sort_limit_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { context, bounded, sort, max_attempts, reservations, metrics };
    let result = final_sort.full_sort_limit_attempt(bounded.limit, &input, 1);

    match result {
        Ok(output) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            final_sort.reservations.release(input.reservation);
            Ok(output)
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.metrics);
            if max_attempts > 1
                && let Some(limit) = bounded.limit
            {
                match final_sort.hierarchical_topk(limit, input) {
                    Ok(output) => {
                        final_sort.metrics.final_sort_time += timer.elapsed();
                        Ok(output)
                    },
                    Err(error) => {
                        final_sort.metrics.final_sort_time += timer.elapsed();
                        Err(error)
                    },
                }
            } else {
                final_sort.metrics.final_sort_time += timer.elapsed();
                final_sort.reservations.release(input.reservation);
                Err(final_sort_retry_facts("retry_attempts_exhausted", final_sort.metrics).attach_to_error(error))
            }
        },
        Err(error) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            final_sort.reservations.release(input.reservation);
            Err(error)
        },
    }
}

struct FinalSortExecutor<'a, 'plan> {
    context: &'a ExecutionContext,
    bounded: &'a BoundedPlan<'plan>,
    sort: &'a SortNode,
    max_attempts: usize,
    reservations: &'a mut ReservationTracker,
    metrics: &'a mut QueryMetrics,
}

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
    fn full_sort_limit_attempt(
        &mut self,
        limit: Option<&LimitNode>,
        input: &CommittedPartial,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let sort = lower_sort_spec(&self.sort.spec)?;
        let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
        let prepare_token = self.final_sort_budget_token("final_sort", attempt)?;
        let timer = Instant::now();
        let prepared = input
            .frame
            .frame()
            .clone()
            .prepare_sort_values_with_budget(sort.columns.iter().map(String::as_str), options, &prepare_token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.final_sort.prepare",
                    "failed to prepare bounded native final sort order",
                    error,
                )
            })
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
        self.metrics.operator_time += timer.elapsed();

        let (prepared_sort, prepare_report) = prepared.into_parts();
        let sorted_order_bytes = final_sort_retained_bytes(&prepare_report)
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
        let sorted_order_request =
            MemoryReservationRequest::new(ReservationKind::FinalSort, sorted_order_bytes, self.reservations.budget())
                .with_memory_boundary("final_sort")
                .with_node_id(self.bounded.aggregate_id)
                .with_stage("final_sort")
                .with_attempt(attempt, self.max_attempts);
        let sorted_order_reservation = self.reservations.try_reserve(sorted_order_request, self.metrics)?;

        let timer = Instant::now();
        let materialize_overrides = execution_overrides(self.context);
        let materialized_output_bytes = prepared_sort
            .materialized_output_reservation_size_bytes_in(&materialize_overrides)
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.final_sort.materialize_size",
                    "failed to size bounded native final sort output",
                    error,
                )
            })
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
        let materialized_output_bytes = u64::try_from(materialized_output_bytes).unwrap_or(u64::MAX);
        let materialized_reservation =
            self.reserve_full_sort_materialized_output(materialized_output_bytes, attempt)?;
        let materialized_frame = match prepared_sort
            .materialize_in(&materialize_overrides)
            .map(GpuDataFrame::new)
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.final_sort.materialize",
                    "failed to materialize bounded native final sort output",
                    error,
                )
            })
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))
        {
            Ok(frame) => frame,
            Err(error) => {
                self.reservations.release(materialized_reservation);
                return Err(error);
            },
        };
        self.metrics.operator_time += timer.elapsed();
        self.metrics.record_device_memory_observation(materialized_output_bytes);

        drop(prepared_sort);
        self.reservations.release(sorted_order_reservation);

        if let Some(limit) = limit {
            if limit_keeps_all_rows(materialized_frame.num_rows(), limit.spec)? {
                return Ok(CommittedPartial { frame: materialized_frame, reservation: materialized_reservation });
            }

            let frame = match execute_optional_limit(self.context, Some(limit), materialized_frame, self.metrics)
                .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))
            {
                Ok(frame) => frame,
                Err(error) => {
                    self.reservations.release(materialized_reservation);
                    return Err(error);
                },
            };
            self.reservations.release(materialized_reservation);
            let output = reserve_final_sort_output(
                self.bounded,
                frame,
                attempt,
                self.max_attempts,
                self.reservations,
                self.metrics,
            );
            output
        } else {
            Ok(CommittedPartial { frame: materialized_frame, reservation: materialized_reservation })
        }
    }

    fn final_sort_budget_token(&self, boundary: &'static str, attempt: usize) -> Result<cudf::ExecutionBudgetToken> {
        let remaining = self.reservations.budget().saturating_sub(self.reservations.reserved_bytes());
        let budget = usize::try_from(remaining).unwrap_or(usize::MAX);
        self.context
            .runtime()
            .execution_budget_token(ExecutionBudgetRequest::new(budget).with_boundary(boundary))
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "cudf_query_engine.final_sort.budget_token",
                    "failed to create bounded native final sort budget token",
                    error,
                )
            })
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))
    }

    fn reserve_full_sort_materialized_output(
        &mut self,
        output_bytes: u64,
        attempt: usize,
    ) -> Result<ActiveReservation> {
        let output_request =
            MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, self.reservations.budget())
                .with_memory_boundary("final_sort")
                .with_node_id(self.bounded.aggregate_id)
                .with_stage("final_sort")
                .with_attempt(attempt, self.max_attempts);
        self.reservations.try_reserve(output_request, self.metrics)
    }

    fn attach_final_sort_resource_facts(&self, error: QueryError, attempt: usize) -> QueryError {
        if error.code() != QueryErrorCode::CudfResourceExhausted {
            return error;
        }

        error
            .with_fact("memory_boundary", "final_sort")
            .with_fact("stage", "final_sort")
            .with_fact("node_id", self.bounded.aggregate_id)
            .with_fact("attempt", attempt)
            .with_fact("max_attempts", self.max_attempts)
    }

    fn hierarchical_topk(&mut self, limit: &LimitNode, input: CommittedPartial) -> Result<CommittedPartial> {
        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            let output = match execute_limit_on_frame(self.context, limit, input.frame.frame().clone(), self.metrics) {
                Ok(output) => output,
                Err(error) => {
                    self.reservations.release(input.reservation);
                    return Err(error);
                },
            };
            self.reservations.release(input.reservation);
            return reserve_final_sort_output(
                self.bounded,
                output,
                2,
                self.max_attempts,
                self.reservations,
                self.metrics,
            );
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

        let output = execute_limit(self.context, limit, candidate.frame, self.metrics).map_err(|error| {
            RuntimeResourceFacts {
                memory_boundary: "final_sort_limit",
                stage: "final_sort",
                node_id: self.bounded.aggregate_id,
                attempt: self.max_attempts,
                max_attempts: self.max_attempts,
                resume_boundary: "final_sort",
                metrics: self.metrics,
            }
            .attach_to(error)
        })?;
        self.reservations.release(candidate.reservation);
        reserve_final_sort_output(
            self.bounded,
            output,
            self.max_attempts,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
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

        let result = input
            .frame
            .frame()
            .clone()
            .with_execution(&execution_overrides(self.context))
            .slice(slice.offset, slice.length)
            .map(GpuDataFrame::new)
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.final_sort.topk_slice",
                    "failed to slice bounded native final sort input",
                    error,
                )
            })
            .and_then(|frame| execute_topk_candidate(self.context, self.sort, frame, slice.keep_rows, self.metrics));
        self.reservations.release(work_reservation);
        let frame = result.map_err(|error| {
            RuntimeResourceFacts {
                memory_boundary: "final_sort_topk_chunk",
                stage: "final_sort",
                node_id: self.bounded.aggregate_id,
                attempt: slice.attempt,
                max_attempts: self.max_attempts,
                resume_boundary: "final_sort",
                metrics: self.metrics,
            }
            .attach_to(error)
        })?;
        reserve_final_sort_output(
            self.bounded,
            frame,
            slice.attempt,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
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
        let request =
            MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, self.reservations.budget())
                .with_memory_boundary("final_sort_topk_merge")
                .with_node_id(self.bounded.aggregate_id)
                .with_stage("final_sort")
                .with_attempt(merge.attempt, self.max_attempts);
        let merge_reservation = self.reservations.try_reserve(request, self.metrics)?;

        let result = concat_committed_frames(self.context, candidates, "cudf_query_engine.final_sort.topk_concat")
            .and_then(|frame| execute_topk_candidate(self.context, self.sort, frame, merge.keep_rows, self.metrics));
        self.reservations.release(merge_reservation);
        let frame = result.map_err(|error| {
            RuntimeResourceFacts {
                memory_boundary: "final_sort_topk_merge",
                stage: "final_sort",
                node_id: self.bounded.aggregate_id,
                attempt: merge.attempt,
                max_attempts: self.max_attempts,
                resume_boundary: "final_sort",
                metrics: self.metrics,
            }
            .attach_to(error)
        })?;
        reserve_final_sort_output(
            self.bounded,
            frame,
            merge.attempt,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
    }
}

fn final_sort_retained_bytes(report: &ExecutionBudgetReport) -> Result<u64> {
    let bytes = report.retained_bytes().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "cudf_query_engine.final_sort.prepare",
            "prepared final sort budget report did not include retained sorted-order bytes",
        )
        .with_fact("budget_boundary", report.request().boundary())
        .with_fact("budget_bytes", report.request().budget_bytes())
        .with_fact("budget_enforcement", report.enforcement().as_str())
        .with_fact("budget_status", report.status().as_str())
    })?;
    Ok(bytes as u64)
}

pub(super) fn execute_topk_candidate(
    context: &ExecutionContext,
    sort: &SortNode,
    input: GpuDataFrame,
    keep_rows: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sorted = execute_sort(context, sort, input, metrics)?;
    execute_limit_spec(context, LimitSpec::fetch(keep_rows), sorted, metrics)
}

pub(super) fn concat_committed_frames(
    context: &ExecutionContext,
    frames: &[CommittedPartial],
    operation: &'static str,
) -> Result<GpuDataFrame> {
    let first_frame = frames[0].frame.frame().clone();
    let other_frames = frames[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let output = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(operation, "failed to concatenate bounded native final sort candidates", error)
            .with_fact("candidate_count", frames.len())
    })?;
    Ok(GpuDataFrame::new(output))
}

pub(super) fn reserve_final_sort_output(
    bounded: &BoundedPlan<'_>,
    frame: GpuDataFrame,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort_output",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request = MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, reservations.budget())
        .with_memory_boundary("final_sort_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) enum TopKLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

pub(super) fn commit_topk_merge_level(
    candidates: Vec<CommittedPartial>,
    level_output: Vec<TopKLevelItem>,
    reservations: &mut ReservationTracker,
) -> Vec<CommittedPartial> {
    let mut candidates = candidates.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_candidates = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            TopKLevelItem::Merged(partial) => next_candidates.push(partial),
            TopKLevelItem::Carry { partial_index } => {
                let partial = candidates[partial_index].take().expect("carry index references a top-k candidate");
                next_candidates.push(partial);
            },
        }
    }

    for partial in candidates.into_iter().flatten() {
        reservations.release(partial.reservation);
    }

    next_candidates
}

pub(super) fn release_topk_level_items(items: Vec<TopKLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let TopKLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

pub(super) fn release_committed_partials(partials: Vec<CommittedPartial>, reservations: &mut ReservationTracker) {
    for partial in partials {
        reservations.release(partial.reservation);
    }
}

pub(super) fn is_retryable_final_sort_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

pub(super) fn record_final_sort_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn final_sort_retry_facts(retry_action: &'static str, metrics: &QueryMetrics) -> RetryFacts {
    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_sort")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}

pub(super) fn final_sort_topk_retry_facts(
    merge_level: usize,
    merge_fan_in: usize,
    metrics: &QueryMetrics,
) -> RetryFacts {
    final_sort_retry_facts(
        if merge_fan_in <= 2 { "fail_min_topk_fan_in" } else { "hierarchical_topk_retry_exhausted" },
        metrics,
    )
    .with_merge_level(merge_level)
    .with_merge_fan_in(merge_fan_in)
}

pub(in crate::exec) fn topk_candidate_rows(spec: LimitSpec) -> Result<usize> {
    spec.skip.checked_add(spec.fetch).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.final_sort.topk_limit",
            "limit skip and fetch overflow usize",
        )
        .with_fact("skip", spec.skip)
        .with_fact("fetch", spec.fetch)
    })
}

pub(super) fn estimated_bytes_per_row(bytes: u64, rows: usize) -> u64 {
    if rows == 0 {
        return bytes.max(1);
    }
    bytes.div_ceil(rows as u64).max(1)
}

pub(super) fn estimate_rows_bytes(bytes_per_row: u64, rows: usize) -> u64 {
    let bytes = u128::from(bytes_per_row) * rows as u128;
    bytes.min(u128::from(u64::MAX)) as u64
}

pub(super) fn final_sort_rows_per_chunk(input_rows: usize, bytes_per_row: u64, available_bytes: u64) -> usize {
    if input_rows == 0 {
        return 1;
    }
    let rows = (available_bytes / bytes_per_row).min(input_rows as u64) as usize;
    rows.max(1)
}
