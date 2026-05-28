use std::collections::VecDeque;

use super::{
    super::source::{
        ScheduledLocalParquetRead, ScheduledLocalParquetReadOutput, local_parquet_execution_frame_schema,
        read_local_parquet_chunk,
    },
    *,
};
use crate::exec::types::{GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey};

const SOURCE_ADJACENT_WORK_RESERVATION_HEADROOM_BYTES: u64 = 1;
const SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES: u64 = 8 * 1024 * 1024;

mod retry;
mod scheduling;
mod source_read;

// Private re-export so sibling submodules can reach `is_prefetch_chunk_error`
// via `use super::*;` without exposing it outside the chunks module.
use retry::is_prefetch_chunk_error;
pub(in crate::exec) use retry::record_chunk_retry_error;
pub(super) use retry::{chunk_retry_facts, is_retryable_chunk_error, split_retryable_chunk};
pub(super) use scheduling::execute_bounded_chunks_with_source_scheduler;
pub(super) use source_read::{
    concatenate_source_read_outputs, read_local_parquet_chunks_required_only,
    read_local_parquet_chunks_required_only_with_observation, release_source_read_outputs_reverse,
};
pub(in crate::exec::bounded) use source_read::{
    insert_bounded_local_parquet_source_cache, try_read_cached_local_parquet_chunk,
};

pub(super) fn execute_bounded_chunk_with_retry(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        context.memory_control_policy()
    else {
        unreachable!("execute_bounded_chunk_with_retry is called only for bounded policies");
    };

    execute_bounded_chunk_retry_attempt(
        context,
        bounded,
        chunk.clone(),
        reservations,
        metrics,
        1,
        max_retry_attempts.max(1),
        min_retry_chunk_row_groups.max(1),
    )
}

pub(super) fn execute_bounded_chunk_retry_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    context.check_cancellation("cudf_query_engine.exec.bounded.chunk_retry")?;
    match execute_bounded_chunk_attempt(context, bounded, &chunk, reservations, metrics, attempt, max_attempts) {
        Ok(partial) => Ok(vec![partial]),
        Err(error) if is_retryable_chunk_error(&error) => retry_after_chunk_error(
            context,
            bounded,
            chunk,
            error,
            reservations,
            metrics,
            attempt,
            max_attempts,
            min_retry_chunk_row_groups,
        ),
        Err(error) => Err(error),
    }
}
fn retry_after_chunk_error(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    error: QueryError,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    record_chunk_retry_error(&error, metrics);
    if aggregate_spec_has_distinct(&bounded.aggregate.0.spec) {
        return Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error));
    }
    if attempt < max_attempts
        && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
    {
        metrics.record_source_chunk_retry();
        let mut partials = execute_bounded_chunk_retry_attempt(
            context,
            bounded,
            left,
            reservations,
            metrics,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?;
        partials.extend(execute_bounded_chunk_retry_attempt(
            context,
            bounded,
            right,
            reservations,
            metrics,
            attempt + 1,
            max_attempts,
            min_retry_chunk_row_groups,
        )?);
        return Ok(partials);
    }

    Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
}

/// Reads a source chunk with its pre-aggregate filter and projection stages
/// applied, without running a partial aggregate. Returns the raw filtered and
/// projected data as a [`CommittedPartial`].
///
/// Used by the distinct aggregate path where all raw source data must be
/// concatenated before a single groupby can produce a correct COUNT(DISTINCT).
pub(in crate::exec::bounded) fn execute_bounded_raw_source_chunk(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    context.check_cancellation("cudf_query_engine.exec.bounded.raw_source_chunk")?;
    let source = read_bounded_chunk_source(
        context,
        bounded.source,
        chunk,
        reservations,
        metrics,
        1,
        1,
        !bounded.has_source_adjacent_stages(),
    )
    .map_err(|error| attach_chunk_resource_facts(error, chunk, "source_chunk_read", 1, 1, metrics))?;
    let input = CommittedPartial { frame: source.frame, reservation: source.reservation };
    execute_source_adjacent_chunk_stages(context, bounded, chunk, input, reservations, metrics, 1, 1)
}

pub(super) fn execute_bounded_chunk_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    context.check_cancellation("cudf_query_engine.exec.bounded.chunk_attempt")?;
    let source = read_bounded_chunk_source(
        context,
        bounded.source,
        chunk,
        reservations,
        metrics,
        attempt,
        max_attempts,
        !bounded.has_source_adjacent_stages(),
    )
    .map_err(|error| attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics))?;
    execute_bounded_chunk_operator_from_source(
        context,
        bounded,
        source,
        reservations,
        metrics,
        attempt,
        max_attempts,
        |_reservations, _metrics| Ok(()),
    )
}
fn read_bounded_chunk_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    grow_source_to_observed: bool,
) -> Result<ScheduledLocalParquetReadOutput> {
    context.check_cancellation("cudf_query_engine.exec.bounded.source_chunk_read")?;
    metrics.record_source_chunk_attempt();
    if let Some(output) = try_read_cached_local_parquet_chunk(context, source, chunk, reservations, metrics)? {
        return Ok(output);
    }

    let source_request = source_read_reservation_request(
        context.memory_control_policy(),
        ReservationKind::SourceRead,
        "source_chunk_read",
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let mut source_reservation = reservations.try_reserve(source_request, metrics)?;
    let source_token_budget = reservations.available_bytes_including(&source_reservation);
    let frame =
        match read_local_parquet_chunk(context, source, chunk, source_token_budget, "source_chunk_read", metrics) {
            Ok(frame) => frame,
            Err(error) => {
                match try_read_local_parquet_chunk_with_corrected_budget_after_error(
                    context,
                    source,
                    chunk,
                    &mut source_reservation,
                    reservations,
                    metrics,
                    "source_chunk_read",
                    &error,
                    attempt,
                    max_attempts,
                ) {
                    Ok(Some(frame)) => frame,
                    Ok(None) => {
                        reservations.release(source_reservation, metrics);
                        return Err(error);
                    },
                    Err(retry_error) => {
                        reservations.release(source_reservation, metrics);
                        return Err(retry_error);
                    },
                }
            },
        };
    let mut output = ScheduledLocalParquetReadOutput {
        chunk: chunk.clone(),
        frame,
        reservation: source_reservation,
        observed_bytes: 0,
    };
    if grow_source_to_observed {
        output.observed_bytes = grow_source_reservation_to_observed(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            "source_chunk_read",
            attempt,
            max_attempts,
        )?;
        insert_bounded_local_parquet_source_cache(
            context,
            source,
            &output.chunk,
            &output.frame,
            &mut output.reservation,
            reservations,
            metrics,
            "source_chunk_read",
            attempt,
            max_attempts,
            output.observed_bytes,
        )?;
    } else {
        output.observed_bytes = match shrink_source_reservation_to_observed_without_grow(
            &output.frame,
            &mut output.reservation,
            reservations,
            &output.chunk,
            metrics,
            "source_chunk_read",
            attempt,
            max_attempts,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(output.frame);
                reservations.release(output.reservation, metrics);
                return Err(error);
            },
        };
    }
    Ok(output)
}
fn execute_bounded_chunk_operator_from_source(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    source: ScheduledLocalParquetReadOutput,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    before_operator: impl FnOnce(&mut ReservationTracker, &mut QueryMetrics) -> Result<()>,
) -> Result<CommittedPartial> {
    let ScheduledLocalParquetReadOutput { chunk, frame, reservation: source_reservation, observed_bytes: _ } = source;
    if let Err(error) = context.check_cancellation("cudf_query_engine.exec.bounded.chunk_operator") {
        drop(frame);
        reservations.release(source_reservation, metrics);
        return Err(error);
    }
    let mut input = CommittedPartial { frame, reservation: source_reservation };
    input = execute_source_adjacent_chunk_stages(
        context,
        bounded,
        &chunk,
        input,
        reservations,
        metrics,
        attempt,
        max_attempts,
    )?;
    let CommittedPartial { frame, reservation: source_reservation } = input;

    let operator_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        "chunk_operator",
        source_reservation.bytes().min(reservations.remaining_bytes()),
        reservations.budget(),
        &chunk,
        attempt,
        max_attempts,
    );
    let operator_reservation = match reservations.try_reserve(operator_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(frame);
            reservations.release(source_reservation, metrics);
            return Err(error);
        },
    };
    let operator_token_budget = reservations.available_bytes_including(&operator_reservation);
    let operator_token = match execution_budget_token_for_boundary(
        context,
        operator_token_budget,
        "chunk_operator",
        "cudf_query_engine.chunk_operator.budget_token",
        "failed to create bounded native chunk operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(operator_reservation, metrics);
            reservations.release(source_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, &chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };
    if let Err(error) = before_operator(reservations, metrics) {
        reservations.release(operator_reservation, metrics);
        reservations.release(source_reservation, metrics);
        return Err(error);
    }

    let partial_input_rows = frame.num_rows() as u64;
    let source_chunk_fingerprint = partial_aggregate_source_chunk_fingerprint(std::slice::from_ref(&chunk));
    let input_evidence = PartialAggregateInputEvidence {
        source_chunk_fingerprint,
        source_adjacent_partial: bounded.has_source_adjacent_stages(),
        source_chunk_row_group_count: chunk.row_group_count() as u64,
        source_chunk_reads_all_row_groups: chunk.reads_all_row_groups,
        source_chunk_known_row_count: chunk.known_row_count,
        source_chunk_estimated_source_read_bytes: chunk.estimated_source_read_bytes,
        source_chunk_size_estimate_based: chunk.size_estimate_based,
        source_adjacent_partition_strategy: "none",
        source_adjacent_partition_count: None,
        source_adjacent_non_empty_partition_count: None,
        source_adjacent_partition_key_count: None,
        source_adjacent_max_partition_fan_in: None,
    };
    let chunk_operator_timer = Instant::now();
    let chunk_output = execute_partial_aggregate_with_budget(
        context,
        Some(bounded.aggregate_id),
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        frame,
        &operator_token,
        Some(input_evidence),
        metrics,
    );
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(operator_reservation, metrics);
            reservations.release(source_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, &chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };

    reservations.release(operator_reservation, metrics);
    reservations.release(source_reservation, metrics);

    let partial_bytes = observe_device_memory_bytes(&frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, &chunk, "partial_state", attempt, max_attempts, metrics))?;
    let partial_request = chunk_reservation_request(
        ReservationKind::PartialState,
        "partial_state",
        partial_bytes,
        reservations.budget(),
        &chunk,
        attempt,
        max_attempts,
    );
    let partial_reservation = reservations.try_reserve(partial_request, metrics)?;
    metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    metrics.record_partial_aggregate(partial_input_rows, frame.num_rows() as u64, partial_reservation.bytes());

    Ok(CommittedPartial { frame, reservation: partial_reservation })
}
fn execute_source_adjacent_chunk_stages(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    mut input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    for (projection, output_schema) in &bounded.pre_filter_projections {
        debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
        input = execute_source_adjacent_chunk_unary(
            context,
            bounded.source_id,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_projection_with_budget(context, projection, input, token, metrics),
        )?;
    }
    if let Some(filter) = bounded.filter {
        input = execute_source_adjacent_chunk_unary(
            context,
            bounded.source_id,
            "source_chunk_filter",
            "source_chunk_filter_output",
            chunk,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_filter_with_budget(context, filter, input, token, metrics),
        )?;
    }
    for (projection, output_schema) in &bounded.post_filter_projections {
        debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());
        input = execute_source_adjacent_chunk_unary(
            context,
            bounded.source_id,
            "source_chunk_projection",
            "source_chunk_projection_output",
            chunk,
            input,
            reservations,
            metrics,
            attempt,
            max_attempts,
            |context, input, token, metrics| execute_projection_with_budget(context, projection, input, token, metrics),
        )?;
    }
    Ok(input)
}
pub(in crate::exec::bounded) fn execute_source_adjacent_chunk_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    chunk: &SourceChunk,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    execute: impl FnOnce(&ExecutionContext, GpuDataFrame, &ExecutionBudgetToken, &mut QueryMetrics) -> Result<GpuDataFrame>,
) -> Result<CommittedPartial> {
    let work_request_bytes =
        source_adjacent_chunk_unary_work_request_bytes(input.reservation.bytes(), reservations.remaining_bytes());
    let mut work_reservation = match reservations.try_reserve_work(
        chunk_reservation_request(
            ReservationKind::ChunkOperator,
            stage,
            work_request_bytes,
            reservations.budget(),
            chunk,
            attempt,
            max_attempts,
        )
        .with_node_id(node_id)
        .with_stage(stage),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input.frame);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        stage,
        "cudf_query_engine.source_chunk_unary.budget_token",
        "failed to create bounded native source-adjacent chunk operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        },
    };
    let result = execute(context, input.frame, &token, metrics);
    let frame = match result {
        Ok(frame) => {
            reservations.release(input.reservation, metrics);
            frame
        },
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, stage, attempt, max_attempts, metrics));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(attach_chunk_resource_facts(error, chunk, output_boundary, attempt, max_attempts, metrics));
        },
    };
    let output_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        output_boundary,
        output_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    )
    .with_node_id(node_id)
    .with_stage(stage);
    if output_bytes > work_reservation.bytes() {
        let delta = output_bytes.saturating_sub(work_reservation.bytes());
        if let Err(error) =
            reservations.try_grow(&mut work_reservation, output_request.clone().with_requested_bytes(delta), metrics)
        {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut work_reservation, output_bytes, metrics);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn source_adjacent_chunk_unary_work_request_bytes(input_reserved_bytes: u64, remaining_bytes: u64) -> u64 {
    let admissible_work_bytes = remaining_bytes.saturating_sub(SOURCE_ADJACENT_WORK_RESERVATION_HEADROOM_BYTES);
    input_reserved_bytes.min(admissible_work_bytes)
}
fn grow_source_reservation_to_observed(
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> Result<u64> {
    let source_observed_bytes = match observe_device_memory_bytes(frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(attach_chunk_resource_facts(error, chunk, memory_boundary, attempt, max_attempts, metrics));
        },
    };
    if source_observed_bytes <= source_reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            source_reservation,
            source_observed_bytes,
            observed_source_read_reservation_request(
                memory_boundary,
                reservations.budget(),
                chunk,
                attempt,
                max_attempts,
            ),
            metrics,
        );
        return Ok(source_observed_bytes);
    }

    let source_delta = source_observed_bytes.saturating_sub(source_reservation.bytes());
    let grow_request =
        observed_source_read_reservation_request(memory_boundary, reservations.budget(), chunk, attempt, max_attempts)
            .with_requested_bytes(source_delta);
    reservations.try_grow(source_reservation, grow_request, metrics)?;
    Ok(source_observed_bytes)
}
fn shrink_source_reservation_to_observed_without_grow(
    frame: &GpuDataFrame,
    source_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
) -> Result<u64> {
    let source_observed_bytes = observe_device_memory_bytes(frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, chunk, memory_boundary, attempt, max_attempts, metrics))?;
    if source_observed_bytes <= source_reservation.bytes() {
        reservations.shrink_to_observed_with_request(
            source_reservation,
            source_observed_bytes,
            observed_source_read_reservation_request(
                memory_boundary,
                reservations.budget(),
                chunk,
                attempt,
                max_attempts,
            ),
            metrics,
        );
    }
    Ok(source_observed_bytes)
}

pub(super) fn chunk_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    source_read_reservation_contract(policy, chunk).requested_bytes
}

pub(super) fn prefetch_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let contract = source_read_reservation_contract(policy, chunk);
    match (policy, contract.quality) {
        (MemoryControlPolicy::Unbounded, _) => contract.requested_bytes,
        (
            MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. },
            SourceReadReservationContractQuality::ConservativeEstimate,
        ) => contract.requested_bytes.max(max_source_chunk_bytes),
        (MemoryControlPolicy::Bounded { .. }, _) => contract.requested_bytes,
    }
}

pub(super) fn join_leaf_prefetch_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> Option<u64> {
    let requested_bytes = prefetch_requested_bytes(policy, chunk);
    let MemoryControlPolicy::Bounded { device_budget_bytes, max_source_chunk_bytes, .. } = policy else {
        return Some(requested_bytes);
    };
    if chunk.zero_column_count {
        return Some(requested_bytes);
    }
    let minimum_read_budget = minimum_parquet_source_read_budget_bytes();
    if minimum_read_budget > max_source_chunk_bytes || minimum_read_budget > device_budget_bytes {
        return None;
    }
    Some(requested_bytes.max(minimum_read_budget))
}

fn minimum_parquet_source_read_budget_bytes() -> u64 {
    u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap_or(u64::MAX)
}

/// Precision of the byte contract used when reserving memory for one bounded
/// source-read chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) enum SourceReadReservationContractQuality {
    /// A lower layer supplied an exact preflight byte contract, so bounded
    /// source reads can reserve the required bytes before decoding starts.
    #[allow(dead_code)]
    ExactLowerLevelContract,
    /// Parquet footer metadata gave projected uncompressed byte counts for the
    /// columns that the source reader must decode.
    ProjectedUncompressedEstimate,
    /// The read is admitted with an initial reservation and then reconciled to
    /// observed device bytes by growing or shrinking after decoding.
    ObservedGrowShrink,
    /// A failed lower-level hard budget reported typed allocation counters and
    /// the retry reservation was corrected from those counters.
    CorrectedLowerLevelBudget,
    /// The reservation is based on file statistics or configured chunk caps;
    /// it may be retried with smaller row-group chunks if execution exceeds it.
    ConservativeEstimate,
}

impl SourceReadReservationContractQuality {
    pub(in crate::exec) const fn as_str(self) -> &'static str {
        match self {
            Self::ExactLowerLevelContract => "exact_lower_level_contract",
            Self::ProjectedUncompressedEstimate => "projected_uncompressed_estimate",
            Self::ObservedGrowShrink => "observed_grow_shrink",
            Self::CorrectedLowerLevelBudget => "corrected_lower_level_budget",
            Self::ConservativeEstimate => "conservative_estimate",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct SourceReadBudgetCorrection {
    previous_budget_bytes: u64,
    corrected_budget_bytes: u64,
    peak_aligned_bytes: u64,
    largest_failed_allocation_bytes: u64,
    failed_allocation_count: u64,
    cap_bytes: u64,
}

impl SourceReadBudgetCorrection {
    pub(in crate::exec) fn attach_to_error(self, error: QueryError) -> QueryError {
        error
            .with_fact_if_absent("budget_retry_previous_budget_bytes", self.previous_budget_bytes)
            .with_fact_if_absent("budget_retry_corrected_budget_bytes", self.corrected_budget_bytes)
            .with_fact_if_absent("budget_retry_peak_aligned_bytes", self.peak_aligned_bytes)
            .with_fact_if_absent("budget_retry_largest_failed_allocation_bytes", self.largest_failed_allocation_bytes)
            .with_fact_if_absent("budget_retry_failed_allocation_count", self.failed_allocation_count)
            .with_fact_if_absent("budget_retry_slack_bytes", SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES)
            .with_fact_if_absent("budget_retry_cap_bytes", self.cap_bytes)
    }

    pub(in crate::exec) fn corrected_budget_bytes(self) -> u64 {
        self.corrected_budget_bytes
    }

    pub(in crate::exec) fn previous_budget_bytes(self) -> u64 {
        self.previous_budget_bytes
    }
}

pub(in crate::exec) fn source_read_budget_correction_from_error(
    error: &QueryError,
    reservations: &ReservationTracker,
    reservation: &ActiveReservation,
) -> Option<SourceReadBudgetCorrection> {
    if error.code() != QueryErrorCode::CudfResourceExhausted {
        return None;
    }
    let detail = error.cudf_budget_exhaustion()?;
    if detail.failed_allocation_count() == 0 || detail.largest_failed_allocation_bytes() == 0 {
        return None;
    }
    let previous_budget_bytes = u64::try_from(detail.hard_budget_bytes()).ok()?;
    let peak_aligned_bytes = u64::try_from(detail.peak_aligned_bytes()).ok()?;
    let largest_failed_allocation_bytes = u64::try_from(detail.largest_failed_allocation_bytes()).ok()?;
    let failed_allocation_count = u64::try_from(detail.failed_allocation_count()).ok()?;
    let lower_bound = previous_budget_bytes.saturating_mul(2).max(
        peak_aligned_bytes
            .saturating_add(largest_failed_allocation_bytes)
            .saturating_add(SOURCE_READ_BUDGET_CORRECTION_SLACK_BYTES),
    );
    let cap_bytes = reservations.available_bytes_including(reservation);
    let corrected_budget_bytes = lower_bound.min(cap_bytes);
    if corrected_budget_bytes <= previous_budget_bytes {
        return None;
    }
    Some(SourceReadBudgetCorrection {
        previous_budget_bytes,
        corrected_budget_bytes,
        peak_aligned_bytes,
        largest_failed_allocation_bytes,
        failed_allocation_count,
        cap_bytes,
    })
}

pub(in crate::exec) fn try_grow_source_read_reservation_for_budget_correction(
    reservations: &mut ReservationTracker,
    reservation: &mut ActiveReservation,
    correction: SourceReadBudgetCorrection,
    mut request: MemoryReservationRequest,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    if correction.corrected_budget_bytes <= reservation.bytes() {
        return Ok(());
    }
    let delta = correction.corrected_budget_bytes.saturating_sub(reservation.bytes());
    request = request.with_requested_bytes(delta).with_source_read_contract(
        SourceReadReservationContractQuality::CorrectedLowerLevelBudget.as_str(),
        Some(correction.previous_budget_bytes),
        Some(correction.corrected_budget_bytes),
    );
    reservations.try_grow(reservation, request, metrics).map_err(|error| correction.attach_to_error(error))
}

fn try_read_local_parquet_chunk_with_corrected_budget_after_error(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    memory_boundary: &'static str,
    error: &QueryError,
    attempt: usize,
    max_attempts: usize,
) -> Result<Option<GpuDataFrame>> {
    context.check_cancellation("cudf_query_engine.exec.bounded.source_chunk_budget_retry")?;
    if attempt >= max_attempts {
        return Ok(None);
    }
    let Some(correction) = source_read_budget_correction_from_error(error, reservations, reservation) else {
        return Ok(None);
    };
    record_chunk_retry_error(error, metrics);
    metrics.record_source_chunk_retry();
    let request = chunk_reservation_request(
        ReservationKind::SourceRead,
        memory_boundary,
        0,
        reservations.budget(),
        chunk,
        attempt + 1,
        max_attempts,
    );
    try_grow_source_read_reservation_for_budget_correction(reservations, reservation, correction, request, metrics)?;
    metrics.record_source_chunk_attempt();
    read_local_parquet_chunk(context, source, chunk, correction.corrected_budget_bytes(), memory_boundary, metrics)
        .map(Some)
        .map_err(|error| correction.attach_to_error(error))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SourceReadReservationContract {
    pub(super) requested_bytes: u64,
    pub(super) planning_bytes: Option<u64>,
    pub(super) estimated_bytes: Option<u64>,
    pub(super) quality: SourceReadReservationContractQuality,
}

pub(super) fn source_read_reservation_contract(
    policy: MemoryControlPolicy,
    chunk: &SourceChunk,
) -> SourceReadReservationContract {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    if let Some(exact_bytes) = chunk.exact_source_read_preflight_bytes {
        return SourceReadReservationContract {
            requested_bytes: exact_bytes,
            planning_bytes: Some(exact_bytes),
            estimated_bytes: chunk.estimated_source_read_bytes,
            quality: SourceReadReservationContractQuality::ExactLowerLevelContract,
        };
    }
    let has_projected_source_read_estimate = chunk.estimated_source_read_bytes.is_some()
        && (chunk.known_projected_uncompressed_bytes.is_some() || chunk.known_pure_filter_uncompressed_bytes.is_some());
    let planning_bytes = if has_projected_source_read_estimate {
        chunk.estimated_source_read_bytes
    } else {
        chunk
            .estimated_chunk_bytes
            .or(chunk.known_projected_uncompressed_bytes)
            .or(chunk.known_row_group_bytes)
            .or(chunk.known_file_bytes)
    };
    let estimated_bytes = chunk
        .estimated_source_read_bytes
        .or(chunk.known_row_group_bytes)
        .or(chunk.estimated_chunk_bytes)
        .or(chunk.known_file_bytes);
    let conservative_bytes = estimated_bytes.unwrap_or(fallback);
    let requested_bytes = match policy {
        MemoryControlPolicy::Unbounded => conservative_bytes,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => conservative_bytes.min(max_source_chunk_bytes),
    };
    SourceReadReservationContract {
        requested_bytes,
        planning_bytes,
        estimated_bytes,
        quality: if has_projected_source_read_estimate {
            SourceReadReservationContractQuality::ProjectedUncompressedEstimate
        } else {
            SourceReadReservationContractQuality::ConservativeEstimate
        },
    }
}

pub(super) fn chunk_reservation_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(chunk.source_node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    if let Some(operator_kind) = legacy_bounded_chunk_operator_kind(kind, memory_boundary) {
        request =
            request.with_compiled_task_id(legacy_bounded_chunk_task_id(operator_kind, memory_boundary, chunk, attempt));
    }
    request
}

fn legacy_bounded_chunk_operator_kind(kind: ReservationKind, memory_boundary: &str) -> Option<&'static str> {
    match kind {
        ReservationKind::SourceRead | ReservationKind::SourceCacheRetained => Some("source_read"),
        ReservationKind::ChunkOperator | ReservationKind::PartialState
            if memory_boundary.contains("filter") || memory_boundary.contains("projection") =>
        {
            Some("unary")
        },
        ReservationKind::ChunkOperator | ReservationKind::PartialState => Some("aggregate"),
        _ => None,
    }
}

fn legacy_bounded_chunk_task_id(
    operator_kind: &str,
    memory_boundary: &str,
    chunk: &SourceChunk,
    attempt: usize,
) -> String {
    format!(
        "legacy_bounded.{operator_kind}.{}.node{}.file{}.chunk{}.attempt{}",
        legacy_bounded_task_id_component(memory_boundary),
        chunk.source_node_id,
        chunk.file_index,
        chunk.chunk_index,
        attempt
    )
}

fn legacy_bounded_task_id_component(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' }).collect()
}

pub(super) fn source_read_reservation_request(
    policy: MemoryControlPolicy,
    kind: ReservationKind,
    memory_boundary: &'static str,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let contract = source_read_reservation_contract(policy, chunk);
    chunk_reservation_request(
        kind,
        memory_boundary,
        contract.requested_bytes,
        budget_bytes,
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(contract.quality.as_str(), contract.planning_bytes, contract.estimated_bytes)
}

pub(super) fn observed_source_read_reservation_request(
    memory_boundary: &'static str,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    chunk_reservation_request(
        ReservationKind::SourceRead,
        memory_boundary,
        0,
        budget_bytes,
        chunk,
        attempt,
        max_attempts,
    )
    .with_source_read_contract(
        SourceReadReservationContractQuality::ObservedGrowShrink.as_str(),
        chunk.estimated_chunk_bytes,
        chunk.estimated_source_read_bytes,
    )
}
