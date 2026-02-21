use cudf::{ExecutionBudgetReport, ExecutionBudgetRequest};

use super::{compiled::ResidentFrameStreamPayload, *};

mod full_sort;
mod hierarchical_topk;
mod limit;
mod reservation;
mod retry;
mod sizing;
mod topk_candidate;
mod topk_level;

use limit::*;
use reservation::*;
use retry::*;
pub(in crate::exec) use sizing::topk_candidate_rows;
use sizing::{
    estimate_rows_bytes, estimated_bytes_per_row, estimated_limited_output_bytes, final_sort_retained_bytes,
    final_sort_rows_per_chunk,
};
use topk_candidate::*;
use topk_level::*;

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
    execute_final_sort_node_with_retry(context, bounded.aggregate_id, sort, bounded.limit, input, reservations, metrics)
}

pub(super) fn execute_final_sort_node_with_retry(
    context: &ExecutionContext,
    node_id: NodeId,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_sort_limit_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { context, node_id, sort, max_attempts, reservations, metrics };

    if let Some(limit) = limit
        && final_sort.should_use_hierarchical_topk(limit, &input)?
    {
        let result = final_sort.hierarchical_topk(limit, input, 1);
        final_sort.metrics.final_sort_time += timer.elapsed();
        return result;
    }

    let result = final_sort.full_sort_limit_attempt(limit, &input, 1);

    match result {
        Ok(output) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            final_sort.reservations.release(input.reservation, final_sort.metrics);
            Ok(output)
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.metrics);
            if max_attempts > 1
                && let Some(limit) = limit
            {
                match final_sort.hierarchical_topk(limit, input, 2) {
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
                final_sort.reservations.release(input.reservation, final_sort.metrics);
                Err(final_sort_retry_facts("retry_attempts_exhausted", final_sort.metrics).attach_to_error(error))
            }
        },
        Err(error) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            final_sort.reservations.release(input.reservation, final_sort.metrics);
            Err(error)
        },
    }
}

pub(in crate::exec::bounded) fn execute_final_sort_node_from_committed_with_retry(
    context: &ExecutionContext,
    node_id: NodeId,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    execute_final_sort_node_with_retry(context, node_id, sort, limit, input, reservations, metrics)
}

pub(in crate::exec::bounded) fn execute_final_sort_node_from_frame_with_retry(
    context: &ExecutionContext,
    node_id: NodeId,
    sort: &SortNode,
    input_frame: GpuDataFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_sort_node_from_frame_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { context, node_id, sort, max_attempts, reservations, metrics };
    let result = final_sort.full_sort_limit_attempt_from_frame(None, &input_frame, 1);

    match result {
        Ok(output) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            Ok(ResidentGpuFrame { frame: output.frame, reservation: output.reservation })
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.metrics);
            final_sort.metrics.final_sort_time += timer.elapsed();
            Err(final_sort_retry_facts("retry_attempts_exhausted", final_sort.metrics).attach_to_error(error))
        },
        Err(error) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            Err(error)
        },
    }
}

pub(in crate::exec::bounded) fn execute_final_sort_node_from_stream_with_retry(
    context: &ExecutionContext,
    node_id: NodeId,
    sort: &SortNode,
    limit: &LimitNode,
    mut input_stream: ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_sort_node_from_stream_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { context, node_id, sort, max_attempts, reservations, metrics };
    let result = final_sort.hierarchical_topk_stream(limit, &mut input_stream, 1);
    input_stream.release_producer_reservations(final_sort.reservations, final_sort.metrics);
    final_sort.metrics.final_sort_time += timer.elapsed();
    result
}

struct FinalSortExecutor<'a> {
    context: &'a ExecutionContext,
    node_id: NodeId,
    sort: &'a SortNode,
    max_attempts: usize,
    reservations: &'a mut ReservationTracker,
    metrics: &'a mut QueryMetrics,
}

impl<'a> FinalSortExecutor<'a> {
    fn final_sort_budget_token(&self, boundary: &'static str, attempt: usize) -> Result<cudf::ExecutionBudgetToken> {
        self.final_sort_budget_token_for_bytes(boundary, attempt, self.reservations.remaining_bytes())
    }

    fn final_sort_budget_token_including(
        &self,
        boundary: &'static str,
        attempt: usize,
        reservation: &ActiveReservation,
    ) -> Result<cudf::ExecutionBudgetToken> {
        self.final_sort_budget_token_for_bytes(
            boundary,
            attempt,
            self.reservations.available_bytes_including(reservation),
        )
    }

    fn final_sort_budget_token_for_bytes(
        &self,
        boundary: &'static str,
        attempt: usize,
        budget_bytes: u64,
    ) -> Result<cudf::ExecutionBudgetToken> {
        let budget = usize::try_from(budget_bytes).unwrap_or(usize::MAX);
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

    fn reserve_estimated_final_sort_output(&mut self, output_bytes: u64, attempt: usize) -> Result<ActiveReservation> {
        reserve_final_sort_output_bytes(
            self.node_id,
            output_bytes,
            attempt,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
    }

    fn finalize_final_sort_output_reservation(
        &mut self,
        frame: GpuDataFrame,
        reservation: ActiveReservation,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        finalize_final_sort_output_reservation(
            self.node_id,
            frame,
            reservation,
            attempt,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
    }

    fn attach_final_sort_resource_facts(&self, error: QueryError, attempt: usize) -> QueryError {
        if error.code() != QueryErrorCode::CudfResourceExhausted {
            return error;
        }

        error
            .with_fact("memory_boundary", "final_sort")
            .with_fact("stage", "final_sort")
            .with_fact("node_id", self.node_id)
            .with_fact("attempt", attempt)
            .with_fact("max_attempts", self.max_attempts)
    }

    fn should_use_hierarchical_topk(&self, limit: &LimitNode, input: &CommittedPartial) -> Result<bool> {
        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            return Ok(true);
        }

        let input_rows = input.frame.num_rows();
        if input_rows == 0 || keep_rows >= input_rows {
            return Ok(false);
        }

        let input_bytes = input.reservation.bytes();
        let limited_output_bytes = estimate_rows_bytes(estimated_bytes_per_row(input_bytes, input_rows), keep_rows);
        let selective_topk_threshold = input_bytes.div_ceil(2).max(1);
        if limited_output_bytes <= selective_topk_threshold {
            return Ok(true);
        }

        let MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } = self.context.memory_control_policy() else {
            unreachable!("final sort execution is called only for bounded policies");
        };
        Ok(input_bytes > max_source_chunk_bytes && limited_output_bytes <= max_source_chunk_bytes)
    }
}
