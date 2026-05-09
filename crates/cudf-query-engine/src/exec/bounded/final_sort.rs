use cudf::{ExecutionBudgetReport, ExecutionBudgetRequest};

use super::*;

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

impl<'a, 'plan> FinalSortExecutor<'a, 'plan> {
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
            self.bounded,
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
            self.bounded,
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
            .with_fact("node_id", self.bounded.aggregate_id)
            .with_fact("attempt", attempt)
            .with_fact("max_attempts", self.max_attempts)
    }
}
