use std::time::Instant;

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
use sizing::{
    estimate_rows_bytes, estimated_bytes_per_row, estimated_limited_output_bytes, final_sort_retained_bytes,
    final_sort_rows_per_chunk,
};
use topk_candidate::*;
use topk_level::*;

pub(super) fn execute_final_sort_limit_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
) -> Result<CommittedPartial> {
    let Some(sort) = bounded.sort else {
        return Ok(input);
    };
    execute_final_sort_node_with_retry(
        execution,
        bounded.aggregate_id,
        sort,
        bounded.limit.map(|(_, limit)| limit),
        input,
    )
}

pub(super) fn execute_final_sort_node_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: CommittedPartial,
) -> Result<CommittedPartial> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = execution.context().memory_control_policy() else {
        unreachable!("execute_final_sort_limit_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { execution, node_id, sort, max_attempts };
    let input_rows = input.frame.num_rows() as u64;
    let (observation, plan_execution_id, parent_span_id) =
        crate::observability::stage_context_from_execution(final_sort.execution.context());
    let stage_guard = crate::observability::RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        crate::observability::RuntimeOperatorStageKind::FinalSort,
        crate::observability::RuntimeOperatorStageIdentity::new()
            .with_node_id(node_id.to_string())
            .with_input_rows(input_rows),
    );

    if let Some(limit) = limit {
        match final_sort.should_use_hierarchical_topk(limit, &input) {
            Ok(true) => {
                let result = final_sort.hierarchical_topk(limit, input, 1);
                final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
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
                final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
                stage_guard.fail(&error);
                return Err(error);
            },
        }
    }

    let result = final_sort.full_sort_limit_attempt(limit, &input, 1);

    match result {
        Ok(output) => {
            final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
            stage_guard.finish(Some(output.frame.num_rows() as u64), None);
            Ok(output)
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.execution.metrics_mut());
            if max_attempts > 1
                && let Some(limit) = limit
            {
                if let Err(backoff_error) = crate::exec::retry::backoff_before_retry(
                    final_sort.execution.context(),
                    "nexus_query_engine.exec.bounded.final_sort_hierarchical_topk_retry",
                ) {
                    stage_guard.fail(&backoff_error);
                    return Err(backoff_error);
                }
                match final_sort.hierarchical_topk(limit, input, 2) {
                    Ok(output) => {
                        final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
                        stage_guard.finish(Some(output.frame.num_rows() as u64), None);
                        Ok(output)
                    },
                    Err(error) => {
                        final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
                        stage_guard.fail(&error);
                        Err(error)
                    },
                }
            } else {
                final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
                let error = final_sort_retry_facts("retry_attempts_exhausted", final_sort.execution.metrics_mut())
                    .attach_to_error(error);
                stage_guard.fail(&error);
                Err(error)
            }
        },
        Err(error) => {
            final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
            stage_guard.fail(&error);
            Err(error)
        },
    }
}

pub(in crate::exec::bounded) fn execute_final_sort_node_from_committed_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    execute_final_sort_node_with_retry(execution, node_id, sort, limit, input)
}

pub(in crate::exec::bounded) fn execute_final_sort_node_from_frame_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    input_frame: GpuDataFrame,
) -> Result<ResidentGpuFrame> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = execution.context().memory_control_policy() else {
        unreachable!("execute_final_sort_node_from_frame_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { execution, node_id, sort, max_attempts };
    let result = final_sort.full_sort_limit_attempt_from_frame(None, &input_frame, 1);

    match result {
        Ok(output) => {
            final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
            Ok(ResidentGpuFrame { frame: output.frame, device_bytes: output.device_bytes })
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.execution.metrics_mut());
            final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
            Err(final_sort_retry_facts("retry_attempts_exhausted", final_sort.execution.metrics_mut())
                .attach_to_error(error))
        },
        Err(error) => {
            final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
            Err(error)
        },
    }
}

pub(in crate::exec::bounded) fn execute_final_sort_node_from_stream_with_retry(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    sort: &SortNode,
    limit: &LimitNode,
    mut input_stream: ResidentFrameStreamPayload,
) -> Result<ResidentGpuFrame> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = execution.context().memory_control_policy() else {
        unreachable!("execute_final_sort_node_from_stream_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let mut final_sort = FinalSortExecutor { execution, node_id, sort, max_attempts };
    let result = final_sort.hierarchical_topk_stream(limit, &mut input_stream, 1);
    input_stream.release_producer(final_sort.execution);
    final_sort.execution.metrics_mut().final_sort_time += timer.elapsed();
    result
}

struct FinalSortExecutor<'a, 'e> {
    execution: &'a mut BoundedExecutionAttempt<'e>,
    node_id: NodeId,
    sort: &'a SortNode,
    max_attempts: usize,
}

fn final_sort_runtime_facts(
    memory_boundary: &'static str,
    node_id: NodeId,
    attempt: usize,
    max_attempts: usize,
    metrics: &mut QueryMetrics,
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

impl FinalSortExecutor<'_, '_> {
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

    fn reserve_estimated_final_sort_output(&mut self, output_bytes: u64, attempt: usize) -> Result<u64> {
        reserve_final_sort_output_bytes(self.execution, self.node_id, output_bytes, attempt, self.max_attempts)
    }

    fn finalize_final_sort_output_reservation(
        &mut self,
        frame: GpuDataFrame,
        reservation: u64,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        finalize_final_sort_output_reservation(
            self.execution,
            self.node_id,
            frame,
            reservation,
            attempt,
            self.max_attempts,
        )
    }

    fn attach_final_sort_resource_facts(&self, error: QueryError, attempt: usize) -> QueryError {
        if crate::exec::retry::resource_failure_recovery(&error)
            == crate::exec::retry::ResourceFailureRecovery::NotResource
        {
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

        let input_bytes = input.device_bytes();
        let limited_output_bytes = estimate_rows_bytes(estimated_bytes_per_row(input_bytes, input_rows), keep_rows);
        let selective_topk_threshold = input_bytes.div_ceil(2).max(1);
        if limited_output_bytes <= selective_topk_threshold {
            return Ok(true);
        }

        let MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } =
            self.execution.context().memory_control_policy()
        else {
            unreachable!("final sort execution is called only for bounded policies");
        };
        Ok(input_bytes > max_source_chunk_bytes && limited_output_bytes <= max_source_chunk_bytes)
    }
}
