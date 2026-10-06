use std::time::Instant;

use cudf::{ExecutionBudgetReport, ExecutionBudgetRequest};
use query_planning::chunk::SourceChunkPlanningPolicy;

use super::{compiled::ResidentFrameStreamPayload, *};

mod full_sort;
mod hierarchical_topk;
mod limit;
mod retry;
mod sizing;
mod sort_lane;
mod topk_candidate;
mod topk_level;

use limit::*;
use query_runtime::observability::{
    RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
};
use retry::*;
use sizing::{estimate_rows_bytes, estimated_bytes_per_row, final_sort_retained_bytes, final_sort_rows_per_chunk};
use sort_lane::*;
use topk_candidate::*;
use topk_level::*;

use super::compiled::lanes::AggregateLaneDispatch;
use crate::exec::retry::{ResourceFailureRecovery, RetryLimits, backoff_before_retry, resource_failure_recovery};

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

struct FinalSortExecutor<'a, 'e, 'd> {
    execution: &'a mut BoundedExecutionAttempt<'e>,
    dispatch: Option<&'d AggregateLaneDispatch>,
    node_id: NodeId,
    sort: &'a SortNode,
    max_attempts: usize,
}

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
