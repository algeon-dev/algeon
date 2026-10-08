use super::{ExecutionMetrics, QueryError, RetryFacts};
use crate::exec::retry::{is_retryable_resource_error, record_retryable_resource_error};

/// Final sort retries share the resource-failure classification of `exec::retry`: only a
/// query-cap rejection is retryable.
pub(super) fn is_retryable_final_sort_error(error: &QueryError) -> bool {
    is_retryable_resource_error(error)
}

pub(super) fn record_final_sort_retry_error(error: &QueryError, metrics: &ExecutionMetrics) {
    record_retryable_resource_error(error, metrics);
}

/// Retry facts for the final sort stage: the action taken and the chunks and partials
/// committed before the failure.
pub(super) fn final_sort_retry_facts(retry_action: &'static str, metrics: &ExecutionMetrics) -> RetryFacts {
    let progress = metrics.progress();
    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_sort")
        .with_committed_chunks(progress.source_chunks_committed as usize)
        .with_committed_partials(progress.partial_aggregate_count as usize)
}

/// Retry facts for a failed merge level. A fan-in of 2 or less cannot shrink further and
/// reports `fail_min_topk_fan_in`; a larger one reports exhausted attempts.
pub(super) fn final_sort_topk_retry_facts(
    merge_level: usize,
    merge_fan_in: usize,
    metrics: &ExecutionMetrics,
) -> RetryFacts {
    final_sort_retry_facts(
        if merge_fan_in <= 2 { "fail_min_topk_fan_in" } else { "hierarchical_topk_retry_exhausted" },
        metrics,
    )
    .with_merge_level(merge_level)
    .with_merge_fan_in(merge_fan_in)
}
