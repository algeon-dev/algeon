use std::time::Duration;

use query_planning::execution_contracts::MemoryControlPolicy;

use super::{ExecutionContext, ExecutionMetrics, QueryError, QueryErrorCode, Result};

const BOUNDED_RETRY_BACKOFF: Duration = Duration::from_millis(50);

/// What a bounded retry boundary may do after one typed resource failure.
///
/// The classification follows the failure's proven source, not its coarse
/// `ResourceExhausted` kind: only an allocation the query-domain allocator
/// rejected against this attempt's immutable cap is evidence that a smaller
/// working set can succeed inside the same grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) enum ResourceFailureRecovery {
    /// The failed unit of work has been released by the boundary that owned it
    /// and the boundary may retry with a smaller batch or fan-in under the same
    /// grant, bounded by the policy's retry attempts and minimum granularity.
    ShrinkWorkingSet,
    /// Physical device exhaustion or an allocation failure without proven
    /// source. Neither proves that this attempt's working set is the cause, so
    /// the attempt fails with the original typed cause; whole-attempt retry
    /// stays with the caller after the sealed report and cleanup.
    Terminal,
    /// Not a resource failure.
    NotResource,
}

pub(in crate::exec) fn resource_failure_recovery(error: &QueryError) -> ResourceFailureRecovery {
    match error.code() {
        QueryErrorCode::QueryMemoryLimitExceeded => ResourceFailureRecovery::ShrinkWorkingSet,
        QueryErrorCode::DeviceOutOfMemory | QueryErrorCode::DeviceAllocationFailed => ResourceFailureRecovery::Terminal,
        _ => ResourceFailureRecovery::NotResource,
    }
}

pub(in crate::exec) fn is_retryable_resource_error(error: &QueryError) -> bool {
    resource_failure_recovery(error) == ResourceFailureRecovery::ShrinkWorkingSet
}

pub(in crate::exec) fn record_retryable_resource_error(error: &QueryError, metrics: &ExecutionMetrics) {
    if is_retryable_resource_error(error) {
        metrics.record_retry_resource_exhausted();
    }
}

/// Waits before a retry. Cancellation or the deadline ends the wait at once.
pub(in crate::exec) fn backoff_before_retry(context: &ExecutionContext, stage: &'static str) -> Result<()> {
    context.check_cancellation(stage)?;
    context.cancellation().sleep(BOUNDED_RETRY_BACKOFF, stage)?;
    context.check_cancellation(stage)
}

/// How far a bounded retry boundary may go: at most `max_attempts` tries of
/// one unit, and no source split below `min_split_row_groups` row groups.
/// Context construction validates both policy fields as nonzero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec) struct RetryLimits {
    pub(in crate::exec) max_attempts: usize,
    pub(in crate::exec) min_split_row_groups: usize,
}

impl RetryLimits {
    pub(in crate::exec) fn of(context: &ExecutionContext) -> Self {
        let MemoryControlPolicy { max_retry_attempts, min_retry_chunk_row_groups, .. } =
            context.memory_control_policy();
        Self { max_attempts: max_retry_attempts, min_split_row_groups: min_retry_chunk_row_groups }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error(code: QueryErrorCode) -> QueryError {
        QueryError::new(code, "query_engine.retry.test", "typed resource failure")
    }

    #[test]
    fn query_cap_rejections_shrink_and_device_failures_are_terminal() {
        assert_eq!(
            resource_failure_recovery(&error(QueryErrorCode::QueryMemoryLimitExceeded)),
            ResourceFailureRecovery::ShrinkWorkingSet
        );
        assert_eq!(
            resource_failure_recovery(&error(QueryErrorCode::DeviceOutOfMemory)),
            ResourceFailureRecovery::Terminal
        );
        assert_eq!(
            resource_failure_recovery(&error(QueryErrorCode::DeviceAllocationFailed)),
            ResourceFailureRecovery::Terminal
        );
        assert_eq!(
            resource_failure_recovery(&error(QueryErrorCode::CudfDependencyFailure)),
            ResourceFailureRecovery::NotResource
        );
        assert!(is_retryable_resource_error(&error(QueryErrorCode::QueryMemoryLimitExceeded)));
        assert!(!is_retryable_resource_error(&error(QueryErrorCode::DeviceOutOfMemory)));
    }

    #[test]
    fn retry_evidence_counts_only_recoverable_failures() {
        let metrics = ExecutionMetrics::detached("retry-evidence");
        record_retryable_resource_error(&error(QueryErrorCode::QueryMemoryLimitExceeded), &metrics);
        record_retryable_resource_error(&error(QueryErrorCode::DeviceOutOfMemory), &metrics);
        record_retryable_resource_error(&error(QueryErrorCode::DeviceAllocationFailed), &metrics);
        assert_eq!(metrics.snapshot().retry_resource_exhausted_count, 1);
    }
}
