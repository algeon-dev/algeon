use std::time::Duration;

use super::{ExecutionContext, QueryError, QueryErrorCode, QueryMetrics, Result};

const BOUNDED_RETRY_BACKOFF: Duration = Duration::from_millis(50);

pub(in crate::exec) fn is_retryable_resource_error(error: &QueryError) -> bool {
    error.code() == QueryErrorCode::CudfResourceExhausted
}

pub(in crate::exec) fn record_retryable_resource_error(error: &QueryError, metrics: &mut QueryMetrics) {
    if error.code() == QueryErrorCode::CudfResourceExhausted {
        metrics.record_retry_resource_exhausted();
    }
}

pub(in crate::exec) fn backoff_before_retry(context: &ExecutionContext, stage: &'static str) -> Result<()> {
    context.check_cancellation(stage)?;
    std::thread::sleep(BOUNDED_RETRY_BACKOFF);
    context.check_cancellation(stage)
}
