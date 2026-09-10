use super::{ExecutionContext, QueryMetrics, Result, RuntimeExecutionObservationScope};

/// One admitted bounded execution attempt.
///
/// The immutable cap is a batching input. Allocation enforcement and resident
/// ownership belong to the query memory domain and grant.
pub(in crate::exec) struct BoundedExecutionAttempt<'a> {
    context: &'a ExecutionContext,
    device_budget_bytes: u64,
    metrics: &'a mut QueryMetrics,
}

impl<'a> BoundedExecutionAttempt<'a> {
    pub(in crate::exec::bounded) fn new(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a mut QueryMetrics,
        _observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Result<Self> {
        Ok(Self { context, device_budget_bytes, metrics })
    }

    pub(in crate::exec) const fn context(&self) -> &'a ExecutionContext {
        self.context
    }

    pub(in crate::exec) fn metrics(&self) -> &QueryMetrics {
        self.metrics
    }

    pub(in crate::exec) fn metrics_mut(&mut self) -> &mut QueryMetrics {
        self.metrics
    }

    pub(in crate::exec) const fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }
}
