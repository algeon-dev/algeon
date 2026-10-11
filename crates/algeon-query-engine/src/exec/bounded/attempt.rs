use query_runtime::observability::SourceReadFact;

use super::{ExecutionContext, ExecutionMetrics};

/// One admitted bounded execution attempt.
///
/// The immutable cap is a batching input. Allocation enforcement and resident
/// ownership belong to the query memory domain and grant.
///
/// This is a borrowed view, cheap to build: the coordinator holds one over the
/// root context, and every lane job builds its own over the lane's derived
/// context so operators run on the lane's stream while reporting into the
/// same metrics owner.
pub(in crate::exec) struct BoundedExecutionAttempt<'a> {
    context: &'a ExecutionContext,
    /// Cap applied to every budget token created under this attempt. Tokens
    /// label stages for diagnostics; the allocator member enforces the grant.
    device_budget_bytes: u64,
    metrics: &'a ExecutionMetrics,
}

impl<'a> BoundedExecutionAttempt<'a> {
    pub(in crate::exec::bounded) fn new(
        context: &'a ExecutionContext,
        device_budget_bytes: u64,
        metrics: &'a ExecutionMetrics,
    ) -> Self {
        Self { context, device_budget_bytes, metrics }
    }

    pub(in crate::exec) const fn context(&self) -> &'a ExecutionContext {
        self.context
    }

    /// Where this attempt reports its metrics. The attempt holds no metrics of
    /// its own; the plan execution's observation owner is the only store.
    pub(in crate::exec) const fn metrics(&self) -> &'a ExecutionMetrics {
        self.metrics
    }

    /// Folds one source-read fact into this attempt's plan execution.
    pub(in crate::exec) fn record_source_read(&self, fact: SourceReadFact) {
        self.metrics.record_source_read(fact);
    }

    pub(in crate::exec) const fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }
}
