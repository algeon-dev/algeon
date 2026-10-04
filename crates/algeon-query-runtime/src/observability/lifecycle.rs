//! Exactly-once query and task lifecycle guards.

use super::{
    MetricsRecorder, PlanExecutionBoundaryKind, PlanExecutionId, QueryMetricsCheckpoint, RuntimeEventContext,
    RuntimeObservationHandle, RuntimeObservationOwner, RuntimeObservationPayload, RuntimeObservationSession,
    RuntimeSpanId, RuntimeTaskId, RuntimeTaskLifecycleObservation, RuntimeTerminalOutcome,
    SealedRuntimeObservationSnapshot, SourceReadFact, TaskAttemptId, TaskLifecycleState,
};
use crate::{attempt::AttemptHandle, metrics::SourceReadRuntimeMetrics};

/// Owner-only query lifecycle. Normal completion consumes the owner; Drop
/// terminalizes abandoned and unwinding executions without exposing seal
/// authority to workers.
#[derive(Debug)]
pub struct RuntimeQueryLifecycle {
    owner: Option<RuntimeObservationOwner>,
    plan_execution_id: PlanExecutionId,
    plan_span_id: RuntimeSpanId,
}

/// One plan invocation inside a query whose owner outlives native execution.
///
/// Finishing this lifecycle records and aggregates the plan outcome, but the
/// admitted query owner remains responsible for cleanup boundaries and sealing.
#[derive(Debug)]
pub struct RuntimePlanExecutionLifecycle {
    observation: RuntimeObservationHandle,
    /// The standalone query owner a finished plan's outcome goes to. An
    /// attempt's plans record only their own boundaries: the attempt commits
    /// its own outcome.
    owner: Option<RuntimeObservationOwner>,
    plan_execution_id: PlanExecutionId,
    plan_span_id: RuntimeSpanId,
    finished: bool,
}

/// Cloneable identity and recording capability for descendants of one plan
/// execution. It deliberately has no seal authority.
#[derive(Debug, Clone)]
pub struct RuntimeExecutionObservationScope {
    observation: RuntimeObservationHandle,
    plan_execution_id: PlanExecutionId,
    plan_span_id: RuntimeSpanId,
    metrics: MetricsRecorder,
}

impl RuntimeExecutionObservationScope {
    /// Returns worker recording access.
    pub const fn observation(&self) -> &RuntimeObservationHandle {
        &self.observation
    }

    /// Returns the plan invocation identity.
    pub const fn plan_execution_id(&self) -> &PlanExecutionId {
        &self.plan_execution_id
    }

    /// Returns the parent span for task attempts.
    pub const fn plan_span_id(&self) -> RuntimeSpanId {
        self.plan_span_id
    }

    /// A scope on `observation` whose metrics go to `metrics`, for driving
    /// plan stages outside a plan-execution lifecycle.
    pub fn for_metrics(observation: RuntimeObservationHandle, metrics: MetricsRecorder) -> Self {
        Self {
            observation,
            plan_execution_id: metrics.plan_execution_id().clone(),
            plan_span_id: RuntimeSpanId::new(1),
            metrics,
        }
    }

    /// Where this plan execution's metrics are reduced.
    pub const fn metrics(&self) -> &MetricsRecorder {
        &self.metrics
    }

    /// Reports this plan execution's metrics to `metrics` instead of the trace
    /// session, for an execution whose metrics owner is bound separately.
    #[must_use]
    pub fn with_metrics(mut self, metrics: MetricsRecorder) -> Self {
        self.metrics = metrics;
        self
    }

    /// Folds one source-read fact into this plan execution's exact totals.
    pub fn record_source_read(&self, fact: SourceReadFact) {
        self.metrics.record_source_read(fact);
    }

    /// Offers one source-read detail row for this plan execution's bounded retention.
    pub fn record_source_read_row(&self, row: SourceReadRuntimeMetrics) {
        self.metrics.record_source_read_runtime(row);
    }

    /// Returns one internally consistent view of this plan execution's metrics.
    pub fn metrics_checkpoint(&self) -> Option<QueryMetricsCheckpoint> {
        self.metrics.checkpoint()
    }

    fn new(
        observation: RuntimeObservationHandle,
        plan_execution_id: PlanExecutionId,
        plan_span_id: RuntimeSpanId,
    ) -> Self {
        let metrics = MetricsRecorder::new(observation.clone(), plan_execution_id.clone());
        Self { observation, plan_execution_id, plan_span_id, metrics }
    }
}

impl RuntimeQueryLifecycle {
    /// Opens one query and plan-execution lifecycle.
    pub fn new(session: RuntimeObservationSession, plan_execution_id: PlanExecutionId) -> Self {
        Self::from_owner(RuntimeObservationOwner::from_session(session), plan_execution_id)
    }

    /// Opens one lifecycle using a shared owner created before admission.
    pub fn from_owner(owner: RuntimeObservationOwner, plan_execution_id: PlanExecutionId) -> Self {
        let lifecycle = Self { owner: Some(owner), plan_execution_id, plan_span_id: RuntimeSpanId::new(1) };
        lifecycle.record_plan(PlanExecutionBoundaryKind::Started);
        lifecycle
    }

    /// Returns the invocation identity attached to all descendant events.
    pub fn plan_execution_id(&self) -> &PlanExecutionId {
        &self.plan_execution_id
    }

    /// Returns the root span for descendant task attempts.
    pub const fn plan_span_id(&self) -> RuntimeSpanId {
        self.plan_span_id
    }

    /// Returns cloneable recording access without terminal authority.
    pub fn handle(&self) -> RuntimeObservationHandle {
        self.owner.as_ref().map_or_else(RuntimeObservationHandle::disabled, RuntimeObservationOwner::handle)
    }

    /// Returns the non-owning scope passed to schedulers and reservations.
    pub fn execution_scope(&self) -> RuntimeExecutionObservationScope {
        RuntimeExecutionObservationScope::new(self.handle(), self.plan_execution_id.clone(), self.plan_span_id)
    }

    /// Records cleanup and exactly one terminal, seals the owner, and consumes
    /// the lifecycle so Drop cannot emit a second terminal.
    pub fn finish(mut self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        self.terminate(outcome)
    }

    fn terminate(&mut self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        let plan_boundary = match &outcome {
            RuntimeTerminalOutcome::Succeeded => PlanExecutionBoundaryKind::Finished,
            RuntimeTerminalOutcome::Cancelled => PlanExecutionBoundaryKind::Cancelled,
            RuntimeTerminalOutcome::Failed { .. }
            | RuntimeTerminalOutcome::Dropped
            | RuntimeTerminalOutcome::Panicked => PlanExecutionBoundaryKind::Failed,
        };
        self.record_plan(plan_boundary);
        self.owner.take().and_then(|owner| owner.finish_query(Some(outcome)))
    }

    fn record_plan(&self, boundary: PlanExecutionBoundaryKind) {
        if let Some(owner) = &self.owner {
            owner.handle().record_with_context(
                RuntimeEventContext::for_plan(self.plan_execution_id.clone()).with_span(self.plan_span_id, None),
                || RuntimeObservationPayload::PlanExecutionBoundary { boundary },
            );
        }
    }
}

impl RuntimePlanExecutionLifecycle {
    /// Opens a plan invocation using query-level owner authority retained by a
    /// caller such as the query-service admission ticket.
    pub fn from_owner(owner: RuntimeObservationOwner, plan_execution_id: PlanExecutionId) -> Self {
        Self::open(owner.handle(), Some(owner), plan_execution_id)
    }

    /// Opens a plan invocation of an admitted attempt.
    pub(crate) fn from_attempt(attempt: &AttemptHandle, plan_execution_id: PlanExecutionId) -> Self {
        Self::open(attempt.observation(), None, plan_execution_id)
    }

    fn open(
        observation: RuntimeObservationHandle,
        owner: Option<RuntimeObservationOwner>,
        plan_execution_id: PlanExecutionId,
    ) -> Self {
        let lifecycle =
            Self { observation, owner, plan_execution_id, plan_span_id: RuntimeSpanId::new(1), finished: false };
        lifecycle.record_plan(PlanExecutionBoundaryKind::Started);
        lifecycle
    }

    /// Opens a plan invocation on a coordinator-owned query observation.
    ///
    /// The handle can record plan/task evidence but carries no terminal-seal
    /// authority. The external query coordinator owns the sole query terminal.
    pub fn from_handle(observation: RuntimeObservationHandle, plan_execution_id: PlanExecutionId) -> Self {
        Self::open(observation, None, plan_execution_id)
    }

    /// Returns cloneable recording access without terminal authority.
    pub fn handle(&self) -> RuntimeObservationHandle {
        self.observation.clone()
    }

    /// Returns the non-owning scope passed to schedulers and reservations.
    pub fn execution_scope(&self) -> RuntimeExecutionObservationScope {
        RuntimeExecutionObservationScope::new(self.handle(), self.plan_execution_id.clone(), self.plan_span_id)
    }

    /// Records the plan terminal and contributes its outcome to the query
    /// owner without sealing the shared observation session.
    pub fn finish(mut self, outcome: RuntimeTerminalOutcome) {
        self.terminate(outcome);
    }

    fn terminate(&mut self, outcome: RuntimeTerminalOutcome) {
        let boundary = match &outcome {
            RuntimeTerminalOutcome::Succeeded => PlanExecutionBoundaryKind::Finished,
            RuntimeTerminalOutcome::Cancelled => PlanExecutionBoundaryKind::Cancelled,
            RuntimeTerminalOutcome::Failed { .. }
            | RuntimeTerminalOutcome::Dropped
            | RuntimeTerminalOutcome::Panicked => PlanExecutionBoundaryKind::Failed,
        };
        self.record_plan(boundary);
        self.finished = true;
        if let Some(owner) = self.owner.take() {
            owner.record_terminal_outcome(outcome);
        }
    }

    fn record_plan(&self, boundary: PlanExecutionBoundaryKind) {
        self.observation.record_with_context(
            RuntimeEventContext::for_plan(self.plan_execution_id.clone()).with_span(self.plan_span_id, None),
            || RuntimeObservationPayload::PlanExecutionBoundary { boundary },
        );
    }
}

impl Drop for RuntimePlanExecutionLifecycle {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let outcome =
            if std::thread::panicking() { RuntimeTerminalOutcome::Panicked } else { RuntimeTerminalOutcome::Dropped };
        self.terminate(outcome);
    }
}

impl Drop for RuntimeQueryLifecycle {
    fn drop(&mut self) {
        if self.owner.is_none() {
            return;
        }
        let outcome =
            if std::thread::panicking() { RuntimeTerminalOutcome::Panicked } else { RuntimeTerminalOutcome::Dropped };
        let _ = self.terminate(outcome);
    }
}

/// Exactly-once lifecycle for one compiled task attempt.
#[derive(Debug)]
pub struct RuntimeTaskLifecycle {
    observation: RuntimeObservationHandle,
    context: RuntimeEventContext,
    attempt_id: TaskAttemptId,
    task_kind: String,
    terminal_recorded: bool,
}

impl RuntimeTaskLifecycle {
    /// Emits ready and start for one explicit task attempt.
    pub fn new(
        observation: RuntimeObservationHandle,
        plan_execution_id: PlanExecutionId,
        attempt_id: TaskAttemptId,
        task_kind: impl Into<String>,
        span_id: RuntimeSpanId,
        parent_span_id: RuntimeSpanId,
    ) -> Self {
        let lifecycle = Self {
            observation,
            context: RuntimeEventContext::for_plan(plan_execution_id).with_span(span_id, Some(parent_span_id)),
            attempt_id,
            task_kind: task_kind.into(),
            terminal_recorded: false,
        };
        lifecycle.record(TaskLifecycleState::Ready);
        lifecycle.record(TaskLifecycleState::Started);
        lifecycle
    }

    /// Emits the sole successful terminal for this task attempt.
    pub fn finish(mut self) {
        self.record_terminal(TaskLifecycleState::Finished);
    }

    /// Emits the sole structured failure or cancellation terminal.
    pub fn fail(mut self, error_code: impl Into<String>, cancelled: bool) {
        let lifecycle = if cancelled {
            TaskLifecycleState::Cancelled
        } else {
            TaskLifecycleState::Failed { error_code: error_code.into() }
        };
        self.record_terminal(lifecycle);
    }

    /// Emits the sole panic terminal after a scheduler catches an unwind long
    /// enough to release live resources before resuming it.
    pub fn panicked(mut self) {
        self.record_terminal(TaskLifecycleState::Panicked);
    }

    fn record_terminal(&mut self, lifecycle: TaskLifecycleState) {
        if self.terminal_recorded {
            return;
        }
        self.record(lifecycle);
        self.terminal_recorded = true;
    }

    fn record(&self, lifecycle: TaskLifecycleState) {
        self.observation.record_with_context(self.context.clone(), || {
            RuntimeObservationPayload::TaskLifecycle(Box::new(RuntimeTaskLifecycleObservation {
                attempt_id: self.attempt_id.clone(),
                task_kind: self.task_kind.clone(),
                lifecycle,
            }))
        });
    }
}

impl Drop for RuntimeTaskLifecycle {
    fn drop(&mut self) {
        if self.terminal_recorded {
            return;
        }
        let lifecycle =
            if std::thread::panicking() { TaskLifecycleState::Panicked } else { TaskLifecycleState::Dropped };
        self.record_terminal(lifecycle);
    }
}

/// Builds one task-attempt identity from an explicit task and attempt number.
pub fn task_attempt_id(task_id: impl Into<RuntimeTaskId>, attempt: u32) -> TaskAttemptId {
    TaskAttemptId::new(task_id.into(), super::RuntimeAttemptId::new(attempt))
}

#[cfg(test)]
mod tests {
    use query_planning::correlation::QueryId;

    use super::*;
    use crate::observability::RuntimeObservationConfig;

    #[test]
    fn coordinator_handle_plan_lifecycle_records_without_sealing_query_root() {
        let owner = RuntimeObservationOwner::new(
            QueryId::new("coordinator-owned"),
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        );
        let lifecycle =
            RuntimePlanExecutionLifecycle::from_handle(owner.handle(), PlanExecutionId::new("coordinator-plan"));

        lifecycle.finish(RuntimeTerminalOutcome::Succeeded);

        assert!(owner.sealed_snapshot().is_none(), "plan completion must not seal a coordinator-owned query root");
        let snapshot = owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).unwrap();
        assert!(snapshot.events().iter().any(|event| matches!(
            event.payload,
            RuntimeObservationPayload::PlanExecutionBoundary { boundary: PlanExecutionBoundaryKind::Finished }
        )));
    }
}
