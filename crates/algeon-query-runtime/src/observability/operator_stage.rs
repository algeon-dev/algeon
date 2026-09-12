//! RAII operator-substage observations for blocking native primitives.
//!
//! Emit [`RuntimeOperatorStageState::Started`] immediately before an FFI call
//! and exactly one terminal state when control returns or unwinds. Pair events
//! share the same [`RuntimeSpanId`].

use std::sync::atomic::{AtomicU64, Ordering};

use query_planning::error::{QueryError, QueryErrorCode};

use super::{
    PlanExecutionId, RuntimeEventContext, RuntimeObservationHandle, RuntimeObservationPayload,
    RuntimeOperatorStageKind, RuntimeOperatorStageObservation, RuntimeOperatorStageState, RuntimeSpanId, RuntimeTaskId,
};

/// Process-local high span ids reserved for operator substages so they do not
/// collide with the small deterministic planner/task span scheme.
static NEXT_OPERATOR_STAGE_SPAN_ID: AtomicU64 = AtomicU64::new(1 << 32);

/// Stable identities that travel with every Started/terminal pair.
#[derive(Debug, Clone, Default)]
pub struct RuntimeOperatorStageIdentity {
    pub node_id: Option<String>,
    pub task_id: Option<RuntimeTaskId>,
    pub chunk_index: Option<usize>,
    pub partition_index: Option<usize>,
    pub attempt: Option<u32>,
    pub strategy_label: Option<String>,
    pub input_rows: Option<u64>,
    pub input_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    pub vector_algorithm: Option<String>,
    pub vector_relation_role: Option<String>,
    pub vector_dimension: Option<usize>,
    pub matrix_layout: Option<String>,
    pub execution_binding_id: Option<u64>,
    pub device_ordinal: Option<usize>,
    pub native_version: Option<(u16, u16, u16)>,
    pub bounded_control_scalars: Option<(u64, u64)>,
}

impl RuntimeOperatorStageIdentity {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    pub fn with_task_id(mut self, task_id: RuntimeTaskId) -> Self {
        self.task_id = Some(task_id);
        self
    }

    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    pub fn with_partition_index(mut self, partition_index: usize) -> Self {
        self.partition_index = Some(partition_index);
        self
    }

    pub fn with_attempt(mut self, attempt: u32) -> Self {
        self.attempt = Some(attempt);
        self
    }

    pub fn with_strategy_label(mut self, strategy_label: impl Into<String>) -> Self {
        self.strategy_label = Some(strategy_label.into());
        self
    }

    pub fn with_input_rows(mut self, input_rows: u64) -> Self {
        self.input_rows = Some(input_rows);
        self
    }

    pub fn with_input_bytes(mut self, input_bytes: u64) -> Self {
        self.input_bytes = Some(input_bytes);
        self
    }

    pub fn with_reserved_bytes(mut self, reserved_bytes: u64) -> Self {
        self.reserved_bytes = Some(reserved_bytes);
        self
    }

    pub fn with_vector_algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.vector_algorithm = Some(algorithm.into());
        self
    }

    pub fn with_vector_relation_role(mut self, role: impl Into<String>) -> Self {
        self.vector_relation_role = Some(role.into());
        self
    }

    pub fn with_vector_dimension(mut self, dimension: usize) -> Self {
        self.vector_dimension = Some(dimension);
        self
    }

    pub fn with_matrix_layout(mut self, layout: impl Into<String>) -> Self {
        self.matrix_layout = Some(layout.into());
        self
    }

    pub fn with_execution_binding_id(mut self, binding_id: u64) -> Self {
        self.execution_binding_id = Some(binding_id);
        self
    }

    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    pub fn with_native_version(mut self, major: u16, minor: u16, patch: u16) -> Self {
        self.native_version = Some((major, minor, patch));
        self
    }

    pub fn with_bounded_control_scalars(mut self, events: u64, bytes: u64) -> Self {
        self.bounded_control_scalars = Some((events, bytes));
        self
    }

    fn apply_identity(&self, mut observation: RuntimeOperatorStageObservation) -> RuntimeOperatorStageObservation {
        if let Some(node_id) = &self.node_id {
            observation = observation.with_node_id(node_id.clone());
        }
        if let Some(task_id) = &self.task_id {
            observation = observation.with_task_id(task_id.clone());
        }
        if let Some(chunk_index) = self.chunk_index {
            observation = observation.with_chunk_index(chunk_index);
        }
        if let Some(partition_index) = self.partition_index {
            observation = observation.with_partition_index(partition_index);
        }
        if let Some(attempt) = self.attempt {
            observation = observation.with_attempt(attempt);
        }
        if let Some(strategy_label) = &self.strategy_label {
            observation = observation.with_strategy_label(strategy_label.clone());
        }
        if let Some(input_rows) = self.input_rows {
            observation = observation.with_input_rows(input_rows);
        }
        if let Some(input_bytes) = self.input_bytes {
            observation = observation.with_input_bytes(input_bytes);
        }
        if let Some(reserved_bytes) = self.reserved_bytes {
            observation = observation.with_reserved_bytes(reserved_bytes);
        }
        if let Some(vector_algorithm) = &self.vector_algorithm {
            observation = observation.with_vector_algorithm(vector_algorithm.clone());
        }
        if let Some(vector_relation_role) = &self.vector_relation_role {
            observation = observation.with_vector_relation_role(vector_relation_role.clone());
        }
        if let Some(vector_dimension) = self.vector_dimension {
            observation = observation.with_vector_dimension(vector_dimension);
        }
        if let Some(matrix_layout) = &self.matrix_layout {
            observation = observation.with_matrix_layout(matrix_layout.clone());
        }
        if let Some(execution_binding_id) = self.execution_binding_id {
            observation = observation.with_execution_binding_id(execution_binding_id);
        }
        if let Some(device_ordinal) = self.device_ordinal {
            observation = observation.with_device_ordinal(device_ordinal);
        }
        if let Some((major, minor, patch)) = self.native_version {
            observation = observation.with_native_version(major, minor, patch);
        }
        if let Some((events, bytes)) = self.bounded_control_scalars {
            observation = observation.with_bounded_control_scalars(events, bytes);
        }
        observation
    }
}

/// Exactly-once Started/terminal guardian for one operator substage.
#[derive(Debug)]
pub struct RuntimeOperatorStageGuard {
    observation: RuntimeObservationHandle,
    context: RuntimeEventContext,
    stage_kind: RuntimeOperatorStageKind,
    identity: RuntimeOperatorStageIdentity,
    terminal_recorded: bool,
}

impl RuntimeOperatorStageGuard {
    /// Emits `Started` immediately and returns a guard that emits one terminal.
    pub fn start(
        observation: RuntimeObservationHandle,
        plan_execution_id: Option<PlanExecutionId>,
        parent_span_id: Option<RuntimeSpanId>,
        stage_kind: RuntimeOperatorStageKind,
        identity: RuntimeOperatorStageIdentity,
    ) -> Self {
        let span_id = RuntimeSpanId::new(NEXT_OPERATOR_STAGE_SPAN_ID.fetch_add(1, Ordering::Relaxed));
        let mut event_context = match plan_execution_id {
            Some(plan_execution_id) => RuntimeEventContext::for_plan(plan_execution_id),
            None => RuntimeEventContext::default(),
        };
        event_context = event_context.with_span(span_id, parent_span_id);
        let guard = Self { observation, context: event_context, stage_kind, identity, terminal_recorded: false };
        guard.record(RuntimeOperatorStageState::Started, None, None);
        guard
    }

    /// Records successful completion with optional output facts.
    pub fn finish(mut self, output_rows: Option<u64>, output_bytes: Option<u64>) {
        self.record(RuntimeOperatorStageState::Finished, output_rows, output_bytes);
        self.terminal_recorded = true;
    }

    /// Attach successful bounded host-control evidence to the terminal event.
    /// The already-emitted Started event intentionally has no speculative
    /// boundary totals.
    pub fn finish_with_bounded_control_scalars(
        mut self,
        output_rows: Option<u64>,
        output_bytes: Option<u64>,
        events: u64,
        bytes: u64,
    ) {
        // `RuntimeOperatorStageGuard` owns a `Drop` implementation so moving
        // its identity out would make the terminal observation itself fail to
        // compile. Keep the guard intact and replace the identity from a
        // cheap clone instead.
        self.identity = self.identity.clone().with_bounded_control_scalars(events, bytes);
        self.record(RuntimeOperatorStageState::Finished, output_rows, output_bytes);
        self.terminal_recorded = true;
    }

    /// Records structured failure or cancellation from a native error.
    pub fn fail(mut self, error: &QueryError) {
        let state = if error.code() == QueryErrorCode::Cancelled {
            RuntimeOperatorStageState::Cancelled
        } else {
            RuntimeOperatorStageState::Failed { error_code: error.code().as_str().to_string() }
        };
        self.record(state, None, None);
        self.terminal_recorded = true;
    }

    /// Records cooperative cancellation without a full error object.
    pub fn cancel(mut self) {
        self.record(RuntimeOperatorStageState::Cancelled, None, None);
        self.terminal_recorded = true;
    }

    fn record(&self, lifecycle: RuntimeOperatorStageState, output_rows: Option<u64>, output_bytes: Option<u64>) {
        let mut stage = RuntimeOperatorStageObservation::new(self.stage_kind, lifecycle);
        stage = self.identity.apply_identity(stage);
        if let Some(output_rows) = output_rows {
            stage = stage.with_output_rows(output_rows);
        }
        if let Some(output_bytes) = output_bytes {
            stage = stage.with_output_bytes(output_bytes);
        }
        self.observation
            .record_with_context(self.context.clone(), || RuntimeObservationPayload::OperatorStage(Box::new(stage)));
    }
}

impl Drop for RuntimeOperatorStageGuard {
    fn drop(&mut self) {
        if self.terminal_recorded {
            return;
        }
        // Unwind or early-drop without an explicit terminal still leaves a
        // truthful terminal rather than an orphaned Started.
        let lifecycle = if std::thread::panicking() {
            RuntimeOperatorStageState::Failed { error_code: "panicked".to_string() }
        } else {
            RuntimeOperatorStageState::Failed { error_code: "dropped".to_string() }
        };
        self.record(lifecycle, None, None);
        self.terminal_recorded = true;
    }
}

/// Convenience helper that scopes one operator substage around a fallible body.
pub fn observe_operator_stage<T, F>(
    observation: &RuntimeObservationHandle,
    plan_execution_id: Option<PlanExecutionId>,
    parent_span_id: Option<RuntimeSpanId>,
    stage_kind: RuntimeOperatorStageKind,
    identity: RuntimeOperatorStageIdentity,
    body: F,
) -> Result<T, QueryError>
where
    F: FnOnce() -> Result<T, QueryError>,
{
    let guard =
        RuntimeOperatorStageGuard::start(observation.clone(), plan_execution_id, parent_span_id, stage_kind, identity);
    match body() {
        Ok(value) => {
            guard.finish(None, None);
            Ok(value)
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

/// Runtime-owned execution scope exposed to operator-stage observation.
///
/// The observation layer deliberately depends on this narrow value contract,
/// rather than on the runtime's concrete execution context.  That keeps the
/// observation-to-runtime edge out of the layering graph while preserving the
/// established [`stage_context_from_execution`] call surface.
pub trait RuntimeExecutionStageContext {
    /// Returns the observation handle, plan execution id, and parent span id
    /// that scope one operator stage.
    ///
    /// The observation handle is always present, so a stage is still recorded
    /// when either id is `None`. A `None` plan execution id records the stage
    /// without associating it with a plan execution; a `None` parent span id
    /// records the stage span without a parent span. `ExecutionContext`
    /// returns both ids when it carries a runtime execution scope and neither
    /// when it does not.
    fn runtime_operator_stage_context(
        &self,
    ) -> (RuntimeObservationHandle, Option<PlanExecutionId>, Option<RuntimeSpanId>);
}

/// Helper that builds stage observation identity from a runtime execution scope.
pub fn stage_context_from_execution(
    context: &impl RuntimeExecutionStageContext,
) -> (RuntimeObservationHandle, Option<PlanExecutionId>, Option<RuntimeSpanId>) {
    context.runtime_operator_stage_context()
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier, Mutex},
        time::Duration,
    };

    use super::*;
    use crate::observability::{
        DeterministicRuntimeObservationClock, RuntimeObservationConfig, RuntimeObservationPayload,
        RuntimeObservationSession, RuntimeOperatorStageKind, RuntimeQueryId, RuntimeTerminalOutcome,
        write_runtime_observation_jsonl,
    };

    fn session_with_limits(max_records: u64) -> RuntimeObservationSession {
        RuntimeObservationSession::with_clock(
            RuntimeQueryId::new("operator-stage"),
            None,
            RuntimeObservationConfig::bounded(max_records, 256 * 1024),
            Arc::new(DeterministicRuntimeObservationClock::new(Some(1_000))),
        )
    }

    fn session() -> RuntimeObservationSession {
        session_with_limits(64)
    }

    #[test]
    fn success_pairs_started_and_finished() {
        let session = session();
        let handle = session.handle();
        {
            let guard = RuntimeOperatorStageGuard::start(
                handle,
                Some(PlanExecutionId::new("plan-1")),
                Some(RuntimeSpanId::new(1)),
                RuntimeOperatorStageKind::RegexReplacePreflight,
                RuntimeOperatorStageIdentity::new().with_input_rows(12),
            );
            guard.finish(Some(12), Some(64));
        }
        let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).expect("seal");
        let stages: Vec<_> = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage) => Some((event.span_id, stage.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].0, stages[1].0);
        assert!(matches!(stages[0].1.lifecycle, RuntimeOperatorStageState::Started));
        assert!(matches!(stages[1].1.lifecycle, RuntimeOperatorStageState::Finished));
        assert_eq!(stages[0].1.input_rows, Some(12));
        assert_eq!(stages[1].1.output_rows, Some(12));
        assert_eq!(stages[1].1.output_bytes, Some(64));
    }

    #[test]
    fn vector_stage_identity_retains_typed_runtime_and_native_facts() {
        let session = session();
        let guard = RuntimeOperatorStageGuard::start(
            session.handle(),
            Some(PlanExecutionId::new("plan-vector")),
            None,
            RuntimeOperatorStageKind::VectorPack,
            RuntimeOperatorStageIdentity::new()
                .with_node_id("7")
                .with_vector_algorithm("brute_force_knn")
                .with_vector_relation_role("dataset")
                .with_vector_dimension(32)
                .with_matrix_layout("row_major")
                .with_execution_binding_id(41)
                .with_device_ordinal(2)
                .with_native_version(26, 6, 0)
                .with_input_rows(128)
                .with_input_bytes(16_384),
        );
        guard.finish(Some(128), Some(16_384));

        let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).expect("seal");
        let stage = snapshot
            .events()
            .iter()
            .find_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage)
                    if matches!(stage.lifecycle, RuntimeOperatorStageState::Started) =>
                {
                    Some(stage)
                },
                _ => None,
            })
            .expect("vector stage started event");
        assert_eq!(stage.stage_kind, RuntimeOperatorStageKind::VectorPack);
        assert_eq!(stage.vector_algorithm.as_deref(), Some("brute_force_knn"));
        assert_eq!(stage.vector_relation_role.as_deref(), Some("dataset"));
        assert_eq!(stage.vector_dimension, Some(32));
        assert_eq!(stage.matrix_layout.as_deref(), Some("row_major"));
        assert_eq!(stage.execution_binding_id, Some(41));
        assert_eq!(stage.device_ordinal, Some(2));
        assert_eq!(stage.native_version_major, Some(26));
        assert_eq!(stage.native_version_minor, Some(6));
        assert_eq!(stage.native_version_patch, Some(0));
    }

    #[test]
    fn error_emits_failed_terminal() {
        let session = session();
        let err = observe_operator_stage(
            &session.handle(),
            Some(PlanExecutionId::new("plan-err")),
            None,
            RuntimeOperatorStageKind::LibcudfGroupby,
            RuntimeOperatorStageIdentity::new(),
            || -> Result<(), QueryError> {
                Err(QueryError::new(QueryErrorCode::CudfResourceExhausted, "test.operator_stage", "boom"))
            },
        )
        .expect_err("body fails");
        assert_eq!(err.code(), QueryErrorCode::CudfResourceExhausted);
        let snapshot = session.seal(RuntimeTerminalOutcome::from_query_error(&err)).expect("seal");
        let states: Vec<_> = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage) => Some(stage.lifecycle.clone()),
                _ => None,
            })
            .collect();
        assert!(matches!(states[0], RuntimeOperatorStageState::Started));
        assert!(matches!(
            states[1],
            RuntimeOperatorStageState::Failed { ref error_code } if error_code == "cudf_resource_exhausted"
        ));
    }

    #[test]
    fn unwind_emits_failed_terminal() {
        let session = session();
        let handle = session.handle();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = RuntimeOperatorStageGuard::start(
                handle.clone(),
                Some(PlanExecutionId::new("plan-panic")),
                None,
                RuntimeOperatorStageKind::PreparedGroupbyPrepare,
                RuntimeOperatorStageIdentity::new(),
            );
            panic!("force unwind");
        }));
        let snapshot = session.seal(RuntimeTerminalOutcome::Panicked).expect("seal");
        let states: Vec<_> = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage) => Some(stage.lifecycle.clone()),
                _ => None,
            })
            .collect();
        assert!(matches!(states[0], RuntimeOperatorStageState::Started));
        assert!(matches!(
            states[1],
            RuntimeOperatorStageState::Failed { ref error_code } if error_code == "panicked"
        ));
    }

    #[test]
    fn live_snapshot_contains_started_before_primitive_returns() {
        let session = session();
        let handle = session.handle();
        let observer = handle.clone();
        let barrier = Arc::new(Barrier::new(2));
        let started_seen = Arc::new(Mutex::new(false));
        let barrier_worker = Arc::clone(&barrier);
        let started_seen_worker = Arc::clone(&started_seen);

        let worker = std::thread::spawn(move || {
            let guard = RuntimeOperatorStageGuard::start(
                handle,
                Some(PlanExecutionId::new("plan-block")),
                None,
                RuntimeOperatorStageKind::ExactCardinalityProbe,
                RuntimeOperatorStageIdentity::new().with_strategy_label("nunique"),
            );
            // Signal that Started should be visible, then block until the
            // observer samples the live snapshot.
            barrier_worker.wait();
            while !*started_seen_worker.lock().expect("lock") {
                std::thread::sleep(Duration::from_millis(1));
            }
            guard.finish(Some(1), None);
        });

        barrier.wait();
        let mut saw_started = false;
        for _ in 0..200 {
            if let Some(live) = observer.live_snapshot() {
                saw_started = live.events().iter().any(|event| {
                    matches!(
                        &event.payload,
                        RuntimeObservationPayload::OperatorStage(stage)
                            if matches!(stage.lifecycle, RuntimeOperatorStageState::Started)
                                && stage.stage_kind == RuntimeOperatorStageKind::ExactCardinalityProbe
                    )
                });
                if saw_started {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(saw_started, "live snapshot must expose Started before the stage returns");
        *started_seen.lock().expect("lock") = true;
        worker.join().expect("worker");
        let _ = session.seal(RuntimeTerminalOutcome::Succeeded);
    }

    #[test]
    fn omitted_terminal_at_record_cap_is_loss_not_fabricated_finish() {
        // Only enough budget for Started; the terminal must be counted as loss.
        let session = session_with_limits(1);
        {
            let guard = RuntimeOperatorStageGuard::start(
                session.handle(),
                Some(PlanExecutionId::new("plan-cap")),
                None,
                RuntimeOperatorStageKind::FinalSort,
                RuntimeOperatorStageIdentity::new(),
            );
            guard.finish(None, None);
        }
        let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).expect("seal");
        let stages: Vec<_> = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::OperatorStage(stage) => Some(stage.lifecycle.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(stages.len(), 1);
        assert!(matches!(stages[0], RuntimeOperatorStageState::Started));
        assert!(snapshot.footer().first_dropped_sequence.is_some());
        assert!(
            snapshot.footer().dropped_by_class.iter().any(|loss| loss.count > 0),
            "terminal drop must produce loss metadata, not a fabricated Finished"
        );
    }

    #[test]
    fn jsonl_round_trip_for_every_stage_kind() {
        let kinds = [
            RuntimeOperatorStageKind::RegexReplacePreflight,
            RuntimeOperatorStageKind::RegexReplaceExecute,
            RuntimeOperatorStageKind::ExactCardinalityProbe,
            RuntimeOperatorStageKind::PreparedGroupbyPrepare,
            RuntimeOperatorStageKind::PreparedGroupbyAggregate,
            RuntimeOperatorStageKind::LibcudfGroupby,
            RuntimeOperatorStageKind::HashPartition,
            RuntimeOperatorStageKind::PartialAggregate,
            RuntimeOperatorStageKind::FinalMerge,
            RuntimeOperatorStageKind::PartitionLocalTopK,
            RuntimeOperatorStageKind::FinalSort,
            RuntimeOperatorStageKind::VectorPack,
            RuntimeOperatorStageKind::VectorNative,
            RuntimeOperatorStageKind::VectorAssemble,
        ];
        let session = session_with_limits(256);
        for kind in kinds {
            let guard = RuntimeOperatorStageGuard::start(
                session.handle(),
                Some(PlanExecutionId::new("plan-jsonl")),
                None,
                kind,
                RuntimeOperatorStageIdentity::new().with_strategy_label(kind.as_str()),
            );
            guard.finish(Some(0), Some(0));
        }
        let snapshot = session.seal(RuntimeTerminalOutcome::Succeeded).expect("seal");
        let mut buffer = Vec::new();
        write_runtime_observation_jsonl(&snapshot, &mut buffer).expect("export");
        let text = String::from_utf8(buffer).expect("utf8");
        for kind in kinds {
            assert!(text.contains(kind.as_str()), "JSONL must retain stage kind {}", kind.as_str());
        }
        for line in text.lines() {
            let value: serde_json::Value = serde_json::from_str(line).expect("json line");
            if value.get("record_type").and_then(|v| v.as_str()) == Some("timeline_event") {
                let event: crate::observability::RuntimeTimelineEvent =
                    serde_json::from_value(value.get("data").cloned().expect("data")).expect("event");
                if let RuntimeObservationPayload::OperatorStage(stage) = event.payload {
                    let _ = stage.stage_kind.as_str();
                }
            }
        }
    }
}
