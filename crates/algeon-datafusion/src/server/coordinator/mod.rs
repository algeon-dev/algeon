//! Single-authority server query lifecycle, observation, and retention registry.
//!
//! Lock ordering is structural. Registry methods never call the native engine,
//! render plans, encode JSONL, or await while holding `state`. Finalization
//! first extracts immutable inputs under the registry mutex, performs sealing
//! and encoding outside it, and then conditionally installs the result for the
//! same [`QueryKey`] generation.

mod artifacts;
mod finalization;
mod maintenance;
mod registry;
mod retention;
mod ticket;
mod types;

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use datafusion::physical_plan::ExecutionPlan;
use tokio::sync::Notify;

#[cfg(test)]
pub(crate) use self::artifacts::install_coordinator_render_probe;
#[cfg(test)]
use self::artifacts::{FinalizationTestGate, run_coordinator_render_probe};
#[cfg(test)]
pub(crate) use self::types::{CoordinatorArtifact, QueryTerminalOutcome, RetainedUnavailableReason};
use self::{
    artifacts::ArtifactEgressLimiter,
    registry::{QueryRegistryState, normalized_correlation_id},
};
pub(crate) use self::{
    maintenance::{QueryCoordinatorMaintenance, QueryCoordinatorShutdownPlan},
    ticket::{QueryExecutionLease, QueryPlanningLease, QueryPlanningReportDeliveryTerminal},
    types::{
        CoordinatorTicketPayload, QueryCoordinatorConfig, QueryCoordinatorError, QueryCoordinatorErrorKind, QueryKey,
        StatementExecutionDomain, logical_capture_reservation_bytes,
    },
};
use super::observability::CapturedQuery;
use crate::server::planner::NativePlanningDiagnostics;

pub(crate) struct QueryCoordinator {
    pub(crate) config: QueryCoordinatorConfig,
    pub(crate) state: Mutex<QueryRegistryState>,
    pub(crate) artifact_egress: Arc<ArtifactEgressLimiter>,
    pub(crate) notify: Notify,
    #[cfg(test)]
    pub(crate) finalization_gate: Mutex<Option<Arc<FinalizationTestGate>>>,
    #[cfg(test)]
    pub(crate) finalization_installs: AtomicUsize,
}

impl QueryCoordinator {
    pub(crate) fn new(config: QueryCoordinatorConfig) -> Arc<Self> {
        let artifact_egress = ArtifactEgressLimiter::new(
            config.max_live_queries.max(1),
            config.retained_max_bytes.max(config.per_query_artifact_max_bytes),
        );
        Arc::new(Self {
            config,
            state: Mutex::new(QueryRegistryState { accepting: true, ..QueryRegistryState::default() }),
            artifact_egress,
            notify: Notify::new(),
            #[cfg(test)]
            finalization_gate: Mutex::new(None),
            #[cfg(test)]
            finalization_installs: AtomicUsize::new(0),
        })
    }

    pub(crate) fn set_sql(&self, query_id: &str, sql: String) {
        let mut state = self.lock_state();
        let Some(key) = state.current.get(query_id).cloned() else { return };
        let Some(record) = state.records.get_mut(&key) else { return };
        if let Some(captured) = &mut record.captured {
            captured.set_sql(sql);
            Self::bump_capture_version(record);
        }
    }

    pub(crate) fn record_correlation_id(&self, query_id: &str, correlation_id: Option<&str>) {
        let mut state = self.lock_state();
        let Some(key) = state.current.get(query_id).cloned() else { return };
        let Some(record) = state.records.get_mut(&key) else { return };
        if record.correlation_id.is_none() {
            record.correlation_id = normalized_correlation_id(correlation_id);
        }
        if let Some(captured) = &mut record.captured {
            captured.record_correlation_id(correlation_id);
            Self::bump_capture_version(record);
        }
    }

    pub(crate) fn capture_planning(
        &self,
        query_id: &str,
        diagnostics: Option<NativePlanningDiagnostics>,
        planned_plan: Arc<dyn ExecutionPlan>,
    ) {
        let snapshot = {
            let state = self.lock_state();
            state.current.get(query_id).and_then(|key| {
                state.records.get(key).and_then(|record| {
                    (record.captured.is_some() && record.terminal_outcome.is_none()).then(|| {
                        (
                            key.clone(),
                            record.phase,
                            record.captured.as_ref().expect("checked capture presence").max_metric_records(),
                        )
                    })
                })
            })
        };
        let Some((key, phase, max_metric_records)) = snapshot else {
            return;
        };

        #[cfg(test)]
        run_coordinator_render_probe("planning render");
        let max_bytes = usize::try_from(self.config.per_query_capture_max_bytes).unwrap_or(usize::MAX);
        let rendered =
            CapturedQuery::render_planning(query_id, diagnostics, planned_plan, max_bytes, max_metric_records);

        let mut state = self.lock_state();
        if state.current.get(query_id) != Some(&key) {
            return;
        }
        let Some(record) = state.records.get_mut(&key) else { return };
        if record.key != key || record.phase != phase || record.terminal_outcome.is_some() {
            return;
        }
        if let Some(captured) = &mut record.captured {
            captured.install_planning(rendered);
            Self::bump_capture_version(record);
        }
    }

    pub(crate) fn capture_planning_failure(
        &self,
        query_id: &str,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) {
        let mut state = self.lock_state();
        let Some(key) = state.current.get(query_id).cloned() else { return };
        let Some(record) = state.records.get_mut(&key) else { return };
        if let Some(captured) = &mut record.captured {
            captured.capture_planning_failure(stage, reason_code, message);
            Self::bump_capture_version(record);
        }
    }

    pub(crate) fn lock_state(&self) -> std::sync::MutexGuard<'_, QueryRegistryState> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
