use std::{sync::Arc, time::Duration};

use algeon_datafusion::{
    attempt::{
        AlgeonAttempt, AttemptOptions, QueryAttemptReport, RuntimeTerminalOutcome, SealedRuntimeObservationSnapshot,
    },
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    error::{Error, ErrorCode, ErrorStatus, Result},
    observability::{BackendMetricsSnapshot, backend_metric_descriptors},
    report::PlanningReport,
};

pub fn backend_contract() -> Result<AlgeonGpuBackend> {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    AlgeonGpuBackend::builder().devices([device]).build()
}

pub fn attempt_contract(backend: &AlgeonGpuBackend, options: AttemptOptions) -> Result<AlgeonAttempt> {
    let attempt = backend.begin_attempt(options)?;
    let _ = attempt.attempt_id();
    let _ = attempt.cancellation();
    let _ = attempt.completion().is_ready();
    attempt.register_sealed_report_observer(
        |_report: Option<Arc<QueryAttemptReport>>, _snapshot: SealedRuntimeObservationSnapshot| {},
    );
    Ok(attempt)
}

pub fn terminal_outcome_contract(outcome: RuntimeTerminalOutcome) -> &'static str {
    outcome.stable_label()
}

pub fn error_contract(error: &Error) -> (ErrorCode, &'static str, ErrorStatus) {
    (error.code(), error.operation(), error.status())
}

pub fn planning_contract() -> PlanningReport {
    PlanningReport::new()
}

pub fn metrics_contract(snapshot: &BackendMetricsSnapshot) -> usize {
    backend_metric_descriptors()
        .iter()
        .map(|descriptor| snapshot.project_backend_metric(descriptor).map_or(0, |samples| samples.len()))
        .sum()
}

pub fn close_contract(backend: &AlgeonGpuBackend) -> Result<()> {
    backend.close_and_wait(Duration::from_secs(30)).map(|_| ())
}
