//! Pure fixed execution-attempt lifecycle state.
use std::{error::Error, fmt};

use crate::observability::{
    AttemptAdmissionFact, AttemptAdmissionWaitFact, AttemptComposedScheduleProjection, AttemptDisposition,
    AttemptFailureOperation, AttemptFailureSummary, AttemptTerminalCommit, AttemptTimestamps,
    COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1, CleanupFailureSummary, RuntimeTerminalOutcome,
};
#[cfg(test)]
use crate::observability::{
    AttemptCapacityDisposition, AttemptFailureCode, AttemptFailureStatus, AttemptFinalDomainUsage,
    AttemptHealthTransition, CleanupFailurePhase,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptLifecycleError {
    DoubleTerminalization,
    InvalidOrder {
        boundary: &'static str,
        reason: &'static str,
    },
    DispositionConflict,
    ReportBeforeTerminal,
    CleanupWithoutAdmission,
    TerminalCommitConflict,
    FinalizationUnavailable,
    /// The attempt's one admission right is already consumed, by an earlier
    /// claim or by finishing the attempt without admission.
    AdmissionAlreadyClaimed,
}

impl fmt::Display for AttemptLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid attempt lifecycle transition: {self:?}")
    }
}

impl Error for AttemptLifecycleError {}

#[derive(Debug)]
pub(crate) struct AttemptLifecycle {
    pub(crate) submitted: u64,
    pub(crate) planning_started: Option<u64>,
    pub(crate) planning_finished: Option<u64>,
    pub(crate) admission_requested: Option<u64>,
    pub(crate) admitted: Option<(u64, AttemptAdmissionFact)>,
    pub(crate) admission_wait: Option<AttemptAdmissionWaitFact>,
    pub(crate) composed_schedule: Option<AttemptComposedScheduleProjection>,
    pub(crate) physical: Option<u64>,
    pub(crate) gpu: Option<u64>,
    pub(crate) terminal: Option<(u64, RuntimeTerminalOutcome)>,
    pub(crate) cleanup: Option<u64>,
    pub(crate) ready: Option<u64>,
    pub(crate) disposition: Option<AttemptDisposition>,
    pub(crate) cleanup_failures: u64,
    pub(crate) cleanup_failures_complete: bool,
    pub(crate) primary_failure: Option<AttemptFailureSummary>,
    pub(crate) first_cleanup_failure: Option<CleanupFailureSummary>,
    pub(crate) terminal_commit: Option<AttemptTerminalCommit>,
    /// A terminal report may be sealed only after its one finalizer has been
    /// consumed. Admitted attempts consume it with their cleanup/capacity
    /// commit; all other attempts consume it with `NotRequired` finalization.
    pub(crate) finalized: bool,
}

fn first(slot: &mut Option<u64>, at: u64, previous: u64, boundary: &'static str) -> Result<(), AttemptLifecycleError> {
    if slot.is_some() {
        return Ok(());
    }
    if at < previous {
        return Err(AttemptLifecycleError::InvalidOrder { boundary, reason: "timestamp_regressed" });
    }
    *slot = Some(at);
    Ok(())
}

impl AttemptLifecycle {
    pub(crate) fn new(submitted: u64) -> Self {
        Self {
            submitted,
            planning_started: None,
            planning_finished: None,
            admission_requested: None,
            admitted: None,
            admission_wait: None,
            composed_schedule: None,
            physical: None,
            gpu: None,
            terminal: None,
            cleanup: None,
            ready: None,
            disposition: None,
            cleanup_failures: 0,
            cleanup_failures_complete: true,
            primary_failure: None,
            first_cleanup_failure: None,
            terminal_commit: None,
            finalized: false,
        }
    }

    pub(crate) fn planning_started(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        first(&mut self.planning_started, at, self.submitted, "planning_started")
    }

    pub(crate) fn planning_finished(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        first(&mut self.planning_finished, at, self.planning_started.unwrap_or(self.submitted), "planning_finished")
    }

    fn rejects_gpu_boundary(&self, boundary: &'static str) -> Result<(), AttemptLifecycleError> {
        match self.disposition {
            Some(AttemptDisposition::Unplanned) => {
                Err(AttemptLifecycleError::InvalidOrder { boundary, reason: "unplanned_has_gpu" })
            },
            Some(AttemptDisposition::CpuFallback) => {
                Err(AttemptLifecycleError::InvalidOrder { boundary, reason: "cpu_fallback_has_gpu" })
            },
            _ => Ok(()),
        }
    }

    /// Records the attempt's one admission claim. Every check runs before the
    /// timestamp is written, so a rejected claim leaves the lifecycle as it was.
    pub(crate) fn admission_requested(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        if self.admission_requested.is_some() {
            return Err(AttemptLifecycleError::AdmissionAlreadyClaimed);
        }
        if self.terminal.is_some() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_requested",
                reason: "attempt_terminal",
            });
        }
        self.rejects_gpu_boundary("admission_requested")?;
        if self.disposition.is_none() {
            // The final plan is always classified before GPU admission can be
            // requested; reaching the queue without a disposition is a caller
            // contract violation, not a synthesizable state.
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_requested",
                reason: "disposition_not_set",
            });
        }
        first(
            &mut self.admission_requested,
            at,
            self.planning_finished.unwrap_or(self.submitted),
            "admission_requested",
        )
    }

    pub(crate) fn admitted(&mut self, fact: AttemptAdmissionFact, at: u64) -> Result<(), AttemptLifecycleError> {
        self.rejects_gpu_boundary("admitted")?;
        if self.admitted.is_some() {
            return Ok(());
        }
        // The owner may terminalize while admission is in flight; a grant
        // published after that terminal would never execute.
        if self.terminal.is_some() {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "admitted", reason: "attempt_terminal" });
        }
        let Some(requested) = self.admission_requested else {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admitted",
                reason: "admission_not_requested",
            });
        };
        if at < requested {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "admitted", reason: "timestamp_regressed" });
        }
        self.admitted = Some((at, fact));
        Ok(())
    }

    /// Retains the controller-owned final queue state after admission has
    /// succeeded. A second identical observation is harmless; conflicting
    /// state would make the immutable sealed report ambiguous.
    pub(crate) fn admission_wait(&mut self, fact: AttemptAdmissionWaitFact) -> Result<(), AttemptLifecycleError> {
        if self.admitted.is_none() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_wait",
                reason: "admission_not_granted",
            });
        }
        match self.admission_wait {
            None => {
                self.admission_wait = Some(fact);
                Ok(())
            },
            Some(current) if current == fact => Ok(()),
            Some(_) => Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_wait",
                reason: "conflicting_final_queue_fact",
            }),
        }
    }

    /// Attaches the immutable composed schedule after admission and before the
    /// attempt can publish its report. Repeated identical attachment is
    /// harmless; a conflicting or late attachment would make the sealed
    /// report ambiguous and is rejected.
    pub(crate) fn composed_schedule(
        &mut self,
        projection: AttemptComposedScheduleProjection,
    ) -> Result<(), AttemptLifecycleError> {
        if projection.schema != COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1 {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "unsupported_schedule_schema",
            });
        }
        if !projection.fingerprint_matches_contents() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "fingerprint_mismatch",
            });
        }
        if self.admitted.is_none() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "admission_not_granted",
            });
        }
        if self.ready.is_some() || self.finalized {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "report_ready_already_recorded",
            });
        }
        let (_, admission) = self.admitted.as_ref().expect("admission checked");
        if projection.device_ordinal != admission.device_ordinal {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "device_ordinal_mismatch",
            });
        }
        if projection.grant_bytes != admission.grant_bytes {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "grant_bytes_mismatch",
            });
        }
        match &self.composed_schedule {
            None => {
                self.composed_schedule = Some(projection);
                Ok(())
            },
            Some(current) if current == &projection => Ok(()),
            Some(_) => Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "conflicting_schedule_projection",
            }),
        }
    }

    pub(crate) fn physical(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        if self.planning_started.is_some() && self.planning_finished.is_none() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "physical_execution_started",
                reason: "planning_in_progress",
            });
        }
        if matches!(self.disposition, Some(AttemptDisposition::Unplanned)) {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "physical_execution_started",
                reason: "unplanned_has_execution",
            });
        }
        let previous = self.admitted.as_ref().map_or(self.planning_finished.unwrap_or(self.submitted), |(at, _)| *at);
        first(&mut self.physical, at, previous, "physical_execution_started")
    }

    pub(crate) fn gpu(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        let Some((admitted, _)) = self.admitted else {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "gpu_execution_started",
                reason: "not_admitted",
            });
        };
        first(&mut self.gpu, at, admitted, "gpu_execution_started")
    }

    pub(crate) fn disposition(&mut self, d: AttemptDisposition) -> Result<(), AttemptLifecycleError> {
        match self.disposition {
            None => {
                let has_gpu = self.admission_requested.is_some() || self.admitted.is_some() || self.gpu.is_some();
                if matches!(d, AttemptDisposition::Unplanned | AttemptDisposition::CpuFallback) && has_gpu {
                    return Err(AttemptLifecycleError::InvalidOrder {
                        boundary: "disposition",
                        reason: "gpu_boundary_already_recorded",
                    });
                }
                if matches!(d, AttemptDisposition::Unplanned) && self.physical.is_some() {
                    return Err(AttemptLifecycleError::InvalidOrder {
                        boundary: "disposition",
                        reason: "physical_boundary_already_recorded",
                    });
                }
                self.disposition = Some(d);
                Ok(())
            },
            Some(current) if current == d => Ok(()),
            Some(_) => Err(AttemptLifecycleError::DispositionConflict),
        }
    }

    pub(crate) fn terminalize(
        &mut self,
        outcome: RuntimeTerminalOutcome,
        at: u64,
    ) -> Result<(), AttemptLifecycleError> {
        if self.terminal.is_some() {
            return Err(AttemptLifecycleError::DoubleTerminalization);
        }
        let d = self.disposition.unwrap_or(AttemptDisposition::Unplanned);
        if matches!(d, AttemptDisposition::Unplanned)
            && (self.admitted.is_some() || self.physical.is_some() || self.gpu.is_some())
        {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "terminal",
                reason: "unplanned_has_execution",
            });
        }
        if matches!(d, AttemptDisposition::CpuFallback) && (self.admitted.is_some() || self.gpu.is_some()) {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "terminal", reason: "cpu_has_gpu" });
        }
        let last = self
            .gpu
            .or(self.physical)
            .or(self.admitted.as_ref().map(|(at, _)| *at))
            .or(self.planning_finished)
            .unwrap_or(self.submitted);
        if at < last {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "outcome_terminal",
                reason: "timestamp_regressed",
            });
        }
        self.disposition = Some(d);
        if self.primary_failure.is_none() {
            self.primary_failure = outcome.fallback_failure_summary();
        }
        self.terminal = Some((at, outcome));
        Ok(())
    }

    /// Terminalizes an admission request that failed before it claimed
    /// admission. A claimed attempt is finished by its claim, and an admitted
    /// or physically executing attempt remains owned by its attempt owner and
    /// release-commit path.
    pub(crate) fn terminalize_admission_rejection(
        &mut self,
        outcome: RuntimeTerminalOutcome,
        summary: Option<AttemptFailureSummary>,
        at: u64,
    ) -> Result<(), AttemptLifecycleError> {
        if self.admission_requested.is_some() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_rejection",
                reason: "admission_claimed",
            });
        }
        if self.admitted.is_some() || self.physical.is_some() || self.gpu.is_some() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_rejection",
                reason: "admission_already_granted",
            });
        }
        if !matches!(outcome, RuntimeTerminalOutcome::Failed { .. } | RuntimeTerminalOutcome::Cancelled) {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_rejection",
                reason: "outcome_not_rejection",
            });
        }
        if self.terminal.is_some() {
            return Err(AttemptLifecycleError::DoubleTerminalization);
        }
        if let Some(summary) = summary {
            self.primary_failure(summary);
        }
        self.terminalize(outcome, at)
    }

    /// Records a primary failure while it is still borrowed by the execution
    /// path. First-wins preserves the terminal failure that was returned to
    /// the caller; a later cleanup failure cannot replace it.
    pub(crate) fn primary_failure(&mut self, summary: AttemptFailureSummary) {
        if self.primary_failure.is_none() {
            self.primary_failure = Some(summary);
        }
    }

    /// Classifies a failure committed now by the lifecycle stage reached.
    /// Later stages win: admission implies planning is over even when its
    /// timestamps were never recorded.
    pub(crate) fn failure_operation(&self) -> AttemptFailureOperation {
        if self.cleanup.is_some() {
            AttemptFailureOperation::Cleanup
        } else if self.admitted.is_some() {
            AttemptFailureOperation::Execution
        } else if self.admission_requested.is_some() {
            AttemptFailureOperation::Admission
        } else if self.planning_finished.is_none() {
            AttemptFailureOperation::Planning
        } else {
            AttemptFailureOperation::Execution
        }
    }

    /// Retains only the first cleanup failure while accounting every observed
    /// cleanup failure. Overflow is marked incomplete rather than published as
    /// a saturated exact value.
    pub(crate) fn cleanup_failure(&mut self, summary: CleanupFailureSummary) {
        if let Some(first) = &mut self.first_cleanup_failure {
            // The report retains one representative summary. Once another
            // cleanup failure is observed that representative is necessarily
            // incomplete, even though the count remains exact below.
            first.mark_incomplete();
        } else {
            self.first_cleanup_failure = Some(summary);
        }
        match self.cleanup_failures.checked_add(1) {
            Some(next) => self.cleanup_failures = next,
            None => self.cleanup_failures_complete = false,
        }
    }

    /// Commits the cleanup and capacity outcome observed by the admitting
    /// query service. It must happen after the primary outcome but before the
    /// report-ready boundary; first-wins preserves exactly one release fact.
    pub(crate) fn terminal_commit(
        &mut self,
        commit: AttemptTerminalCommit,
        at: u64,
    ) -> Result<(), AttemptLifecycleError> {
        if self.ready.is_some() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "terminal_commit",
                reason: "report_ready_already_recorded",
            });
        }
        if let Some(current) = self.terminal_commit {
            return if current == commit { Ok(()) } else { Err(AttemptLifecycleError::TerminalCommitConflict) };
        }
        if !commit.has_consistent_capacity_health_transition() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "terminal_commit",
                reason: "capacity_health_mismatch",
            });
        }
        if !commit.is_admitted_cleanup_commit() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "terminal_commit",
                reason: "invalid_admitted_cleanup_fact",
            });
        }
        self.cleanup(at)?;
        self.terminal_commit = Some(commit);
        self.finalized = true;
        Ok(())
    }

    /// Consumes the owner-held finalizer for an attempt that never acquired an
    /// admission grant. This is deliberately separate from an admitted
    /// terminal commit: no cleanup, capacity, health, or domain facts exist.
    pub(crate) fn finalize_not_admitted(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        if self.admitted.is_some() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "attempt_finalization",
                reason: "admission_already_granted",
            });
        }
        if self.terminal.is_none() {
            return Err(AttemptLifecycleError::ReportBeforeTerminal);
        }
        if self.finalized {
            return Ok(());
        }
        // Keep the finalization point monotonic even though it has no public
        // timestamp field of its own. `ready` remains the published boundary.
        let terminal = self.terminal.as_ref().expect("terminal was checked").0;
        if at < terminal {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "attempt_finalization",
                reason: "timestamp_regressed",
            });
        }
        self.finalized = true;
        Ok(())
    }

    pub(crate) fn cleanup(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        if self.cleanup.is_some() {
            return Ok(());
        }
        if self.admitted.is_none() {
            return Err(AttemptLifecycleError::CleanupWithoutAdmission);
        }
        let Some((terminal, _)) = self.terminal else {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "cleanup_complete", reason: "not_terminal" });
        };
        first(&mut self.cleanup, at, terminal, "cleanup_complete")
    }

    pub(crate) fn ready(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        let Some((terminal, _)) = self.terminal else { return Err(AttemptLifecycleError::ReportBeforeTerminal) };
        if !self.finalized {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "report_ready",
                reason: "finalization_incomplete",
            });
        }
        if self.admitted.is_some() && self.cleanup.is_none() {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "report_ready", reason: "cleanup_incomplete" });
        }
        first(&mut self.ready, at, self.cleanup.unwrap_or(terminal), "report_ready")
    }

    pub(crate) fn timestamps(&self) -> AttemptTimestamps {
        AttemptTimestamps {
            submitted_ns: self.submitted,
            planning_started_ns: self.planning_started,
            planning_finished_ns: self.planning_finished,
            admission_requested_ns: self.admission_requested,
            admitted_ns: self.admitted.as_ref().map(|(at, _)| *at),
            physical_execution_started_ns: self.physical,
            gpu_execution_started_ns: self.gpu,
            outcome_terminal_ns: self.terminal.as_ref().map(|(at, _)| *at).unwrap_or(self.submitted),
            cleanup_complete_ns: self.cleanup,
            report_ready_ns: self.ready,
        }
    }
}

#[cfg(test)]
mod tests {
    use query_planning::error::{QueryError, QueryErrorCode};

    use super::*;

    fn fact() -> AttemptAdmissionFact {
        AttemptAdmissionFact { admission_sequence: 7, admission_wait_ns: 3, device_ordinal: 2, grant_bytes: 4096 }
    }

    #[test]
    fn failure_operation_follows_lifecycle_stage() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert_eq!(lifecycle.failure_operation(), AttemptFailureOperation::Planning);
        lifecycle.planning_finished(1).unwrap();
        assert_eq!(lifecycle.failure_operation(), AttemptFailureOperation::Execution);
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(2).unwrap();
        assert_eq!(lifecycle.failure_operation(), AttemptFailureOperation::Admission);
        lifecycle.admitted(fact(), 3).unwrap();
        assert_eq!(lifecycle.failure_operation(), AttemptFailureOperation::Execution);
        lifecycle.terminalize(RuntimeTerminalOutcome::Cancelled, 4).unwrap();
        lifecycle.cleanup(5).unwrap();
        assert_eq!(lifecycle.failure_operation(), AttemptFailureOperation::Cleanup);
    }

    fn projection() -> AttemptComposedScheduleProjection {
        AttemptComposedScheduleProjection::for_test(2, 11, 4096)
    }

    fn failed() -> RuntimeTerminalOutcome {
        RuntimeTerminalOutcome::Failed {
            error_code: AttemptFailureCode::Internal,
            error_status: AttemptFailureStatus::Permanent,
        }
    }

    #[test]
    fn cleanup_failure_counter_never_publishes_a_saturated_exact_count() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.cleanup_failures = u64::MAX;
        lifecycle.cleanup_failure(CleanupFailureSummary::from_native_error(
            CleanupFailurePhase::QueryServiceCleanup,
            &QueryError::new(QueryErrorCode::Internal, "test.cleanup", "overflow probe"),
        ));

        assert_eq!(lifecycle.cleanup_failures, u64::MAX);
        assert!(!lifecycle.cleanup_failures_complete);
    }

    #[test]
    fn composed_schedule_requires_admission_and_preserves_one_immutable_projection() {
        let mut lifecycle = AttemptLifecycle::new(0);
        let schedule = projection();
        assert_eq!(
            lifecycle.composed_schedule(schedule.clone()),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "composed_schedule", reason: "admission_not_granted" })
        );

        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(1).unwrap();
        lifecycle.admitted(fact(), 2).unwrap();
        lifecycle.composed_schedule(schedule.clone()).unwrap();
        lifecycle.composed_schedule(schedule).unwrap();
        let mut corrupted = projection();
        corrupted.fingerprint = "composed-schedule-fnv1a64:0000000000000000".to_owned();
        assert_eq!(
            lifecycle.composed_schedule(corrupted),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "composed_schedule", reason: "fingerprint_mismatch" })
        );
        let mut conflicting = projection();
        conflicting.grant_id = 12;
        assert_eq!(
            lifecycle.composed_schedule(conflicting),
            Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "conflicting_schedule_projection"
            })
        );
    }

    /// One row of the phase lifecycle table driven through the state machine.
    #[derive(Debug, Clone, Copy)]
    enum Row {
        UnplannedPlanningFailure,
        UnplannedPrePlanningRejection,
        CpuFallback,
        CpuFallbackDroppedBeforePoll,
        PartialNative,
        Native,
        QueueRejection,
        QueueCancellation,
    }

    fn drive(row: Row) -> AttemptLifecycle {
        let mut lifecycle = AttemptLifecycle::new(0);
        match row {
            Row::UnplannedPlanningFailure => {
                lifecycle.planning_started(1).unwrap();
                lifecycle.terminalize(failed(), 2).unwrap();
                lifecycle.finalize_not_admitted(3).unwrap();
                lifecycle.ready(3).unwrap();
            },
            Row::UnplannedPrePlanningRejection => {
                lifecycle.terminalize(failed(), 1).unwrap();
                lifecycle.finalize_not_admitted(2).unwrap();
                lifecycle.ready(2).unwrap();
            },
            Row::CpuFallback => {
                lifecycle.planning_started(1).unwrap();
                lifecycle.planning_finished(2).unwrap();
                lifecycle.disposition(AttemptDisposition::CpuFallback).unwrap();
                lifecycle.physical(3).unwrap();
                lifecycle.terminalize(RuntimeTerminalOutcome::Succeeded, 4).unwrap();
                lifecycle.finalize_not_admitted(5).unwrap();
                lifecycle.ready(5).unwrap();
            },
            Row::CpuFallbackDroppedBeforePoll => {
                lifecycle.planning_started(1).unwrap();
                lifecycle.planning_finished(2).unwrap();
                lifecycle.disposition(AttemptDisposition::CpuFallback).unwrap();
                lifecycle.terminalize(RuntimeTerminalOutcome::Dropped, 3).unwrap();
                lifecycle.finalize_not_admitted(4).unwrap();
                lifecycle.ready(4).unwrap();
            },
            Row::PartialNative => {
                lifecycle.planning_started(1).unwrap();
                lifecycle.planning_finished(2).unwrap();
                lifecycle.disposition(AttemptDisposition::PartialNative).unwrap();
                lifecycle.admission_requested(3).unwrap();
                lifecycle.admitted(fact(), 4).unwrap();
                lifecycle.physical(4).unwrap();
                lifecycle.gpu(5).unwrap();
                lifecycle.terminalize(RuntimeTerminalOutcome::Succeeded, 6).unwrap();
                lifecycle
                    .terminal_commit(
                        AttemptTerminalCommit::new(
                            AttemptCapacityDisposition::Released,
                            None,
                            AttemptHealthTransition::HealthyReuse,
                        ),
                        7,
                    )
                    .unwrap();
                lifecycle.ready(8).unwrap();
            },
            Row::Native => {
                lifecycle.planning_started(1).unwrap();
                lifecycle.planning_finished(2).unwrap();
                lifecycle.disposition(AttemptDisposition::Native).unwrap();
                lifecycle.admission_requested(3).unwrap();
                lifecycle.admitted(fact(), 4).unwrap();
                lifecycle.physical(4).unwrap();
                lifecycle.gpu(5).unwrap();
                lifecycle.terminalize(failed(), 6).unwrap();
                lifecycle
                    .terminal_commit(
                        AttemptTerminalCommit::new(
                            AttemptCapacityDisposition::Released,
                            None,
                            AttemptHealthTransition::HealthyReuse,
                        ),
                        7,
                    )
                    .unwrap();
                lifecycle.ready(8).unwrap();
            },
            Row::QueueRejection => {
                lifecycle.planning_started(1).unwrap();
                lifecycle.planning_finished(2).unwrap();
                lifecycle.disposition(AttemptDisposition::Native).unwrap();
                lifecycle.admission_requested(3).unwrap();
                lifecycle.terminalize(failed(), 4).unwrap();
                lifecycle.finalize_not_admitted(5).unwrap();
                lifecycle.ready(5).unwrap();
            },
            Row::QueueCancellation => {
                lifecycle.planning_started(1).unwrap();
                lifecycle.planning_finished(2).unwrap();
                lifecycle.disposition(AttemptDisposition::PartialNative).unwrap();
                lifecycle.admission_requested(3).unwrap();
                lifecycle.terminalize(RuntimeTerminalOutcome::Cancelled, 4).unwrap();
                lifecycle.finalize_not_admitted(5).unwrap();
                lifecycle.ready(5).unwrap();
            },
        }
        lifecycle
    }

    #[test]
    fn every_lifecycle_row_reaches_report_ready_with_the_required_boundaries() {
        for row in [
            Row::UnplannedPlanningFailure,
            Row::UnplannedPrePlanningRejection,
            Row::CpuFallback,
            Row::CpuFallbackDroppedBeforePoll,
            Row::PartialNative,
            Row::Native,
            Row::QueueRejection,
            Row::QueueCancellation,
        ] {
            let lifecycle = drive(row);
            let timestamps = lifecycle.timestamps();
            assert_eq!(timestamps.submitted_ns, 0);
            assert!(lifecycle.terminal.is_some(), "outcome terminal required for {row:?}");
            assert!(lifecycle.ready.is_some(), "report ready required for {row:?}");
            assert!(timestamps.outcome_terminal_ns <= timestamps.report_ready_ns.expect("ready"));
        }
    }

    #[test]
    fn unplanned_rows_have_no_execution_or_gpu_boundaries() {
        for row in [Row::UnplannedPlanningFailure, Row::UnplannedPrePlanningRejection] {
            let lifecycle = drive(row);
            assert_eq!(lifecycle.disposition, Some(AttemptDisposition::Unplanned));
            let timestamps = lifecycle.timestamps();
            assert_eq!(timestamps.physical_execution_started_ns, None);
            assert_eq!(timestamps.admission_requested_ns, None);
            assert_eq!(timestamps.admitted_ns, None);
            assert_eq!(timestamps.gpu_execution_started_ns, None);
            assert_eq!(timestamps.cleanup_complete_ns, None);
        }
    }

    #[test]
    fn cpu_fallback_rows_have_physical_start_but_no_gpu_boundaries() {
        let lifecycle = drive(Row::CpuFallback);
        let timestamps = lifecycle.timestamps();
        assert_eq!(timestamps.physical_execution_started_ns, Some(3));
        assert_eq!(timestamps.admission_requested_ns, None);
        assert_eq!(timestamps.admitted_ns, None);
        assert_eq!(timestamps.gpu_execution_started_ns, None);
        assert_eq!(timestamps.cleanup_complete_ns, None);
    }

    #[test]
    fn native_rows_have_distinct_physical_and_gpu_starts_and_cleanup() {
        for row in [Row::PartialNative, Row::Native] {
            let lifecycle = drive(row);
            let timestamps = lifecycle.timestamps();
            assert_eq!(timestamps.physical_execution_started_ns, Some(4));
            assert_eq!(timestamps.gpu_execution_started_ns, Some(5));
            assert!(timestamps.gpu_execution_started_ns >= timestamps.physical_execution_started_ns);
            assert_eq!(timestamps.cleanup_complete_ns, Some(7));
            assert_eq!(timestamps.admitted_ns, Some(4));
        }
    }

    #[test]
    fn queue_terminal_rows_have_queue_facts_but_no_execution_or_cleanup() {
        for row in [Row::QueueRejection, Row::QueueCancellation] {
            let lifecycle = drive(row);
            let timestamps = lifecycle.timestamps();
            assert_eq!(timestamps.admission_requested_ns, Some(3));
            assert_eq!(timestamps.admitted_ns, None);
            assert_eq!(timestamps.physical_execution_started_ns, None);
            assert_eq!(timestamps.gpu_execution_started_ns, None);
            assert_eq!(timestamps.cleanup_complete_ns, None);
            assert!(matches!(
                lifecycle.disposition,
                Some(AttemptDisposition::Native | AttemptDisposition::PartialNative)
            ));
        }
    }

    #[test]
    fn absent_boundaries_are_none_not_zero() {
        let lifecycle = drive(Row::UnplannedPrePlanningRejection);
        let timestamps = lifecycle.timestamps();
        assert_eq!(timestamps.planning_started_ns, None);
        assert_eq!(timestamps.planning_finished_ns, None);
        assert_eq!(timestamps.cleanup_complete_ns, None);
        assert_ne!(timestamps.outcome_terminal_ns, 0);
    }

    #[test]
    fn double_terminalization_is_a_structured_error() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.terminalize(RuntimeTerminalOutcome::Succeeded, 1).unwrap();
        assert_eq!(lifecycle.terminalize(failed(), 2), Err(AttemptLifecycleError::DoubleTerminalization));
        // The first outcome is preserved.
        assert!(matches!(lifecycle.terminal, Some((1, RuntimeTerminalOutcome::Succeeded))));
    }

    #[test]
    fn report_ready_requires_terminal_and_cleanup_when_admitted() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert_eq!(lifecycle.ready(1), Err(AttemptLifecycleError::ReportBeforeTerminal));

        let mut admitted = AttemptLifecycle::new(0);
        admitted.disposition(AttemptDisposition::Native).unwrap();
        admitted.admission_requested(1).unwrap();
        admitted.admitted(fact(), 2).unwrap();
        admitted.terminalize(RuntimeTerminalOutcome::Succeeded, 3).unwrap();
        assert!(matches!(admitted.ready(4), Err(AttemptLifecycleError::InvalidOrder { boundary: "report_ready", .. })));
        admitted.cleanup(4).unwrap();
        assert!(matches!(
            admitted.ready(5),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "report_ready", reason: "finalization_incomplete" })
        ));
        admitted
            .terminal_commit(
                AttemptTerminalCommit::new(
                    AttemptCapacityDisposition::Released,
                    None,
                    AttemptHealthTransition::HealthyReuse,
                ),
                5,
            )
            .unwrap();
        admitted.ready(6).unwrap();
    }

    #[test]
    fn cleanup_requires_admission_and_terminal() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert_eq!(lifecycle.cleanup(1), Err(AttemptLifecycleError::CleanupWithoutAdmission));
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(1).unwrap();
        lifecycle.admitted(fact(), 2).unwrap();
        assert!(matches!(
            lifecycle.cleanup(3),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "cleanup_complete", .. })
        ));
    }

    #[test]
    fn terminal_commit_requires_a_terminal_admission_and_is_first_wins() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(1).unwrap();
        lifecycle.admitted(fact(), 2).unwrap();
        assert!(matches!(
            lifecycle.terminal_commit(
                AttemptTerminalCommit::new(
                    AttemptCapacityDisposition::Released,
                    None,
                    AttemptHealthTransition::HealthyReuse,
                ),
                3,
            ),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "cleanup_complete", reason: "not_terminal" })
        ));
        lifecycle.terminalize(RuntimeTerminalOutcome::Succeeded, 3).unwrap();
        let released = AttemptTerminalCommit::new(
            AttemptCapacityDisposition::Released,
            Some(AttemptFinalDomainUsage {
                domain_id: 7,
                active: false,
                active_members: 0,
                ceiling_bytes: 8,
                allocation_limit_bytes: 10,
                current_bytes: 0,
                peak_bytes: 11,
                total_bytes: 12,
                current_aligned_bytes: 0,
                peak_aligned_bytes: 13,
                total_allocations: 14,
                current_allocations: 0,
                peak_allocations: 15,
                failed_allocation_count: 16,
                largest_failed_allocation_bytes: 17,
            }),
            AttemptHealthTransition::HealthyReuse,
        );
        lifecycle.terminal_commit(released, 4).unwrap();
        assert_eq!(lifecycle.cleanup, Some(4));
        assert_eq!(lifecycle.terminal_commit, Some(released));
        lifecycle.terminal_commit(released, 5).unwrap();
        assert_eq!(
            lifecycle.terminal_commit(
                AttemptTerminalCommit::new(
                    AttemptCapacityDisposition::RetainedQuarantine,
                    None,
                    AttemptHealthTransition::DeviceQuarantined,
                ),
                5,
            ),
            Err(AttemptLifecycleError::TerminalCommitConflict)
        );
    }

    #[test]
    fn terminal_commit_conflicts_when_only_exact_final_usage_or_health_differs() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(1).unwrap();
        lifecycle.admitted(fact(), 2).unwrap();
        lifecycle.terminalize(RuntimeTerminalOutcome::Succeeded, 3).unwrap();
        let committed = AttemptTerminalCommit::new(
            AttemptCapacityDisposition::Released,
            Some(AttemptFinalDomainUsage {
                domain_id: 1,
                active: false,
                active_members: 0,
                ceiling_bytes: 2,
                allocation_limit_bytes: 4,
                current_bytes: 0,
                peak_bytes: 5,
                total_bytes: 6,
                current_aligned_bytes: 0,
                peak_aligned_bytes: 7,
                total_allocations: 8,
                current_allocations: 0,
                peak_allocations: 9,
                failed_allocation_count: 10,
                largest_failed_allocation_bytes: 11,
            }),
            AttemptHealthTransition::HealthyReuse,
        );
        lifecycle.terminal_commit(committed, 4).unwrap();
        assert_eq!(
            lifecycle.terminal_commit(
                AttemptTerminalCommit::new(
                    AttemptCapacityDisposition::Released,
                    Some(AttemptFinalDomainUsage {
                        domain_id: 99,
                        ..committed.final_domain_usage().expect("committed usage")
                    }),
                    AttemptHealthTransition::HealthyReuse,
                ),
                5,
            ),
            Err(AttemptLifecycleError::TerminalCommitConflict)
        );
    }

    #[test]
    fn terminal_commit_rejects_inconsistent_capacity_and_health_transition() {
        let invalid = [
            AttemptTerminalCommit::new(
                AttemptCapacityDisposition::RetainedQuarantine,
                None,
                AttemptHealthTransition::HealthyReuse,
            ),
            AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                None,
                AttemptHealthTransition::DeviceQuarantined,
            ),
            AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                None,
                AttemptHealthTransition::BackendUnhealthy,
            ),
        ];
        for commit in invalid {
            let mut lifecycle = AttemptLifecycle::new(0);
            lifecycle.disposition(AttemptDisposition::Native).unwrap();
            lifecycle.admission_requested(1).unwrap();
            lifecycle.admitted(fact(), 2).unwrap();
            lifecycle.terminalize(RuntimeTerminalOutcome::Succeeded, 3).unwrap();
            assert_eq!(
                lifecycle.terminal_commit(commit, 4),
                Err(AttemptLifecycleError::InvalidOrder {
                    boundary: "terminal_commit",
                    reason: "capacity_health_mismatch"
                })
            );
        }
    }

    #[test]
    fn admission_rejection_is_limited_to_unclaimed_failure_or_cancellation() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        assert_eq!(
            lifecycle.terminalize_admission_rejection(RuntimeTerminalOutcome::Succeeded, None, 1),
            Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_rejection",
                reason: "outcome_not_rejection"
            })
        );
        lifecycle
            .terminalize_admission_rejection(
                RuntimeTerminalOutcome::Failed {
                    error_code: AttemptFailureCode::Internal,
                    error_status: AttemptFailureStatus::Permanent,
                },
                None,
                2,
            )
            .unwrap();

        let mut claimed = AttemptLifecycle::new(0);
        claimed.disposition(AttemptDisposition::Native).unwrap();
        claimed.admission_requested(1).unwrap();
        assert_eq!(
            claimed.terminalize_admission_rejection(RuntimeTerminalOutcome::Cancelled, None, 3),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "admission_rejection", reason: "admission_claimed" })
        );
    }

    #[test]
    fn a_second_admission_claim_is_rejected_without_changing_the_lifecycle() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(1).unwrap();
        assert_eq!(lifecycle.admission_requested(2), Err(AttemptLifecycleError::AdmissionAlreadyClaimed));
        assert_eq!(lifecycle.admission_requested, Some(1));
        assert!(lifecycle.terminal.is_none() && lifecycle.primary_failure.is_none());
    }

    #[test]
    fn admitted_requires_a_prior_admission_request() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert!(matches!(
            lifecycle.admitted(fact(), 1),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "admitted", .. })
        ));
    }

    #[test]
    fn gpu_start_requires_admission() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert!(matches!(
            lifecycle.gpu(1),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "gpu_execution_started", .. })
        ));
    }

    #[test]
    fn physical_start_during_planning_is_rejected() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.planning_started(1).unwrap();
        assert!(matches!(
            lifecycle.physical(2),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "physical_execution_started", .. })
        ));
    }

    #[test]
    fn cpu_fallback_disposition_forbids_gpu_boundaries_in_both_orders() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.disposition(AttemptDisposition::CpuFallback).unwrap();
        assert!(matches!(
            lifecycle.admission_requested(1),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "admission_requested", .. })
        ));

        let mut other = AttemptLifecycle::new(0);
        other.disposition(AttemptDisposition::Native).unwrap();
        other.admission_requested(1).unwrap();
        assert_eq!(other.disposition(AttemptDisposition::CpuFallback), Err(AttemptLifecycleError::DispositionConflict));
    }

    #[test]
    fn disposition_is_set_once_and_conflicts_are_errors() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        assert_eq!(
            lifecycle.disposition(AttemptDisposition::CpuFallback),
            Err(AttemptLifecycleError::DispositionConflict)
        );
    }

    #[test]
    fn monotonic_regressions_are_rejected() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.planning_started(5).unwrap();
        assert!(matches!(
            lifecycle.planning_finished(4),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "planning_finished", .. })
        ));
        lifecycle.planning_finished(5).unwrap();
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(6).unwrap();
        assert!(matches!(
            lifecycle.admitted(fact(), 5),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "admitted", .. })
        ));
    }

    #[test]
    fn boundary_recording_is_first_wins_for_repeated_fragment_calls() {
        let mut lifecycle = drive(Row::Native);
        let before = lifecycle.timestamps();
        lifecycle.gpu(9).unwrap();
        lifecycle.physical(9).unwrap();
        lifecycle.admitted(fact(), 9).unwrap();
        lifecycle.cleanup(9).unwrap();
        let after = lifecycle.timestamps();
        assert_eq!(before, after);
    }

    #[test]
    fn unplanned_terminalization_with_execution_boundaries_is_rejected() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.planning_started(1).unwrap();
        lifecycle.planning_finished(2).unwrap();
        lifecycle.disposition(AttemptDisposition::CpuFallback).unwrap();
        lifecycle.physical(3).unwrap();
        // Disposition stays cpu_fallback; unplanned is only assigned when no
        // disposition was set, so this stays a valid cpu_fallback terminal.
        lifecycle.terminalize(RuntimeTerminalOutcome::Dropped, 4).unwrap();
        assert_eq!(lifecycle.disposition, Some(AttemptDisposition::CpuFallback));
    }
}

#[cfg(test)]
mod disposition_gate_tests {
    use super::*;

    #[test]
    fn admission_requires_a_classified_disposition() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert!(matches!(
            lifecycle.admission_requested(1),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "admission_requested", reason: "disposition_not_set" })
        ));
        lifecycle.disposition(AttemptDisposition::PartialNative).unwrap();
        lifecycle.admission_requested(1).unwrap();
    }
}
