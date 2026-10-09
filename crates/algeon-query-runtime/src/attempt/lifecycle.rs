//! Pure fixed execution-attempt lifecycle state.
//!
//! `AttemptLifecycle` is a state machine with no clock, lock, or I/O of its
//! own. The attempt owner holds it under the attempt lock and supplies every
//! timestamp, so each transition is atomic with respect to the report that
//! completion receivers read. Transitions validate before they mutate; a
//! rejected transition leaves the lifecycle unchanged.
//!
//! The phase order is `Unclaimed -> Claimed -> Admitted -> Terminal ->
//! Finalized -> Ready`. An attempt that never requests a device skips
//! `Claimed` and `Admitted`; a request rejected or cancelled in the queue
//! skips `Admitted`. `Terminal` fixes the outcome and every later phase
//! carries that same value. `Ready` is absorbing. The design contract lives in
//! `website/docs/design/admission.mdx`.
use std::{error::Error, fmt};

#[cfg(test)]
use query_planning::error::AttemptFailureCode;

use crate::observability::{
    AttemptAdmissionFact, AttemptAdmissionWaitFact, AttemptComposedScheduleProjection, AttemptDisposition,
    AttemptFailureOperation, AttemptFailureSummary, AttemptTerminalCommit, AttemptTimestamps,
    COMPOSED_SCHEDULE_PROJECTION_SCHEMA, RuntimeTerminalOutcome,
};
#[cfg(test)]
use crate::{
    observability::AttemptCapacityDisposition, observability::AttemptFailureStatus,
    observability::AttemptFinalDomainUsage, observability::AttemptHealthTransition,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptLifecycleError {
    /// A second terminal outcome was offered after one was fixed. The first
    /// stands; paths that race to terminalize read this as another path
    /// having decided already.
    DoubleTerminalization,
    /// A boundary arrived out of order, or with a timestamp earlier than the
    /// boundary before it. `reason` names the check that failed.
    InvalidOrder { boundary: &'static str, reason: &'static str },
    /// A disposition that differs from the recorded one. Repeating the same
    /// disposition is accepted.
    DispositionConflict,
    /// Finalization or report-ready was requested before the outcome was
    /// fixed.
    ReportBeforeTerminal,
    /// A cleanup commit for an attempt that never held a grant.
    CleanupWithoutAdmission,
    /// A second cleanup commit with different facts. An identical repeat is
    /// accepted.
    TerminalCommitConflict,
    /// The attempt's finalizer was already consumed, by an admission claim
    /// that still holds it or by an earlier not-admitted finalization. Raised
    /// by the owner module; the state machine itself never produces it.
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

/// The attempt's published admission grant and the facts attached to it.
#[derive(Debug, Clone)]
struct AttemptAdmitted {
    requested_ns: u64,
    admitted_ns: u64,
    fact: AttemptAdmissionFact,
    wait: Option<AttemptAdmissionWaitFact>,
    composed_schedule: Option<AttemptComposedScheduleProjection>,
}

/// How far admission got when the outcome became terminal.
#[derive(Debug, Clone)]
enum TerminalAdmission {
    Unclaimed,
    Claimed {
        requested_ns: u64,
    },
    /// The cleanup commit is still owed.
    Admitted(Box<AttemptAdmitted>),
}

/// How the attempt's one finalizer was consumed.
#[derive(Debug, Clone)]
enum FinalizedAdmission {
    NotAdmitted { requested_ns: Option<u64> },
    Admitted(Box<AdmittedCleanup>),
}

/// An admitted attempt's grant and the cleanup commit that consumed its
/// finalizer.
#[derive(Debug, Clone)]
struct AdmittedCleanup {
    admitted: AttemptAdmitted,
    cleanup_ns: u64,
    commit: AttemptTerminalCommit,
}

/// The attempt's progress through its one admission and its terminal
/// publication. Each phase carries the facts that exist at that point: an
/// admission fact without a claim, a cleanup commit without a grant, and
/// report-ready before finalization have no representation.
#[derive(Debug, Clone)]
enum AttemptPhase {
    /// Before the attempt claims admission.
    Unclaimed,
    /// The attempt holds its one admission claim.
    Claimed { requested_ns: u64 },
    /// Admission published a grant.
    Admitted(Box<AttemptAdmitted>),
    /// The outcome is terminal. `admission` keeps how far admission got so
    /// the cleanup commit or not-admitted finalization that follows can be
    /// validated against it.
    Terminal { terminal_ns: u64, outcome: RuntimeTerminalOutcome, admission: TerminalAdmission },
    /// The finalizer is consumed; the report is not ready yet.
    Finalized { terminal_ns: u64, outcome: RuntimeTerminalOutcome, admission: FinalizedAdmission },
    /// The report-ready boundary is recorded. No transition leaves this phase.
    Ready { terminal_ns: u64, ready_ns: u64, outcome: RuntimeTerminalOutcome, admission: FinalizedAdmission },
}

/// One attempt's lifecycle. Disposition, the planning and execution-start
/// timestamps, and the primary failure are set-once facts recorded
/// independently of admission order, so they stay fields beside the phase.
#[derive(Debug)]
pub(crate) struct AttemptLifecycle {
    submitted: u64,
    planning_started: Option<u64>,
    planning_finished: Option<u64>,
    physical: Option<u64>,
    gpu: Option<u64>,
    disposition: Option<AttemptDisposition>,
    primary_failure: Option<AttemptFailureSummary>,
    phase: AttemptPhase,
}

/// Records a set-once timestamp. A repeat is accepted without comparison
/// because the first value is the published one; a first value earlier than
/// `previous` is rejected as a regression.
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

impl AttemptPhase {
    fn requested_ns(&self) -> Option<u64> {
        match self {
            Self::Unclaimed => None,
            Self::Claimed { requested_ns } => Some(*requested_ns),
            Self::Admitted(admitted) => Some(admitted.requested_ns),
            Self::Terminal { admission, .. } => match admission {
                TerminalAdmission::Unclaimed => None,
                TerminalAdmission::Claimed { requested_ns } => Some(*requested_ns),
                TerminalAdmission::Admitted(admitted) => Some(admitted.requested_ns),
            },
            Self::Finalized { admission, .. } | Self::Ready { admission, .. } => match admission {
                FinalizedAdmission::NotAdmitted { requested_ns } => *requested_ns,
                FinalizedAdmission::Admitted(cleanup) => Some(cleanup.admitted.requested_ns),
            },
        }
    }

    fn admitted(&self) -> Option<&AttemptAdmitted> {
        match self {
            Self::Admitted(admitted) | Self::Terminal { admission: TerminalAdmission::Admitted(admitted), .. } => {
                Some(admitted)
            },
            Self::Finalized { admission: FinalizedAdmission::Admitted(cleanup), .. }
            | Self::Ready { admission: FinalizedAdmission::Admitted(cleanup), .. } => Some(&cleanup.admitted),
            Self::Unclaimed
            | Self::Claimed { .. }
            | Self::Terminal { .. }
            | Self::Finalized { .. }
            | Self::Ready { .. } => None,
        }
    }

    /// The admitted facts that may still change: before finalization.
    fn open_admitted(&mut self) -> Option<&mut AttemptAdmitted> {
        match self {
            Self::Admitted(admitted) | Self::Terminal { admission: TerminalAdmission::Admitted(admitted), .. } => {
                Some(admitted)
            },
            Self::Unclaimed
            | Self::Claimed { .. }
            | Self::Terminal { .. }
            | Self::Finalized { .. }
            | Self::Ready { .. } => None,
        }
    }

    fn terminal(&self) -> Option<(u64, &RuntimeTerminalOutcome)> {
        match self {
            Self::Terminal { terminal_ns, outcome, .. }
            | Self::Finalized { terminal_ns, outcome, .. }
            | Self::Ready { terminal_ns, outcome, .. } => Some((*terminal_ns, outcome)),
            Self::Unclaimed | Self::Claimed { .. } | Self::Admitted(_) => None,
        }
    }

    fn cleanup(&self) -> Option<(u64, AttemptTerminalCommit)> {
        match self {
            Self::Finalized { admission: FinalizedAdmission::Admitted(cleanup), .. }
            | Self::Ready { admission: FinalizedAdmission::Admitted(cleanup), .. } => {
                Some((cleanup.cleanup_ns, cleanup.commit))
            },
            Self::Unclaimed
            | Self::Claimed { .. }
            | Self::Admitted(_)
            | Self::Terminal { .. }
            | Self::Finalized { .. }
            | Self::Ready { .. } => None,
        }
    }
}

impl AttemptLifecycle {
    pub(crate) fn new(submitted: u64) -> Self {
        Self {
            submitted,
            planning_started: None,
            planning_finished: None,
            physical: None,
            gpu: None,
            disposition: None,
            primary_failure: None,
            phase: AttemptPhase::Unclaimed,
        }
    }

    pub(crate) fn recorded_disposition(&self) -> Option<AttemptDisposition> {
        self.disposition
    }

    pub(crate) fn primary_failure_summary(&self) -> Option<&AttemptFailureSummary> {
        self.primary_failure.as_ref()
    }

    /// Whether the attempt took its one admission claim.
    pub(crate) fn is_claimed(&self) -> bool {
        self.phase.requested_ns().is_some()
    }

    pub(crate) fn admission_fact(&self) -> Option<&AttemptAdmissionFact> {
        self.phase.admitted().map(|admitted| &admitted.fact)
    }

    pub(crate) fn admission_wait_fact(&self) -> Option<AttemptAdmissionWaitFact> {
        self.phase.admitted().and_then(|admitted| admitted.wait)
    }

    pub(crate) fn composed_schedule_projection(&self) -> Option<&AttemptComposedScheduleProjection> {
        self.phase.admitted().and_then(|admitted| admitted.composed_schedule.as_ref())
    }

    pub(crate) fn terminal_outcome(&self) -> Option<&RuntimeTerminalOutcome> {
        self.phase.terminal().map(|(_, outcome)| outcome)
    }

    pub(crate) fn cleanup_commit(&self) -> Option<AttemptTerminalCommit> {
        self.phase.cleanup().map(|(_, commit)| commit)
    }

    pub(crate) fn is_ready(&self) -> bool {
        matches!(self.phase, AttemptPhase::Ready { .. })
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
    /// phase changes, so a rejected claim leaves the lifecycle as it was.
    pub(crate) fn admission_requested(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        if self.is_claimed() {
            return Err(AttemptLifecycleError::AdmissionAlreadyClaimed);
        }
        if self.terminal_outcome().is_some() {
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
        if at < self.planning_finished.unwrap_or(self.submitted) {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_requested",
                reason: "timestamp_regressed",
            });
        }
        self.phase = AttemptPhase::Claimed { requested_ns: at };
        Ok(())
    }

    /// Records the published grant, moving the claim to `Admitted`. A repeat
    /// after admission is accepted and changes nothing.
    pub(crate) fn admitted(&mut self, fact: AttemptAdmissionFact, at: u64) -> Result<(), AttemptLifecycleError> {
        self.rejects_gpu_boundary("admitted")?;
        if self.phase.admitted().is_some() {
            return Ok(());
        }
        // The owner may terminalize while admission is in flight; a grant
        // published after that terminal would never execute.
        if self.terminal_outcome().is_some() {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "admitted", reason: "attempt_terminal" });
        }
        let AttemptPhase::Claimed { requested_ns } = self.phase else {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admitted",
                reason: "admission_not_requested",
            });
        };
        if at < requested_ns {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "admitted", reason: "timestamp_regressed" });
        }
        self.phase = AttemptPhase::Admitted(Box::new(AttemptAdmitted {
            requested_ns,
            admitted_ns: at,
            fact,
            wait: None,
            composed_schedule: None,
        }));
        Ok(())
    }

    /// Retains the controller-owned final queue state after admission has
    /// succeeded. A second identical observation is harmless; conflicting
    /// state would make the immutable sealed report ambiguous.
    pub(crate) fn admission_wait(&mut self, fact: AttemptAdmissionWaitFact) -> Result<(), AttemptLifecycleError> {
        let Some(admitted) = self.phase.open_admitted() else {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_wait",
                reason: "admission_not_granted",
            });
        };
        match admitted.wait {
            None => {
                admitted.wait = Some(fact);
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
        if projection.schema != COMPOSED_SCHEDULE_PROJECTION_SCHEMA {
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
        if self.phase.admitted().is_none() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "admission_not_granted",
            });
        }
        let Some(admitted) = self.phase.open_admitted() else {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "report_ready_already_recorded",
            });
        };
        if projection.device_ordinal != admitted.fact.device_ordinal {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "device_ordinal_mismatch",
            });
        }
        if projection.grant_bytes != admitted.fact.grant_bytes {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "grant_bytes_mismatch",
            });
        }
        match &admitted.composed_schedule {
            None => {
                admitted.composed_schedule = Some(projection);
                Ok(())
            },
            Some(current) if current == &projection => Ok(()),
            Some(_) => Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "conflicting_schedule_projection",
            }),
        }
    }

    /// Records the start of physical execution, once. Allowed without a grant
    /// because a CPU-fallback attempt executes physically with no device;
    /// refused while planning is open and for an unplanned attempt.
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
        let previous = self
            .phase
            .admitted()
            .map_or(self.planning_finished.unwrap_or(self.submitted), |admitted| admitted.admitted_ns);
        first(&mut self.physical, at, previous, "physical_execution_started")
    }

    /// Records the first GPU work, once. It requires a grant, because GPU
    /// execution without admission would mean device allocations the ledger
    /// never committed.
    pub(crate) fn gpu(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        let Some(admitted_ns) = self.phase.admitted().map(|admitted| admitted.admitted_ns) else {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "gpu_execution_started",
                reason: "not_admitted",
            });
        };
        first(&mut self.gpu, at, admitted_ns, "gpu_execution_started")
    }

    /// Classifies the attempt, once. `Unplanned` is refused after any
    /// admission, GPU, or physical boundary and `CpuFallback` after any
    /// admission or GPU boundary, so a disposition can never understate what
    /// the attempt already did.
    pub(crate) fn disposition(&mut self, d: AttemptDisposition) -> Result<(), AttemptLifecycleError> {
        match self.disposition {
            None => {
                let has_gpu = self.is_claimed() || self.gpu.is_some();
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

    /// Fixes the attempt's outcome. This is the only transition into
    /// `Terminal`; the admission-rejection path and the owner's release paths
    /// all pass through it.
    ///
    /// A missing disposition defaults to `Unplanned` and is then checked
    /// against the recorded boundaries like an explicit one. When no primary
    /// failure was captured, one is derived from `outcome`, which is how a
    /// control outcome (`Cancelled`, `Dropped`, `Panicked`) still publishes a
    /// text-free summary.
    pub(crate) fn terminalize(
        &mut self,
        outcome: RuntimeTerminalOutcome,
        at: u64,
    ) -> Result<(), AttemptLifecycleError> {
        let admission = match &self.phase {
            AttemptPhase::Unclaimed => TerminalAdmission::Unclaimed,
            AttemptPhase::Claimed { requested_ns } => TerminalAdmission::Claimed { requested_ns: *requested_ns },
            AttemptPhase::Admitted(admitted) => TerminalAdmission::Admitted(admitted.clone()),
            AttemptPhase::Terminal { .. } | AttemptPhase::Finalized { .. } | AttemptPhase::Ready { .. } => {
                return Err(AttemptLifecycleError::DoubleTerminalization);
            },
        };
        let admitted_ns = self.phase.admitted().map(|admitted| admitted.admitted_ns);
        let d = self.disposition.unwrap_or(AttemptDisposition::Unplanned);
        if matches!(d, AttemptDisposition::Unplanned)
            && (admitted_ns.is_some() || self.physical.is_some() || self.gpu.is_some())
        {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "terminal",
                reason: "unplanned_has_execution",
            });
        }
        if matches!(d, AttemptDisposition::CpuFallback) && (admitted_ns.is_some() || self.gpu.is_some()) {
            return Err(AttemptLifecycleError::InvalidOrder { boundary: "terminal", reason: "cpu_has_gpu" });
        }
        let last = self.gpu.or(self.physical).or(admitted_ns).or(self.planning_finished).unwrap_or(self.submitted);
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
        self.phase = AttemptPhase::Terminal { terminal_ns: at, outcome, admission };
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
        if self.is_claimed() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "admission_rejection",
                reason: "admission_claimed",
            });
        }
        if self.physical.is_some() || self.gpu.is_some() {
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
        if self.terminal_outcome().is_some() {
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
        if self.phase.cleanup().is_some() {
            AttemptFailureOperation::Cleanup
        } else if self.phase.admitted().is_some() {
            AttemptFailureOperation::Execution
        } else if self.is_claimed() {
            AttemptFailureOperation::Admission
        } else if self.planning_finished.is_none() {
            AttemptFailureOperation::Planning
        } else {
            AttemptFailureOperation::Execution
        }
    }

    /// Commits the cleanup and capacity outcome observed by the admitting
    /// query service, consuming the admitted attempt's finalizer. It must
    /// happen after the primary outcome but before the report-ready boundary;
    /// first-wins preserves exactly one release fact.
    pub(crate) fn terminal_commit(
        &mut self,
        commit: AttemptTerminalCommit,
        at: u64,
    ) -> Result<(), AttemptLifecycleError> {
        if self.is_ready() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "terminal_commit",
                reason: "report_ready_already_recorded",
            });
        }
        if let Some((_, current)) = self.phase.cleanup() {
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
        let (terminal_ns, outcome, admitted) = match &self.phase {
            AttemptPhase::Terminal { terminal_ns, outcome, admission: TerminalAdmission::Admitted(admitted) } => {
                (*terminal_ns, outcome.clone(), admitted.clone())
            },
            AttemptPhase::Admitted(_) => {
                return Err(AttemptLifecycleError::InvalidOrder {
                    boundary: "cleanup_complete",
                    reason: "not_terminal",
                });
            },
            AttemptPhase::Unclaimed
            | AttemptPhase::Claimed { .. }
            | AttemptPhase::Terminal { .. }
            | AttemptPhase::Finalized { .. }
            | AttemptPhase::Ready { .. } => return Err(AttemptLifecycleError::CleanupWithoutAdmission),
        };
        if at < terminal_ns {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "cleanup_complete",
                reason: "timestamp_regressed",
            });
        }
        self.phase = AttemptPhase::Finalized {
            terminal_ns,
            outcome,
            admission: FinalizedAdmission::Admitted(Box::new(AdmittedCleanup {
                admitted: *admitted,
                cleanup_ns: at,
                commit,
            })),
        };
        Ok(())
    }

    /// Consumes the owner-held finalizer for an attempt that never acquired an
    /// admission grant. This is deliberately separate from an admitted
    /// terminal commit: no cleanup, capacity, health, or domain facts exist.
    pub(crate) fn finalize_not_admitted(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        if self.phase.admitted().is_some() {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "attempt_finalization",
                reason: "admission_already_granted",
            });
        }
        let (terminal_ns, outcome, requested_ns) = match &self.phase {
            AttemptPhase::Terminal { terminal_ns, outcome, admission } => {
                let requested_ns = match admission {
                    TerminalAdmission::Unclaimed => None,
                    TerminalAdmission::Claimed { requested_ns } => Some(*requested_ns),
                    TerminalAdmission::Admitted(_) => unreachable!("an admitted phase was rejected above"),
                };
                (*terminal_ns, outcome.clone(), requested_ns)
            },
            AttemptPhase::Finalized { .. } | AttemptPhase::Ready { .. } => return Ok(()),
            AttemptPhase::Unclaimed | AttemptPhase::Claimed { .. } | AttemptPhase::Admitted(_) => {
                return Err(AttemptLifecycleError::ReportBeforeTerminal);
            },
        };
        // Keep the finalization point monotonic even though it has no public
        // timestamp field of its own. `ready` remains the published boundary.
        if at < terminal_ns {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "attempt_finalization",
                reason: "timestamp_regressed",
            });
        }
        self.phase = AttemptPhase::Finalized {
            terminal_ns,
            outcome,
            admission: FinalizedAdmission::NotAdmitted { requested_ns },
        };
        Ok(())
    }

    /// Records the report-ready boundary, the last transition. The finalizer
    /// must have been consumed first; from `Terminal` the reason is
    /// `finalization_incomplete`, which distinguishes a skipped finalizer from
    /// a missing outcome. Idempotent once ready.
    pub(crate) fn ready(&mut self, at: u64) -> Result<(), AttemptLifecycleError> {
        let (terminal_ns, outcome, admission) = match &self.phase {
            AttemptPhase::Finalized { terminal_ns, outcome, admission } => {
                (*terminal_ns, outcome.clone(), admission.clone())
            },
            AttemptPhase::Ready { .. } => return Ok(()),
            AttemptPhase::Terminal { .. } => {
                return Err(AttemptLifecycleError::InvalidOrder {
                    boundary: "report_ready",
                    reason: "finalization_incomplete",
                });
            },
            AttemptPhase::Unclaimed | AttemptPhase::Claimed { .. } | AttemptPhase::Admitted(_) => {
                return Err(AttemptLifecycleError::ReportBeforeTerminal);
            },
        };
        let previous = match &admission {
            FinalizedAdmission::Admitted(cleanup) => cleanup.cleanup_ns,
            FinalizedAdmission::NotAdmitted { .. } => terminal_ns,
        };
        if at < previous {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "report_ready",
                reason: "timestamp_regressed",
            });
        }
        self.phase = AttemptPhase::Ready { terminal_ns, ready_ns: at, outcome, admission };
        Ok(())
    }

    /// Projects the timeline into the report. `outcome_terminal_ns` falls back
    /// to the submission time while the attempt is still running; the owner
    /// builds the report only after `ready`, so a published report never
    /// carries that fallback.
    pub(crate) fn timestamps(&self) -> AttemptTimestamps {
        let terminal_ns = self.phase.terminal().map(|(at, _)| at);
        AttemptTimestamps {
            submitted_ns: self.submitted,
            planning_started_ns: self.planning_started,
            planning_finished_ns: self.planning_finished,
            admission_requested_ns: self.phase.requested_ns(),
            admitted_ns: self.phase.admitted().map(|admitted| admitted.admitted_ns),
            physical_execution_started_ns: self.physical,
            gpu_execution_started_ns: self.gpu,
            outcome_terminal_ns: terminal_ns.unwrap_or(self.submitted),
            cleanup_complete_ns: self.phase.cleanup().map(|(at, _)| at),
            report_ready_ns: match self.phase {
                AttemptPhase::Ready { ready_ns, .. } => Some(ready_ns),
                AttemptPhase::Unclaimed
                | AttemptPhase::Claimed { .. }
                | AttemptPhase::Admitted(_)
                | AttemptPhase::Terminal { .. }
                | AttemptPhase::Finalized { .. } => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use query_planning::error::QueryErrorCode;

    use super::*;

    fn fact() -> AttemptAdmissionFact {
        AttemptAdmissionFact { admission_sequence: 7, admission_wait_ns: 3, device_ordinal: 2, grant_bytes: 4096 }
    }

    fn projection() -> AttemptComposedScheduleProjection {
        AttemptComposedScheduleProjection::for_test(2, 11, 4096)
    }

    fn failed() -> RuntimeTerminalOutcome {
        RuntimeTerminalOutcome::Failed {
            error_code: AttemptFailureCode::Query(QueryErrorCode::Internal),
            error_status: AttemptFailureStatus::Permanent,
        }
    }

    fn released() -> AttemptTerminalCommit {
        AttemptTerminalCommit::new(AttemptCapacityDisposition::Released, None, AttemptHealthTransition::HealthyReuse)
    }

    fn phase_name(lifecycle: &AttemptLifecycle) -> &'static str {
        match lifecycle.phase {
            AttemptPhase::Unclaimed => "unclaimed",
            AttemptPhase::Claimed { .. } => "claimed",
            AttemptPhase::Admitted(_) => "admitted",
            AttemptPhase::Terminal { .. } => "terminal",
            AttemptPhase::Finalized { .. } => "finalized",
            AttemptPhase::Ready { .. } => "ready",
        }
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
        assert_eq!(lifecycle.failure_operation(), AttemptFailureOperation::Execution);
        lifecycle.terminal_commit(released(), 5).unwrap();
        assert_eq!(lifecycle.failure_operation(), AttemptFailureOperation::Cleanup);
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

    /// One attempt path from begin to report-ready.
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

    const ROWS: [Row; 8] = [
        Row::UnplannedPlanningFailure,
        Row::UnplannedPrePlanningRejection,
        Row::CpuFallback,
        Row::CpuFallbackDroppedBeforePoll,
        Row::PartialNative,
        Row::Native,
        Row::QueueRejection,
        Row::QueueCancellation,
    ];

    /// Drives `row` through the lifecycle and returns it with the phases it
    /// passed through, in order.
    fn drive(row: Row) -> (AttemptLifecycle, Vec<&'static str>) {
        let mut lifecycle = AttemptLifecycle::new(0);
        let mut phases = vec![phase_name(&lifecycle)];
        let mut step = |lifecycle: &mut AttemptLifecycle, apply: &dyn Fn(&mut AttemptLifecycle)| {
            apply(lifecycle);
            let phase = phase_name(lifecycle);
            if phases.last() != Some(&phase) {
                phases.push(phase);
            }
        };
        match row {
            Row::UnplannedPlanningFailure => {
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_started(1).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.terminalize(failed(), 2).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.finalize_not_admitted(3).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.ready(3).unwrap());
            },
            Row::UnplannedPrePlanningRejection => {
                step(&mut lifecycle, &|lifecycle| lifecycle.terminalize(failed(), 1).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.finalize_not_admitted(2).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.ready(2).unwrap());
            },
            Row::CpuFallback => {
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_started(1).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_finished(2).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.disposition(AttemptDisposition::CpuFallback).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.physical(3).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.terminalize(RuntimeTerminalOutcome::Succeeded, 4).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.finalize_not_admitted(5).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.ready(5).unwrap());
            },
            Row::CpuFallbackDroppedBeforePoll => {
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_started(1).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_finished(2).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.disposition(AttemptDisposition::CpuFallback).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.terminalize(RuntimeTerminalOutcome::Dropped, 3).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.finalize_not_admitted(4).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.ready(4).unwrap());
            },
            Row::PartialNative | Row::Native => {
                let disposition = match row {
                    Row::PartialNative => AttemptDisposition::PartialNative,
                    _ => AttemptDisposition::Native,
                };
                let outcome = match row {
                    Row::PartialNative => RuntimeTerminalOutcome::Succeeded,
                    _ => failed(),
                };
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_started(1).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_finished(2).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.disposition(disposition).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.admission_requested(3).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.admitted(fact(), 4).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.physical(4).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.gpu(5).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.terminalize(outcome.clone(), 6).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.terminal_commit(released(), 7).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.ready(8).unwrap());
            },
            Row::QueueRejection | Row::QueueCancellation => {
                let (disposition, outcome) = match row {
                    Row::QueueRejection => (AttemptDisposition::Native, failed()),
                    _ => (AttemptDisposition::PartialNative, RuntimeTerminalOutcome::Cancelled),
                };
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_started(1).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.planning_finished(2).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.disposition(disposition).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.admission_requested(3).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.terminalize(outcome.clone(), 4).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.finalize_not_admitted(5).unwrap());
                step(&mut lifecycle, &|lifecycle| lifecycle.ready(5).unwrap());
            },
        }
        (lifecycle, phases)
    }

    #[test]
    fn every_row_moves_through_its_phases_in_order() {
        for row in ROWS {
            let expected: &[&str] = match row {
                Row::UnplannedPlanningFailure
                | Row::UnplannedPrePlanningRejection
                | Row::CpuFallback
                | Row::CpuFallbackDroppedBeforePoll => &["unclaimed", "terminal", "finalized", "ready"],
                Row::PartialNative | Row::Native => {
                    &["unclaimed", "claimed", "admitted", "terminal", "finalized", "ready"]
                },
                Row::QueueRejection | Row::QueueCancellation => {
                    &["unclaimed", "claimed", "terminal", "finalized", "ready"]
                },
            };
            assert_eq!(drive(row).1, expected, "{row:?}");
        }
    }

    #[test]
    fn every_row_reaches_report_ready_after_its_terminal_boundary() {
        for row in ROWS {
            let (lifecycle, _) = drive(row);
            let timestamps = lifecycle.timestamps();
            assert_eq!(timestamps.submitted_ns, 0);
            assert!(lifecycle.terminal_outcome().is_some(), "outcome terminal required for {row:?}");
            assert!(lifecycle.is_ready(), "report ready required for {row:?}");
            assert!(timestamps.outcome_terminal_ns <= timestamps.report_ready_ns.expect("ready"));
        }
    }

    #[test]
    fn unplanned_rows_have_no_execution_or_gpu_boundaries() {
        for row in [Row::UnplannedPlanningFailure, Row::UnplannedPrePlanningRejection] {
            let (lifecycle, _) = drive(row);
            assert_eq!(lifecycle.recorded_disposition(), Some(AttemptDisposition::Unplanned));
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
        let (lifecycle, _) = drive(Row::CpuFallback);
        assert_eq!(lifecycle.recorded_disposition(), Some(AttemptDisposition::CpuFallback));
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
            let (lifecycle, _) = drive(row);
            let timestamps = lifecycle.timestamps();
            assert_eq!(timestamps.physical_execution_started_ns, Some(4));
            assert_eq!(timestamps.gpu_execution_started_ns, Some(5));
            assert_eq!(timestamps.cleanup_complete_ns, Some(7));
            assert_eq!(timestamps.admitted_ns, Some(4));
            assert_eq!(lifecycle.cleanup_commit(), Some(released()));
        }
    }

    #[test]
    fn queue_terminal_rows_keep_their_claim_but_no_execution_or_cleanup() {
        for row in [Row::QueueRejection, Row::QueueCancellation] {
            let (lifecycle, _) = drive(row);
            let timestamps = lifecycle.timestamps();
            assert_eq!(timestamps.admission_requested_ns, Some(3));
            assert_eq!(timestamps.admitted_ns, None);
            assert_eq!(timestamps.physical_execution_started_ns, None);
            assert_eq!(timestamps.gpu_execution_started_ns, None);
            assert_eq!(timestamps.cleanup_complete_ns, None);
            assert!(lifecycle.is_claimed());
            assert!(matches!(
                lifecycle.recorded_disposition(),
                Some(AttemptDisposition::Native | AttemptDisposition::PartialNative)
            ));
        }
    }

    #[test]
    fn absent_boundaries_are_none_not_zero() {
        let (lifecycle, _) = drive(Row::UnplannedPrePlanningRejection);
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
        assert_eq!(lifecycle.terminal_outcome(), Some(&RuntimeTerminalOutcome::Succeeded));
        assert_eq!(lifecycle.timestamps().outcome_terminal_ns, 1);
    }

    #[test]
    fn report_ready_requires_terminal_and_finalization() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert_eq!(lifecycle.ready(1), Err(AttemptLifecycleError::ReportBeforeTerminal));

        let mut admitted = AttemptLifecycle::new(0);
        admitted.disposition(AttemptDisposition::Native).unwrap();
        admitted.admission_requested(1).unwrap();
        admitted.admitted(fact(), 2).unwrap();
        assert_eq!(admitted.ready(3), Err(AttemptLifecycleError::ReportBeforeTerminal));
        admitted.terminalize(RuntimeTerminalOutcome::Succeeded, 3).unwrap();
        assert_eq!(
            admitted.ready(4),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "report_ready", reason: "finalization_incomplete" })
        );
        assert_eq!(
            admitted.finalize_not_admitted(4),
            Err(AttemptLifecycleError::InvalidOrder {
                boundary: "attempt_finalization",
                reason: "admission_already_granted"
            })
        );
        admitted.terminal_commit(released(), 5).unwrap();
        assert_eq!(
            admitted.ready(4),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "report_ready", reason: "timestamp_regressed" })
        );
        admitted.ready(6).unwrap();
        assert_eq!(admitted.timestamps().report_ready_ns, Some(6));
    }

    #[test]
    fn cleanup_commit_requires_admission_and_terminal() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert_eq!(lifecycle.terminal_commit(released(), 1), Err(AttemptLifecycleError::CleanupWithoutAdmission));
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(1).unwrap();
        assert_eq!(lifecycle.terminal_commit(released(), 2), Err(AttemptLifecycleError::CleanupWithoutAdmission));
        lifecycle.admitted(fact(), 2).unwrap();
        assert_eq!(
            lifecycle.terminal_commit(released(), 3),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "cleanup_complete", reason: "not_terminal" })
        );

        let mut not_admitted = AttemptLifecycle::new(0);
        not_admitted.terminalize(failed(), 1).unwrap();
        not_admitted.finalize_not_admitted(2).unwrap();
        assert_eq!(not_admitted.terminal_commit(released(), 3), Err(AttemptLifecycleError::CleanupWithoutAdmission));
    }

    #[test]
    fn terminal_commit_requires_a_terminal_admission_and_is_first_wins() {
        let mut lifecycle = AttemptLifecycle::new(0);
        lifecycle.disposition(AttemptDisposition::Native).unwrap();
        lifecycle.admission_requested(1).unwrap();
        lifecycle.admitted(fact(), 2).unwrap();
        assert!(matches!(
            lifecycle.terminal_commit(released(), 3),
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
        assert_eq!(lifecycle.timestamps().cleanup_complete_ns, Some(4));
        assert_eq!(lifecycle.cleanup_commit(), Some(released));
        lifecycle.terminal_commit(released, 5).unwrap();
        assert_eq!(lifecycle.timestamps().cleanup_complete_ns, Some(4));
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
                    error_code: AttemptFailureCode::Query(QueryErrorCode::Internal),
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
        assert_eq!(lifecycle.timestamps().admission_requested_ns, Some(1));
        assert!(lifecycle.terminal_outcome().is_none() && lifecycle.primary_failure_summary().is_none());
    }

    fn admitted_requires_a_prior_admission_request_case() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert!(matches!(
            lifecycle.admitted(fact(), 1),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "admitted", .. })
        ));
    }

    #[test]
    fn admitted_requires_a_prior_admission_request() {
        admitted_requires_a_prior_admission_request_case();
        gpu_start_requires_admission();
        physical_start_during_planning_is_rejected();
    }

    fn gpu_start_requires_admission() {
        let mut lifecycle = AttemptLifecycle::new(0);
        assert!(matches!(
            lifecycle.gpu(1),
            Err(AttemptLifecycleError::InvalidOrder { boundary: "gpu_execution_started", .. })
        ));
    }

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
        let (mut lifecycle, _) = drive(Row::Native);
        let before = lifecycle.timestamps();
        lifecycle.gpu(9).unwrap();
        lifecycle.physical(9).unwrap();
        lifecycle.admitted(fact(), 9).unwrap();
        lifecycle.ready(9).unwrap();
        let after = lifecycle.timestamps();
        assert_eq!(before, after);
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
