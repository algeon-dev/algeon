//! Attempt ownership, boundary recording, and independent completion delivery.
use std::{
    fmt,
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex, atomic::Ordering},
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

use super::{
    AttemptBackendCounters, AttemptCancellation, AttemptCompletionQueue, AttemptLifecycle, AttemptLifecycleError,
};
#[cfg(test)]
use crate::observability::{AttemptFinalDomainUsage, AttemptHealthTransition};
use crate::{
    error::QueryError,
    observability::{
        AttemptAdmissionFact, AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptId,
        AttemptQueryId, AttemptTerminalCommit, AttemptTraceContext, QUERY_ATTEMPT_REPORT_SCHEMA_V1, QueryAttemptReport,
        RuntimeCleanupFailureObservation, RuntimeObservationClock, RuntimeObservationConfig, RuntimeObservationFooter,
        RuntimeObservationHandle, RuntimeObservationOwner, RuntimeObservationPayload, RuntimeQueryId,
        RuntimeTerminalOutcome, RuntimeTraceId, SealedRuntimeObservationSnapshot, SystemRuntimeObservationClock,
    },
    runtime::ObservationPermit,
};

/// Caller inputs to begin one execution attempt before planning.
///
/// Correlation values are validated by their constructors; oversized input is
/// rejected structurally before planning and is never truncated.
pub struct BeginAttemptRequest {
    pub query_id: Option<AttemptQueryId>,
    pub trace_context: Option<AttemptTraceContext>,
    pub deadline: Option<Instant>,
    pub observation: RuntimeObservationConfig,
}

impl Default for BeginAttemptRequest {
    fn default() -> Self {
        Self {
            query_id: None,
            trace_context: None,
            deadline: None,
            observation: RuntimeObservationConfig::native_execution(),
        }
    }
}

impl fmt::Debug for BeginAttemptRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BeginAttemptRequest")
            .field("query_id", &self.query_id)
            .field("trace_context", &self.trace_context)
            .field("deadline", &self.deadline)
            .finish_non_exhaustive()
    }
}

/// The attempt owner and its independent completion, created together at begin.
pub struct AttemptBegin {
    pub owner: AttemptOwner,
    pub completion: AttemptCompletion,
}

impl fmt::Debug for AttemptBegin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptBegin").field("attempt_id", &self.owner.attempt_id()).finish_non_exhaustive()
    }
}

/// Terminalization authority for one execution attempt.
///
/// This type is intentionally not [`Clone`]: boundary recording is shared
/// through [`AttemptHandle`], but exactly one owner may terminalize.
pub struct AttemptOwner(Arc<AttemptInner>);

/// Cloneable record-only capability shared by every native fragment of one
/// invocation. It cannot terminalize the attempt.
pub struct AttemptHandle(Arc<AttemptCore>);

/// Opaque authority for rejecting an admission request before it receives a
/// grant.
///
/// This is intentionally narrower than [`AttemptOwner`]: it can only commit
/// a structured pre-admission rejection or cancellation, and it cannot seal
/// an observation, terminalize an admitted attempt, or record arbitrary
/// lifecycle boundaries.
pub struct AttemptAdmissionRejection(Arc<AttemptCore>);

/// Opaque, identity-bound admission attachment for one execution attempt.
///
/// [`AttemptOwner`] mints this as one unit so an admission request can never
/// combine one attempt's observation owner with another attempt's cleanup
/// capability. It exposes neither terminalization nor observation-seal
/// authority to the request caller.
#[derive(Clone)]
pub struct AttemptAdmissionBinding {
    observation_owner: RuntimeObservationOwner,
    release_commit: AttemptReleaseCommit,
}

/// Crate-private cleanup authority carried by [`AttemptAdmissionBinding`].
///
/// Only query-service cleanup may apply its terminal facts. Keeping it behind
/// the binding prevents external request construction from pairing it with a
/// different attempt's observation owner.
pub(crate) struct AttemptReleaseCommit(Arc<AttemptCore>);

impl Clone for AttemptHandle {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl Clone for AttemptAdmissionRejection {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl Clone for AttemptReleaseCommit {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

/// Independent receiver for the one bounded terminal attempt report.
///
/// Cloning or dropping receivers never cancels cleanup, report production, or
/// exact metric updates, and holding a receiver retains no device lease.
///
/// Receivers are counted so publication can route never-received reports to
/// the optional abandoned-report queue. Adapter internals hold
/// [`Self::uncounted`] factories instead, so an attempt object or result
/// stream never masks caller abandonment.
pub struct AttemptCompletion {
    cell: Arc<AttemptCompletionCell>,
    counted: bool,
}

impl Clone for AttemptCompletion {
    fn clone(&self) -> Self {
        let mut state = self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.receivers = state.receivers.saturating_add(1);
        drop(state);
        Self { cell: Arc::clone(&self.cell), counted: true }
    }
}

#[derive(Debug)]
struct CompletionState {
    report: Option<Arc<QueryAttemptReport>>,
    wakers: Vec<Waker>,
    // Receiver counting shares the publication mutex so clone/drop linearize
    // against the abandoned-at-publish decision.
    receivers: u64,
}

#[derive(Debug)]
struct AttemptCompletionCell {
    state: Mutex<CompletionState>,
    changed: Condvar,
}

struct AttemptCore {
    attempt_id: AttemptId,
    query_id: Option<AttemptQueryId>,
    trace_context: Option<AttemptTraceContext>,
    lifecycle: Mutex<AttemptLifecycle>,
    clock: Arc<dyn RuntimeObservationClock>,
    cancellation: AttemptCancellation,
    // Set once during construction after the observation owner exists. A
    // `RuntimeObservationHandle` is cloneable record-only state, so holding it
    // here cannot cycle back into the attempt core.
    observation: std::sync::OnceLock<RuntimeObservationHandle>,
    cell: Arc<AttemptCompletionCell>,
    counters: Arc<AttemptBackendCounters>,
    completion_queue: Option<Arc<AttemptCompletionQueue>>,
    // The begin-vs-close linearization artifact. It fences observation
    // quiescence only until the report is published: a caller holding a
    // finished stream or completion must never delay backend close.
    observation_permit: Mutex<Option<ObservationPermit>>,
}

struct AttemptInner {
    core: Arc<AttemptCore>,
    observation_owner: RuntimeObservationOwner,
}

impl fmt::Debug for AttemptCore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptCore").field("attempt_id", &self.attempt_id).finish_non_exhaustive()
    }
}

impl AttemptCore {
    fn lock_lifecycle(&self) -> std::sync::MutexGuard<'_, AttemptLifecycle> {
        self.lifecycle.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn monotonic_ns(&self) -> u64 {
        self.clock.monotonic_ns()
    }

    /// Builds the fixed report from lifecycle state. The sole seal commit
    /// applies observation completeness before this value is made visible to
    /// either the sealed snapshot or completion receivers.
    fn report(&self) -> QueryAttemptReport {
        let lifecycle = self.lock_lifecycle();
        let terminal_commit = lifecycle.terminal_commit;
        let cleanup_failure_count = lifecycle.cleanup_failures;
        QueryAttemptReport {
            schema: QUERY_ATTEMPT_REPORT_SCHEMA_V1.to_string(),
            attempt_id: self.attempt_id,
            query_id: self.query_id.clone(),
            trace_context: self.trace_context.clone(),
            disposition: lifecycle.disposition.unwrap_or(AttemptDisposition::Unplanned),
            outcome: lifecycle
                .terminal
                .as_ref()
                .map(|(_, outcome)| outcome.clone())
                .unwrap_or(RuntimeTerminalOutcome::Dropped),
            device_ordinal: lifecycle.admitted.as_ref().map(|(_, fact)| fact.device_ordinal),
            admission: lifecycle.admitted.as_ref().map(|(_, fact)| fact.clone()),
            timestamps: lifecycle.timestamps(),
            cleanup_disposition: terminal_commit.map_or_else(
                || {
                    if lifecycle.admitted.is_some() {
                        if cleanup_failure_count == 0 {
                            AttemptCleanupDisposition::Completed
                        } else {
                            AttemptCleanupDisposition::CompletedWithFailures
                        }
                    } else {
                        AttemptCleanupDisposition::NotRequired
                    }
                },
                |commit| {
                    if cleanup_failure_count == 0 {
                        commit.cleanup_disposition()
                    } else {
                        AttemptCleanupDisposition::CompletedWithFailures
                    }
                },
            ),
            cleanup_failure_count,
            capacity_disposition: terminal_commit.map_or_else(
                || {
                    if lifecycle.admitted.is_some() {
                        AttemptCapacityDisposition::PendingRelease
                    } else {
                        AttemptCapacityDisposition::NotAdmitted
                    }
                },
                AttemptTerminalCommit::capacity_disposition,
            ),
            final_domain_usage: terminal_commit.and_then(AttemptTerminalCommit::final_domain_usage),
            health_transition: terminal_commit.map(AttemptTerminalCommit::health_transition),
            observation_lost_records: 0,
            observation_retained_records: 0,
            observation_retained_bytes: 0,
            observation_truncated: false,
            observation_degraded: false,
            unix_epoch_ns: self.clock.unix_epoch_ns(),
        }
    }

    /// Establishes the lifecycle's sole terminal outcome. A terminal already
    /// committed by the attempt owner always wins over a release fallback.
    fn terminalize_for_release(
        &self,
        fallback: RuntimeTerminalOutcome,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        let mut lifecycle = self.lock_lifecycle();
        if let Some((_, outcome)) = &lifecycle.terminal {
            return Ok(outcome.clone());
        }
        lifecycle.terminalize(fallback, self.monotonic_ns())?;
        Ok(lifecycle.terminal.as_ref().expect("terminal outcome was recorded").1.clone())
    }

    /// Resolves the sole terminal outcome at seal. Recorder-local pending
    /// outcomes are only an early observation; the attempt lifecycle remains
    /// canonical once it has a terminal state.
    fn canonical_terminal_outcome(&self, fallback: RuntimeTerminalOutcome) -> RuntimeTerminalOutcome {
        self.terminalize_for_release(fallback.clone()).unwrap_or(fallback)
    }

    /// The single seal-and-completion commit: cleanup commit (admitted only),
    /// exact aggregates, report-ready, and publication of one immutable
    /// report. The returned `Arc` is embedded directly in the sealed snapshot.
    fn finish_with_footer(
        &self,
        footer: &RuntimeObservationFooter,
    ) -> Result<Arc<QueryAttemptReport>, AttemptLifecycleError> {
        {
            let mut lifecycle = self.lock_lifecycle();
            if lifecycle.ready.is_none() {
                if lifecycle.admitted.is_some() {
                    lifecycle.cleanup(self.monotonic_ns())?;
                }
                lifecycle.ready(self.monotonic_ns())?;
            }
        }
        let report = {
            let mut report = self.report();
            report.set_observation_completeness(footer);
            Arc::new(report)
        };
        let mut state = self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = &state.report {
            return Ok(Arc::clone(existing));
        }
        self.counters.reports_ready.fetch_add(1, Ordering::AcqRel);
        state.report = Some(Arc::clone(&report));
        let wakers = std::mem::take(&mut state.wakers);
        let abandoned = state.receivers == 0;
        drop(state);
        self.cell.changed.notify_all();
        for waker in wakers {
            waker.wake();
        }
        // Abandoned-report delivery is best-effort: overflow is counted and
        // never blocks planning, GPU workers, or cleanup.
        if abandoned && let Some(queue) = &self.completion_queue {
            if queue.try_push(Arc::clone(&report)) {
                self.counters.completion_queue_pushed.fetch_add(1, Ordering::AcqRel);
            } else {
                self.counters.completion_queue_overflowed.fetch_add(1, Ordering::AcqRel);
            }
        }
        // The report is durable: the attempt's observation permit has nothing
        // left to fence and is released before any waiter observes close
        // quiescence. Report and completion state carry no lease.
        self.observation_permit.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        Ok(report)
    }

    fn finish_without_observation(&self) -> Result<Arc<QueryAttemptReport>, AttemptLifecycleError> {
        self.finish_with_footer(&RuntimeObservationFooter {
            complete: true,
            terminal: None,
            retained_records: 0,
            retained_bytes: 0,
            first_dropped_sequence: None,
            last_attempted_sequence: None,
            dropped_by_class: Vec::new(),
            dropped_by_reason: Vec::new(),
            truncated: false,
            degraded: false,
            poison_recoveries: 0,
            counter_overflow: false,
        })
    }
}

impl AttemptOwner {
    pub(crate) fn begin(
        request: BeginAttemptRequest,
        permit: ObservationPermit,
        counters: Arc<AttemptBackendCounters>,
        queue: Option<Arc<AttemptCompletionQueue>>,
    ) -> AttemptBegin {
        Self::begin_with_clock(request, permit, counters, queue, Arc::new(SystemRuntimeObservationClock::new()))
    }

    pub(crate) fn begin_with_clock(
        request: BeginAttemptRequest,
        permit: ObservationPermit,
        counters: Arc<AttemptBackendCounters>,
        queue: Option<Arc<AttemptCompletionQueue>>,
        clock: Arc<dyn RuntimeObservationClock>,
    ) -> AttemptBegin {
        // A disabled diagnostic config is normalized to a zero-limit bounded
        // session so the recorder still seals and delivers observers: the
        // fixed summary and completion must survive any diagnostic loss.
        let observation = if request.observation.is_enabled() {
            request.observation
        } else {
            RuntimeObservationConfig::bounded(0, 0)
        };
        let cell = Arc::new(AttemptCompletionCell {
            state: Mutex::new(CompletionState { report: None, wakers: Vec::new(), receivers: 1 }),
            changed: Condvar::new(),
        });
        let core = Arc::new(AttemptCore {
            attempt_id: AttemptId::mint(),
            query_id: request.query_id,
            trace_context: request.trace_context,
            lifecycle: Mutex::new(AttemptLifecycle::new(clock.monotonic_ns())),
            clock: Arc::clone(&clock),
            cancellation: AttemptCancellation::new(request.deadline),
            observation: std::sync::OnceLock::new(),
            cell: Arc::clone(&cell),
            counters,
            completion_queue: queue,
            observation_permit: Mutex::new(None),
        });
        // The observation manifest uses the canonical unassigned sentinel when
        // the caller supplied no correlation; the attempt id is the identity.
        let query_id = core
            .query_id
            .as_ref()
            .map(|query_id| RuntimeQueryId::new(query_id.as_str()))
            .unwrap_or_else(|| RuntimeQueryId::new(""));
        let trace_id = core.trace_context.as_ref().map(|trace| RuntimeTraceId::new(trace.as_str()));
        let terminal_outcome_core = Arc::clone(&core);
        let report_core = Arc::clone(&core);
        let owner = RuntimeObservationOwner::with_clock(query_id, trace_id, observation, clock)
            .with_terminal_outcome_resolver(move |fallback| terminal_outcome_core.canonical_terminal_outcome(fallback))
            .with_attempt_report_seal_provider(move |footer| {
                report_core
                    .finish_with_footer(footer)
                    .expect("attempt lifecycle must complete while sealing its sole immutable report")
            });
        core.observation.set(owner.handle()).expect("attempt observation handle is set exactly once");
        core.observation_permit.lock().unwrap_or_else(std::sync::PoisonError::into_inner).replace(permit);
        AttemptBegin {
            owner: Self(Arc::new(AttemptInner { core, observation_owner: owner })),
            completion: AttemptCompletion { cell, counted: true },
        }
    }

    /// The engine-minted identity of this invocation.
    pub fn attempt_id(&self) -> AttemptId {
        self.0.core.attempt_id
    }

    /// Record-only capability for native fragments of this invocation.
    pub fn handle(&self) -> AttemptHandle {
        AttemptHandle(Arc::clone(&self.0.core))
    }

    /// Opaque capability that can terminalize only an admission request that
    /// was rejected before an admission grant was published.
    pub fn admission_rejection(&self) -> AttemptAdmissionRejection {
        AttemptAdmissionRejection(Arc::clone(&self.0.core))
    }

    /// Opaque, atomic attachment for the query-service admission request.
    ///
    /// Its observation owner and release capability originate from this same
    /// attempt and are consumed together by admission. Standalone observation
    /// users continue to attach only a [`RuntimeObservationOwner`].
    pub fn admission_binding(&self) -> AttemptAdmissionBinding {
        AttemptAdmissionBinding {
            observation_owner: self.0.observation_owner.clone(),
            release_commit: AttemptReleaseCommit(Arc::clone(&self.0.core)),
        }
    }

    /// An independent receiver for the terminal report.
    pub fn completion(&self) -> AttemptCompletion {
        self.0.core.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).receivers += 1;
        AttemptCompletion { cell: Arc::clone(&self.0.core.cell), counted: true }
    }

    /// The attempt-scoped cooperative cancellation signal and deadline.
    pub fn cancellation(&self) -> AttemptCancellation {
        self.0.core.cancellation.clone()
    }

    /// Test-only access to the owner behind an attempt binding.
    ///
    /// Production callers receive only the read-only APIs on
    /// [`AttemptAdmissionBinding`], so they cannot detach observation sealing
    /// from the corresponding release commitment.
    #[cfg(test)]
    pub(crate) fn observation_owner_for_test(&self) -> RuntimeObservationOwner {
        self.0.observation_owner.clone()
    }

    /// The one terminal commit hook: outcome terminal, then (for admitted
    /// attempts) cleanup-complete and report-ready are deferred to the
    /// post-GPU-cleanup seal; otherwise the commit completes immediately.
    pub fn commit_terminal(&self, outcome: RuntimeTerminalOutcome) -> Result<(), AttemptLifecycleError> {
        let admitted = {
            let mut lifecycle = self.0.core.lock_lifecycle();
            lifecycle.terminalize(outcome.clone(), self.0.core.monotonic_ns())?;
            lifecycle.admitted.is_some()
        };
        if !admitted {
            self.0.observation_owner.finish_query(outcome);
            // Seal-observer delivery is the primary completion path; finish
            // here as an idempotent backstop so no delivery ordering can lose
            // the report.
            let report_ready = self.0.core.lock_lifecycle().ready.is_some();
            if !report_ready {
                match self.0.observation_owner.sealed_snapshot() {
                    Some(snapshot) => {
                        let _ = self.0.core.finish_with_footer(snapshot.footer())?;
                    },
                    None => {
                        let _ = self.0.core.finish_without_observation()?;
                    },
                }
            }
        }
        Ok(())
    }
}

impl Drop for AttemptOwner {
    fn drop(&mut self) {
        let report_ready = self.0.core.lock_lifecycle().ready.is_some();
        if !report_ready {
            let _ = self.commit_terminal(if std::thread::panicking() {
                RuntimeTerminalOutcome::Panicked
            } else {
                RuntimeTerminalOutcome::Dropped
            });
        }
    }
}

impl fmt::Debug for AttemptOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptOwner").field("attempt_id", &self.attempt_id()).finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptHandle").field("attempt_id", &self.attempt_id()).finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptAdmissionRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttemptAdmissionRejection")
            .field("attempt_id", &self.0.attempt_id)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptReleaseCommit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptReleaseCommit").field("attempt_id", &self.0.attempt_id).finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptAdmissionBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttemptAdmissionBinding")
            .field("attempt_id", &self.release_commit.0.attempt_id)
            .finish_non_exhaustive()
    }
}

impl AttemptAdmissionBinding {
    /// Returns recording-only access to the attempt observation.
    ///
    /// The returned handle cannot seal the observation.  Admission must take
    /// this binding as a whole to receive the paired cleanup commitment.
    pub fn observation_handle(&self) -> RuntimeObservationHandle {
        self.observation_owner.handle()
    }

    /// Registers a read-only notification for the immutable attempt seal.
    ///
    /// This preserves the owner's delivery ordering without exposing any
    /// terminal-seal authority to callers.
    pub fn register_seal_observer<F>(&self, observer: F)
    where F: Fn(SealedRuntimeObservationSnapshot) + Send + Sync + 'static {
        self.observation_owner.register_seal_observer(observer);
    }

    pub(crate) fn into_parts(self) -> (RuntimeObservationOwner, AttemptReleaseCommit) {
        (self.observation_owner, self.release_commit)
    }
}

impl AttemptAdmissionRejection {
    /// Records the structured outcome of a rejected or cancelled admission
    /// request. The lifecycle accepts it only after admission was requested
    /// and before a grant or physical execution boundary exists.
    pub fn reject(&self, error: &QueryError) -> Result<(), AttemptLifecycleError> {
        let outcome = if error.fact_value("reason") == Some("backend_closing") {
            RuntimeTerminalOutcome::Failed {
                error_code: "backend_closing".to_string(),
                error_status: "permanent".to_string(),
            }
        } else if self.0.cancellation.is_cancelled() || error.code() == crate::error::QueryErrorCode::Cancelled {
            RuntimeTerminalOutcome::Cancelled
        } else {
            RuntimeTerminalOutcome::Failed {
                error_code: error.code().as_str().to_string(),
                error_status: error.status().as_str().to_string(),
            }
        };
        self.0.lock_lifecycle().terminalize_admission_rejection(outcome, self.0.monotonic_ns())
    }
}

impl AttemptReleaseCommit {
    /// Establishes the lifecycle terminal outcome before committing admitted
    /// cleanup and capacity facts. The resolved outcome must then be passed
    /// to the observation seal so footer, report, and terminal boundary agree.
    pub(crate) fn terminalize_for_release(
        &self,
        fallback: RuntimeTerminalOutcome,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        self.0.terminalize_for_release(fallback)
    }

    /// Applies the sole release result before report seal. This remains
    /// crate-private so a request caller cannot manufacture cleanup or
    /// capacity facts for an unrelated attempt.
    pub(crate) fn commit(&self, commit: AttemptTerminalCommit) -> Result<(), AttemptLifecycleError> {
        let mut lifecycle = self.0.lock_lifecycle();
        if lifecycle.terminal_commit == Some(commit) {
            return Ok(());
        }
        lifecycle.terminal_commit(commit, self.0.monotonic_ns())?;
        Ok(())
    }
}

impl AttemptHandle {
    /// The engine-minted identity of this invocation.
    pub fn attempt_id(&self) -> AttemptId {
        self.0.attempt_id
    }

    /// The attempt-scoped cooperative cancellation signal and deadline.
    pub fn cancellation(&self) -> AttemptCancellation {
        self.0.cancellation.clone()
    }

    /// Returns whether the attempt outcome is already terminal.
    pub fn is_terminal(&self) -> bool {
        self.0.lock_lifecycle().terminal.is_some()
    }

    /// The current disposition, when planning or execution has classified it.
    pub fn disposition(&self) -> Option<AttemptDisposition> {
        self.0.lock_lifecycle().disposition
    }

    pub fn record_planning_started(&self) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().planning_started(self.0.monotonic_ns())
    }

    pub fn record_planning_finished(&self) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().planning_finished(self.0.monotonic_ns())
    }

    pub fn set_disposition(&self, disposition: AttemptDisposition) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().disposition(disposition)
    }

    pub fn record_admission_requested(&self) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().admission_requested(self.0.monotonic_ns())
    }

    pub fn record_admitted(&self, fact: AttemptAdmissionFact) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().admitted(fact, self.0.monotonic_ns())
    }

    pub fn record_physical_execution_started(&self) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().physical(self.0.monotonic_ns())
    }

    pub fn record_gpu_execution_started(&self) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().gpu(self.0.monotonic_ns())
    }

    /// Counts one secondary cleanup failure exactly and records a bounded
    /// timeline fact. The count never replaces the primary outcome.
    pub fn record_cleanup_failure(&self, cleanup_error: &QueryError) {
        let primary_error_code = {
            let mut lifecycle = self.0.lock_lifecycle();
            lifecycle.cleanup_failure();
            match &lifecycle.terminal.as_ref().map(|(_, outcome)| outcome.clone()) {
                Some(RuntimeTerminalOutcome::Failed { error_code, .. }) => Some(error_code.clone()),
                _ => None,
            }
        };
        let Some(observation) = self.0.observation.get() else { return };
        observation.record(|| {
            RuntimeObservationPayload::CleanupFailure(Box::new(RuntimeCleanupFailureObservation {
                phase: "attempt_cleanup".to_string(),
                primary_error_code,
                error_code: cleanup_error.code().to_string(),
                error_kind: cleanup_error.kind().to_string(),
                error_status: cleanup_error.status().as_str().to_string(),
                error_operation: cleanup_error.operation().to_string(),
            }))
        });
    }
}

impl AttemptCompletion {
    /// An uncounted handle for adapter internals that mint counted receivers
    /// on demand. It awaits and reads the report exactly like a receiver but
    /// is invisible to the abandoned-report decision.
    pub fn uncounted(&self) -> Self {
        Self { cell: Arc::clone(&self.cell), counted: false }
    }

    /// Returns whether the terminal report has been published.
    pub fn is_ready(&self) -> bool {
        self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).report.is_some()
    }

    /// Returns the published report without waiting, when ready.
    pub fn try_report(&self) -> Option<Arc<QueryAttemptReport>> {
        self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).report.clone()
    }

    /// Blocks the current thread until the report is published or the timeout
    /// elapses. Intended for synchronous tests and diagnostics.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Arc<QueryAttemptReport>> {
        let state = self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (state, _) = self
            .cell
            .changed
            .wait_timeout_while(state, timeout, |state| state.report.is_none())
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.report.clone()
    }
}

impl Future for AttemptCompletion {
    type Output = Arc<QueryAttemptReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(report) = &state.report {
            return Poll::Ready(Arc::clone(report));
        }
        if !state.wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
            state.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

impl Drop for AttemptCompletion {
    fn drop(&mut self) {
        if self.counted {
            let mut state = self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            state.receivers = state.receivers.saturating_sub(1);
        }
    }
}

impl fmt::Debug for AttemptCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptCompletion").field("ready", &self.is_ready()).finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::{QueryError, QueryErrorCode},
        observability::{DeterministicRuntimeObservationClock, RuntimeObservationConfig},
        runtime::BackendLifecycle,
    };

    fn test_begin(
        observation: RuntimeObservationConfig,
        queue: Option<Arc<AttemptCompletionQueue>>,
    ) -> (AttemptBegin, Arc<DeterministicRuntimeObservationClock>, Arc<AttemptBackendCounters>) {
        let lifecycle = BackendLifecycle::new();
        let permit = lifecycle.try_begin_observation().expect("test permit");
        let counters = Arc::new(AttemptBackendCounters::default());
        let clock = Arc::new(DeterministicRuntimeObservationClock::new(Some(42)));
        let begin = AttemptOwner::begin_with_clock(
            BeginAttemptRequest { observation, ..BeginAttemptRequest::default() },
            permit,
            Arc::clone(&counters),
            queue,
            clock.clone(),
        );
        (begin, clock, counters)
    }

    fn native_disposition(begin: &AttemptBegin) {
        let handle = begin.owner.handle();
        handle.record_planning_started().unwrap();
        handle.record_planning_finished().unwrap();
        handle.set_disposition(AttemptDisposition::Native).unwrap();
    }

    #[test]
    fn non_admitted_commit_publishes_immediately_in_order() {
        let (begin, clock, counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.clone();
        native_disposition(&begin);
        clock.set_monotonic_ns(10);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("report published");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
        assert_eq!(report.disposition, AttemptDisposition::Native);
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::NotRequired);
        assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::NotAdmitted);
        assert_eq!(report.final_domain_usage, None, "unadmitted attempts have no final domain sample");
        assert_eq!(report.health_transition, None, "unadmitted attempts have no release health transition");
        assert_eq!(report.timestamps.cleanup_complete_ns, None);
        assert_eq!(report.timestamps.report_ready_ns, Some(10));
        assert_eq!(report.timestamps.outcome_terminal_ns, 10);
        assert_eq!(counters.snapshot().reports_ready, 1);
        // The sealed observation and completion share one immutable report.
        let snapshot = begin.owner.observation_owner_for_test().sealed_snapshot().expect("sealed");
        let embedded = snapshot.attempt_report().expect("attempt report embedded");
        assert_eq!(embedded, report.as_ref());
        assert!(Arc::ptr_eq(
            snapshot.data().attempt_report.as_ref().expect("attempt report allocation embedded"),
            &report
        ));
    }

    #[test]
    fn seal_uses_the_attempt_terminal_outcome_for_footer_report_and_boundary() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.clone();
        native_disposition(&begin);
        let observation_owner = begin.owner.observation_owner_for_test();
        observation_owner.record_terminal_outcome(RuntimeTerminalOutcome::Failed {
            error_code: "inner_failure".to_string(),
            error_status: "permanent".to_string(),
        });

        begin.owner.commit_terminal(RuntimeTerminalOutcome::Cancelled).unwrap();
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("report published");
        let snapshot = observation_owner.sealed_snapshot().expect("sealed");

        assert_eq!(report.outcome, RuntimeTerminalOutcome::Cancelled);
        assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Cancelled));
        assert_eq!(snapshot.attempt_report().expect("attempt report embedded"), report.as_ref());
        assert!(snapshot.events().iter().any(|event| {
            matches!(
                event.payload,
                RuntimeObservationPayload::QueryBoundary {
                    boundary: crate::observability::QueryBoundaryKind::Cancelled
                }
            )
        }));
    }

    #[test]
    fn admitted_commit_defers_until_post_cleanup_seal() {
        let (begin, clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.clone();
        native_disposition(&begin);
        let handle = begin.owner.handle();
        let (_, release_commit) = begin.owner.admission_binding().into_parts();
        handle.record_admission_requested().unwrap();
        handle
            .record_admitted(AttemptAdmissionFact {
                admission_sequence: 3,
                admission_wait_ns: 1,
                device_ordinal: 1,
                grant_bytes: 1024,
            })
            .unwrap();
        handle.record_physical_execution_started().unwrap();
        handle.record_gpu_execution_started().unwrap();
        clock.set_monotonic_ns(20);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        assert!(!completion.is_ready(), "admitted attempt must wait for the post-cleanup seal");
        // The engine seals the observation owner after GPU cleanup.
        clock.set_monotonic_ns(30);
        let final_domain_usage = AttemptFinalDomainUsage::from(cudf::QueryMemoryDomainUsage {
            domain_id: 101,
            active: false,
            active_members: 0,
            ceiling_bytes: 1024,
            effective_driver_ceiling_bytes: 2048,
            driver_ceiling_verified: true,
            allocation_limit_bytes: 768,
            current_bytes: 0,
            peak_bytes: 512,
            total_bytes: 1024,
            current_aligned_bytes: 0,
            peak_aligned_bytes: 640,
            total_allocations: 9,
            current_allocations: 0,
            peak_allocations: 4,
            failed_allocation_count: 2,
            largest_failed_allocation_bytes: 64,
        });
        release_commit
            .commit(AttemptTerminalCommit::completed(
                AttemptCapacityDisposition::Released,
                Some(final_domain_usage),
                AttemptHealthTransition::HealthyReuse,
            ))
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(RuntimeTerminalOutcome::Succeeded);
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("report published after seal");
        assert_eq!(report.timestamps.outcome_terminal_ns, 20);
        assert_eq!(report.timestamps.cleanup_complete_ns, Some(30));
        assert_eq!(report.timestamps.report_ready_ns, Some(30));
        assert_eq!(report.device_ordinal, Some(1));
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed);
        assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::Released);
        assert_eq!(report.final_domain_usage, Some(final_domain_usage), "report preserves every final sample scalar");
        assert_eq!(report.health_transition, Some(AttemptHealthTransition::HealthyReuse));
        assert_eq!(report.admission.as_ref().map(|fact| fact.grant_bytes), Some(1024));
        let snapshot = begin.owner.observation_owner_for_test().sealed_snapshot().expect("sealed");
        assert_eq!(snapshot.attempt_report().expect("attempt report embedded"), report.as_ref());
    }

    #[test]
    fn terminal_commit_projects_actual_cleanup_and_capacity_before_seal() {
        let (begin, clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.clone();
        native_disposition(&begin);
        let handle = begin.owner.handle();
        let (_, release_commit) = begin.owner.admission_binding().into_parts();
        handle.record_admission_requested().unwrap();
        handle
            .record_admitted(AttemptAdmissionFact {
                admission_sequence: 3,
                admission_wait_ns: 1,
                device_ordinal: 1,
                grant_bytes: 1024,
            })
            .unwrap();
        clock.set_monotonic_ns(20);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        clock.set_monotonic_ns(30);
        release_commit
            .commit(AttemptTerminalCommit::failed_cleanup(
                AttemptCapacityDisposition::RetainedQuarantine,
                None,
                AttemptHealthTransition::BackendUnhealthy,
            ))
            .unwrap();
        release_commit
            .commit(AttemptTerminalCommit::failed_cleanup(
                AttemptCapacityDisposition::RetainedQuarantine,
                None,
                AttemptHealthTransition::BackendUnhealthy,
            ))
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(RuntimeTerminalOutcome::Succeeded);
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("report");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::CompletedWithFailures);
        assert_eq!(report.cleanup_failure_count, 1);
        assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::RetainedQuarantine);
        assert_eq!(report.final_domain_usage, None, "unavailable sample is never replaced with all-zero usage");
        assert_eq!(report.health_transition, Some(AttemptHealthTransition::BackendUnhealthy));
        assert_eq!(report.timestamps.cleanup_complete_ns, Some(30));
        assert_eq!(report.timestamps.report_ready_ns, Some(30));
        let snapshot = begin.owner.observation_owner_for_test().sealed_snapshot().expect("sealed");
        assert_eq!(snapshot.attempt_report().expect("attempt report embedded"), report.as_ref());
    }

    #[test]
    fn double_terminalization_is_exactly_once() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Dropped).unwrap();
        assert_eq!(
            begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded),
            Err(AttemptLifecycleError::DoubleTerminalization)
        );
        let report = begin.completion.wait_timeout(Duration::from_secs(1)).expect("report");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
    }

    #[test]
    fn owner_drop_backstop_never_loses_the_report() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.clone();
        drop(begin.owner);
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("drop publishes the report");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
    }

    #[test]
    fn fixed_summary_survives_disabled_and_zero_limit_diagnostics() {
        for observation in [RuntimeObservationConfig::disabled(), RuntimeObservationConfig::bounded(0, 0)] {
            let (begin, _clock, _counters) = test_begin(observation, None);
            let completion = begin.completion.clone();
            native_disposition(&begin);
            begin
                .owner
                .commit_terminal(RuntimeTerminalOutcome::Failed {
                    error_code: "internal".to_string(),
                    error_status: "permanent".to_string(),
                })
                .unwrap();
            let report = completion.wait_timeout(Duration::from_secs(1)).expect("report");
            assert_eq!(report.disposition, AttemptDisposition::Native);
            assert!(matches!(report.outcome, RuntimeTerminalOutcome::Failed { .. }));
            assert!(report.timestamps.report_ready_ns.is_some());
            assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::NotRequired);
        }
    }

    #[test]
    fn receiver_abandonment_still_publishes_and_overflow_is_counted() {
        let queue = Arc::new(AttemptCompletionQueue::new(1).expect("queue"));
        let (begin, _clock, counters) =
            test_begin(RuntimeObservationConfig::native_execution(), Some(Arc::clone(&queue)));
        // Abandon the only receiver before the report exists.
        drop(begin.completion);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        assert_eq!(counters.snapshot().reports_ready, 1);
        assert_eq!(counters.snapshot().completion_queue_pushed, 1);
        assert_eq!(counters.snapshot().completion_queue_overflowed, 0);
        let drained = queue.drain(8);
        assert_eq!(drained.len(), 1);
        assert!(matches!(drained[0].outcome, RuntimeTerminalOutcome::Succeeded));

        // Second abandoned report overflows the capacity-one queue.
        let (begin2, _clock2, counters2) =
            test_begin(RuntimeObservationConfig::native_execution(), Some(Arc::clone(&queue)));
        drop(begin2.completion);
        begin2.owner.commit_terminal(RuntimeTerminalOutcome::Dropped).unwrap();
        // Queue is empty after the drain, so this push succeeds.
        assert_eq!(counters2.snapshot().completion_queue_pushed, 1);
        let (begin3, _clock3, counters3) =
            test_begin(RuntimeObservationConfig::native_execution(), Some(Arc::clone(&queue)));
        drop(begin3.completion);
        begin3.owner.commit_terminal(RuntimeTerminalOutcome::Dropped).unwrap();
        assert_eq!(counters3.snapshot().completion_queue_overflowed, 1);
        // The capacity-one queue still holds the second report; the third was
        // counted as delivery loss without blocking the commit.
        assert_eq!(queue.drain(8).len(), 1);
    }

    #[test]
    fn held_receivers_do_not_route_to_the_abandoned_queue() {
        let queue = Arc::new(AttemptCompletionQueue::new(4).expect("queue"));
        let (begin, _clock, counters) = test_begin(RuntimeObservationConfig::native_execution(), Some(queue.clone()));
        let _held = begin.completion.clone();
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        assert_eq!(counters.snapshot().completion_queue_pushed, 0);
        assert!(queue.drain(4).is_empty());
        assert!(_held.is_ready());
    }

    #[test]
    fn cleanup_failures_are_counted_exactly_without_replacing_the_primary_outcome() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        native_disposition(&begin);
        let handle = begin.owner.handle();
        handle.record_admission_requested().unwrap();
        handle
            .record_admitted(AttemptAdmissionFact {
                admission_sequence: 1,
                admission_wait_ns: 0,
                device_ordinal: 0,
                grant_bytes: 1,
            })
            .unwrap();
        begin
            .owner
            .commit_terminal(RuntimeTerminalOutcome::Failed {
                error_code: "cudf_temporary_failure".to_string(),
                error_status: "temporary".to_string(),
            })
            .unwrap();
        let cleanup_error = QueryError::new(QueryErrorCode::Internal, "test.cleanup", "injected cleanup failure");
        handle.record_cleanup_failure(&cleanup_error);
        handle.record_cleanup_failure(&cleanup_error);
        begin.owner.observation_owner_for_test().finish_query(RuntimeTerminalOutcome::Failed {
            error_code: "cudf_temporary_failure".to_string(),
            error_status: "temporary".to_string(),
        });
        let report = begin.completion.wait_timeout(Duration::from_secs(1)).expect("report");
        assert_eq!(report.cleanup_failure_count, 2);
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::CompletedWithFailures);
        assert!(matches!(
            report.outcome,
            RuntimeTerminalOutcome::Failed { ref error_code, .. } if error_code == "cudf_temporary_failure"
        ));
        // The bounded timeline carries the same facts when capture allows.
        let snapshot = begin.owner.observation_owner_for_test().sealed_snapshot().expect("sealed");
        let cleanup_events = snapshot
            .events()
            .iter()
            .filter(|event| matches!(event.payload, RuntimeObservationPayload::CleanupFailure(_)))
            .count();
        assert_eq!(cleanup_events, 2);
    }

    #[test]
    fn cancellation_projects_into_the_engine_execution_signal() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let cancellation = begin.owner.cancellation();
        assert!(!cancellation.is_cancelled());
        cancellation.cancel();
        assert!(cancellation.is_cancelled());
        assert!(cancellation.execution_cancellation().is_cancelled());
        assert!(cancellation.check("test.operation").is_err());
    }
}

#[cfg(test)]
mod terminalize_race_tests {
    use super::*;
    use crate::{observability::DeterministicRuntimeObservationClock, runtime::BackendLifecycle};

    #[test]
    fn admission_rejection_terminalized_at_the_binding_wins_over_the_wrapper_commit() {
        let lifecycle = BackendLifecycle::new();
        let permit = lifecycle.try_begin_observation().expect("test permit");
        let counters = Arc::new(AttemptBackendCounters::default());
        let clock = Arc::new(DeterministicRuntimeObservationClock::new(Some(1)));
        let begin = AttemptOwner::begin_with_clock(BeginAttemptRequest::default(), permit, counters, None, clock);
        let handle = begin.owner.handle();
        let admission_rejection = begin.owner.admission_rejection();
        handle.record_planning_started().unwrap();
        handle.record_planning_finished().unwrap();
        handle.set_disposition(AttemptDisposition::Native).unwrap();
        handle.record_admission_requested().unwrap();
        // The admission binding terminalizes the rejection at its source.
        admission_rejection
            .reject(&QueryError::new(
                crate::error::QueryErrorCode::ServiceOverloaded,
                "test.admission_rejection",
                "injected admission rejection",
            ))
            .unwrap();
        // The wrapper's later commit loses and observes the structured error.
        assert_eq!(
            begin.owner.commit_terminal(RuntimeTerminalOutcome::Failed {
                error_code: "datafusion_error".to_string(),
                error_status: "permanent".to_string(),
            }),
            Err(AttemptLifecycleError::DoubleTerminalization)
        );
        // The engine post-rejection seal publishes the truthful outcome.
        begin.owner.observation_owner_for_test().finish_query(RuntimeTerminalOutcome::Failed {
            error_code: "service_overloaded".to_string(),
            error_status: "temporary".to_string(),
        });
        let report = begin.completion.wait_timeout(Duration::from_secs(1)).expect("report");
        assert!(matches!(
            report.outcome,
            RuntimeTerminalOutcome::Failed { ref error_code, .. } if error_code == "service_overloaded"
        ));
        assert_eq!(report.disposition, AttemptDisposition::Native);
        assert_eq!(report.timestamps.admission_requested_ns, Some(0));
        assert_eq!(report.timestamps.admitted_ns, None);
        assert_eq!(report.timestamps.physical_execution_started_ns, None);
    }
}

#[cfg(test)]
mod abandonment_race_tests {
    use super::*;
    use crate::{
        observability::{DeterministicRuntimeObservationClock, RuntimeObservationConfig},
        runtime::BackendLifecycle,
    };

    fn begin(queue: Option<Arc<AttemptCompletionQueue>>) -> AttemptBegin {
        let lifecycle = BackendLifecycle::new();
        let permit = lifecycle.try_begin_observation().expect("test permit");
        AttemptOwner::begin_with_clock(
            BeginAttemptRequest::default(),
            permit,
            Arc::new(AttemptBackendCounters::default()),
            queue,
            Arc::new(DeterministicRuntimeObservationClock::new(Some(1))),
        )
    }

    #[test]
    fn last_receiver_drop_and_publish_have_one_coherent_linearization() {
        for _ in 0..64 {
            let queue = Arc::new(AttemptCompletionQueue::new(4).expect("queue"));
            let begin = begin(Some(Arc::clone(&queue)));
            let completion = begin.completion;
            let (dropped_tx, dropped_rx) = std::sync::mpsc::channel::<()>();
            let publisher = std::thread::spawn(move || {
                dropped_rx.recv().expect("drop signal");
                begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
                begin.owner
            });
            drop(completion);
            dropped_tx.send(()).expect("publish signal");
            let owner = publisher.join().expect("publisher joins");
            // The drop strictly precedes the publish in this interleaving, so
            // the report must be routed to the abandoned queue.
            assert_eq!(queue.drain(4).len(), 1);
            drop(owner);
        }
    }

    #[test]
    fn clone_during_publication_is_delivery_loss_never_report_loss() {
        for _ in 0..64 {
            let queue = Arc::new(AttemptCompletionQueue::new(4).expect("queue"));
            let begin = begin(Some(Arc::clone(&queue)));
            let factory = begin.completion.uncounted();
            drop(begin.completion);
            let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
            let cloner_factory = factory.uncounted();
            let cloner = std::thread::spawn(move || {
                go_rx.recv().expect("go");
                cloner_factory.clone()
            });
            go_tx.send(()).expect("go");
            begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
            let receiver = cloner.join().expect("cloner joins");
            // Whatever the interleaving, the cloned receiver can always read
            // the published report; the queue decision was made under one lock.
            assert!(matches!(
                receiver.wait_timeout(Duration::from_secs(1)).map(|report| report.outcome.clone()),
                Some(RuntimeTerminalOutcome::Succeeded)
            ));
            drop(receiver);
        }
    }

    #[test]
    fn deadline_expiry_is_a_distinct_typed_cancellation() {
        let lifecycle = BackendLifecycle::new();
        let permit = lifecycle.try_begin_observation().expect("test permit");
        let begin = AttemptOwner::begin_with_clock(
            BeginAttemptRequest {
                deadline: Some(Instant::now() - Duration::from_millis(1)),
                observation: RuntimeObservationConfig::native_execution(),
                ..BeginAttemptRequest::default()
            },
            permit,
            Arc::new(AttemptBackendCounters::default()),
            None,
            Arc::new(DeterministicRuntimeObservationClock::new(Some(1))),
        );
        let cancellation = begin.owner.cancellation();
        assert!(cancellation.is_expired());
        let error = cancellation.check("test.deadline").unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
        assert_eq!(error.fact_value("cancellation_requested"), Some("true"));
        assert!(!cancellation.is_cancelled(), "expiry is not a caller cancel signal");
    }
}
