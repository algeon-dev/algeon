//! Attempt ownership, boundary recording, and independent completion delivery.
use std::{
    fmt,
    future::Future,
    mem, panic,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync,
    sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, atomic::Ordering},
    task::{Context, Poll, Waker},
    thread,
    time::{Duration, Instant},
};

use query_planning::error::QueryError;

use super::{
    AttemptBackendCounters, AttemptCancellation, AttemptCompletionQueue, AttemptLifecycle, AttemptLifecycleError,
};
#[cfg(test)]
use crate::observability::{
    AttemptDeviceHealth, AttemptDeviceHealthStatus, AttemptFailureCode, AttemptFailureFactKey, AttemptFailureFactValue,
    AttemptFailureOrigin, AttemptFailureStatus, AttemptFaultIsolation, AttemptFinalDomainUsage,
    AttemptHealthTransition, AttemptNativeFault, AttemptNativeFaultEvidence, AttemptNativeFaultKind,
    AttemptNativeStatus, AttemptTeardownHealth, QueryBoundaryKind,
};
#[cfg(test)]
use crate::observability::{
    CleanupFailurePhase, CleanupFailureSummary, RuntimeCleanupFailureObservation, RuntimeObservationPayload,
};
use crate::{
    observability::{
        AttemptAdmissionFact, AttemptAdmissionWaitFact, AttemptAlternateDeviceEligibility, AttemptCapacityDisposition,
        AttemptCleanupDisposition, AttemptComposedScheduleProjection, AttemptDisposition, AttemptFailureOperation,
        AttemptFailureSummary, AttemptId, AttemptQueryId, AttemptRetryAdvice, AttemptTerminalCommit,
        AttemptTraceContext, QUERY_ATTEMPT_REPORT_SCHEMA_V2, QueryAttemptReport, RuntimeObservationClock,
        RuntimeObservationConfig, RuntimeObservationFooter, RuntimeObservationHandle, RuntimeObservationOwner,
        RuntimeQueryId, RuntimeTerminalOutcome, RuntimeTraceId, SealedRuntimeObservationSnapshot,
        SystemRuntimeObservationClock,
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

/// The attempt owner and its completion factory, created together at begin.
pub struct AttemptBegin {
    pub owner: AttemptOwner,
    pub completion: AttemptCompletionFactory,
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
/// a structured pre-admission rejection or cancellation, and it cannot
/// terminalize an admitted attempt or record arbitrary lifecycle boundaries.
pub struct AttemptAdmissionRejection {
    core: Arc<AttemptCore>,
    observation_owner: RuntimeObservationOwner,
    finalizer_slot: AttemptFinalizerSlot,
}

/// Opaque, identity-bound admission attachment for one execution attempt.
///
/// [`AttemptOwner`] mints this as one unit so an admission request can never
/// combine one attempt's observation owner with another attempt's cleanup
/// capability. It exposes neither terminalization nor observation-seal
/// authority to the request caller.
#[derive(Clone)]
pub struct AttemptAdmissionBinding {
    attempt_id: AttemptId,
    observation_owner: RuntimeObservationOwner,
    finalizer_slot: AttemptFinalizerSlot,
}

/// The one non-cloneable post-cleanup authority minted for an attempt.
///
/// It may record the canonical terminal outcome before cleanup, but consuming
/// it is the only way to establish finalization and therefore report-ready for
/// an admitted attempt.
pub(crate) struct AttemptFinalizer(Arc<AttemptCore>);

/// The attempt's one admission right, taken by
/// [`AttemptAdmissionBinding::claim`] before the admission request touches a
/// queue, ledger, stream, or the attempt lifecycle.
///
/// It owns the attempt's finalizer, so while it exists the attempt owner only
/// records a terminal outcome; the admission finishes the attempt, either as
/// not admitted or after GPU cleanup.
pub(crate) struct AttemptAdmission(AttemptFinalizer);

/// Shared routing for one [`AttemptFinalizer`]. Cloning an admission binding
/// clones this routing handle, never the finalizer itself. Before admission,
/// the owner-side path consumes it with `NotRequired`; after a grant is
/// published, the admission envelope takes it exactly once.
#[derive(Clone)]
pub(crate) struct AttemptFinalizerSlot(Arc<Mutex<Option<AttemptFinalizer>>>);

impl Clone for AttemptHandle {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl Clone for AttemptAdmissionRejection {
    fn clone(&self) -> Self {
        Self {
            core: Arc::clone(&self.core),
            observation_owner: self.observation_owner.clone(),
            finalizer_slot: self.finalizer_slot.clone(),
        }
    }
}

/// Independent receiver for the one bounded terminal attempt report.
///
/// Cloning or dropping receivers never cancels cleanup, report production, or
/// exact metric updates, and holding a receiver retains no device lease.
///
/// Every receiver is counted so publication can route never-received reports
/// to the optional abandoned-report queue. Adapter internals hold an
/// [`AttemptCompletionFactory`] instead, so an attempt object or result
/// stream never masks caller abandonment.
pub struct AttemptCompletion {
    cell: Arc<AttemptCompletionCell>,
}

impl Clone for AttemptCompletion {
    fn clone(&self) -> Self {
        let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.receivers = state.receivers.saturating_add(1);
        Self { cell: Arc::clone(&self.cell) }
    }
}

/// Adapter-internal minting capability for one attempt's completion receivers.
///
/// The factory crosses crates so the adapter attempt object can hand out
/// independent counted receivers on demand and read the sealed report for its
/// seal observer. It is never an application receiver itself: cloning or
/// dropping a factory never changes the active receiver count, so it cannot
/// mask caller abandonment. It cannot terminalize the attempt or store a
/// second report.
pub struct AttemptCompletionFactory {
    cell: Arc<AttemptCompletionCell>,
}

impl Clone for AttemptCompletionFactory {
    fn clone(&self) -> Self {
        Self { cell: Arc::clone(&self.cell) }
    }
}

impl AttemptCompletionFactory {
    /// Mints one independent counted receiver for the attempt's immutable
    /// terminal report.
    pub fn completion(&self) -> AttemptCompletion {
        let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.receivers = state.receivers.saturating_add(1);
        AttemptCompletion { cell: Arc::clone(&self.cell) }
    }

    /// Returns the published report without waiting, when ready.
    pub fn try_report(&self) -> Option<Arc<QueryAttemptReport>> {
        let state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.report_visible.then(|| state.report.as_ref().map(Arc::clone)).flatten()
    }
}

/// A read-only callback invoked while the one immutable report is being
/// published. It intentionally has no terminalization, retry, or storage
/// authority; consumers may only project the report they receive.
struct SealedAttemptReportSink {
    notify: Box<dyn Fn(Arc<QueryAttemptReport>) + Send + Sync>,
}

impl SealedAttemptReportSink {
    fn invoke(&self, report: Arc<QueryAttemptReport>) {
        // A projection must never prevent the owning attempt from publishing
        // its already-finalized report.
        let _ = panic::catch_unwind(AssertUnwindSafe(|| (self.notify)(report)));
    }
}

struct CompletionState {
    // The canonical report allocation. It remains unreadable until every
    // pre-registered projection consumed it, so there is no second sink-side
    // report store and no completion observer can race ahead of a projection.
    report: Option<Arc<QueryAttemptReport>>,
    report_visible: bool,
    sealed_report_sinks: Vec<SealedAttemptReportSink>,
    wakers: Vec<Waker>,
    // Receiver counting shares the publication mutex so clone/drop linearize
    // against the abandoned-at-publish decision.
    receivers: u64,
}

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
    observation: OnceLock<RuntimeObservationHandle>,
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
    finalizer_slot: AttemptFinalizerSlot,
}

impl fmt::Debug for AttemptCore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptCore").field("attempt_id", &self.attempt_id).finish_non_exhaustive()
    }
}

impl AttemptCore {
    fn lock_lifecycle(&self) -> MutexGuard<'_, AttemptLifecycle> {
        self.lifecycle.lock().unwrap_or_else(sync::PoisonError::into_inner)
    }

    fn monotonic_ns(&self) -> u64 {
        self.clock.monotonic_ns()
    }

    fn register_sealed_report_sink<F>(&self, sink: F)
    where F: Fn(Arc<QueryAttemptReport>) + Send + Sync + 'static {
        let mut sink = Some(SealedAttemptReportSink { notify: Box::new(sink) });
        let published = {
            let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
            match &state.report {
                Some(report) if state.report_visible => Some(Arc::clone(report)),
                Some(_) => {
                    // Publication is draining the pre-visibility queue. Join
                    // it rather than invoke independently, so a concurrent
                    // late registration cannot race completion visibility.
                    state.sealed_report_sinks.push(sink.take().expect("sink exists"));
                    None
                },
                None => {
                    state.sealed_report_sinks.push(sink.take().expect("sink exists"));
                    None
                },
            }
        };
        if let Some(report) = published {
            sink.expect("unregistered sink exists").invoke(report);
        }
    }

    /// Builds the fixed report from lifecycle state. The sole seal commit
    /// applies observation completeness before this value is made visible to
    /// either the sealed snapshot or completion receivers.
    fn report(&self) -> QueryAttemptReport {
        let lifecycle = self.lock_lifecycle();
        let terminal_commit = lifecycle.terminal_commit;
        let cleanup_failure_count = lifecycle.cleanup_failures;
        let outcome =
            lifecycle.terminal.as_ref().map(|(_, outcome)| outcome.clone()).unwrap_or(RuntimeTerminalOutcome::Dropped);
        let primary_failure = lifecycle.primary_failure.clone();
        let capacity_disposition = terminal_commit
            .map(AttemptTerminalCommit::capacity_disposition)
            .unwrap_or(AttemptCapacityDisposition::NotAdmitted);
        let health_transition = terminal_commit.map(AttemptTerminalCommit::health_transition);
        let native_fault_evidence = terminal_commit.and_then(AttemptTerminalCommit::native_fault_evidence);
        let resource_evidence = terminal_commit.and_then(AttemptTerminalCommit::resource_evidence);
        let alternate_device_eligibility = terminal_commit
            .map(AttemptTerminalCommit::alternate_device_eligibility)
            .unwrap_or(AttemptAlternateDeviceEligibility::Unavailable);
        let retry_advice = AttemptRetryAdvice::from_sealed_attempt(
            &outcome,
            primary_failure.as_ref(),
            capacity_disposition,
            health_transition,
            native_fault_evidence,
            resource_evidence,
            alternate_device_eligibility,
        );
        QueryAttemptReport {
            schema: QUERY_ATTEMPT_REPORT_SCHEMA_V2.to_string(),
            attempt_id: self.attempt_id,
            query_id: self.query_id.clone(),
            trace_context: self.trace_context.clone(),
            disposition: lifecycle.disposition.unwrap_or(AttemptDisposition::Unplanned),
            outcome,
            primary_failure,
            retry_advice,
            device_ordinal: lifecycle.admitted.as_ref().map(|(_, fact)| fact.device_ordinal),
            admission: lifecycle.admitted.as_ref().map(|(_, fact)| fact.clone()),
            admission_wait: lifecycle.admission_wait,
            composed_schedule: lifecycle.composed_schedule.clone(),
            timestamps: lifecycle.timestamps(),
            cleanup_disposition: if lifecycle.admitted.is_some() {
                if cleanup_failure_count == 0 {
                    AttemptCleanupDisposition::Completed
                } else {
                    AttemptCleanupDisposition::CompletedWithFailures
                }
            } else {
                AttemptCleanupDisposition::NotRequired
            },
            cleanup_failure_count,
            cleanup_failure_count_complete: lifecycle.cleanup_failures_complete,
            first_cleanup_failure: lifecycle.first_cleanup_failure.clone(),
            // `finish_with_footer` requires finalization before publication.
            // An admitted report therefore always has a terminal commit;
            // `NotAdmitted` is the only valid no-commit projection.
            capacity_disposition,
            final_domain_usage: terminal_commit.and_then(AttemptTerminalCommit::final_domain_usage),
            health_transition,
            native_fault_evidence,
            resource_evidence,
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
        if lifecycle.terminal.is_some() {
            return Err(AttemptLifecycleError::DoubleTerminalization);
        }
        lifecycle.terminalize(fallback, self.monotonic_ns())?;
        Ok(lifecycle.terminal.as_ref().expect("terminal outcome was recorded").1.clone())
    }

    fn terminalize_for_release_with_primary_failure(
        &self,
        outcome: RuntimeTerminalOutcome,
        summary: AttemptFailureSummary,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        let mut lifecycle = self.lock_lifecycle();
        if lifecycle.terminal.is_some() {
            return Err(AttemptLifecycleError::DoubleTerminalization);
        }
        lifecycle.primary_failure(summary);
        lifecycle.terminalize(outcome, self.monotonic_ns())?;
        Ok(lifecycle.terminal.as_ref().expect("terminal outcome was recorded").1.clone())
    }

    /// Resolves the sole terminal outcome at seal. Recorder-local pending
    /// outcomes are only an early observation; the attempt lifecycle remains
    /// canonical once it has a terminal state.
    fn canonical_terminal_outcome(&self, fallback: RuntimeTerminalOutcome) -> RuntimeTerminalOutcome {
        let mut lifecycle = self.lock_lifecycle();
        if let Some((_, outcome)) = &lifecycle.terminal {
            return outcome.clone();
        }
        // A caller can capture a borrowed native failure before it hands the
        // error back to a transport wrapper.  In particular, an admitted
        // handle may start cleanup before that outer wrapper receives the
        // error.  The stored summary is therefore the only truthful terminal
        // classification available at this boundary; do not replace it with
        // a mechanical drop/cancellation fallback.
        let fallback =
            lifecycle.primary_failure.as_ref().map(RuntimeTerminalOutcome::from_primary_failure).unwrap_or(fallback);
        let at = self.monotonic_ns();
        lifecycle.terminalize(fallback.clone(), at).unwrap_or(());
        lifecycle.terminal.as_ref().map(|(_, outcome)| outcome.clone()).unwrap_or(fallback)
    }

    /// The single seal-and-completion commit: a previously consumed finalizer,
    /// exact aggregates, report-ready, and publication of one immutable report.
    /// The returned `Arc` is embedded directly in the sealed snapshot.
    fn finish_with_footer(
        &self,
        footer: &RuntimeObservationFooter,
    ) -> Result<Arc<QueryAttemptReport>, AttemptLifecycleError> {
        {
            let mut lifecycle = self.lock_lifecycle();
            if lifecycle.ready.is_none() {
                lifecycle.ready(self.monotonic_ns())?;
            }
        }
        let report = {
            let mut report = self.report();
            report.set_observation_completeness(footer);
            Arc::new(report)
        };
        // The canonical completion cell first owns the report without making
        // it observable. Sinks consume that exact Arc before the visibility
        // flag wakes any completion receiver; late registrations clone this
        // canonical allocation directly rather than retaining another copy.
        let report = loop {
            let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
            if let Some(existing) = &state.report {
                if state.report_visible {
                    return Ok(Arc::clone(existing));
                }
                let _guard = self.cell.changed.wait(state).unwrap_or_else(sync::PoisonError::into_inner);
                continue;
            }
            state.report = Some(Arc::clone(&report));
            break report;
        };

        // Keep draining under a visibility fence. A sink may register another
        // sink, and another thread may register while this publication is in
        // flight; every such pre-visibility sink receives this exact Arc
        // before any completion reader can observe it.
        let (wakers, abandoned) = loop {
            let sinks = {
                let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
                mem::take(&mut state.sealed_report_sinks)
            };
            for sink in sinks {
                sink.invoke(Arc::clone(&report));
            }
            let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
            if !state.sealed_report_sinks.is_empty() {
                continue;
            }
            debug_assert!(state.report.as_ref().is_some_and(|existing| Arc::ptr_eq(existing, &report)));
            state.report_visible = true;
            break (mem::take(&mut state.wakers), state.receivers == 0);
        };
        self.counters.reports_ready.fetch_add(1, Ordering::AcqRel);
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
        self.observation_permit.lock().unwrap_or_else(sync::PoisonError::into_inner).take();
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
            attempt_report: None,
            counter_overflow: false,
        })
    }
}

impl AttemptOwner {
    #[doc(hidden)]
    pub fn begin(
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
            state: Mutex::new(CompletionState {
                report: None,
                report_visible: false,
                sealed_report_sinks: Vec::new(),
                wakers: Vec::new(),
                receivers: 0,
            }),
            changed: Condvar::new(),
        });
        let core = Arc::new(AttemptCore {
            attempt_id: AttemptId::mint(),
            query_id: request.query_id,
            trace_context: request.trace_context,
            lifecycle: Mutex::new(AttemptLifecycle::new(clock.monotonic_ns())),
            clock: Arc::clone(&clock),
            cancellation: AttemptCancellation::new(request.deadline),
            observation: sync::OnceLock::new(),
            cell: Arc::clone(&cell),
            counters,
            completion_queue: queue,
            observation_permit: Mutex::new(None),
        });
        let finalizer_slot = AttemptFinalizerSlot::new(AttemptFinalizer(Arc::clone(&core)));
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
        let cleanup_failure_core = Arc::downgrade(&core);
        let owner = RuntimeObservationOwner::with_clock(query_id, trace_id, observation, clock)
            .with_terminal_outcome_resolver(move |fallback| terminal_outcome_core.canonical_terminal_outcome(fallback))
            .with_attempt_report_seal_provider(move |footer| {
                report_core
                    .finish_with_footer(footer)
                    .expect("attempt lifecycle must complete while sealing its sole immutable report")
            })
            .with_attempt_cleanup_failure_recorder(move |summary| {
                if let Some(core) = cleanup_failure_core.upgrade() {
                    core.lock_lifecycle().cleanup_failure(summary);
                }
            });
        core.observation.set(owner.handle()).expect("attempt observation handle is set exactly once");
        core.observation_permit.lock().unwrap_or_else(sync::PoisonError::into_inner).replace(permit);
        AttemptBegin {
            owner: Self(Arc::new(AttemptInner { core, observation_owner: owner, finalizer_slot })),
            completion: AttemptCompletionFactory { cell },
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

    /// Register a read-only projection of the immutable terminal report.
    ///
    /// The sink receives the report before completion receivers are notified,
    /// and late registration receives the already-published report. It has no
    /// lifecycle, retry, or aggregation authority; backend metrics consumes
    /// the exact sealed allocation without introducing another store.
    pub fn register_sealed_report_sink<F>(&self, sink: F)
    where F: Fn(Arc<QueryAttemptReport>) + Send + Sync + 'static {
        self.0.core.register_sealed_report_sink(sink);
    }

    /// Opaque capability that can terminalize only an admission request that
    /// was rejected before an admission grant was published.
    pub fn admission_rejection(&self) -> AttemptAdmissionRejection {
        AttemptAdmissionRejection {
            core: Arc::clone(&self.0.core),
            observation_owner: self.0.observation_owner.clone(),
            finalizer_slot: self.0.finalizer_slot.clone(),
        }
    }

    /// Opaque, atomic attachment for the query-service admission request.
    ///
    /// Its observation owner and release capability originate from this same
    /// attempt and are consumed together by admission. Standalone observation
    /// users continue to attach only a [`RuntimeObservationOwner`].
    pub fn admission_binding(&self) -> AttemptAdmissionBinding {
        AttemptAdmissionBinding {
            attempt_id: self.0.core.attempt_id,
            observation_owner: self.0.observation_owner.clone(),
            finalizer_slot: self.0.finalizer_slot.clone(),
        }
    }

    /// An independent receiver for the terminal report.
    pub fn completion(&self) -> AttemptCompletion {
        self.0.core.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner).receivers += 1;
        AttemptCompletion { cell: Arc::clone(&self.0.core.cell) }
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
        let terminal = self.0.core.terminalize_for_release(outcome)?;
        self.commit_resolved_terminal(terminal)
    }

    /// Publishes an already-linearized terminal outcome without attempting a
    /// second terminal lifecycle write. Borrowed-error capture records the
    /// primary summary and terminal outcome together before reaching this
    /// common finalization path.
    fn commit_resolved_terminal(&self, terminal: RuntimeTerminalOutcome) -> Result<(), AttemptLifecycleError> {
        match self.0.finalizer_slot.finalize_not_admitted(terminal) {
            Ok(terminal) => {
                let _ = self.0.observation_owner.finish_query(Some(terminal));
            },
            // An admission claim holds the finalizer, or a not-admitted
            // commit already consumed it. Either way the terminal outcome is
            // recorded and its holder publishes the report.
            Err(AttemptLifecycleError::FinalizationUnavailable) => return Ok(()),
            Err(error) => return Err(error),
        }
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
        Ok(())
    }

    /// Captures the native error by reference before reducing it to the
    /// closed terminal outcome. The caller retains the untouched error and
    /// its complete source chain.
    pub fn commit_query_error(&self, error: &QueryError) -> Result<(), AttemptLifecycleError> {
        let operation = self.0.core.lock_lifecycle().failure_operation();
        let outcome = RuntimeTerminalOutcome::from_query_error(error);
        let terminal = match RuntimeTerminalOutcome::primary_failure_from_query_error(error, operation) {
            Some(summary) => self.0.core.terminalize_for_release_with_primary_failure(outcome, summary)?,
            None => self.0.core.terminalize_for_release(outcome)?,
        };
        self.commit_resolved_terminal(terminal)
    }

    /// Commits a closed adapter fallback without retaining adapter display or
    /// source-chain text in the engine-owned report.
    pub fn commit_failure_summary(&self, summary: AttemptFailureSummary) -> Result<(), AttemptLifecycleError> {
        let terminal = self
            .0
            .core
            .terminalize_for_release_with_primary_failure(RuntimeTerminalOutcome::failed(&summary), summary)?;
        self.commit_resolved_terminal(terminal)
    }
}

impl Drop for AttemptOwner {
    fn drop(&mut self) {
        let report_ready = self.0.core.lock_lifecycle().ready.is_some();
        if !report_ready {
            let _ = self.commit_terminal(if thread::panicking() {
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
            .field("attempt_id", &self.core.attempt_id)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptFinalizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptFinalizer").field("attempt_id", &self.0.attempt_id).finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptAdmission").field("attempt_id", &self.0.0.attempt_id).finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptFinalizerSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptFinalizerSlot").finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptAdmissionBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptAdmissionBinding").finish_non_exhaustive()
    }
}

impl AttemptAdmissionBinding {
    /// Engine-minted identity retained only for post-callback allocator-event
    /// correlation. Request callers cannot replace or synthesize it.
    pub(crate) const fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

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

    /// Takes the attempt's only admission right. It must be the first step of
    /// an admission request, before any queue, ledger, stream, or lifecycle
    /// change. A rejected claim changes nothing and returns
    /// [`AttemptLifecycleError::AdmissionAlreadyClaimed`] once the right is
    /// consumed.
    pub(crate) fn claim(&self) -> Result<AttemptAdmission, AttemptLifecycleError> {
        self.finalizer_slot.claim()
    }

    pub(crate) fn into_observation_owner(self) -> RuntimeObservationOwner {
        self.observation_owner
    }
}

impl AttemptAdmissionRejection {
    /// Records the structured outcome of a rejected or cancelled admission
    /// request. The lifecycle accepts it only after admission was requested
    /// and before a grant or physical execution boundary exists.
    pub fn reject(&self, error: &QueryError) -> Result<(), AttemptLifecycleError> {
        let cancellation_error = self.core.cancellation.is_cancelled().then(|| {
            self.core
                .cancellation
                .check("query_engine.attempt.admission_rejection")
                .expect_err("an effective cancellation must produce a terminal error")
        });
        let (outcome, summary) = if let Some(cancellation_error) = cancellation_error {
            (
                RuntimeTerminalOutcome::Cancelled,
                RuntimeTerminalOutcome::primary_failure_from_query_error(
                    &cancellation_error,
                    AttemptFailureOperation::Admission,
                ),
            )
        } else {
            let summary = if error.fact_value("reason") == Some("backend_closing") {
                Some(AttemptFailureSummary::backend_closing())
            } else {
                RuntimeTerminalOutcome::primary_failure_from_query_error(error, AttemptFailureOperation::Admission)
            };
            (RuntimeTerminalOutcome::from_query_error(error), summary)
        };
        {
            let mut lifecycle = self.core.lock_lifecycle();
            // A claimed admission, including one whose second claim was just
            // rejected, is finished by its claim; this rejection changes nothing.
            if lifecycle.admission_requested.is_some() {
                return Ok(());
            }
            lifecycle.terminalize_admission_rejection(outcome.clone(), summary, self.core.monotonic_ns())?;
        }
        let outcome = self.finalizer_slot.finalize_not_admitted(outcome)?;
        self.observation_owner.finish_query(Some(outcome));
        Ok(())
    }
}

impl AttemptFinalizerSlot {
    fn new(finalizer: AttemptFinalizer) -> Self {
        Self(Arc::new(Mutex::new(Some(finalizer))))
    }

    fn take(&self) -> Result<AttemptFinalizer, AttemptLifecycleError> {
        self.0
            .lock()
            .unwrap_or_else(sync::PoisonError::into_inner)
            .take()
            .ok_or(AttemptLifecycleError::FinalizationUnavailable)
    }

    /// Takes the finalizer as the attempt's admission right. The slot lock is
    /// held while the lifecycle validates the claim, so a rejected claim
    /// leaves both the slot and the lifecycle unchanged.
    fn claim(&self) -> Result<AttemptAdmission, AttemptLifecycleError> {
        let mut slot = self.0.lock().unwrap_or_else(sync::PoisonError::into_inner);
        let core = &slot.as_ref().ok_or(AttemptLifecycleError::AdmissionAlreadyClaimed)?.0;
        core.lock_lifecycle().admission_requested(core.monotonic_ns())?;
        Ok(AttemptAdmission(slot.take().expect("the claimed finalizer is present")))
    }

    pub(crate) fn finalize_not_admitted(
        &self,
        fallback: RuntimeTerminalOutcome,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        self.take()?.finalize_not_admitted(fallback)
    }
}

impl AttemptFinalizer {
    /// Samples the attempt-owned monotonic clock for a release-commit fact.
    ///
    /// The query service must not manufacture a timestamp when no admission
    /// binding transferred this linear finalizer into its ownership.
    pub(crate) fn monotonic_ns(&self) -> u64 {
        self.0.monotonic_ns()
    }

    pub(crate) fn record_primary_failure(&self, summary: AttemptFailureSummary) {
        self.0.lock_lifecycle().primary_failure(summary);
    }

    /// Establishes the lifecycle terminal outcome before committing admitted
    /// cleanup and capacity facts. The resolved outcome must then be passed
    /// to the observation seal so footer, report, and terminal boundary agree.
    pub(crate) fn terminalize_for_release(
        &self,
        fallback: RuntimeTerminalOutcome,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        // The result stream may have already committed the primary terminal
        // outcome before its owned admission handle begins GPU cleanup. The
        // linear finalizer must resolve that existing outcome, not contend
        // with the public terminal writer and manufacture a second commit.
        Ok(self.0.canonical_terminal_outcome(fallback))
    }

    fn finalize_not_admitted(
        self,
        fallback: RuntimeTerminalOutcome,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        let mut lifecycle = self.0.lock_lifecycle();
        let outcome = match &lifecycle.terminal {
            Some((_, outcome)) => outcome.clone(),
            None => {
                lifecycle.terminalize(fallback, self.0.monotonic_ns())?;
                lifecycle.terminal.as_ref().expect("terminal outcome was recorded").1.clone()
            },
        };
        lifecycle.finalize_not_admitted(self.0.monotonic_ns())?;
        Ok(outcome)
    }

    /// Consumes the sole post-cleanup finalizer. This remains crate-private so
    /// request callers cannot manufacture cleanup/capacity facts for another
    /// attempt or race the envelope that owns the immutable grant.
    pub(crate) fn finalize_admitted(self, commit: AttemptTerminalCommit) -> Result<(), AttemptLifecycleError> {
        let mut lifecycle = self.0.lock_lifecycle();
        lifecycle.terminal_commit(commit, self.0.monotonic_ns())?;
        Ok(())
    }
}

impl AttemptAdmission {
    pub(crate) fn monotonic_ns(&self) -> u64 {
        self.0.monotonic_ns()
    }

    pub(crate) fn record_primary_failure(&self, summary: AttemptFailureSummary) {
        self.0.record_primary_failure(summary);
    }

    /// Records the published grant and final queue state. The query service
    /// calls this under its admission mutex while publishing the envelope;
    /// it fails when the attempt owner committed a terminal outcome first.
    pub(crate) fn record_admitted(
        &self,
        fact: AttemptAdmissionFact,
        wait: Option<AttemptAdmissionWaitFact>,
    ) -> Result<(), AttemptLifecycleError> {
        let mut lifecycle = self.0.0.lock_lifecycle();
        lifecycle.admitted(fact, self.0.monotonic_ns())?;
        match wait {
            Some(wait) => lifecycle.admission_wait(wait),
            None => Ok(()),
        }
    }

    pub(crate) fn is_admitted(&self) -> bool {
        self.0.0.lock_lifecycle().admitted.is_some()
    }

    pub(crate) fn terminalize_for_release(
        &self,
        fallback: RuntimeTerminalOutcome,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        self.0.terminalize_for_release(fallback)
    }

    /// Finishes an admission that ended without a grant: rejection,
    /// cancellation, or queue timeout. A terminal outcome the owner already
    /// committed wins over `fallback`.
    pub(crate) fn finish_not_admitted(
        self,
        fallback: RuntimeTerminalOutcome,
    ) -> Result<RuntimeTerminalOutcome, AttemptLifecycleError> {
        self.0.finalize_not_admitted(fallback)
    }

    /// Consumes the admission after GPU cleanup with its capacity and health
    /// facts.
    pub(crate) fn finish_admitted(self, commit: AttemptTerminalCommit) -> Result<(), AttemptLifecycleError> {
        self.0.finalize_admitted(commit)
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

    /// Publishes the engine-owned composed schedule into this attempt's
    /// immutable report. The lifecycle verifies the projection's attempt
    /// placement against the already-recorded admission fact; callers cannot
    /// replace the schedule after finalization.
    pub fn record_composed_schedule_projection(
        &self,
        projection: AttemptComposedScheduleProjection,
    ) -> Result<(), AttemptLifecycleError> {
        if projection.attempt_id != self.0.attempt_id {
            return Err(AttemptLifecycleError::InvalidOrder {
                boundary: "composed_schedule",
                reason: "attempt_id_mismatch",
            });
        }
        self.0.lock_lifecycle().composed_schedule(projection)
    }

    pub fn record_physical_execution_started(&self) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().physical(self.0.monotonic_ns())
    }

    pub fn record_gpu_execution_started(&self) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().gpu(self.0.monotonic_ns())
    }

    /// Retains the safe, closed projection of a borrowed native error without
    /// granting terminalization or sealing authority.
    ///
    /// The record-only capability is used by adapter paths whose admitted
    /// handle may begin cleanup before the outer DataFusion wrapper returns
    /// the original error to the attempt owner.  The full error remains on
    /// that returned path; this records only its bounded stable summary.
    pub fn record_native_failure(&self, error: &QueryError) {
        let operation = self.0.lock_lifecycle().failure_operation();
        if let Some(summary) = RuntimeTerminalOutcome::primary_failure_from_query_error(error, operation) {
            self.0.lock_lifecycle().primary_failure(summary);
        }
    }

    /// Counts one secondary cleanup failure in the lifecycle before trying to
    /// retain its optional timeline projection. Timeline loss therefore never
    /// changes the sealed report's exact cleanup accounting.
    #[cfg(test)]
    pub(crate) fn record_cleanup_failure(&self, phase: CleanupFailurePhase, cleanup_error: &QueryError) {
        let summary = CleanupFailureSummary::from_native_error(phase, cleanup_error);
        self.0.lock_lifecycle().cleanup_failure(summary.clone());
        let Some(observation) = self.0.observation.get() else { return };
        observation.record(|| {
            RuntimeObservationPayload::CleanupFailure(Box::new(RuntimeCleanupFailureObservation { summary }))
        });
    }
}

impl AttemptCompletion {
    /// Returns whether the terminal report has been published.
    pub fn is_ready(&self) -> bool {
        self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner).report_visible
    }

    /// Returns the published report without waiting, when ready.
    pub fn try_report(&self) -> Option<Arc<QueryAttemptReport>> {
        let state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.report_visible.then(|| state.report.as_ref().map(Arc::clone)).flatten()
    }

    /// Blocks the current thread until the report is published or the timeout
    /// elapses. Intended for synchronous tests and diagnostics.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Arc<QueryAttemptReport>> {
        let state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        let (state, _) = self
            .cell
            .changed
            .wait_timeout_while(state, timeout, |state| !state.report_visible)
            .unwrap_or_else(sync::PoisonError::into_inner);
        state.report_visible.then(|| state.report.as_ref().map(Arc::clone)).flatten()
    }
}

impl Future for AttemptCompletion {
    type Output = Arc<QueryAttemptReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        if state.report_visible
            && let Some(report) = &state.report
        {
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
        let mut state = self.cell.state.lock().unwrap_or_else(sync::PoisonError::into_inner);
        state.receivers = state.receivers.saturating_sub(1);
    }
}

impl fmt::Debug for AttemptCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptCompletion").field("ready", &self.is_ready()).finish_non_exhaustive()
    }
}

impl fmt::Debug for AttemptCompletionFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AttemptCompletionFactory").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use query_planning::error::{QueryError, QueryErrorCode};

    use super::*;
    use crate::{
        observability::{
            DeterministicRuntimeObservationClock, RuntimeObservationConfig, RuntimeObservationRecord,
            write_runtime_observation_jsonl,
        },
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
            Arc::clone(&clock) as _,
        );
        (begin, clock, counters)
    }

    fn native_disposition(begin: &AttemptBegin) {
        let handle = begin.owner.handle();
        handle.record_planning_started().unwrap();
        handle.record_planning_finished().unwrap();
        handle.set_disposition(AttemptDisposition::Native).unwrap();
    }

    fn claim_admitted(begin: &AttemptBegin, fact: AttemptAdmissionFact) -> AttemptAdmission {
        let admission = begin.owner.admission_binding().claim().unwrap();
        admission.record_admitted(fact, None).unwrap();
        admission
    }

    #[test]
    fn non_admitted_commit_publishes_immediately_in_order() {
        let (begin, clock, counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.completion();
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
            snapshot.footer().attempt_report.as_ref().expect("attempt report allocation embedded"),
            &report
        ));
        let mut jsonl = Vec::new();
        write_runtime_observation_jsonl(&snapshot, &mut jsonl).expect("attempt JSONL exports");
        let exported_report = std::str::from_utf8(&jsonl)
            .expect("attempt JSONL is UTF-8")
            .lines()
            .map(|line| serde_json::from_str::<RuntimeObservationRecord>(line).expect("attempt JSONL record parses"))
            .find_map(|record| match record {
                RuntimeObservationRecord::Footer(footer) => footer.attempt_report,
                _ => None,
            })
            .expect("JSONL footer retains the attempt report");
        assert_eq!(exported_report.as_ref(), report.as_ref());
    }

    #[test]
    fn sealed_report_sink_receives_the_exact_report_before_completion_visibility() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let factory = begin.completion.clone();
        let completion_for_sink = factory.clone();
        let completion = factory.completion();
        let delivered = Arc::new(Mutex::new(Vec::new()));
        let delivered_for_sink = Arc::clone(&delivered);
        let observed_before_completion = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_before_completion_for_sink = Arc::clone(&observed_before_completion);
        begin.owner.register_sealed_report_sink(move |report| {
            observed_before_completion_for_sink
                .store(completion_for_sink.try_report().is_none(), std::sync::atomic::Ordering::Release);
            delivered_for_sink.lock().unwrap().push(report);
        });

        native_disposition(&begin);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();

        let report = completion.try_report().expect("completion is published after sink delivery");
        let delivered = delivered.lock().unwrap();
        assert_eq!(delivered.len(), 1);
        assert!(Arc::ptr_eq(&delivered[0], &report));
        assert!(observed_before_completion.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn seal_uses_the_attempt_terminal_outcome_for_footer_report_and_boundary() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.completion();
        native_disposition(&begin);
        let observation_owner = begin.owner.observation_owner_for_test();
        observation_owner.record_terminal_outcome(RuntimeTerminalOutcome::Failed {
            error_code: AttemptFailureCode::Internal,
            error_status: AttemptFailureStatus::Permanent,
        });

        begin.owner.commit_terminal(RuntimeTerminalOutcome::Cancelled).unwrap();
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("report published");
        let snapshot = observation_owner.sealed_snapshot().expect("sealed");

        assert_eq!(report.outcome, RuntimeTerminalOutcome::Cancelled);
        assert_eq!(snapshot.footer().terminal, Some(RuntimeTerminalOutcome::Cancelled));
        assert_eq!(snapshot.attempt_report().expect("attempt report embedded"), report.as_ref());
        assert!(snapshot.events().iter().any(|event| {
            matches!(event.payload, RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Cancelled })
        }));
    }

    #[test]
    fn admitted_commit_defers_until_post_cleanup_seal() {
        let (begin, clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.completion();
        native_disposition(&begin);
        let handle = begin.owner.handle();
        let admission = claim_admitted(
            &begin,
            AttemptAdmissionFact { admission_sequence: 3, admission_wait_ns: 1, device_ordinal: 1, grant_bytes: 1024 },
        );
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
        admission
            .finish_admitted(AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                Some(final_domain_usage),
                AttemptHealthTransition::HealthyReuse,
            ))
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(Some(RuntimeTerminalOutcome::Succeeded));
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
    fn admitted_terminal_commit_projects_typed_native_fault_evidence() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        native_disposition(&begin);
        let admission = claim_admitted(
            &begin,
            AttemptAdmissionFact { admission_sequence: 3, admission_wait_ns: 1, device_ordinal: 1, grant_bytes: 1024 },
        );
        let native_fault_evidence = AttemptNativeFaultEvidence {
            isolation: AttemptFaultIsolation::DeviceLocal,
            teardown_health: AttemptTeardownHealth::Completed,
            device_health: Some(AttemptDeviceHealth {
                status: AttemptDeviceHealthStatus::Unhealthy,
                fault_count: 1,
                rejected_operation_count: 0,
                quarantined_teardown_count: 0,
            }),
            first_fault: Some(AttemptNativeFault {
                kind: AttemptNativeFaultKind::FatalDeviceOperation,
                // The cuDF ABI is unsigned; this sentinel guards against
                // narrowing it through an i32-only report field.
                status: Some(AttemptNativeStatus::Cudf(u32::MAX)),
            }),
        };
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        admission
            .finish_admitted(
                AttemptTerminalCommit::new(
                    AttemptCapacityDisposition::RetainedQuarantine,
                    None,
                    AttemptHealthTransition::DeviceQuarantined,
                )
                .with_native_fault_evidence(native_fault_evidence),
            )
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(Some(RuntimeTerminalOutcome::Succeeded));

        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::RetainedQuarantine);
        assert_eq!(report.health_transition, Some(AttemptHealthTransition::DeviceQuarantined));
        assert_eq!(report.native_fault_evidence, Some(native_fault_evidence));
    }

    #[test]
    fn terminal_commit_projects_actual_cleanup_and_capacity_before_seal() {
        let (begin, clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.completion();
        native_disposition(&begin);
        let admission = claim_admitted(
            &begin,
            AttemptAdmissionFact { admission_sequence: 3, admission_wait_ns: 1, device_ordinal: 1, grant_bytes: 1024 },
        );
        clock.set_monotonic_ns(20);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        clock.set_monotonic_ns(30);
        admission
            .finish_admitted(AttemptTerminalCommit::new(
                AttemptCapacityDisposition::RetainedQuarantine,
                None,
                AttemptHealthTransition::BackendUnhealthy,
            ))
            .unwrap();
        assert!(
            matches!(begin.owner.admission_binding().claim(), Err(AttemptLifecycleError::AdmissionAlreadyClaimed)),
            "the admission right is linear and cannot be claimed a second time"
        );
        begin.owner.observation_owner_for_test().finish_query(Some(RuntimeTerminalOutcome::Succeeded));
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("report");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed);
        assert_eq!(report.cleanup_failure_count, 0);
        assert!(report.cleanup_failure_count_complete);
        assert!(report.first_cleanup_failure.is_none());
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
        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
    }

    #[test]
    fn owner_drop_backstop_never_loses_the_report() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.completion();
        drop(begin.owner);
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("drop publishes the report");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Dropped));
    }

    #[test]
    fn fixed_summary_survives_disabled_and_zero_limit_diagnostics() {
        for observation in [RuntimeObservationConfig::disabled(), RuntimeObservationConfig::bounded(0, 0)] {
            let (begin, _clock, _counters) = test_begin(observation, None);
            let completion = begin.completion.completion();
            native_disposition(&begin);
            begin
                .owner
                .commit_terminal(RuntimeTerminalOutcome::Failed {
                    error_code: AttemptFailureCode::Internal,
                    error_status: AttemptFailureStatus::Permanent,
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
        drop(begin.completion.completion());
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
        drop(begin2.completion.completion());
        begin2.owner.commit_terminal(RuntimeTerminalOutcome::Dropped).unwrap();
        assert_eq!(counters2.snapshot().completion_queue_pushed, 1);
        let (begin3, _clock3, counters3) =
            test_begin(RuntimeObservationConfig::native_execution(), Some(Arc::clone(&queue)));
        drop(begin3.completion.completion());
        begin3.owner.commit_terminal(RuntimeTerminalOutcome::Dropped).unwrap();
        assert_eq!(counters3.snapshot().completion_queue_overflowed, 1);
        // The capacity-one queue still holds the second report; the third was
        // counted as delivery loss without blocking the commit.
        assert_eq!(queue.drain(8).len(), 1);
    }

    #[test]
    fn contended_completion_push_blocks_for_capacity_instead_of_reporting_overflow() {
        let queue = Arc::new(AttemptCompletionQueue::new(4).expect("queue"));
        let (begin, _clock, _counters) =
            test_begin(RuntimeObservationConfig::native_execution(), Some(Arc::clone(&queue)));
        drop(begin.completion.completion());
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        let report = queue.drain(8).pop().expect("abandoned report");
        // Hold the queue mutex while a pusher arrives: with spare capacity
        // the push must wait for the lock, not fail as if the queue were
        // full. The barrier plus hold window prove the pusher attempted its
        // push while contended.
        let guard = queue.hold_lock_for_test();
        let barrier = sync::Barrier::new(2);
        let (pushed_tx, pushed_rx) = sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                pushed_tx.send(queue.try_push(Arc::clone(&report))).expect("push result");
            });
            barrier.wait();
            std::thread::sleep(Duration::from_millis(20));
            assert!(pushed_rx.try_recv().is_err(), "contended push must block, not fail fast");
            drop(guard);
            assert!(pushed_rx.recv().expect("push result after release"));
        });
        assert_eq!(queue.drain(8).len(), 1);
    }

    #[test]
    fn held_receivers_do_not_route_to_the_abandoned_queue() {
        let queue = Arc::new(AttemptCompletionQueue::new(4).expect("queue"));
        let (begin, _clock, counters) =
            test_begin(RuntimeObservationConfig::native_execution(), Some(Arc::clone(&queue)));
        let _held = begin.completion.completion();
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
        let admission = claim_admitted(
            &begin,
            AttemptAdmissionFact { admission_sequence: 1, admission_wait_ns: 0, device_ordinal: 0, grant_bytes: 1 },
        );
        let primary_error =
            QueryError::new(QueryErrorCode::CudfTemporaryFailure, "test.primary", "primary native failure");
        begin.owner.commit_query_error(&primary_error).unwrap();
        let cleanup_error = QueryError::new(QueryErrorCode::Internal, "test.cleanup", "injected cleanup failure");
        handle.record_cleanup_failure(CleanupFailurePhase::QueryServiceCleanup, &cleanup_error);
        handle.record_cleanup_failure(CleanupFailurePhase::QueryServiceCleanup, &cleanup_error);
        admission
            .finish_admitted(AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                None,
                AttemptHealthTransition::HealthyReuse,
            ))
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(Some(RuntimeTerminalOutcome::Failed {
            error_code: AttemptFailureCode::CudfTemporaryFailure,
            error_status: AttemptFailureStatus::Temporary,
        }));
        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        assert_eq!(report.cleanup_failure_count, 2);
        assert!(report.cleanup_failure_count_complete);
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::CompletedWithFailures);
        assert!(matches!(
            report.outcome,
            RuntimeTerminalOutcome::Failed { ref error_code, .. }
                if *error_code == AttemptFailureCode::CudfTemporaryFailure
        ));
        let primary = report.primary_failure.as_ref().expect("primary failure summary");
        assert_eq!(primary.code(), AttemptFailureCode::CudfTemporaryFailure);
        assert!(primary.text().is_none(), "native error text must not enter the stable report");
        let first_cleanup = report.first_cleanup_failure.as_ref().expect("first cleanup failure summary");
        assert_eq!(first_cleanup.phase(), CleanupFailurePhase::QueryServiceCleanup);
        assert_eq!(first_cleanup.failure().code(), AttemptFailureCode::Internal);
        assert!(!first_cleanup.complete(), "the first retained summary is incomplete after a later cleanup failure");
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
    fn borrowed_primary_capture_preserves_the_returned_error_source_chain() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        native_disposition(&begin);
        let error = QueryError::new(QueryErrorCode::CudfTemporaryFailure, "test.primary", "native failure")
            .with_source(std::io::Error::other("original source cause"));
        assert_eq!(error.source_error().map(ToString::to_string).as_deref(), Some("original source cause"));

        begin.owner.commit_query_error(&error).unwrap();

        assert_eq!(error.source_error().map(ToString::to_string).as_deref(), Some("original source cause"));
        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        let summary = report.primary_failure.as_ref().expect("borrowed native summary");
        assert_eq!(summary.code(), AttemptFailureCode::CudfTemporaryFailure);
        assert_eq!(summary.status(), AttemptFailureStatus::Temporary);
        assert!(summary.text().is_none(), "the returned error, not the report, retains native message text");
    }

    #[test]
    fn borrowed_primary_capture_wins_when_admitted_cleanup_finalizes_before_outer_error_handling() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        native_disposition(&begin);
        let handle = begin.owner.handle();
        let admission = claim_admitted(
            &begin,
            AttemptAdmissionFact { admission_sequence: 1, admission_wait_ns: 0, device_ordinal: 0, grant_bytes: 1 },
        );

        // This models synchronous `input.execute` failure: the adapter still
        // borrows the native error while dropping the admitted handle starts
        // cleanup and finalization before the outer DataFusion boundary can
        // commit its own generic fallback.
        let error = QueryError::new(
            QueryErrorCode::CudfTemporaryFailure,
            "test.native_admission_execute",
            "application-specific native failure text must remain on the returned error",
        )
        .with_fact("reason", "deadline_expired")
        .with_source(std::io::Error::other("original source cause"));
        handle.record_native_failure(&error);

        let terminal = admission.terminalize_for_release(RuntimeTerminalOutcome::Dropped).unwrap();
        admission
            .finish_admitted(AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                None,
                AttemptHealthTransition::HealthyReuse,
            ))
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(Some(terminal));

        assert_eq!(error.source_error().map(ToString::to_string).as_deref(), Some("original source cause"));
        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        assert!(matches!(
            report.outcome,
            RuntimeTerminalOutcome::Failed {
                error_code: AttemptFailureCode::CudfTemporaryFailure,
                error_status: AttemptFailureStatus::Temporary,
            }
        ));
        let summary = report.primary_failure.as_ref().expect("captured native primary summary");
        assert_eq!(summary.code(), AttemptFailureCode::CudfTemporaryFailure);
        assert_eq!(summary.status(), AttemptFailureStatus::Temporary);
        assert_eq!(summary.operation(), AttemptFailureOperation::Execution);
        assert_eq!(summary.facts().len(), 1);
        assert_eq!(summary.facts()[0].key(), AttemptFailureFactKey::Reason);
        assert_eq!(summary.facts()[0].value(), AttemptFailureFactValue::DeadlineExpired);
        assert!(summary.text().is_none());
    }

    #[test]
    fn execution_failure_with_plan_in_operation_seals_as_execution() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        native_disposition(&begin);
        let handle = begin.owner.handle();
        let admission = claim_admitted(
            &begin,
            AttemptAdmissionFact { admission_sequence: 1, admission_wait_ns: 0, device_ordinal: 0, grant_bytes: 1 },
        );

        let error = QueryError::new(
            QueryErrorCode::CudfTemporaryFailure,
            "query_engine.exec.execute_query_plan",
            "execution failure",
        );
        handle.record_native_failure(&error);

        let terminal = admission.terminalize_for_release(RuntimeTerminalOutcome::Dropped).unwrap();
        admission
            .finish_admitted(AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                None,
                AttemptHealthTransition::HealthyReuse,
            ))
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(Some(terminal));

        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        let summary = report.primary_failure.as_ref().expect("captured native primary summary");
        assert_eq!(summary.operation(), AttemptFailureOperation::Execution);
    }

    #[test]
    fn cleanup_facts_survive_a_full_timeline_ring() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::bounded(0, 0), None);
        native_disposition(&begin);
        let handle = begin.owner.handle();
        let admission = claim_admitted(
            &begin,
            AttemptAdmissionFact { admission_sequence: 1, admission_wait_ns: 0, device_ordinal: 0, grant_bytes: 1 },
        );
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        let cleanup_error = QueryError::new(QueryErrorCode::Internal, "test.cleanup", "ring-full cleanup failure");
        handle.record_cleanup_failure(CleanupFailurePhase::QueryServiceCleanup, &cleanup_error);
        admission
            .finish_admitted(AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                None,
                AttemptHealthTransition::HealthyReuse,
            ))
            .unwrap();
        begin.owner.observation_owner_for_test().finish_query(Some(RuntimeTerminalOutcome::Succeeded));

        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        assert_eq!(report.cleanup_failure_count, 1);
        assert!(report.cleanup_failure_count_complete);
        assert_eq!(
            report.first_cleanup_failure.as_ref().map(CleanupFailureSummary::phase),
            Some(CleanupFailurePhase::QueryServiceCleanup)
        );
        let snapshot = begin.owner.observation_owner_for_test().sealed_snapshot().expect("sealed");
        assert!(
            snapshot
                .events()
                .iter()
                .all(|event| !matches!(event.payload, RuntimeObservationPayload::CleanupFailure(_)))
        );
    }

    #[test]
    fn deadline_expired_cancelled_error_retains_its_bounded_reason() {
        let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let error = QueryError::new(QueryErrorCode::Cancelled, "test.deadline", "deadline elapsed")
            .with_fact("reason", "deadline_expired")
            .with_fact("cancellation_requested", true);

        begin.owner.commit_query_error(&error).unwrap();

        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        assert_eq!(report.outcome, RuntimeTerminalOutcome::Cancelled);
        let summary = report.primary_failure.as_ref().expect("deadline summary");
        assert_eq!(summary.code(), AttemptFailureCode::Cancelled);
        assert_eq!(summary.origin(), AttemptFailureOrigin::Native);
        assert_eq!(summary.facts().len(), 1);
        assert_eq!(summary.facts()[0].value(), AttemptFailureFactValue::DeadlineExpired);
    }

    #[test]
    fn caller_and_generic_cancellation_keep_the_control_summary() {
        for error in [
            QueryError::new(QueryErrorCode::Cancelled, "test.caller_cancelled", "caller cancelled")
                .with_fact("reason", "caller_cancelled"),
            QueryError::new(QueryErrorCode::Cancelled, "test.generic_cancelled", "cancelled"),
        ] {
            let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
            begin.owner.commit_query_error(&error).unwrap();

            let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
            assert_eq!(report.outcome, RuntimeTerminalOutcome::Cancelled);
            let summary = report.primary_failure.as_ref().expect("control cancellation summary");
            assert_eq!(summary.code(), AttemptFailureCode::Cancelled);
            assert_eq!(summary.origin(), AttemptFailureOrigin::AttemptControl);
            assert!(summary.facts().is_empty());
        }
    }

    #[test]
    fn control_terminal_paths_publish_text_free_primary_summaries() {
        for (outcome, code) in [
            (RuntimeTerminalOutcome::Cancelled, AttemptFailureCode::Cancelled),
            (RuntimeTerminalOutcome::Dropped, AttemptFailureCode::Dropped),
            (RuntimeTerminalOutcome::Panicked, AttemptFailureCode::Panicked),
        ] {
            let (begin, _clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
            begin.owner.commit_terminal(outcome).unwrap();
            let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
            let summary = report.primary_failure.as_ref().expect("control summary");
            assert_eq!(summary.code(), code);
            assert_eq!(summary.origin(), AttemptFailureOrigin::AttemptControl);
            assert_eq!(summary.status(), AttemptFailureStatus::NotApplicable);
            assert!(summary.text().is_none());
        }
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
    use query_planning::error::QueryErrorCode;

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
        // The admission binding terminalizes the rejection at its source.
        admission_rejection
            .reject(&QueryError::new(
                QueryErrorCode::ServiceOverloaded,
                "test.admission_rejection",
                "injected admission rejection",
            ))
            .unwrap();
        // The wrapper's later commit loses and observes the structured error.
        assert_eq!(
            begin.owner.commit_terminal(RuntimeTerminalOutcome::Failed {
                error_code: AttemptFailureCode::DatafusionError,
                error_status: AttemptFailureStatus::Permanent,
            }),
            Err(AttemptLifecycleError::DoubleTerminalization)
        );
        // The engine post-rejection seal publishes the truthful outcome.
        begin.owner.observation_owner_for_test().finish_query(Some(RuntimeTerminalOutcome::Failed {
            error_code: AttemptFailureCode::ServiceOverloaded,
            error_status: AttemptFailureStatus::Temporary,
        }));
        let report = begin.completion.completion().wait_timeout(Duration::from_secs(1)).expect("report");
        assert!(matches!(
            report.outcome,
            RuntimeTerminalOutcome::Failed { ref error_code, .. }
                if *error_code == AttemptFailureCode::ServiceOverloaded
        ));
        assert_eq!(report.disposition, AttemptDisposition::Native);
        assert_eq!(report.timestamps.admission_requested_ns, None);
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
            let completion = begin.completion.completion();
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
            let factory = begin.completion.clone();
            let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
            let cloner_factory = factory.clone();
            let cloner = std::thread::spawn(move || {
                go_rx.recv().expect("go");
                cloner_factory.completion()
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
        let execution_cancellation = cancellation.execution_cancellation();
        assert!(execution_cancellation.is_expired(), "native execution must retain the attempt deadline");
        assert_eq!(execution_cancellation.deadline(), cancellation.deadline());
        let error = cancellation.check("test.deadline").unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
        assert_eq!(error.fact_value("cancellation_requested"), Some("true"));
        let execution_error = execution_cancellation.check("test.native_deadline").unwrap_err();
        assert_eq!(execution_error.fact_value("reason"), Some("deadline_expired"));
        assert!(cancellation.is_cancelled(), "expiry is an effective cancellation at native safe boundaries");
    }
}
