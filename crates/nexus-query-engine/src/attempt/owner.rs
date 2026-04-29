//! Attempt ownership, boundary recording, and independent completion delivery.
use std::{
    fmt,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

use super::{
    AttemptBackendCounters, AttemptCancellation, AttemptCompletionQueue, AttemptLifecycle, AttemptLifecycleError,
};
use crate::{
    error::QueryError,
    observability::{
        AttemptAdmissionFact, AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptDisposition, AttemptId,
        AttemptQueryId, AttemptTraceContext, QUERY_ATTEMPT_REPORT_SCHEMA_V1, QueryAttemptReport,
        RuntimeCleanupFailureObservation, RuntimeObservationClock, RuntimeObservationConfig, RuntimeObservationFooter,
        RuntimeObservationHandle, RuntimeObservationOwner, RuntimeObservationPayload, RuntimeQueryId,
        RuntimeTerminalOutcome, RuntimeTraceId, SystemRuntimeObservationClock,
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

impl Clone for AttemptHandle {
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
    cleanup_failure_count: AtomicU64,
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

    /// Builds the fixed report from lifecycle state. Observation completeness
    /// facts start zeroed; the recorder seal path patches the embedded copy
    /// from the computed footer and [`Self::finish_with_footer`] patches the
    /// delivered copy from the sealed snapshot footer.
    fn report(&self) -> QueryAttemptReport {
        let lifecycle = self.lock_lifecycle();
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
            cleanup_disposition: if lifecycle.admitted.is_some() {
                if self.cleanup_failure_count.load(Ordering::Acquire) == 0 {
                    AttemptCleanupDisposition::Completed
                } else {
                    AttemptCleanupDisposition::CompletedWithFailures
                }
            } else {
                AttemptCleanupDisposition::NotRequired
            },
            cleanup_failure_count: self.cleanup_failure_count.load(Ordering::Acquire),
            capacity_disposition: if lifecycle.admitted.is_some() {
                AttemptCapacityDisposition::PendingRelease
            } else {
                AttemptCapacityDisposition::NotAdmitted
            },
            observation_lost_records: 0,
            observation_retained_records: 0,
            observation_retained_bytes: 0,
            observation_truncated: false,
            observation_degraded: false,
            unix_epoch_ns: self.clock.unix_epoch_ns(),
        }
    }

    /// The single commit completion path: cleanup commit (admitted only),
    /// exact aggregates, report-ready, then publish. Idempotent; the first
    /// call wins so the seal observer and the owner backstop cannot race a
    /// second publication.
    fn finish_with_footer(&self, footer: &RuntimeObservationFooter) -> Result<(), AttemptLifecycleError> {
        let report = {
            let mut lifecycle = self.lock_lifecycle();
            if lifecycle.ready.is_some() {
                return Ok(());
            }
            if lifecycle.admitted.is_some() {
                lifecycle.cleanup(self.monotonic_ns())?;
            }
            lifecycle.ready(self.monotonic_ns())?;
            drop(lifecycle);
            let mut report = self.report();
            report.set_observation_completeness(footer);
            Arc::new(report)
        };
        self.counters.reports_ready.fetch_add(1, Ordering::AcqRel);
        let mut state = self.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.report.is_some() {
            return Ok(());
        }
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
            if queue.try_push(report) {
                self.counters.completion_queue_pushed.fetch_add(1, Ordering::AcqRel);
            } else {
                self.counters.completion_queue_overflowed.fetch_add(1, Ordering::AcqRel);
            }
        }
        // The report is durable: the attempt's observation permit has nothing
        // left to fence and is released before any waiter observes close
        // quiescence. Report and completion state carry no lease.
        self.observation_permit.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        Ok(())
    }

    fn finish_without_observation(&self) -> Result<(), AttemptLifecycleError> {
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
            cleanup_failure_count: AtomicU64::new(0),
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
        let provider_core = Arc::clone(&core);
        let owner = RuntimeObservationOwner::with_clock(query_id, trace_id, observation, clock)
            .with_attempt_report_provider(Arc::new(move || provider_core.report()));
        // The seal observer completes the deferred commit for admitted
        // attempts: the engine seals this owner only after GPU cleanup, so
        // cleanup-complete and report-ready are recorded at delivery. A seal
        // that arrives without a wrapper commit is terminalized as dropped so
        // completion delivery cannot be lost.
        let observer_core = Arc::clone(&core);
        owner.register_seal_observer(move |snapshot| {
            {
                let mut lifecycle = observer_core.lock_lifecycle();
                if lifecycle.terminal.is_none() {
                    let _ = lifecycle.terminalize(RuntimeTerminalOutcome::Dropped, observer_core.monotonic_ns());
                }
            }
            let _ = observer_core.finish_with_footer(snapshot.footer());
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

    /// An independent receiver for the terminal report.
    pub fn completion(&self) -> AttemptCompletion {
        self.0.core.cell.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).receivers += 1;
        AttemptCompletion { cell: Arc::clone(&self.0.core.cell), counted: true }
    }

    /// The attempt-scoped cooperative cancellation signal and deadline.
    pub fn cancellation(&self) -> AttemptCancellation {
        self.0.core.cancellation.clone()
    }

    /// The observation owner handed to the admission ticket so the engine
    /// seals this attempt's observation after GPU cleanup.
    pub fn observation_owner(&self) -> RuntimeObservationOwner {
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
                    Some(snapshot) => self.0.core.finish_with_footer(snapshot.footer())?,
                    None => self.0.core.finish_without_observation()?,
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

    /// First-wins terminalization for the admission binding.
    ///
    /// The [`AttemptOwner`] remains the primary terminalization authority.
    /// This handle-level path exists for exactly one window: when GPU
    /// admission itself is rejected, cancelled, or closed before physical
    /// execution starts, the engine seals the attempt observation immediately
    /// and the terminal fact must already be recorded or the report would
    /// degrade to a synthetic `Dropped`. Exactly-once is enforced by the
    /// state machine; a racing owner commit observes
    /// [`AttemptLifecycleError::DoubleTerminalization`].
    pub fn terminalize(&self, outcome: RuntimeTerminalOutcome) -> Result<(), AttemptLifecycleError> {
        self.0.lock_lifecycle().terminalize(outcome, self.0.monotonic_ns())
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
        self.0.cleanup_failure_count.fetch_add(1, Ordering::AcqRel);
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
        assert_eq!(report.timestamps.cleanup_complete_ns, None);
        assert_eq!(report.timestamps.report_ready_ns, Some(10));
        assert_eq!(report.timestamps.outcome_terminal_ns, 10);
        assert_eq!(counters.snapshot().reports_ready, 1);
        // The sealed observation snapshot embeds the same fixed summary.
        let snapshot = begin.owner.observation_owner().sealed_snapshot().expect("sealed");
        let embedded = snapshot.attempt_report().expect("attempt report embedded");
        assert_eq!(embedded.attempt_id, report.attempt_id);
        assert_eq!(embedded.disposition, report.disposition);
    }

    #[test]
    fn admitted_commit_defers_until_post_cleanup_seal() {
        let (begin, clock, _counters) = test_begin(RuntimeObservationConfig::native_execution(), None);
        let completion = begin.completion.clone();
        native_disposition(&begin);
        let handle = begin.owner.handle();
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
        begin.owner.observation_owner().finish_query(RuntimeTerminalOutcome::Succeeded);
        let report = completion.wait_timeout(Duration::from_secs(1)).expect("report published after seal");
        assert_eq!(report.timestamps.outcome_terminal_ns, 20);
        assert_eq!(report.timestamps.cleanup_complete_ns, Some(30));
        assert_eq!(report.timestamps.report_ready_ns, Some(30));
        assert_eq!(report.device_ordinal, Some(1));
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed);
        assert_eq!(report.capacity_disposition, AttemptCapacityDisposition::PendingRelease);
        assert_eq!(report.admission.as_ref().map(|fact| fact.grant_bytes), Some(1024));
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
        begin.owner.observation_owner().finish_query(RuntimeTerminalOutcome::Failed {
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
        let snapshot = begin.owner.observation_owner().sealed_snapshot().expect("sealed");
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
    use crate::{
        observability::{DeterministicRuntimeObservationClock, RuntimeObservationConfig},
        runtime::BackendLifecycle,
    };

    #[test]
    fn admission_rejection_terminalized_at_the_binding_wins_over_the_wrapper_commit() {
        let lifecycle = BackendLifecycle::new();
        let permit = lifecycle.try_begin_observation().expect("test permit");
        let counters = Arc::new(AttemptBackendCounters::default());
        let clock = Arc::new(DeterministicRuntimeObservationClock::new(Some(1)));
        let begin = AttemptOwner::begin_with_clock(BeginAttemptRequest::default(), permit, counters, None, clock);
        let handle = begin.owner.handle();
        handle.record_planning_started().unwrap();
        handle.record_planning_finished().unwrap();
        handle.set_disposition(AttemptDisposition::Native).unwrap();
        handle.record_admission_requested().unwrap();
        // The admission binding terminalizes the rejection at its source.
        handle
            .terminalize(RuntimeTerminalOutcome::Failed {
                error_code: "service_overloaded".to_string(),
                error_status: "temporary".to_string(),
            })
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
        begin.owner.observation_owner().finish_query(RuntimeTerminalOutcome::Failed {
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
