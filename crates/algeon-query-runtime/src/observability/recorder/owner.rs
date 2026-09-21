//! Shared terminal ownership, lifecycle fence, and observer delivery.

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    cell::RefCell,
    fmt,
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

use super::{
    AttemptCleanupFailureRecorder, AttemptReportProvider, AttemptReportSealProvider, CleanupFailureSummary,
    DeferredRuntimeObservationSeal, LiveRuntimeObservationSnapshot, QueryAttemptReport, QueryBoundaryKind,
    QueryCleanupSealDelivery, RuntimeObservationClock, RuntimeObservationConfig, RuntimeObservationFooter,
    RuntimeObservationHandle, RuntimeObservationPayload, RuntimeObservationSealObserver, RuntimeObservationSession,
    RuntimeObservationTerminalOutcomeResolver, RuntimeQueryId, RuntimeTerminalOutcome, RuntimeTraceId,
    SealedRuntimeObservationSnapshot,
};

/// Cloneable shared ownership of one query observation session.
///
/// Clones carry the same terminal authority and the recorder's first-terminal
/// rule makes sealing idempotent. Execution contexts still receive only
/// [`RuntimeObservationHandle`].
#[derive(Clone)]
pub struct RuntimeObservationOwner {
    session: Arc<RuntimeObservationSession>,
    lifecycle: Arc<RuntimeObservationOwnerLifecycle>,
    attempt_report_provider: Option<AttemptReportProvider>,
    attempt_report_seal_provider: Option<AttemptReportSealProvider>,
    terminal_outcome_resolver: Option<RuntimeObservationTerminalOutcomeResolver>,
    attempt_cleanup_failure_recorder: Option<AttemptCleanupFailureRecorder>,
    #[cfg(test)]
    deferred_delivery_gate: Arc<Mutex<Option<Arc<RuntimeObservationDeliveryGate>>>>,
}

#[derive(Debug)]
pub(super) struct RuntimeObservationOwnerLifecycle {
    state: Mutex<RuntimeObservationOwnerLifecycleState>,
    finished: Condvar,
}

/// Test-only barrier at the start of ordinary deferred observer delivery.
///
/// The terminal seal commit deliberately runs before this barrier. It lets
/// lifecycle tests prove that report-ready does not depend on observer
/// delivery or capacity reuse.
#[cfg(test)]
#[derive(Debug)]
#[doc(hidden)]
pub struct RuntimeObservationDeliveryGate {
    entered: AtomicBool,
    gate: std::sync::Barrier,
}

#[cfg(test)]
impl RuntimeObservationDeliveryGate {
    #[doc(hidden)]
    pub fn entered(&self) -> bool {
        self.entered.load(Ordering::Acquire)
    }

    #[doc(hidden)]
    pub fn open(&self) {
        self.gate.wait();
    }
}

#[derive(Debug)]
pub(super) enum RuntimeObservationOwnerLifecycleState {
    Open {
        cleanup_started: bool,
        pending_outcome: Option<RuntimeTerminalOutcome>,
    },
    Finishing {
        late_observers: Vec<RuntimeObservationSealObserver>,
    },
    Delivering {
        snapshot: Option<SealedRuntimeObservationSnapshot>,
        late_observers: Vec<RuntimeObservationSealObserver>,
    },
    Sealed(Option<SealedRuntimeObservationSnapshot>),
}

thread_local! {
    static RUNTIME_OBSERVATION_DELIVERY_STACK: RefCell<Vec<Option<*const RuntimeObservationOwnerLifecycle>>> =
        const { RefCell::new(Vec::new()) };
}

/// Returns whether the current thread is synchronously delivering a runtime-observation seal observer.
///
/// Backend close-and-wait uses this to reject synchronous reentry, which would
/// otherwise wait for the observer currently executing on this thread.
#[doc(hidden)]
pub fn in_runtime_observation_delivery() -> bool {
    RUNTIME_OBSERVATION_DELIVERY_STACK.with(|stack| !stack.borrow().is_empty())
}

fn in_owner_observation_delivery(lifecycle: &Arc<RuntimeObservationOwnerLifecycle>) -> bool {
    let lifecycle = Arc::as_ptr(lifecycle);
    RUNTIME_OBSERVATION_DELIVERY_STACK
        .with(|stack| stack.borrow().iter().any(|active| active.is_some_and(|active| active == lifecycle)))
}

pub(super) struct RuntimeObservationDeliveryGuard {
    lifecycle: Option<*const RuntimeObservationOwnerLifecycle>,
}

impl RuntimeObservationDeliveryGuard {
    pub(super) fn enter(lifecycle: Option<&Arc<RuntimeObservationOwnerLifecycle>>) -> Self {
        let lifecycle = lifecycle.map(Arc::as_ptr);
        RUNTIME_OBSERVATION_DELIVERY_STACK.with(|stack| stack.borrow_mut().push(lifecycle));
        Self { lifecycle }
    }
}

impl Drop for RuntimeObservationDeliveryGuard {
    fn drop(&mut self) {
        RUNTIME_OBSERVATION_DELIVERY_STACK.with(|stack| {
            let active = stack.borrow_mut().pop();
            debug_assert_eq!(active, Some(self.lifecycle));
        });
    }
}

impl Default for RuntimeObservationOwnerLifecycle {
    fn default() -> Self {
        Self {
            state: Mutex::new(RuntimeObservationOwnerLifecycleState::Open {
                cleanup_started: false,
                pending_outcome: None,
            }),
            finished: Condvar::new(),
        }
    }
}

impl fmt::Debug for RuntimeObservationOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeObservationOwner")
            .field("query_id", &self.query_id().map(RuntimeQueryId::as_str))
            .field("enabled", &self.handle().is_enabled())
            .finish_non_exhaustive()
    }
}

impl RuntimeObservationOwner {
    /// Creates shared owner authority for a new query observation session.
    pub fn new(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
    ) -> Self {
        Self::from_session(RuntimeObservationSession::new(query_id, trace_id, config))
    }

    /// Creates shared owner authority with an injected monotonic clock.
    pub fn with_clock(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        clock: Arc<dyn RuntimeObservationClock>,
    ) -> Self {
        Self::from_session(RuntimeObservationSession::with_clock(query_id, trace_id, config, clock))
    }

    /// Wraps an existing unique session in shared one-shot ownership.
    pub fn from_session(session: RuntimeObservationSession) -> Self {
        let owner = Self {
            session: Arc::new(session),
            lifecycle: Arc::new(RuntimeObservationOwnerLifecycle::default()),
            attempt_report_provider: None,
            attempt_report_seal_provider: None,
            terminal_outcome_resolver: None,
            attempt_cleanup_failure_recorder: None,
            #[cfg(test)]
            deferred_delivery_gate: Arc::new(Mutex::new(None)),
        };
        owner.handle().record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::Started });
        owner
    }

    /// Installs a construction-time fixed attempt-summary provider.
    pub fn with_attempt_report_provider(mut self, provider: AttemptReportProvider) -> Self {
        self.attempt_report_provider = Some(provider);
        self
    }

    /// Installs attempt-owned report construction at the sole seal point.
    /// This remains internal so no adapter can manufacture a second report for
    /// a live attempt.
    #[doc(hidden)]
    pub fn with_attempt_report_seal_provider<F>(mut self, provider: F) -> Self
    where F: Fn(&RuntimeObservationFooter) -> Arc<QueryAttemptReport> + Send + Sync + 'static {
        self.attempt_report_seal_provider = Some(Arc::new(provider));
        self
    }

    /// Installs the attempt-owned terminal outcome resolver. It runs after
    /// cleanup evidence is recorded but before the recorder seals, so the
    /// footer and attempt report cannot disagree on the terminal outcome.
    #[doc(hidden)]
    pub fn with_terminal_outcome_resolver<F>(mut self, resolver: F) -> Self
    where F: Fn(RuntimeTerminalOutcome) -> RuntimeTerminalOutcome + Send + Sync + 'static {
        self.terminal_outcome_resolver = Some(Arc::new(resolver));
        self
    }

    /// Installs the engine-owned attempt cleanup recorder. This internal hook
    /// is only routing: lifecycle state remains the sole owner of the count
    /// and first-summary facts.
    #[doc(hidden)]
    pub fn with_attempt_cleanup_failure_recorder<F>(mut self, recorder: F) -> Self
    where F: Fn(CleanupFailureSummary) + Send + Sync + 'static {
        self.attempt_cleanup_failure_recorder = Some(Arc::new(recorder));
        self
    }

    /// Blocks ordinary observer delivery after the terminal commit. This is
    /// test-only and scoped to this observation owner.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn arm_deferred_delivery_gate(&self) -> Arc<RuntimeObservationDeliveryGate> {
        let gate = Arc::new(RuntimeObservationDeliveryGate {
            entered: AtomicBool::new(false),
            gate: std::sync::Barrier::new(2),
        });
        let mut slot = self.deferred_delivery_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(slot.is_none(), "an observation owner can arm only one deferred-delivery gate");
        *slot = Some(Arc::clone(&gate));
        gate
    }

    /// Creates a disabled shared owner.
    pub fn disabled() -> Self {
        Self::from_session(RuntimeObservationSession::disabled())
    }

    /// Returns cloneable recording access without terminal authority.
    pub fn handle(&self) -> RuntimeObservationHandle {
        let mut handle = self.session.handle();
        handle.attempt_cleanup_failure_recorder = self.attempt_cleanup_failure_recorder.clone();
        handle
    }

    /// Returns the immutable query identity.
    pub fn query_id(&self) -> Option<&RuntimeQueryId> {
        self.session.query_id()
    }

    /// Returns the immutable optional trace identity.
    pub fn trace_id(&self) -> Option<&RuntimeTraceId> {
        self.session.trace_id()
    }

    /// Returns the immutable recorder configuration used to create this owner.
    pub fn config(&self) -> RuntimeObservationConfig {
        self.session.config()
    }

    /// Fixes the first terminal and returns the shared sealed snapshot.
    pub fn seal(&self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        let outcome = loop {
            let mut lifecycle = self.lock_lifecycle();
            match &mut *lifecycle {
                RuntimeObservationOwnerLifecycleState::Open { .. } => {
                    *lifecycle = RuntimeObservationOwnerLifecycleState::Finishing { late_observers: Vec::new() };
                    break outcome;
                },
                RuntimeObservationOwnerLifecycleState::Finishing { .. } => {
                    lifecycle = match self.lifecycle.finished.wait(lifecycle) {
                        Ok(lifecycle) => lifecycle,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if let RuntimeObservationOwnerLifecycleState::Sealed(snapshot) = &*lifecycle {
                        return snapshot.clone();
                    }
                },
                RuntimeObservationOwnerLifecycleState::Delivering { snapshot, .. } => {
                    if in_owner_observation_delivery(&self.lifecycle) {
                        return snapshot.clone();
                    }
                    lifecycle = match self.lifecycle.finished.wait(lifecycle) {
                        Ok(lifecycle) => lifecycle,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if let RuntimeObservationOwnerLifecycleState::Sealed(snapshot) = &*lifecycle {
                        return snapshot.clone();
                    }
                },
                RuntimeObservationOwnerLifecycleState::Sealed(snapshot) => return snapshot.clone(),
            }
        };
        self.finish_owner_terminal(Some(outcome), |_| {})
    }

    /// Retains one plan or admission outcome until the query owner completes cleanup.
    ///
    /// The first non-success outcome remains authoritative, so later cleanup or
    /// cancellation cannot reclassify an already structured native failure.
    pub fn record_terminal_outcome(&self, outcome: RuntimeTerminalOutcome) {
        let mut lifecycle = self.lock_lifecycle();
        if let RuntimeObservationOwnerLifecycleState::Open { pending_outcome, .. } = &mut *lifecycle {
            *pending_outcome = Some(match pending_outcome.take() {
                None | Some(RuntimeTerminalOutcome::Succeeded) => outcome,
                Some(existing) => existing,
            });
        }
    }

    /// Returns the terminal outcome that a cleanup release must establish
    /// before it commits its capacity facts. A recorded execution outcome
    /// wins over the mechanical fallback; a seal already in progress is
    /// allowed to finish so its immutable terminal outcome remains canonical.
    #[doc(hidden)]
    pub fn terminal_outcome_for_cleanup(&self, fallback: RuntimeTerminalOutcome) -> RuntimeTerminalOutcome {
        let mut lifecycle = self.lock_lifecycle();
        loop {
            match &*lifecycle {
                RuntimeObservationOwnerLifecycleState::Open { pending_outcome, .. } => {
                    return pending_outcome.clone().unwrap_or(fallback);
                },
                RuntimeObservationOwnerLifecycleState::Finishing { .. }
                | RuntimeObservationOwnerLifecycleState::Delivering { snapshot: None, .. } => {
                    lifecycle = match self.lifecycle.finished.wait(lifecycle) {
                        Ok(lifecycle) => lifecycle,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                },
                RuntimeObservationOwnerLifecycleState::Delivering { snapshot: Some(snapshot), .. }
                | RuntimeObservationOwnerLifecycleState::Sealed(Some(snapshot)) => {
                    return snapshot.footer().terminal.clone().unwrap_or(fallback);
                },
                RuntimeObservationOwnerLifecycleState::Sealed(None) => return fallback,
            }
        }
    }

    /// Records that query-owned cleanup has begun without sealing the recorder.
    #[doc(hidden)]
    pub fn begin_query_cleanup(&self) {
        let should_record = {
            let mut lifecycle = self.lock_lifecycle();
            match &mut *lifecycle {
                RuntimeObservationOwnerLifecycleState::Open { cleanup_started, .. } if !*cleanup_started => {
                    *cleanup_started = true;
                    true
                },
                RuntimeObservationOwnerLifecycleState::Open { .. }
                | RuntimeObservationOwnerLifecycleState::Finishing { .. }
                | RuntimeObservationOwnerLifecycleState::Delivering { .. }
                | RuntimeObservationOwnerLifecycleState::Sealed(_) => false,
            }
        };
        if should_record {
            self.handle()
                .record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupStarted });
        }
    }

    /// Seals after query-owned cleanup, using `fallback` only when execution
    /// never produced a more specific outcome.
    #[doc(hidden)]
    pub fn finish_query_cleanup(&self, fallback: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        self.seal_query_cleanup(fallback).deliver()
    }

    /// Records cleanup completion and seals the recorder state so the report
    /// is published, returning the
    /// deferred observer delivery. Capacity release may proceed once this
    /// returns; [`QueryCleanupSealDelivery::deliver`] runs afterwards and
    /// never gates reuse.
    #[doc(hidden)]
    pub fn seal_query_cleanup(&self, fallback: RuntimeTerminalOutcome) -> QueryCleanupSealDelivery {
        self.seal_query_cleanup_with_fallback(Some(fallback))
    }

    fn seal_query_cleanup_with_fallback(&self, fallback: Option<RuntimeTerminalOutcome>) -> QueryCleanupSealDelivery {
        self.begin_query_cleanup();
        let outcome = loop {
            let mut lifecycle = self.lock_lifecycle();
            match &mut *lifecycle {
                RuntimeObservationOwnerLifecycleState::Open { pending_outcome, .. } => {
                    let outcome = pending_outcome.take().or_else(|| fallback.clone());
                    *lifecycle = RuntimeObservationOwnerLifecycleState::Finishing { late_observers: Vec::new() };
                    break outcome;
                },
                RuntimeObservationOwnerLifecycleState::Finishing { .. } => {
                    lifecycle = match self.lifecycle.finished.wait(lifecycle) {
                        Ok(lifecycle) => lifecycle,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if let RuntimeObservationOwnerLifecycleState::Sealed(snapshot) = &*lifecycle {
                        return QueryCleanupSealDelivery::already_sealed(snapshot.clone());
                    }
                },
                RuntimeObservationOwnerLifecycleState::Delivering { snapshot, .. } => {
                    if in_owner_observation_delivery(&self.lifecycle) {
                        return QueryCleanupSealDelivery::already_sealed(snapshot.clone());
                    }
                    lifecycle = match self.lifecycle.finished.wait(lifecycle) {
                        Ok(lifecycle) => lifecycle,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if let RuntimeObservationOwnerLifecycleState::Sealed(snapshot) = &*lifecycle {
                        return QueryCleanupSealDelivery::already_sealed(snapshot.clone());
                    }
                },
                RuntimeObservationOwnerLifecycleState::Sealed(snapshot) => {
                    return QueryCleanupSealDelivery::already_sealed(snapshot.clone());
                },
            }
        };
        let deferred = self.seal_owner_terminal(outcome, |outcome| {
            self.handle()
                .record(|| RuntimeObservationPayload::QueryBoundary { boundary: QueryBoundaryKind::CleanupFinished });
            if let Some(outcome) = outcome {
                let boundary = query_terminal_boundary(outcome);
                self.handle().record(|| RuntimeObservationPayload::QueryBoundary { boundary });
            }
        });
        QueryCleanupSealDelivery { owner: Some(self.clone()), deferred: Some(deferred), already_sealed: None }
    }

    /// Completes a query that has no admitted handle responsible for cleanup.
    ///
    /// `None` is reserved for coordinator-owned records that have no attempt
    /// lifecycle. Their observation still seals, but its terminal field stays
    /// absent rather than minting an engine attempt outcome from delivery data.
    pub fn finish_query(&self, outcome: Option<RuntimeTerminalOutcome>) -> Option<SealedRuntimeObservationSnapshot> {
        match outcome {
            Some(outcome) => {
                self.record_terminal_outcome(outcome.clone());
                self.finish_query_cleanup(outcome)
            },
            None => self.seal_query_cleanup_with_fallback(None).deliver(),
        }
    }

    /// Returns a live incomplete snapshot before terminal sealing.
    pub fn live_snapshot(&self) -> Option<LiveRuntimeObservationSnapshot> {
        self.session.live_snapshot()
    }

    /// Returns the immutable sealed snapshot after a terminal wins.
    pub fn sealed_snapshot(&self) -> Option<SealedRuntimeObservationSnapshot> {
        self.session.sealed_snapshot()
    }

    /// Registers an infallible notification for this session's immutable seal.
    ///
    /// The observer runs at most once for this registration and outside
    /// recorder locks. Registrations racing terminal publication join the
    /// owner's delivery fence; registrations after that fence has completed
    /// deliver the existing snapshot immediately. A panic is isolated from
    /// query execution and cleanup.
    pub fn register_seal_observer<F>(&self, observer: F)
    where F: Fn(SealedRuntimeObservationSnapshot) + Send + Sync + 'static {
        let mut observer = Some(RuntimeObservationSealObserver { notify: Box::new(observer) });
        let immediate = {
            let mut lifecycle = self.lock_lifecycle();
            match &mut *lifecycle {
                RuntimeObservationOwnerLifecycleState::Open { .. } => {
                    self.handle().register_seal_observer(observer.take().expect("observer exists"))
                },
                RuntimeObservationOwnerLifecycleState::Finishing { late_observers }
                | RuntimeObservationOwnerLifecycleState::Delivering { late_observers, .. } => {
                    late_observers.push(observer.take().expect("observer exists"));
                    None
                },
                RuntimeObservationOwnerLifecycleState::Sealed(snapshot) => {
                    snapshot.clone().map(|snapshot| (observer.take().expect("observer exists"), snapshot))
                },
            }
        };
        if let Some((observer, snapshot)) = immediate {
            observer.invoke(snapshot);
        }
    }

    fn finish_owner_terminal<F>(
        &self,
        outcome: Option<RuntimeTerminalOutcome>,
        before_seal: F,
    ) -> Option<SealedRuntimeObservationSnapshot>
    where
        F: FnOnce(Option<&RuntimeTerminalOutcome>),
    {
        let deferred = self.seal_owner_terminal(outcome, before_seal);
        self.complete_observer_delivery(deferred)
    }

    /// Seals the recorder state and returns the deferred observer delivery.
    fn seal_owner_terminal<F>(
        &self,
        outcome: Option<RuntimeTerminalOutcome>,
        before_seal: F,
    ) -> DeferredRuntimeObservationSeal
    where
        F: FnOnce(Option<&RuntimeTerminalOutcome>),
    {
        let finishing = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let outcome = outcome.map(|outcome| match &self.terminal_outcome_resolver {
                Some(resolver) => resolver(outcome),
                None => outcome,
            });
            before_seal(outcome.as_ref());
            let attempt_report_provider = outcome.as_ref().and_then(|_| {
                self.attempt_report_seal_provider.clone().or_else(|| {
                    self.attempt_report_provider.as_ref().map(|provider| {
                        let provider = Arc::clone(provider);
                        Arc::new(move |footer: &RuntimeObservationFooter| {
                            let mut report = provider();
                            report.set_observation_completeness(footer);
                            Arc::new(report)
                        }) as AttemptReportSealProvider
                    })
                })
            });
            self.session.seal_deferred(outcome, attempt_report_provider.as_ref())
        }));
        match finishing {
            Ok(deferred) => deferred,
            Err(panic) => {
                self.handle().mark_degraded();
                let deferred = self.session.seal_deferred(Some(RuntimeTerminalOutcome::Panicked), None);
                let _ = self.complete_observer_delivery(deferred);
                std::panic::resume_unwind(panic);
            },
        }
    }

    pub(super) fn complete_observer_delivery(
        &self,
        deferred: DeferredRuntimeObservationSeal,
    ) -> Option<SealedRuntimeObservationSnapshot> {
        let snapshot = deferred.snapshot.clone();
        let mut observers = deferred.observers;
        {
            let mut lifecycle = self.lock_lifecycle();
            let late_observers = match &mut *lifecycle {
                RuntimeObservationOwnerLifecycleState::Finishing { late_observers } => std::mem::take(late_observers),
                _ => unreachable!("terminal owner must publish before observer delivery"),
            };
            if snapshot.is_some() {
                observers.extend(late_observers);
            }
            *lifecycle = RuntimeObservationOwnerLifecycleState::Delivering {
                snapshot: snapshot.clone(),
                late_observers: Vec::new(),
            };
        }
        // The test gate deliberately sits before every ordinary observer. The
        // terminal seal commit has already run by this point.
        #[cfg(test)]
        if let Some(gate) = self.deferred_delivery_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
        {
            gate.entered.store(true, Ordering::Release);
            gate.gate.wait();
        }
        let _delivery = RuntimeObservationDeliveryGuard::enter(Some(&self.lifecycle));
        loop {
            if let Some(snapshot) = &snapshot {
                for observer in observers.drain(..) {
                    observer.invoke(snapshot.clone());
                }
            }
            let mut lifecycle = self.lock_lifecycle();
            let late_observers = match &mut *lifecycle {
                RuntimeObservationOwnerLifecycleState::Delivering { late_observers, .. } => {
                    std::mem::take(late_observers)
                },
                _ => unreachable!("observer delivery must retain terminal owner state"),
            };
            if late_observers.is_empty() || snapshot.is_none() {
                *lifecycle = RuntimeObservationOwnerLifecycleState::Sealed(snapshot.clone());
                self.lifecycle.finished.notify_all();
                return snapshot;
            }
            observers = late_observers;
        }
    }

    pub(super) fn lock_lifecycle(&self) -> MutexGuard<'_, RuntimeObservationOwnerLifecycleState> {
        match self.lifecycle.state.lock() {
            Ok(lifecycle) => lifecycle,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl Default for RuntimeObservationOwner {
    fn default() -> Self {
        Self::disabled()
    }
}

fn query_terminal_boundary(outcome: &RuntimeTerminalOutcome) -> QueryBoundaryKind {
    match outcome {
        RuntimeTerminalOutcome::Succeeded => QueryBoundaryKind::Succeeded,
        RuntimeTerminalOutcome::Failed { .. } => QueryBoundaryKind::Failed,
        RuntimeTerminalOutcome::Cancelled => QueryBoundaryKind::Cancelled,
        RuntimeTerminalOutcome::Dropped => QueryBoundaryKind::Dropped,
        RuntimeTerminalOutcome::Panicked => QueryBoundaryKind::Panicked,
    }
}
