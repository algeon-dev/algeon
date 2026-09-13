//! Query observation session, seal observers, and recorder event writes.

use std::{
    fmt,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use super::{
    CleanupFailureSummary, LiveRuntimeObservationSnapshot, PlanExecutionId, QueryAttemptReport, RetainedTopologyRecord,
    RuntimeEventContext, RuntimeObservationClassLoss, RuntimeObservationClock, RuntimeObservationConfig,
    RuntimeObservationDeliveryGuard, RuntimeObservationFooter, RuntimeObservationHandle, RuntimeObservationManifest,
    RuntimeObservationOwner, RuntimeObservationPayload, RuntimeObservationReasonLoss, RuntimeObservationRecordClass,
    RuntimeObservationRecordDisposition, RuntimeObservationRecorderState, RuntimeObservationSnapshotData,
    RuntimeQueryId, RuntimeTerminalOutcome, RuntimeTimelineEvent, RuntimeTopologyRecord, RuntimeTraceId,
    SealedRuntimeObservationSnapshot, SourceLocationDigestKeyState,
};
#[cfg(any(test, feature = "test-hooks"))]
use super::{SystemRuntimeObservationClock, TestSourceLocationDigestKeyProvider};

/// Owner-only capability for one query observation lifetime.
///
/// This type is intentionally not [`Clone`]. Cloneable execution contexts and
/// workers receive only [`RuntimeObservationHandle`], which cannot seal.
#[derive(Debug, Default)]
pub struct RuntimeObservationSession {
    pub(super) handle: RuntimeObservationHandle,
}

/// Construction-time provider of the fixed execution-attempt summary embedded at seal.
///
/// This compatibility hook is retained for recorder-only owners. Attempt
/// owners use the crate-private seal provider below so the embedded value and
/// completion delivery share the same immutable allocation.
pub type AttemptReportProvider = Arc<dyn Fn() -> QueryAttemptReport + Send + Sync>;

/// Attempt-owned report construction after the recorder has computed its
/// complete footer.
pub(super) type AttemptReportSealProvider =
    Arc<dyn Fn(&RuntimeObservationFooter) -> Arc<QueryAttemptReport> + Send + Sync>;

/// Crate-internal authority that selects the canonical attempt outcome before
/// the recorder fixes the footer. Attempt-owned observations use this to
/// collapse recorder-local pending outcomes into the attempt lifecycle's one
/// terminal state.
pub(super) type RuntimeObservationTerminalOutcomeResolver =
    Arc<dyn Fn(RuntimeTerminalOutcome) -> RuntimeTerminalOutcome + Send + Sync>;

/// Engine-owned lifecycle routing for a cleanup failure. The recorder invokes
/// this before optional timeline retention, so bounded capture loss cannot
/// change the immutable attempt report.
pub(super) type AttemptCleanupFailureRecorder = Arc<dyn Fn(CleanupFailureSummary) + Send + Sync>;

pub(super) struct RuntimeObservationSessionInner {
    pub(super) manifest: RuntimeObservationManifest,
    pub(super) source_location_digest_key: SourceLocationDigestKeyState,
    pub(super) clock: Arc<dyn RuntimeObservationClock>,
    pub(super) state: Mutex<RuntimeObservationRecorderState>,
    pub(super) sealed: AtomicBool,
}

pub(super) struct RuntimeObservationSealObserver {
    pub(super) notify: Box<dyn Fn(SealedRuntimeObservationSnapshot) + Send + Sync>,
}

impl fmt::Debug for RuntimeObservationSealObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RuntimeObservationSealObserver").finish_non_exhaustive()
    }
}

impl RuntimeObservationSealObserver {
    pub(super) fn invoke(&self, snapshot: SealedRuntimeObservationSnapshot) {
        let _delivery = RuntimeObservationDeliveryGuard::enter(None);
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.notify)(snapshot))).is_err() {
            eprintln!("[query-engine] warning: runtime observation seal observer panicked");
        }
    }
}

/// Deferred observer delivery for an already-sealed query cleanup.
/// The seal has published the report; delivery runs after capacity release
/// and never gates reuse.
#[doc(hidden)]
pub struct QueryCleanupSealDelivery {
    pub(super) owner: Option<RuntimeObservationOwner>,
    pub(super) deferred: Option<DeferredRuntimeObservationSeal>,
    pub(super) already_sealed: Option<SealedRuntimeObservationSnapshot>,
}

impl QueryCleanupSealDelivery {
    pub(super) fn already_sealed(snapshot: Option<SealedRuntimeObservationSnapshot>) -> Self {
        Self { owner: None, deferred: None, already_sealed: snapshot }
    }

    /// Runs observer delivery, returning the sealed snapshot.
    #[doc(hidden)]
    pub fn deliver(self) -> Option<SealedRuntimeObservationSnapshot> {
        if let Some(owner) = self.owner
            && let Some(deferred) = self.deferred
        {
            return owner.complete_observer_delivery(deferred);
        }
        self.already_sealed
    }
}

pub(super) struct DeferredRuntimeObservationSeal {
    pub(super) snapshot: Option<SealedRuntimeObservationSnapshot>,
    pub(super) observers: Vec<RuntimeObservationSealObserver>,
}

impl DeferredRuntimeObservationSeal {
    pub(super) fn invoke_observers(self) -> Option<SealedRuntimeObservationSnapshot> {
        if let Some(snapshot) = &self.snapshot {
            for observer in self.observers {
                observer.invoke(snapshot.clone());
            }
        }
        self.snapshot
    }
}

impl fmt::Debug for RuntimeObservationSessionInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeObservationSessionInner")
            .field("manifest", &self.manifest)
            .field("sealed", &self.sealed.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl RuntimeObservationSession {
    /// Creates an owner and recording handle using a system monotonic clock.
    pub fn new(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
    ) -> Self {
        Self { handle: RuntimeObservationHandle::new(query_id, trace_id, config) }
    }

    /// Creates an owner and recording handle using an injected clock.
    pub fn with_clock(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        clock: Arc<dyn RuntimeObservationClock>,
    ) -> Self {
        Self { handle: RuntimeObservationHandle::with_clock(query_id, trace_id, config, clock) }
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn with_source_location_test_key(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        key: [u8; blake3::KEY_LEN],
    ) -> Self {
        Self {
            handle: RuntimeObservationHandle::with_clock_and_source_location_digest_key(
                query_id,
                trace_id,
                config,
                Arc::new(SystemRuntimeObservationClock::new()),
                &TestSourceLocationDigestKeyProvider { key: Some(key) },
            ),
        }
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn with_unavailable_source_location_test_key(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
    ) -> Self {
        Self {
            handle: RuntimeObservationHandle::with_clock_and_source_location_digest_key(
                query_id,
                trace_id,
                config,
                Arc::new(SystemRuntimeObservationClock::new()),
                &TestSourceLocationDigestKeyProvider { key: None },
            ),
        }
    }

    /// Creates a disabled owner and handle.
    pub const fn disabled() -> Self {
        Self { handle: RuntimeObservationHandle::disabled() }
    }

    /// Returns a cloneable recording capability without seal authority.
    pub fn handle(&self) -> RuntimeObservationHandle {
        self.handle.clone()
    }

    /// Returns the immutable recorder configuration.
    pub const fn config(&self) -> RuntimeObservationConfig {
        self.handle.config()
    }

    /// Returns the immutable query identity when recording is enabled.
    pub fn query_id(&self) -> Option<&RuntimeQueryId> {
        self.handle.query_id()
    }

    /// Returns the immutable optional parent trace identity.
    pub fn trace_id(&self) -> Option<&RuntimeTraceId> {
        self.handle.trace_id()
    }

    /// Lazily constructs and records a query-level event.
    pub fn record<F>(&self, build_payload: F) -> RuntimeObservationRecordDisposition
    where F: FnOnce() -> RuntimeObservationPayload {
        self.handle.record(build_payload)
    }

    /// Lazily constructs and records an event for one plan execution.
    pub fn record_for_plan<F>(
        &self,
        plan_execution_id: PlanExecutionId,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        self.handle.record_for_plan(plan_execution_id, build_payload)
    }

    /// Lazily constructs an event before acquiring recorder order and records it infallibly.
    pub fn record_with_context<F>(
        &self,
        context: RuntimeEventContext,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        self.handle.record_with_context(context, build_payload)
    }

    /// Lazily constructs and records one static topology row.
    pub fn record_topology<F>(
        &self,
        class: RuntimeObservationRecordClass,
        build_record: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeTopologyRecord,
    {
        self.handle.record_topology(class, build_record)
    }

    /// Captures an immutable incomplete view. Returns none after sealing or when disabled.
    pub fn live_snapshot(&self) -> Option<LiveRuntimeObservationSnapshot> {
        self.handle.live_snapshot()
    }

    /// Fixes the first terminal outcome and returns the same immutable snapshot to authorized racers.
    pub fn seal(&self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        self.handle.seal_owned(outcome)
    }

    pub(super) fn seal_deferred(
        &self,
        outcome: Option<RuntimeTerminalOutcome>,
        attempt_report_provider: Option<&AttemptReportSealProvider>,
    ) -> DeferredRuntimeObservationSeal {
        self.handle.seal_owned_deferred(outcome, attempt_report_provider)
    }

    /// Returns the sealed immutable snapshot when a terminal outcome already won.
    pub fn sealed_snapshot(&self) -> Option<SealedRuntimeObservationSnapshot> {
        self.handle.sealed_snapshot()
    }
}

impl RuntimeObservationSessionInner {
    pub(super) fn lock_state(&self) -> MutexGuard<'_, RuntimeObservationRecorderState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.degraded = true;
                state.poison_recoveries = state.poison_recoveries.checked_add(1).unwrap_or_else(|| {
                    state.counter_overflow = true;
                    u64::MAX
                });
                self.state.clear_poison();
                state
            },
        }
    }

    pub(super) fn record_event_lazy<F>(
        &self,
        context: RuntimeEventContext,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        self.record_event_timestamped(context, |_monotonic_ns| build_payload()).0
    }

    pub(super) fn record_event_sized<F>(
        &self,
        context: RuntimeEventContext,
        logical_bytes: u64,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        let mut state = self.lock_state();
        if state.sealed_snapshot.is_some() {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        let sequence = state.take_sequence();
        if let Some(reason) = state.rejection_reason(self.manifest.limits, logical_bytes) {
            state.record_drop(RuntimeObservationRecordClass::TimelineEvent, sequence, reason);
            return RuntimeObservationRecordDisposition::Dropped { sequence, reason };
        }
        let monotonic_ns = self.clock.monotonic_ns().max(state.last_monotonic_ns);
        state.last_monotonic_ns = monotonic_ns;
        let event = RuntimeTimelineEvent {
            sequence,
            monotonic_ns,
            plan_execution_id: context.plan_execution_id,
            span_id: context.span_id,
            parent_span_id: context.parent_span_id,
            payload: build_payload(),
        };
        let actual_bytes = event.logical_retained_bytes();
        if actual_bytes != logical_bytes {
            state.degraded = true;
        }
        match state.admit(self.manifest.limits, RuntimeObservationRecordClass::TimelineEvent, sequence, actual_bytes) {
            Ok(()) => {
                state.events.push(event);
                RuntimeObservationRecordDisposition::Retained { sequence, logical_bytes: actual_bytes }
            },
            Err(reason) => RuntimeObservationRecordDisposition::Dropped { sequence, reason },
        }
    }

    pub(super) fn record_event_timestamped<F>(
        &self,
        context: RuntimeEventContext,
        build_payload: F,
    ) -> (RuntimeObservationRecordDisposition, Option<u64>)
    where
        F: FnOnce(u64) -> RuntimeObservationPayload,
    {
        let mut state = self.lock_state();
        if state.sealed_snapshot.is_some() {
            return (RuntimeObservationRecordDisposition::Sealed, None);
        }
        let sequence = state.take_sequence();
        let monotonic_ns = self.clock.monotonic_ns().max(state.last_monotonic_ns);
        state.last_monotonic_ns = monotonic_ns;
        const MIN_TIMELINE_EVENT_BYTES: u64 = 42;
        if let Some(reason) = state.rejection_reason(self.manifest.limits, MIN_TIMELINE_EVENT_BYTES) {
            state.record_drop(RuntimeObservationRecordClass::TimelineEvent, sequence, reason);
            return (RuntimeObservationRecordDisposition::Dropped { sequence, reason }, Some(monotonic_ns));
        }
        let event = RuntimeTimelineEvent {
            sequence,
            monotonic_ns,
            plan_execution_id: context.plan_execution_id,
            span_id: context.span_id,
            parent_span_id: context.parent_span_id,
            payload: build_payload(monotonic_ns),
        };
        let logical_bytes = event.logical_retained_bytes();
        let disposition = match state.admit(
            self.manifest.limits,
            RuntimeObservationRecordClass::TimelineEvent,
            sequence,
            logical_bytes,
        ) {
            Ok(()) => {
                state.events.push(event);
                RuntimeObservationRecordDisposition::Retained { sequence, logical_bytes }
            },
            Err(reason) => RuntimeObservationRecordDisposition::Dropped { sequence, reason },
        };
        (disposition, Some(monotonic_ns))
    }

    pub(super) fn record_topology_lazy<F>(
        &self,
        class: RuntimeObservationRecordClass,
        build_record: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeTopologyRecord,
    {
        let mut state = self.lock_state();
        if state.sealed_snapshot.is_some() {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        let sequence = state.take_sequence();
        const MIN_TOPOLOGY_RECORD_BYTES: u64 = 24;
        if let Some(reason) = state.rejection_reason(self.manifest.limits, MIN_TOPOLOGY_RECORD_BYTES) {
            state.record_drop(class, sequence, reason);
            return RuntimeObservationRecordDisposition::Dropped { sequence, reason };
        }
        let record = build_record();
        if record.class() != class {
            state.degraded = true;
        }
        let logical_bytes = record.logical_retained_bytes();
        match state.admit(self.manifest.limits, record.class(), sequence, logical_bytes) {
            Ok(()) => {
                state.topology.push(RetainedTopologyRecord { sequence, record });
                RuntimeObservationRecordDisposition::Retained { sequence, logical_bytes }
            },
            Err(reason) => RuntimeObservationRecordDisposition::Dropped { sequence, reason },
        }
    }

    pub(super) fn snapshot_data(
        &self,
        state: &RuntimeObservationRecorderState,
        complete: bool,
        terminal: Option<RuntimeTerminalOutcome>,
    ) -> RuntimeObservationSnapshotData {
        let mut topology = state.topology.clone();
        topology.sort_by(|left, right| left.record.cmp(&right.record).then(left.sequence.cmp(&right.sequence)));
        let mut events = state.events.clone();
        events.sort_by_key(|event| event.sequence);
        let dropped_by_class = state
            .dropped_by_class
            .iter()
            .map(|(class, count)| RuntimeObservationClassLoss { class: *class, count: *count })
            .collect();
        let dropped_by_reason = state
            .dropped_by_reason
            .iter()
            .map(|(reason, count)| RuntimeObservationReasonLoss { reason: *reason, count: *count })
            .collect();
        RuntimeObservationSnapshotData {
            manifest: self.manifest.clone(),
            topology,
            events,
            footer: RuntimeObservationFooter {
                complete,
                terminal,
                retained_records: state.retained_records,
                retained_bytes: state.retained_bytes,
                first_dropped_sequence: state.first_dropped_sequence,
                last_attempted_sequence: state.last_attempted_sequence,
                dropped_by_class,
                dropped_by_reason,
                truncated: state.first_dropped_sequence.is_some(),
                degraded: state.degraded,
                poison_recoveries: state.poison_recoveries,
                attempt_report: None,
                counter_overflow: state.counter_overflow,
            },
        }
    }

    pub(super) fn take_sealed_snapshot_data(
        &self,
        state: &mut RuntimeObservationRecorderState,
        outcome: Option<RuntimeTerminalOutcome>,
        attempt_report_provider: Option<&AttemptReportSealProvider>,
    ) -> RuntimeObservationSnapshotData {
        let mut topology = std::mem::take(&mut state.topology);
        topology.sort_by(|left, right| left.record.cmp(&right.record).then(left.sequence.cmp(&right.sequence)));
        let mut events = std::mem::take(&mut state.events);
        events.sort_by_key(|event| event.sequence);
        let dropped_by_class = state
            .dropped_by_class
            .iter()
            .map(|(class, count)| RuntimeObservationClassLoss { class: *class, count: *count })
            .collect();
        let dropped_by_reason = state
            .dropped_by_reason
            .iter()
            .map(|(reason, count)| RuntimeObservationReasonLoss { reason: *reason, count: *count })
            .collect();
        let mut footer = RuntimeObservationFooter {
            complete: true,
            terminal: outcome,
            retained_records: state.retained_records,
            retained_bytes: state.retained_bytes,
            first_dropped_sequence: state.first_dropped_sequence,
            last_attempted_sequence: state.last_attempted_sequence,
            dropped_by_class,
            dropped_by_reason,
            truncated: state.first_dropped_sequence.is_some(),
            degraded: state.degraded,
            poison_recoveries: state.poison_recoveries,
            attempt_report: None,
            counter_overflow: state.counter_overflow,
        };
        // The provider runs only after the complete footer exists. It returns
        // the exact immutable report that completion receivers publish, rather
        // than a pre-seal report which would be missing report-ready facts.
        footer.attempt_report = attempt_report_provider.map(|provider| provider(&footer));
        debug_assert!(
            footer.attempt_report.as_ref().is_none_or(|report| { footer.terminal.as_ref() == Some(&report.outcome) })
        );
        RuntimeObservationSnapshotData { manifest: self.manifest.clone(), topology, events, footer }
    }
}
