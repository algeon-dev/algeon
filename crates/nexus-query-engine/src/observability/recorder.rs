//! Query-scoped bounded synchronous runtime observation recorder.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use super::{
    CleanupFailureSummary, LogicalRetainedBytes, ObservationSequence, PlanExecutionId, QueryAttemptReport,
    QueryBoundaryKind, RuntimeObservationClassLoss, RuntimeObservationClock, RuntimeObservationFooter,
    RuntimeObservationLimits, RuntimeObservationLossReason, RuntimeObservationManifest, RuntimeObservationPayload,
    RuntimeObservationReasonLoss, RuntimeObservationRecordClass, RuntimeQueryId, RuntimeSourceLocationId,
    RuntimeSpanId, RuntimeTerminalOutcome, RuntimeTimelineEvent, RuntimeTopologyRecord, RuntimeTraceId,
    SystemRuntimeObservationClock,
};
use crate::observability::{RUNTIME_OBSERVATION_JSONL_SCHEMA, RuntimeClockDomain};

/// Query-local recorder configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeObservationConfig {
    enabled: bool,
    limits: RuntimeObservationLimits,
}

impl RuntimeObservationConfig {
    /// Default bounded capture used by native execution adapters until server
    /// retention policy is introduced in Phase 4.
    pub const NATIVE_EXECUTION_MAX_RECORDS: u64 = 65_536;
    /// Deterministic logical-byte cap paired with [`Self::NATIVE_EXECUTION_MAX_RECORDS`].
    pub const NATIVE_EXECUTION_MAX_BYTES: u64 = 8 * 1024 * 1024;

    /// Disables runtime recording.
    pub const fn disabled() -> Self {
        Self { enabled: false, limits: RuntimeObservationLimits { max_retained_records: 0, max_retained_bytes: 0 } }
    }

    /// Enables recording with deterministic record and logical-byte caps.
    pub const fn bounded(max_retained_records: u64, max_retained_bytes: u64) -> Self {
        Self { enabled: true, limits: RuntimeObservationLimits { max_retained_records, max_retained_bytes } }
    }

    /// Enables the bounded per-query native execution capture profile.
    pub const fn native_execution() -> Self {
        Self::bounded(Self::NATIVE_EXECUTION_MAX_RECORDS, Self::NATIVE_EXECUTION_MAX_BYTES)
    }

    /// Returns whether payload construction and recording are enabled.
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }

    /// Returns the configured retention limits.
    pub const fn limits(self) -> RuntimeObservationLimits {
        self.limits
    }
}

impl Default for RuntimeObservationConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Optional event identity attached outside the payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeEventContext {
    pub plan_execution_id: Option<PlanExecutionId>,
    pub span_id: Option<RuntimeSpanId>,
    pub parent_span_id: Option<RuntimeSpanId>,
}

impl RuntimeEventContext {
    /// Creates context for one plan execution.
    pub fn for_plan(plan_execution_id: PlanExecutionId) -> Self {
        Self { plan_execution_id: Some(plan_execution_id), ..Self::default() }
    }

    /// Attaches one span and its optional parent.
    pub const fn with_span(mut self, span_id: RuntimeSpanId, parent_span_id: Option<RuntimeSpanId>) -> Self {
        self.span_id = Some(span_id);
        self.parent_span_id = parent_span_id;
        self
    }
}

/// Infallible result of one hot-path record attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeObservationRecordDisposition {
    Disabled,
    Retained { sequence: ObservationSequence, logical_bytes: u64 },
    Dropped { sequence: ObservationSequence, reason: RuntimeObservationLossReason },
    Sealed,
}

/// Immutable live snapshot. Its footer is always incomplete and has no terminal outcome.
#[derive(Debug, Clone)]
pub struct LiveRuntimeObservationSnapshot {
    data: Arc<RuntimeObservationSnapshotData>,
}

/// Immutable sealed snapshot accepted by the JSONL exporter.
#[derive(Debug, Clone)]
pub struct SealedRuntimeObservationSnapshot {
    data: Arc<RuntimeObservationSnapshotData>,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeObservationSnapshotData {
    pub(crate) manifest: RuntimeObservationManifest,
    pub(crate) topology: Vec<RetainedTopologyRecord>,
    pub(crate) events: Vec<RuntimeTimelineEvent>,
    pub(crate) footer: RuntimeObservationFooter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RetainedTopologyRecord {
    pub(crate) sequence: ObservationSequence,
    pub(crate) record: RuntimeTopologyRecord,
}

macro_rules! snapshot_accessors {
    ($snapshot:ident) => {
        impl $snapshot {
            /// Immutable session manifest.
            pub fn manifest(&self) -> &RuntimeObservationManifest {
                &self.data.manifest
            }

            /// Static topology sorted by typed row identity.
            pub fn topology(&self) -> impl ExactSizeIterator<Item = &RuntimeTopologyRecord> {
                self.data.topology.iter().map(|record| &record.record)
            }

            /// Dynamic events sorted by recorder sequence.
            pub fn events(&self) -> &[RuntimeTimelineEvent] {
                &self.data.events
            }

            /// Completeness, terminal, loss, and degradation metadata.
            pub fn footer(&self) -> &RuntimeObservationFooter {
                &self.data.footer
            }

            /// Fixed execution-attempt summary, when this observation belongs to an attempt.
            ///
            /// The embedded report is the same immutable value delivered by
            /// the attempt completion. In particular, `report_ready_ns` is
            /// fixed before this snapshot is published and the JSONL footer
            /// serializes this exact allocation.
            pub fn attempt_report(&self) -> Option<&QueryAttemptReport> {
                self.data.footer.attempt_report.as_deref()
            }
        }
    };
}

snapshot_accessors!(LiveRuntimeObservationSnapshot);
snapshot_accessors!(SealedRuntimeObservationSnapshot);

impl SealedRuntimeObservationSnapshot {
    pub(crate) fn data(&self) -> &RuntimeObservationSnapshotData {
        &self.data
    }

    /// Returns whether both values are views of the same immutable seal.
    ///
    /// Query identifiers are intentionally insufficient for this comparison:
    /// independently-created recorder sessions may share one external query
    /// identity. Ownership boundaries use this identity check to make repeated
    /// delivery idempotent without silently accepting another session.
    pub fn is_same_snapshot(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

/// Cheaply cloneable recording capability without terminal-seal authority.
#[derive(Clone, Default)]
pub struct RuntimeObservationHandle {
    config: RuntimeObservationConfig,
    inner: Option<Arc<RuntimeObservationSessionInner>>,
    attempt_cleanup_failure_recorder: Option<AttemptCleanupFailureRecorder>,
}

impl fmt::Debug for RuntimeObservationHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeObservationHandle")
            .field("enabled", &self.is_enabled())
            .field("limits", &self.config.limits())
            .finish_non_exhaustive()
    }
}

/// Owner-only capability for one query observation lifetime.
///
/// This type is intentionally not [`Clone`]. Cloneable execution contexts and
/// workers receive only [`RuntimeObservationHandle`], which cannot seal.
#[derive(Debug, Default)]
pub struct RuntimeObservationSession {
    handle: RuntimeObservationHandle,
}

/// Construction-time provider of the fixed execution-attempt summary embedded at seal.
///
/// This compatibility hook is retained for recorder-only owners. Attempt
/// owners use the crate-private seal provider below so the embedded value and
/// completion delivery share the same immutable allocation.
pub type AttemptReportProvider = Arc<dyn Fn() -> QueryAttemptReport + Send + Sync>;

/// Attempt-owned report construction after the recorder has computed its
/// complete footer.
type AttemptReportSealProvider = Arc<dyn Fn(&RuntimeObservationFooter) -> Arc<QueryAttemptReport> + Send + Sync>;

/// Crate-internal authority that selects the canonical attempt outcome before
/// the recorder fixes the footer. Attempt-owned observations use this to
/// collapse recorder-local pending outcomes into the attempt lifecycle's one
/// terminal state.
type RuntimeObservationTerminalOutcomeResolver =
    Arc<dyn Fn(RuntimeTerminalOutcome) -> RuntimeTerminalOutcome + Send + Sync>;

/// Engine-owned lifecycle routing for a cleanup failure. The recorder invokes
/// this before optional timeline retention, so bounded capture loss cannot
/// change the immutable attempt report.
type AttemptCleanupFailureRecorder = Arc<dyn Fn(CleanupFailureSummary) + Send + Sync>;

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
struct RuntimeObservationOwnerLifecycle {
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
pub(crate) struct RuntimeObservationDeliveryGate {
    entered: AtomicBool,
    gate: std::sync::Barrier,
}

#[cfg(test)]
impl RuntimeObservationDeliveryGate {
    pub(crate) fn entered(&self) -> bool {
        self.entered.load(Ordering::Acquire)
    }

    pub(crate) fn open(&self) {
        self.gate.wait();
    }
}

#[derive(Debug)]
enum RuntimeObservationOwnerLifecycleState {
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
pub(crate) fn in_runtime_observation_delivery() -> bool {
    RUNTIME_OBSERVATION_DELIVERY_STACK.with(|stack| !stack.borrow().is_empty())
}

fn in_owner_observation_delivery(lifecycle: &Arc<RuntimeObservationOwnerLifecycle>) -> bool {
    let lifecycle = Arc::as_ptr(lifecycle);
    RUNTIME_OBSERVATION_DELIVERY_STACK
        .with(|stack| stack.borrow().iter().any(|active| active.is_some_and(|active| active == lifecycle)))
}

struct RuntimeObservationDeliveryGuard {
    lifecycle: Option<*const RuntimeObservationOwnerLifecycle>,
}

impl RuntimeObservationDeliveryGuard {
    fn enter(lifecycle: Option<&Arc<RuntimeObservationOwnerLifecycle>>) -> Self {
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

struct RuntimeObservationSessionInner {
    manifest: RuntimeObservationManifest,
    source_location_digest_key: SourceLocationDigestKeyState,
    clock: Arc<dyn RuntimeObservationClock>,
    state: Mutex<RuntimeObservationRecorderState>,
    sealed: AtomicBool,
}

struct RuntimeObservationSealObserver {
    notify: Box<dyn Fn(SealedRuntimeObservationSnapshot) + Send + Sync>,
}

impl fmt::Debug for RuntimeObservationSealObserver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RuntimeObservationSealObserver").finish_non_exhaustive()
    }
}

impl RuntimeObservationSealObserver {
    fn invoke(&self, snapshot: SealedRuntimeObservationSnapshot) {
        let _delivery = RuntimeObservationDeliveryGuard::enter(None);
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.notify)(snapshot))).is_err() {
            eprintln!("[nexus-query-engine] warning: runtime observation seal observer panicked");
        }
    }
}

/// Deferred observer delivery for an already-sealed query cleanup (P2.7).
/// The seal has published the report; delivery runs after capacity release
/// and never gates reuse.
pub(crate) struct QueryCleanupSealDelivery {
    owner: Option<RuntimeObservationOwner>,
    deferred: Option<DeferredRuntimeObservationSeal>,
    already_sealed: Option<SealedRuntimeObservationSnapshot>,
}

impl QueryCleanupSealDelivery {
    fn already_sealed(snapshot: Option<SealedRuntimeObservationSnapshot>) -> Self {
        Self { owner: None, deferred: None, already_sealed: snapshot }
    }

    /// Runs observer delivery, returning the sealed snapshot.
    pub(crate) fn deliver(self) -> Option<SealedRuntimeObservationSnapshot> {
        if let Some(owner) = self.owner
            && let Some(deferred) = self.deferred
        {
            return owner.complete_observer_delivery(deferred);
        }
        self.already_sealed
    }
}

struct DeferredRuntimeObservationSeal {
    snapshot: Option<SealedRuntimeObservationSnapshot>,
    observers: Vec<RuntimeObservationSealObserver>,
}

impl DeferredRuntimeObservationSeal {
    fn invoke_observers(self) -> Option<SealedRuntimeObservationSnapshot> {
        if let Some(snapshot) = &self.snapshot {
            for observer in self.observers {
                observer.invoke(snapshot.clone());
            }
        }
        self.snapshot
    }
}

trait SourceLocationDigestKeyProvider {
    fn source_location_digest_key(&self) -> Result<SourceLocationDigestKey, SourceLocationDigestKeyUnavailable>;
}

struct OsSourceLocationDigestKeyProvider;

impl SourceLocationDigestKeyProvider for OsSourceLocationDigestKeyProvider {
    fn source_location_digest_key(&self) -> Result<SourceLocationDigestKey, SourceLocationDigestKeyUnavailable> {
        let mut key = [0_u8; blake3::KEY_LEN];
        getrandom::fill(&mut key).map_err(|_| SourceLocationDigestKeyUnavailable)?;
        Ok(SourceLocationDigestKey(key))
    }
}

struct SourceLocationDigestKeyUnavailable;

enum SourceLocationDigestKeyState {
    Available(SourceLocationDigestKey),
    Unavailable,
}

struct SourceLocationDigestKey([u8; blake3::KEY_LEN]);

impl SourceLocationDigestKey {
    fn source_location_id(&self, value: &str) -> RuntimeSourceLocationId {
        let mut hasher = blake3::Hasher::new_keyed(&self.0);
        hasher.update(b"nexus-query-engine/runtime-source-location/v2\0");
        hasher.update(value.as_bytes());
        RuntimeSourceLocationId::from_keyed_digest(format!(
            "source_location:blake3_keyed_256:{}",
            hasher.finalize().to_hex()
        ))
    }
}

impl SourceLocationDigestKeyState {
    fn source_location_id(&self, value: &str) -> Option<RuntimeSourceLocationId> {
        match self {
            Self::Available(key) => Some(key.source_location_id(value)),
            Self::Unavailable => None,
        }
    }

    const fn retained_id_bytes(&self) -> Option<u64> {
        match self {
            Self::Available(_) => Some(("source_location:blake3_keyed_256:".len() + 64) as u64),
            Self::Unavailable => None,
        }
    }
}

#[cfg(test)]
struct TestSourceLocationDigestKeyProvider {
    key: Option<[u8; blake3::KEY_LEN]>,
}

#[cfg(test)]
impl SourceLocationDigestKeyProvider for TestSourceLocationDigestKeyProvider {
    fn source_location_digest_key(&self) -> Result<SourceLocationDigestKey, SourceLocationDigestKeyUnavailable> {
        self.key.map(SourceLocationDigestKey).ok_or(SourceLocationDigestKeyUnavailable)
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

#[derive(Debug, Default)]
struct RuntimeObservationRecorderState {
    next_sequence: u64,
    last_monotonic_ns: u64,
    retained_records: u64,
    retained_bytes: u64,
    topology: Vec<RetainedTopologyRecord>,
    events: Vec<RuntimeTimelineEvent>,
    first_dropped_sequence: Option<ObservationSequence>,
    last_attempted_sequence: Option<ObservationSequence>,
    dropped_by_class: BTreeMap<RuntimeObservationRecordClass, u64>,
    dropped_by_reason: BTreeMap<RuntimeObservationLossReason, u64>,
    degraded: bool,
    poison_recoveries: u64,
    counter_overflow: bool,
    sealed_snapshot: Option<SealedRuntimeObservationSnapshot>,
    seal_observers: Vec<RuntimeObservationSealObserver>,
}

impl RuntimeObservationHandle {
    /// Creates a session using a system monotonic clock. Disabled config allocates no recorder.
    fn new(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
    ) -> Self {
        if !config.is_enabled() {
            return Self { config, inner: None, attempt_cleanup_failure_recorder: None };
        }
        Self::with_clock(query_id, trace_id, config, Arc::new(SystemRuntimeObservationClock::new()))
    }

    /// Creates a session using an injected clock.
    fn with_clock(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        clock: Arc<dyn RuntimeObservationClock>,
    ) -> Self {
        if !config.is_enabled() {
            return Self { config, inner: None, attempt_cleanup_failure_recorder: None };
        }
        Self::with_clock_and_source_location_digest_key(
            query_id,
            trace_id,
            config,
            clock,
            &OsSourceLocationDigestKeyProvider,
        )
    }

    fn with_clock_and_source_location_digest_key(
        query_id: impl Into<RuntimeQueryId>,
        trace_id: Option<RuntimeTraceId>,
        config: RuntimeObservationConfig,
        clock: Arc<dyn RuntimeObservationClock>,
        source_location_digest_key_provider: &dyn SourceLocationDigestKeyProvider,
    ) -> Self {
        if !config.is_enabled() {
            return Self { config, inner: None, attempt_cleanup_failure_recorder: None };
        }
        let manifest = RuntimeObservationManifest {
            schema: RUNTIME_OBSERVATION_JSONL_SCHEMA.to_string(),
            query_id: query_id.into(),
            trace_id,
            unix_epoch_ns: clock.unix_epoch_ns(),
            clock_domain: RuntimeClockDomain::HostMonotonic,
            limits: config.limits(),
        };
        let (source_location_digest_key, state) = match source_location_digest_key_provider.source_location_digest_key()
        {
            Ok(key) => (
                SourceLocationDigestKeyState::Available(key),
                RuntimeObservationRecorderState { next_sequence: 1, ..Default::default() },
            ),
            Err(SourceLocationDigestKeyUnavailable) => {
                let mut dropped_by_reason = BTreeMap::new();
                dropped_by_reason.insert(RuntimeObservationLossReason::SourceLocationKeyUnavailable, 1);
                (
                    SourceLocationDigestKeyState::Unavailable,
                    RuntimeObservationRecorderState {
                        next_sequence: 1,
                        dropped_by_reason,
                        degraded: true,
                        ..Default::default()
                    },
                )
            },
        };
        Self {
            config,
            inner: Some(Arc::new(RuntimeObservationSessionInner {
                manifest,
                source_location_digest_key,
                clock,
                state: Mutex::new(state),
                sealed: AtomicBool::new(false),
            })),
            attempt_cleanup_failure_recorder: None,
        }
    }

    /// Returns a disabled session.
    pub const fn disabled() -> Self {
        Self { config: RuntimeObservationConfig::disabled(), inner: None, attempt_cleanup_failure_recorder: None }
    }

    /// Returns whether this session constructs and records payloads.
    pub const fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Returns the immutable recorder configuration.
    pub const fn config(&self) -> RuntimeObservationConfig {
        self.config
    }

    /// Routes an attempt-owned cleanup failure into the lifecycle before the
    /// caller attempts an optional timeline projection.
    pub(crate) fn record_attempt_cleanup_failure(&self, summary: CleanupFailureSummary) {
        if let Some(recorder) = &self.attempt_cleanup_failure_recorder {
            recorder(summary);
        }
    }

    /// Returns the immutable query identity when recording is enabled.
    pub fn query_id(&self) -> Option<&RuntimeQueryId> {
        self.inner.as_ref().map(|inner| &inner.manifest.query_id)
    }

    /// Returns the immutable optional parent trace identity.
    pub fn trace_id(&self) -> Option<&RuntimeTraceId> {
        self.inner.as_ref().and_then(|inner| inner.manifest.trace_id.as_ref())
    }

    /// Returns a keyed, query-scoped source-location identity when recording is enabled.
    pub fn source_location_id(&self, value: &str) -> Option<RuntimeSourceLocationId> {
        self.inner.as_ref().and_then(|inner| inner.source_location_digest_key.source_location_id(value))
    }

    /// Exact retained byte size of a keyed source-location id without hashing
    /// or allocating the id itself.
    pub fn source_location_id_retained_bytes(&self) -> Option<u64> {
        self.inner.as_ref().and_then(|inner| inner.source_location_digest_key.retained_id_bytes())
    }

    /// Lazily constructs and records a query-level event.
    pub fn record<F>(&self, build_payload: F) -> RuntimeObservationRecordDisposition
    where F: FnOnce() -> RuntimeObservationPayload {
        self.record_with_context(RuntimeEventContext::default(), build_payload)
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
        self.record_with_context(RuntimeEventContext::for_plan(plan_execution_id), build_payload)
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
        let Some(inner) = &self.inner else {
            return RuntimeObservationRecordDisposition::Disabled;
        };
        if inner.sealed.load(Ordering::Acquire) {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        inner.record_event_lazy(context, build_payload)
    }

    /// Records an event after an exact borrowed-data size preflight.
    ///
    /// The payload closure is not invoked when the record or byte cap already
    /// rejects `logical_bytes`, so variable-size strings and row-group vectors
    /// are never cloned merely to discover that capture is exhausted.
    pub fn record_with_context_sized<F>(
        &self,
        context: RuntimeEventContext,
        logical_bytes: u64,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        let Some(inner) = &self.inner else {
            return RuntimeObservationRecordDisposition::Disabled;
        };
        if inner.sealed.load(Ordering::Acquire) {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        inner.record_event_sized(context, logical_bytes, build_payload)
    }

    /// Records an event whose payload depends on its exact recorder timestamp.
    ///
    /// The payload builder runs while recorder order is held and receives the
    /// same query-relative monotonic timestamp retained on the event. The
    /// timestamp is returned even when bounded capture drops the event, and
    /// `None` is returned only when recording is disabled or already sealed.
    pub(crate) fn record_with_context_timestamped<F>(
        &self,
        context: RuntimeEventContext,
        build_payload: F,
    ) -> Option<u64>
    where
        F: FnOnce(u64) -> RuntimeObservationPayload,
    {
        let inner = self.inner.as_ref()?;
        if inner.sealed.load(Ordering::Acquire) {
            return None;
        }
        inner.record_event_timestamped(context, build_payload).1
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
        let Some(inner) = &self.inner else {
            return RuntimeObservationRecordDisposition::Disabled;
        };
        if inner.sealed.load(Ordering::Acquire) {
            return RuntimeObservationRecordDisposition::Sealed;
        }
        inner.record_topology_lazy(class, build_record)
    }

    /// Captures an immutable incomplete view. Returns none after sealing or when disabled.
    pub fn live_snapshot(&self) -> Option<LiveRuntimeObservationSnapshot> {
        let inner = self.inner.as_ref()?;
        let state = inner.lock_state();
        if state.sealed_snapshot.is_some() {
            return None;
        }
        Some(LiveRuntimeObservationSnapshot { data: Arc::new(inner.snapshot_data(&state, false, None)) })
    }

    fn seal_owned(&self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        self.seal_owned_deferred(Some(outcome), None).invoke_observers()
    }

    fn seal_owned_deferred(
        &self,
        outcome: Option<RuntimeTerminalOutcome>,
        attempt_report_provider: Option<&AttemptReportSealProvider>,
    ) -> DeferredRuntimeObservationSeal {
        let Some(inner) = self.inner.as_ref() else {
            return DeferredRuntimeObservationSeal { snapshot: None, observers: Vec::new() };
        };
        let mut state = inner.lock_state();
        if let Some(snapshot) = &state.sealed_snapshot {
            return DeferredRuntimeObservationSeal { snapshot: Some(snapshot.clone()), observers: Vec::new() };
        }
        inner.sealed.store(true, Ordering::Release);
        let snapshot = SealedRuntimeObservationSnapshot {
            data: Arc::new(inner.take_sealed_snapshot_data(&mut state, outcome, attempt_report_provider)),
        };
        state.sealed_snapshot = Some(snapshot.clone());
        let observers = std::mem::take(&mut state.seal_observers);
        DeferredRuntimeObservationSeal { snapshot: Some(snapshot), observers }
    }

    fn mark_degraded(&self) {
        if let Some(inner) = &self.inner {
            inner.lock_state().degraded = true;
        }
    }

    fn register_seal_observer(
        &self,
        observer: RuntimeObservationSealObserver,
    ) -> Option<(RuntimeObservationSealObserver, SealedRuntimeObservationSnapshot)> {
        let inner = self.inner.as_ref()?;
        let mut state = inner.lock_state();
        if let Some(snapshot) = &state.sealed_snapshot {
            Some((observer, snapshot.clone()))
        } else {
            state.seal_observers.push(observer);
            None
        }
    }

    /// Returns the sealed immutable snapshot when a terminal outcome already won.
    pub fn sealed_snapshot(&self) -> Option<SealedRuntimeObservationSnapshot> {
        let inner = self.inner.as_ref()?;
        inner.lock_state().sealed_snapshot.clone()
    }

    /// Returns whether both handles refer to the same recorder session.
    ///
    /// Disabled handles have no recorder identity and therefore never compare
    /// as the same session. This is intended for ownership handoff boundaries
    /// that must reject two independently-created sessions for one external
    /// query instead of silently replacing the first.
    pub fn is_same_session(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            (None, _) | (_, None) => false,
        }
    }

    /// Returns whether a sealed snapshot was produced by this recorder session.
    pub fn owns_snapshot(&self, snapshot: &SealedRuntimeObservationSnapshot) -> bool {
        self.sealed_snapshot().is_some_and(|sealed| sealed.is_same_snapshot(snapshot))
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

    #[cfg(test)]
    pub(crate) fn with_source_location_test_key(
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

    #[cfg(test)]
    pub(crate) fn with_unavailable_source_location_test_key(
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

    fn seal_deferred(
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
    pub(crate) fn with_attempt_report_seal_provider<F>(mut self, provider: F) -> Self
    where F: Fn(&RuntimeObservationFooter) -> Arc<QueryAttemptReport> + Send + Sync + 'static {
        self.attempt_report_seal_provider = Some(Arc::new(provider));
        self
    }

    /// Installs the attempt-owned terminal outcome resolver. It runs after
    /// cleanup evidence is recorded but before the recorder seals, so the
    /// footer and attempt report cannot disagree on the terminal outcome.
    pub(crate) fn with_terminal_outcome_resolver<F>(mut self, resolver: F) -> Self
    where F: Fn(RuntimeTerminalOutcome) -> RuntimeTerminalOutcome + Send + Sync + 'static {
        self.terminal_outcome_resolver = Some(Arc::new(resolver));
        self
    }

    /// Installs the engine-owned attempt cleanup recorder. This internal hook
    /// is only routing: lifecycle state remains the sole owner of the count
    /// and first-summary facts.
    pub(crate) fn with_attempt_cleanup_failure_recorder<F>(mut self, recorder: F) -> Self
    where F: Fn(CleanupFailureSummary) + Send + Sync + 'static {
        self.attempt_cleanup_failure_recorder = Some(Arc::new(recorder));
        self
    }

    /// Blocks ordinary observer delivery after the terminal commit. This is
    /// test-only and scoped to this observation owner.
    #[cfg(test)]
    pub(crate) fn arm_deferred_delivery_gate(&self) -> Arc<RuntimeObservationDeliveryGate> {
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
    pub(crate) fn terminal_outcome_for_cleanup(&self, fallback: RuntimeTerminalOutcome) -> RuntimeTerminalOutcome {
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
    pub(crate) fn begin_query_cleanup(&self) {
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
    pub(crate) fn finish_query_cleanup(
        &self,
        fallback: RuntimeTerminalOutcome,
    ) -> Option<SealedRuntimeObservationSnapshot> {
        self.seal_query_cleanup(fallback).deliver()
    }

    /// Phase one of the cleanup commit (P2.7): records cleanup completion and
    /// seals the recorder state so the report is published, returning the
    /// deferred observer delivery. Capacity release may proceed once this
    /// returns; [`QueryCleanupSealDelivery::deliver`] runs afterwards and
    /// never gates reuse.
    pub(crate) fn seal_query_cleanup(&self, fallback: RuntimeTerminalOutcome) -> QueryCleanupSealDelivery {
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

    fn complete_observer_delivery(
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

    fn lock_lifecycle(&self) -> MutexGuard<'_, RuntimeObservationOwnerLifecycleState> {
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

impl RuntimeObservationSessionInner {
    fn lock_state(&self) -> MutexGuard<'_, RuntimeObservationRecorderState> {
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

    fn record_event_lazy<F>(
        &self,
        context: RuntimeEventContext,
        build_payload: F,
    ) -> RuntimeObservationRecordDisposition
    where
        F: FnOnce() -> RuntimeObservationPayload,
    {
        self.record_event_timestamped(context, |_monotonic_ns| build_payload()).0
    }

    fn record_event_sized<F>(
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

    fn record_event_timestamped<F>(
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

    fn record_topology_lazy<F>(
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

    fn snapshot_data(
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

    fn take_sealed_snapshot_data(
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

impl RuntimeObservationRecorderState {
    fn take_sequence(&mut self) -> ObservationSequence {
        let value = self.next_sequence.max(1);
        self.next_sequence = value.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        let sequence = ObservationSequence::new(value);
        self.last_attempted_sequence = Some(sequence);
        sequence
    }

    fn rejection_reason(
        &self,
        limits: RuntimeObservationLimits,
        logical_bytes: u64,
    ) -> Option<RuntimeObservationLossReason> {
        if self.retained_records >= limits.max_retained_records {
            Some(RuntimeObservationLossReason::RecordLimit)
        } else if logical_bytes > limits.max_retained_bytes {
            Some(RuntimeObservationLossReason::Oversized)
        } else if self.retained_bytes.checked_add(logical_bytes).is_none_or(|sum| sum > limits.max_retained_bytes) {
            Some(RuntimeObservationLossReason::ByteLimit)
        } else {
            None
        }
    }

    fn record_drop(
        &mut self,
        class: RuntimeObservationRecordClass,
        sequence: ObservationSequence,
        reason: RuntimeObservationLossReason,
    ) {
        self.first_dropped_sequence.get_or_insert(sequence);
        let class_count = self.dropped_by_class.get(&class).copied().unwrap_or(0);
        let next_class_count = class_count.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        self.dropped_by_class.insert(class, next_class_count);
        let reason_count = self.dropped_by_reason.get(&reason).copied().unwrap_or(0);
        let next_reason_count = reason_count.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        self.dropped_by_reason.insert(reason, next_reason_count);
    }

    fn admit(
        &mut self,
        limits: RuntimeObservationLimits,
        class: RuntimeObservationRecordClass,
        sequence: ObservationSequence,
        logical_bytes: u64,
    ) -> Result<(), RuntimeObservationLossReason> {
        if let Some(reason) = self.rejection_reason(limits, logical_bytes) {
            self.record_drop(class, sequence, reason);
            return Err(reason);
        }
        self.retained_records = self.retained_records.checked_add(1).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        self.retained_bytes = self.retained_bytes.checked_add(logical_bytes).unwrap_or_else(|| {
            self.counter_overflow = true;
            u64::MAX
        });
        Ok(())
    }
}

#[cfg(test)]
#[path = "recorder_tests.rs"]
mod tests;
