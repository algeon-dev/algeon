//! Single-authority server query lifecycle, observation, and retention registry.
//!
//! Lock ordering is structural: registry methods never call the native engine,
//! render plans, encode JSONL, or await while holding `state`. Finalization
//! first extracts immutable inputs under the registry mutex, performs sealing
//! and encoding outside it, and then conditionally installs the result for the
//! same [`QueryKey`] generation.

#[cfg(test)]
use std::{cell::RefCell, sync::Barrier};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    future::Future,
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use datafusion::physical_plan::ExecutionPlan;
use nexus_query_engine::{
    observability::{
        RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationOwner, RuntimeQueryId,
        RuntimeTerminalOutcome, RuntimeTraceId, write_runtime_observation_jsonl,
    },
    runtime::{ExecutionCancellation, QueryActivityLease},
};
use serde::Serialize;
use tokio::{
    sync::{Notify, futures::Notified},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::observability::{CapturedError, CapturedQuery, QueryStatsPayload, ShutdownCaptureSnapshot, StableValue};
use crate::{
    error::{Error, ErrorCode},
    native_exec::NativeObservationActivityFactory,
    planner::NativePlanningDiagnostics,
};

const RUNTIME_ARTIFACT_CONTROL_SCHEMA: &str = "datafusion_nexus.query_runtime_artifact.control.v1";
const TOMBSTONE_STATS_MAX_BYTES: u64 = 1024;
const QUERY_STATS_HEADER_LINE: &str = "# datafusion_nexus_query_stats=1\n";
type QueryArtifactFrames = (Arc<[u8]>, Option<Arc<[u8]>>);

#[cfg(test)]
type CoordinatorRenderProbe = Box<dyn Fn(&'static str)>;

#[cfg(test)]
thread_local! {
    static COORDINATOR_RENDER_PROBE: RefCell<Option<CoordinatorRenderProbe>> = RefCell::new(None);
}

#[cfg(test)]
fn run_coordinator_render_probe(operation: &'static str) {
    COORDINATOR_RENDER_PROBE.with(|probe| {
        if let Some(probe) = probe.borrow().as_ref() {
            probe(operation);
        }
    });
}

#[cfg(test)]
fn install_coordinator_render_probe(probe: impl Fn(&'static str) + 'static) -> CoordinatorRenderProbeGuard {
    COORDINATOR_RENDER_PROBE.with(|slot| {
        assert!(slot.borrow_mut().replace(Box::new(probe)).is_none(), "render probe already installed");
    });
    CoordinatorRenderProbeGuard
}

#[cfg(test)]
struct CoordinatorRenderProbeGuard;

#[cfg(test)]
impl Drop for CoordinatorRenderProbeGuard {
    fn drop(&mut self) {
        COORDINATOR_RENDER_PROBE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

#[cfg(test)]
struct FinalizationTestGate {
    entered: Barrier,
    release: Barrier,
}

#[cfg(test)]
impl FinalizationTestGate {
    fn new() -> Arc<Self> {
        Arc::new(Self { entered: Barrier::new(2), release: Barrier::new(2) })
    }

    fn wait(&self) {
        self.entered.wait();
        self.release.wait();
    }
}

#[cfg(test)]
impl fmt::Debug for FinalizationTestGate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FinalizationTestGate")
    }
}

#[derive(Serialize)]
struct RuntimeArtifactControl<'a> {
    schema: &'static str,
    version: u32,
    query_id: &'a str,
    state: &'static str,
    retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    #[serde(skip_serializing_if = "is_false")]
    incomplete: bool,
}

const fn is_false(value: &bool) -> bool {
    !*value
}

fn runtime_control(query_id: &str, state: &'static str, retryable: bool, reason: Option<&'static str>) -> Arc<[u8]> {
    Arc::from(
        serde_json::to_vec(&RuntimeArtifactControl {
            schema: RUNTIME_ARTIFACT_CONTROL_SCHEMA,
            version: 1,
            query_id,
            state,
            retryable,
            reason,
            incomplete: false,
        })
        .expect("runtime artifact control contains only infallibly serializable fields"),
    )
}

fn incomplete_runtime_control(query_id: &str, reason: RetainedUnavailableReason) -> Arc<[u8]> {
    Arc::from(
        serde_json::to_vec(&RuntimeArtifactControl {
            schema: RUNTIME_ARTIFACT_CONTROL_SCHEMA,
            version: 1,
            query_id,
            state: "unavailable",
            retryable: false,
            reason: Some(reason.as_str()),
            incomplete: true,
        })
        .expect("runtime artifact control contains only infallibly serializable fields"),
    )
}

fn unavailable_stats_payload(
    query_id: &str,
    outcome: &str,
    reason: RetainedUnavailableReason,
    max_bytes: usize,
) -> QueryStatsPayload {
    let max_bytes = max_bytes.min(usize::try_from(TOMBSTONE_STATS_MAX_BYTES).unwrap_or(usize::MAX));
    let mut bytes = Vec::with_capacity(max_bytes.min(1024));
    for line in [
        QUERY_STATS_HEADER_LINE.to_string(),
        format!("# query_id={}\n", StableValue(query_id)),
        format!("# outcome={outcome}\n"),
        "# sealed=true\n".to_string(),
        "# incomplete=true\n".to_string(),
        format!("# unavailable_reason={}\n", reason.as_str()),
    ] {
        if line.len() > max_bytes.saturating_sub(bytes.len()) {
            break;
        }
        bytes.extend_from_slice(line.as_bytes());
    }
    QueryStatsPayload { bytes: Arc::from(bytes), incomplete: true }
}

struct ComposedTerminalArtifact {
    bundle: Arc<RetainedQueryBundle>,
    stats_incomplete: bool,
}

fn compose_terminal_artifact<F>(
    control: Arc<[u8]>,
    runtime_jsonl: Option<Arc<[u8]>>,
    artifact_max_bytes: u64,
    render_stats: F,
) -> Option<ComposedTerminalArtifact>
where
    F: FnOnce(usize) -> QueryStatsPayload,
{
    let fixed_bytes = u64::try_from(control.len())
        .unwrap_or(u64::MAX)
        .checked_add(runtime_jsonl.as_ref().map_or(0, |jsonl| u64::try_from(jsonl.len()).unwrap_or(u64::MAX)))?;
    let remaining = artifact_max_bytes.checked_sub(fixed_bytes)?;
    let payload = render_stats(usize::try_from(remaining).unwrap_or(usize::MAX));
    let payload_len = payload.bytes.len();
    let max_stats_bytes = usize::try_from(remaining).unwrap_or(usize::MAX);
    let bounded_len = payload_len.min(max_stats_bytes);
    let bounded = &payload.bytes[..bounded_len];
    let keep = if bounded.ends_with(b"\n") {
        bounded.len()
    } else {
        bounded.iter().rposition(|byte| *byte == b'\n').map_or(0, |index| index + 1)
    };
    let stats = if keep == payload_len { payload.bytes } else { Arc::from(&payload.bytes[..keep]) };
    if !stats.starts_with(QUERY_STATS_HEADER_LINE.as_bytes()) {
        return None;
    }
    let stats_incomplete = payload.incomplete || keep != payload_len;
    let charge_bytes = fixed_bytes.checked_add(u64::try_from(stats.len()).unwrap_or(u64::MAX))?;
    debug_assert!(charge_bytes <= artifact_max_bytes);
    Some(ComposedTerminalArtifact {
        bundle: Arc::new(RetainedQueryBundle { stats, runtime_control: control, runtime_jsonl, charge_bytes }),
        stats_incomplete,
    })
}

fn unavailable_tombstone_stats(
    query_id: &str,
    outcome: &str,
    reason: RetainedUnavailableReason,
    artifact_max_bytes: u64,
) -> Option<Arc<[u8]>> {
    let control = incomplete_runtime_control(query_id, reason);
    compose_terminal_artifact(control, None, artifact_max_bytes, |remaining| {
        unavailable_stats_payload(query_id, outcome, reason, remaining)
    })
    .map(|artifact| Arc::clone(&artifact.bundle.stats))
}

struct LimitedVecWriter {
    bytes: Vec<u8>,
    limit: usize,
    overflowed: bool,
}

impl LimitedVecWriter {
    fn new(limit: usize) -> Self {
        Self { bytes: Vec::with_capacity(limit.min(64 * 1024)), limit, overflowed: false }
    }
}

impl io::Write for LimitedVecWriter {
    fn write(&mut self, value: &[u8]) -> io::Result<usize> {
        let remaining = self.limit.saturating_sub(self.bytes.len());
        if value.len() > remaining {
            self.overflowed = true;
            return Err(io::Error::new(io::ErrorKind::WriteZero, "runtime artifact byte limit reached"));
        }
        self.bytes.extend_from_slice(value);
        Ok(value.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

enum ArtifactFrameSnapshot {
    Encode { state: &'static str, retryable: bool, reason: Option<&'static str> },
    Incomplete { reason: RetainedUnavailableReason },
    // Public framing is state-shaped: unavailable is control-only, while a
    // sealed artifact owns exactly one complete JSONL frame.
    Control { control: Arc<[u8]> },
    Sealed { control: Arc<[u8]>, jsonl: Arc<[u8]> },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct QueryKey {
    query_id: Arc<str>,
    generation: u64,
}

impl QueryKey {
    pub(super) fn query_id(&self) -> &str {
        &self.query_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ObservationId(Arc<str>);

#[derive(Debug, Clone)]
pub(super) struct QueryCoordinatorConfig {
    pub(super) max_live_queries: usize,
    pub(super) max_ticketed_queries: usize,
    pub(super) max_active_logical_capture_bytes: u64,
    pub(super) per_query_capture_max_bytes: u64,
    pub(super) per_query_artifact_max_bytes: u64,
    pub(super) retained_max_count: usize,
    pub(super) retained_max_bytes: u64,
    pub(super) retained_ttl: Duration,
    pub(super) tombstone_max_count: usize,
    pub(super) tombstone_ttl: Duration,
    pub(super) shutdown_grace: Duration,
}

impl QueryCoordinatorConfig {
    pub(super) fn from_observability(config: &super::config::ServerObservabilityConfig) -> Self {
        let max_live_queries = 1024;
        Self {
            max_live_queries,
            max_ticketed_queries: 1024,
            max_active_logical_capture_bytes: config.active_logical_capture_max_bytes,
            per_query_capture_max_bytes: config.per_query_max_bytes,
            per_query_artifact_max_bytes: config
                .per_query_max_bytes
                .saturating_mul(2)
                .min(config.global_sealed_max_bytes),
            retained_max_count: config.global_sealed_max_count.max(1),
            retained_max_bytes: config.global_sealed_max_bytes,
            retained_ttl: config.sealed_ttl,
            tombstone_max_count: config.global_sealed_max_count.max(1),
            tombstone_ttl: config.sealed_ttl,
            shutdown_grace: config.shutdown_grace,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QueryCoordinatorErrorKind {
    RegistrationCapacity,
    DuplicateLiveQueryId,
    LogicalCaptureBytes,
    TicketCapacity,
    UnknownQuery,
    StaleGeneration,
    InvalidTransition,
    TicketExpired,
    GenerationExhausted,
    Shutdown,
    ArtifactEgressCapacity,
    ArtifactRenderFailed,
}

impl QueryCoordinatorErrorKind {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::RegistrationCapacity => "registration_capacity",
            Self::DuplicateLiveQueryId => "duplicate_live_query_id",
            Self::LogicalCaptureBytes => "logical_capture_bytes",
            Self::TicketCapacity => "ticket_capacity",
            Self::UnknownQuery => "unknown_query",
            Self::StaleGeneration => "stale_generation",
            Self::InvalidTransition => "invalid_transition",
            Self::TicketExpired => "ticket_expired",
            Self::GenerationExhausted => "generation_exhausted",
            Self::Shutdown => "shutdown",
            Self::ArtifactEgressCapacity => "artifact_egress_capacity",
            Self::ArtifactRenderFailed => "artifact_render_failed",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct QueryCoordinatorError {
    kind: QueryCoordinatorErrorKind,
    message: String,
    query_id: Option<String>,
}

impl QueryCoordinatorError {
    fn new(kind: QueryCoordinatorErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), query_id: None }
    }

    fn with_query_id(mut self, query_id: impl Into<String>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    pub(super) fn query_id(&self) -> Option<&str> {
        self.query_id.as_deref()
    }

    pub(super) const fn kind(&self) -> QueryCoordinatorErrorKind {
        self.kind
    }

    pub(super) fn adapter_error(&self, operation: &'static str) -> Error {
        let code = match self.kind {
            QueryCoordinatorErrorKind::RegistrationCapacity
            | QueryCoordinatorErrorKind::DuplicateLiveQueryId
            | QueryCoordinatorErrorKind::LogicalCaptureBytes
            | QueryCoordinatorErrorKind::TicketCapacity
            | QueryCoordinatorErrorKind::ArtifactEgressCapacity => ErrorCode::ServiceOverloaded,
            QueryCoordinatorErrorKind::UnknownQuery | QueryCoordinatorErrorKind::TicketExpired => {
                ErrorCode::InvalidRequest
            },
            QueryCoordinatorErrorKind::StaleGeneration | QueryCoordinatorErrorKind::InvalidTransition => {
                ErrorCode::Internal
            },
            QueryCoordinatorErrorKind::GenerationExhausted | QueryCoordinatorErrorKind::ArtifactRenderFailed => {
                ErrorCode::Internal
            },
            QueryCoordinatorErrorKind::Shutdown => ErrorCode::Cancelled,
        };
        Error::new(code, operation, self.message.clone())
            .with_log_safe_fact("query_coordinator_error", self.kind.as_str())
    }
}

impl fmt::Display for QueryCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for QueryCoordinatorError {}

#[derive(Debug, Clone)]
pub(super) enum QueryTerminal {
    Succeeded,
    Failed { stage: &'static str, error: Arc<CapturedError> },
    Cancelled { stage: &'static str, error: Arc<CapturedError> },
    Dropped,
    Panicked,
}

impl QueryTerminal {
    fn runtime_outcome(&self) -> RuntimeTerminalOutcome {
        match self {
            Self::Succeeded => RuntimeTerminalOutcome::Succeeded,
            Self::Cancelled { .. } => RuntimeTerminalOutcome::Cancelled,
            Self::Failed { error, .. } => RuntimeTerminalOutcome::Failed { error_code: error.code.clone() },
            Self::Dropped => RuntimeTerminalOutcome::Dropped,
            Self::Panicked => RuntimeTerminalOutcome::Panicked,
        }
    }

    const fn compact_outcome(&self) -> QueryTerminalOutcome {
        match self {
            Self::Succeeded => QueryTerminalOutcome::Succeeded,
            Self::Failed { .. } => QueryTerminalOutcome::Failed,
            Self::Cancelled { .. } => QueryTerminalOutcome::Cancelled,
            Self::Dropped => QueryTerminalOutcome::Dropped,
            Self::Panicked => QueryTerminalOutcome::Panicked,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QueryTerminalOutcome {
    Succeeded,
    Failed,
    Cancelled,
    Dropped,
    Panicked,
    ShutdownTimeout,
}

impl QueryTerminalOutcome {
    const fn stats_outcome(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Dropped | Self::ShutdownTimeout => "dropped",
            Self::Panicked => "panicked",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryPhase {
    Planning,
    Ticketed,
    Executing,
    Finalizing,
    Retained,
    Tombstone,
}

#[derive(Debug, Default)]
pub(super) struct QueryHotCounters {
    output_batches: AtomicUsize,
    output_rows: AtomicUsize,
    output_bytes: AtomicUsize,
    overflow: AtomicBool,
}

impl QueryHotCounters {
    pub(super) fn record_output(&self, rows: usize, bytes: usize) {
        saturating_atomic_add(&self.output_batches, 1, &self.overflow);
        saturating_atomic_add(&self.output_rows, rows, &self.overflow);
        saturating_atomic_add(&self.output_bytes, bytes, &self.overflow);
    }

    fn snapshot(&self) -> QueryHotCounterSnapshot {
        QueryHotCounterSnapshot {
            output_batches: self.output_batches.load(Ordering::Relaxed),
            output_rows: self.output_rows.load(Ordering::Relaxed),
            output_bytes: self.output_bytes.load(Ordering::Relaxed),
            overflow: self.overflow.load(Ordering::Relaxed),
        }
    }
}

fn saturating_atomic_add(counter: &AtomicUsize, delta: usize, overflow: &AtomicBool) {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.checked_add(delta).unwrap_or_else(|| {
            overflow.store(true, Ordering::Relaxed);
            usize::MAX
        });
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryHotCounterSnapshot {
    output_batches: usize,
    output_rows: usize,
    output_bytes: usize,
    overflow: bool,
}

#[derive(Debug, Clone)]
pub(super) struct CoordinatorTicketPayload {
    pub(super) sql: String,
    pub(super) correlation_id: Option<String>,
    pub(super) prepared_handle: Option<String>,
    pub(super) kind: u8,
    pub(super) native_cancellation: ExecutionCancellation,
    pub(super) started_at: Instant,
    pub(super) deadline: Instant,
}

#[derive(Debug, Clone)]
pub(super) struct RetainedQueryBundle {
    pub(super) stats: Arc<[u8]>,
    pub(super) runtime_control: Arc<[u8]>,
    pub(super) runtime_jsonl: Option<Arc<[u8]>>,
    pub(super) charge_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RetainedUnavailableReason {
    CaptureDisabled,
    TooLarge,
    EncodeFailed,
    Evicted,
    TicketExpired,
    ShutdownTimeout,
    FinalizationPanicked,
    CaptureTruncated,
    CounterOverflow,
}

impl RetainedUnavailableReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::CaptureDisabled => "capture_disabled",
            Self::TooLarge => "global_budget",
            Self::EncodeFailed => "encode_failed",
            Self::Evicted => "evicted",
            Self::TicketExpired => "no_native_execution",
            Self::ShutdownTimeout => "shutdown_timeout",
            Self::FinalizationPanicked => "finalization_panicked",
            Self::CaptureTruncated => "capture_budget",
            Self::CounterOverflow => "counter_overflow",
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) enum CoordinatorArtifact {
    Pending,
    Active,
    Finalizing,
    Retained(Arc<RetainedQueryBundle>),
    Unavailable(RetainedUnavailableReason),
}

#[derive(Debug)]
struct QueryRecord {
    key: QueryKey,
    observation_id: ObservationId,
    captured: Option<CapturedQuery>,
    phase: QueryPhase,
    terminal: Option<QueryTerminal>,
    terminal_outcome: Option<QueryTerminalOutcome>,
    active_leases: usize,
    active_activities: usize,
    observation_owner: Option<RuntimeObservationOwner>,
    hot: Arc<QueryHotCounters>,
    logical_capture_bytes: u64,
    artifact_max_bytes: u64,
    capture_version: u64,
    live_stats_cache: Arc<LiveStatsCache>,
    ticket: Option<CoordinatorTicketPayload>,
    execution_cancellation: Option<ExecutionCancellation>,
    retained: Option<Arc<RetainedQueryBundle>>,
    unavailable: Option<RetainedUnavailableReason>,
    retention_deadline: Option<Instant>,
    retention_order: Option<u64>,
    tombstone_deadline: Option<Instant>,
    tombstone_stats: Option<Arc<[u8]>>,
    finalization_claim: Option<u64>,
}

#[derive(Debug)]
struct LiveStatsCache {
    state: Mutex<LiveStatsCacheState>,
    changed: Condvar,
    #[cfg(test)]
    wait_observer: Mutex<Option<std::sync::mpsc::Sender<()>>>,
}

#[derive(Debug)]
struct LiveStatsCacheState {
    open: bool,
    active_renderers: usize,
    waiting_readers: usize,
    rendering_key: Option<(u64, QueryHotCounterSnapshot)>,
    completed_key: Option<(u64, QueryHotCounterSnapshot)>,
    stats: Option<Arc<[u8]>>,
    render_failed: bool,
    terminal_stats: Option<Arc<[u8]>>,
    abandoned: bool,
}

impl LiveStatsCache {
    fn new() -> Self {
        Self {
            state: Mutex::new(LiveStatsCacheState {
                open: true,
                active_renderers: 0,
                waiting_readers: 0,
                rendering_key: None,
                completed_key: None,
                stats: None,
                render_failed: false,
                terminal_stats: None,
                abandoned: false,
            }),
            changed: Condvar::new(),
            #[cfg(test)]
            wait_observer: Mutex::new(None),
        }
    }

    fn begin_read(self: &Arc<Self>, key: (u64, QueryHotCounterSnapshot)) -> LiveStatsRead {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.open {
            if let Some(stats) = &state.terminal_stats {
                return LiveStatsRead::Hit(Arc::clone(stats));
            }
            if state.abandoned {
                return LiveStatsRead::Closed;
            }
            state.waiting_readers = state.waiting_readers.checked_add(1).expect("live stats waiter count overflow");
            return LiveStatsRead::Wait(LiveStatsWaiter { cache: Arc::clone(self), key, completed: false });
        }
        if state.completed_key == Some(key) {
            return if state.render_failed {
                LiveStatsRead::Failed
            } else {
                state.stats.as_ref().map_or(LiveStatsRead::Failed, |stats| LiveStatsRead::Hit(Arc::clone(stats)))
            };
        }
        if state.rendering_key.is_none() {
            state.rendering_key = Some(key);
            state.active_renderers = state.active_renderers.checked_add(1).expect("live stats renderer count overflow");
            return LiveStatsRead::Render(LiveStatsReadLease { cache: Arc::clone(self), key, completed: false });
        }
        state.waiting_readers = state.waiting_readers.checked_add(1).expect("live stats waiter count overflow");
        LiveStatsRead::Wait(LiveStatsWaiter { cache: Arc::clone(self), key, completed: false })
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.open = false;
        state.completed_key = None;
        state.stats = None;
        state.render_failed = false;
        self.changed.notify_all();
    }

    fn wait_drained(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.active_renderers != 0 {
            state = self.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn publish_optional_terminal(&self, stats: Option<Arc<[u8]>>) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(!state.open, "terminal stats require a closed live cache");
        state.terminal_stats = stats;
        state.abandoned = state.terminal_stats.is_none();
        self.changed.notify_all();
    }

    fn abandon(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.open = false;
        state.stats = None;
        if state.terminal_stats.is_none() {
            state.abandoned = true;
        }
        self.changed.notify_all();
    }
}

enum LiveStatsRead {
    Hit(Arc<[u8]>),
    Render(LiveStatsReadLease),
    Wait(LiveStatsWaiter),
    Failed,
    Closed,
}

enum LiveStatsWaitResult {
    Hit(Arc<[u8]>),
    Failed,
    Retry,
    Closed,
}

struct LiveStatsWaiter {
    cache: Arc<LiveStatsCache>,
    key: (u64, QueryHotCounterSnapshot),
    completed: bool,
}

impl LiveStatsWaiter {
    fn wait(mut self) -> LiveStatsWaitResult {
        let mut state = self.cache.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        #[cfg(test)]
        let mut observed_wait = false;
        let result = loop {
            if !state.open {
                if let Some(stats) = &state.terminal_stats {
                    break LiveStatsWaitResult::Hit(Arc::clone(stats));
                }
                if state.abandoned {
                    break LiveStatsWaitResult::Closed;
                }
                #[cfg(test)]
                if !observed_wait {
                    if let Some(observer) =
                        self.cache.wait_observer.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref()
                    {
                        let _ = observer.send(());
                    }
                    observed_wait = true;
                }
                state = self.cache.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
                continue;
            }
            if state.completed_key == Some(self.key) {
                break if state.render_failed {
                    LiveStatsWaitResult::Failed
                } else {
                    state
                        .stats
                        .as_ref()
                        .map_or(LiveStatsWaitResult::Failed, |stats| LiveStatsWaitResult::Hit(Arc::clone(stats)))
                };
            }
            if state.rendering_key.is_none() {
                break LiveStatsWaitResult::Retry;
            }
            #[cfg(test)]
            if !observed_wait {
                if let Some(observer) =
                    self.cache.wait_observer.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref()
                {
                    let _ = observer.send(());
                }
                observed_wait = true;
            }
            state = self.cache.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        };
        state.waiting_readers = state.waiting_readers.checked_sub(1).expect("live stats waiter count underflow");
        self.completed = true;
        result
    }
}

impl Drop for LiveStatsWaiter {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let mut state = self.cache.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.waiting_readers = state.waiting_readers.checked_sub(1).expect("live stats waiter count underflow");
        self.cache.changed.notify_all();
    }
}

struct LiveStatsReadLease {
    cache: Arc<LiveStatsCache>,
    key: (u64, QueryHotCounterSnapshot),
    completed: bool,
}

impl LiveStatsReadLease {
    fn complete(mut self, rendered: Arc<[u8]>) -> Option<Arc<[u8]>> {
        let mut state = self.cache.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.open {
            debug_assert_eq!(state.rendering_key, Some(self.key));
            state.rendering_key = None;
            state.completed_key = Some(self.key);
            state.stats = Some(Arc::clone(&rendered));
            state.render_failed = false;
            state.active_renderers =
                state.active_renderers.checked_sub(1).expect("live stats renderer count underflow");
            self.completed = true;
            self.cache.changed.notify_all();
            return Some(rendered);
        }

        state.rendering_key = None;
        state.active_renderers = state.active_renderers.checked_sub(1).expect("live stats renderer count underflow");
        self.completed = true;
        self.cache.changed.notify_all();
        while state.terminal_stats.is_none() && !state.abandoned {
            state = self.cache.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.terminal_stats.as_ref().map(Arc::clone)
    }
}

impl Drop for LiveStatsReadLease {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let mut state = self.cache.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.open {
            state.completed_key = Some(self.key);
            state.stats = None;
            state.render_failed = true;
        }
        state.rendering_key = None;
        state.active_renderers = state.active_renderers.checked_sub(1).expect("live stats renderer count underflow");
        self.cache.changed.notify_all();
    }
}

#[derive(Debug)]
struct ArtifactEgressLimiter {
    max_streams: usize,
    max_bytes: u64,
    state: Mutex<ArtifactEgressState>,
}

#[derive(Debug, Default)]
struct ArtifactEgressState {
    streams: usize,
    bytes: u64,
    peak_streams: usize,
    peak_bytes: u64,
}

impl ArtifactEgressLimiter {
    fn new(max_streams: usize, max_bytes: u64) -> Arc<Self> {
        Arc::new(Self { max_streams, max_bytes, state: Mutex::new(ArtifactEgressState::default()) })
    }

    fn try_acquire(self: &Arc<Self>, bytes: u64) -> Result<ArtifactEgressLease, QueryCoordinatorError> {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(next_streams) = state.streams.checked_add(1) else {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::ArtifactEgressCapacity,
                "runtime artifact response concurrency accounting overflowed",
            ));
        };
        let Some(next_bytes) = state.bytes.checked_add(bytes) else {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::ArtifactEgressCapacity,
                "runtime artifact response byte accounting overflowed",
            ));
        };
        if next_streams > self.max_streams || next_bytes > self.max_bytes {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::ArtifactEgressCapacity,
                "runtime artifact response capacity is exhausted",
            ));
        }
        state.streams = next_streams;
        state.bytes = next_bytes;
        state.peak_streams = state.peak_streams.max(next_streams);
        state.peak_bytes = state.peak_bytes.max(next_bytes);
        Ok(ArtifactEgressLease { limiter: Arc::clone(self), bytes })
    }
}

#[derive(Debug)]
struct ArtifactEgressLease {
    limiter: Arc<ArtifactEgressLimiter>,
    bytes: u64,
}

impl ArtifactEgressLease {
    fn reconcile(&mut self, bytes: u64) -> Result<(), QueryCoordinatorError> {
        let mut state = self.limiter.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if bytes > self.bytes {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::ArtifactEgressCapacity,
                "runtime artifact response exceeded its configured provisional capacity",
            ));
        }
        state.bytes =
            state.bytes.checked_sub(self.bytes - bytes).expect("artifact egress byte reconciliation is exact");
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for ArtifactEgressLease {
    fn drop(&mut self) {
        let mut state = self.limiter.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.streams = state.streams.checked_sub(1).expect("artifact egress stream release is exact");
        state.bytes = state.bytes.checked_sub(self.bytes).expect("artifact egress byte release is exact");
    }
}

#[derive(Debug)]
pub(super) struct LeasedArtifactResponse {
    frames: std::collections::VecDeque<Bytes>,
    _lease: ArtifactEgressLease,
}

impl LeasedArtifactResponse {
    pub(super) fn next_frame(&mut self) -> Option<Bytes> {
        self.frames.pop_front()
    }
}

#[derive(Debug, Default)]
struct QueryRegistryState {
    next_generation: u64,
    current: HashMap<String, QueryKey>,
    records: BTreeMap<QueryKey, QueryRecord>,
    accepting: bool,
    active_logical_capture_bytes: u64,
    ticketed_count: usize,
    retained_count: usize,
    retained_bytes: u64,
    tombstone_count: usize,
    tombstone_bytes: u64,
    next_retention_order: u64,
    shutdown_deadline: Option<Instant>,
    shutdown_forced: bool,
}

#[derive(Debug)]
pub(super) struct QueryCoordinator {
    config: QueryCoordinatorConfig,
    state: Mutex<QueryRegistryState>,
    artifact_egress: Arc<ArtifactEgressLimiter>,
    notify: Notify,
    #[cfg(test)]
    finalization_gate: Mutex<Option<Arc<FinalizationTestGate>>>,
    #[cfg(test)]
    finalization_installs: AtomicUsize,
}

impl QueryCoordinator {
    pub(super) fn new(config: QueryCoordinatorConfig) -> Arc<Self> {
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

    #[cfg(test)]
    pub(super) fn artifact_egress_usage(&self) -> (usize, u64) {
        let state = self.artifact_egress.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.streams, state.bytes)
    }

    pub(super) fn register_new(
        self: &Arc<Self>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<(String, QueryPlanningLease), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let query_id = Uuid::new_v4().hyphenated().to_string();
        let lease = self
            .register_locked(&mut state, query_id.clone(), sql.into(), correlation_id, trace_id, observation_config)
            .map_err(|error| error.with_query_id(query_id.clone()))?;
        Ok((query_id, lease))
    }

    #[cfg(test)]
    pub(super) fn register_with_id(
        self: &Arc<Self>,
        query_id: impl Into<String>,
        sql: impl Into<String>,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<QueryPlanningLease, QueryCoordinatorError> {
        let query_id = query_id.into();
        let mut state = self.lock_state();
        self.register_locked(&mut state, query_id.clone(), sql.into(), correlation_id, trace_id, observation_config)
            .map_err(|error| error.with_query_id(query_id))
    }

    fn register_locked(
        self: &Arc<Self>,
        state: &mut QueryRegistryState,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        trace_id: Option<String>,
        observation_config: RuntimeObservationConfig,
    ) -> Result<QueryPlanningLease, QueryCoordinatorError> {
        let runtime_capture_bytes = observation_config.limits().max_retained_bytes;
        let logical_capture_bytes = runtime_capture_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(self.config.per_query_capture_max_bytes))
            .and_then(|bytes| bytes.checked_add(self.config.per_query_artifact_max_bytes))
            .ok_or_else(|| {
                QueryCoordinatorError::new(
                    QueryCoordinatorErrorKind::LogicalCaptureBytes,
                    "per-query observation capture reservation overflows u64",
                )
            })?;
        if !state.accepting {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::RegistrationCapacity,
                "query registration is closed",
            ));
        }
        if state.current.get(&query_id).and_then(|key| state.records.get(key)).is_some_and(|record| {
            matches!(
                record.phase,
                QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
            )
        }) {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::DuplicateLiveQueryId,
                "query id is already registered to a live query",
            ));
        }
        let live_count = state
            .records
            .values()
            .filter(|record| {
                matches!(
                    record.phase,
                    QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                )
            })
            .count();
        if live_count >= self.config.max_live_queries {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::RegistrationCapacity,
                "maximum live query count reached",
            ));
        }
        let Some(next_active_capture_bytes) = state.active_logical_capture_bytes.checked_add(logical_capture_bytes)
        else {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::LogicalCaptureBytes,
                "active logical observation capture accounting overflowed",
            ));
        };
        if next_active_capture_bytes > self.config.max_active_logical_capture_bytes {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::LogicalCaptureBytes,
                "active logical observation capture budget exhausted",
            ));
        }
        let generation = state.next_generation.checked_add(1).ok_or_else(|| {
            QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::GenerationExhausted,
                "query generation space exhausted",
            )
        })?;
        state.next_generation = generation;
        let key = QueryKey { query_id: Arc::from(query_id.as_str()), generation };
        let observation_id = ObservationId(Arc::from(format!("{query_id}:{generation}")));
        let owner = RuntimeObservationOwner::new(
            RuntimeQueryId::new(query_id.clone()),
            trace_id.map(RuntimeTraceId::new),
            observation_config,
        );
        let captured = CapturedQuery::new(
            query_id.clone(),
            sql,
            correlation_id,
            Instant::now(),
            self.config.per_query_capture_max_bytes,
            observation_config.limits().max_retained_records,
        );
        let hot = Arc::new(QueryHotCounters::default());
        state.active_logical_capture_bytes = next_active_capture_bytes;
        state.current.insert(query_id.clone(), key.clone());
        state.records.insert(
            key.clone(),
            QueryRecord {
                key: key.clone(),
                observation_id: observation_id.clone(),
                captured: Some(captured),
                phase: QueryPhase::Planning,
                terminal: None,
                terminal_outcome: None,
                active_leases: 1,
                active_activities: 0,
                observation_owner: Some(owner),
                hot: Arc::clone(&hot),
                logical_capture_bytes,
                artifact_max_bytes: self.config.per_query_artifact_max_bytes,
                capture_version: 0,
                live_stats_cache: Arc::new(LiveStatsCache::new()),
                ticket: None,
                execution_cancellation: None,
                retained: None,
                unavailable: None,
                retention_deadline: None,
                retention_order: None,
                tombstone_deadline: None,
                tombstone_stats: None,
                finalization_claim: None,
            },
        );
        Ok(QueryPlanningLease { coordinator: Arc::downgrade(self), key, observation_id, hot, released: false })
    }

    pub(super) fn consume_ticket(
        self: &Arc<Self>,
        query_id: &str,
        now: Instant,
    ) -> Result<(QueryExecutionLease, CoordinatorTicketPayload), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let key = state.current.get(query_id).cloned().ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query ticket")
        })?;
        let (ticket, observation_id, hot, expired) = {
            let record = state.records.get_mut(&key).expect("current query key must reference a record");
            if record.phase != QueryPhase::Ticketed {
                return Err(QueryCoordinatorError::new(
                    QueryCoordinatorErrorKind::InvalidTransition,
                    "query is not ticketed",
                ));
            }
            let ticket = record.ticket.take().expect("ticketed query must own a ticket payload");
            let expired = now >= ticket.deadline;
            if expired {
                if record.terminal_outcome.is_none() {
                    let mut terminal = QueryTerminal::Dropped;
                    Self::apply_terminal_to_capture(record, &mut terminal, now);
                    record.terminal_outcome = Some(terminal.compact_outcome());
                    record.terminal = Some(terminal);
                }
                record.unavailable = Some(RetainedUnavailableReason::TicketExpired);
                record.phase = QueryPhase::Finalizing;
            } else {
                record.phase = QueryPhase::Executing;
                record.active_leases = 1;
                record.execution_cancellation = Some(ticket.native_cancellation.clone());
            }
            (ticket, record.observation_id.clone(), Arc::clone(&record.hot), expired)
        };
        state.ticketed_count = state.ticketed_count.checked_sub(1).expect("consumed ticket count underflow");
        if expired {
            drop(state);
            self.finalize_ready();
            return Err(QueryCoordinatorError::new(QueryCoordinatorErrorKind::TicketExpired, "query ticket expired"));
        }
        let lease =
            QueryExecutionLease { coordinator: Arc::downgrade(self), key, observation_id, hot, released: false };
        Ok((lease, ticket))
    }

    #[cfg(test)]
    pub(super) fn artifact(&self, query_id: &str) -> Option<CoordinatorArtifact> {
        self.finalize_ready();
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        let record = state.records.get(key)?;
        Some(match record.phase {
            QueryPhase::Planning | QueryPhase::Ticketed => CoordinatorArtifact::Pending,
            QueryPhase::Executing => CoordinatorArtifact::Active,
            QueryPhase::Finalizing => CoordinatorArtifact::Finalizing,
            QueryPhase::Retained => match record.unavailable {
                Some(reason) => CoordinatorArtifact::Unavailable(reason),
                None => CoordinatorArtifact::Retained(Arc::clone(
                    record.retained.as_ref().expect("retained phase owns a terminal bundle"),
                )),
            },
            QueryPhase::Tombstone => {
                CoordinatorArtifact::Unavailable(record.unavailable.unwrap_or(RetainedUnavailableReason::Evicted))
            },
        })
    }

    #[cfg(test)]
    pub(super) fn terminal(&self, query_id: &str) -> Option<QueryTerminalOutcome> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        state.records.get(key)?.terminal_outcome
    }

    #[cfg(test)]
    pub(super) fn query_stats(&self, query_id: &str) -> Option<Arc<[u8]>> {
        self.query_stats_result(query_id).ok().flatten()
    }

    fn query_stats_result(&self, query_id: &str) -> Result<Option<Arc<[u8]>>, QueryCoordinatorError> {
        self.finalize_ready();
        loop {
            let (mut captured, hot, artifact_max_bytes, read) = {
                let state = self.lock_state();
                let Some(key) = state.current.get(query_id) else { return Ok(None) };
                let Some(record) = state.records.get(key) else { return Ok(None) };
                if let Some(bundle) = &record.retained {
                    return Ok(Some(Arc::clone(&bundle.stats)));
                }
                if let Some(stats) = &record.tombstone_stats {
                    return Ok(Some(Arc::clone(stats)));
                }
                let hot = record.hot.snapshot();
                let cache_key = (record.capture_version, hot);
                match record.live_stats_cache.begin_read(cache_key) {
                    LiveStatsRead::Hit(stats) => return Ok(Some(stats)),
                    LiveStatsRead::Failed => return Err(Self::artifact_render_failed(query_id)),
                    LiveStatsRead::Closed => return Ok(None),
                    LiveStatsRead::Wait(waiter) => {
                        drop(state);
                        match waiter.wait() {
                            LiveStatsWaitResult::Hit(stats) => return Ok(Some(stats)),
                            LiveStatsWaitResult::Failed => return Err(Self::artifact_render_failed(query_id)),
                            LiveStatsWaitResult::Closed => return Ok(None),
                            LiveStatsWaitResult::Retry => continue,
                        }
                    },
                    LiveStatsRead::Render(read) => {
                        let Some(captured) = record.captured.as_ref().cloned() else {
                            drop(read);
                            return Ok(None);
                        };
                        (captured, hot, record.artifact_max_bytes, read)
                    },
                }
            };
            #[cfg(test)]
            run_coordinator_render_probe("query-stats projection");
            captured.set_output_metrics(hot.output_batches, hot.output_rows, hot.output_bytes, hot.overflow);
            let max_bytes = usize::try_from(artifact_max_bytes).unwrap_or(usize::MAX);
            let stats = captured.to_query_stats_payload(max_bytes).bytes;
            drop(captured);
            return Ok(read.complete(stats));
        }
    }

    fn artifact_render_failed(query_id: &str) -> QueryCoordinatorError {
        QueryCoordinatorError::new(
            QueryCoordinatorErrorKind::ArtifactRenderFailed,
            "live query stats rendering failed before publication",
        )
        .with_query_id(query_id)
    }

    pub(super) fn artifact_frames(&self, query_id: &str) -> Option<QueryArtifactFrames> {
        self.finalize_ready();
        let snapshot = {
            let state = self.lock_state();
            let key = state.current.get(query_id)?;
            let record = state.records.get(key)?;
            match record.phase {
                QueryPhase::Planning | QueryPhase::Ticketed => {
                    ArtifactFrameSnapshot::Encode { state: "pending", retryable: true, reason: None }
                },
                QueryPhase::Executing | QueryPhase::Finalizing => {
                    ArtifactFrameSnapshot::Encode { state: "active", retryable: true, reason: None }
                },
                QueryPhase::Retained => {
                    let bundle = record.retained.as_ref()?;
                    if record.unavailable.is_some() {
                        ArtifactFrameSnapshot::Control { control: Arc::clone(&bundle.runtime_control) }
                    } else {
                        ArtifactFrameSnapshot::Sealed {
                            control: Arc::clone(&bundle.runtime_control),
                            jsonl: bundle.runtime_jsonl.as_ref().map(Arc::clone)?,
                        }
                    }
                },
                QueryPhase::Tombstone => ArtifactFrameSnapshot::Incomplete {
                    reason: record.unavailable.unwrap_or(RetainedUnavailableReason::Evicted),
                },
            }
        };
        match snapshot {
            ArtifactFrameSnapshot::Encode { state, retryable, reason } => {
                #[cfg(test)]
                run_coordinator_render_probe("runtime control encoding");
                Some((runtime_control(query_id, state, retryable, reason), None))
            },
            ArtifactFrameSnapshot::Incomplete { reason } => Some((incomplete_runtime_control(query_id, reason), None)),
            ArtifactFrameSnapshot::Control { control } => Some((control, None)),
            ArtifactFrameSnapshot::Sealed { control, jsonl } => Some((control, Some(jsonl))),
        }
    }

    pub(super) fn lease_query_stats(
        &self,
        query_id: &str,
    ) -> Result<Option<LeasedArtifactResponse>, QueryCoordinatorError> {
        self.finalize_ready();
        let provisional_bytes = {
            let state = self.lock_state();
            let Some(key) = state.current.get(query_id) else { return Ok(None) };
            let Some(record) = state.records.get(key) else { return Ok(None) };
            record.artifact_max_bytes
        };
        let mut lease = self.artifact_egress.try_acquire(provisional_bytes)?;
        let Some(stats) = self.query_stats_result(query_id)? else {
            return Ok(None);
        };
        let actual_bytes = u64::try_from(stats.len()).unwrap_or(u64::MAX);
        lease.reconcile(actual_bytes)?;
        let frames = std::iter::once(stats).map(Bytes::from_owner).collect();
        Ok(Some(LeasedArtifactResponse { frames, _lease: lease }))
    }

    pub(super) fn lease_runtime_artifact(
        &self,
        query_id: &str,
    ) -> Result<Option<LeasedArtifactResponse>, QueryCoordinatorError> {
        let Some((control, jsonl)) = self.artifact_frames(query_id) else {
            return Ok(None);
        };
        let mut payloads = Vec::with_capacity(2);
        payloads.push(control);
        if let Some(jsonl) = jsonl {
            payloads.push(jsonl);
        }
        self.lease_artifact_payloads(payloads)
    }

    fn lease_artifact_payloads(
        &self,
        payloads: impl IntoIterator<Item = Arc<[u8]>>,
    ) -> Result<Option<LeasedArtifactResponse>, QueryCoordinatorError> {
        let payloads = payloads.into_iter().collect::<Vec<_>>();
        if payloads.is_empty() {
            return Ok(None);
        }
        let bytes = payloads
            .iter()
            .try_fold(0u64, |total, payload| total.checked_add(u64::try_from(payload.len()).unwrap_or(u64::MAX)));
        let bytes = bytes.ok_or_else(|| {
            QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::ArtifactEgressCapacity,
                "runtime artifact response size overflowed",
            )
        })?;
        let lease = self.artifact_egress.try_acquire(bytes)?;
        let frames = payloads.into_iter().map(Bytes::from_owner).collect();
        Ok(Some(LeasedArtifactResponse { frames, _lease: lease }))
    }

    pub(super) fn set_sql(&self, query_id: &str, sql: String) {
        let mut state = self.lock_state();
        let Some(key) = state.current.get(query_id).cloned() else { return };
        let Some(record) = state.records.get_mut(&key) else { return };
        if let Some(captured) = &mut record.captured {
            captured.set_sql(sql);
            Self::bump_capture_version(record);
        }
    }

    pub(super) fn record_correlation_id(&self, query_id: &str, correlation_id: Option<&str>) {
        let mut state = self.lock_state();
        let Some(key) = state.current.get(query_id).cloned() else { return };
        let Some(record) = state.records.get_mut(&key) else { return };
        if let Some(captured) = &mut record.captured {
            captured.record_correlation_id(correlation_id);
            Self::bump_capture_version(record);
        }
    }

    pub(super) fn capture_planning(
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

    pub(super) fn capture_planning_failure(
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

    pub(super) fn maintenance_at(&self, now: Instant) {
        let shutdown_due = {
            let mut state = self.lock_state();
            let expired_tickets = state
                .records
                .iter()
                .filter(|(_, record)| {
                    record.phase == QueryPhase::Ticketed
                        && record.ticket.as_ref().is_some_and(|ticket| now >= ticket.deadline)
                })
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            for key in expired_tickets {
                if let Some(record) = state.records.get_mut(&key) {
                    record.ticket = None;
                    if record.terminal_outcome.is_none() {
                        let mut terminal = QueryTerminal::Dropped;
                        Self::apply_terminal_to_capture(record, &mut terminal, now);
                        record.terminal_outcome = Some(terminal.compact_outcome());
                        record.terminal = Some(terminal);
                    }
                    record.unavailable = Some(RetainedUnavailableReason::TicketExpired);
                    record.phase = QueryPhase::Finalizing;
                    state.ticketed_count = state.ticketed_count.checked_sub(1).expect("expired ticket count underflow");
                }
            }
            let expired_retained = state
                .records
                .iter()
                .filter(|(_, record)| {
                    record.phase == QueryPhase::Retained
                        && record.retention_deadline.is_some_and(|deadline| now >= deadline)
                })
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            for key in expired_retained {
                Self::evict_retained_locked(&self.config, &mut state, &key, now);
            }
            let expired_tombstones = state
                .records
                .iter()
                .filter(|(_, record)| {
                    record.phase == QueryPhase::Tombstone
                        && record.tombstone_deadline.is_some_and(|deadline| now >= deadline)
                })
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>();
            for key in expired_tombstones {
                Self::remove_tombstone_locked(&mut state, &key);
            }
            Self::enforce_retention_locked(&self.config, &mut state, now);
            state.shutdown_deadline.is_some_and(|deadline| now >= deadline) && Self::live_count_locked(&state) > 0
        };
        self.finalize_ready();
        if shutdown_due {
            self.force_shutdown_timeout(now);
        }
    }

    pub(super) fn cancel_ticket(&self, query_id: &str, error: CapturedError) -> bool {
        let cancelled = {
            let mut state = self.lock_state();
            let Some(key) = state.current.get(query_id).cloned() else { return false };
            let Some(record) = state.records.get_mut(&key) else { return false };
            if record.phase != QueryPhase::Ticketed {
                return false;
            }
            if let Some(ticket) = record.ticket.take() {
                ticket.native_cancellation.cancel();
            }
            if record.terminal_outcome.is_none() {
                let mut terminal = QueryTerminal::Cancelled { stage: "cancel_query", error: Arc::new(error) };
                Self::apply_terminal_to_capture(record, &mut terminal, Instant::now());
                record.terminal_outcome = Some(terminal.compact_outcome());
                record.terminal = Some(terminal);
            }
            record.phase = QueryPhase::Finalizing;
            state.ticketed_count = state.ticketed_count.checked_sub(1).expect("cancelled ticket count underflow");
            true
        };
        if cancelled {
            self.notify.notify_waiters();
            self.finalize_ready();
        }
        cancelled
    }

    #[cfg(all(test, feature = "iceberg"))]
    pub(super) fn ticket_query_ids(&self) -> Vec<String> {
        let state = self.lock_state();
        state
            .records
            .values()
            .filter(|record| record.phase == QueryPhase::Ticketed)
            .map(|record| record.key.query_id().to_string())
            .collect()
    }

    #[cfg(test)]
    pub(super) fn ticket_correlation_id(&self, query_id: &str) -> Option<Option<String>> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        let record = state.records.get(key)?;
        if record.phase != QueryPhase::Ticketed {
            return None;
        }
        Some(record.ticket.as_ref()?.correlation_id.clone())
    }

    #[cfg(test)]
    pub(super) fn ticket_cancellation(&self, query_id: &str) -> Option<ExecutionCancellation> {
        let state = self.lock_state();
        let key = state.current.get(query_id)?;
        let record = state.records.get(key)?;
        if record.phase != QueryPhase::Ticketed {
            return None;
        }
        Some(record.ticket.as_ref()?.native_cancellation.clone())
    }

    /// Close registration, record a server-shutdown cancellation cause, and
    /// request cancellation for every ticketed or executing native query.
    pub(super) fn begin_shutdown(&self, now: Instant) -> QueryCoordinatorShutdownPlan {
        let shutdown_error = Arc::new(CapturedError::from_adapter_error(&Error::new(
            ErrorCode::Cancelled,
            "datafusion_nexus.server.flight.shutdown",
            "server shutting down; statement aborted",
        )));
        let (live_count, deadline) = {
            let mut state = self.lock_state();
            state.accepting = false;
            let deadline = *state.shutdown_deadline.get_or_insert(checked_deadline(now, self.config.shutdown_grace));
            let mut removed_tickets = 0usize;
            for record in state.records.values_mut() {
                if !matches!(
                    record.phase,
                    QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                ) {
                    continue;
                }
                if let Some(ticket) = record.ticket.take() {
                    ticket.native_cancellation.cancel();
                    removed_tickets = removed_tickets.checked_add(1).expect("removed ticket count overflow");
                }
                if let Some(cancellation) = &record.execution_cancellation {
                    cancellation.cancel();
                }
                if record.terminal_outcome.is_none() {
                    let mut terminal =
                        QueryTerminal::Cancelled { stage: "server_shutdown", error: shutdown_error.clone() };
                    Self::apply_terminal_to_capture(record, &mut terminal, now);
                    record.terminal_outcome = Some(terminal.compact_outcome());
                    record.terminal = Some(terminal);
                }
                if record.phase == QueryPhase::Ticketed {
                    record.phase = QueryPhase::Finalizing;
                }
            }
            state.ticketed_count =
                state.ticketed_count.checked_sub(removed_tickets).expect("shutdown ticket count underflow");
            (Self::live_count_locked(&state), deadline)
        };
        self.notify.notify_waiters();
        QueryCoordinatorShutdownPlan { deadline, live_count }
    }

    pub(super) fn notified(&self) -> Notified<'_> {
        self.notify.notified()
    }

    pub(super) fn next_deadline(&self) -> Option<Instant> {
        let state = self.lock_state();
        let shutdown_deadline = (Self::live_count_locked(&state) > 0).then_some(state.shutdown_deadline).flatten();
        state
            .records
            .values()
            .flat_map(|record| {
                [
                    record.ticket.as_ref().map(|ticket| ticket.deadline),
                    record.retention_deadline,
                    record.tombstone_deadline,
                ]
                .into_iter()
                .flatten()
            })
            .chain(shutdown_deadline)
            .min()
    }

    pub(super) fn is_drained(&self) -> bool {
        Self::live_count_locked(&self.lock_state()) == 0
    }

    #[cfg(test)]
    pub(super) fn shutdown_deadline(&self) -> Option<Instant> {
        self.lock_state().shutdown_deadline
    }

    pub(super) fn shutdown_forced(&self) -> bool {
        self.lock_state().shutdown_forced
    }

    fn force_shutdown_timeout(&self, now: Instant) -> usize {
        let snapshots = {
            let state = self.lock_state();
            state
                .records
                .iter()
                .filter(|(_, record)| {
                    matches!(
                        record.phase,
                        QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                    )
                })
                .map(|(key, record)| ShutdownTimeoutSnapshot {
                    key: key.clone(),
                    captured: record.captured.as_ref().map(CapturedQuery::shutdown_snapshot),
                    hot: record.hot.snapshot(),
                    artifact_max_bytes: record.artifact_max_bytes,
                    live_stats_cache: {
                        let cache = Arc::clone(&record.live_stats_cache);
                        cache.close();
                        cache
                    },
                })
                .collect::<Vec<_>>()
        };
        if snapshots.is_empty() {
            let mut state = self.lock_state();
            state.shutdown_deadline = None;
            return 0;
        }

        let outputs = snapshots
            .into_iter()
            .map(|snapshot| {
                snapshot.live_stats_cache.wait_drained();
                let reason = RetainedUnavailableReason::ShutdownTimeout;
                let control = incomplete_runtime_control(snapshot.key.query_id(), reason);
                let composed = compose_terminal_artifact(control, None, snapshot.artifact_max_bytes, |remaining| {
                    snapshot.captured.as_ref().map_or_else(
                        || unavailable_stats_payload(snapshot.key.query_id(), "dropped", reason, remaining),
                        |captured| {
                            captured.to_query_stats_payload(
                                now,
                                snapshot.hot.output_batches,
                                snapshot.hot.output_rows,
                                snapshot.hot.output_bytes,
                                remaining,
                            )
                        },
                    )
                });
                let bundle = composed.map(|artifact| artifact.bundle);
                let tombstone_stats = bundle
                    .is_none()
                    .then(|| {
                        unavailable_tombstone_stats(
                            snapshot.key.query_id(),
                            "dropped",
                            reason,
                            snapshot.artifact_max_bytes,
                        )
                    })
                    .flatten();
                ShutdownTimeoutOutput {
                    key: snapshot.key,
                    live_stats_cache: snapshot.live_stats_cache,
                    bundle,
                    tombstone_stats,
                }
            })
            .collect::<Vec<_>>();

        let (converted, discarded) = {
            let mut state = self.lock_state();
            let mut converted = 0usize;
            let mut discarded = Vec::new();
            for output in outputs {
                let is_live = state.records.get(&output.key).is_some_and(|record| {
                    matches!(
                        record.phase,
                        QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                    )
                });
                if !is_live {
                    output.live_stats_cache.abandon();
                    continue;
                }
                let charge_bytes = output.bundle.as_ref().map_or(0, |bundle| bundle.charge_bytes);
                let tombstone_bytes =
                    output.tombstone_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
                let terminal_stats = output
                    .bundle
                    .as_ref()
                    .map(|bundle| Arc::clone(&bundle.stats))
                    .or_else(|| output.tombstone_stats.as_ref().map(Arc::clone));
                let retention_order = if output.bundle.is_some() {
                    state.next_retention_order =
                        state.next_retention_order.checked_add(1).expect("retention order overflow");
                    Some(state.next_retention_order)
                } else {
                    None
                };
                let (logical_capture_bytes, was_ticketed, heavy) = {
                    let record = state.records.get_mut(&output.key).expect("live shutdown record remains present");
                    let was_ticketed = record.ticket.is_some();
                    let heavy = ShutdownDiscarded {
                        _captured: record.captured.take(),
                        _observation_owner: record.observation_owner.take(),
                        _terminal: record.terminal.take(),
                        _ticket: record.ticket.take(),
                        _execution_cancellation: record.execution_cancellation.take(),
                    };
                    let logical_capture_bytes = std::mem::take(&mut record.logical_capture_bytes);
                    record.terminal_outcome = Some(QueryTerminalOutcome::ShutdownTimeout);
                    record.active_leases = 0;
                    record.active_activities = 0;
                    record.finalization_claim = None;
                    record.phase = if output.bundle.is_some() { QueryPhase::Retained } else { QueryPhase::Tombstone };
                    record.retained = output.bundle;
                    record.unavailable = Some(RetainedUnavailableReason::ShutdownTimeout);
                    record.retention_deadline =
                        retention_order.map(|_| checked_deadline(now, self.config.retained_ttl));
                    record.retention_order = retention_order;
                    record.tombstone_deadline =
                        retention_order.is_none().then(|| checked_deadline(now, self.config.tombstone_ttl));
                    record.tombstone_stats = output.tombstone_stats;
                    (logical_capture_bytes, was_ticketed, heavy)
                };
                output.live_stats_cache.publish_optional_terminal(terminal_stats);
                if was_ticketed {
                    state.ticketed_count = state.ticketed_count.checked_sub(1).expect("ticketed count underflow");
                }
                state.active_logical_capture_bytes = state
                    .active_logical_capture_bytes
                    .checked_sub(logical_capture_bytes)
                    .expect("forced shutdown releases each active capture reservation exactly once");
                if retention_order.is_some() {
                    state.retained_count = state.retained_count.checked_add(1).expect("retained count overflow");
                    state.retained_bytes =
                        state.retained_bytes.checked_add(charge_bytes).expect("retained bytes overflow");
                } else {
                    state.tombstone_count = state.tombstone_count.checked_add(1).expect("tombstone count overflow");
                    state.tombstone_bytes =
                        state.tombstone_bytes.checked_add(tombstone_bytes).expect("tombstone bytes overflow");
                }
                converted = converted.checked_add(1).expect("converted query count overflow");
                discarded.push(heavy);
            }
            state.shutdown_deadline = None;
            state.shutdown_forced |= converted > 0;
            Self::enforce_retention_locked(&self.config, &mut state, now);
            (converted, discarded)
        };
        drop(discarded);
        self.notify.notify_waiters();
        converted
    }

    fn live_count_locked(state: &QueryRegistryState) -> usize {
        state
            .records
            .values()
            .filter(|record| {
                matches!(
                    record.phase,
                    QueryPhase::Planning | QueryPhase::Ticketed | QueryPhase::Executing | QueryPhase::Finalizing
                )
            })
            .count()
    }

    fn issue_ticket(&self, key: &QueryKey, payload: CoordinatorTicketPayload) -> Result<(), QueryCoordinatorError> {
        let mut state = self.lock_state();
        if state.ticketed_count >= self.config.max_ticketed_queries {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::TicketCapacity,
                "maximum ticketed query count reached",
            ));
        }
        let record = Self::record_mut(&mut state, key)?;
        if record.phase != QueryPhase::Planning || record.active_leases != 1 || record.terminal_outcome.is_some() {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "planning query cannot transition to ticketed",
            ));
        }
        record.phase = QueryPhase::Ticketed;
        record.ticket = Some(payload);
        record.active_leases = 0;
        state.ticketed_count = state.ticketed_count.checked_add(1).expect("ticketed count overflow");
        self.notify.notify_waiters();
        Ok(())
    }

    fn promote_execution(&self, key: &QueryKey) -> Result<(), QueryCoordinatorError> {
        let mut state = self.lock_state();
        let record = Self::record_mut(&mut state, key)?;
        if matches!(record.terminal, Some(QueryTerminal::Cancelled { stage: "server_shutdown", .. })) {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::Shutdown,
                "server shutdown cancelled query before execution began",
            ));
        }
        if record.phase != QueryPhase::Planning || record.active_leases != 1 || record.terminal_outcome.is_some() {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "planning query cannot transition to execution",
            ));
        }
        record.phase = QueryPhase::Executing;
        Ok(())
    }

    fn acquire_activity(
        self: &Arc<Self>,
        key: &QueryKey,
        observation_id: &ObservationId,
    ) -> Result<ObservationActivityLease, QueryCoordinatorError> {
        let mut state = self.lock_state();
        let record = Self::record_mut(&mut state, key)?;
        if &record.observation_id != observation_id {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::StaleGeneration,
                "observation identity does not match the query generation",
            ));
        }
        if record.terminal_outcome.is_some()
            || matches!(record.phase, QueryPhase::Finalizing | QueryPhase::Retained | QueryPhase::Tombstone)
        {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "terminal query cannot acquire observation activity",
            ));
        }
        record.active_activities = record.active_activities.checked_add(1).ok_or_else(|| {
            QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::InvalidTransition,
                "query observation activity count overflowed",
            )
        })?;
        Ok(ObservationActivityLease { coordinator: Arc::downgrade(self), key: key.clone(), released: false })
    }

    fn observation_handle(
        &self,
        key: &QueryKey,
        observation_id: &ObservationId,
    ) -> Result<RuntimeObservationHandle, QueryCoordinatorError> {
        let state = self.lock_state();
        let record = state.records.get(key).ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query generation")
        })?;
        if &record.observation_id != observation_id {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::StaleGeneration,
                "observation identity does not match the query generation",
            ));
        }
        Ok(record
            .observation_owner
            .as_ref()
            .map_or_else(RuntimeObservationHandle::disabled, RuntimeObservationOwner::handle))
    }

    fn set_terminal(&self, key: &QueryKey, mut terminal: QueryTerminal) {
        let should_finalize = {
            let mut state = self.lock_state();
            let Ok(record) = Self::record_mut(&mut state, key) else { return };
            if record.terminal_outcome.is_none() {
                Self::apply_terminal_to_capture(record, &mut terminal, Instant::now());
                record.terminal_outcome = Some(terminal.compact_outcome());
                record.terminal = Some(terminal);
                Self::bump_capture_version(record);
            }
            if record.active_leases > 0 {
                record.active_leases -= 1;
            }
            record.active_leases == 0
                && record.active_activities == 0
                && !matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone)
        };
        self.notify.notify_waiters();
        if should_finalize {
            self.finalize_ready();
        }
    }

    fn apply_terminal_to_capture(record: &mut QueryRecord, terminal: &mut QueryTerminal, now: Instant) {
        match terminal {
            QueryTerminal::Succeeded => {
                if let Some(captured) = &mut record.captured {
                    captured.finish_success(now);
                }
            },
            QueryTerminal::Failed { stage, error } | QueryTerminal::Cancelled { stage, error } => {
                if let Some(captured) = &mut record.captured {
                    captured.bound_error(Arc::make_mut(error));
                    captured.finish_failed(stage, Arc::clone(error), now);
                }
            },
            QueryTerminal::Dropped | QueryTerminal::Panicked => {
                if let Some(captured) = &mut record.captured {
                    captured.finish_dropped(now);
                }
            },
        }
    }

    fn release_activity(&self, key: &QueryKey) {
        let should_finalize = {
            let mut state = self.lock_state();
            let Ok(record) = Self::record_mut(&mut state, key) else { return };
            if matches!(record.phase, QueryPhase::Retained | QueryPhase::Tombstone) {
                return;
            }
            if record.active_activities > 0 {
                record.active_activities -= 1;
            }
            record.terminal_outcome.is_some() && record.active_leases == 0 && record.active_activities == 0
        };
        self.notify.notify_waiters();
        if should_finalize {
            self.finalize_ready();
        }
    }

    fn finalize_ready(&self) {
        loop {
            let Some(inputs) = self.take_finalization_inputs() else { break };
            let failure = FinalizationFailureContext::from_inputs(&inputs);
            let output = catch_unwind(AssertUnwindSafe(|| Self::finalize_outside_lock(inputs)))
                .unwrap_or_else(|_| FinalizationOutput::panicked(failure));
            self.install_finalization(output, Instant::now());
        }
    }

    fn take_finalization_inputs(&self) -> Option<FinalizationInputs> {
        #[cfg(test)]
        let finalization_gate =
            self.finalization_gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let mut state = self.lock_state();
        let key = state.records.iter().find_map(|(key, record)| {
            (record.terminal_outcome.is_some()
                && record.active_leases == 0
                && record.active_activities == 0
                && record.finalization_claim.is_none()
                && matches!(record.phase, QueryPhase::Planning | QueryPhase::Executing | QueryPhase::Finalizing))
            .then(|| key.clone())
        })?;
        let record = state.records.get_mut(&key).expect("selected finalization record remains present");
        record.phase = QueryPhase::Finalizing;
        let claim = key.generation;
        record.finalization_claim = Some(claim);
        let terminal = record.terminal.take().expect("finalization requires a terminal payload");
        let terminal_outcome = terminal.compact_outcome();
        debug_assert_eq!(record.terminal_outcome, Some(terminal_outcome));
        let runtime_outcome = terminal.runtime_outcome();
        drop(terminal);
        let live_stats_cache = Arc::clone(&record.live_stats_cache);
        live_stats_cache.close();
        let inputs = FinalizationInputs {
            key,
            claim,
            terminal_outcome,
            runtime_outcome,
            captured: record.captured.take().expect("finalizing live query retains captured stats inputs"),
            observation_owner: record.observation_owner.take(),
            live_stats_cache: Arc::clone(&live_stats_cache),
            hot: record.hot.snapshot(),
            artifact_max_bytes: record.artifact_max_bytes,
            preselected_unavailable: record.unavailable,
            #[cfg(test)]
            finalization_gate,
        };
        drop(state);
        live_stats_cache.wait_drained();
        Some(inputs)
    }

    fn finalize_outside_lock(mut inputs: FinalizationInputs) -> FinalizationOutput {
        #[cfg(test)]
        if let Some(gate) = &inputs.finalization_gate {
            gate.wait();
        }
        #[cfg(test)]
        run_coordinator_render_probe("finalization projection");
        inputs.captured.set_output_metrics(
            inputs.hot.output_batches,
            inputs.hot.output_rows,
            inputs.hot.output_bytes,
            inputs.hot.overflow,
        );
        inputs.captured.seal(Instant::now());
        let stats_outcome = inputs.terminal_outcome.stats_outcome();
        let Some(owner) = inputs.observation_owner else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::CaptureDisabled,
                inputs.artifact_max_bytes,
            );
        };
        let Some(snapshot) = owner.finish_query(inputs.runtime_outcome) else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::CaptureDisabled,
                inputs.artifact_max_bytes,
            );
        };
        let mut incomplete_reason = snapshot
            .footer()
            .counter_overflow
            .then_some(RetainedUnavailableReason::CounterOverflow)
            .or_else(|| inputs.hot.overflow.then_some(RetainedUnavailableReason::CounterOverflow))
            .or(inputs.preselected_unavailable)
            .or_else(|| {
                (snapshot.footer().truncated || snapshot.footer().degraded)
                    .then_some(RetainedUnavailableReason::CaptureTruncated)
            });
        let control = incomplete_reason.map_or_else(
            || runtime_control(inputs.key.query_id(), "sealed", false, None),
            |reason| incomplete_runtime_control(inputs.key.query_id(), reason),
        );
        let Some(jsonl_limit) = inputs.artifact_max_bytes.checked_sub(u64::try_from(control.len()).unwrap_or(u64::MAX))
        else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::TooLarge,
                inputs.artifact_max_bytes,
            );
        };
        let mut jsonl = LimitedVecWriter::new(usize::try_from(jsonl_limit).unwrap_or(usize::MAX));
        if write_runtime_observation_jsonl(&snapshot, &mut jsonl).is_err() {
            let reason = if jsonl.overflowed {
                RetainedUnavailableReason::TooLarge
            } else {
                RetainedUnavailableReason::EncodeFailed
            };
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                reason,
                inputs.artifact_max_bytes,
            );
        }
        let jsonl = Arc::<[u8]>::from(jsonl.bytes);
        let Some(mut composed) =
            compose_terminal_artifact(control, Some(Arc::clone(&jsonl)), inputs.artifact_max_bytes, |remaining| {
                inputs.captured.to_query_stats_payload(remaining)
            })
        else {
            return FinalizationOutput::unavailable(
                inputs.key,
                inputs.claim,
                inputs.live_stats_cache,
                stats_outcome,
                RetainedUnavailableReason::TooLarge,
                inputs.artifact_max_bytes,
            );
        };
        if incomplete_reason.is_none() && composed.stats_incomplete {
            incomplete_reason = Some(RetainedUnavailableReason::CaptureTruncated);
            let control =
                incomplete_runtime_control(inputs.key.query_id(), RetainedUnavailableReason::CaptureTruncated);
            let Some(recomposed) =
                compose_terminal_artifact(control, Some(jsonl), inputs.artifact_max_bytes, |remaining| {
                    inputs.captured.to_query_stats_payload(remaining)
                })
            else {
                return FinalizationOutput::unavailable(
                    inputs.key,
                    inputs.claim,
                    inputs.live_stats_cache,
                    stats_outcome,
                    RetainedUnavailableReason::TooLarge,
                    inputs.artifact_max_bytes,
                );
            };
            composed = recomposed;
        }
        FinalizationOutput {
            key: inputs.key,
            claim: inputs.claim,
            live_stats_cache: inputs.live_stats_cache,
            bundle: Some(composed.bundle),
            unavailable: incomplete_reason,
            tombstone_stats: None,
        }
    }

    fn install_finalization(&self, mut output: FinalizationOutput, now: Instant) {
        let mut state = self.lock_state();
        let Some(record) = state.records.get(&output.key) else {
            drop(state);
            output.live_stats_cache.abandon();
            return;
        };
        if record.phase != QueryPhase::Finalizing
            || record.key != output.key
            || record.finalization_claim != Some(output.claim)
        {
            drop(state);
            output.live_stats_cache.abandon();
            return;
        }
        #[cfg(test)]
        self.finalization_installs.fetch_add(1, Ordering::SeqCst);
        let logical_capture_bytes = record.logical_capture_bytes;
        let terminal_outcome = record.terminal_outcome.map_or("unknown", QueryTerminalOutcome::stats_outcome);
        state.active_logical_capture_bytes = state
            .active_logical_capture_bytes
            .checked_sub(logical_capture_bytes)
            .expect("live finalization releases its active capture reservation exactly once");
        {
            let record = state.records.get_mut(&output.key).expect("validated finalization record remains present");
            record.logical_capture_bytes = 0;
            record.captured = None;
            record.execution_cancellation = None;
            record.finalization_claim = None;
        }
        let charge_bytes = output.bundle.as_ref().map_or(0, |bundle| bundle.charge_bytes);
        if charge_bytes > self.config.retained_max_bytes {
            output.unavailable = Some(RetainedUnavailableReason::TooLarge);
            output.bundle = None;
            output.tombstone_stats = unavailable_tombstone_stats(
                output.key.query_id(),
                terminal_outcome,
                RetainedUnavailableReason::TooLarge,
                self.config.per_query_artifact_max_bytes,
            );
        }
        let terminal_stats = match output.bundle.take() {
            None => {
                let reason = output.unavailable.unwrap_or(RetainedUnavailableReason::TooLarge);
                let (tombstone_bytes, terminal_stats) = {
                    let record =
                        state.records.get_mut(&output.key).expect("validated finalization record remains present");
                    record.phase = QueryPhase::Tombstone;
                    record.unavailable = Some(reason);
                    record.retention_deadline = None;
                    record.retention_order = None;
                    record.tombstone_deadline = Some(checked_deadline(now, self.config.tombstone_ttl));
                    record.retained = None;
                    record.tombstone_stats = output.tombstone_stats.take().or_else(|| {
                        unavailable_tombstone_stats(
                            output.key.query_id(),
                            terminal_outcome,
                            reason,
                            self.config.per_query_artifact_max_bytes,
                        )
                    });
                    let terminal_stats = record.tombstone_stats.as_ref().map(Arc::clone);
                    let tombstone_bytes =
                        terminal_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
                    (tombstone_bytes, terminal_stats)
                };
                state.tombstone_count = state.tombstone_count.checked_add(1).expect("tombstone count overflow");
                state.tombstone_bytes =
                    state.tombstone_bytes.checked_add(tombstone_bytes).expect("tombstone bytes overflow");
                terminal_stats
            },
            Some(bundle) => {
                state.next_retention_order =
                    state.next_retention_order.checked_add(1).expect("retention order overflow");
                let retention_order = state.next_retention_order;
                let record = state.records.get_mut(&output.key).expect("validated finalization record remains present");
                record.phase = QueryPhase::Retained;
                record.retention_deadline = Some(checked_deadline(now, self.config.retained_ttl));
                record.retention_order = Some(retention_order);
                record.tombstone_deadline = None;
                let terminal_stats = Arc::clone(&bundle.stats);
                record.retained = Some(bundle);
                record.unavailable = output.unavailable;
                record.tombstone_stats = None;
                state.retained_count = state.retained_count.checked_add(1).expect("retained count overflow");
                state.retained_bytes = state.retained_bytes.checked_add(charge_bytes).expect("retained bytes overflow");
                Some(terminal_stats)
            },
        };
        Self::enforce_retention_locked(&self.config, &mut state, now);
        if Self::live_count_locked(&state) == 0 {
            state.shutdown_deadline = None;
        }
        drop(state);
        output.live_stats_cache.publish_optional_terminal(terminal_stats);
        self.notify.notify_waiters();
    }

    fn enforce_retention_locked(config: &QueryCoordinatorConfig, state: &mut QueryRegistryState, now: Instant) {
        while state.retained_count > config.retained_max_count || state.retained_bytes > config.retained_max_bytes {
            let victim = state
                .records
                .iter()
                .filter_map(|(key, record)| {
                    (record.phase == QueryPhase::Retained)
                        .then_some((record.retention_order.unwrap_or(u64::MAX), key.clone()))
                })
                .min()
                .map(|(_, key)| key)
                .expect("retained pressure requires a retained victim");
            Self::evict_retained_locked(config, state, &victim, now);
        }
        let tombstone_max_bytes =
            u64::try_from(config.tombstone_max_count).unwrap_or(u64::MAX).saturating_mul(TOMBSTONE_STATS_MAX_BYTES);
        while state.tombstone_count > config.tombstone_max_count || state.tombstone_bytes > tombstone_max_bytes {
            let victim = state
                .records
                .iter()
                .filter_map(|(key, record)| {
                    (record.phase == QueryPhase::Tombstone)
                        .then_some((record.tombstone_deadline.unwrap_or(now), key.clone()))
                })
                .min()
                .map(|(_, key)| key)
                .expect("tombstone pressure requires a tombstone victim");
            Self::remove_tombstone_locked(state, &victim);
        }
    }

    fn remove_tombstone_locked(state: &mut QueryRegistryState, key: &QueryKey) -> bool {
        let Some(record) = state.records.get(key) else { return false };
        if record.phase != QueryPhase::Tombstone {
            return false;
        }
        let bytes = record.tombstone_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
        state.records.remove(key);
        state.tombstone_count = state.tombstone_count.checked_sub(1).expect("tombstone count underflow");
        state.tombstone_bytes = state.tombstone_bytes.checked_sub(bytes).expect("tombstone bytes underflow");
        if state.current.get(key.query_id()).is_some_and(|current| current == key) {
            state.current.remove(key.query_id());
        }
        true
    }

    fn evict_retained_locked(
        config: &QueryCoordinatorConfig,
        state: &mut QueryRegistryState,
        key: &QueryKey,
        now: Instant,
    ) {
        let Some(record) = state.records.get_mut(key) else { return };
        if record.phase != QueryPhase::Retained {
            return;
        }
        let terminal_outcome = record.terminal_outcome.map_or("unknown", QueryTerminalOutcome::stats_outcome);
        if let Some(bundle) = record.retained.take() {
            state.retained_bytes = state
                .retained_bytes
                .checked_sub(bundle.charge_bytes)
                .expect("retained eviction releases an installed bundle charge exactly once");
        }
        state.retained_count = state.retained_count.checked_sub(1).expect("retained count underflow");
        state.tombstone_count = state.tombstone_count.checked_add(1).expect("tombstone count overflow");
        record.phase = QueryPhase::Tombstone;
        record.unavailable = Some(RetainedUnavailableReason::Evicted);
        record.tombstone_stats = unavailable_tombstone_stats(
            key.query_id(),
            terminal_outcome,
            RetainedUnavailableReason::Evicted,
            config.per_query_artifact_max_bytes,
        );
        let tombstone_bytes =
            record.tombstone_stats.as_ref().map_or(0, |stats| u64::try_from(stats.len()).unwrap_or(u64::MAX));
        record.live_stats_cache.publish_optional_terminal(record.tombstone_stats.as_ref().map(Arc::clone));
        state.tombstone_bytes = state.tombstone_bytes.checked_add(tombstone_bytes).expect("tombstone bytes overflow");
        record.retention_deadline = None;
        record.retention_order = None;
        record.tombstone_deadline = Some(checked_deadline(now, config.tombstone_ttl));
    }

    fn record_mut<'a>(
        state: &'a mut QueryRegistryState,
        key: &QueryKey,
    ) -> Result<&'a mut QueryRecord, QueryCoordinatorError> {
        let current = state.current.get(key.query_id()).ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query generation")
        })?;
        if current != key {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::StaleGeneration,
                "stale query generation",
            ));
        }
        state.records.get_mut(key).ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "unknown query generation")
        })
    }

    fn bump_capture_version(record: &mut QueryRecord) {
        record.capture_version = record.capture_version.checked_add(1).unwrap_or_else(|| {
            if let Some(captured) = &mut record.captured {
                captured.mark_counter_overflow();
            }
            u64::MAX
        });
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, QueryRegistryState> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

trait CoordinatorMaintenanceClock: Send + Sync + 'static {
    fn now(&self) -> Instant;

    fn sleep_until(&self, deadline: Instant) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

struct TokioCoordinatorMaintenanceClock;

impl CoordinatorMaintenanceClock for TokioCoordinatorMaintenanceClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep_until(&self, deadline: Instant) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)))
    }
}

/// The one server-owned task that advances ticket/retention/shutdown deadlines.
pub(super) struct QueryCoordinatorMaintenance {
    coordinator: Arc<QueryCoordinator>,
    clock: Arc<dyn CoordinatorMaintenanceClock>,
    stop: CancellationToken,
    task: JoinHandle<()>,
}

impl QueryCoordinatorMaintenance {
    pub(super) fn start(coordinator: Arc<QueryCoordinator>) -> Self {
        Self::start_with_clock(coordinator, Arc::new(TokioCoordinatorMaintenanceClock))
    }

    fn start_with_clock(coordinator: Arc<QueryCoordinator>, clock: Arc<dyn CoordinatorMaintenanceClock>) -> Self {
        let stop = CancellationToken::new();
        let task =
            tokio::spawn(run_coordinator_maintenance(Arc::clone(&coordinator), Arc::clone(&clock), stop.clone()));
        Self { coordinator, clock, stop, task }
    }

    /// Join only after graceful finalization or the configured forced,
    /// incomplete shutdown conversion has made the live registry empty.
    pub(super) async fn finish(
        mut self,
        plan: QueryCoordinatorShutdownPlan,
    ) -> std::result::Result<bool, tokio::task::JoinError> {
        let mut deadline_expired = false;
        loop {
            let now = self.clock.now();
            if now >= plan.deadline() {
                self.coordinator.force_shutdown_timeout(now);
                deadline_expired = true;
                break;
            }
            if self.coordinator.is_drained() {
                break;
            }
            let mut notified = Box::pin(self.coordinator.notified());
            notified.as_mut().enable();
            if self.coordinator.is_drained() {
                let now = self.clock.now();
                if now >= plan.deadline() {
                    self.coordinator.force_shutdown_timeout(now);
                    deadline_expired = true;
                }
                break;
            }
            tokio::select! {
                () = &mut notified => {},
                () = self.clock.sleep_until(plan.deadline()) => {
                    self.coordinator.force_shutdown_timeout(self.clock.now());
                    deadline_expired = true;
                    break;
                },
            }
        }
        let forced = self.coordinator.shutdown_forced();
        self.stop.cancel();
        (&mut self.task).await?;
        Ok(forced || deadline_expired || self.clock.now() >= plan.deadline())
    }
}

/// Immutable coordinator shutdown handoff captured before mutable coordinator
/// state can be finalized or discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct QueryCoordinatorShutdownPlan {
    deadline: Instant,
    live_count: usize,
}

impl QueryCoordinatorShutdownPlan {
    pub(super) fn deadline(self) -> Instant {
        self.deadline
    }

    pub(super) fn live_count(self) -> usize {
        self.live_count
    }
}

impl Drop for QueryCoordinatorMaintenance {
    fn drop(&mut self) {
        self.stop.cancel();
        self.task.abort();
    }
}

async fn run_coordinator_maintenance(
    coordinator: Arc<QueryCoordinator>,
    clock: Arc<dyn CoordinatorMaintenanceClock>,
    stop: CancellationToken,
) {
    loop {
        let mut notified = Box::pin(coordinator.notified());
        notified.as_mut().enable();
        coordinator.maintenance_at(clock.now());
        if stop.is_cancelled() {
            return;
        }
        match coordinator.next_deadline() {
            Some(deadline) => {
                tokio::select! {
                    () = stop.cancelled() => return,
                    () = &mut notified => {},
                    () = clock.sleep_until(deadline) => {},
                }
            },
            None => {
                tokio::select! {
                    () = stop.cancelled() => return,
                    () = &mut notified => {},
                }
            },
        }
    }
}

struct FinalizationInputs {
    key: QueryKey,
    claim: u64,
    terminal_outcome: QueryTerminalOutcome,
    runtime_outcome: RuntimeTerminalOutcome,
    captured: CapturedQuery,
    observation_owner: Option<RuntimeObservationOwner>,
    live_stats_cache: Arc<LiveStatsCache>,
    hot: QueryHotCounterSnapshot,
    artifact_max_bytes: u64,
    preselected_unavailable: Option<RetainedUnavailableReason>,
    #[cfg(test)]
    finalization_gate: Option<Arc<FinalizationTestGate>>,
}

struct FinalizationFailureContext {
    key: QueryKey,
    claim: u64,
    live_stats_cache: Arc<LiveStatsCache>,
    outcome: &'static str,
    artifact_max_bytes: u64,
}

impl FinalizationFailureContext {
    fn from_inputs(inputs: &FinalizationInputs) -> Self {
        let outcome = inputs.terminal_outcome.stats_outcome();
        Self {
            key: inputs.key.clone(),
            claim: inputs.claim,
            live_stats_cache: Arc::clone(&inputs.live_stats_cache),
            outcome,
            artifact_max_bytes: inputs.artifact_max_bytes,
        }
    }
}

struct ShutdownTimeoutSnapshot {
    key: QueryKey,
    captured: Option<ShutdownCaptureSnapshot>,
    hot: QueryHotCounterSnapshot,
    artifact_max_bytes: u64,
    live_stats_cache: Arc<LiveStatsCache>,
}

struct ShutdownTimeoutOutput {
    key: QueryKey,
    live_stats_cache: Arc<LiveStatsCache>,
    bundle: Option<Arc<RetainedQueryBundle>>,
    tombstone_stats: Option<Arc<[u8]>>,
}

struct ShutdownDiscarded {
    _captured: Option<CapturedQuery>,
    _observation_owner: Option<RuntimeObservationOwner>,
    _terminal: Option<QueryTerminal>,
    _ticket: Option<CoordinatorTicketPayload>,
    _execution_cancellation: Option<ExecutionCancellation>,
}

struct FinalizationOutput {
    key: QueryKey,
    claim: u64,
    live_stats_cache: Arc<LiveStatsCache>,
    bundle: Option<Arc<RetainedQueryBundle>>,
    unavailable: Option<RetainedUnavailableReason>,
    tombstone_stats: Option<Arc<[u8]>>,
}

impl FinalizationOutput {
    fn unavailable(
        key: QueryKey,
        claim: u64,
        live_stats_cache: Arc<LiveStatsCache>,
        outcome: &'static str,
        reason: RetainedUnavailableReason,
        artifact_max_bytes: u64,
    ) -> Self {
        let control = incomplete_runtime_control(key.query_id(), reason);
        let composed = compose_terminal_artifact(control, None, artifact_max_bytes, |remaining| {
            unavailable_stats_payload(key.query_id(), outcome, reason, remaining)
        });
        let bundle = composed.map(|artifact| artifact.bundle);
        let tombstone_stats = bundle
            .is_none()
            .then(|| unavailable_tombstone_stats(key.query_id(), outcome, reason, artifact_max_bytes))
            .flatten();
        Self { key, claim, live_stats_cache, bundle, unavailable: Some(reason), tombstone_stats }
    }

    fn panicked(context: FinalizationFailureContext) -> Self {
        Self::unavailable(
            context.key,
            context.claim,
            context.live_stats_cache,
            context.outcome,
            RetainedUnavailableReason::FinalizationPanicked,
            context.artifact_max_bytes,
        )
    }
}

fn checked_deadline(now: Instant, duration: Duration) -> Instant {
    now.checked_add(duration).unwrap_or(now)
}

#[derive(Debug)]
pub(super) struct QueryPlanningLease {
    coordinator: Weak<QueryCoordinator>,
    key: QueryKey,
    observation_id: ObservationId,
    hot: Arc<QueryHotCounters>,
    released: bool,
}

impl QueryPlanningLease {
    pub(super) fn key(&self) -> &QueryKey {
        &self.key
    }

    #[cfg(test)]
    pub(super) fn observation_id(&self) -> &ObservationId {
        &self.observation_id
    }

    #[cfg(test)]
    pub(super) fn hot_counters(&self) -> Arc<QueryHotCounters> {
        Arc::clone(&self.hot)
    }

    pub(super) fn observation_handle(&self) -> Result<RuntimeObservationHandle, QueryCoordinatorError> {
        self.coordinator
            .upgrade()
            .ok_or_else(|| QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped"))?
            .observation_handle(&self.key, &self.observation_id)
    }

    pub(super) fn activity_factory(&self) -> NativeObservationActivityFactory {
        observation_activity_factory(self.coordinator.clone(), self.key.clone(), self.observation_id.clone())
    }

    pub(super) fn issue_ticket(mut self, payload: CoordinatorTicketPayload) -> Result<QueryKey, QueryCoordinatorError> {
        let coordinator = self.coordinator.upgrade().ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped")
        })?;
        if let Err(error) = coordinator.issue_ticket(&self.key, payload) {
            let captured = CapturedError::from_adapter_error(
                &error.adapter_error("datafusion_nexus.server.query_coordinator.issue_ticket"),
            );
            self.finish(QueryTerminal::Failed { stage: "ticket", error: Arc::new(captured) });
            return Err(error);
        }
        self.released = true;
        Ok(self.key.clone())
    }

    pub(super) fn into_execution(mut self) -> Result<QueryExecutionLease, QueryCoordinatorError> {
        let coordinator = self.coordinator.upgrade().ok_or_else(|| {
            QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped")
        })?;
        if let Err(error) = coordinator.promote_execution(&self.key) {
            let captured = CapturedError::from_adapter_error(
                &error.adapter_error("datafusion_nexus.server.query_coordinator.promote_execution"),
            );
            self.finish(QueryTerminal::Failed { stage: "planning", error: Arc::new(captured) });
            return Err(error);
        }
        self.released = true;
        Ok(QueryExecutionLease {
            coordinator: self.coordinator.clone(),
            key: self.key.clone(),
            observation_id: self.observation_id.clone(),
            hot: Arc::clone(&self.hot),
            released: false,
        })
    }

    pub(super) fn fail(mut self, stage: &'static str, error: CapturedError) {
        self.finish(QueryTerminal::Failed { stage, error: Arc::new(error) });
    }

    pub(super) fn succeed(mut self) {
        self.finish(QueryTerminal::Succeeded);
    }

    pub(super) fn finish_captured(&mut self, stage: &'static str, status: &tonic::Status, error: CapturedError) {
        let terminal = if status.code() == tonic::Code::Cancelled {
            QueryTerminal::Cancelled { stage, error: Arc::new(error) }
        } else {
            QueryTerminal::Failed { stage, error: Arc::new(error) }
        };
        self.finish(terminal);
    }

    fn finish(&mut self, terminal: QueryTerminal) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.set_terminal(&self.key, terminal);
        }
    }
}

impl Drop for QueryPlanningLease {
    fn drop(&mut self) {
        let terminal = if std::thread::panicking() { QueryTerminal::Panicked } else { QueryTerminal::Dropped };
        self.finish(terminal);
    }
}

#[derive(Debug)]
pub(super) struct QueryExecutionLease {
    coordinator: Weak<QueryCoordinator>,
    key: QueryKey,
    observation_id: ObservationId,
    hot: Arc<QueryHotCounters>,
    released: bool,
}

impl QueryExecutionLease {
    #[cfg(test)]
    pub(super) fn observation_id(&self) -> &ObservationId {
        &self.observation_id
    }

    pub(super) fn hot_counters(&self) -> Arc<QueryHotCounters> {
        Arc::clone(&self.hot)
    }

    pub(super) fn observation_handle(&self) -> Result<RuntimeObservationHandle, QueryCoordinatorError> {
        self.coordinator
            .upgrade()
            .ok_or_else(|| QueryCoordinatorError::new(QueryCoordinatorErrorKind::UnknownQuery, "coordinator dropped"))?
            .observation_handle(&self.key, &self.observation_id)
    }

    pub(super) fn activity_factory(&self) -> NativeObservationActivityFactory {
        observation_activity_factory(self.coordinator.clone(), self.key.clone(), self.observation_id.clone())
    }

    pub(super) fn succeed(mut self) {
        self.finish_success();
    }

    #[cfg(test)]
    pub(super) fn fail(mut self, stage: &'static str, error: CapturedError) {
        self.finish_failed(stage, error);
    }

    pub(super) fn finish_success(&mut self) {
        self.finish(QueryTerminal::Succeeded);
    }

    pub(super) fn finish_failed(&mut self, stage: &'static str, error: CapturedError) {
        self.finish(QueryTerminal::Failed { stage, error: Arc::new(error) });
    }

    pub(super) fn finish_cancelled(&mut self, stage: &'static str, error: CapturedError) {
        self.finish(QueryTerminal::Cancelled { stage, error: Arc::new(error) });
    }

    pub(super) fn finish_panicked(&mut self) {
        self.finish(QueryTerminal::Panicked);
    }

    #[cfg(test)]
    pub(super) fn finish_status(&mut self, stage: &'static str, status: &tonic::Status) {
        let error = CapturedError::from_status(status);
        self.finish_captured(stage, status, error);
    }

    pub(super) fn finish_captured(&mut self, stage: &'static str, status: &tonic::Status, error: CapturedError) {
        if status.code() == tonic::Code::Cancelled {
            self.finish_cancelled(stage, error);
        } else {
            self.finish_failed(stage, error);
        }
    }

    fn finish(&mut self, terminal: QueryTerminal) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.set_terminal(&self.key, terminal);
        }
    }
}

impl Drop for QueryExecutionLease {
    fn drop(&mut self) {
        let terminal = if std::thread::panicking() { QueryTerminal::Panicked } else { QueryTerminal::Dropped };
        self.finish(terminal);
    }
}

fn observation_activity_factory(
    coordinator: Weak<QueryCoordinator>,
    key: QueryKey,
    observation_id: ObservationId,
) -> NativeObservationActivityFactory {
    Arc::new(move || {
        let coordinator = coordinator.upgrade()?;
        coordinator.acquire_activity(&key, &observation_id).ok().map(QueryActivityLease::new)
    })
}

#[derive(Debug)]
pub(super) struct ObservationActivityLease {
    coordinator: Weak<QueryCoordinator>,
    key: QueryKey,
    released: bool,
}

impl Drop for ObservationActivityLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        if let Some(coordinator) = self.coordinator.upgrade() {
            coordinator.release_activity(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::mpsc,
    };

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::physical_plan::{empty::EmptyExec, union::UnionExec};
    use nexus_query_engine::{
        metrics::QueryMetrics,
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, LiteralBatchSource, SourceNode},
    };

    use super::*;
    use crate::{
        native_exec::{NativeCudfComposedExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec},
        report::{GpuCandidate, PlanningReport},
    };

    #[derive(Debug)]
    struct TestMaintenanceClock {
        now: Mutex<Instant>,
        advanced: Notify,
        sleepers: AtomicUsize,
    }

    impl TestMaintenanceClock {
        fn new(now: Instant) -> Arc<Self> {
            Arc::new(Self { now: Mutex::new(now), advanced: Notify::new(), sleepers: AtomicUsize::new(0) })
        }

        fn advance(&self, duration: Duration) {
            let mut now = self.now.lock().unwrap();
            *now += duration;
            drop(now);
            self.advanced.notify_waiters();
        }

        fn advance_without_notifying(&self, duration: Duration) {
            *self.now.lock().unwrap() += duration;
        }

        async fn wait_for_sleepers(&self, expected: usize) {
            while self.sleepers.load(Ordering::SeqCst) < expected {
                tokio::task::yield_now().await;
            }
        }
    }

    impl CoordinatorMaintenanceClock for TestMaintenanceClock {
        fn now(&self) -> Instant {
            *self.now.lock().unwrap()
        }

        fn sleep_until(&self, deadline: Instant) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            self.sleepers.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                loop {
                    let mut advanced = Box::pin(self.advanced.notified());
                    advanced.as_mut().enable();
                    if self.now() >= deadline {
                        return;
                    }
                    advanced.await;
                }
            })
        }
    }

    fn config() -> QueryCoordinatorConfig {
        QueryCoordinatorConfig {
            max_live_queries: 2,
            max_ticketed_queries: 1,
            max_active_logical_capture_bytes: 1024 * 1024,
            per_query_capture_max_bytes: 64 * 1024,
            per_query_artifact_max_bytes: 64 * 1024,
            retained_max_count: 1,
            retained_max_bytes: 64 * 1024,
            retained_ttl: Duration::from_secs(10),
            tombstone_max_count: 2,
            tombstone_ttl: Duration::from_secs(20),
            shutdown_grace: Duration::from_secs(5),
        }
    }

    fn register(coordinator: &Arc<QueryCoordinator>, query_id: &str) -> QueryPlanningLease {
        coordinator
            .register_with_id(query_id, "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
            .unwrap()
    }

    fn accepted_native_diagnostics() -> NativePlanningDiagnostics {
        let mut report = PlanningReport::new();
        report.push_rewritten(GpuCandidate::new("test_native_plan", 1));
        NativePlanningDiagnostics {
            input_plan_root: "EmptyExec".to_string(),
            input_plan_shape: "EmptyExec".to_string(),
            input_join_diagnostics: Vec::new(),
            report,
            native_optimizer: Default::default(),
            source_diagnostics: Vec::new(),
            strict_bounded_contract: Default::default(),
            adaptive_pipeline: Default::default(),
        }
    }

    fn captured_error_with_stage(stage: &str) -> CapturedError {
        CapturedError {
            code: "native_execution".to_string(),
            kind: "execution".to_string(),
            status: "transient".to_string(),
            operation: "datafusion_nexus.test".to_string(),
            message: "native execution failed".to_string(),
            facts: vec![("native_datafusion_stage".to_string(), stage.to_string())],
            diagnostic: "native execution failed".to_string(),
            source_chain: None,
            root_cause: None,
        }
    }

    fn ticket_payload(started_at: Instant, deadline: Instant) -> CoordinatorTicketPayload {
        CoordinatorTicketPayload {
            sql: "select 1".to_string(),
            correlation_id: None,
            prepared_handle: None,
            kind: 0,
            native_cancellation: ExecutionCancellation::new(),
            started_at,
            deadline,
        }
    }

    fn phase(coordinator: &QueryCoordinator, query_id: &str) -> Option<QueryPhase> {
        let state = coordinator.state.lock().unwrap();
        let key = state.current.get(query_id)?;
        state.records.get(key).map(|record| record.phase)
    }

    fn wait_for_phase_sync(coordinator: &QueryCoordinator, query_id: &str, expected: QueryPhase) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while phase(coordinator, query_id) != Some(expected) {
            assert!(Instant::now() < deadline, "query {query_id} did not reach phase {expected:?}");
            std::thread::yield_now();
        }
    }

    async fn wait_for_phase(coordinator: &QueryCoordinator, query_id: &str, expected: Option<QueryPhase>) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if phase(coordinator, query_id) == expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("query {query_id} did not reach phase {expected:?}"));
    }

    fn tsv_value<'a>(payload: &'a str, section: &str, column: &str) -> &'a str {
        let marker = format!("[{section}]");
        let mut lines = payload
            .lines()
            .skip_while(|line| *line != marker)
            .skip(1)
            .filter(|line| !line.is_empty() && !line.starts_with('#'));
        let header = lines.next().unwrap().split('\t').collect::<Vec<_>>();
        let row = lines.next().unwrap().split('\t').collect::<Vec<_>>();
        row[header.iter().position(|name| *name == column).unwrap()]
    }

    fn test_native_producer(input_name: &str) -> (NativeCudfGpuProducerExec, Arc<Schema>) {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
        let plan = builder.finish_with_host_sink(source).clone_with_gpu_resident_sink();
        (NativeCudfGpuProducerExec::try_new(input_name, plan).unwrap(), schema)
    }

    fn record_one_gpu_handoff(producer: &NativeCudfGpuProducerExec) {
        let mut metrics = QueryMetrics::default();
        metrics.boundary.record_gpu_resident_input_handoff();
        producer.record_test_query_metrics(&metrics);
    }

    fn capture_successful_plan(
        coordinator: &Arc<QueryCoordinator>,
        query_id: &str,
        plan: Arc<dyn ExecutionPlan>,
    ) -> String {
        let planning = register(coordinator, query_id);
        coordinator.capture_planning(query_id, Some(accepted_native_diagnostics()), plan);
        planning.into_execution().unwrap().succeed();
        String::from_utf8(coordinator.query_stats(query_id).unwrap().to_vec()).unwrap()
    }

    #[test]
    fn terminal_artifact_composer_reserves_fixed_frames_and_keeps_complete_utf8_lines() {
        let control = Arc::<[u8]>::from(&b"control"[..]);
        let jsonl = Arc::<[u8]>::from(&b"jsonl\n"[..]);
        let unicode_line = "# detail=資料\n";
        let payload = Arc::<[u8]>::from(format!("{QUERY_STATS_HEADER_LINE}{unicode_line}# later=ok\n").into_bytes());
        let cap = u64::try_from(control.len() + jsonl.len() + QUERY_STATS_HEADER_LINE.len() + unicode_line.len() - 1)
            .unwrap();

        let composed = compose_terminal_artifact(Arc::clone(&control), Some(Arc::clone(&jsonl)), cap, |_| {
            QueryStatsPayload { bytes: Arc::clone(&payload), incomplete: false }
        })
        .expect("the header fits after fixed frames");

        assert_eq!(composed.bundle.stats.as_ref(), QUERY_STATS_HEADER_LINE.as_bytes());
        assert!(std::str::from_utf8(&composed.bundle.stats).is_ok());
        assert!(composed.stats_incomplete);
        assert_eq!(
            composed.bundle.charge_bytes,
            u64::try_from(
                composed.bundle.stats.len()
                    + composed.bundle.runtime_control.len()
                    + composed.bundle.runtime_jsonl.as_ref().map_or(0, |value| value.len()),
            )
            .unwrap()
        );
        assert!(composed.bundle.charge_bytes <= cap);

        let exact = u64::try_from(control.len() + QUERY_STATS_HEADER_LINE.len()).unwrap();
        assert!(
            compose_terminal_artifact(Arc::clone(&control), None, exact - 1, |_| QueryStatsPayload {
                bytes: Arc::from(QUERY_STATS_HEADER_LINE.as_bytes()),
                incomplete: false,
            })
            .is_none(),
            "a terminal bundle cannot retain a partial stats header"
        );
        let exact_bundle = compose_terminal_artifact(control, None, exact, |_| QueryStatsPayload {
            bytes: Arc::from(QUERY_STATS_HEADER_LINE.as_bytes()),
            incomplete: false,
        })
        .expect("the exact control plus header boundary fits");
        assert_eq!(exact_bundle.bundle.charge_bytes, exact);
    }

    #[test]
    fn undersized_terminal_cap_installs_stats_free_tombstone_without_leaking_egress() {
        let query_id = "tiny-資料";
        let minimum = incomplete_runtime_control(query_id, RetainedUnavailableReason::TooLarge).len()
            + QUERY_STATS_HEADER_LINE.len();
        let mut tiny = config();
        tiny.per_query_artifact_max_bytes = u64::try_from(minimum - 1).unwrap();
        let coordinator = QueryCoordinator::new(tiny);

        register(&coordinator, query_id).into_execution().unwrap().succeed();

        assert_eq!(phase(&coordinator, query_id), Some(QueryPhase::Tombstone));
        assert!(matches!(
            coordinator.artifact(query_id),
            Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::TooLarge))
        ));
        assert!(coordinator.query_stats(query_id).is_none());
        assert!(coordinator.lease_query_stats(query_id).unwrap().is_none());
        assert_eq!(coordinator.artifact_egress_usage(), (0, 0));
        let state = coordinator.lock_state();
        let key = state.current.get(query_id).unwrap();
        let record = state.records.get(key).unwrap();
        assert!(record.retained.is_none());
        assert!(record.tombstone_stats.is_none());
        assert_eq!(state.tombstone_bytes, 0);
    }

    #[test]
    fn forced_shutdown_with_undersized_cap_installs_stats_free_tombstone() {
        let query_id = "forced-tiny-資料";
        let minimum = incomplete_runtime_control(query_id, RetainedUnavailableReason::ShutdownTimeout).len()
            + QUERY_STATS_HEADER_LINE.len();
        let mut tiny = config();
        tiny.per_query_artifact_max_bytes = u64::try_from(minimum - 1).unwrap();
        let coordinator = QueryCoordinator::new(tiny);
        let execution = register(&coordinator, query_id).into_execution().unwrap();
        let activity = execution.activity_factory()().expect("activity lease");
        let now = Instant::now();
        coordinator.begin_shutdown(now);

        assert_eq!(coordinator.force_shutdown_timeout(now + Duration::from_secs(5)), 1);
        assert_eq!(phase(&coordinator, query_id), Some(QueryPhase::Tombstone));
        assert!(matches!(
            coordinator.artifact(query_id),
            Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::ShutdownTimeout))
        ));
        assert!(coordinator.query_stats(query_id).is_none());
        let state = coordinator.lock_state();
        assert_eq!(state.active_logical_capture_bytes, 0);
        assert_eq!(state.retained_count, 0);
        assert_eq!(state.tombstone_count, 1);
        assert_eq!(state.tombstone_bytes, 0);
        drop(state);
        drop(activity);
        drop(execution);
    }

    #[test]
    fn terminal_waits_for_last_activity_then_finalizes_without_access() {
        let coordinator = QueryCoordinator::new(config());
        let planning = register(&coordinator, "delayed");
        let execution = planning.into_execution().unwrap();
        let activity = execution.activity_factory()().expect("activity factory acquires an opaque native token");

        execution.succeed();
        assert!(matches!(coordinator.artifact("delayed"), Some(CoordinatorArtifact::Active)));

        drop(activity);
        assert_eq!(
            coordinator
                .state
                .lock()
                .unwrap()
                .records
                .values()
                .find(|record| record.key.query_id() == "delayed")
                .map(|record| record.phase),
            Some(QueryPhase::Retained),
            "dropping the last native activity must finalize without retrieval traffic"
        );
        let Some(CoordinatorArtifact::Retained(bundle)) = coordinator.artifact("delayed") else {
            panic!("last activity must trigger finalization");
        };
        assert!(bundle.runtime_jsonl.is_some());
    }

    #[test]
    fn overlapping_finalizers_share_one_claim_and_install_one_complete_artifact() {
        let mut finalization_config = config();
        finalization_config.retained_max_count = 2;
        let coordinator = QueryCoordinator::new(finalization_config);
        let execution = register(&coordinator, "overlapping-finalizers").into_execution().unwrap();
        let gate = FinalizationTestGate::new();
        *coordinator.finalization_gate.lock().unwrap() = Some(Arc::clone(&gate));

        let finalizer = std::thread::spawn(move || execution.succeed());
        gate.entered.wait();

        let concurrent = Arc::clone(&coordinator);
        std::thread::spawn(move || concurrent.finalize_ready()).join().unwrap();
        assert_eq!(phase(&coordinator, "overlapping-finalizers"), Some(QueryPhase::Finalizing));
        assert_eq!(coordinator.finalization_installs.load(Ordering::SeqCst), 0);

        gate.release.wait();
        finalizer.join().unwrap();

        assert_eq!(coordinator.finalization_installs.load(Ordering::SeqCst), 1);
        let Some(CoordinatorArtifact::Retained(bundle)) = coordinator.artifact("overlapping-finalizers") else {
            panic!("the claimed finalizer must retain the complete artifact");
        };
        assert!(bundle.runtime_jsonl.is_some(), "a competing finalizer must not install capture_disabled");
        for line in std::str::from_utf8(bundle.runtime_jsonl.as_ref().unwrap()).unwrap().lines() {
            serde_json::from_str::<serde_json::Value>(line).expect("every retained JSONL record is valid JSON");
        }
        let state = coordinator.state.lock().unwrap();
        let key = state.current.get("overlapping-finalizers").unwrap();
        let record = state.records.get(key).unwrap();
        assert!(record.finalization_claim.is_none());
        assert!(record.unavailable.is_none());
    }

    #[test]
    fn stats_reader_after_cache_close_waits_for_terminal_publication() {
        let coordinator = QueryCoordinator::new(config());
        let execution = register(&coordinator, "post-close-reader").into_execution().unwrap();
        let cache = {
            let state = coordinator.lock_state();
            let key = state.current.get("post-close-reader").unwrap();
            Arc::clone(&state.records.get(key).unwrap().live_stats_cache)
        };
        let (wait_tx, wait_rx) = mpsc::channel();
        *cache.wait_observer.lock().unwrap() = Some(wait_tx);
        let gate = FinalizationTestGate::new();
        *coordinator.finalization_gate.lock().unwrap() = Some(Arc::clone(&gate));

        let finalizer = std::thread::spawn(move || execution.succeed());
        gate.entered.wait();

        let reader_coordinator = Arc::clone(&coordinator);
        let reader = std::thread::spawn(move || {
            reader_coordinator
                .query_stats("post-close-reader")
                .expect("a finalizing query remains addressable until terminal stats are published")
        });
        wait_rx.recv_timeout(Duration::from_secs(2)).expect("the post-close reader must wait for terminal publication");

        gate.release.wait();
        finalizer.join().unwrap();
        let stats = reader.join().unwrap();
        assert!(std::str::from_utf8(&stats).unwrap().contains("# outcome=succeeded"));
        assert_eq!(cache.state.lock().unwrap().waiting_readers, 0);
    }

    #[test]
    fn stale_generation_and_observation_identity_cannot_mutate_replacement() {
        let coordinator = QueryCoordinator::new(config());
        let first = register(&coordinator, "same-id");
        let first_key = first.key().clone();
        let first_observation = first.observation_id().clone();
        drop(first);
        let second = register(&coordinator, "same-id");

        let error = coordinator.acquire_activity(&first_key, &first_observation).unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::StaleGeneration);
        assert_ne!(first_observation, *second.observation_id());
    }

    #[test]
    fn duplicate_live_query_id_is_rejected_before_registry_or_budget_mutation() {
        let coordinator = QueryCoordinator::new(config());
        let first = coordinator
            .register_with_id("duplicate-live", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
            .unwrap();
        let before = {
            let state = coordinator.state.lock().unwrap();
            (
                state.next_generation,
                state.active_logical_capture_bytes,
                state.ticketed_count,
                state.records.len(),
                state.current.len(),
            )
        };

        let error = coordinator
            .register_with_id(
                "duplicate-live",
                "select 2",
                Some("replacement-correlation".to_string()),
                Some("replacement-trace".to_string()),
                RuntimeObservationConfig::bounded(1, 1),
            )
            .unwrap_err();

        assert_eq!(error.kind(), QueryCoordinatorErrorKind::DuplicateLiveQueryId);
        assert_eq!(error.query_id(), Some("duplicate-live"));
        let state = coordinator.state.lock().unwrap();
        assert_eq!(
            (
                state.next_generation,
                state.active_logical_capture_bytes,
                state.ticketed_count,
                state.records.len(),
                state.current.len(),
            ),
            before
        );
        drop(state);
        drop(first);
    }

    #[test]
    fn planning_render_runs_unlocked_and_stale_generation_install_is_rejected() {
        let mut planning_config = config();
        planning_config.retained_max_count = 2;
        let coordinator = QueryCoordinator::new(planning_config);
        let planning = register(&coordinator, "planning-stale");
        let stale_key = planning.key().clone();
        let replacement = Arc::new(Mutex::new(None));
        let replacement_for_probe = Arc::clone(&replacement);
        let weak = Arc::downgrade(&coordinator);
        let stale_key_for_probe = stale_key.clone();
        let probe = install_coordinator_render_probe(move |operation| {
            assert_eq!(operation, "planning render");
            let coordinator = weak.upgrade().unwrap();
            assert!(coordinator.state.try_lock().is_ok(), "planning rendered under the registry lock");
            coordinator.set_terminal(&stale_key_for_probe, QueryTerminal::Dropped);
            let fresh = register(&coordinator, "planning-stale");
            replacement_for_probe.lock().unwrap().replace(fresh);
        });
        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));

        coordinator.capture_planning(
            "planning-stale",
            Some(accepted_native_diagnostics()),
            Arc::new(EmptyExec::new(schema)),
        );
        drop(probe);
        drop(planning);

        let replacement = replacement.lock().unwrap().take().expect("probe registered a replacement generation");
        assert_ne!(replacement.key(), &stale_key);
        let payload = String::from_utf8(coordinator.query_stats("planning-stale").unwrap().to_vec()).unwrap();
        assert!(!payload.contains("[physical_plan]"), "stale planning render mutated the replacement: {payload}");
        drop(replacement);
    }

    #[test]
    fn high_churn_reuse_keeps_registry_metadata_bounded_and_rejects_stale_key() {
        let mut churn_config = config();
        churn_config.retained_ttl = Duration::ZERO;
        churn_config.tombstone_ttl = Duration::ZERO;
        let coordinator = QueryCoordinator::new(churn_config);
        let mut stale = None;

        for generation in 1..=256 {
            let planning = register(&coordinator, "00000000-0000-0000-0000-000000000001");
            if generation == 1 {
                stale = Some((planning.key().clone(), planning.observation_id().clone()));
            }
            planning.into_execution().unwrap().succeed();
            coordinator.maintenance_at(Instant::now());
            let state = coordinator.state.lock().unwrap();
            assert!(state.records.is_empty(), "expired churn generation remained resident");
            assert!(state.current.is_empty(), "expired churn generation remained current");
        }

        let replacement = register(&coordinator, "00000000-0000-0000-0000-000000000001");
        let (stale_key, stale_observation) = stale.unwrap();
        let error = coordinator.acquire_activity(&stale_key, &stale_observation).unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::StaleGeneration);
        let state = coordinator.state.lock().unwrap();
        assert_eq!(state.next_generation, 257);
        assert_eq!(state.records.len(), 1);
        assert_eq!(state.current.len(), 1);
        drop(state);
        drop(replacement);
    }

    #[test]
    fn generation_overflow_is_explicit_and_does_not_register() {
        let coordinator = QueryCoordinator::new(config());
        coordinator.state.lock().unwrap().next_generation = u64::MAX;

        let error = coordinator
            .register_with_id("overflow", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
            .unwrap_err();

        assert_eq!(error.kind(), QueryCoordinatorErrorKind::GenerationExhausted);
        let state = coordinator.state.lock().unwrap();
        assert!(state.records.is_empty());
        assert!(state.current.is_empty());
    }

    #[test]
    fn query_wide_observation_identity_is_stable_across_three_native_epochs() {
        let coordinator = QueryCoordinator::new(config());
        let planning = register(&coordinator, "stable-observation");
        let observation_id = planning.observation_id().clone();
        let root = planning.observation_handle().unwrap();
        let execution = planning.into_execution().unwrap();

        assert_eq!(execution.observation_id(), &observation_id);
        for _ in 0..3 {
            let epoch = execution.observation_handle().unwrap();
            assert!(root.is_same_session(&epoch));
        }
        execution.succeed();
    }

    #[test]
    fn dropped_planning_and_execution_leases_terminalize_without_explicit_finish() {
        let mut lease_config = config();
        lease_config.retained_max_count = 2;
        let coordinator = QueryCoordinator::new(lease_config);

        drop(register(&coordinator, "planning-drop"));
        drop(register(&coordinator, "execution-drop").into_execution().unwrap());

        for query_id in ["planning-drop", "execution-drop"] {
            let stats = String::from_utf8(coordinator.query_stats(query_id).unwrap().to_vec()).unwrap();
            assert!(stats.contains("# outcome=dropped"), "{stats}");
            assert!(stats.contains("# sealed=true"), "{stats}");
            assert!(matches!(coordinator.artifact(query_id), Some(CoordinatorArtifact::Retained(_))));
        }
    }

    #[test]
    fn query_stats_projection_and_runtime_control_encoding_run_without_registry_lock() {
        let coordinator = QueryCoordinator::new(config());
        let weak = Arc::downgrade(&coordinator);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observed_for_probe = Arc::clone(&observed);
        let _probe = install_coordinator_render_probe(move |operation| {
            let coordinator = weak.upgrade().unwrap();
            assert!(coordinator.state.try_lock().is_ok(), "{operation} ran under the registry lock");
            observed_for_probe.lock().unwrap().push(operation);
        });
        let planning = register(&coordinator, "lock-probe");

        assert!(coordinator.query_stats("lock-probe").is_some());
        assert!(coordinator.artifact_frames("lock-probe").is_some());

        let execution = planning.into_execution().unwrap();
        assert!(coordinator.query_stats("lock-probe").is_some());
        assert!(coordinator.artifact_frames("lock-probe").is_some());
        execution.succeed();

        assert_eq!(
            *observed.lock().unwrap(),
            [
                "query-stats projection",
                "runtime control encoding",
                "runtime control encoding",
                "finalization projection",
            ]
        );
    }

    #[test]
    fn finalization_projection_panic_installs_bounded_unavailable_without_poisoning_registry() {
        let coordinator = QueryCoordinator::new(config());
        let weak = Arc::downgrade(&coordinator);
        let _probe = install_coordinator_render_probe(move |operation| {
            let coordinator = weak.upgrade().unwrap();
            assert!(coordinator.state.try_lock().is_ok(), "{operation} ran under the registry lock");
            if operation == "finalization projection" {
                panic!("deterministic finalization projection panic");
            }
        });
        let execution = register(&coordinator, "finalization-panic").into_execution().unwrap();

        execution.succeed();

        assert!(matches!(
            coordinator.artifact("finalization-panic"),
            Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::FinalizationPanicked))
        ));
        assert_eq!(coordinator.terminal("finalization-panic"), Some(QueryTerminalOutcome::Succeeded));
        let state = coordinator.lock_state();
        let key = state.current.get("finalization-panic").unwrap();
        let record = state.records.get(key).unwrap();
        assert_eq!(record.phase, QueryPhase::Retained);
        assert_eq!(record.logical_capture_bytes, 0);
        assert_eq!(record.finalization_claim, None);
        let bundle = record.retained.as_ref().unwrap();
        assert_eq!(
            bundle.charge_bytes,
            u64::try_from(
                bundle.stats.len()
                    + bundle.runtime_control.len()
                    + bundle.runtime_jsonl.as_ref().map_or(0, |value| value.len()),
            )
            .unwrap()
        );
        assert!(bundle.charge_bytes <= config().per_query_artifact_max_bytes);
        assert_eq!(state.active_logical_capture_bytes, 0);
        drop(state);
        let stats = String::from_utf8(coordinator.query_stats("finalization-panic").unwrap().to_vec()).unwrap();
        assert!(stats.contains("# query_id=finalization-panic"), "{stats}");
        assert!(stats.contains("# outcome=succeeded"), "{stats}");
        assert!(stats.contains("# incomplete=true"), "{stats}");
        assert!(stats.contains("# unavailable_reason=finalization_panicked"), "{stats}");
        let (control, runtime) = coordinator.artifact_frames("finalization-panic").unwrap();
        let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
        assert_eq!(control["reason"], "finalization_panicked");
        assert_eq!(control["incomplete"], true);
        assert!(runtime.is_none());

        drop(register(&coordinator, "after-finalization-panic"));
        assert!(coordinator.is_drained());
    }

    #[test]
    fn checked_deadline_overflow_is_immediate_and_never_panics() {
        let now = Instant::now();
        assert_eq!(checked_deadline(now, Duration::MAX), now);
    }

    #[test]
    fn coordinator_stats_project_producer_only_composed_metrics_exactly_once() {
        let mut metrics_config = config();
        metrics_config.retained_max_bytes = 1024 * 1024;
        let coordinator = QueryCoordinator::new(metrics_config);
        let (producer, schema) = test_native_producer("producer-only");
        record_one_gpu_handoff(&producer);

        let mut consumer_builder = QueryPlanBuilder::new();
        let source =
            consumer_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("producer-only", schema)));
        let consumer = consumer_builder.finish_with_host_sink(source);
        let composed =
            NativeCudfComposedExec::try_new(consumer, vec![Arc::new(producer)], NativeCudfExecutionConfig::unbounded())
                .unwrap();
        let plan = Arc::new(composed) as Arc<dyn ExecutionPlan>;

        let payload = capture_successful_plan(&coordinator, "producer-composed-metrics", plan);

        assert_eq!(tsv_value(&payload, "execution", "gpu_resident_input_handoffs"), "1", "{payload}");
    }

    #[test]
    fn coordinator_stats_deduplicate_shared_dag_nodes_and_metric_arcs_exactly_once() {
        let mut metrics_config = config();
        metrics_config.retained_max_bytes = 1024 * 1024;
        let coordinator = QueryCoordinator::new(metrics_config);
        let (producer, _) = test_native_producer("shared-metrics");
        record_one_gpu_handoff(&producer);
        let shared = Arc::new(producer.clone()) as Arc<dyn ExecutionPlan>;
        let same_node = Arc::clone(&shared);
        let distinct_node_with_shared_metrics = Arc::new(producer) as Arc<dyn ExecutionPlan>;
        let plan = UnionExec::try_new(vec![shared, same_node, distinct_node_with_shared_metrics]).unwrap();

        let payload = capture_successful_plan(&coordinator, "shared-dag-metrics", plan);

        assert_eq!(tsv_value(&payload, "execution", "gpu_resident_input_handoffs"), "1", "{payload}");
    }

    #[test]
    fn planning_lease_unwind_records_panicked_once_and_does_not_poison_registry() {
        let coordinator = QueryCoordinator::new(config());
        let planning = register(&coordinator, "planning-panic");
        let key = planning.key().clone();

        assert!(
            catch_unwind(AssertUnwindSafe(move || {
                let _planning = planning;
                panic!("planning lease unwind probe");
            }))
            .is_err()
        );
        coordinator.set_terminal(&key, QueryTerminal::Succeeded);

        let state = coordinator.state.lock().expect("lease unwind must not poison registry mutex");
        let record = state.records.get(&key).unwrap();
        assert_eq!(record.terminal_outcome, Some(QueryTerminalOutcome::Panicked));
        assert!(record.terminal.is_none(), "retained records keep only the compact terminal outcome");
        assert_eq!(record.phase, QueryPhase::Retained);
        assert_eq!(record.active_leases, 0);
    }

    #[test]
    fn execution_lease_unwind_records_panicked_once_and_does_not_poison_registry() {
        let coordinator = QueryCoordinator::new(config());
        let execution = register(&coordinator, "execution-panic").into_execution().unwrap();
        let key = execution.key.clone();

        assert!(
            catch_unwind(AssertUnwindSafe(move || {
                let _execution = execution;
                panic!("execution lease unwind probe");
            }))
            .is_err()
        );
        coordinator.set_terminal(&key, QueryTerminal::Succeeded);

        let state = coordinator.state.lock().expect("lease unwind must not poison registry mutex");
        let record = state.records.get(&key).unwrap();
        assert_eq!(record.terminal_outcome, Some(QueryTerminalOutcome::Panicked));
        assert!(record.terminal.is_none(), "retained records keep only the compact terminal outcome");
        assert_eq!(record.phase, QueryPhase::Retained);
        assert_eq!(record.active_leases, 0);
    }

    #[test]
    fn activity_lease_unwind_cannot_overwrite_first_terminal_or_poison_registry() {
        let coordinator = QueryCoordinator::new(config());
        let execution = register(&coordinator, "activity-panic").into_execution().unwrap();
        let key = execution.key.clone();
        let activity = execution.activity_factory()().unwrap();
        execution.succeed();

        assert!(
            catch_unwind(AssertUnwindSafe(move || {
                let _activity = activity;
                panic!("activity lease unwind probe");
            }))
            .is_err()
        );

        let state = coordinator.state.lock().expect("activity unwind must not poison registry mutex");
        let record = state.records.get(&key).unwrap();
        assert_eq!(record.terminal_outcome, Some(QueryTerminalOutcome::Succeeded));
        assert!(record.terminal.is_none(), "retained records keep only the compact terminal outcome");
        assert_eq!(record.phase, QueryPhase::Retained);
        assert_eq!(record.active_activities, 0);
    }

    #[test]
    fn shutdown_race_into_execution_is_structured_cancelled_and_preserves_terminal() {
        let coordinator = QueryCoordinator::new(config());
        let planning = register(&coordinator, "do-put-shutdown-race");

        assert_eq!(coordinator.begin_shutdown(Instant::now()).live_count(), 1);
        let error = planning.into_execution().unwrap_err();

        assert_eq!(error.kind(), QueryCoordinatorErrorKind::Shutdown);
        let adapter = error.adapter_error("datafusion_nexus.server.flight.do_put_statement_update.into_execution");
        assert_eq!(adapter.code(), ErrorCode::Cancelled);
        assert_eq!(adapter.fact_value("query_coordinator_error"), Some("shutdown"));
        assert!(matches!(coordinator.terminal("do-put-shutdown-race"), Some(QueryTerminalOutcome::Cancelled)));
        assert!(matches!(
            coordinator.artifact("do-put-shutdown-race"),
            Some(CoordinatorArtifact::Retained(_) | CoordinatorArtifact::Unavailable(_))
        ));
    }

    #[tokio::test]
    async fn manual_clock_expires_exact_ticket_retention_and_tombstone_boundaries_without_traffic() {
        let coordinator = QueryCoordinator::new(config());
        let clock = TestMaintenanceClock::new(Instant::now());
        let maintenance = QueryCoordinatorMaintenance::start_with_clock(
            Arc::clone(&coordinator),
            Arc::<TestMaintenanceClock>::clone(&clock),
        );
        let base = clock.now();
        let planning = register(&coordinator, "ticket");
        planning.issue_ticket(ticket_payload(base, base + Duration::from_secs(1))).unwrap();

        clock.wait_for_sleepers(1).await;
        clock.advance(Duration::from_secs(1) - Duration::from_nanos(1));
        tokio::task::yield_now().await;
        assert_eq!(phase(&coordinator, "ticket"), Some(QueryPhase::Ticketed));
        clock.advance(Duration::from_nanos(1));
        wait_for_phase(&coordinator, "ticket", Some(QueryPhase::Retained)).await;

        clock.wait_for_sleepers(2).await;
        let retention_deadline = {
            let state = coordinator.state.lock().unwrap();
            let key = state.current.get("ticket").unwrap();
            state.records.get(key).unwrap().retention_deadline.unwrap()
        };
        let until_retention = retention_deadline.duration_since(clock.now());
        clock.advance(until_retention - Duration::from_nanos(1));
        tokio::task::yield_now().await;
        assert_eq!(phase(&coordinator, "ticket"), Some(QueryPhase::Retained));
        clock.advance(Duration::from_nanos(1));
        wait_for_phase(&coordinator, "ticket", Some(QueryPhase::Tombstone)).await;

        clock.wait_for_sleepers(3).await;
        let tombstone_deadline = {
            let state = coordinator.state.lock().unwrap();
            let key = state.current.get("ticket").unwrap();
            state.records.get(key).unwrap().tombstone_deadline.unwrap()
        };
        let until_tombstone = tombstone_deadline.duration_since(clock.now());
        clock.advance(until_tombstone - Duration::from_nanos(1));
        tokio::task::yield_now().await;
        assert_eq!(phase(&coordinator, "ticket"), Some(QueryPhase::Tombstone));
        clock.advance(Duration::from_nanos(1));
        wait_for_phase(&coordinator, "ticket", None).await;

        assert!(clock.sleepers.load(Ordering::SeqCst) >= 3, "maintenance must sleep on every exact deadline");
        let plan = coordinator.begin_shutdown(clock.now());
        assert!(!maintenance.finish(plan).await.unwrap());
    }

    #[test]
    fn maintenance_retained_expiry_batch_enforces_tombstone_cap_immediately() {
        let mut expiry_config = config();
        expiry_config.retained_max_count = 8;
        expiry_config.retained_max_bytes = 1024 * 1024;
        expiry_config.retained_ttl = Duration::from_secs(1);
        expiry_config.tombstone_max_count = 2;
        let coordinator = QueryCoordinator::new(expiry_config);

        for query_id in ["expire-0", "expire-1", "expire-2", "expire-3", "expire-4"] {
            register(&coordinator, query_id).into_execution().unwrap().succeed();
        }
        coordinator.maintenance_at(Instant::now() + Duration::from_secs(2));

        let state = coordinator.state.lock().unwrap();
        assert_eq!(state.retained_count, 0);
        assert_eq!(state.tombstone_count, 2);
        assert_eq!(state.records.len(), 2, "the same maintenance transaction must enforce the tombstone cap");
        let remaining = state.records.keys().map(|key| key.query_id()).collect::<Vec<_>>();
        assert_eq!(remaining, ["expire-3", "expire-4"], "deadline ties use QueryKey ordering deterministically");
    }

    #[tokio::test]
    async fn maintenance_shutdown_joins_after_last_execution_and_activity_release() {
        let coordinator = QueryCoordinator::new(config());
        let clock = TestMaintenanceClock::new(Instant::now());
        let maintenance = QueryCoordinatorMaintenance::start_with_clock(
            Arc::clone(&coordinator),
            Arc::<TestMaintenanceClock>::clone(&clock),
        );
        let planning = register(&coordinator, "graceful-shutdown");
        let execution = planning.into_execution().unwrap();
        let activity = execution.activity_factory()().expect("activity lease");

        let plan = coordinator.begin_shutdown(clock.now());
        assert_eq!(plan.live_count(), 1);
        let error = coordinator
            .register_with_id("closed", "select 2", None, None, RuntimeObservationConfig::bounded(16, 4096))
            .unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::RegistrationCapacity);
        let finish = tokio::spawn(maintenance.finish(plan));

        drop(execution);
        assert!(!coordinator.is_drained(), "native activity keeps finalization live during grace");
        drop(activity);

        assert!(!finish.await.unwrap().unwrap());
        assert!(matches!(coordinator.artifact("graceful-shutdown"), Some(CoordinatorArtifact::Retained(_))));
    }

    #[tokio::test]
    async fn maintenance_rejects_quiescence_first_observed_at_absolute_deadline() {
        let coordinator = QueryCoordinator::new(config());
        let clock = TestMaintenanceClock::new(Instant::now());
        let maintenance = QueryCoordinatorMaintenance::start_with_clock(
            Arc::clone(&coordinator),
            Arc::<TestMaintenanceClock>::clone(&clock),
        );
        let execution = register(&coordinator, "late-graceful-shutdown").into_execution().unwrap();
        let activity = execution.activity_factory()().expect("activity lease");

        let plan = coordinator.begin_shutdown(clock.now());
        let finish = tokio::spawn(maintenance.finish(plan));
        clock.wait_for_sleepers(1).await;
        clock.advance_without_notifying(config().shutdown_grace);
        drop(execution);
        drop(activity);

        assert!(finish.await.unwrap().unwrap(), "quiescence at the deadline cannot report graceful success");
        assert!(matches!(coordinator.artifact("late-graceful-shutdown"), Some(CoordinatorArtifact::Retained(_))));
    }

    #[tokio::test]
    async fn maintenance_shutdown_timeout_retains_bounded_incomplete_diagnostic_and_late_drops_are_noops() {
        let coordinator = QueryCoordinator::new(config());
        let clock = TestMaintenanceClock::new(Instant::now());
        let maintenance = QueryCoordinatorMaintenance::start_with_clock(
            Arc::clone(&coordinator),
            Arc::<TestMaintenanceClock>::clone(&clock),
        );
        let planning = coordinator
            .register_with_id(
                "forced-shutdown",
                "select secret_literal",
                Some("correlation".to_string()),
                None,
                RuntimeObservationConfig::bounded(16, 4096),
            )
            .unwrap();
        let execution = planning.into_execution().unwrap();
        execution.hot_counters().record_output(3, 64);
        let activity = execution.activity_factory()().expect("activity lease");

        let plan = coordinator.begin_shutdown(clock.now());
        assert_eq!(plan.live_count(), 1);
        let finish = tokio::spawn(maintenance.finish(plan));
        clock.wait_for_sleepers(1).await;
        clock.advance(Duration::from_secs(6));

        assert!(finish.await.unwrap().unwrap());
        assert!(matches!(
            coordinator.artifact("forced-shutdown"),
            Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::ShutdownTimeout))
        ));
        let stats = String::from_utf8(coordinator.query_stats("forced-shutdown").unwrap().to_vec()).unwrap();
        assert!(stats.contains("# incomplete=true"), "{stats}");
        assert!(stats.contains("# unavailable_reason=shutdown_timeout"), "{stats}");
        assert!(stats.contains("# output_batches=1"), "{stats}");
        assert!(!stats.contains("secret_literal"), "{stats}");
        let (control, runtime_jsonl) = coordinator.artifact_frames("forced-shutdown").unwrap();
        let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
        assert_eq!(control["reason"], "shutdown_timeout");
        assert_eq!(control["incomplete"], true);
        assert!(runtime_jsonl.is_none());
        let before = coordinator.query_stats("forced-shutdown").unwrap();
        {
            let state = coordinator.state.lock().unwrap();
            let record = state.records.values().find(|record| record.key.query_id() == "forced-shutdown").unwrap();
            assert!(record.captured.is_none());
            assert!(record.observation_owner.is_none());
            assert_eq!(record.active_leases, 0);
            assert_eq!(record.active_activities, 0);
            let bundle = record.retained.as_ref().expect("forced shutdown retains its bounded fallback bundle");
            assert_eq!(
                bundle.charge_bytes,
                u64::try_from(
                    bundle.stats.len()
                        + bundle.runtime_control.len()
                        + bundle.runtime_jsonl.as_ref().map_or(0, |value| value.len()),
                )
                .unwrap()
            );
            assert!(bundle.charge_bytes <= config().per_query_artifact_max_bytes);
        }

        drop(activity);
        drop(execution);
        assert_eq!(coordinator.query_stats("forced-shutdown").unwrap().as_ref(), before.as_ref());
    }

    #[test]
    fn registration_cap_rejects_before_work() {
        let mut limited = config();
        limited.max_live_queries = 1;
        let coordinator = QueryCoordinator::new(limited);
        let _first = register(&coordinator, "first");
        let error = coordinator
            .register_with_id("second", "select 2", None, None, RuntimeObservationConfig::bounded(16, 4096))
            .unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::RegistrationCapacity);
    }

    #[test]
    fn ticket_and_logical_capture_byte_caps_reject_before_expensive_work() {
        let now = Instant::now();
        let mut ticket_limited = config();
        ticket_limited.max_live_queries = 3;
        ticket_limited.max_ticketed_queries = 1;
        let coordinator = QueryCoordinator::new(ticket_limited);
        register(&coordinator, "first-ticket")
            .issue_ticket(ticket_payload(now, now + Duration::from_secs(30)))
            .unwrap();
        let error = register(&coordinator, "second-ticket")
            .issue_ticket(ticket_payload(now, now + Duration::from_secs(30)))
            .unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::TicketCapacity);
        let stats = String::from_utf8(coordinator.query_stats("second-ticket").unwrap().to_vec()).unwrap();
        assert!(stats.contains("code=service_overloaded"), "{stats}");
        assert!(stats.contains("kind=resource_exhausted"), "{stats}");
        assert!(stats.contains("status=temporary"), "{stats}");
        assert!(stats.contains("operation=datafusion_nexus.server.query_coordinator.issue_ticket"), "{stats}");

        let mut capture_limited = config();
        capture_limited.max_live_queries = 3;
        capture_limited.per_query_capture_max_bytes = 1;
        capture_limited.per_query_artifact_max_bytes = 1;
        capture_limited.max_active_logical_capture_bytes = 8194;
        let coordinator = QueryCoordinator::new(capture_limited);
        let _first = coordinator
            .register_with_id("first-capture", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
            .unwrap();
        let error = coordinator
            .register_with_id("second-capture", "select 2", None, None, RuntimeObservationConfig::bounded(16, 1))
            .unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::LogicalCaptureBytes);
    }

    #[test]
    fn hot_counter_updates_do_not_require_registry_lock() {
        let coordinator = QueryCoordinator::new(config());
        let planning = register(&coordinator, "hot");
        let hot = planning.hot_counters();
        let state = coordinator.state.lock().unwrap();
        hot.record_output(3, 64);
        assert_eq!(hot.snapshot().output_batches, 1);
        drop(state);
    }

    #[test]
    fn artifact_frames_preserve_control_schema_escape_ids_and_never_expose_sql() {
        let coordinator = QueryCoordinator::new(config());
        let query_id = "quoted-\"query\nline";
        let planning = coordinator
            .register_with_id(
                query_id,
                "select 'private-sql-literal'",
                None,
                None,
                RuntimeObservationConfig::bounded(16, 4096),
            )
            .unwrap();
        let (pending, jsonl) = coordinator.artifact_frames(query_id).unwrap();
        assert!(jsonl.is_none());
        let pending: serde_json::Value = serde_json::from_slice(&pending).unwrap();
        assert_eq!(pending["schema"], RUNTIME_ARTIFACT_CONTROL_SCHEMA);
        assert_eq!(pending["version"], 1);
        assert_eq!(pending["query_id"], query_id);
        assert_eq!(pending["state"], "pending");
        assert_eq!(pending["retryable"], true);

        let execution = planning.into_execution().unwrap();
        let (active, jsonl) = coordinator.artifact_frames(query_id).unwrap();
        assert!(jsonl.is_none());
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&active).unwrap()["state"], "active");
        execution.succeed();

        let (sealed, jsonl) = coordinator.artifact_frames(query_id).unwrap();
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&sealed).unwrap()["state"], "sealed");
        let jsonl = jsonl.expect("terminal artifact has one complete JSONL frame");
        let jsonl = std::str::from_utf8(&jsonl).unwrap();
        assert!(jsonl.contains("\"record_type\":\"manifest\""));
        assert!(jsonl.contains("\"record_type\":\"footer\""));
        assert!(!jsonl.contains("private-sql-literal"));
    }

    #[test]
    fn retention_charges_exact_serialized_bytes_and_evicts_oldest_terminal_only() {
        let mut limited = config();
        limited.retained_max_count = 1;
        let coordinator = QueryCoordinator::new(limited);
        let active = register(&coordinator, "active").into_execution().unwrap();

        for query_id in ["older", "newer"] {
            register(&coordinator, query_id).into_execution().unwrap().succeed();
        }

        assert!(matches!(coordinator.artifact("active"), Some(CoordinatorArtifact::Active)));
        assert!(matches!(
            coordinator.artifact("older"),
            Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::Evicted))
        ));
        let Some(CoordinatorArtifact::Retained(bundle)) = coordinator.artifact("newer") else {
            panic!("newest terminal query must be retained");
        };
        let expected = u64::try_from(bundle.stats.len() + bundle.runtime_control.len()).unwrap()
            + u64::try_from(bundle.runtime_jsonl.as_ref().unwrap().len()).unwrap();
        assert_eq!(bundle.charge_bytes, expected);
        let state = coordinator.state.lock().unwrap();
        assert_eq!(state.retained_count, 1);
        assert_eq!(state.retained_bytes, expected);
        drop(state);
        drop(active);
    }

    #[test]
    fn structured_failure_stats_keep_log_safe_facts_and_first_terminal() {
        let coordinator = QueryCoordinator::new(config());
        let mut execution = register(&coordinator, "failed-stats").into_execution().unwrap();
        execution.hot_counters().record_output(7, 128);
        execution.finish_failed(
            "collect",
            CapturedError {
                code: "native_execution".to_string(),
                kind: "execution".to_string(),
                status: "transient".to_string(),
                operation: "test.operation".to_string(),
                message: "safe failure".to_string(),
                facts: vec![("reason".to_string(), "reservation_pressure".to_string())],
                diagnostic: "safe diagnostic".to_string(),
                source_chain: None,
                root_cause: None,
            },
        );
        execution.finish_success();

        let payload = coordinator.query_stats("failed-stats").unwrap();
        let payload = std::str::from_utf8(&payload).unwrap();
        assert!(payload.contains("# outcome=failed\n"), "{payload}");
        assert!(payload.contains("# sealed=true\n"), "{payload}");
        assert!(payload.contains("fact.reason=reservation_pressure\n"), "{payload}");
        assert!(payload.contains("operation=test.operation\n"), "{payload}");
    }

    #[test]
    fn over_budget_terminal_becomes_explicit_unavailable_tombstone() {
        let mut limited = config();
        limited.retained_max_bytes = 1;
        let coordinator = QueryCoordinator::new(limited);
        register(&coordinator, "too-large").into_execution().unwrap().succeed();

        assert!(matches!(
            coordinator.artifact("too-large"),
            Some(CoordinatorArtifact::Unavailable(RetainedUnavailableReason::TooLarge))
        ));
        let (control, jsonl) = coordinator.artifact_frames("too-large").unwrap();
        assert!(jsonl.is_none());
        let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
        assert_eq!(control["state"], "unavailable");
        assert_eq!(control["reason"], "global_budget");
        assert_eq!(control["incomplete"], true);
        let stats = String::from_utf8(coordinator.query_stats("too-large").unwrap().to_vec()).unwrap();
        assert!(stats.contains("# incomplete=true"), "{stats}");
        assert!(stats.contains("# unavailable_reason=global_budget"), "{stats}");
        let state = coordinator.state.lock().unwrap();
        assert_eq!(state.tombstone_count, 1);
        assert_eq!(state.tombstone_bytes, stats.len() as u64);
    }

    #[test]
    fn adversarial_capture_inputs_and_transient_artifacts_stay_bounded_with_structured_loss() {
        let mut bounded = config();
        bounded.max_active_logical_capture_bytes = 64 * 1024;
        bounded.per_query_capture_max_bytes = 1024;
        bounded.per_query_artifact_max_bytes = 8 * 1024;
        bounded.retained_max_bytes = 8 * 1024;
        let coordinator = QueryCoordinator::new(bounded.clone());
        let planning = coordinator
            .register_with_id(
                "adversarial-bounded",
                "s".repeat(2 * 1024 * 1024),
                Some("c".repeat(2 * 1024 * 1024)),
                None,
                RuntimeObservationConfig::bounded(8, 1024),
            )
            .unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
        let plan = UnionExec::try_new(
            (0..4096).map(|_| Arc::new(EmptyExec::new(Arc::clone(&schema))) as Arc<dyn ExecutionPlan>).collect(),
        )
        .unwrap();
        let mut diagnostics = accepted_native_diagnostics();
        diagnostics.input_plan_shape = "diagnostic".repeat(256 * 1024);
        coordinator.capture_planning("adversarial-bounded", Some(diagnostics), plan);

        let huge = "error".repeat(256 * 1024);
        planning.into_execution().unwrap().fail(
            "runtime_execution",
            CapturedError {
                code: huge.clone(),
                kind: huge.clone(),
                status: huge.clone(),
                operation: huge.clone(),
                message: huge.clone(),
                facts: (0..4096).map(|index| (format!("key-{index}"), "v".repeat(1024))).collect(),
                diagnostic: huge.clone(),
                source_chain: Some(huge.clone()),
                root_cause: Some(huge),
            },
        );

        let stats = coordinator.query_stats("adversarial-bounded").expect("bounded terminal stats");
        assert!(stats.len() <= bounded.per_query_artifact_max_bytes as usize);
        let stats = String::from_utf8(stats.to_vec()).unwrap();
        assert!(stats.contains("# incomplete=true"), "{stats}");
        assert!(
            stats.contains("capture_loss.retained_byte_limit") || stats.contains("# unavailable_reason=global_budget"),
            "{stats}"
        );
        let state = coordinator.state.lock().unwrap();
        assert_eq!(state.active_logical_capture_bytes, 0, "terminal install releases the reservation once");
        assert!(state.retained_bytes <= bounded.retained_max_bytes);
    }

    #[test]
    fn different_key_waiter_sleeps_until_current_render_finishes_then_retries_once() {
        let cache = Arc::new(LiveStatsCache::new());
        let first_key =
            (1, QueryHotCounterSnapshot { output_batches: 1, output_rows: 1, output_bytes: 1, overflow: false });
        let second_key =
            (2, QueryHotCounterSnapshot { output_batches: 2, output_rows: 2, output_bytes: 2, overflow: false });
        let LiveStatsRead::Render(first_renderer) = cache.begin_read(first_key) else {
            panic!("first key must elect the renderer");
        };
        let (wait_tx, wait_rx) = mpsc::channel();
        *cache.wait_observer.lock().unwrap() = Some(wait_tx);
        let LiveStatsRead::Wait(second_waiter) = cache.begin_read(second_key) else {
            panic!("different key must wait behind the active renderer");
        };
        let waiter = std::thread::spawn(move || second_waiter.wait());
        wait_rx.recv_timeout(Duration::from_secs(2)).expect("different-key waiter must enter the condvar wait");
        {
            let state = cache.state.lock().unwrap();
            assert_eq!(state.active_renderers, 1);
            assert_eq!(state.waiting_readers, 1);
        }

        first_renderer.complete(Arc::from(&b"first"[..]));
        assert!(matches!(waiter.join().unwrap(), LiveStatsWaitResult::Retry));
        let LiveStatsRead::Render(second_renderer) = cache.begin_read(second_key) else {
            panic!("second key must render once after the first flight completes");
        };
        let second = second_renderer.complete(Arc::from(&b"second"[..])).unwrap();
        assert_eq!(second.as_ref(), b"second");
        let state = cache.state.lock().unwrap();
        assert_eq!(state.active_renderers, 0);
        assert_eq!(state.waiting_readers, 0);
        assert_eq!(state.completed_key, Some(second_key));
    }

    #[test]
    fn renderer_drop_wakes_all_waiters_with_one_shared_error_and_exact_accounting() {
        let cache = Arc::new(LiveStatsCache::new());
        let key = (1, QueryHotCounterSnapshot { output_batches: 1, output_rows: 1, output_bytes: 1, overflow: false });
        let LiveStatsRead::Render(renderer) = cache.begin_read(key) else {
            panic!("first reader must render");
        };
        let (wait_tx, wait_rx) = mpsc::channel();
        *cache.wait_observer.lock().unwrap() = Some(wait_tx);
        let waiters = (0..8)
            .map(|_| {
                let LiveStatsRead::Wait(waiter) = cache.begin_read(key) else {
                    panic!("same-key reader must join the flight");
                };
                std::thread::spawn(move || waiter.wait())
            })
            .collect::<Vec<_>>();
        for _ in 0..waiters.len() {
            wait_rx.recv_timeout(Duration::from_secs(2)).expect("every waiter must block behind the renderer");
        }

        drop(renderer);
        assert!(waiters.into_iter().all(|waiter| matches!(waiter.join().unwrap(), LiveStatsWaitResult::Failed)));
        let state = cache.state.lock().unwrap();
        assert_eq!(state.active_renderers, 0);
        assert_eq!(state.waiting_readers, 0);
        assert_eq!(state.rendering_key, None);
        assert_eq!(state.completed_key, Some(key));
        assert!(state.render_failed);
    }

    #[test]
    fn concurrent_repeated_live_stats_reads_share_one_bounded_snapshot() {
        let coordinator = QueryCoordinator::new(config());
        let execution = register(&coordinator, "live-stats-cache").into_execution().unwrap();
        let cache = {
            let state = coordinator.lock_state();
            let key = state.current.get("live-stats-cache").unwrap();
            Arc::clone(&state.records.get(key).unwrap().live_stats_cache)
        };
        let start = Arc::new(Barrier::new(33));
        let render_entered = Arc::new(Barrier::new(2));
        let render_release = Arc::new(Barrier::new(2));
        let render_count = Arc::new(AtomicUsize::new(0));
        let readers = (0..32)
            .map(|_| {
                let coordinator = Arc::clone(&coordinator);
                let start = Arc::clone(&start);
                let render_entered = Arc::clone(&render_entered);
                let render_release = Arc::clone(&render_release);
                let render_count = Arc::clone(&render_count);
                std::thread::spawn(move || {
                    let _probe = install_coordinator_render_probe(move |operation| {
                        if operation == "query-stats projection" && render_count.fetch_add(1, Ordering::SeqCst) == 0 {
                            render_entered.wait();
                            render_release.wait();
                        }
                    });
                    start.wait();
                    coordinator.query_stats("live-stats-cache").unwrap()
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        render_entered.wait();
        let wait_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if cache.state.lock().unwrap().waiting_readers == 31 {
                break;
            }
            assert!(Instant::now() < wait_deadline, "all concurrent cache misses must join the single flight");
            std::thread::yield_now();
        }
        render_release.wait();
        let snapshots = readers.into_iter().map(|reader| reader.join().unwrap()).collect::<Vec<_>>();
        assert_eq!(render_count.load(Ordering::SeqCst), 1, "only the elected builder may render the artifact");
        assert!(snapshots.iter().all(|snapshot| Arc::ptr_eq(snapshot, &snapshots[0])));
        assert!(snapshots[0].len() <= config().per_query_artifact_max_bytes as usize);
        assert_eq!(cache.state.lock().unwrap().waiting_readers, 0);
        execution.succeed();
    }

    #[test]
    fn live_stats_egress_reserves_bound_before_render_and_reconciles_to_actual_bytes() {
        let bounded = config();
        let coordinator = QueryCoordinator::new(bounded.clone());
        let execution = register(&coordinator, "live-egress-preflight").into_execution().unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let request_coordinator = Arc::clone(&coordinator);
        let request_entered = Arc::clone(&entered);
        let request_release = Arc::clone(&release);
        let request = std::thread::spawn(move || {
            let _probe = install_coordinator_render_probe(move |operation| {
                if operation == "query-stats projection" {
                    request_entered.wait();
                    request_release.wait();
                }
            });
            request_coordinator.lease_query_stats("live-egress-preflight").unwrap().unwrap()
        });
        entered.wait();
        {
            let state = coordinator.artifact_egress.state.lock().unwrap();
            assert_eq!(state.streams, 1);
            assert_eq!(state.bytes, bounded.per_query_artifact_max_bytes);
            assert_eq!(state.peak_streams, 1);
            assert_eq!(state.peak_bytes, bounded.per_query_artifact_max_bytes);
        }
        release.wait();
        let mut response = request.join().unwrap();
        let actual_bytes = u64::try_from(response.next_frame().unwrap().len()).unwrap();
        {
            let state = coordinator.artifact_egress.state.lock().unwrap();
            assert_eq!(state.streams, 1);
            assert_eq!(state.bytes, actual_bytes);
            assert_eq!(state.peak_bytes, bounded.per_query_artifact_max_bytes);
        }
        drop(response);
        let state = coordinator.artifact_egress.state.lock().unwrap();
        assert_eq!((state.streams, state.bytes), (0, 0));
        drop(state);
        execution.succeed();
    }

    #[test]
    fn live_stats_render_panic_releases_provisional_egress_and_publishes_shared_error() {
        let coordinator = QueryCoordinator::new(config());
        let execution = register(&coordinator, "live-render-panic").into_execution().unwrap();
        let request_coordinator = Arc::clone(&coordinator);
        let panicked = std::thread::spawn(move || {
            catch_unwind(AssertUnwindSafe(|| {
                let _probe = install_coordinator_render_probe(|operation| {
                    if operation == "query-stats projection" {
                        panic!("injected stats renderer panic");
                    }
                });
                request_coordinator.lease_query_stats("live-render-panic")
            }))
        })
        .join()
        .unwrap();
        assert!(panicked.is_err());
        {
            let state = coordinator.artifact_egress.state.lock().unwrap();
            assert_eq!((state.streams, state.bytes), (0, 0));
        }
        let error = coordinator.lease_query_stats("live-render-panic").unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::ArtifactRenderFailed);
        let state = coordinator.artifact_egress.state.lock().unwrap();
        assert_eq!((state.streams, state.bytes), (0, 0));
        drop(state);
        execution.succeed();
    }

    #[test]
    fn finalization_closes_and_drains_in_flight_live_stats_without_late_repopulation() {
        let coordinator = QueryCoordinator::new(config());
        let execution = register(&coordinator, "live-finalize-race").into_execution().unwrap();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let reader_coordinator = Arc::clone(&coordinator);
        let reader_entered = Arc::clone(&entered);
        let reader_release = Arc::clone(&release);
        let reader = std::thread::spawn(move || {
            let _probe = install_coordinator_render_probe(move |operation| {
                if operation == "query-stats projection" {
                    reader_entered.wait();
                    reader_release.wait();
                }
            });
            reader_coordinator.query_stats("live-finalize-race").expect("reader receives terminal stats")
        });
        entered.wait();

        let finalizer = std::thread::spawn(move || execution.succeed());
        wait_for_phase_sync(&coordinator, "live-finalize-race", QueryPhase::Finalizing);
        release.wait();
        let raced_stats = reader.join().unwrap();
        finalizer.join().unwrap();

        let retained = coordinator.query_stats("live-finalize-race").unwrap();
        assert!(Arc::ptr_eq(&raced_stats, &retained));
        let state = coordinator.lock_state();
        assert_eq!(state.active_logical_capture_bytes, 0);
        let key = state.current.get("live-finalize-race").unwrap();
        let cache = &state.records.get(key).unwrap().live_stats_cache;
        let cache = cache.state.lock().unwrap();
        assert!(!cache.open);
        assert_eq!(cache.active_renderers, 0);
        assert_eq!(cache.waiting_readers, 0);
        assert!(cache.stats.is_none(), "live projection cache must be drained before active charge release");
        assert!(cache.terminal_stats.as_ref().is_some_and(|stats| Arc::ptr_eq(stats, &retained)));
    }

    #[test]
    fn artifact_egress_lease_bounds_concurrency_and_releases_on_response_drop() {
        let mut limited = config();
        limited.max_live_queries = 1;
        let coordinator = QueryCoordinator::new(limited);
        register(&coordinator, "egress").into_execution().unwrap().succeed();

        let mut first = coordinator.lease_runtime_artifact("egress").unwrap().unwrap();
        let first_frame = first.next_frame().expect("control frame");
        assert!(!first_frame.is_empty());
        let error = coordinator.lease_runtime_artifact("egress").unwrap_err();
        assert_eq!(error.kind(), QueryCoordinatorErrorKind::ArtifactEgressCapacity);
        drop(first);
        {
            let state = coordinator.artifact_egress.state.lock().unwrap();
            assert_eq!(state.streams, 0);
            assert_eq!(state.bytes, 0);
        }
        let replacement = coordinator.lease_runtime_artifact("egress").unwrap().unwrap();
        drop(replacement);
        let state = coordinator.artifact_egress.state.lock().unwrap();
        assert_eq!(state.streams, 0);
        assert_eq!(state.bytes, 0);
    }

    #[test]
    fn retained_unavailable_runtime_artifact_egress_is_one_control_frame() {
        let coordinator = QueryCoordinator::new(config());
        let planning = coordinator
            .register_with_id(
                "bounded-capture",
                "x".repeat(128 * 1024),
                None,
                None,
                RuntimeObservationConfig::bounded(16, 4096),
            )
            .unwrap();
        planning.into_execution().unwrap().succeed();

        {
            let state = coordinator.lock_state();
            let key = state.current.get("bounded-capture").unwrap();
            let record = state.records.get(key).unwrap();
            assert_eq!(record.phase, QueryPhase::Retained);
            assert_eq!(record.unavailable, Some(RetainedUnavailableReason::CaptureTruncated));
            assert!(
                record.retained.as_ref().unwrap().runtime_jsonl.is_some(),
                "the regression requires retained JSONL behind an unavailable control"
            );
        }

        let (control, jsonl) = coordinator.artifact_frames("bounded-capture").unwrap();
        let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
        assert_eq!(control["state"], "unavailable");
        assert_eq!(control["reason"], "capture_budget");
        assert!(jsonl.is_none(), "unavailable artifacts must not expose a JSONL frame");

        let mut response = coordinator.lease_runtime_artifact("bounded-capture").unwrap().unwrap();
        assert!(response.next_frame().is_some(), "unavailable artifact has one control frame");
        assert!(response.next_frame().is_none(), "unavailable artifact must have exactly one frame");
        drop(response);
        assert_eq!(coordinator.artifact_egress_usage(), (0, 0));
    }

    #[test]
    fn planning_capture_projects_and_drops_the_full_execution_plan() {
        let coordinator = QueryCoordinator::new(config());
        let _planning = register(&coordinator, "plan-release");
        let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(schema));
        coordinator.capture_planning("plan-release", Some(accepted_native_diagnostics()), Arc::clone(&plan));
        assert_eq!(Arc::strong_count(&plan), 1, "capture must retain bounded projections, not the full plan DAG");
    }

    #[test]
    fn hot_counter_saturation_is_truthfully_exported_as_incomplete() {
        let coordinator = QueryCoordinator::new(config());
        let execution = register(&coordinator, "hot-overflow").into_execution().unwrap();
        let hot = execution.hot_counters();
        hot.output_rows.store(usize::MAX, Ordering::Relaxed);
        hot.record_output(1, 1);
        execution.succeed();

        let stats = String::from_utf8(coordinator.query_stats("hot-overflow").unwrap().to_vec()).unwrap();
        assert!(stats.contains("capture_loss.counter_overflow"), "{stats}");
        assert!(stats.contains("# incomplete=true"), "{stats}");
        let (control, _) = coordinator.artifact_frames("hot-overflow").unwrap();
        let control: serde_json::Value = serde_json::from_slice(&control).unwrap();
        assert_eq!(control["reason"], "counter_overflow");
        assert_eq!(control["incomplete"], true);
    }

    #[test]
    fn full_stats_projection_distinguishes_context_and_runtime_failures() {
        for (query_id, stage, expected_started, expected_outcome) in [
            ("context-failure", "runtime_context", "false", "not_started"),
            ("execution-failure", "runtime_execution", "true", "failed"),
        ] {
            let coordinator = QueryCoordinator::new(config());
            let planning = register(&coordinator, query_id);
            let schema = Arc::new(Schema::new(vec![Field::new("one", DataType::Int64, false)]));
            coordinator.capture_planning(
                query_id,
                Some(accepted_native_diagnostics()),
                Arc::new(EmptyExec::new(schema)) as Arc<dyn ExecutionPlan>,
            );
            planning.into_execution().unwrap().fail("stream", captured_error_with_stage(stage));

            let payload = coordinator.query_stats(query_id).unwrap();
            let payload = std::str::from_utf8(&payload).unwrap();
            assert_eq!(tsv_value(payload, "execution", "native_runtime_started"), expected_started, "{payload}");
            assert_eq!(tsv_value(payload, "execution", "runtime_outcome"), expected_outcome, "{payload}");
        }
    }

    #[test]
    fn status_terminal_preserves_adapter_source_chain() {
        let coordinator = QueryCoordinator::new(config());
        let mut execution = register(&coordinator, "status-source").into_execution().unwrap();
        let error = crate::error::Error::new(
            crate::error::ErrorCode::Internal,
            "datafusion_nexus.server.flight.test.encode",
            "failed to encode response",
        )
        .with_source(std::io::Error::other("socket closed"));
        let status = crate::server::error::adapter_error_to_status(&error);
        execution.finish_status("flight_encode", &status);

        let payload = coordinator.query_stats("status-source").unwrap();
        let payload = std::str::from_utf8(&payload).unwrap();
        assert!(payload.contains("operation=datafusion_nexus.server.flight.test.encode"), "{payload}");
        assert!(payload.contains("source_chain=socket closed"), "{payload}");
        assert!(payload.contains("diagnostic=socket closed"), "{payload}");
    }
}
