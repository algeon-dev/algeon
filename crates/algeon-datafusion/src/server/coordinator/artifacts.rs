//! Stats/runtime artifact composition, live stats cache, and egress leases.

#[cfg(test)]
use std::fmt;
#[cfg(test)]
use std::{cell::RefCell, sync::Barrier};
use std::{
    io,
    sync::{Arc, Condvar, Mutex},
};

use bytes::Bytes;
use query_engine::observability::{QueryAttemptReport, SealedAttemptProjection};
use serde::Serialize;

use super::{
    super::observability::{QUERY_STATS_HEADER_LINE, QueryStatsPayload, StableValue},
    QueryCoordinator,
    types::{
        QueryCoordinatorError, QueryCoordinatorErrorKind, QueryHotCounterSnapshot, QueryPhase, RetainedQueryBundle,
        RetainedUnavailableReason,
    },
};

pub(crate) const RUNTIME_ARTIFACT_CONTROL_SCHEMA: &str = "algeon_datafusion.query_runtime_artifact.control";
pub(crate) const TOMBSTONE_STATS_MAX_BYTES: u64 = 1024;
pub(crate) type QueryArtifactFrames = (Arc<[u8]>, Option<Arc<[u8]>>);

#[cfg(test)]
pub(crate) type CoordinatorRenderProbe = Box<dyn Fn(&'static str)>;

#[cfg(test)]
thread_local! {
    static COORDINATOR_RENDER_PROBE: RefCell<Option<CoordinatorRenderProbe>> = RefCell::new(None);
}

#[cfg(test)]
pub(crate) fn run_coordinator_render_probe(operation: &'static str) {
    COORDINATOR_RENDER_PROBE.with(|probe| {
        if let Some(probe) = probe.borrow().as_ref() {
            probe(operation);
        }
    });
}

#[cfg(test)]
pub(crate) fn install_coordinator_render_probe(probe: impl Fn(&'static str) + 'static) -> CoordinatorRenderProbeGuard {
    COORDINATOR_RENDER_PROBE.with(|slot| {
        assert!(slot.borrow_mut().replace(Box::new(probe)).is_none(), "render probe already installed");
    });
    CoordinatorRenderProbeGuard
}

#[cfg(test)]
pub(crate) struct CoordinatorRenderProbeGuard;

#[cfg(test)]
impl Drop for CoordinatorRenderProbeGuard {
    fn drop(&mut self) {
        COORDINATOR_RENDER_PROBE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

#[cfg(test)]
pub(crate) struct FinalizationTestGate {
    pub(crate) entered: Barrier,
    pub(crate) release: Barrier,
}

#[cfg(test)]
impl FinalizationTestGate {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self { entered: Barrier::new(2), release: Barrier::new(2) })
    }

    pub(crate) fn wait(&self) {
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

/// The first frame of every runtime artifact response. Every state
/// serializes every field, so a reader sees one shape whose values depend on
/// the state.
#[derive(Serialize)]
pub(crate) struct RuntimeArtifactControl<'a> {
    pub(crate) schema: &'static str,
    pub(crate) query_id: &'a str,
    pub(crate) state: &'static str,
    pub(crate) retryable: bool,
    pub(crate) reason: Option<&'static str>,
    pub(crate) incomplete: bool,
    /// Whether exactly one runtime JSONL frame follows this control in the
    /// same response. Only a `sealed` control of a statement that ran with a
    /// runtime observation sets it. Live and `unavailable` controls, and the
    /// sealed control of a statement without a runtime observation (such as
    /// a metadata statement), are the whole artifact.
    pub(crate) runtime_jsonl_follows: bool,
    /// The bounded projection of the attempt's sealed report, set on a
    /// terminal control rendered from that report and null otherwise. The
    /// complete report is in the footer of the runtime JSONL frame when that
    /// frame follows.
    pub(crate) attempt: Option<SealedAttemptProjection>,
}

pub(crate) fn runtime_control(
    query_id: &str,
    state: &'static str,
    retryable: bool,
    reason: Option<&'static str>,
) -> Arc<[u8]> {
    Arc::from(
        serde_json::to_vec(&RuntimeArtifactControl {
            schema: RUNTIME_ARTIFACT_CONTROL_SCHEMA,
            query_id,
            state,
            retryable,
            reason,
            incomplete: false,
            runtime_jsonl_follows: false,
            attempt: None,
        })
        .expect("runtime artifact control contains only infallibly serializable fields"),
    )
}

pub(crate) fn incomplete_runtime_control(query_id: &str, reason: RetainedUnavailableReason) -> Arc<[u8]> {
    Arc::from(
        serde_json::to_vec(&RuntimeArtifactControl {
            schema: RUNTIME_ARTIFACT_CONTROL_SCHEMA,
            query_id,
            state: "unavailable",
            retryable: false,
            reason: Some(reason.as_str()),
            incomplete: true,
            runtime_jsonl_follows: false,
            attempt: None,
        })
        .expect("runtime artifact control contains only infallibly serializable fields"),
    )
}

/// Encodes the terminal control of a finalized artifact from the attempt's
/// immutable report, when the statement has one. `runtime_jsonl` states
/// whether the artifact retains a runtime JSONL frame; an incomplete artifact
/// is `unavailable` and never exposes that frame.
pub(crate) fn sealed_runtime_control(
    query_id: &str,
    report: Option<&QueryAttemptReport>,
    runtime_jsonl: bool,
    incomplete_reason: Option<RetainedUnavailableReason>,
) -> Arc<[u8]> {
    Arc::from(
        serde_json::to_vec(&RuntimeArtifactControl {
            schema: RUNTIME_ARTIFACT_CONTROL_SCHEMA,
            query_id,
            state: incomplete_reason.map_or("sealed", |_| "unavailable"),
            // This remains delivery/frame retryability, never retry advice.
            retryable: false,
            reason: incomplete_reason.map(RetainedUnavailableReason::as_str),
            incomplete: incomplete_reason.is_some(),
            runtime_jsonl_follows: runtime_jsonl && incomplete_reason.is_none(),
            attempt: report.map(QueryAttemptReport::sealed_projection),
        })
        .expect("runtime artifact control contains only infallibly serializable fields"),
    )
}

pub(crate) fn unavailable_stats_payload(
    query_id: &str,
    outcome: &str,
    delivery_outcome: Option<&str>,
    reason: RetainedUnavailableReason,
    max_bytes: usize,
) -> QueryStatsPayload {
    let mut lines = vec![
        QUERY_STATS_HEADER_LINE.to_string(),
        format!("# query_id={}\n", StableValue(query_id)),
        format!("# outcome={outcome}\n"),
    ];
    if let Some(delivery_outcome) = delivery_outcome {
        lines.push(format!("# delivery_outcome={delivery_outcome}\n"));
    }
    lines.extend([
        "# sealed=true\n".to_string(),
        "# incomplete=true\n".to_string(),
        format!("# unavailable_reason={}\n", reason.as_str()),
    ]);
    QueryStatsPayload { bytes: bounded_stats_lines(lines, max_bytes), incomplete: true }
}

/// Live stats of a query registered beyond the active capture budget. It
/// keeps no capture to render, so its stats carry the running outcome and
/// the reason.
pub(crate) fn uncaptured_live_stats_payload(query_id: &str, max_bytes: usize) -> Arc<[u8]> {
    let lines = [
        QUERY_STATS_HEADER_LINE.to_string(),
        format!("# query_id={}\n", StableValue(query_id)),
        "# outcome=running\n".to_string(),
        "# sealed=false\n".to_string(),
        "# incomplete=true\n".to_string(),
        format!("# unavailable_reason={}\n", RetainedUnavailableReason::ActiveCaptureBudget.as_str()),
    ];
    bounded_stats_lines(lines, max_bytes)
}

fn bounded_stats_lines(lines: impl IntoIterator<Item = String>, max_bytes: usize) -> Arc<[u8]> {
    let max_bytes = max_bytes.min(TOMBSTONE_STATS_MAX_BYTES as usize);
    let mut bytes = Vec::with_capacity(max_bytes.min(1024));
    for line in lines {
        if line.len() > max_bytes.saturating_sub(bytes.len()) {
            break;
        }
        bytes.extend_from_slice(line.as_bytes());
    }
    Arc::from(bytes)
}

pub(crate) struct ComposedTerminalArtifact {
    pub(crate) bundle: Arc<RetainedQueryBundle>,
    pub(crate) stats_incomplete: bool,
}

pub(crate) fn compose_terminal_artifact<F>(
    control: Arc<[u8]>,
    runtime_jsonl: Option<Arc<[u8]>>,
    artifact_max_bytes: u64,
    render_stats: F,
) -> Option<ComposedTerminalArtifact>
where
    F: FnOnce(usize) -> QueryStatsPayload,
{
    let fixed_bytes =
        (control.len() as u64).checked_add(runtime_jsonl.as_ref().map_or(0, |jsonl| jsonl.len() as u64))?;
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
    let charge_bytes = fixed_bytes.checked_add(stats.len() as u64)?;
    debug_assert!(charge_bytes <= artifact_max_bytes);
    Some(ComposedTerminalArtifact {
        bundle: Arc::new(RetainedQueryBundle { stats, runtime_control: control, runtime_jsonl, charge_bytes }),
        stats_incomplete,
    })
}

pub(crate) fn unavailable_tombstone_stats(
    query_id: &str,
    outcome: &str,
    delivery_outcome: Option<&str>,
    reason: RetainedUnavailableReason,
    artifact_max_bytes: u64,
) -> Option<Arc<[u8]>> {
    let control = incomplete_runtime_control(query_id, reason);
    compose_terminal_artifact(control, None, artifact_max_bytes, |remaining| {
        unavailable_stats_payload(query_id, outcome, delivery_outcome, reason, remaining)
    })
    .map(|artifact| Arc::clone(&artifact.bundle.stats))
}

pub(crate) struct LimitedVecWriter {
    pub(crate) bytes: Vec<u8>,
    pub(crate) limit: usize,
    pub(crate) overflowed: bool,
}

impl LimitedVecWriter {
    pub(crate) fn new(limit: usize) -> Self {
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

pub(crate) enum ArtifactFrameSnapshot {
    Encode { state: &'static str, retryable: bool, reason: Option<&'static str> },
    Incomplete { reason: RetainedUnavailableReason },
    // An unavailable artifact, and a sealed one without a runtime
    // observation, is control-only; any other sealed artifact owns exactly
    // one complete JSONL frame, as its control's `runtime_jsonl_follows`
    // states.
    Control { control: Arc<[u8]> },
    Sealed { control: Arc<[u8]>, jsonl: Arc<[u8]> },
}

#[derive(Debug)]
pub(crate) struct LiveStatsCache {
    pub(crate) state: Mutex<LiveStatsCacheState>,
    pub(crate) changed: Condvar,
    #[cfg(test)]
    pub(crate) wait_observer: Mutex<Option<std::sync::mpsc::Sender<()>>>,
}

#[derive(Debug)]
pub(crate) struct LiveStatsCacheState {
    pub(crate) open: bool,
    pub(crate) active_renderers: usize,
    pub(crate) waiting_readers: usize,
    pub(crate) rendering_key: Option<(u64, QueryHotCounterSnapshot)>,
    pub(crate) completed_key: Option<(u64, QueryHotCounterSnapshot)>,
    pub(crate) stats: Option<Arc<[u8]>>,
    pub(crate) render_failed: bool,
    pub(crate) terminal_stats: Option<Arc<[u8]>>,
    pub(crate) abandoned: bool,
}

fn fail_closed_live_stats_cache(state: &mut LiveStatsCacheState) {
    // Once the lease counters disagree with their owners, serving another
    // mutable projection would make the cache's single-flight state
    // untrustworthy. A terminal snapshot, if already installed, stays valid.
    state.open = false;
    state.rendering_key = None;
    state.completed_key = None;
    state.stats = None;
    state.render_failed = true;
    state.abandoned = state.terminal_stats.is_none();
}

impl LiveStatsCache {
    pub(crate) fn new() -> Self {
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

    pub(crate) fn begin_read(self: &Arc<Self>, key: (u64, QueryHotCounterSnapshot)) -> LiveStatsRead {
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

    pub(crate) fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.open = false;
        state.completed_key = None;
        state.stats = None;
        state.render_failed = false;
        self.changed.notify_all();
    }

    pub(crate) fn wait_drained(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.active_renderers != 0 {
            state = self.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    pub(crate) fn publish_optional_terminal(&self, stats: Option<Arc<[u8]>>) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(!state.open, "terminal stats require a closed live cache");
        state.terminal_stats = stats;
        state.abandoned = state.terminal_stats.is_none();
        self.changed.notify_all();
    }

    pub(crate) fn abandon(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.open = false;
        state.stats = None;
        if state.terminal_stats.is_none() {
            state.abandoned = true;
        }
        self.changed.notify_all();
    }
}

pub(crate) enum LiveStatsRead {
    Hit(Arc<[u8]>),
    Render(LiveStatsReadLease),
    Wait(LiveStatsWaiter),
    Failed,
    Closed,
}

pub(crate) enum LiveStatsWaitResult {
    Hit(Arc<[u8]>),
    Failed,
    Retry,
    Closed,
}

pub(crate) struct LiveStatsWaiter {
    pub(crate) cache: Arc<LiveStatsCache>,
    pub(crate) key: (u64, QueryHotCounterSnapshot),
    pub(crate) completed: bool,
}

impl LiveStatsWaiter {
    pub(crate) fn wait(mut self) -> LiveStatsWaitResult {
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
        let underflow = {
            let mut state = self.cache.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            match state.waiting_readers.checked_sub(1) {
                Some(waiting_readers) => {
                    state.waiting_readers = waiting_readers;
                    self.cache.changed.notify_all();
                    false
                },
                None => {
                    fail_closed_live_stats_cache(&mut state);
                    self.cache.changed.notify_all();
                    true
                },
            }
        };
        if underflow && !std::thread::panicking() {
            panic!("live stats waiter count underflow");
        }
    }
}

pub(crate) struct LiveStatsReadLease {
    pub(crate) cache: Arc<LiveStatsCache>,
    pub(crate) key: (u64, QueryHotCounterSnapshot),
    pub(crate) completed: bool,
}

impl LiveStatsReadLease {
    pub(crate) fn complete(mut self, rendered: Arc<[u8]>) -> Option<Arc<[u8]>> {
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
        let underflow = {
            let mut state = self.cache.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.open {
                state.completed_key = Some(self.key);
                state.stats = None;
                state.render_failed = true;
            }
            state.rendering_key = None;
            match state.active_renderers.checked_sub(1) {
                Some(active_renderers) => {
                    state.active_renderers = active_renderers;
                    self.cache.changed.notify_all();
                    false
                },
                None => {
                    fail_closed_live_stats_cache(&mut state);
                    self.cache.changed.notify_all();
                    true
                },
            }
        };
        if underflow && !std::thread::panicking() {
            panic!("live stats renderer count underflow");
        }
    }
}

#[derive(Debug)]
pub(crate) struct ArtifactEgressLimiter {
    pub(crate) max_streams: usize,
    pub(crate) max_bytes: u64,
    pub(crate) state: Mutex<ArtifactEgressState>,
}

#[derive(Debug, Default)]
pub(crate) struct ArtifactEgressState {
    pub(crate) streams: usize,
    pub(crate) bytes: u64,
    pub(crate) peak_streams: usize,
    pub(crate) peak_bytes: u64,
    pub(crate) closed: bool,
}

impl ArtifactEgressLimiter {
    pub(crate) fn new(max_streams: usize, max_bytes: u64) -> Arc<Self> {
        Arc::new(Self { max_streams, max_bytes, state: Mutex::new(ArtifactEgressState::default()) })
    }

    pub(crate) fn try_acquire(self: &Arc<Self>, bytes: u64) -> Result<ArtifactEgressLease, QueryCoordinatorError> {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::ArtifactEgressCapacity,
                "runtime artifact response capacity is unavailable after an accounting failure",
            ));
        }
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
pub(crate) struct ArtifactEgressLease {
    pub(crate) limiter: Arc<ArtifactEgressLimiter>,
    pub(crate) bytes: u64,
}

impl ArtifactEgressLease {
    pub(crate) fn reconcile(&mut self, bytes: u64) -> Result<(), QueryCoordinatorError> {
        let mut state = self.limiter.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed {
            return Err(QueryCoordinatorError::new(
                QueryCoordinatorErrorKind::ArtifactEgressCapacity,
                "runtime artifact response capacity is unavailable after an accounting failure",
            ));
        }
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
        let underflow = {
            let mut state = self.limiter.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            match (state.streams.checked_sub(1), state.bytes.checked_sub(self.bytes)) {
                (Some(streams), Some(bytes)) => {
                    state.streams = streams;
                    state.bytes = bytes;
                    false
                },
                _ => {
                    state.streams = self.limiter.max_streams;
                    state.bytes = self.limiter.max_bytes;
                    state.closed = true;
                    true
                },
            }
        };
        if underflow && !std::thread::panicking() {
            panic!("artifact egress lease accounting underflow");
        }
    }
}

#[derive(Debug)]
pub(crate) struct LeasedArtifactResponse {
    pub(crate) frames: std::collections::VecDeque<Bytes>,
    pub(crate) _lease: ArtifactEgressLease,
}

impl LeasedArtifactResponse {
    pub(crate) fn next_frame(&mut self) -> Option<Bytes> {
        self.frames.pop_front()
    }
}

impl QueryCoordinator {
    #[cfg(test)]
    pub(crate) fn artifact_egress_usage(&self) -> (usize, u64) {
        let state = self.artifact_egress.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.streams, state.bytes)
    }

    #[cfg(test)]
    pub(crate) fn query_stats(&self, query_id: &str) -> Option<Arc<[u8]>> {
        self.query_stats_result(query_id).ok().flatten()
    }

    pub(crate) fn query_stats_result(&self, query_id: &str) -> Result<Option<Arc<[u8]>>, QueryCoordinatorError> {
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
                        // Only a query registered beyond the active capture
                        // budget is live without a capture; finalization
                        // closes the cache before it takes one.
                        let Some(captured) = record.captured.as_ref().cloned() else {
                            let max_bytes = usize::try_from(record.artifact_max_bytes).unwrap_or(usize::MAX);
                            let stats = uncaptured_live_stats_payload(query_id, max_bytes);
                            drop(state);
                            return Ok(read.complete(stats));
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

    pub(crate) fn artifact_render_failed(query_id: &str) -> QueryCoordinatorError {
        QueryCoordinatorError::new(
            QueryCoordinatorErrorKind::ArtifactRenderFailed,
            "live query stats rendering failed before publication",
        )
        .with_query_id(query_id)
    }

    pub(crate) fn artifact_frames(&self, query_id: &str) -> Option<QueryArtifactFrames> {
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
                    } else if let Some(jsonl) = bundle.runtime_jsonl.as_ref() {
                        ArtifactFrameSnapshot::Sealed {
                            control: Arc::clone(&bundle.runtime_control),
                            jsonl: Arc::clone(jsonl),
                        }
                    } else {
                        ArtifactFrameSnapshot::Control { control: Arc::clone(&bundle.runtime_control) }
                    }
                },
                QueryPhase::Tombstone => ArtifactFrameSnapshot::Incomplete {
                    reason: record.unavailable.expect("tombstone phase owns an unavailable reason"),
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

    pub(crate) fn lease_query_stats(
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
        let actual_bytes = stats.len() as u64;
        lease.reconcile(actual_bytes)?;
        let frames = std::iter::once(stats).map(Bytes::from_owner).collect();
        Ok(Some(LeasedArtifactResponse { frames, _lease: lease }))
    }

    pub(crate) fn lease_runtime_artifact(
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

    pub(crate) fn lease_artifact_payloads(
        &self,
        payloads: impl IntoIterator<Item = Arc<[u8]>>,
    ) -> Result<Option<LeasedArtifactResponse>, QueryCoordinatorError> {
        let payloads = payloads.into_iter().collect::<Vec<_>>();
        if payloads.is_empty() {
            return Ok(None);
        }
        let bytes = payloads.iter().try_fold(0u64, |total, payload| total.checked_add(payload.len() as u64));
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
}
