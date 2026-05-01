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
use serde::Serialize;

use super::{
    super::observability::{QueryStatsPayload, StableValue},
    QueryCoordinator,
    types::{
        QueryCoordinatorError, QueryCoordinatorErrorKind, QueryHotCounterSnapshot, QueryPhase, RetainedQueryBundle,
        RetainedUnavailableReason,
    },
};

pub(in crate::server) const RUNTIME_ARTIFACT_CONTROL_SCHEMA: &str =
    "datafusion_nexus.query_runtime_artifact.control.v1";
pub(in crate::server) const TOMBSTONE_STATS_MAX_BYTES: u64 = 1024;
pub(in crate::server) const QUERY_STATS_HEADER_LINE: &str = "# datafusion_nexus_query_stats=1\n";
pub(in crate::server) type QueryArtifactFrames = (Arc<[u8]>, Option<Arc<[u8]>>);

#[cfg(test)]
pub(in crate::server) type CoordinatorRenderProbe = Box<dyn Fn(&'static str)>;

#[cfg(test)]
thread_local! {
    static COORDINATOR_RENDER_PROBE: RefCell<Option<CoordinatorRenderProbe>> = RefCell::new(None);
}

#[cfg(test)]
pub(in crate::server) fn run_coordinator_render_probe(operation: &'static str) {
    COORDINATOR_RENDER_PROBE.with(|probe| {
        if let Some(probe) = probe.borrow().as_ref() {
            probe(operation);
        }
    });
}

#[cfg(test)]
pub(in crate::server) fn install_coordinator_render_probe(
    probe: impl Fn(&'static str) + 'static,
) -> CoordinatorRenderProbeGuard {
    COORDINATOR_RENDER_PROBE.with(|slot| {
        assert!(slot.borrow_mut().replace(Box::new(probe)).is_none(), "render probe already installed");
    });
    CoordinatorRenderProbeGuard
}

#[cfg(test)]
pub(in crate::server) struct CoordinatorRenderProbeGuard;

#[cfg(test)]
impl Drop for CoordinatorRenderProbeGuard {
    fn drop(&mut self) {
        COORDINATOR_RENDER_PROBE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

#[cfg(test)]
pub(in crate::server) struct FinalizationTestGate {
    pub(in crate::server) entered: Barrier,
    pub(in crate::server) release: Barrier,
}

#[cfg(test)]
impl FinalizationTestGate {
    pub(in crate::server) fn new() -> Arc<Self> {
        Arc::new(Self { entered: Barrier::new(2), release: Barrier::new(2) })
    }

    pub(in crate::server) fn wait(&self) {
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
pub(in crate::server) struct RuntimeArtifactControl<'a> {
    pub(in crate::server) schema: &'static str,
    pub(in crate::server) version: u32,
    pub(in crate::server) query_id: &'a str,
    pub(in crate::server) state: &'static str,
    pub(in crate::server) retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(in crate::server) reason: Option<&'static str>,
    #[serde(skip_serializing_if = "is_false")]
    pub(in crate::server) incomplete: bool,
}

pub(in crate::server) const fn is_false(value: &bool) -> bool {
    !*value
}

pub(in crate::server) fn runtime_control(
    query_id: &str,
    state: &'static str,
    retryable: bool,
    reason: Option<&'static str>,
) -> Arc<[u8]> {
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

pub(in crate::server) fn incomplete_runtime_control(query_id: &str, reason: RetainedUnavailableReason) -> Arc<[u8]> {
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

pub(in crate::server) fn unavailable_stats_payload(
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

pub(in crate::server) struct ComposedTerminalArtifact {
    pub(in crate::server) bundle: Arc<RetainedQueryBundle>,
    pub(in crate::server) stats_incomplete: bool,
}

pub(in crate::server) fn compose_terminal_artifact<F>(
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

pub(in crate::server) fn unavailable_tombstone_stats(
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

pub(in crate::server) struct LimitedVecWriter {
    pub(in crate::server) bytes: Vec<u8>,
    pub(in crate::server) limit: usize,
    pub(in crate::server) overflowed: bool,
}

impl LimitedVecWriter {
    pub(in crate::server) fn new(limit: usize) -> Self {
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

pub(in crate::server) enum ArtifactFrameSnapshot {
    Encode { state: &'static str, retryable: bool, reason: Option<&'static str> },
    Incomplete { reason: RetainedUnavailableReason },
    // Public framing is state-shaped: unavailable is control-only, while a
    // sealed artifact owns exactly one complete JSONL frame.
    Control { control: Arc<[u8]> },
    Sealed { control: Arc<[u8]>, jsonl: Arc<[u8]> },
}

#[derive(Debug)]
pub(in crate::server) struct LiveStatsCache {
    pub(in crate::server) state: Mutex<LiveStatsCacheState>,
    pub(in crate::server) changed: Condvar,
    #[cfg(test)]
    pub(in crate::server) wait_observer: Mutex<Option<std::sync::mpsc::Sender<()>>>,
}

#[derive(Debug)]
pub(in crate::server) struct LiveStatsCacheState {
    pub(in crate::server) open: bool,
    pub(in crate::server) active_renderers: usize,
    pub(in crate::server) waiting_readers: usize,
    pub(in crate::server) rendering_key: Option<(u64, QueryHotCounterSnapshot)>,
    pub(in crate::server) completed_key: Option<(u64, QueryHotCounterSnapshot)>,
    pub(in crate::server) stats: Option<Arc<[u8]>>,
    pub(in crate::server) render_failed: bool,
    pub(in crate::server) terminal_stats: Option<Arc<[u8]>>,
    pub(in crate::server) abandoned: bool,
}

impl LiveStatsCache {
    pub(in crate::server) fn new() -> Self {
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

    pub(in crate::server) fn begin_read(self: &Arc<Self>, key: (u64, QueryHotCounterSnapshot)) -> LiveStatsRead {
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

    pub(in crate::server) fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.open = false;
        state.completed_key = None;
        state.stats = None;
        state.render_failed = false;
        self.changed.notify_all();
    }

    pub(in crate::server) fn wait_drained(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.active_renderers != 0 {
            state = self.changed.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    pub(in crate::server) fn publish_optional_terminal(&self, stats: Option<Arc<[u8]>>) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(!state.open, "terminal stats require a closed live cache");
        state.terminal_stats = stats;
        state.abandoned = state.terminal_stats.is_none();
        self.changed.notify_all();
    }

    pub(in crate::server) fn abandon(&self) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.open = false;
        state.stats = None;
        if state.terminal_stats.is_none() {
            state.abandoned = true;
        }
        self.changed.notify_all();
    }
}

pub(in crate::server) enum LiveStatsRead {
    Hit(Arc<[u8]>),
    Render(LiveStatsReadLease),
    Wait(LiveStatsWaiter),
    Failed,
    Closed,
}

pub(in crate::server) enum LiveStatsWaitResult {
    Hit(Arc<[u8]>),
    Failed,
    Retry,
    Closed,
}

pub(in crate::server) struct LiveStatsWaiter {
    pub(in crate::server) cache: Arc<LiveStatsCache>,
    pub(in crate::server) key: (u64, QueryHotCounterSnapshot),
    pub(in crate::server) completed: bool,
}

impl LiveStatsWaiter {
    pub(in crate::server) fn wait(mut self) -> LiveStatsWaitResult {
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

pub(in crate::server) struct LiveStatsReadLease {
    pub(in crate::server) cache: Arc<LiveStatsCache>,
    pub(in crate::server) key: (u64, QueryHotCounterSnapshot),
    pub(in crate::server) completed: bool,
}

impl LiveStatsReadLease {
    pub(in crate::server) fn complete(mut self, rendered: Arc<[u8]>) -> Option<Arc<[u8]>> {
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
pub(in crate::server) struct ArtifactEgressLimiter {
    pub(in crate::server) max_streams: usize,
    pub(in crate::server) max_bytes: u64,
    pub(in crate::server) state: Mutex<ArtifactEgressState>,
}

#[derive(Debug, Default)]
pub(in crate::server) struct ArtifactEgressState {
    pub(in crate::server) streams: usize,
    pub(in crate::server) bytes: u64,
    pub(in crate::server) peak_streams: usize,
    pub(in crate::server) peak_bytes: u64,
}

impl ArtifactEgressLimiter {
    pub(in crate::server) fn new(max_streams: usize, max_bytes: u64) -> Arc<Self> {
        Arc::new(Self { max_streams, max_bytes, state: Mutex::new(ArtifactEgressState::default()) })
    }

    pub(in crate::server) fn try_acquire(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<ArtifactEgressLease, QueryCoordinatorError> {
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
pub(in crate::server) struct ArtifactEgressLease {
    pub(in crate::server) limiter: Arc<ArtifactEgressLimiter>,
    pub(in crate::server) bytes: u64,
}

impl ArtifactEgressLease {
    pub(in crate::server) fn reconcile(&mut self, bytes: u64) -> Result<(), QueryCoordinatorError> {
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
pub(in crate::server) struct LeasedArtifactResponse {
    pub(in crate::server) frames: std::collections::VecDeque<Bytes>,
    pub(in crate::server) _lease: ArtifactEgressLease,
}

impl LeasedArtifactResponse {
    pub(in crate::server) fn next_frame(&mut self) -> Option<Bytes> {
        self.frames.pop_front()
    }
}

impl QueryCoordinator {
    #[cfg(test)]
    pub(in crate::server) fn artifact_egress_usage(&self) -> (usize, u64) {
        let state = self.artifact_egress.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.streams, state.bytes)
    }

    #[cfg(test)]
    pub(in crate::server) fn query_stats(&self, query_id: &str) -> Option<Arc<[u8]>> {
        self.query_stats_result(query_id).ok().flatten()
    }

    pub(in crate::server) fn query_stats_result(
        &self,
        query_id: &str,
    ) -> Result<Option<Arc<[u8]>>, QueryCoordinatorError> {
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

    pub(in crate::server) fn artifact_render_failed(query_id: &str) -> QueryCoordinatorError {
        QueryCoordinatorError::new(
            QueryCoordinatorErrorKind::ArtifactRenderFailed,
            "live query stats rendering failed before publication",
        )
        .with_query_id(query_id)
    }

    pub(in crate::server) fn artifact_frames(&self, query_id: &str) -> Option<QueryArtifactFrames> {
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

    pub(in crate::server) fn lease_query_stats(
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

    pub(in crate::server) fn lease_runtime_artifact(
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

    pub(in crate::server) fn lease_artifact_payloads(
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
}
