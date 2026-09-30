//! Frontend-neutral graph execution services and admission contracts.
//!
//! The adapter core owns process-local coordination and the DataFusion plan
//! discovery SPI. Graph frontends register implementations without creating a
//! core-to-frontend dependency.
//!
//! Exact-request singleflight for expensive BFS calls.
//!
//! This holds only a transient published result until its joined followers
//! consume it. The graph identity is a coalescing key, not a
//! device-resident cache claim; the resident cache owns any retained device
//! allocation separately.
//!
//! libcugraph multi-source BFS produces one nearest-source forest, not one
//! independent result per source. Serving coalescing therefore groups only
//! requests with identical traversal semantics and publishes the completed
//! result to followers. A GPU-resident publication borrows the leader
//! attempt's attached domain members while followers copy it, so the leader
//! rendezvouses with its joined followers before it may release: publication
//! resolves the generation, and the leader waits until every joined follower
//! has copied or dropped the handoff.

use std::sync::{Mutex, OnceLock};
#[cfg(feature = "cugraph")]
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

#[cfg(feature = "cugraph")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "cugraph")]
use datafusion::execution::memory_pool::MemoryReservation;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    physical_plan::ExecutionPlan,
};
use query_engine::admission::AttemptCapabilityEvidence;
#[cfg(feature = "cugraph")]
use query_engine::{
    observability::AttemptId,
    plan::ResidentGraphKey,
    runtime::{ExecutionCancellation, cache_observation_fingerprint, resident::GpuResultHandle},
};
#[cfg(feature = "cugraph")]
use tokio::sync::{Notify, watch};
#[cfg(feature = "cugraph")]
use tokio::time::sleep;

#[cfg(feature = "cugraph")]
use crate::error::{Error, ErrorCode, Result};
use crate::native_exec::SealedAttemptMetricsTarget;

#[cfg(feature = "cugraph")]
const OPERATION: &str = "algeon_datafusion.graph_execution.bfs_singleflight";
#[cfg(feature = "cugraph")]
const DEFAULT_MAX_BATCH_WIDTH: usize = 32;
/// The coordinator remains a serving primitive, not an unbounded event
/// recorder. A fresh same-key cohort is far smaller than this bounded projection.
#[cfg(feature = "cugraph")]
const MAX_BFS_SINGLEFLIGHT_EVENT_SNAPSHOTS: usize = 256;

/// Frontend-neutral graph-model diagnostics exposed by a graph execution node.
///
/// Capability evidence issuance remains in the adapter planner after it has
/// recognized this crate-private execution type; graph frontends do not get a
/// caller-controlled evidence SPI.
pub(crate) trait GraphExecutionPlan: Send + Sync {
    /// Issues capability evidence for the exact native graph fragment this
    /// execution node submits after staging its GPU-resident inputs.
    fn graph_execution_capability_evidence(&self) -> DataFusionResult<AttemptCapabilityEvidence>;

    /// Returns the one fixed-cardinality post-seal metric target for this
    /// graph execution root. The adapter discovers it through this neutral
    /// SPI, so core execution does not depend on a graph frontend.
    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget;

    /// Returns a typed rejection when this graph root intentionally owns its
    /// result at the host boundary and cannot participate as a GPU producer.
    fn gpu_resident_output_rejection(&self) -> Option<DataFusionError> {
        None
    }
}

/// A graph frontend hook that recognizes one of its DataFusion execution nodes.
pub(crate) type GraphExecutionPlanDiscovery = for<'a> fn(&'a dyn ExecutionPlan) -> Option<&'a dyn GraphExecutionPlan>;

static GRAPH_EXECUTION_PLAN_DISCOVERIES: OnceLock<Mutex<Vec<GraphExecutionPlanDiscovery>>> = OnceLock::new();

/// Register a graph frontend's execution-plan recognizer.
///
/// Registration is idempotent for the same function and does not retain a
/// frontend value. The core planner only sees the frontend-neutral trait.
#[cfg(any(feature = "cugraph", test))]
pub(crate) fn register_graph_execution_plan_discovery(discovery: GraphExecutionPlanDiscovery) {
    let mut discoveries = GRAPH_EXECUTION_PLAN_DISCOVERIES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("graph execution-plan discovery registry mutex cannot be poisoned");
    if !discoveries.iter().any(|candidate| std::ptr::fn_addr_eq(*candidate, discovery)) {
        discoveries.push(discovery);
    }
}

pub(crate) fn graph_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
    let discoveries = {
        GRAPH_EXECUTION_PLAN_DISCOVERIES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .expect("graph execution-plan discovery registry mutex cannot be poisoned")
            .clone()
    };
    discoveries.iter().find_map(|discovery| discovery(plan))
}

/// Shared process-local services carried by native execution configuration.
///
/// The state is graph-runtime owned and contains no SQL frontend types.
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Default)]
pub(crate) struct GraphExecutionServices {
    bfs_singleflight: GraphBfsSingleflight,
}

#[cfg(feature = "cugraph")]
impl GraphExecutionServices {
    pub(crate) fn bfs_singleflight(&self) -> GraphBfsSingleflight {
        self.bfs_singleflight.clone()
    }

    pub(crate) fn bfs_singleflight_snapshot(&self) -> GraphBfsSingleflightSnapshot {
        self.bfs_singleflight.snapshot()
    }

    pub(crate) fn bfs_singleflight_enabled(&self) -> bool {
        true
    }
}

/// Exact compatibility key for one device-local independent BFS traversal.
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GraphBfsSingleflightKey {
    pub(crate) device_ordinal: usize,
    pub(crate) resident_graph_key: Option<ResidentGraphKey>,
    pub(crate) source_vertex: i64,
    pub(crate) normalized_output: bool,
    pub(crate) projection: Option<Vec<usize>>,
    pub(crate) limit: Option<usize>,
    pub(crate) max_output_staging_bytes: usize,
}

/// Additive diagnostics for exact-request BFS serving coalescing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphBfsSingleflightSnapshot {
    /// Requests accepted by the exact deep-BFS eligibility guard.
    pub eligible_requests: u64,
    /// Leader traversal attempts submitted to native query execution.
    pub physical_traversals: u64,
    /// Followers that reused and imported a published leader result.
    pub follower_copies: u64,
    /// Device-local coalescing generations created.
    pub groups_formed: u64,
    /// Generations created because an earlier group reached its width cap.
    pub overflow_groups: u64,
    /// Leader executions that failed and released followers to retry.
    pub leader_failures: u64,
    /// Leader or follower waits dropped or cooperatively cancelled.
    pub cancelled_waiters: u64,
    /// Largest number of logical requests joined to one generation.
    pub peak_width: usize,
    /// Bounded coordinator-emitted events. These are not derived from the
    /// aggregate counters above and contain only one-way fingerprints.
    pub events: Vec<GraphBfsSingleflightEventSnapshot>,
}

/// One bounded BFS singleflight fact emitted at the coordinator that owns the
/// group lifecycle. Exact key and caller correlation are one-way fingerprints
/// so diagnostic snapshots never disclose source identities or query ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphBfsSingleflightEventSnapshot {
    /// Per-coordinator monotonically increasing sequence number.
    pub sequence: u64,
    pub device_ordinal: usize,
    pub singleflight_key_fingerprint: String,
    pub correlation_fingerprint: Option<String>,
    pub generation: u64,
    /// `leader_join`, `follower_join`, `leader_traversal`,
    /// `leader_published`, `follower_copy`, or `leader_failed`.
    pub action: &'static str,
}

/// Process-local coordinator shared by clones of one backend execution config.
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone)]
pub(crate) struct GraphBfsSingleflight {
    inner: Arc<SingleflightInner>,
}

#[cfg(feature = "cugraph")]
impl Default for GraphBfsSingleflight {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_BATCH_WIDTH)
    }
}

#[cfg(feature = "cugraph")]
impl GraphBfsSingleflight {
    fn new(max_batch_width: usize) -> Self {
        debug_assert!(max_batch_width > 0);
        Self {
            inner: Arc::new(SingleflightInner {
                max_batch_width,
                state: Mutex::new(SingleflightState::default()),
                counters: SingleflightCounters::default(),
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn join(&self, key: GraphBfsSingleflightKey) -> Result<GraphBfsSingleflightJoin> {
        self.join_with_correlation(key, None)
    }

    /// Joins a coalescing group while retaining only the caller-supplied
    /// one-way correlation fingerprint in its bounded event projection.
    #[cfg(test)]
    pub(crate) fn join_with_correlation(
        &self,
        key: GraphBfsSingleflightKey,
        correlation_fingerprint: Option<String>,
    ) -> Result<GraphBfsSingleflightJoin> {
        self.join_with_attempt(key, correlation_fingerprint, None)
    }

    /// Joins a coalescing group and retains the leader's engine-minted attempt
    /// id only in the in-process follower state. It is projected later by the
    /// follower's own sealed attempt snapshot, never by coordinator events.
    pub(crate) fn join_with_attempt(
        &self,
        key: GraphBfsSingleflightKey,
        correlation_fingerprint: Option<String>,
        leader_attempt_id: Option<AttemptId>,
    ) -> Result<GraphBfsSingleflightJoin> {
        self.inner.counters.eligible_requests.fetch_add(1, Ordering::Relaxed);
        let mut state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
        let follower = state.groups.get_mut(&key).and_then(|groups| {
            groups.iter_mut().find(|group| group.participants < self.inner.max_batch_width).map(|group| {
                group.participants += 1;
                update_peak(&self.inner.counters.peak_width, group.participants);
                (group.generation, group.leader_attempt_id, group.sender.subscribe())
            })
        });
        if let Some((generation, leader_attempt_id, receiver)) = follower {
            state.record_event(&key, correlation_fingerprint.clone(), generation, "follower_join");
            return Ok(GraphBfsSingleflightJoin::Follower(GraphBfsFollower {
                coordinator: self.clone(),
                key,
                generation,
                leader_attempt_id,
                correlation_fingerprint,
                receiver,
                completed: false,
                handoff_settled: AtomicBool::new(false),
            }));
        }

        state.next_generation = state.next_generation.saturating_add(1);
        let generation = state.next_generation;
        let (sender, _receiver) = watch::channel(PublishedResult::Pending);
        let overflow = {
            let groups = state.groups.entry(key.clone()).or_default();
            let overflow = !groups.is_empty();
            groups.push(SingleflightGroup { generation, participants: 1, leader_attempt_id, sender });
            overflow
        };
        if overflow {
            self.inner.counters.overflow_groups.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.counters.groups_formed.fetch_add(1, Ordering::Relaxed);
        update_peak(&self.inner.counters.peak_width, 1);
        state.record_event(&key, correlation_fingerprint.clone(), generation, "leader_join");
        Ok(GraphBfsSingleflightJoin::Leader(GraphBfsLeader {
            coordinator: self.clone(),
            key,
            generation,
            correlation_fingerprint,
            traversal_submitted: false,
            resolved: false,
        }))
    }

    fn record_follower_copy(
        &self,
        key: &GraphBfsSingleflightKey,
        generation: u64,
        correlation_fingerprint: Option<String>,
    ) {
        self.inner.counters.follower_copies.fetch_add(1, Ordering::Relaxed);
        self.record_event(key, correlation_fingerprint, generation, "follower_copy");
    }

    pub(crate) fn snapshot(&self) -> GraphBfsSingleflightSnapshot {
        let counters = &self.inner.counters;
        GraphBfsSingleflightSnapshot {
            eligible_requests: counters.eligible_requests.load(Ordering::Acquire),
            physical_traversals: counters.physical_traversals.load(Ordering::Acquire),
            follower_copies: counters.follower_copies.load(Ordering::Acquire),
            groups_formed: counters.groups_formed.load(Ordering::Acquire),
            overflow_groups: counters.overflow_groups.load(Ordering::Acquire),
            leader_failures: counters.leader_failures.load(Ordering::Acquire),
            cancelled_waiters: counters.cancelled_waiters.load(Ordering::Acquire),
            peak_width: usize::try_from(counters.peak_width.load(Ordering::Acquire)).unwrap_or(usize::MAX),
            events: self
                .inner
                .state
                .lock()
                .map(|state| state.recent_events.iter().cloned().collect())
                .unwrap_or_default(),
        }
    }

    fn publish(
        &self,
        key: &GraphBfsSingleflightKey,
        generation: u64,
        correlation_fingerprint: Option<String>,
        result: PublishedResult,
    ) -> Result<()> {
        let sender = {
            let mut state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
            let published = matches!(&result, PublishedResult::Ready(_));
            let gpu_handoff_followers = match &result {
                PublishedResult::Ready(publication) if publication.kind() == GraphBfsPublicationKind::Gpu => state
                    .groups
                    .get(key)
                    .and_then(|groups| groups.iter().find(|group| group.generation == generation))
                    .map(|group| group.participants.saturating_sub(1)),
                _ => None,
            };
            let sender = remove_group(&mut state, key, generation).map(|group| group.sender);
            if published && sender.is_some() {
                state.record_event(key, correlation_fingerprint, generation, "leader_published");
            }
            if let Some(follower_count) = gpu_handoff_followers.filter(|count| *count > 0) {
                state
                    .gpu_handoff_drains
                    .insert((key.clone(), generation), Arc::new(GpuHandoffDrain::new(follower_count)));
            }
            sender
        };
        if let Some(sender) = sender {
            sender.send_replace(result);
        }
        Ok(())
    }

    /// Clones the GPU handoff drain for a published generation, if the leader
    /// published a GPU-resident result to joined followers. Host publications
    /// never create one: host batches need no leader device attachment.
    fn handoff_drain(&self, key: &GraphBfsSingleflightKey, generation: u64) -> Result<Option<Arc<GpuHandoffDrain>>> {
        let state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
        Ok(state.gpu_handoff_drains.get(&(key.clone(), generation)).cloned())
    }

    /// Signals one joined follower's copy settlement for a GPU handoff drain.
    /// A copy settles exactly once per follower; a drop settles through
    /// [`leave_follower`](Self::leave_follower), which checks the group and
    /// the drain under one lock hold. Followers of host publications or
    /// retried generations find no drain and are unaffected.
    fn signal_handoff_copy_settled(&self, key: &GraphBfsSingleflightKey, generation: u64) {
        let Ok(mut state) = self.inner.state.lock() else {
            tracing::warn!("failed to signal poisoned BFS singleflight GPU handoff drain");
            return;
        };
        state.signal_handoff_drain_locked(key, generation);
    }

    fn participant_count(&self, key: &GraphBfsSingleflightKey, generation: u64) -> Result<usize> {
        let state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
        Ok(state
            .groups
            .get(key)
            .and_then(|groups| groups.iter().find(|group| group.generation == generation))
            .map_or(0, |group| group.participants))
    }

    fn abandon(
        &self,
        key: &GraphBfsSingleflightKey,
        generation: u64,
        correlation_fingerprint: Option<String>,
        cancelled: bool,
    ) {
        let sender = match self.inner.state.lock() {
            Ok(mut state) => {
                let sender = remove_group(&mut state, key, generation).map(|group| group.sender);
                if !cancelled && sender.is_some() {
                    state.record_event(key, correlation_fingerprint, generation, "leader_failed");
                }
                sender
            },
            Err(_error) => {
                tracing::warn!("failed to abandon poisoned BFS singleflight state");
                None
            },
        };
        if cancelled {
            self.inner.counters.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
        } else {
            self.inner.counters.leader_failures.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(sender) = sender {
            sender.send_replace(PublishedResult::Retry);
        }
    }

    /// Leaves a follower's generation through one locked section: a generation
    /// that already published a GPU handoff settles this follower's drain
    /// share, otherwise the live group shrinks before the leader counts it.
    /// Publish holds the same lock, so it cannot land between the drain check
    /// and the group update and strand a counted follower without its settle.
    fn leave_follower(&self, key: &GraphBfsSingleflightKey, generation: u64, cancelled: bool) {
        if cancelled {
            self.inner.counters.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
        }
        let Ok(mut state) = self.inner.state.lock() else {
            tracing::warn!("failed to leave poisoned BFS singleflight state");
            return;
        };
        if state.signal_handoff_drain_locked(key, generation) {
            return;
        }
        let Some(groups) = state.groups.get_mut(key) else {
            return;
        };
        if let Some(group) = groups.iter_mut().find(|group| group.generation == generation) {
            group.participants = group.participants.saturating_sub(1);
        }
    }

    fn record_event(
        &self,
        key: &GraphBfsSingleflightKey,
        correlation_fingerprint: Option<String>,
        generation: u64,
        action: &'static str,
    ) {
        let Ok(mut state) = self.inner.state.lock() else {
            tracing::warn!("failed to record BFS singleflight event after state mutex poisoning");
            return;
        };
        state.record_event(key, correlation_fingerprint, generation, action);
    }
}

#[cfg(feature = "cugraph")]
#[derive(Debug)]
struct SingleflightInner {
    max_batch_width: usize,
    state: Mutex<SingleflightState>,
    counters: SingleflightCounters,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Default)]
struct SingleflightState {
    groups: HashMap<GraphBfsSingleflightKey, Vec<SingleflightGroup>>,
    gpu_handoff_drains: HashMap<(GraphBfsSingleflightKey, u64), Arc<GpuHandoffDrain>>,
    next_generation: u64,
    next_event_sequence: u64,
    recent_events: VecDeque<GraphBfsSingleflightEventSnapshot>,
}

#[cfg(feature = "cugraph")]
impl SingleflightState {
    /// Signals one follower's share of this generation's GPU handoff drain,
    /// removing the drain at zero. Returns whether a drain existed. The
    /// caller holds the state lock, so the check, the signal, and the removal
    /// are atomic with respect to publish and to other followers leaving.
    fn signal_handoff_drain_locked(&mut self, key: &GraphBfsSingleflightKey, generation: u64) -> bool {
        let Some(drain) = self.gpu_handoff_drains.get(&(key.clone(), generation)).cloned() else {
            return false;
        };
        drain.signal_copy_settled();
        if drain.remaining.load(Ordering::Acquire) == 0 {
            self.gpu_handoff_drains.remove(&(key.clone(), generation));
        }
        true
    }

    fn record_event(
        &mut self,
        key: &GraphBfsSingleflightKey,
        correlation_fingerprint: Option<String>,
        generation: u64,
        action: &'static str,
    ) {
        if self.recent_events.len() == MAX_BFS_SINGLEFLIGHT_EVENT_SNAPSHOTS {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(GraphBfsSingleflightEventSnapshot {
            sequence: self.next_event_sequence,
            device_ordinal: key.device_ordinal,
            singleflight_key_fingerprint: graph_bfs_singleflight_key_fingerprint(key),
            correlation_fingerprint,
            generation,
            action,
        });
        self.next_event_sequence = self.next_event_sequence.saturating_add(1);
    }
}

#[cfg(feature = "cugraph")]
#[derive(Debug)]
struct SingleflightGroup {
    generation: u64,
    participants: usize,
    leader_attempt_id: Option<AttemptId>,
    sender: watch::Sender<PublishedResult>,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Default)]
struct SingleflightCounters {
    eligible_requests: AtomicU64,
    physical_traversals: AtomicU64,
    follower_copies: AtomicU64,
    groups_formed: AtomicU64,
    overflow_groups: AtomicU64,
    leader_failures: AtomicU64,
    cancelled_waiters: AtomicU64,
    peak_width: AtomicU64,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone)]
enum PublishedResult {
    Pending,
    Ready(GraphBfsPublishedResult),
    Retry,
}

/// One exact BFS result retained only until every joined follower has imported
/// or returned it. GPU publication keeps the producer's live grant attached;
/// a follower reaches the engine's explicit cross-attempt GPU handoff before
/// any downstream operator consumes it.
#[cfg(feature = "cugraph")]
#[derive(Clone)]
pub(crate) enum GraphBfsPublishedResult {
    Host { batches: Arc<Vec<RecordBatch>>, reservation: Arc<MemoryReservation> },
    Gpu(Box<GpuResultHandle>),
}

/// The storage boundary selected by the singleflight leader.
#[cfg(feature = "cugraph")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphBfsPublicationKind {
    Host,
    Gpu,
}

/// The explicit conversion, if any, required to deliver a published BFS
/// result to a follower's requested sink.
#[cfg(feature = "cugraph")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphBfsFollowerRoute {
    ReuseHost,
    UploadHost,
    HandoffGpu,
    MaterializeGpu,
}

#[cfg(feature = "cugraph")]
impl GraphBfsFollowerRoute {
    pub(crate) const fn for_gpu_follower(publication: GraphBfsPublicationKind) -> Self {
        match publication {
            GraphBfsPublicationKind::Host => Self::UploadHost,
            GraphBfsPublicationKind::Gpu => Self::HandoffGpu,
        }
    }

    pub(crate) const fn for_host_follower(publication: GraphBfsPublicationKind) -> Self {
        match publication {
            GraphBfsPublicationKind::Host => Self::ReuseHost,
            GraphBfsPublicationKind::Gpu => Self::MaterializeGpu,
        }
    }
}

#[cfg(feature = "cugraph")]
impl GraphBfsPublishedResult {
    /// Constructs the host representation required by a host-sink leader.
    pub(crate) fn new(batches: Arc<Vec<RecordBatch>>, reservation: Arc<MemoryReservation>) -> Self {
        Self::Host { batches, reservation }
    }

    /// Retains a GPU-resident leader output without forcing its complete
    /// materialization merely to satisfy same-device followers.
    pub(crate) fn gpu(handle: GpuResultHandle) -> Self {
        Self::Gpu(Box::new(handle))
    }

    pub(crate) const fn kind(&self) -> GraphBfsPublicationKind {
        match self {
            Self::Host { .. } => GraphBfsPublicationKind::Host,
            Self::Gpu(_) => GraphBfsPublicationKind::Gpu,
        }
    }

    pub(crate) fn host_batches(&self) -> Option<&[RecordBatch]> {
        match self {
            Self::Host { batches, .. } => Some(batches.as_slice()),
            Self::Gpu(_) => None,
        }
    }

    pub(crate) fn host_shared_parts(&self) -> Option<(Arc<Vec<RecordBatch>>, Arc<MemoryReservation>)> {
        match self {
            Self::Host { batches, reservation } => Some((Arc::clone(batches), Arc::clone(reservation))),
            Self::Gpu(_) => None,
        }
    }

    pub(crate) fn gpu_handle(&self) -> Option<GpuResultHandle> {
        match self {
            Self::Host { .. } => None,
            Self::Gpu(handle) => Some((**handle).clone()),
        }
    }
}

#[cfg(feature = "cugraph")]
impl fmt::Debug for GraphBfsPublishedResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host { batches, .. } => {
                formatter.debug_struct("GraphBfsPublishedResult::Host").field("batch_count", &batches.len()).finish()
            },
            Self::Gpu(handle) => formatter
                .debug_struct("GraphBfsPublishedResult::Gpu")
                .field("rows", &handle.num_rows())
                .field("columns", &handle.num_columns())
                .finish(),
        }
    }
}

#[cfg(feature = "cugraph")]
pub(crate) enum GraphBfsSingleflightJoin {
    Leader(GraphBfsLeader),
    Follower(GraphBfsFollower),
}

/// Rendezvous for one GPU-resident publication's joined followers.
///
/// The published frame borrows the leader attempt's attached domain members:
/// follower copies read leader-domain columns through the leader's member
/// attachment, and the leader's terminal cleanup fails closed on residual
/// domain usage. The drain counts the generation's joined followers down to
/// zero as each one copies or drops the handoff; the leader waits on it after
/// publishing and before returning, so leader teardown cannot race a follower
/// copy. Publication resolves the generation, therefore no request can join
/// afterwards and the count only decreases.
#[cfg(feature = "cugraph")]
#[derive(Debug)]
pub(crate) struct GpuHandoffDrain {
    remaining: AtomicUsize,
    notify: Notify,
}

#[cfg(feature = "cugraph")]
impl GpuHandoffDrain {
    fn new(follower_count: usize) -> Self {
        Self { remaining: AtomicUsize::new(follower_count), notify: Notify::new() }
    }

    async fn wait_until_drained(&self, cancellation: &ExecutionCancellation) -> Result<()> {
        loop {
            if self.remaining.load(Ordering::Acquire) == 0 {
                return Ok(());
            }
            if let Err(error) = cancellation.check(OPERATION) {
                return Err(Error::from_native(
                    OPERATION,
                    "cancelled while waiting for joined BFS followers to copy the GPU handoff",
                    error,
                ));
            }
            tokio::select! {
                () = self.notify.notified() => {},
                () = sleep(Duration::from_millis(5)) => {},
            }
        }
    }

    fn signal_copy_settled(&self) {
        self.remaining.fetch_sub(1, Ordering::AcqRel);
        self.notify.notify_waiters();
    }
}

#[cfg(feature = "cugraph")]
pub(crate) struct GraphBfsLeader {
    coordinator: GraphBfsSingleflight,
    key: GraphBfsSingleflightKey,
    generation: u64,
    correlation_fingerprint: Option<String>,
    traversal_submitted: bool,
    resolved: bool,
}

#[cfg(feature = "cugraph")]
impl GraphBfsLeader {
    pub(crate) fn record_traversal_submission(&mut self) {
        debug_assert!(!self.traversal_submitted, "singleflight leader submitted more than one traversal");
        if !self.traversal_submitted {
            self.coordinator.inner.counters.physical_traversals.fetch_add(1, Ordering::Relaxed);
            self.coordinator.record_event(
                &self.key,
                self.correlation_fingerprint.clone(),
                self.generation,
                "leader_traversal",
            );
            self.traversal_submitted = true;
        }
    }

    pub(crate) fn has_followers(&self) -> Result<bool> {
        self.coordinator.participant_count(&self.key, self.generation).map(|count| count > 1)
    }

    /// Clones this generation's GPU handoff drain after publishing a
    /// GPU-resident result. Returns `None` for host publications and for
    /// generations without joined followers, which need no rendezvous.
    pub(crate) fn handoff_drain(&self) -> Result<Option<Arc<GpuHandoffDrain>>> {
        self.coordinator.handoff_drain(&self.key, self.generation)
    }

    /// Waits until every joined follower has copied or dropped the published
    /// GPU handoff before the leader may return. The leader attempt's domain
    /// members stay attached across this wait, so follower copies cannot race
    /// leader teardown.
    pub(crate) async fn wait_for_handoff_drain(
        &self,
        drain: &GpuHandoffDrain,
        cancellation: &ExecutionCancellation,
    ) -> Result<()> {
        drain.wait_until_drained(cancellation).await
    }

    pub(crate) fn publish(&mut self, result: GraphBfsPublishedResult) -> Result<()> {
        self.coordinator.publish(
            &self.key,
            self.generation,
            self.correlation_fingerprint.clone(),
            PublishedResult::Ready(result),
        )?;
        self.resolved = true;
        Ok(())
    }

    pub(crate) fn complete_without_result(&mut self) -> Result<()> {
        self.coordinator.publish(
            &self.key,
            self.generation,
            self.correlation_fingerprint.clone(),
            PublishedResult::Retry,
        )?;
        self.resolved = true;
        Ok(())
    }

    pub(crate) fn fail(mut self) {
        self.coordinator.abandon(&self.key, self.generation, self.correlation_fingerprint.clone(), false);
        self.resolved = true;
    }
}

#[cfg(feature = "cugraph")]
impl Drop for GraphBfsLeader {
    fn drop(&mut self) {
        if !self.resolved {
            self.coordinator.abandon(&self.key, self.generation, self.correlation_fingerprint.clone(), true);
            self.resolved = true;
        }
    }
}

#[cfg(feature = "cugraph")]
pub(crate) struct GraphBfsFollower {
    coordinator: GraphBfsSingleflight,
    key: GraphBfsSingleflightKey,
    generation: u64,
    leader_attempt_id: Option<AttemptId>,
    correlation_fingerprint: Option<String>,
    receiver: watch::Receiver<PublishedResult>,
    completed: bool,
    handoff_settled: AtomicBool,
}

#[cfg(feature = "cugraph")]
impl GraphBfsFollower {
    /// The engine-minted identity of the physical leader, when this
    /// coordinator was joined by an attempt-bound request.
    pub(crate) const fn leader_attempt_id(&self) -> Option<AttemptId> {
        self.leader_attempt_id
    }

    pub(crate) fn record_copy(&self) {
        self.coordinator.record_follower_copy(&self.key, self.generation, self.correlation_fingerprint.clone());
        self.signal_handoff_settled();
    }

    /// Settles this follower's copied share of a GPU handoff drain exactly once.
    /// A copy implies the generation published, so the drain exists for GPU
    /// publications; a follower of a host publication or a retried generation
    /// finds no drain and only flips its local flag. Drop consults the same
    /// flag and leaves through `leave_follower`, which settles an unsettled
    /// follower under the same lock hold that checks the live group.
    fn signal_handoff_settled(&self) {
        if self.handoff_settled.swap(true, Ordering::AcqRel) {
            return;
        }
        self.coordinator.signal_handoff_copy_settled(&self.key, self.generation);
    }

    pub(crate) async fn wait(&mut self, cancellation: &ExecutionCancellation) -> Result<GraphBfsFollowerOutcome> {
        loop {
            if let Err(error) = cancellation.check(OPERATION) {
                return Err(Error::from_native(OPERATION, "cancelled while waiting for a coalesced BFS result", error));
            }
            match self.receiver.borrow().clone() {
                PublishedResult::Ready(batch) => {
                    self.completed = true;
                    return Ok(GraphBfsFollowerOutcome::Ready(batch));
                },
                PublishedResult::Retry => {
                    self.completed = true;
                    return Ok(GraphBfsFollowerOutcome::Retry);
                },
                PublishedResult::Pending => {},
            }
            tokio::select! {
                changed = self.receiver.changed() => {
                    if changed.is_err() {
                        self.completed = true;
                        return Ok(GraphBfsFollowerOutcome::Retry);
                    }
                },
                () = sleep(Duration::from_millis(5)) => {},
            }
        }
    }
}

#[cfg(feature = "cugraph")]
impl Drop for GraphBfsFollower {
    fn drop(&mut self) {
        // A copy already settled this follower's drain share, and the
        // generation published before the copy, so no group membership
        // remains to leave. Anything else leaves through one locked section
        // that either shrinks the live group or settles the published drain.
        if self.handoff_settled.swap(true, Ordering::AcqRel) {
            return;
        }
        self.coordinator.leave_follower(&self.key, self.generation, !self.completed);
    }
}

#[cfg(feature = "cugraph")]
pub(crate) enum GraphBfsFollowerOutcome {
    Ready(GraphBfsPublishedResult),
    Retry,
}

#[cfg(feature = "cugraph")]
fn remove_group(
    state: &mut SingleflightState,
    key: &GraphBfsSingleflightKey,
    generation: u64,
) -> Option<SingleflightGroup> {
    let groups = state.groups.get_mut(key)?;
    let position = groups.iter().position(|group| group.generation == generation)?;
    let group = groups.remove(position);
    if groups.is_empty() {
        state.groups.remove(key);
    }
    Some(group)
}

#[cfg(feature = "cugraph")]
fn update_peak(peak: &AtomicU64, value: usize) {
    peak.fetch_max(u64::try_from(value).unwrap_or(u64::MAX), Ordering::AcqRel);
}

#[cfg(feature = "cugraph")]
fn graph_bfs_singleflight_key_fingerprint(key: &GraphBfsSingleflightKey) -> String {
    cache_observation_fingerprint(&format!("algeon-datafusion/bfs-singleflight-key/v1/{key:?}"))
}

#[cfg(feature = "cugraph")]
fn state_poisoned_error(device_ordinal: usize) -> Error {
    Error::new(ErrorCode::Internal, OPERATION, "BFS singleflight state mutex was poisoned")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("reason", "state_poisoned")
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use arrow::datatypes::Schema;
    use datafusion::physical_plan::{ExecutionPlan, empty::EmptyExec};

    use super::*;

    static REENTRANT_REGISTRATION_RAN: AtomicBool = AtomicBool::new(false);

    fn no_graph_match(_: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
        None
    }

    fn register_while_discovering(_: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
        register_graph_execution_plan_discovery(no_graph_match);
        REENTRANT_REGISTRATION_RAN.store(true, Ordering::Release);
        None
    }

    #[test]
    fn discovery_callbacks_run_without_holding_the_registry_lock() {
        REENTRANT_REGISTRATION_RAN.store(false, Ordering::Release);
        register_graph_execution_plan_discovery(register_while_discovering);
        let plan = EmptyExec::new(Arc::new(Schema::empty()));

        assert!(graph_execution_plan(&plan).is_none());
        assert!(REENTRANT_REGISTRATION_RAN.load(Ordering::Acquire));
    }

    #[cfg(feature = "cugraph")]
    mod singleflight_tests {
        use std::sync::Barrier;

        use arrow::{
            array::{ArrayRef, Int32Array},
            datatypes::{DataType, Field},
            record_batch::RecordBatch,
        };
        use datafusion::execution::{TaskContext, memory_pool::MemoryConsumer};
        use query_engine::{
            admission::AttemptCapabilityEvidence,
            plan::QueryPlanBuilder,
            runtime::{
                QueryAdmissionRequest,
                resident::{GpuDataFrame, GpuResultHandle},
            },
            source::{LiteralBatchSource, SourceNode},
        };
        use tokio::task::spawn_blocking;

        use super::*;
        use crate::backend::InstalledTestGpuBackend;

        fn key(source_vertex: i64) -> GraphBfsSingleflightKey {
            GraphBfsSingleflightKey {
                device_ordinal: 0,
                resident_graph_key: None,
                source_vertex,
                normalized_output: true,
                projection: None,
                limit: None,
                max_output_staging_bytes: 1024,
            }
        }

        fn empty_result() -> GraphBfsPublishedResult {
            let task_context = TaskContext::default();
            let reservation = MemoryConsumer::new("GraphBfsSingleflight[test]").register(task_context.memory_pool());
            GraphBfsPublishedResult::new(
                Arc::new(vec![RecordBatch::new_empty(Arc::new(Schema::empty()))]),
                Arc::new(reservation),
            )
        }

        #[tokio::test]
        async fn identical_requests_share_one_physical_traversal() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(mut leader) = coordinator.join(key(1)).unwrap() else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
                panic!("second request must follow")
            };

            leader.record_traversal_submission();
            leader.publish(empty_result()).unwrap();
            assert!(matches!(
                follower.wait(&ExecutionCancellation::new()).await.unwrap(),
                GraphBfsFollowerOutcome::Ready(_)
            ));
            let snapshot = coordinator.snapshot();
            assert_eq!(snapshot.eligible_requests, 2);
            assert_eq!(snapshot.physical_traversals, 1);
            assert_eq!(snapshot.groups_formed, 1);
            assert_eq!(snapshot.peak_width, 2);
        }

        #[test]
        fn attempt_bound_follower_retains_its_physical_leader_identity() {
            let coordinator = GraphBfsSingleflight::new(4);
            let leader_attempt_id = AttemptId::mint();
            let GraphBfsSingleflightJoin::Leader(_leader) =
                coordinator.join_with_attempt(key(1), None, Some(leader_attempt_id)).unwrap()
            else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(follower) =
                coordinator.join_with_attempt(key(1), None, Some(AttemptId::mint())).unwrap()
            else {
                panic!("second request must follow")
            };

            assert_eq!(follower.leader_attempt_id(), Some(leader_attempt_id));
        }

        #[tokio::test]
        async fn correlated_events_are_emitted_by_the_singleflight_owner() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(mut leader) =
                coordinator.join_with_correlation(key(1), Some("leader-correlation".to_string())).unwrap()
            else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(mut follower) =
                coordinator.join_with_correlation(key(1), Some("follower-correlation".to_string())).unwrap()
            else {
                panic!("second request must follow")
            };

            leader.record_traversal_submission();
            leader.publish(empty_result()).unwrap();
            assert!(matches!(
                follower.wait(&ExecutionCancellation::new()).await.unwrap(),
                GraphBfsFollowerOutcome::Ready(_)
            ));
            follower.record_copy();

            let snapshot = coordinator.snapshot();
            let actions = snapshot.events.iter().map(|event| event.action).collect::<Vec<_>>();
            assert_eq!(
                actions,
                vec!["leader_join", "follower_join", "leader_traversal", "leader_published", "follower_copy"]
            );
            assert!(snapshot.events.iter().all(|event| event.device_ordinal == 0));
            assert_eq!(
                snapshot
                    .events
                    .iter()
                    .map(|event| &event.singleflight_key_fingerprint)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                1
            );
            assert_eq!(snapshot.events[0].correlation_fingerprint.as_deref(), Some("leader-correlation"));
            assert_eq!(snapshot.events[1].correlation_fingerprint.as_deref(), Some("follower-correlation"));
        }

        #[tokio::test]
        async fn published_multi_batch_result_retains_one_shared_reservation_for_followers() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(mut leader) = coordinator.join(key(1)).unwrap() else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
                panic!("second request must follow")
            };
            let task_context = TaskContext::default();
            let pool = Arc::clone(task_context.memory_pool());
            let reservation = MemoryConsumer::new("GraphBfsSingleflight[multi-batch-test]").register(&pool);
            reservation.try_grow(128).unwrap();
            let result = GraphBfsPublishedResult::new(
                Arc::new(vec![
                    RecordBatch::new_empty(Arc::new(Schema::empty())),
                    RecordBatch::new_empty(Arc::new(Schema::empty())),
                ]),
                Arc::new(reservation),
            );

            leader.publish(result).unwrap();
            let GraphBfsFollowerOutcome::Ready(result) = follower.wait(&ExecutionCancellation::new()).await.unwrap()
            else {
                panic!("published result must be ready")
            };
            assert_eq!(result.host_batches().expect("host result must expose batches").len(), 2);
            assert_eq!(pool.reserved(), 128);
            drop(result);
            assert_eq!(pool.reserved(), 128, "the follower receiver still retains the published result");
            drop(follower);
            assert_eq!(pool.reserved(), 0);
        }

        #[test]
        fn published_result_routing_matches_follower_sink() {
            assert_eq!(
                GraphBfsFollowerRoute::for_gpu_follower(GraphBfsPublicationKind::Host),
                GraphBfsFollowerRoute::UploadHost,
            );
            assert_eq!(
                GraphBfsFollowerRoute::for_gpu_follower(GraphBfsPublicationKind::Gpu),
                GraphBfsFollowerRoute::HandoffGpu,
            );
            assert_eq!(
                GraphBfsFollowerRoute::for_host_follower(GraphBfsPublicationKind::Host),
                GraphBfsFollowerRoute::ReuseHost,
            );
            assert_eq!(
                GraphBfsFollowerRoute::for_host_follower(GraphBfsPublicationKind::Gpu),
                GraphBfsFollowerRoute::MaterializeGpu,
            );
        }

        #[tokio::test]
        async fn gpu_handoff_drain_waits_until_every_follower_settles() {
            let drain = Arc::new(GpuHandoffDrain::new(2));
            let waiter = Arc::clone(&drain);
            let settled = tokio::spawn(async move { waiter.wait_until_drained(&ExecutionCancellation::new()).await });
            sleep(Duration::from_millis(20)).await;
            assert!(!settled.is_finished(), "the leader waits while followers still hold the handoff");
            drain.signal_copy_settled();
            sleep(Duration::from_millis(20)).await;
            assert!(!settled.is_finished(), "one outstanding follower still holds the leader");
            drain.signal_copy_settled();
            settled.await.expect("drain wait does not panic").expect("a settled handoff releases the leader");
        }

        #[test]
        fn gpu_handoff_drain_releases_the_leader_exactly_once_per_follower() {
            let coordinator = GraphBfsSingleflight::new(4);
            let join_key = key(1);
            let GraphBfsSingleflightJoin::Leader(leader) = coordinator.join(join_key.clone()).unwrap() else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(first) = coordinator.join(join_key.clone()).unwrap() else {
                panic!("second request must follow")
            };
            let GraphBfsSingleflightJoin::Follower(second) = coordinator.join(join_key.clone()).unwrap() else {
                panic!("third request must follow")
            };
            // A GPU publication to two joined followers opens one drain counting both.
            coordinator
                .inner
                .state
                .lock()
                .unwrap()
                .gpu_handoff_drains
                .insert((join_key, leader.generation), Arc::new(GpuHandoffDrain::new(2)));
            let remaining = || {
                leader
                    .handoff_drain()
                    .map(|drain| drain.map(|drain| drain.remaining.load(Ordering::Acquire)).unwrap_or(0))
                    .unwrap()
            };

            first.record_copy();
            assert_eq!(remaining(), 1, "one follower copy settles one drain share");
            first.record_copy();
            assert_eq!(remaining(), 1, "a duplicate copy must not over-release the leader");
            drop(second);
            assert!(
                leader.handoff_drain().unwrap().is_none(),
                "a follower that never copies still releases the leader on drop"
            );
        }

        #[tokio::test]
        async fn host_publication_creates_no_handoff_drain() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(mut leader) = coordinator.join(key(1)).unwrap() else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
                panic!("second request must follow")
            };

            leader.publish(empty_result()).unwrap();
            assert!(leader.handoff_drain().unwrap().is_none(), "host publications need no leader rendezvous");
            assert!(matches!(
                follower.wait(&ExecutionCancellation::new()).await.unwrap(),
                GraphBfsFollowerOutcome::Ready(_)
            ));
            follower.record_copy();
            drop(follower);
            drop(leader);
            assert!(
                coordinator.inner.state.lock().unwrap().gpu_handoff_drains.is_empty(),
                "host generations leave no drain behind"
            );
        }

        #[test]
        fn requests_with_different_output_caps_never_form_a_leader_follower_group() {
            let coordinator = GraphBfsSingleflight::new(4);
            let loose_key = key(1);
            let mut strict_key = loose_key.clone();
            strict_key.max_output_staging_bytes -= 1;

            let GraphBfsSingleflightJoin::Leader(_loose_leader) = coordinator.join(loose_key).unwrap() else {
                panic!("first output-cap request must lead")
            };
            let GraphBfsSingleflightJoin::Leader(_strict_leader) = coordinator.join(strict_key).unwrap() else {
                panic!("a request with a different output cap must start an independent generation")
            };

            let snapshot = coordinator.snapshot();
            assert_eq!(snapshot.groups_formed, 2);
            assert_eq!(snapshot.peak_width, 1);
        }

        #[test]
        fn batch_width_overflow_starts_an_independent_generation() {
            let coordinator = GraphBfsSingleflight::new(2);
            let GraphBfsSingleflightJoin::Leader(mut first) = coordinator.join(key(1)).unwrap() else {
                panic!("first request must lead")
            };
            let second = coordinator.join(key(1)).unwrap();
            let GraphBfsSingleflightJoin::Leader(mut overflow) = coordinator.join(key(1)).unwrap() else {
                panic!("overflow request must lead")
            };
            first.record_traversal_submission();
            overflow.record_traversal_submission();
            assert!(matches!(second, GraphBfsSingleflightJoin::Follower(_)));

            let snapshot = coordinator.snapshot();
            assert_eq!(snapshot.physical_traversals, 2);
            assert_eq!(snapshot.overflow_groups, 1);
        }

        #[tokio::test]
        async fn failed_leader_releases_followers_to_retry() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(leader) = coordinator.join(key(1)).unwrap() else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
                panic!("second request must follow")
            };
            leader.fail();

            assert!(matches!(
                follower.wait(&ExecutionCancellation::new()).await.unwrap(),
                GraphBfsFollowerOutcome::Retry
            ));
            assert_eq!(coordinator.snapshot().leader_failures, 1);
            assert_eq!(coordinator.snapshot().physical_traversals, 0);
            assert!(matches!(coordinator.join(key(1)).unwrap(), GraphBfsSingleflightJoin::Leader(_)));
        }

        #[tokio::test]
        async fn dropped_leader_is_cancelled_and_releases_followers() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(leader) = coordinator.join(key(1)).unwrap() else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
                panic!("second request must follow")
            };

            drop(leader);
            assert!(matches!(
                follower.wait(&ExecutionCancellation::new()).await.unwrap(),
                GraphBfsFollowerOutcome::Retry
            ));
            let snapshot = coordinator.snapshot();
            assert_eq!(snapshot.cancelled_waiters, 1);
            assert_eq!(snapshot.leader_failures, 0);
            assert_eq!(snapshot.physical_traversals, 0);
            assert!(matches!(coordinator.join(key(1)).unwrap(), GraphBfsSingleflightJoin::Leader(_)));
        }

        #[tokio::test]
        async fn cancelled_follower_leaves_the_leader_running() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(leader) = coordinator.join(key(1)).unwrap() else {
                panic!("first request must lead")
            };
            let GraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
                panic!("second request must follow")
            };
            let cancellation = ExecutionCancellation::new();
            cancellation.cancel();

            assert!(follower.wait(&cancellation).await.is_err());
            drop(follower);
            assert!(!leader.has_followers().unwrap());
            assert_eq!(coordinator.snapshot().cancelled_waiters, 1);
        }

        #[test]
        fn distinct_sources_never_coalesce() {
            let coordinator = GraphBfsSingleflight::new(4);
            let GraphBfsSingleflightJoin::Leader(mut first) = coordinator.join(key(1)).unwrap() else {
                panic!("first source must lead")
            };
            let GraphBfsSingleflightJoin::Leader(mut second) = coordinator.join(key(2)).unwrap() else {
                panic!("second source must lead")
            };
            first.record_traversal_submission();
            second.record_traversal_submission();
            assert_eq!(coordinator.snapshot().physical_traversals, 2);
        }

        /// Races follower drops against a GPU publish from one barrier, so
        /// the publish wedges between drop and group bookkeeping however the
        /// threads interleave. Every joined follower either leaves the group
        /// before publish (and is never counted) or settles its drain share
        /// after publish, so the leader's drain always reaches zero and is
        /// removed. A drop that settles the drain and leaves the group in two
        /// lock sections strands the drain when publish lands between them.
        #[tokio::test]
        async fn follower_drops_racing_publish_still_drain_the_leader() {
            const FOLLOWER_COUNT: usize = 31;
            const ITERATIONS: usize = 100;

            let backend = InstalledTestGpuBackend::new();
            let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
            let batch =
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef])
                    .expect("literal batch");
            let mut builder = QueryPlanBuilder::new();
            let source =
                builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
            let plan = builder.finish_gpu_resident(source);
            let attempt = backend
                .backend()
                .session_capability()
                .admit_query_with_request(
                    QueryAdmissionRequest::new()
                        .with_query_plan_capability_evidence(AttemptCapabilityEvidence::for_query_plan(&plan)),
                )
                .expect("attempt admission");
            let admitted = attempt.acquire_context().expect("admitted context");
            let context = admitted.context();
            let runtime = context.runtime().expect("attempt runtime");
            let upload =
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef])
                    .expect("upload batch");
            let publication = GpuResultHandle::new(
                context,
                GpuDataFrame::new(cudf::DataFrame::from_record_batch(&runtime, &upload).expect("resident frame")),
                Arc::clone(&schema),
            )
            .expect("resident handle");

            for iteration in 0..ITERATIONS {
                let coordinator = GraphBfsSingleflight::new(FOLLOWER_COUNT + 1);
                let join_key = key(i64::try_from(iteration).unwrap());
                let GraphBfsSingleflightJoin::Leader(mut leader) = coordinator.join(join_key.clone()).unwrap() else {
                    panic!("iteration {iteration}: first request must lead")
                };
                let mut followers = Vec::with_capacity(FOLLOWER_COUNT);
                for _ in 0..FOLLOWER_COUNT {
                    let GraphBfsSingleflightJoin::Follower(follower) = coordinator.join(join_key.clone()).unwrap()
                    else {
                        panic!("iteration {iteration}: follower requests must follow")
                    };
                    followers.push(follower);
                }
                let barrier = Arc::new(Barrier::new(FOLLOWER_COUNT + 1));
                let mut tasks = Vec::with_capacity(FOLLOWER_COUNT + 1);
                for follower in followers {
                    let gate = Arc::clone(&barrier);
                    tasks.push(spawn_blocking(move || {
                        gate.wait();
                        drop(follower);
                    }));
                }
                let result = GraphBfsPublishedResult::Gpu(Box::new(publication.clone()));
                let gate = Arc::clone(&barrier);
                tasks.push(spawn_blocking(move || {
                    gate.wait();
                    leader.publish(result).expect("publish must succeed");
                }));
                for task in tasks {
                    task.await.expect("worker must not panic");
                }
                assert!(
                    coordinator.inner.state.lock().unwrap().gpu_handoff_drains.is_empty(),
                    "iteration {iteration}: every joined follower either left before publish or settled after it"
                );
            }

            drop(publication);
            drop(admitted);
            attempt.finish();
        }
    }
}
