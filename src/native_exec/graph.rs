//! Frontend-neutral graph execution services and admission contracts.
//!
//! The adapter core owns process-local coordination and the DataFusion plan
//! discovery SPI. Graph frontends register implementations without creating a
//! core-to-frontend dependency.
//!
//! Exact-request singleflight for expensive BFS calls.
//!
//! This holds only a transient host result until its followers consume it. The
//! graph identity is a coalescing key, not a device-resident cache claim; the
//! resident cache owns any retained device allocation separately.
//!
//! libcugraph multi-source BFS produces one nearest-source forest, not one
//! independent result per source. Serving coalescing therefore groups only
//! requests with identical traversal semantics and publishes the completed
//! host result to followers.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use arrow::record_batch::RecordBatch;
use datafusion::{
    common::Result as DataFusionResult, execution::memory_pool::MemoryReservation, physical_plan::ExecutionPlan,
};
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof, graph_memory::GraphMemoryAdmissionEvidence, plan::ResidentGraphKey,
    runtime::ExecutionCancellation,
};
use tokio::sync::watch;

use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::SealedAttemptMetricsTarget,
};

const OPERATION: &str = "datafusion_nexus.graph_execution.bfs_singleflight";
const DEFAULT_MAX_BATCH_WIDTH: usize = 32;

/// Opaque construction-inclusive and execution-only graph-model output.
///
/// The installed profile and ledger still own immutable-grant eligibility and
/// capacity. The cold estimate may only refine the target inside that range,
/// through the engine-issued proof that attests the exact graph execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphAdmissionModelAsk {
    cold_model_requested_bytes: u64,
    warm_model_requested_bytes: u64,
    cold_model_evidence: GraphMemoryAdmissionEvidence,
}

impl GraphAdmissionModelAsk {
    /// Forwards engine-issued graph-model evidence into the adapter-private
    /// planning/diagnostic path.
    pub(crate) fn from_graph_memory_admission_evidence(evidence: GraphMemoryAdmissionEvidence) -> Self {
        Self {
            cold_model_requested_bytes: evidence.cold_model_requested_bytes(),
            warm_model_requested_bytes: evidence.warm_model_requested_bytes(),
            cold_model_evidence: evidence,
        }
    }

    /// Construction-inclusive graph-model estimate retained for planner diagnostics only.
    #[cfg(test)]
    pub(crate) fn cold_model_requested_bytes(&self) -> u64 {
        self.cold_model_requested_bytes
    }

    /// Execution-only graph-model estimate retained for planner diagnostics only.
    #[cfg(test)]
    pub(crate) fn warm_model_requested_bytes(&self) -> u64 {
        self.warm_model_requested_bytes
    }

    pub(crate) fn cold_model_evidence(&self) -> &GraphMemoryAdmissionEvidence {
        &self.cold_model_evidence
    }

    pub(crate) fn max(self, other: Self) -> Self {
        let Self {
            cold_model_requested_bytes: self_cold,
            warm_model_requested_bytes: self_warm,
            cold_model_evidence: self_evidence,
        } = self;
        let Self {
            cold_model_requested_bytes: other_cold,
            warm_model_requested_bytes: other_warm,
            cold_model_evidence: other_evidence,
        } = other;
        let (cold_model_requested_bytes, cold_model_evidence) =
            if other_cold > self_cold { (other_cold, other_evidence) } else { (self_cold, self_evidence) };
        Self { cold_model_requested_bytes, warm_model_requested_bytes: self_warm.max(other_warm), cold_model_evidence }
    }
}

/// Frontend-neutral graph-model diagnostics exposed by a graph execution node.
///
/// Capability proof issuance remains in the adapter planner after it has
/// recognized this crate-private execution type; graph frontends do not get a
/// caller-controlled proof SPI.
pub(crate) trait GraphExecutionPlan: Send + Sync {
    fn graph_model_ask_for_admission(&self) -> DataFusionResult<Option<GraphAdmissionModelAsk>>;

    /// Issues capability evidence for the exact native graph fragment this
    /// execution node submits after staging its GPU-resident inputs.
    fn graph_execution_capability_proof(&self) -> DataFusionResult<QueryPlanCapabilityProof>;

    /// Returns the one fixed-cardinality post-seal metric target for this
    /// graph execution root. The adapter discovers it through this neutral
    /// SPI, so core execution does not depend on a graph frontend.
    fn sealed_attempt_metrics_target(&self) -> SealedAttemptMetricsTarget;
}

/// A graph frontend hook that recognizes one of its DataFusion execution nodes.
pub(crate) type GraphExecutionPlanDiscovery = for<'a> fn(&'a dyn ExecutionPlan) -> Option<&'a dyn GraphExecutionPlan>;

static GRAPH_EXECUTION_PLAN_DISCOVERIES: OnceLock<Mutex<Vec<GraphExecutionPlanDiscovery>>> = OnceLock::new();

/// Register a graph frontend's execution-plan recognizer.
///
/// Registration is idempotent for the same function and does not retain a
/// frontend value. The core planner only sees the frontend-neutral trait.
pub(crate) fn register_graph_execution_plan_discovery(discovery: GraphExecutionPlanDiscovery) {
    let mut discoveries = GRAPH_EXECUTION_PLAN_DISCOVERIES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !discoveries.iter().any(|candidate| std::ptr::fn_addr_eq(*candidate, discovery)) {
        discoveries.push(discovery);
    }
}

pub(crate) fn graph_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn GraphExecutionPlan> {
    let discoveries = {
        GRAPH_EXECUTION_PLAN_DISCOVERIES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    };
    discoveries.iter().find_map(|discovery| discovery(plan))
}

/// Shared process-local services carried by native execution configuration.
///
/// The state is graph-runtime owned and contains no SQL frontend types.
#[derive(Debug, Clone, Default)]
pub(crate) struct GraphExecutionServices {
    bfs_singleflight: GraphBfsSingleflight,
}

impl GraphExecutionServices {
    pub(crate) fn bfs_singleflight(&self) -> GraphBfsSingleflight {
        self.bfs_singleflight.clone()
    }

    pub(crate) fn bfs_singleflight_snapshot(&self) -> GraphBfsSingleflightSnapshot {
        self.bfs_singleflight.snapshot()
    }

    pub(crate) fn set_bfs_singleflight_enabled(&self, enabled: bool) {
        self.bfs_singleflight.set_enabled(enabled);
    }

    pub(crate) fn bfs_singleflight_enabled(&self) -> bool {
        self.bfs_singleflight.is_enabled()
    }
}

/// Exact compatibility key for one device-local independent BFS traversal.
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
}

/// Process-local coordinator shared by clones of one backend execution config.
#[derive(Debug, Clone)]
pub(crate) struct GraphBfsSingleflight {
    inner: Arc<SingleflightInner>,
}

impl Default for GraphBfsSingleflight {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_BATCH_WIDTH)
    }
}

impl GraphBfsSingleflight {
    fn new(max_batch_width: usize) -> Self {
        debug_assert!(max_batch_width > 0);
        Self {
            inner: Arc::new(SingleflightInner {
                max_batch_width,
                enabled: AtomicBool::new(true),
                state: Mutex::new(SingleflightState::default()),
                counters: SingleflightCounters::default(),
            }),
        }
    }

    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.inner.enabled.store(enabled, Ordering::Release);
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Acquire)
    }

    pub(crate) fn join(&self, key: GraphBfsSingleflightKey) -> Result<GraphBfsSingleflightJoin> {
        self.inner.counters.eligible_requests.fetch_add(1, Ordering::Relaxed);
        let mut state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
        if let Some(groups) = state.groups.get_mut(&key)
            && let Some(group) = groups.iter_mut().find(|group| group.participants < self.inner.max_batch_width)
        {
            group.participants += 1;
            update_peak(&self.inner.counters.peak_width, group.participants);
            return Ok(GraphBfsSingleflightJoin::Follower(GraphBfsFollower {
                coordinator: self.clone(),
                key,
                generation: group.generation,
                receiver: group.sender.subscribe(),
                completed: false,
            }));
        }

        state.next_generation = state.next_generation.saturating_add(1);
        let generation = state.next_generation;
        let (sender, _receiver) = watch::channel(PublishedResult::Pending);
        let groups = state.groups.entry(key.clone()).or_default();
        if !groups.is_empty() {
            self.inner.counters.overflow_groups.fetch_add(1, Ordering::Relaxed);
        }
        groups.push(SingleflightGroup { generation, participants: 1, sender });
        self.inner.counters.groups_formed.fetch_add(1, Ordering::Relaxed);
        update_peak(&self.inner.counters.peak_width, 1);
        Ok(GraphBfsSingleflightJoin::Leader(GraphBfsLeader {
            coordinator: self.clone(),
            key,
            generation,
            traversal_submitted: false,
            resolved: false,
        }))
    }

    pub(crate) fn record_follower_copy(&self) {
        self.inner.counters.follower_copies.fetch_add(1, Ordering::Relaxed);
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
        }
    }

    fn publish(&self, key: &GraphBfsSingleflightKey, generation: u64, result: PublishedResult) -> Result<()> {
        let sender = {
            let mut state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
            remove_group(&mut state, key, generation).map(|group| group.sender)
        };
        if let Some(sender) = sender {
            sender.send_replace(result);
        }
        Ok(())
    }

    fn participant_count(&self, key: &GraphBfsSingleflightKey, generation: u64) -> Result<usize> {
        let state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
        Ok(state
            .groups
            .get(key)
            .and_then(|groups| groups.iter().find(|group| group.generation == generation))
            .map_or(0, |group| group.participants))
    }

    fn abandon(&self, key: &GraphBfsSingleflightKey, generation: u64, cancelled: bool) {
        let sender = match self.inner.state.lock() {
            Ok(mut state) => remove_group(&mut state, key, generation).map(|group| group.sender),
            Err(_error) => {
                eprintln!("[datafusion-nexus] warning: failed to abandon poisoned BFS singleflight state");
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

    fn leave_follower(&self, key: &GraphBfsSingleflightKey, generation: u64, cancelled: bool) {
        if cancelled {
            self.inner.counters.cancelled_waiters.fetch_add(1, Ordering::Relaxed);
        }
        let Ok(mut state) = self.inner.state.lock() else {
            eprintln!("[datafusion-nexus] warning: failed to leave poisoned BFS singleflight state");
            return;
        };
        let Some(groups) = state.groups.get_mut(key) else {
            return;
        };
        if let Some(group) = groups.iter_mut().find(|group| group.generation == generation) {
            group.participants = group.participants.saturating_sub(1);
        }
    }
}

#[derive(Debug)]
struct SingleflightInner {
    max_batch_width: usize,
    enabled: AtomicBool,
    state: Mutex<SingleflightState>,
    counters: SingleflightCounters,
}

#[derive(Debug, Default)]
struct SingleflightState {
    groups: HashMap<GraphBfsSingleflightKey, Vec<SingleflightGroup>>,
    next_generation: u64,
}

#[derive(Debug)]
struct SingleflightGroup {
    generation: u64,
    participants: usize,
    sender: watch::Sender<PublishedResult>,
}

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

#[derive(Debug, Clone)]
enum PublishedResult {
    Pending,
    Ready(GraphBfsPublishedResult),
    Retry,
}

/// Bounded host batches published by one BFS leader and retained until every
/// joined follower has imported or returned them.
#[derive(Debug, Clone)]
pub(crate) struct GraphBfsPublishedResult {
    batches: Arc<Vec<RecordBatch>>,
    reservation: Arc<MemoryReservation>,
}

impl GraphBfsPublishedResult {
    pub(crate) fn new(batches: Arc<Vec<RecordBatch>>, reservation: Arc<MemoryReservation>) -> Self {
        Self { batches, reservation }
    }

    pub(crate) fn batches(&self) -> &[RecordBatch] {
        self.batches.as_slice()
    }

    pub(crate) fn shared_parts(&self) -> (Arc<Vec<RecordBatch>>, Arc<MemoryReservation>) {
        (Arc::clone(&self.batches), Arc::clone(&self.reservation))
    }
}

pub(crate) enum GraphBfsSingleflightJoin {
    Leader(GraphBfsLeader),
    Follower(GraphBfsFollower),
}

pub(crate) struct GraphBfsLeader {
    coordinator: GraphBfsSingleflight,
    key: GraphBfsSingleflightKey,
    generation: u64,
    traversal_submitted: bool,
    resolved: bool,
}

impl GraphBfsLeader {
    pub(crate) fn record_traversal_submission(&mut self) {
        debug_assert!(!self.traversal_submitted, "singleflight leader submitted more than one traversal");
        if !self.traversal_submitted {
            self.coordinator.inner.counters.physical_traversals.fetch_add(1, Ordering::Relaxed);
            self.traversal_submitted = true;
        }
    }

    pub(crate) fn has_followers(&self) -> Result<bool> {
        self.coordinator.participant_count(&self.key, self.generation).map(|count| count > 1)
    }

    pub(crate) fn publish(&mut self, result: GraphBfsPublishedResult) -> Result<()> {
        self.coordinator.publish(&self.key, self.generation, PublishedResult::Ready(result))?;
        self.resolved = true;
        Ok(())
    }

    pub(crate) fn complete_without_result(&mut self) -> Result<()> {
        self.coordinator.publish(&self.key, self.generation, PublishedResult::Retry)?;
        self.resolved = true;
        Ok(())
    }

    pub(crate) fn fail(mut self) {
        self.coordinator.abandon(&self.key, self.generation, false);
        self.resolved = true;
    }
}

impl Drop for GraphBfsLeader {
    fn drop(&mut self) {
        if !self.resolved {
            self.coordinator.abandon(&self.key, self.generation, true);
            self.resolved = true;
        }
    }
}

pub(crate) struct GraphBfsFollower {
    coordinator: GraphBfsSingleflight,
    key: GraphBfsSingleflightKey,
    generation: u64,
    receiver: watch::Receiver<PublishedResult>,
    completed: bool,
}

impl GraphBfsFollower {
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
                () = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        }
    }
}

impl Drop for GraphBfsFollower {
    fn drop(&mut self) {
        self.coordinator.leave_follower(&self.key, self.generation, !self.completed);
    }
}

pub(crate) enum GraphBfsFollowerOutcome {
    Ready(GraphBfsPublishedResult),
    Retry,
}

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

fn update_peak(peak: &AtomicU64, value: usize) {
    peak.fetch_max(u64::try_from(value).unwrap_or(u64::MAX), Ordering::AcqRel);
}

fn state_poisoned_error(device_ordinal: usize) -> Error {
    Error::new(ErrorCode::Internal, OPERATION, "BFS singleflight state mutex was poisoned")
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("reason", "state_poisoned")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use arrow::{datatypes::Schema, record_batch::RecordBatch};
    use datafusion::{
        execution::{TaskContext, memory_pool::MemoryConsumer},
        physical_plan::{ExecutionPlan, empty::EmptyExec},
    };

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
        let GraphBfsFollowerOutcome::Ready(result) = follower.wait(&ExecutionCancellation::new()).await.unwrap() else {
            panic!("published result must be ready")
        };
        assert_eq!(result.batches().len(), 2);
        assert_eq!(pool.reserved(), 128);
        drop(result);
        assert_eq!(pool.reserved(), 128, "the follower receiver still retains the published result");
        drop(follower);
        assert_eq!(pool.reserved(), 0);
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

        assert!(matches!(follower.wait(&ExecutionCancellation::new()).await.unwrap(), GraphBfsFollowerOutcome::Retry));
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
        assert!(matches!(follower.wait(&ExecutionCancellation::new()).await.unwrap(), GraphBfsFollowerOutcome::Retry));
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
}
