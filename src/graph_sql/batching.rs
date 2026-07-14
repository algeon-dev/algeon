//! Exact-request singleflight for expensive, cache-resident BFS calls.
//!
//! libcugraph multi-source BFS produces one nearest-source forest, not one
//! independent result per source. Serving coalescing therefore groups only
//! requests with identical traversal semantics and publishes the completed
//! host result to followers.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use arrow::record_batch::RecordBatch;
use nexus_query_engine::{plan::ResidentGraphKey, runtime::ExecutionCancellation};
use tokio::sync::watch;

use crate::error::{Error, ErrorCode, Result};

const OPERATION: &str = "datafusion_nexus.cugraph_sql.bfs_singleflight";
const DEFAULT_MAX_BATCH_WIDTH: usize = 32;

/// Exact compatibility key for one device-local independent BFS traversal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CugraphBfsSingleflightKey {
    pub(crate) device_ordinal: usize,
    pub(crate) resident_graph_key: Option<ResidentGraphKey>,
    pub(crate) source_vertex: i64,
    pub(crate) normalized_output: bool,
    pub(crate) projection: Option<Vec<usize>>,
    pub(crate) limit: Option<usize>,
}

/// Additive diagnostics for exact-request BFS serving coalescing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CugraphBfsSingleflightSnapshot {
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
pub(crate) struct CugraphBfsSingleflight {
    inner: Arc<SingleflightInner>,
}

impl Default for CugraphBfsSingleflight {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_BATCH_WIDTH)
    }
}

impl CugraphBfsSingleflight {
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

    pub(crate) fn join(&self, key: CugraphBfsSingleflightKey) -> Result<CugraphBfsSingleflightJoin> {
        self.inner.counters.eligible_requests.fetch_add(1, Ordering::Relaxed);
        let mut state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
        if let Some(groups) = state.groups.get_mut(&key)
            && let Some(group) = groups.iter_mut().find(|group| group.participants < self.inner.max_batch_width)
        {
            group.participants += 1;
            update_peak(&self.inner.counters.peak_width, group.participants);
            return Ok(CugraphBfsSingleflightJoin::Follower(CugraphBfsFollower {
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
        Ok(CugraphBfsSingleflightJoin::Leader(CugraphBfsLeader {
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

    pub(crate) fn snapshot(&self) -> CugraphBfsSingleflightSnapshot {
        let counters = &self.inner.counters;
        CugraphBfsSingleflightSnapshot {
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

    fn publish(&self, key: &CugraphBfsSingleflightKey, generation: u64, result: PublishedResult) -> Result<()> {
        let sender = {
            let mut state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
            remove_group(&mut state, key, generation).map(|group| group.sender)
        };
        if let Some(sender) = sender {
            sender.send_replace(result);
        }
        Ok(())
    }

    fn participant_count(&self, key: &CugraphBfsSingleflightKey, generation: u64) -> Result<usize> {
        let state = self.inner.state.lock().map_err(|_error| state_poisoned_error(key.device_ordinal))?;
        Ok(state
            .groups
            .get(key)
            .and_then(|groups| groups.iter().find(|group| group.generation == generation))
            .map_or(0, |group| group.participants))
    }

    fn abandon(&self, key: &CugraphBfsSingleflightKey, generation: u64, cancelled: bool) {
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

    fn leave_follower(&self, key: &CugraphBfsSingleflightKey, generation: u64, cancelled: bool) {
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
    groups: HashMap<CugraphBfsSingleflightKey, Vec<SingleflightGroup>>,
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
    Ready(Arc<RecordBatch>),
    Retry,
}

pub(crate) enum CugraphBfsSingleflightJoin {
    Leader(CugraphBfsLeader),
    Follower(CugraphBfsFollower),
}

pub(crate) struct CugraphBfsLeader {
    coordinator: CugraphBfsSingleflight,
    key: CugraphBfsSingleflightKey,
    generation: u64,
    traversal_submitted: bool,
    resolved: bool,
}

impl CugraphBfsLeader {
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

    pub(crate) fn publish(&mut self, batch: Arc<RecordBatch>) -> Result<()> {
        self.coordinator.publish(&self.key, self.generation, PublishedResult::Ready(batch))?;
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

impl Drop for CugraphBfsLeader {
    fn drop(&mut self) {
        if !self.resolved {
            self.coordinator.abandon(&self.key, self.generation, true);
            self.resolved = true;
        }
    }
}

pub(crate) struct CugraphBfsFollower {
    coordinator: CugraphBfsSingleflight,
    key: CugraphBfsSingleflightKey,
    generation: u64,
    receiver: watch::Receiver<PublishedResult>,
    completed: bool,
}

impl CugraphBfsFollower {
    pub(crate) async fn wait(&mut self, cancellation: &ExecutionCancellation) -> Result<CugraphBfsFollowerOutcome> {
        loop {
            if cancellation.is_cancelled() {
                return Err(Error::from_native(
                    OPERATION,
                    "cancelled while waiting for a coalesced BFS result",
                    cancellation.check(OPERATION).expect_err("cancellation checked above"),
                ));
            }
            match self.receiver.borrow().clone() {
                PublishedResult::Ready(batch) => {
                    self.completed = true;
                    return Ok(CugraphBfsFollowerOutcome::Ready(batch));
                },
                PublishedResult::Retry => {
                    self.completed = true;
                    return Ok(CugraphBfsFollowerOutcome::Retry);
                },
                PublishedResult::Pending => {},
            }
            tokio::select! {
                changed = self.receiver.changed() => {
                    if changed.is_err() {
                        self.completed = true;
                        return Ok(CugraphBfsFollowerOutcome::Retry);
                    }
                },
                () = tokio::time::sleep(Duration::from_millis(5)) => {},
            }
        }
    }
}

impl Drop for CugraphBfsFollower {
    fn drop(&mut self) {
        self.coordinator.leave_follower(&self.key, self.generation, !self.completed);
    }
}

pub(crate) enum CugraphBfsFollowerOutcome {
    Ready(Arc<RecordBatch>),
    Retry,
}

fn remove_group(
    state: &mut SingleflightState,
    key: &CugraphBfsSingleflightKey,
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
    use arrow::{datatypes::Schema, record_batch::RecordBatch};

    use super::*;

    fn key(source_vertex: i64) -> CugraphBfsSingleflightKey {
        CugraphBfsSingleflightKey {
            device_ordinal: 0,
            resident_graph_key: None,
            source_vertex,
            normalized_output: true,
            projection: None,
            limit: None,
        }
    }

    fn empty_batch() -> Arc<RecordBatch> {
        Arc::new(RecordBatch::new_empty(Arc::new(Schema::empty())))
    }

    #[tokio::test]
    async fn identical_requests_share_one_physical_traversal() {
        let coordinator = CugraphBfsSingleflight::new(4);
        let CugraphBfsSingleflightJoin::Leader(mut leader) = coordinator.join(key(1)).unwrap() else {
            panic!("first request must lead")
        };
        let CugraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
            panic!("second request must follow")
        };

        leader.record_traversal_submission();
        leader.publish(empty_batch()).unwrap();
        assert!(matches!(
            follower.wait(&ExecutionCancellation::new()).await.unwrap(),
            CugraphBfsFollowerOutcome::Ready(_)
        ));
        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.eligible_requests, 2);
        assert_eq!(snapshot.physical_traversals, 1);
        assert_eq!(snapshot.groups_formed, 1);
        assert_eq!(snapshot.peak_width, 2);
    }

    #[test]
    fn batch_width_overflow_starts_an_independent_generation() {
        let coordinator = CugraphBfsSingleflight::new(2);
        let CugraphBfsSingleflightJoin::Leader(mut first) = coordinator.join(key(1)).unwrap() else {
            panic!("first request must lead")
        };
        let second = coordinator.join(key(1)).unwrap();
        let CugraphBfsSingleflightJoin::Leader(mut overflow) = coordinator.join(key(1)).unwrap() else {
            panic!("overflow request must lead")
        };
        first.record_traversal_submission();
        overflow.record_traversal_submission();
        assert!(matches!(second, CugraphBfsSingleflightJoin::Follower(_)));

        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.physical_traversals, 2);
        assert_eq!(snapshot.overflow_groups, 1);
    }

    #[tokio::test]
    async fn failed_leader_releases_followers_to_retry() {
        let coordinator = CugraphBfsSingleflight::new(4);
        let CugraphBfsSingleflightJoin::Leader(leader) = coordinator.join(key(1)).unwrap() else {
            panic!("first request must lead")
        };
        let CugraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
            panic!("second request must follow")
        };
        leader.fail();

        assert!(matches!(
            follower.wait(&ExecutionCancellation::new()).await.unwrap(),
            CugraphBfsFollowerOutcome::Retry
        ));
        assert_eq!(coordinator.snapshot().leader_failures, 1);
        assert_eq!(coordinator.snapshot().physical_traversals, 0);
        assert!(matches!(coordinator.join(key(1)).unwrap(), CugraphBfsSingleflightJoin::Leader(_)));
    }

    #[tokio::test]
    async fn dropped_leader_is_cancelled_and_releases_followers() {
        let coordinator = CugraphBfsSingleflight::new(4);
        let CugraphBfsSingleflightJoin::Leader(leader) = coordinator.join(key(1)).unwrap() else {
            panic!("first request must lead")
        };
        let CugraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
            panic!("second request must follow")
        };

        drop(leader);
        assert!(matches!(
            follower.wait(&ExecutionCancellation::new()).await.unwrap(),
            CugraphBfsFollowerOutcome::Retry
        ));
        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.cancelled_waiters, 1);
        assert_eq!(snapshot.leader_failures, 0);
        assert_eq!(snapshot.physical_traversals, 0);
        assert!(matches!(coordinator.join(key(1)).unwrap(), CugraphBfsSingleflightJoin::Leader(_)));
    }

    #[tokio::test]
    async fn cancelled_follower_leaves_the_leader_running() {
        let coordinator = CugraphBfsSingleflight::new(4);
        let CugraphBfsSingleflightJoin::Leader(leader) = coordinator.join(key(1)).unwrap() else {
            panic!("first request must lead")
        };
        let CugraphBfsSingleflightJoin::Follower(mut follower) = coordinator.join(key(1)).unwrap() else {
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
        let coordinator = CugraphBfsSingleflight::new(4);
        let CugraphBfsSingleflightJoin::Leader(mut first) = coordinator.join(key(1)).unwrap() else {
            panic!("first source must lead")
        };
        let CugraphBfsSingleflightJoin::Leader(mut second) = coordinator.join(key(2)).unwrap() else {
            panic!("second source must lead")
        };
        first.record_traversal_submission();
        second.record_traversal_submission();
        assert_eq!(coordinator.snapshot().physical_traversals, 2);
    }
}
