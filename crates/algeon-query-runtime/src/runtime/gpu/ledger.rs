//! Versioned per-device immutable-grant ledger.
//!
//! One [`DeviceLedger`] owns admission accounting for exactly one CUDA device:
//! an immutable [`DeviceResourceProfile`] plus a mutable, versioned state of
//! active or cleaning attempt grants, frozen resident charges, device
//! admission health, and cache-internal accounting.
//!
//! The capacity contract splits the frozen managed capacity `M` into a cache
//! share `K` (`cache_cap_bytes`) and a query share `Q = M - K`. Query
//! admission and cache eviction each own one share, so no admission order can
//! promise the same bytes twice. The admission equation is normative:
//!
//! ```text
//! R + Σ Gᵢ <= Q            (frozen residents + live grants fit the query share)
//! C <= K                   (cache charges fit the cache share)
//! active_attempts + 1 <= max_active_attempts
//! new admission: R + Σ Gᵢ + Gnew <= Q
//! ```
//!
//! The query cap is explicit or is resolved inside the admission transaction
//! as `min(default_query_limit, query_share - resident_bytes)`. It is
//! never inferred from a planner estimate and never clamped to transient free
//! capacity left by other live grants.

use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use cudf::runtime::CudaStream;
use query_planning::error::{QueryError, QueryErrorCode, Result};
use serde::Serialize;

use super::DeviceResourceProfile;
#[cfg(test)]
use super::{DeviceCapabilities, DeviceResourceEnvelope};
use crate::{
    observability::{AdmissionRequestId, AttemptId},
    runtime::context::QueryServiceResourceEpoch,
};

const OPERATION: &str = "query_engine.gpu.device_ledger";

/// Admission health of one device ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeviceAdmissionHealth {
    /// New attempts may be admitted.
    Open,
    /// New admission is closed; in-flight attempts finish and release.
    Closed { reason: String, generation: u64 },
}

impl DeviceAdmissionHealth {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Closed { .. } => "closed",
        }
    }
}

/// Atomic read of one device ledger at one observation sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LedgerSnapshot {
    pub(crate) device_ordinal: usize,
    pub(crate) sequence: u64,
    pub(crate) managed_capacity_bytes: u64,
    /// Cache share `K` carved out of the managed capacity.
    pub(crate) cache_cap_bytes: u64,
    /// Query share `Q = M - K` promisable to grants plus frozen residents.
    pub(crate) query_share_bytes: u64,
    pub(crate) default_query_limit_bytes: u64,
    pub(crate) max_active_attempts: usize,
    pub(crate) active_attempts: usize,
    pub(crate) live_grants_bytes: u64,
    /// Bytes owned by frozen resident domains.
    pub(crate) frozen_resident_bytes: u64,
    /// Aggregate bytes retained by live resident owners.
    pub(crate) resident_owner_bytes: u64,
    /// Logical root stream slots still available to a new admitted attempt.
    ///
    /// These values are derived from the immutable profile and active grants
    /// while holding the ledger lock. They are not a separately sampled CUDA
    /// stream-pool counter, so placement never combines independently read
    /// resource gauges into an unsound admission fact.
    pub(crate) root_streams_available: usize,
    /// Logical auxiliary stream slots still available to admitted attempts.
    pub(crate) auxiliary_streams_available: usize,
    /// Logical root worker slots still available to a new admitted attempt.
    pub(crate) root_worker_slots_available: usize,
    /// Logical auxiliary worker slots still available to admitted attempts.
    pub(crate) auxiliary_worker_slots_available: usize,
    /// Worker slots committed by active attempts under the frozen profile.
    pub(crate) worker_depth: usize,
    pub(crate) protected_slot: bool,
    pub(crate) protected_reserve_bytes: u64,
    pub(crate) cache_evictable_resident_bytes: u64,
    pub(crate) cache_pinned_in_use_bytes: u64,
    pub(crate) health: DeviceAdmissionHealth,
}

impl LedgerSnapshot {
    /// Total logical stream slots available under the immutable profile.
    pub(crate) fn stream_slots_available(&self) -> usize {
        self.root_streams_available.saturating_add(self.auxiliary_streams_available)
    }

    /// Resolves and checks one query cap against this atomic snapshot.
    ///
    /// The check is `R + Σ Gᵢ + Gnew <= Q`: frozen residents plus live
    /// grants plus the candidate grant must fit the query share. The cache
    /// share never participates, so filling the cache cannot shrink query
    /// admission and admitting queries cannot spend cache bytes.
    pub(crate) fn admissible_query_cap_bytes(&self, requested_cap_bytes: Option<u64>) -> Option<u64> {
        self.active_attempts.checked_add(1).filter(|active| *active <= self.max_active_attempts)?;
        let resident_capacity = self.query_share_bytes.checked_sub(self.frozen_resident_bytes)?;
        let query_cap = requested_cap_bytes.unwrap_or_else(|| self.default_query_limit_bytes.min(resident_capacity));
        if query_cap == 0 {
            return None;
        }
        self.frozen_resident_bytes
            .checked_add(self.live_grants_bytes)?
            .checked_add(query_cap)
            .filter(|total| *total <= self.query_share_bytes)
            .map(|_| query_cap)
    }

    /// Whether this snapshot's resident ownership leaves enough query-share
    /// capacity for the requested cap, ignoring active attempts.
    pub(crate) fn can_ever_fit_query_cap(&self, requested_cap_bytes: Option<u64>) -> bool {
        let Some(resident_capacity) = self.query_share_bytes.checked_sub(self.frozen_resident_bytes) else {
            return false;
        };
        let cap = requested_cap_bytes.unwrap_or_else(|| self.default_query_limit_bytes.min(resident_capacity));
        cap > 0 && cap <= resident_capacity
    }
}

/// Public diagnostic report for one device ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceLedgerReport {
    pub device_ordinal: usize,
    pub sequence: u64,
    pub managed_capacity_bytes: u64,
    pub(crate) default_query_limit_bytes: u64,
    pub max_active_attempts: usize,
    pub active_attempts: usize,
    pub live_grants_bytes: u64,
    pub frozen_resident_bytes: u64,
    /// Aggregate bytes retained by resident owners at this report's sequence.
    pub resident_owner_bytes: u64,
    /// Logical root stream slots available at this report's sequence.
    pub root_streams_available: usize,
    /// Logical auxiliary stream slots available at this report's sequence.
    pub(crate) auxiliary_streams_available: usize,
    /// Logical root worker slots available at this report's sequence.
    pub root_worker_slots_available: usize,
    /// Logical auxiliary worker slots available at this report's sequence.
    pub(crate) auxiliary_worker_slots_available: usize,
    /// Worker slots committed by active attempts at this report's sequence.
    pub(crate) worker_depth: usize,
    /// Query-share bytes not currently covered by live grants or frozen
    /// residents (`Q - R - Σ Gᵢ`). Cache charges never participate.
    pub available_bytes: u64,
    /// Immediately grantable query-share bytes, projected from the same
    /// snapshot as `available_bytes` (`Q - R - Σ Gᵢ`).
    pub(crate) free_bytes: u64,
    /// Cache-internal evictable accounting, reported separately from active
    /// grants and pinned cache bytes.
    pub evictable_bytes: u64,
    pub(crate) protected_slot: bool,
    pub(crate) protected_reserve_bytes: u64,
    pub cache_evictable_resident_bytes: u64,
    /// Resident-cache bytes temporarily pinned by an active cache lease.
    pub cache_pinned_in_use_bytes: u64,
    pub health: String,
    pub grants: Vec<DeviceLedgerGrantRow>,
}

/// One active grant row in [`DeviceLedgerReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceLedgerGrantRow {
    pub grant_id: u64,
    pub admission_request_id: AdmissionRequestId,
    pub grant_bytes: u64,
    pub resident_owner_bytes: u64,
    pub(crate) frozen_resident_bytes: Option<u64>,
}

/// Immutable identity assigned when a device ledger commits one attempt grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GrantIdentity {
    pub(crate) attempt_id: AttemptId,
    pub(crate) generation: u64,
}

/// The ledger mutation that advanced an observation sequence.
///
/// A transition observer receives the matching [`LedgerSnapshot`] while the
/// ledger state mutex is still held. Observers must therefore treat the
/// snapshot as their complete input and must not synchronously re-enter the
/// same ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LedgerTransitionKind {
    AdmissionClosed,
    GrantAdmitted,
    GrantFrozenResident,
    GrantReleased,
    /// An evictable resident-cache charge was added.
    EvictableCharged,
    /// An evictable resident-cache charge was released.
    EvictableReleased,
    /// A resident-cache entry became in-use and therefore non-evictable.
    CachePinned,
    /// The final in-use reference left a resident-cache entry.
    CacheUnpinned,
    ResidentOwnerCharged,
    ResidentOwnerReleased,
}

/// One immutable ledger mutation observed atomically with its state change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LedgerTransition {
    pub(crate) kind: LedgerTransitionKind,
    pub(crate) snapshot: LedgerSnapshot,
}

/// In-lock observer of one atomic ledger transition.
pub(crate) type LedgerTransitionObserver = Arc<dyn Fn(&LedgerTransition) + Send + Sync + 'static>;

/// Post-lock observer of a grant release.
///
/// Release observers preserve the queue wakeup ordering: the grant has left
/// the ledger and the state mutex is no longer held when they run.
pub(crate) type LedgerReleaseObserver = Arc<dyn Fn(&LedgerTransition) + Send + Sync + 'static>;

/// Per-device versioned immutable-grant ledger.
#[derive(Clone)]
pub struct DeviceLedger {
    inner: Arc<DeviceLedgerInner>,
}

struct DeviceLedgerInner {
    profile: DeviceResourceProfile,
    state: Mutex<DeviceLedgerState>,
    // Registration is copy-on-write.  Transition delivery only clones this
    // Arc while the ledger state mutex is held; it must never clone a Vec or
    // allocate on an admission/resource update path.
    transition_observers: Mutex<Arc<[LedgerTransitionObserver]>>,
    // Release observers are outside the state mutex, but share the same
    // allocation-free dispatch rule as transition observers.
    release_observers: Mutex<Arc<[LedgerReleaseObserver]>>,
}

const MAX_LEDGER_TRANSITION_OBSERVER_DEPTH: usize = 64;

#[derive(Clone, Copy)]
struct LedgerTransitionObserverStack {
    ledgers: [*const DeviceLedgerInner; MAX_LEDGER_TRANSITION_OBSERVER_DEPTH],
    len: usize,
}

impl LedgerTransitionObserverStack {
    const EMPTY: Self = Self { ledgers: [std::ptr::null(); MAX_LEDGER_TRANSITION_OBSERVER_DEPTH], len: 0 };

    fn contains(&self, ledger: *const DeviceLedgerInner) -> bool {
        self.ledgers[..self.len].contains(&ledger)
    }
}

thread_local! {
    // The reentrancy guard is a fixed stack: in-lock observer dispatch must
    // remain allocation-free even on a thread's first transition.
    static LEDGER_TRANSITION_OBSERVER_STACK: Cell<LedgerTransitionObserverStack> = const { Cell::new(LedgerTransitionObserverStack::EMPTY) };
}

struct LedgerTransitionObserverGuard {
    ledger: *const DeviceLedgerInner,
}

impl LedgerTransitionObserverGuard {
    fn enter(ledger: &Arc<DeviceLedgerInner>) -> Self {
        let ledger = Arc::as_ptr(ledger);
        LEDGER_TRANSITION_OBSERVER_STACK.with(|stack| {
            let mut active = stack.get();
            assert!(
                active.len < MAX_LEDGER_TRANSITION_OBSERVER_DEPTH,
                "ledger transition observer nesting exceeds its fixed bound"
            );
            active.ledgers[active.len] = ledger;
            active.len += 1;
            stack.set(active);
        });
        Self { ledger }
    }
}

impl Drop for LedgerTransitionObserverGuard {
    fn drop(&mut self) {
        LEDGER_TRANSITION_OBSERVER_STACK.with(|stack| {
            let mut active = stack.get();
            debug_assert!(active.len > 0);
            active.len = active.len.saturating_sub(1);
            debug_assert_eq!(active.ledgers[active.len], self.ledger);
            active.ledgers[active.len] = std::ptr::null();
            stack.set(active);
        });
    }
}

impl fmt::Debug for DeviceLedgerInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceLedgerInner")
            .field("device_ordinal", &self.profile.device_ordinal)
            .field("managed_capacity_bytes", &self.profile.managed_capacity_bytes)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct DeviceLedgerState {
    sequence: u64,
    next_grant_generation: u64,
    grants: BTreeMap<u64, ActiveGrant>,
    frozen_resident_bytes: u64,
    health: DeviceAdmissionHealth,
    evictable_bytes: u64,
    pinned_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveGrant {
    admission_request_id: AdmissionRequestId,
    grant_bytes: u64,
    resident_owner_bytes: u64,
    pending_frozen_resident_bytes: Option<u64>,
    frozen_resident_bytes: Option<u64>,
}

impl DeviceLedgerState {
    fn live_grants_bytes(&self) -> u64 {
        self.grants.values().fold(0u64, |sum, grant| {
            sum.saturating_add(if grant.frozen_resident_bytes.is_some() { 0 } else { grant.grant_bytes })
        })
    }

    fn resident_owner_bytes(&self) -> u64 {
        self.grants.values().fold(0u64, |sum, grant| sum.saturating_add(grant.resident_owner_bytes))
    }
}

impl DeviceLedger {
    /// Creates one ledger over an immutable per-device profile.
    pub(crate) fn new(profile: DeviceResourceProfile) -> Self {
        Self {
            inner: Arc::new(DeviceLedgerInner {
                profile,
                state: Mutex::new(DeviceLedgerState {
                    sequence: 0,
                    next_grant_generation: 1,
                    grants: BTreeMap::new(),
                    frozen_resident_bytes: 0,
                    health: DeviceAdmissionHealth::Open,
                    evictable_bytes: 0,
                    pinned_bytes: 0,
                }),
                transition_observers: Mutex::new(Arc::from([])),
                release_observers: Mutex::new(Arc::from([])),
            }),
        }
    }

    /// Test constructor deriving a valid profile from raw capacity inputs.
    #[cfg(test)]
    pub(crate) fn for_test(device_ordinal: usize, managed_capacity_bytes: u64, max_active_attempts: usize) -> Self {
        let max_active_attempts = max_active_attempts.max(1);
        let source_work = 1usize;
        let slots = max_active_attempts.saturating_mul(2).max(max_active_attempts);
        let default_query_limit_bytes = DeviceResourceProfile::derive_default_query_limit_bytes(
            device_ordinal,
            managed_capacity_bytes,
            0,
            max_active_attempts,
        )
        .expect("test ledger profile must resolve");
        let profile = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal,
            device_capabilities: DeviceCapabilities::default(),
            configured_limit_bytes: managed_capacity_bytes,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes,
            max_active_attempts,
            source_work_per_attempt: source_work,
            stream_pool_size: slots,
            worker_job_capacity: slots,
            cache_cap_bytes: 0,
        })
        .expect("test ledger profile must resolve");
        Self::new(profile)
    }

    /// Test convenience: admit one attempt with an explicit query cap.
    #[cfg(test)]
    pub(crate) fn admit_with_configured_floor_and_soft_working_set_hint_for_test(
        &self,
        waiter_id: AdmissionRequestId,
        query_cap_bytes: u64,
        _unused_soft_working_set_hint_bytes: u64,
    ) -> Result<Option<MemoryGrant>> {
        self.try_admit_for_attempt(waiter_id, AttemptId::mint(), Some(query_cap_bytes))
    }

    /// Immutable per-device resource profile behind this ledger.
    pub fn profile(&self) -> &DeviceResourceProfile {
        &self.inner.profile
    }

    /// CUDA device ordinal owned by this ledger.
    pub fn device_ordinal(&self) -> usize {
        self.inner.profile.device_ordinal
    }

    /// Managed capacity in bytes (immutable).
    pub(crate) fn managed_capacity_bytes(&self) -> u64 {
        self.inner.profile.managed_capacity_bytes
    }

    fn in_transition_observer(&self) -> bool {
        let ledger = Arc::as_ptr(&self.inner);
        LEDGER_TRANSITION_OBSERVER_STACK.with(|stack| stack.get().contains(ledger))
    }

    fn transition_observer_reentrancy_error(&self, action: &'static str) -> QueryError {
        QueryError::new(
            QueryErrorCode::Internal,
            OPERATION,
            "device ledger transition observer synchronously re-entered the same ledger",
        )
        .with_fact("action", action)
        .with_fact("reason", "ledger_observer_reentrancy")
    }

    #[cold]
    fn panic_on_transition_observer_reentrancy(&self, action: &'static str) -> ! {
        std::panic::panic_any(self.transition_observer_reentrancy_error(action))
    }

    fn lock_state(&self, action: &'static str) -> Result<MutexGuard<'_, DeviceLedgerState>> {
        if self.in_transition_observer() {
            return Err(self.transition_observer_reentrancy_error(action));
        }
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "device ledger mutex was poisoned")
                .with_fact("action", action)
                .with_fact("reason", "ledger_mutex_poisoned")
        })
    }

    fn transition_observers(&self) -> Option<Arc<[LedgerTransitionObserver]>> {
        self.inner
            .transition_observers
            .lock()
            .map(|observers| Arc::clone(&observers))
            .map_err(|_error| {
                eprintln!("[query-engine] warning: device ledger transition-observer mutex was poisoned");
            })
            .ok()
    }

    fn release_observers(&self) -> Option<Arc<[LedgerReleaseObserver]>> {
        self.inner
            .release_observers
            .lock()
            .map(|observers| Arc::clone(&observers))
            .map_err(|_error| {
                eprintln!("[query-engine] warning: device ledger release-observer mutex was poisoned");
            })
            .ok()
    }

    fn bump_sequence_and_notify_locked(
        &self,
        state: &mut DeviceLedgerState,
        kind: LedgerTransitionKind,
    ) -> LedgerTransition {
        state.sequence = state.sequence.saturating_add(1);
        let transition = LedgerTransition { kind, snapshot: self.snapshot_locked(state) };
        if let Some(observers) = self.transition_observers() {
            for observer in observers.iter() {
                let _delivery = LedgerTransitionObserverGuard::enter(&self.inner);
                if catch_unwind(AssertUnwindSafe(|| observer(&transition))).is_err() {
                    eprintln!("[query-engine] warning: device ledger transition observer panicked");
                }
            }
        }
        transition
    }

    /// Registers an in-lock observer for every ledger sequence transition.
    ///
    /// Registration is additive. Each invocation receives a snapshot whose
    /// sequence is exactly the transition sequence; observer panics are
    /// isolated from admission and release.
    pub(crate) fn register_transition_observer(&self, observer: LedgerTransitionObserver) -> Result<()> {
        let mut registered = self.inner.transition_observers.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "device ledger transition-observer mutex was poisoned")
                .with_fact("reason", "transition_observer_mutex_poisoned")
        })?;
        let mut next = Vec::with_capacity(registered.len().saturating_add(1));
        next.extend(registered.iter().cloned());
        next.push(observer);
        *registered = Arc::from(next);
        Ok(())
    }

    /// Registers an additive post-lock grant-release observer.
    ///
    /// The observer runs only after a grant release has committed and the
    /// ledger mutex has been released. This preserves the queue-wakeup
    /// contract without conflating it with atomic state observation.
    pub(crate) fn register_release_observer(&self, observer: LedgerReleaseObserver) -> Result<()> {
        let mut registered = self.inner.release_observers.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "device ledger release-observer mutex was poisoned")
                .with_fact("reason", "release_observer_mutex_poisoned")
        })?;
        let mut next = Vec::with_capacity(registered.len().saturating_add(1));
        next.extend(registered.iter().cloned());
        next.push(observer);
        *registered = Arc::from(next);
        Ok(())
    }

    /// One atomic ledger snapshot for placement, reports, and metrics.
    pub(crate) fn snapshot(&self) -> Result<LedgerSnapshot> {
        let state = self.lock_state("snapshot")?;
        Ok(self.snapshot_locked(&state))
    }

    fn snapshot_locked(&self, state: &DeviceLedgerState) -> LedgerSnapshot {
        let profile = &self.inner.profile;
        let active_attempts = state.grants.values().filter(|grant| grant.frozen_resident_bytes.is_none()).count();
        // Admission reserves the complete root plus bounded auxiliary envelope
        // for every active attempt. Derive availability from that immutable
        // envelope under the ledger lock instead of reading mutable executor or
        // stream-pool counters independently.
        let committed_auxiliary_slots = active_attempts.saturating_mul(profile.source_work_per_attempt);
        LedgerSnapshot {
            device_ordinal: profile.device_ordinal,
            sequence: state.sequence,
            managed_capacity_bytes: profile.managed_capacity_bytes,
            cache_cap_bytes: profile.cache_cap_bytes,
            query_share_bytes: profile.query_share_bytes(),
            default_query_limit_bytes: profile.default_query_limit_bytes,
            max_active_attempts: profile.max_active_attempts,
            active_attempts,
            live_grants_bytes: state.live_grants_bytes(),
            frozen_resident_bytes: state.frozen_resident_bytes,
            resident_owner_bytes: state.resident_owner_bytes(),
            root_streams_available: profile.root_stream_count.saturating_sub(active_attempts),
            auxiliary_streams_available: profile.aux_source_work_capacity.saturating_sub(committed_auxiliary_slots),
            root_worker_slots_available: profile.root_lane_count.saturating_sub(active_attempts),
            auxiliary_worker_slots_available: profile
                .aux_source_work_capacity
                .saturating_sub(committed_auxiliary_slots),
            worker_depth: active_attempts.saturating_add(committed_auxiliary_slots),
            protected_slot: false,
            protected_reserve_bytes: 0,
            // Resident-cache charges share the same atomic ledger snapshot.
            cache_evictable_resident_bytes: state.evictable_bytes,
            cache_pinned_in_use_bytes: state.pinned_bytes,
            health: state.health.clone(),
        }
    }

    /// Public diagnostic report for this ledger.
    pub(crate) fn report(&self) -> Result<DeviceLedgerReport> {
        let state = self.lock_state("report")?;
        let snapshot = self.snapshot_locked(&state);
        let grants = state
            .grants
            .iter()
            .map(|(grant_id, active)| DeviceLedgerGrantRow {
                grant_id: *grant_id,
                admission_request_id: active.admission_request_id,
                grant_bytes: active.grant_bytes,
                resident_owner_bytes: active.resident_owner_bytes,
                frozen_resident_bytes: active.frozen_resident_bytes,
            })
            .collect();
        Ok(DeviceLedgerReport {
            device_ordinal: snapshot.device_ordinal,
            sequence: snapshot.sequence,
            managed_capacity_bytes: snapshot.managed_capacity_bytes,
            default_query_limit_bytes: snapshot.default_query_limit_bytes,
            max_active_attempts: snapshot.max_active_attempts,
            active_attempts: snapshot.active_attempts,
            live_grants_bytes: snapshot.live_grants_bytes,
            frozen_resident_bytes: snapshot.frozen_resident_bytes,
            resident_owner_bytes: snapshot.resident_owner_bytes,
            root_streams_available: snapshot.root_streams_available,
            auxiliary_streams_available: snapshot.auxiliary_streams_available,
            root_worker_slots_available: snapshot.root_worker_slots_available,
            auxiliary_worker_slots_available: snapshot.auxiliary_worker_slots_available,
            worker_depth: snapshot.worker_depth,
            available_bytes: snapshot
                .query_share_bytes
                .saturating_sub(snapshot.live_grants_bytes)
                .saturating_sub(snapshot.frozen_resident_bytes),
            free_bytes: snapshot
                .query_share_bytes
                .saturating_sub(snapshot.live_grants_bytes)
                .saturating_sub(snapshot.frozen_resident_bytes),
            evictable_bytes: state.evictable_bytes,
            protected_slot: snapshot.protected_slot,
            protected_reserve_bytes: snapshot.protected_reserve_bytes,
            cache_evictable_resident_bytes: snapshot.cache_evictable_resident_bytes,
            cache_pinned_in_use_bytes: snapshot.cache_pinned_in_use_bytes,
            health: snapshot.health.as_str().to_string(),
            grants,
        })
    }

    /// Current device admission health.
    pub(crate) fn health(&self) -> Result<DeviceAdmissionHealth> {
        let state = self.lock_state("health")?;
        Ok(state.health.clone())
    }

    /// Closes new admission on this device; in-flight grants still release.
    pub(crate) fn close_admission(&self, reason: impl Into<String>) -> Result<u64> {
        let mut state = self.lock_state("close_admission")?;
        let generation = match &state.health {
            DeviceAdmissionHealth::Open => 1,
            DeviceAdmissionHealth::Closed { generation, .. } => generation.saturating_add(1),
        };
        state.health = DeviceAdmissionHealth::Closed { reason: reason.into(), generation };
        self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::AdmissionClosed);
        Ok(generation)
    }

    /// Closes new admission only when this ledger is still open.
    ///
    /// Native device health is process-lifetime state. Multiple service
    /// contexts can observe the same fault concurrently, but only the first
    /// observation may advance the ledger health generation and define its
    /// closure reason.
    pub(crate) fn close_admission_if_open(&self, reason: impl Into<String>) -> Result<bool> {
        let mut state = self.lock_state("close_admission_if_open")?;
        if !matches!(state.health, DeviceAdmissionHealth::Open) {
            return Ok(false);
        }
        state.health = DeviceAdmissionHealth::Closed { reason: reason.into(), generation: 1 };
        self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::AdmissionClosed);
        Ok(true)
    }

    /// Returns the cap that could be admitted right now.
    #[cfg(test)]
    pub(crate) fn candidate_query_cap_bytes(&self, requested_cap_bytes: Option<u64>) -> Result<Option<u64>> {
        let state = self.lock_state("candidate_max_new_grant_bytes")?;
        let snapshot = self.snapshot_locked(&state);
        Ok(snapshot.admissible_query_cap_bytes(requested_cap_bytes))
    }

    /// Attempts one immutable grant admission for `waiter_id`.
    ///
    /// The grant is exactly the explicit cap, or the default cap resolved
    /// against resident ownership in this transaction. It is never clamped to
    /// transient capacity and planner estimates do not participate.
    pub(crate) fn try_admit_for_attempt(
        &self,
        waiter_id: AdmissionRequestId,
        attempt_id: AttemptId,
        requested_cap_bytes: Option<u64>,
    ) -> Result<Option<MemoryGrant>> {
        if requested_cap_bytes == Some(0) {
            return Err(QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, "query cap must be greater than zero")
                .with_fact("reason", "zero_query_cap")
                .with_fact("admission_request_id", waiter_id));
        }
        let mut state = self.lock_state("try_admit")?;
        if !matches!(state.health, DeviceAdmissionHealth::Open) {
            return Ok(None);
        }
        let snapshot = self.snapshot_locked(&state);
        let Some(query_cap_bytes) = snapshot.admissible_query_cap_bytes(requested_cap_bytes) else {
            return Ok(None);
        };
        self.commit_grant_locked(&mut state, waiter_id, attempt_id, query_cap_bytes).map(Some)
    }

    #[cfg(test)]
    pub(crate) fn try_admit(
        &self,
        waiter_id: AdmissionRequestId,
        requested_cap_bytes: Option<u64>,
    ) -> Result<Option<MemoryGrant>> {
        self.try_admit_for_attempt(waiter_id, AttemptId::mint(), requested_cap_bytes)
    }

    fn commit_grant_locked(
        &self,
        state: &mut DeviceLedgerState,
        waiter_id: AdmissionRequestId,
        attempt_id: AttemptId,
        grant_bytes: u64,
    ) -> Result<MemoryGrant> {
        let generation = state.next_grant_generation;
        let next_grant_generation = generation.checked_add(1).ok_or_else(|| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "device ledger grant generation exhausted")
                .with_fact("reason", "grant_generation_exhausted")
                .with_fact("device_ordinal", self.inner.profile.device_ordinal)
                .with_fact("attempt_id", attempt_id)
        })?;
        let identity = GrantIdentity { attempt_id, generation };
        state.next_grant_generation = next_grant_generation;
        state.grants.insert(
            generation,
            ActiveGrant {
                admission_request_id: waiter_id,
                grant_bytes,
                resident_owner_bytes: 0,
                pending_frozen_resident_bytes: None,
                frozen_resident_bytes: None,
            },
        );
        self.bump_sequence_and_notify_locked(state, LedgerTransitionKind::GrantAdmitted);
        Ok(MemoryGrant {
            inner: Arc::new(MemoryGrantInner {
                ledger: Arc::clone(&self.inner),
                identity,
                admission_request_id: waiter_id,
                grant_bytes,
                resident_owner_bytes: Default::default(),
                next_resident_owner_id: std::sync::atomic::AtomicU64::new(1),
                resident_owner_allocations: Mutex::new(BTreeMap::new()),
                frozen: Default::default(),
                frozen_domain: Mutex::new(None),
            }),
            resource_epoch: None,
        })
    }

    /// Releases one grant after its active or resident ownership ends.
    fn release_grant(&self, grant_id: u64) {
        if self.in_transition_observer() {
            self.panic_on_transition_observer_reentrancy("release_grant");
        }
        // The state is integer accounting; returning early on a poisoned lock would leak the grant for good.
        let mut state = self.inner.state.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(active) = state.grants.remove(&grant_id) else {
            return;
        };
        if let Some(resident_bytes) = active.frozen_resident_bytes {
            state.frozen_resident_bytes = state.frozen_resident_bytes.saturating_sub(resident_bytes);
        }
        let transition = self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::GrantReleased);
        // Release observers lock service admission mutexes, so they run outside the ledger lock.
        drop(state);
        if let Some(observers) = self.release_observers() {
            for observer in observers.iter() {
                if catch_unwind(AssertUnwindSafe(|| observer(&transition))).is_err() {
                    eprintln!("[query-engine] warning: device ledger release observer panicked");
                }
            }
        }
    }

    /// Attempts to charge bytes retained by an evictable cache.
    ///
    /// Cache eviction owns only the cache share (`C <= K`): the check never
    /// consults live query grants or frozen residents, so cache-first and
    /// query-first admission orders promise identical capacity. The query
    /// share bound (`R + Σ Gᵢ <= Q`) independently keeps the combined
    /// commitment within `M`.
    pub(crate) fn try_reserve_evictable_bytes(&self, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        let mut state = self.lock_state("try_reserve_evictable_bytes")?;
        if state.evictable_bytes.saturating_add(state.pinned_bytes).saturating_add(bytes)
            > self.inner.profile.cache_cap_bytes
        {
            return Ok(false);
        }
        state.evictable_bytes = state.evictable_bytes.saturating_add(bytes);
        self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::EvictableCharged);
        Ok(true)
    }

    /// Moves one exact resident-cache charge from evictable to pinned.
    ///
    /// The aggregate cache charge is unchanged. This narrow transition is
    /// deliberately private to the shared residency state machine: callers
    /// must not maintain independent pinned counters beside the ledger.
    pub(crate) fn pin_cache_residency_charge(&self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut state = self.lock_state("pin_cache_residency_charge")?;
        if state.evictable_bytes < bytes {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident-cache pin is missing its evictable ledger charge",
            )
            .with_fact("reason", "cache_evictable_charge_missing")
            .with_fact("requested_bytes", bytes)
            .with_fact("evictable_bytes", state.evictable_bytes));
        }
        let pinned_bytes = state.pinned_bytes.checked_add(bytes).ok_or_else(|| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "resident-cache pinned ledger charge overflowed")
                .with_fact("reason", "cache_pinned_charge_overflow")
                .with_fact("requested_bytes", bytes)
                .with_fact("pinned_bytes", state.pinned_bytes)
        })?;
        state.evictable_bytes -= bytes;
        state.pinned_bytes = pinned_bytes;
        self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::CachePinned);
        Ok(())
    }

    /// Moves one exact resident-cache charge from pinned back to evictable.
    pub(crate) fn unpin_cache_residency_charge(&self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut state = self.lock_state("unpin_cache_residency_charge")?;
        if state.pinned_bytes < bytes {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident-cache unpin is missing its pinned ledger charge",
            )
            .with_fact("reason", "cache_pinned_charge_missing")
            .with_fact("requested_bytes", bytes)
            .with_fact("pinned_bytes", state.pinned_bytes));
        }
        let evictable_bytes = state.evictable_bytes.checked_add(bytes).ok_or_else(|| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "resident-cache evictable ledger charge overflowed")
                .with_fact("reason", "cache_evictable_charge_overflow")
                .with_fact("requested_bytes", bytes)
                .with_fact("evictable_bytes", state.evictable_bytes)
        })?;
        state.pinned_bytes -= bytes;
        state.evictable_bytes = evictable_bytes;
        self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::CacheUnpinned);
        Ok(())
    }

    /// Releases the complete exact charge of a destroyed resident-cache entry.
    ///
    /// Unlike a permissive aggregate release, this refuses an
    /// undercharged release. The residency state machine calls it only after
    /// GPU-safe destruction has completed, so a mismatched byte count must
    /// fail closed instead of releasing another entry's charge.
    pub(crate) fn release_cache_residency_charge_after_destroy(&self, bytes: u64) -> Result<()> {
        self.release_exact_evictable_cache_charge(bytes, "release_cache_residency_charge_after_destroy")
    }

    /// Returns unused conservative pre-reservation bytes before a cache entry
    /// has been committed. No physical allocation owns this excess yet.
    pub(crate) fn release_cache_residency_precharge_excess(&self, bytes: u64) -> Result<()> {
        self.release_exact_evictable_cache_charge(bytes, "release_cache_residency_precharge_excess")
    }

    fn release_exact_evictable_cache_charge(&self, bytes: u64, action: &'static str) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut state = self.lock_state(action)?;
        if state.evictable_bytes < bytes {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident-cache exact release is missing its evictable ledger charge",
            )
            .with_fact("reason", "cache_exact_charge_missing")
            .with_fact("action", action)
            .with_fact("requested_bytes", bytes)
            .with_fact("evictable_bytes", state.evictable_bytes));
        }
        state.evictable_bytes -= bytes;
        self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::EvictableReleased);
        Ok(())
    }
}

/// RAII immutable attempt grant. Dropping the final clone returns its budget
/// to the device ledger; the budget never changes while the grant is live.
#[derive(Clone)]
pub struct MemoryGrant {
    inner: Arc<MemoryGrantInner>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
}

/// Shared grant retention held by GPU-resident outputs.
#[derive(Clone)]
#[doc(hidden)]
pub struct ResidentOwnerLease {
    inner: Arc<ResidentOwnerLeaseInner>,
}

struct ResidentOwnerLeaseInner {
    grant: MemoryGrant,
    bytes: u64,
    owner_id: u64,
    completion_stream: Mutex<Option<CudaStream>>,
}

struct MemoryGrantInner {
    ledger: Arc<DeviceLedgerInner>,
    identity: GrantIdentity,
    admission_request_id: AdmissionRequestId,
    grant_bytes: u64,
    resident_owner_bytes: std::sync::atomic::AtomicU64,
    next_resident_owner_id: std::sync::atomic::AtomicU64,
    resident_owner_allocations: Mutex<BTreeMap<u64, Arc<[usize]>>>,
    frozen: std::sync::atomic::AtomicBool,
    frozen_domain: Mutex<Option<Arc<cudf::QueryMemoryDomain>>>,
}

impl Drop for MemoryGrantInner {
    fn drop(&mut self) {
        let ledger = DeviceLedger { inner: Arc::clone(&self.ledger) };
        let frozen_cleanup_failure = match self.frozen_domain.get_mut() {
            Ok(Some(domain)) => match domain.usage() {
                Ok(usage)
                    if usage.active_members == 0
                        && usage.current_allocations == 0
                        && usage.current_bytes == 0
                        && usage.current_aligned_bytes == 0 =>
                {
                    None
                },
                Ok(usage) => Some(format!(
                    "frozen resident domain did not reach zero at final release (domain_id={}, active_members={}, current_allocations={}, current_bytes={}, current_aligned_bytes={})",
                    usage.domain_id,
                    usage.active_members,
                    usage.current_allocations,
                    usage.current_bytes,
                    usage.current_aligned_bytes
                )),
                Err(error) => Some(format!("frozen resident domain final usage sample failed: {error}")),
            },
            Ok(None) => None,
            Err(_error) => Some("frozen resident domain owner mutex was poisoned".to_string()),
        };
        if let Some(reason) = frozen_cleanup_failure
            && let Err(error) = ledger.close_admission_if_open(reason)
        {
            eprintln!("[query-engine] warning: failed to close admission after frozen resident cleanup: {error}");
        }
        ledger.release_grant(self.identity.generation);
    }
}

impl MemoryGrant {
    pub(crate) fn with_resource_epoch(&self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        Self { inner: Arc::clone(&self.inner), resource_epoch: Some(resource_epoch) }
    }

    fn check_resource_epoch(&self, operation: &'static str) -> Result<()> {
        self.resource_epoch.as_ref().map_or(Ok(()), |epoch| epoch.check_open(operation))
    }

    /// Unique ledger-local grant id.
    pub(crate) fn grant_id(&self) -> u64 {
        self.inner.identity.generation
    }

    /// Immutable physical-attempt identity committed by the selected ledger.
    pub fn identity(&self) -> GrantIdentity {
        self.inner.identity
    }

    /// Service-local admission request that owns this grant.
    pub fn admission_request_id(&self) -> AdmissionRequestId {
        self.inner.admission_request_id
    }

    pub(crate) fn identity_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Immutable grant budget in bytes. This value never changes while the
    /// grant is live; v1 has no live grant growth, shrink, borrowing, or
    /// revocation.
    pub fn budget_bytes(&self) -> u64 {
        self.inner.grant_bytes
    }

    #[cfg(test)]
    pub(crate) fn freeze_resident(&self, resident_bytes: u64) -> Result<()> {
        self.prepare_frozen_resident_charge(resident_bytes, None)?;
        self.publish_frozen_resident()
    }

    #[cfg(test)]
    pub(super) fn prepare_frozen_resident_for_test(&self, resident_bytes: u64) -> Result<()> {
        self.prepare_frozen_resident_charge(resident_bytes, None)
    }

    /// Atomically replaces this active grant with a measured frozen-resident
    /// charge and retains its native owner through final release.
    ///
    /// `resident_bytes` is the allocation registry's own measurement. The
    /// resident-owner charge is reconciled to that same measurement so the
    /// owner and frozen charges remain one accounting fact.
    pub(crate) fn freeze_resident_with_domain(
        &self,
        resident_bytes: u64,
        domain: Arc<cudf::QueryMemoryDomain>,
    ) -> Result<()> {
        self.reconcile_resident_owner_bytes(resident_bytes)?;
        self.prepare_frozen_resident_charge(resident_bytes, Some(domain))
    }

    /// Rebases resident ownership onto the allocator's measured bytes.
    fn reconcile_resident_owner_bytes(&self, measured_bytes: u64) -> Result<()> {
        if measured_bytes > self.inner.grant_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.reconcile_resident_owner_bytes",
                "measured resident bytes exceed the immutable query cap",
            )
            .with_fact("measured_bytes", measured_bytes)
            .with_fact("query_cap_bytes", self.inner.grant_bytes));
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("reconcile_resident_owner_bytes")?;
        let Some(active) = state.grants.get_mut(&self.inner.identity.generation) else {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.reconcile_resident_owner_bytes",
                "query grant is missing from its device ledger",
            )
            .with_fact("grant_id", self.grant_id()));
        };
        if active.resident_owner_bytes == measured_bytes {
            return Ok(());
        }
        active.resident_owner_bytes = measured_bytes;
        self.inner.resident_owner_bytes.store(measured_bytes, std::sync::atomic::Ordering::Release);
        ledger.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::ResidentOwnerCharged);
        Ok(())
    }

    fn prepare_frozen_resident_charge(
        &self,
        resident_bytes: u64,
        domain: Option<Arc<cudf::QueryMemoryDomain>>,
    ) -> Result<()> {
        if resident_bytes > self.inner.grant_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.freeze_resident",
                "frozen resident charge must fit the query cap",
            )
            .with_fact("resident_bytes", resident_bytes)
            .with_fact("query_cap_bytes", self.inner.grant_bytes));
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("freeze_resident")?;
        let active = state.grants.get(&self.inner.identity.generation).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.freeze_resident",
                "query grant is missing from its device ledger",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        let existing = active.frozen_resident_bytes.or(active.pending_frozen_resident_bytes);
        if let Some(existing_bytes) = existing {
            let frozen_domain = self.inner.frozen_domain.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InternalInvariantViolation,
                    "query_engine.runtime.gpu.memory_grant.freeze_resident",
                    "frozen resident domain owner mutex was poisoned",
                )
            })?;
            let same_domain = match (frozen_domain.as_ref(), domain.as_ref()) {
                (Some(existing), Some(domain)) => Arc::ptr_eq(existing, domain),
                (None, None) => true,
                _ => false,
            };
            return (existing_bytes == resident_bytes && same_domain).then_some(()).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InternalInvariantViolation,
                    "query_engine.runtime.gpu.memory_grant.freeze_resident",
                    "query grant was frozen with a different resident charge",
                )
                .with_fact("grant_id", self.grant_id())
                .with_fact("existing_resident_bytes", existing_bytes)
                .with_fact("resident_bytes", resident_bytes)
            });
        }
        state
            .grants
            .get_mut(&self.inner.identity.generation)
            .expect("grant existence was checked under the same ledger lock")
            .pending_frozen_resident_bytes = Some(resident_bytes);
        if let Some(domain) = domain {
            *self.inner.frozen_domain.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InternalInvariantViolation,
                    "query_engine.runtime.gpu.memory_grant.freeze_resident",
                    "frozen resident domain owner mutex was poisoned",
                )
            })? = Some(domain);
        }
        self.inner.frozen.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Publishes a prepared B-to-R transition after the attempt report is
    /// visible. Until this commit the ledger continues to account the full
    /// active grant, so no released capacity can race report publication.
    /// Proves the prepared B-to-R transition can still be published.
    ///
    /// The publish itself runs after the report is sealed, where a failure can
    /// no longer be recorded as a cleanup disposition. Checking the same
    /// preconditions before the seal keeps that failure on the reportable side
    /// of the boundary.
    pub(crate) fn validate_frozen_resident_publish(&self) -> Result<()> {
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let state = ledger.lock_state("validate_frozen_resident_publish")?;
        let active = state.grants.get(&self.inner.identity.generation).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.validate_frozen_resident_publish",
                "prepared frozen resident grant is missing from its device ledger",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        if active.frozen_resident_bytes.is_some() {
            return Ok(());
        }
        let resident_bytes = active.pending_frozen_resident_bytes.ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.validate_frozen_resident_publish",
                "frozen resident transition was not prepared",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        state.frozen_resident_bytes.checked_add(resident_bytes).map(|_| ()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.validate_frozen_resident_publish",
                "frozen resident accounting would overflow",
            )
            .with_fact("grant_id", self.grant_id())
            .with_fact("resident_bytes", resident_bytes)
        })
    }

    pub(crate) fn publish_frozen_resident(&self) -> Result<()> {
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("publish_frozen_resident")?;
        let active = state.grants.get_mut(&self.inner.identity.generation).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.publish_frozen_resident",
                "prepared frozen resident grant is missing from its device ledger",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        if active.frozen_resident_bytes.is_some() {
            return Ok(());
        }
        let resident_bytes = active.pending_frozen_resident_bytes.take().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.publish_frozen_resident",
                "frozen resident transition was not prepared",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        state.frozen_resident_bytes = state.frozen_resident_bytes.checked_add(resident_bytes).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.publish_frozen_resident",
                "frozen resident accounting overflowed",
            )
        })?;
        state
            .grants
            .get_mut(&self.inner.identity.generation)
            .expect("grant existence was checked under the same ledger lock")
            .frozen_resident_bytes = Some(resident_bytes);
        ledger.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::GrantFrozenResident);
        Ok(())
    }

    pub(crate) fn is_frozen_resident(&self) -> bool {
        self.inner.frozen.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Device ordinal that owns this grant.
    pub fn device_ordinal(&self) -> usize {
        self.inner.ledger.profile.device_ordinal
    }

    /// Device-resident bytes currently owned by live values under this grant.
    pub(crate) fn resident_owner_bytes(&self) -> u64 {
        self.inner.resident_owner_bytes.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Records newly published device-resident bytes under this grant.
    fn add_resident_owner_bytes(&self, bytes: u64) -> Result<()> {
        self.check_resource_epoch("query_engine.runtime.gpu.memory_grant.add_resident_owner_bytes")?;
        if bytes == 0 {
            return Ok(());
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("add_resident_owner_bytes")?;
        let Some(active) = state.grants.get_mut(&self.inner.identity.generation) else {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.add_resident_owner_bytes",
                "resident owner was published after its grant left the device ledger",
            ));
        };
        if active.frozen_resident_bytes.is_some() {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.add_resident_owner_bytes",
                "resident owner was published after the grant was frozen",
            ));
        }
        let next_resident_owner_bytes = active.resident_owner_bytes.checked_add(bytes).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.memory_grant.add_resident_owner_bytes",
                "resident owner byte accounting overflowed",
            )
        })?;
        active.resident_owner_bytes = next_resident_owner_bytes;
        self.inner.resident_owner_bytes.store(next_resident_owner_bytes, std::sync::atomic::Ordering::Release);
        ledger.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::ResidentOwnerCharged);
        Ok(())
    }

    fn release_resident_owner_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        if ledger.in_transition_observer() {
            ledger.panic_on_transition_observer_reentrancy("release_resident_owner_bytes");
        }
        let mut state = match ledger.inner.state.lock() {
            Ok(state) => state,
            Err(_error) => {
                eprintln!("[query-engine] warning: failed to release resident owner bytes");
                return;
            },
        };
        self.release_locked(&ledger, &mut state, bytes);
    }

    /// Applies one exact release while the ledger state lock is held.
    fn release_locked(&self, ledger: &DeviceLedger, state: &mut DeviceLedgerState, bytes: u64) {
        let Some(active) = state.grants.get_mut(&self.inner.identity.generation) else {
            return;
        };
        let next_resident_owner_bytes = match active.resident_owner_bytes.checked_sub(bytes) {
            Some(next) => next,
            // Frozen ownership is rebased onto the allocator measurement.
            // Aliased owners can therefore release a larger pre-freeze charge.
            None if active.frozen_resident_bytes.is_some() || active.pending_frozen_resident_bytes.is_some() => 0,
            None => {
                eprintln!("[query-engine] warning: resident owner release exceeded its active charge");
                return;
            },
        };
        active.resident_owner_bytes = next_resident_owner_bytes;
        self.inner.resident_owner_bytes.store(next_resident_owner_bytes, std::sync::atomic::Ordering::Release);
        ledger.bump_sequence_and_notify_locked(state, LedgerTransitionKind::ResidentOwnerReleased);
    }

    /// Records and retains a newly published resident owner.
    #[doc(hidden)]
    pub fn retain_resident_owner(&self, bytes: u64) -> Result<Option<ResidentOwnerLease>> {
        if bytes == 0 {
            return Ok(None);
        }
        self.add_resident_owner_bytes(bytes)?;
        Ok(Some(self.new_resident_owner_lease(bytes)?))
    }

    fn new_resident_owner_lease(&self, bytes: u64) -> Result<ResidentOwnerLease> {
        let owner_id = self.inner.next_resident_owner_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner
            .resident_owner_allocations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(owner_id, Arc::from([]));
        // The derived owner shares only the grant's backend service
        // generation, never its query anchor: residents outlive their
        // attempt, and retaining the query epoch would stall report
        // publication and the frozen-resident handoff. The generation still
        // cannot be torn down and reused while any alias retains GPU bytes.
        // Derived owners never mint a second ledger or a second generation.
        let resource_epoch = self
            .resource_epoch
            .as_ref()
            .map(|epoch| {
                epoch.service_generation_only("query_engine.runtime.gpu.memory_grant.new_resident_owner_lease")
            })
            .transpose()?;
        Ok(ResidentOwnerLease {
            inner: Arc::new(ResidentOwnerLeaseInner {
                grant: MemoryGrant { inner: Arc::clone(&self.inner), resource_epoch },
                bytes,
                owner_id,
                completion_stream: Mutex::new(None),
            }),
        })
    }

    pub(crate) fn published_resident_owner_count(&self) -> usize {
        self.inner.resident_owner_allocations.lock().unwrap_or_else(std::sync::PoisonError::into_inner).len()
    }

    pub(crate) fn published_resident_allocation_pointers(&self) -> std::collections::BTreeSet<usize> {
        self.inner
            .resident_owner_allocations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .flat_map(|pointers| pointers.iter().copied())
            .collect()
    }
}

impl ResidentOwnerLease {
    pub(crate) fn set_completion_stream(&self, stream: CudaStream) -> Result<()> {
        let mut completion_stream = self.inner.completion_stream.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.gpu.frozen_resident_lease.completion_stream",
                "frozen resident completion stream mutex was poisoned",
            )
        })?;
        if let Some(existing) = completion_stream.as_ref() {
            if existing.identity() != stream.identity() {
                return Err(QueryError::new(
                    QueryErrorCode::InternalInvariantViolation,
                    "query_engine.runtime.gpu.frozen_resident_lease.completion_stream",
                    "a resident owner cannot be rebound to a different completion stream",
                )
                .with_fact("existing_stream", existing.identity())
                .with_fact("replacement_stream", stream.identity()));
            }
            return Ok(());
        }
        *completion_stream = Some(stream);
        Ok(())
    }

    pub(crate) fn set_backing_allocation_pointers(&self, mut pointers: Vec<usize>) {
        pointers.retain(|pointer| *pointer != 0);
        pointers.sort_unstable();
        pointers.dedup();
        self.inner
            .grant
            .inner
            .resident_owner_allocations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(self.inner.owner_id, Arc::from(pointers));
    }

    pub(crate) fn is_frozen_resident(&self) -> bool {
        self.inner.grant.is_frozen_resident()
    }

    /// Bytes retained by this handle in the owning memory grant.
    pub(crate) fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Whether this retention consumes capacity from `grant`.
    pub(crate) fn belongs_to(&self, grant: &MemoryGrant) -> bool {
        self.inner.grant.identity_key() == grant.identity_key()
    }

    /// Whether this owner shares its grant's backend generation lease.
    ///
    /// A leased owner keeps the device generation alive until its GPU bytes
    /// are released. Grants admitted straight from a ledger, without a query
    /// service epoch, have no generation to share.
    #[cfg(test)]
    pub(crate) fn has_resource_epoch(&self) -> bool {
        self.inner.grant.resource_epoch.is_some()
    }

    /// Whether this owner also retains its query anchor.
    ///
    /// Resident owners must never retain it: the query epoch has to retire
    /// for the attempt report and the frozen-resident handoff to publish.
    #[cfg(test)]
    pub(crate) fn retains_query_anchor(&self) -> bool {
        self.inner.grant.resource_epoch.as_ref().is_some_and(|epoch| epoch.has_query_anchor())
    }

    pub(crate) fn identity_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Retains a derived output owner under the same grant.
    pub(crate) fn retain_derived_owner(&self, bytes: u64) -> Result<Option<Self>> {
        self.inner.grant.retain_resident_owner(bytes)
    }
}

impl Drop for ResidentOwnerLeaseInner {
    fn drop(&mut self) {
        let cleanup_failure = match (self.grant.is_frozen_resident(), self.completion_stream.get_mut()) {
            (false, _) => None,
            (true, Ok(Some(stream))) => stream
                .synchronize_for_teardown("query_engine.runtime.gpu.frozen_resident_lease.drop")
                .err()
                .map(|error| format!("frozen resident completion fence failed: {error}")),
            (true, Ok(None)) => {
                Some("frozen resident owner reached final release without a completion stream".to_string())
            },
            (true, Err(_error)) => Some("frozen resident completion stream mutex was poisoned".to_string()),
        };
        if let Some(reason) = cleanup_failure {
            let ledger = DeviceLedger { inner: Arc::clone(&self.grant.inner.ledger) };
            if let Err(error) = ledger.close_admission_if_open(reason) {
                eprintln!(
                    "[query-engine] warning: failed to close admission after resident completion failure: {error}"
                );
            }
        }
        self.grant
            .inner
            .resident_owner_allocations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.owner_id);
        self.grant.release_resident_owner_bytes(self.bytes);
    }
}

impl fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("identity", &self.inner.identity)
            .field("admission_request_id", &self.inner.admission_request_id)
            .field("grant_bytes", &self.inner.grant_bytes)
            .field("resident_owner_bytes", &self.resident_owner_bytes())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for DeviceLedger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceLedger")
            .field("device_ordinal", &self.inner.profile.device_ordinal)
            .field("managed_capacity_bytes", &self.inner.profile.managed_capacity_bytes)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod tests;
