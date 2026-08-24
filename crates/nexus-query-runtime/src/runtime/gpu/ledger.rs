//! Versioned per-device immutable-grant ledger.
//!
//! One [`DeviceLedger`] owns admission accounting for exactly one CUDA device:
//! an immutable [`DeviceResourceProfile`] plus a mutable, versioned state of
//! active or cleaning attempt grants, frozen resident charges, device
//! admission health, and cache-internal accounting. Cache accounting remains
//! diagnostic state and is not part of query admission.
//!
//! The admission equation is normative:
//!
//! ```text
//! active_attempts + 1 <= max_active_attempts
//! resident_bytes + live_grants + query_cap <= managed_capacity
//! ```
//!
//! The query cap is explicit or is resolved inside the admission transaction
//! as `min(default_query_limit, managed_capacity - resident_bytes)`. It is
//! never inferred from a planner estimate and never clamped to transient free
//! capacity.

use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, MutexGuard},
};

use nexus_query_planning::error::{QueryError, QueryErrorCode, Result};
use serde::Serialize;

use super::DeviceResourceProfile;
#[cfg(any(test, feature = "test-hooks"))]
use super::{DeviceCapabilities, DeviceResourceEnvelope};
use crate::{
    observability::{AdmissionRequestId, AttemptId},
    runtime::context::QueryServiceResourceEpoch,
};

const OPERATION: &str = "nexus_query_engine.gpu.device_ledger";

/// Admission health of one device ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceAdmissionHealth {
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
pub struct LedgerSnapshot {
    pub device_ordinal: usize,
    pub sequence: u64,
    pub managed_capacity_bytes: u64,
    pub default_query_limit_bytes: u64,
    pub max_active_attempts: usize,
    pub active_attempts: usize,
    pub live_grants_bytes: u64,
    /// Bytes owned by frozen resident domains.
    pub frozen_resident_bytes: u64,
    /// Aggregate query-owned bytes currently reserved inside live grants.
    ///
    /// This is deliberately distinct from `live_grants_bytes`: grants prove
    /// admission capacity while live reservations describe the bytes currently
    /// retained by query work or escaping results.
    pub live_reserved_bytes: u64,
    /// Logical root stream slots still available to a new admitted attempt.
    ///
    /// These values are derived from the immutable profile and active grants
    /// while holding the ledger lock. They are not a separately sampled CUDA
    /// stream-pool counter, so placement never combines independently read
    /// resource gauges into an unsound admission fact.
    pub root_streams_available: usize,
    /// Logical auxiliary stream slots still available to admitted attempts.
    pub auxiliary_streams_available: usize,
    /// Logical root worker slots still available to a new admitted attempt.
    pub root_worker_slots_available: usize,
    /// Logical auxiliary worker slots still available to admitted attempts.
    pub auxiliary_worker_slots_available: usize,
    /// Worker slots committed by active attempts under the frozen profile.
    pub worker_depth: usize,
    pub protected_slot: bool,
    pub protected_reserve_bytes: u64,
    pub cache_evictable_resident_bytes: u64,
    pub cache_pinned_in_use_bytes: u64,
    pub health: DeviceAdmissionHealth,
}

impl LedgerSnapshot {
    /// Total logical stream slots available under the immutable profile.
    pub fn stream_slots_available(&self) -> usize {
        self.root_streams_available.saturating_add(self.auxiliary_streams_available)
    }

    /// Total logical worker slots available under the immutable profile.
    pub fn worker_slots_available(&self) -> usize {
        self.root_worker_slots_available.saturating_add(self.auxiliary_worker_slots_available)
    }

    /// Resolves and checks one query cap against this atomic snapshot.
    pub fn admissible_query_cap_bytes(&self, requested_cap_bytes: Option<u64>) -> Option<u64> {
        self.active_attempts.checked_add(1).filter(|active| *active <= self.max_active_attempts)?;
        let resident_capacity = self.managed_capacity_bytes.checked_sub(self.frozen_resident_bytes)?;
        let query_cap = requested_cap_bytes.unwrap_or_else(|| self.default_query_limit_bytes.min(resident_capacity));
        if query_cap == 0 {
            return None;
        }
        self.frozen_resident_bytes
            .checked_add(self.live_grants_bytes)?
            .checked_add(query_cap)
            .filter(|total| *total <= self.managed_capacity_bytes)
            .map(|_| query_cap)
    }

    /// Whether this snapshot's resident ownership leaves enough configured
    /// capacity for the requested cap, ignoring active attempts.
    pub fn can_ever_fit_query_cap(&self, requested_cap_bytes: Option<u64>) -> bool {
        let Some(resident_capacity) = self.managed_capacity_bytes.checked_sub(self.frozen_resident_bytes) else {
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
    pub default_query_limit_bytes: u64,
    pub max_active_attempts: usize,
    pub active_attempts: usize,
    pub live_grants_bytes: u64,
    pub frozen_resident_bytes: u64,
    /// Aggregate query-owned live reservations at this report's sequence.
    pub live_reserved_bytes: u64,
    /// Logical root stream slots available at this report's sequence.
    pub root_streams_available: usize,
    /// Logical auxiliary stream slots available at this report's sequence.
    pub auxiliary_streams_available: usize,
    /// Logical root worker slots available at this report's sequence.
    pub root_worker_slots_available: usize,
    /// Logical auxiliary worker slots available at this report's sequence.
    pub auxiliary_worker_slots_available: usize,
    /// Worker slots committed by active attempts at this report's sequence.
    pub worker_depth: usize,
    /// Managed bytes not currently covered by live grants or resident domains.
    pub available_bytes: u64,
    /// Immediately grantable bytes: managed minus live grants minus
    /// cache-internal evictable charges.
    pub free_bytes: u64,
    /// Cache-internal evictable accounting, reported separately from active
    /// grants and pinned cache bytes.
    pub evictable_bytes: u64,
    pub protected_slot: bool,
    pub protected_reserve_bytes: u64,
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
    pub live_reserved_bytes: u64,
    pub frozen_resident_bytes: Option<u64>,
}

/// Immutable identity assigned when a device ledger commits one attempt grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GrantIdentity {
    pub attempt_id: AttemptId,
    pub generation: u64,
}

/// The ledger mutation that advanced an observation sequence.
///
/// A transition observer receives the matching [`LedgerSnapshot`] while the
/// ledger state mutex is still held. Observers must therefore treat the
/// snapshot as their complete input and must not synchronously re-enter the
/// same ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum LedgerTransitionKind {
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
    LiveReservationCharged,
    LiveReservationReleased,
    AttemptPhaseActivated,
    AttemptPhaseReleased,
}

/// One immutable ledger mutation observed atomically with its state change.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct LedgerTransition {
    pub kind: LedgerTransitionKind,
    pub snapshot: LedgerSnapshot,
}

/// In-lock observer of one atomic ledger transition.
#[doc(hidden)]
pub type LedgerTransitionObserver = Arc<dyn Fn(&LedgerTransition) + Send + Sync + 'static>;

/// Post-lock observer of a grant release.
///
/// Release observers preserve the queue wakeup ordering: the grant has left
/// the ledger and the state mutex is no longer held when they run.
#[doc(hidden)]
pub type LedgerReleaseObserver = Arc<dyn Fn(&LedgerTransition) + Send + Sync + 'static>;

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
    next_phase_generation: u64,
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
    live_reserved_bytes: u64,
    active_phase: Option<ActiveGrantPhase>,
    pending_frozen_resident_bytes: Option<u64>,
    frozen_resident_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct ActiveGrantPhase {
    generation: u64,
    required_bytes: u64,
}

impl DeviceLedgerState {
    fn live_grants_bytes(&self) -> u64 {
        self.grants.values().fold(0u64, |sum, grant| {
            sum.saturating_add(if grant.frozen_resident_bytes.is_some() { 0 } else { grant.grant_bytes })
        })
    }

    fn live_reserved_bytes(&self) -> u64 {
        self.grants.values().fold(0u64, |sum, grant| sum.saturating_add(grant.live_reserved_bytes))
    }
}

impl DeviceLedger {
    /// Creates one ledger over an immutable per-device profile.
    #[doc(hidden)]
    pub fn new(profile: DeviceResourceProfile) -> Self {
        Self {
            inner: Arc::new(DeviceLedgerInner {
                profile,
                state: Mutex::new(DeviceLedgerState {
                    sequence: 0,
                    next_grant_generation: 1,
                    next_phase_generation: 1,
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
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn for_test(device_ordinal: usize, managed_capacity_bytes: u64, max_active_attempts: usize) -> Self {
        let max_active_attempts = max_active_attempts.max(1);
        let source_work = 1usize;
        let slots = max_active_attempts.saturating_mul(2).max(max_active_attempts);
        let profile = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal,
            device_capabilities: DeviceCapabilities::default(),
            configured_limit_bytes: managed_capacity_bytes,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: (managed_capacity_bytes
                / u64::try_from(max_active_attempts).unwrap_or(1).max(8))
            .max(1),
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
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn admit_with_configured_floor_and_soft_working_set_hint_for_test(
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
    pub fn managed_capacity_bytes(&self) -> u64 {
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
                eprintln!("[nexus-query-engine] warning: device ledger transition-observer mutex was poisoned");
            })
            .ok()
    }

    fn release_observers(&self) -> Option<Arc<[LedgerReleaseObserver]>> {
        self.inner
            .release_observers
            .lock()
            .map(|observers| Arc::clone(&observers))
            .map_err(|_error| {
                eprintln!("[nexus-query-engine] warning: device ledger release-observer mutex was poisoned");
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
                    eprintln!("[nexus-query-engine] warning: device ledger transition observer panicked");
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
    #[doc(hidden)]
    pub fn register_transition_observer(&self, observer: LedgerTransitionObserver) -> Result<()> {
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
    #[doc(hidden)]
    pub fn register_release_observer(&self, observer: LedgerReleaseObserver) -> Result<()> {
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
    pub fn snapshot(&self) -> Result<LedgerSnapshot> {
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
            default_query_limit_bytes: profile.default_query_limit_bytes,
            max_active_attempts: profile.max_active_attempts,
            active_attempts,
            live_grants_bytes: state.live_grants_bytes(),
            frozen_resident_bytes: state.frozen_resident_bytes,
            live_reserved_bytes: state.live_reserved_bytes(),
            root_streams_available: profile.root_stream_count.saturating_sub(active_attempts),
            auxiliary_streams_available: profile.aux_source_work_capacity.saturating_sub(committed_auxiliary_slots),
            root_worker_slots_available: profile.root_lane_count.saturating_sub(active_attempts),
            auxiliary_worker_slots_available: profile
                .aux_source_work_capacity
                .saturating_sub(committed_auxiliary_slots),
            worker_depth: active_attempts.saturating_add(committed_auxiliary_slots),
            protected_slot: false,
            protected_reserve_bytes: 0,
            // Resident-cache charges are ledger-owned and share this atomic
            // snapshot sequence with grants and reservations.
            cache_evictable_resident_bytes: state.evictable_bytes,
            cache_pinned_in_use_bytes: state.pinned_bytes,
            health: state.health.clone(),
        }
    }

    /// Public diagnostic report for this ledger.
    pub fn report(&self) -> Result<DeviceLedgerReport> {
        let state = self.lock_state("report")?;
        let snapshot = self.snapshot_locked(&state);
        let grants = state
            .grants
            .iter()
            .map(|(grant_id, active)| DeviceLedgerGrantRow {
                grant_id: *grant_id,
                admission_request_id: active.admission_request_id,
                grant_bytes: active.grant_bytes,
                live_reserved_bytes: active.live_reserved_bytes,
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
            live_reserved_bytes: snapshot.live_reserved_bytes,
            root_streams_available: snapshot.root_streams_available,
            auxiliary_streams_available: snapshot.auxiliary_streams_available,
            root_worker_slots_available: snapshot.root_worker_slots_available,
            auxiliary_worker_slots_available: snapshot.auxiliary_worker_slots_available,
            worker_depth: snapshot.worker_depth,
            available_bytes: snapshot
                .managed_capacity_bytes
                .saturating_sub(snapshot.live_grants_bytes)
                .saturating_sub(snapshot.frozen_resident_bytes),
            free_bytes: snapshot
                .managed_capacity_bytes
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
    pub fn health(&self) -> Result<DeviceAdmissionHealth> {
        let state = self.lock_state("health")?;
        Ok(state.health.clone())
    }

    /// Closes new admission on this device; in-flight grants still release.
    pub fn close_admission(&self, reason: impl Into<String>) -> Result<u64> {
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
    #[doc(hidden)]
    pub fn close_admission_if_open(&self, reason: impl Into<String>) -> Result<bool> {
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
    #[doc(hidden)]
    pub fn candidate_query_cap_bytes(&self, requested_cap_bytes: Option<u64>) -> Result<Option<u64>> {
        let state = self.lock_state("candidate_max_new_grant_bytes")?;
        let snapshot = self.snapshot_locked(&state);
        Ok(snapshot.admissible_query_cap_bytes(requested_cap_bytes))
    }

    /// Attempts one immutable grant admission for `waiter_id`.
    ///
    /// The grant is exactly the explicit cap, or the default cap resolved
    /// against resident ownership in this transaction. It is never clamped to
    /// transient capacity and planner estimates do not participate.
    #[doc(hidden)]
    pub fn try_admit_for_attempt(
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
    #[doc(hidden)]
    pub fn try_admit(
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
                live_reserved_bytes: 0,
                active_phase: None,
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
                live_reserved_bytes: Default::default(),
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
        let Ok(mut state) = self.inner.state.lock() else {
            eprintln!("[nexus-query-engine] warning: device ledger release observed a poisoned mutex");
            return;
        };
        let Some(active) = state.grants.remove(&grant_id) else {
            return;
        };
        if let Some(resident_bytes) = active.frozen_resident_bytes {
            state.frozen_resident_bytes = state.frozen_resident_bytes.saturating_sub(resident_bytes);
        }
        let transition = self.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::GrantReleased);
        drop(state);
        if let Some(observers) = self.release_observers() {
            for observer in observers.iter() {
                if catch_unwind(AssertUnwindSafe(|| observer(&transition))).is_err() {
                    eprintln!("[nexus-query-engine] warning: device ledger release observer panicked");
                }
            }
        }
    }

    /// Attempts to charge bytes retained by an evictable cache.
    #[doc(hidden)]
    pub fn try_reserve_evictable_bytes(&self, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        let mut state = self.lock_state("try_reserve_evictable_bytes")?;
        if state.evictable_bytes.saturating_add(state.pinned_bytes).saturating_add(bytes)
            > self.inner.profile.cache_cap_bytes
            || self.free_bytes_locked(&state) < bytes
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
    #[doc(hidden)]
    pub fn pin_cache_residency_charge(&self, bytes: u64) -> Result<()> {
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
    #[doc(hidden)]
    pub fn unpin_cache_residency_charge(&self, bytes: u64) -> Result<()> {
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
    #[doc(hidden)]
    pub fn release_cache_residency_charge_after_destroy(&self, bytes: u64) -> Result<()> {
        self.release_exact_evictable_cache_charge(bytes, "release_cache_residency_charge_after_destroy")
    }

    /// Returns unused conservative pre-reservation bytes before a cache entry
    /// has been committed. No physical allocation owns this excess yet.
    #[doc(hidden)]
    pub fn release_cache_residency_precharge_excess(&self, bytes: u64) -> Result<()> {
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

    /// Bytes currently charged to evictable caches.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn evictable_bytes(&self) -> Result<u64> {
        let state = self.lock_state("evictable_bytes")?;
        Ok(state.evictable_bytes)
    }

    fn free_bytes_locked(&self, state: &DeviceLedgerState) -> u64 {
        self.inner.profile.managed_capacity_bytes.saturating_sub(
            state
                .live_grants_bytes()
                .saturating_add(state.frozen_resident_bytes)
                .saturating_add(state.evictable_bytes)
                .saturating_add(state.pinned_bytes),
        )
    }
}

/// RAII immutable attempt grant. Dropping the final clone returns its budget
/// to the device ledger; the budget never changes while the grant is live.
#[derive(Clone)]
pub struct MemoryGrant {
    inner: Arc<MemoryGrantInner>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
}

/// Shared live-reservation retention held by GPU-resident outputs that
/// outlive their executor-side reservation tracker.
#[derive(Clone)]
#[doc(hidden)]
pub struct FrozenResidentLease {
    inner: Arc<FrozenResidentLeaseInner>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct MemoryGrantPhase {
    grant: MemoryGrant,
    generation: u64,
    required_bytes: u64,
}

struct FrozenResidentLeaseInner {
    grant: MemoryGrant,
    bytes: u64,
}

struct MemoryGrantInner {
    ledger: Arc<DeviceLedgerInner>,
    identity: GrantIdentity,
    admission_request_id: AdmissionRequestId,
    grant_bytes: u64,
    live_reserved_bytes: std::sync::atomic::AtomicU64,
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
            eprintln!("[nexus-query-engine] warning: failed to close admission after frozen resident cleanup: {error}");
        }
        ledger.release_grant(self.identity.generation);
    }
}

impl MemoryGrant {
    #[doc(hidden)]
    pub fn with_resource_epoch(&self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        Self { inner: Arc::clone(&self.inner), resource_epoch: Some(resource_epoch) }
    }

    fn check_resource_epoch(&self, operation: &'static str) -> Result<()> {
        self.resource_epoch.as_ref().map_or(Ok(()), |epoch| epoch.check_open(operation))
    }

    /// Unique ledger-local grant id.
    pub fn grant_id(&self) -> u64 {
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

    #[doc(hidden)]
    pub fn identity_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Immutable grant budget in bytes. This value never changes while the
    /// grant is live; v1 has no live grant growth, shrink, borrowing, or
    /// revocation.
    pub fn budget_bytes(&self) -> u64 {
        self.inner.grant_bytes
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn freeze_resident(&self, resident_bytes: u64) -> Result<()> {
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
    /// grant's live reservation is reconciled to that same measurement so the
    /// retained charge and the frozen charge are one accounting fact rather
    /// than two independently derived totals.
    pub(crate) fn freeze_resident_with_domain(
        &self,
        resident_bytes: u64,
        domain: Arc<cudf::QueryMemoryDomain>,
    ) -> Result<()> {
        self.reconcile_live_reservation_to_measurement(resident_bytes)?;
        self.prepare_frozen_resident_charge(resident_bytes, Some(domain))
    }

    /// Rebases every live reservation this grant holds onto the measured
    /// resident bytes.
    ///
    /// Executor-side trackers charge what each stage requested, which can
    /// differ from what the allocator actually retained (concat outputs are
    /// smaller than the sum of their chunks, for example). The measurement is
    /// the truth source, so the difference is reconciled instead of being
    /// treated as an ownership mismatch.
    fn reconcile_live_reservation_to_measurement(&self, measured_bytes: u64) -> Result<()> {
        if measured_bytes > self.inner.grant_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "nexus_query_engine.runtime.gpu.memory_grant.reconcile_live_reservation",
                "measured resident bytes exceed the immutable query cap",
            )
            .with_fact("measured_bytes", measured_bytes)
            .with_fact("query_cap_bytes", self.inner.grant_bytes));
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("reconcile_live_reservation_to_measurement")?;
        let Some(active) = state.grants.get_mut(&self.inner.identity.generation) else {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "nexus_query_engine.runtime.gpu.memory_grant.reconcile_live_reservation",
                "query grant is missing from its device ledger",
            )
            .with_fact("grant_id", self.grant_id()));
        };
        if active.live_reserved_bytes == measured_bytes {
            return Ok(());
        }
        active.live_reserved_bytes = measured_bytes;
        self.inner.live_reserved_bytes.store(measured_bytes, std::sync::atomic::Ordering::Release);
        ledger.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::LiveReservationCharged);
        Ok(())
    }

    fn prepare_frozen_resident_charge(
        &self,
        resident_bytes: u64,
        domain: Option<Arc<cudf::QueryMemoryDomain>>,
    ) -> Result<()> {
        if resident_bytes == 0 || resident_bytes > self.inner.grant_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "nexus_query_engine.runtime.gpu.memory_grant.freeze_resident",
                "frozen resident charge must be positive and fit the query cap",
            )
            .with_fact("resident_bytes", resident_bytes)
            .with_fact("query_cap_bytes", self.inner.grant_bytes));
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("freeze_resident")?;
        let active = state.grants.get(&self.inner.identity.generation).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "nexus_query_engine.runtime.gpu.memory_grant.freeze_resident",
                "query grant is missing from its device ledger",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        let existing = active.frozen_resident_bytes.or(active.pending_frozen_resident_bytes);
        if let Some(existing_bytes) = existing {
            let frozen_domain = self.inner.frozen_domain.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InternalInvariantViolation,
                    "nexus_query_engine.runtime.gpu.memory_grant.freeze_resident",
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
                    "nexus_query_engine.runtime.gpu.memory_grant.freeze_resident",
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
                    "nexus_query_engine.runtime.gpu.memory_grant.freeze_resident",
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
                "nexus_query_engine.runtime.gpu.memory_grant.validate_frozen_resident_publish",
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
                "nexus_query_engine.runtime.gpu.memory_grant.validate_frozen_resident_publish",
                "frozen resident transition was not prepared",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        state.frozen_resident_bytes.checked_add(resident_bytes).map(|_| ()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "nexus_query_engine.runtime.gpu.memory_grant.validate_frozen_resident_publish",
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
                "nexus_query_engine.runtime.gpu.memory_grant.publish_frozen_resident",
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
                "nexus_query_engine.runtime.gpu.memory_grant.publish_frozen_resident",
                "frozen resident transition was not prepared",
            )
            .with_fact("grant_id", self.grant_id())
        })?;
        state.frozen_resident_bytes = state.frozen_resident_bytes.checked_add(resident_bytes).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "nexus_query_engine.runtime.gpu.memory_grant.publish_frozen_resident",
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

    pub fn is_frozen_resident(&self) -> bool {
        self.inner.frozen.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Device ordinal that owns this grant.
    pub fn device_ordinal(&self) -> usize {
        self.inner.ledger.profile.device_ordinal
    }

    /// Device-resident bytes currently protected by all reservation trackers
    /// sharing this grant.
    #[doc(hidden)]
    pub fn live_reserved_bytes(&self) -> u64 {
        self.inner.live_reserved_bytes.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Remaining bytes this grant can admit across all active reservation
    /// trackers sharing it.
    #[doc(hidden)]
    pub fn remaining_live_reservation_bytes(&self) -> Result<u64> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.remaining_live_reservation_bytes")?;
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let state = ledger.lock_state("remaining_live_reservation_bytes")?;
        let Some(active) = state.grants.get(&self.inner.identity.generation) else {
            return Ok(0);
        };
        let active_limit = active.active_phase.map_or(active.grant_bytes, |phase| phase.required_bytes);
        Ok(active_limit.saturating_sub(active.live_reserved_bytes))
    }

    #[doc(hidden)]
    pub fn activate_phase(&self, required_bytes: u64) -> Result<MemoryGrantPhase> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.activate_phase")?;
        if required_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.gpu.memory_grant.activate_phase",
                "an attempt phase requires a positive immutable byte bound",
            )
            .with_fact("reason", "zero_attempt_phase_bound"));
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("activate_phase")?;
        let phase_generation = state.next_phase_generation;
        let next_phase_generation = phase_generation.checked_add(1).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.activate_phase",
                "attempt phase generation exhausted",
            )
            .with_fact("reason", "attempt_phase_generation_exhausted")
        })?;
        let active = state.grants.get_mut(&self.inner.identity.generation).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::MemoryReservationFailed,
                "nexus_query_engine.runtime.gpu.memory_grant.activate_phase",
                "the immutable attempt grant is no longer active",
            )
            .with_fact("reason", "attempt_phase_grant_inactive")
        })?;
        if active.frozen_resident_bytes.is_some() {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "nexus_query_engine.runtime.gpu.memory_grant.activate_phase",
                "a frozen resident domain cannot activate execution work",
            )
            .with_fact("reason", "frozen_resident_domain_is_not_executable"));
        }
        if active.active_phase.is_some() {
            return Err(QueryError::new(
                QueryErrorCode::MemoryReservationFailed,
                "nexus_query_engine.runtime.gpu.memory_grant.activate_phase",
                "the immutable attempt grant already has an active phase",
            )
            .with_fact("reason", "attempt_phase_already_active"));
        }
        if required_bytes > active.grant_bytes || active.live_reserved_bytes > required_bytes {
            return Err(QueryError::new(
                QueryErrorCode::MemoryReservationFailed,
                "nexus_query_engine.runtime.gpu.memory_grant.activate_phase",
                "the next attempt phase does not fit the immutable grant and retained input",
            )
            .with_fact("reason", "attempt_phase_exceeds_immutable_grant")
            .with_fact("required_phase_bytes", required_bytes)
            .with_fact("grant_bytes", active.grant_bytes)
            .with_fact("retained_live_bytes", active.live_reserved_bytes));
        }
        active.active_phase = Some(ActiveGrantPhase { generation: phase_generation, required_bytes });
        state.next_phase_generation = next_phase_generation;
        ledger.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::AttemptPhaseActivated);
        Ok(MemoryGrantPhase { grant: self.clone(), generation: phase_generation, required_bytes })
    }

    /// Attempts to charge new device-resident bytes to this grant.
    #[doc(hidden)]
    pub fn try_reserve_live_bytes(&self, bytes: u64) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.try_reserve_live_bytes")?;
        if bytes == 0 {
            return Ok(true);
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("try_reserve_live_bytes")?;
        let Some(active) = state.grants.get_mut(&self.inner.identity.generation) else {
            return Ok(false);
        };
        if active.frozen_resident_bytes.is_some() {
            return Ok(false);
        }
        let Some(next_live_reserved_bytes) = active.live_reserved_bytes.checked_add(bytes) else {
            return Ok(false);
        };
        active.live_reserved_bytes = next_live_reserved_bytes;
        self.inner.live_reserved_bytes.store(next_live_reserved_bytes, std::sync::atomic::Ordering::Release);
        ledger.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::LiveReservationCharged);
        Ok(true)
    }

    /// Releases live bytes previously admitted by
    /// [`Self::try_reserve_live_bytes`].
    #[doc(hidden)]
    pub fn release_live_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        if ledger.in_transition_observer() {
            ledger.panic_on_transition_observer_reentrancy("release_live_bytes");
        }
        let mut state = match ledger.inner.state.lock() {
            Ok(state) => state,
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to release memory grant live reservation bytes");
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
        let next_live_reserved_bytes = match active.live_reserved_bytes.checked_sub(bytes) {
            Some(next) => next,
            // A frozen grant's live reservation was rebased onto the
            // allocator's measurement, so a lease created from a larger
            // executor-side charge can release more than the rebased total.
            // The measurement stays authoritative and the total floors at
            // zero; only a non-frozen grant can under-run its own accounting.
            None if active.frozen_resident_bytes.is_some() || active.pending_frozen_resident_bytes.is_some() => 0,
            None => {
                eprintln!(
                    "[nexus-query-engine] warning: exact memory grant live-reservation release exceeded its active charge"
                );
                return;
            },
        };
        active.live_reserved_bytes = next_live_reserved_bytes;
        self.inner.live_reserved_bytes.store(next_live_reserved_bytes, std::sync::atomic::Ordering::Release);
        ledger.bump_sequence_and_notify_locked(state, LedgerTransitionKind::LiveReservationReleased);
    }

    /// Charges and retains newly escaping grant-scoped live bytes.
    ///
    /// Clones share one retention and the final drop releases it. Unlike an
    /// ownership transfer, this is a new charge and therefore must fit the
    /// immutable grant before the retention is created.
    #[doc(hidden)]
    pub fn retain_live_reservation(&self, bytes: u64) -> Result<Option<FrozenResidentLease>> {
        if bytes == 0 {
            return Ok(None);
        }
        if !self.try_reserve_live_bytes(bytes)? {
            return Err(QueryError::new(
                QueryErrorCode::MemoryReservationFailed,
                "nexus_query_engine.runtime.gpu.memory_grant.retain_live_reservation",
                "immutable memory grant cannot cover retained resident GPU bytes",
            )
            .with_fact("requested_bytes", bytes)
            .with_fact("budget_bytes", self.budget_bytes())
            .with_fact("live_reserved_bytes", self.live_reserved_bytes())
            .with_fact("reason", "retained_reservation_exceeds_immutable_grant"));
        }
        Ok(Some(self.transfer_live_reservation(bytes).expect("positive retained reservation has a grant charge")))
    }

    /// Transfers an already charged reservation into a result lifetime
    /// without briefly adding the bytes a second time.
    ///
    /// The caller must suppress the source reservation's eventual ledger
    /// release, because the returned retention owns that exact existing
    /// charge. This method intentionally does not mutate the ledger: the
    /// source charge and the result ownership are one atomic accounting fact.
    #[doc(hidden)]
    pub fn transfer_live_reservation(&self, bytes: u64) -> Option<FrozenResidentLease> {
        (bytes > 0).then(|| FrozenResidentLease {
            inner: Arc::new(FrozenResidentLeaseInner {
                grant: MemoryGrant { inner: Arc::clone(&self.inner), resource_epoch: None },
                bytes,
            }),
        })
    }
}

impl FrozenResidentLease {
    pub(crate) fn is_frozen_resident(&self) -> bool {
        self.inner.grant.is_frozen_resident()
    }

    /// Bytes retained by this handle in the owning memory grant.
    #[doc(hidden)]
    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Whether this retention consumes capacity from `grant`.
    #[doc(hidden)]
    pub fn belongs_to(&self, grant: &MemoryGrant) -> bool {
        self.inner.grant.identity_key() == grant.identity_key()
    }

    #[doc(hidden)]
    pub fn identity_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Retains a derived reservation for output bytes produced under the
    /// retained footprint.
    ///
    /// Immutable grants cannot grow to cover derived bytes, so the derived
    /// retention must fit inside the grant's remaining live-reservation
    /// capacity or the retention fails with a structured reservation error.
    #[doc(hidden)]
    pub fn retain_derived_live_reservation(&self, bytes: u64) -> Result<Option<Self>> {
        self.inner.grant.retain_live_reservation(bytes).map_err(|error| {
            if error.code() != QueryErrorCode::MemoryReservationFailed {
                return error;
            }
            QueryError::new(
                QueryErrorCode::MemoryReservationFailed,
                "nexus_query_engine.runtime.gpu.memory_grant.retain_derived_live_reservation",
                "immutable memory grant cannot cover derived resident GPU frame",
            )
            .with_fact("requested_bytes", bytes)
            .with_fact("budget_bytes", self.inner.grant.budget_bytes())
            .with_fact("live_reserved_bytes", self.inner.grant.live_reserved_bytes())
            .with_fact("reason", "derived_reservation_exceeds_immutable_grant")
        })
    }
}

impl Drop for FrozenResidentLeaseInner {
    fn drop(&mut self) {
        self.grant.release_live_bytes(self.bytes);
    }
}

impl MemoryGrantPhase {
    #[doc(hidden)]
    pub fn grant_identity(&self) -> GrantIdentity {
        self.grant.identity()
    }

    #[doc(hidden)]
    pub const fn required_bytes(&self) -> u64 {
        self.required_bytes
    }
}

impl Drop for MemoryGrantPhase {
    fn drop(&mut self) {
        let ledger = DeviceLedger { inner: Arc::clone(&self.grant.inner.ledger) };
        if ledger.in_transition_observer() {
            ledger.panic_on_transition_observer_reentrancy("release_phase");
        }
        let Ok(mut state) = ledger.inner.state.lock() else {
            eprintln!("[nexus-query-engine] warning: failed to release immutable attempt phase");
            return;
        };
        let Some(active) = state.grants.get_mut(&self.grant.identity().generation) else {
            return;
        };
        let Some(phase) = active.active_phase else {
            return;
        };
        if phase.generation != self.generation || phase.required_bytes != self.required_bytes {
            eprintln!("[nexus-query-engine] warning: immutable attempt phase identity mismatch during release");
            return;
        }
        active.active_phase = None;
        ledger.bump_sequence_and_notify_locked(&mut state, LedgerTransitionKind::AttemptPhaseReleased);
    }
}

impl fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("identity", &self.inner.identity)
            .field("admission_request_id", &self.inner.admission_request_id)
            .field("grant_bytes", &self.inner.grant_bytes)
            .field("live_reserved_bytes", &self.live_reserved_bytes())
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
