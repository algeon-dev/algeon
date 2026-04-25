//! Per-device soft memory grant ledger.

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    observability::AdmissionRequestId,
    runtime::{context::QueryServiceResourceEpoch, gpu::QueryQos},
};

type ReleaseObserver = Arc<dyn Fn() + Send + Sync + 'static>;
pub(crate) type BudgetChangeObserver = Arc<dyn Fn(BudgetChange) + Send + Sync + 'static>;
/// Callback used by caches to release evictable device bytes under grant pressure.
///
/// The returned byte count is advisory diagnostics only; the governor always
/// recomputes pressure from its own ledger after each observer callback.
pub type EvictableMemoryObserver = Arc<dyn Fn(EvictableMemoryPressure) -> Result<u64> + Send + Sync + 'static>;
const NO_SHRINK_REQUEST: u64 = 0;

/// Ordered memory-grant budget update delivered to observers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BudgetChange {
    pub(crate) budget_bytes: u64,
    pub(crate) sequence: u64,
}

/// Memory-pressure request delivered to evictable-cache observers.
///
/// Observers should evict up to `requested_evict_bytes` on this device and
/// return the number of bytes actually released from the governor's evictable
/// ledger. They are invoked without holding the governor state mutex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictableMemoryPressure {
    pub device_ordinal: usize,
    pub admission_request_id: AdmissionRequestId,
    pub requested_evict_bytes: u64,
    pub min_grant_bytes: u64,
    pub want_grant_bytes: u64,
    pub outstanding_bytes: u64,
    pub evictable_bytes: u64,
    pub free_bytes: u64,
}

/// Request for a variable per-query GPU memory grant.
#[derive(Debug, Clone, Copy)]
pub struct GrantRequest {
    /// Service-local admission request requesting memory.
    pub admission_request_id: AdmissionRequestId,
    /// Query QoS label. Scheduling behavior is deferred, but the label is part
    /// of the public request shape from Phase 4 onward.
    pub qos: QueryQos,
    /// Minimum grant that can make progress.
    pub min_bytes: u64,
    /// Preferred grant. The governor may grant less when pressure exists, but
    /// never below `min_bytes`.
    pub want_bytes: u64,
    /// Optional absolute wait deadline for blocking admission.
    pub deadline: Option<Instant>,
}

impl GrantRequest {
    /// Creates a normal-priority grant request.
    pub const fn new(admission_request_id: AdmissionRequestId, min_bytes: u64, want_bytes: u64) -> Self {
        Self {
            admission_request_id,
            qos: QueryQos::new(super::Priority::Normal),
            min_bytes,
            want_bytes,
            deadline: None,
        }
    }

    /// Sets QoS metadata.
    pub const fn with_qos(mut self, qos: QueryQos) -> Self {
        self.qos = qos;
        self
    }

    /// Sets an absolute blocking wait deadline.
    pub const fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// Soft memory governor for one CUDA device.
#[derive(Clone, Debug)]
pub struct MemoryGovernor {
    inner: Arc<MemoryGovernorInner>,
}

struct MemoryGovernorInner {
    device_ordinal: usize,
    capacity_bytes: u64,
    state: Mutex<GovernorState>,
    evictable_memory_observers: Mutex<Vec<EvictableMemoryObserver>>,
    available: Condvar,
}

impl std::fmt::Debug for MemoryGovernorInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGovernorInner")
            .field("device_ordinal", &self.device_ordinal)
            .field("capacity_bytes", &self.capacity_bytes)
            .field("state", &self.state)
            .field(
                "evictable_memory_observer_count",
                &self.evictable_memory_observers.lock().ok().map(|observers| observers.len()),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct GovernorState {
    next_grant_id: u64,
    outstanding_bytes: u64,
    evictable_bytes: u64,
    grants: BTreeMap<u64, ActiveGrant>,
}

#[derive(Debug, Clone, Copy)]
struct ActiveGrant {
    admission_request_id: AdmissionRequestId,
    qos: QueryQos,
    min_bytes: u64,
    budget_bytes: u64,
    live_reserved_bytes: u64,
    requested_shrink_target_bytes: u64,
}

/// RAII query memory grant. Dropping the final clone returns its budget.
#[derive(Clone)]
pub struct MemoryGrant {
    inner: Arc<MemoryGrantInner>,
    resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
}

/// Shared live-reservation retention held by GPU-resident outputs that outlive
/// their executor-side reservation tracker.
#[derive(Clone)]
pub(crate) struct MemoryGrantLiveReservation {
    inner: Arc<MemoryGrantLiveReservationInner>,
}

struct MemoryGrantLiveReservationInner {
    grant: MemoryGrant,
    bytes: u64,
}

struct MemoryGrantInner {
    governor: Arc<MemoryGovernorInner>,
    grant_id: u64,
    admission_request_id: AdmissionRequestId,
    qos: QueryQos,
    min_bytes: u64,
    budget_bytes: AtomicU64,
    hard_cap_bytes: AtomicU64,
    live_reserved_bytes: AtomicU64,
    budget_change_sequence: AtomicU64,
    release_observers: Mutex<Vec<ReleaseObserver>>,
    budget_decrease_observers: Mutex<Vec<ReleaseObserver>>,
    budget_change_observers: Mutex<Vec<BudgetChangeObserver>>,
}

/// Point-in-time governor report. This is intentionally separate from strict
/// and adaptive execution reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryGovernorReport {
    pub device_ordinal: usize,
    pub capacity_bytes: u64,
    pub outstanding_bytes: u64,
    pub evictable_bytes: u64,
    pub free_bytes: u64,
    pub available_bytes: u64,
    pub grants: Vec<MemoryGovernorGrantRow>,
}

/// One active grant row in [`MemoryGovernorReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryGovernorGrantRow {
    pub grant_id: u64,
    pub admission_request_id: AdmissionRequestId,
    pub priority: &'static str,
    pub budget_bytes: u64,
}

impl MemoryGovernor {
    /// Creates a per-device governor with fixed soft capacity.
    pub fn new(device_ordinal: usize, capacity_bytes: u64) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.gpu.memory_governor",
                "memory governor capacity must be greater than zero",
            )
            .with_fact("field", "capacity_bytes")
            .with_fact("reason", "zero"));
        }
        Ok(Self {
            inner: Arc::new(MemoryGovernorInner {
                device_ordinal,
                capacity_bytes,
                state: Mutex::new(GovernorState {
                    next_grant_id: 1,
                    outstanding_bytes: 0,
                    evictable_bytes: 0,
                    grants: BTreeMap::new(),
                }),
                evictable_memory_observers: Mutex::new(Vec::new()),
                available: Condvar::new(),
            }),
        })
    }

    /// CUDA device ordinal governed by this ledger.
    pub fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
    }

    /// Soft capacity in bytes.
    pub fn capacity_bytes(&self) -> u64 {
        self.inner.capacity_bytes
    }

    /// Returns bytes grantable to queries, including bytes that can be made
    /// free by synchronous evictable-cache observers.
    pub fn available_bytes(&self) -> Result<u64> {
        let state = self.lock_state("available_bytes")?;
        Ok(Self::grantable_bytes_locked(self.inner.capacity_bytes, &state))
    }

    /// Returns immediate unreserved bytes, excluding bytes held by evictable caches.
    pub fn free_bytes(&self) -> Result<u64> {
        let state = self.lock_state("free_bytes")?;
        Ok(Self::free_bytes_locked(self.inner.capacity_bytes, &state))
    }

    /// Returns bytes currently charged to evictable caches.
    pub fn evictable_bytes(&self) -> Result<u64> {
        let state = self.lock_state("evictable_bytes")?;
        Ok(state.evictable_bytes)
    }

    /// Registers a synchronous evictable-cache pressure observer.
    pub fn register_evictable_memory_observer(&self, observer: EvictableMemoryObserver) -> Result<()> {
        self.inner
            .evictable_memory_observers
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.runtime.gpu.memory_governor.register_evictable_memory_observer",
                    "memory governor evictable observer mutex was poisoned",
                )
            })?
            .push(observer);
        Ok(())
    }

    /// Attempts to charge bytes retained by an evictable cache.
    ///
    /// The charge succeeds only when immediate free capacity can cover the
    /// bytes. Existing evictable entries should evict and call
    /// [`Self::release_evictable_bytes`] before retrying a larger insertion.
    pub fn try_reserve_evictable_bytes(&self, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        let mut state = self.lock_state("try_reserve_evictable_bytes")?;
        if Self::free_bytes_locked(self.inner.capacity_bytes, &state) < bytes {
            return Ok(false);
        }
        state.evictable_bytes = state.evictable_bytes.saturating_add(bytes);
        Ok(true)
    }

    /// Returns whether an active grant from this governor has enough budget to
    /// cover bytes that are being retained as evictable cache memory.
    #[cfg(any(feature = "cugraph", test))]
    pub(crate) fn grant_can_cover_evictable_bytes(&self, grant: &MemoryGrant, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        if !Arc::ptr_eq(&self.inner, &grant.inner.governor) {
            return Ok(false);
        }
        let state = self.lock_state("grant_can_cover_evictable_bytes")?;
        let Some(active) = state.grants.get(&grant.inner.grant_id) else {
            return Ok(false);
        };
        Ok(bytes <= active.budget_bytes)
    }

    /// Charges evictable cache bytes that are already covered by an active
    /// grant from this governor.
    ///
    /// This intentionally does not reduce the grant's outstanding bytes. While
    /// the grant is alive the governor may be temporarily over-accounted, which
    /// prevents admitting additional work. Once the grant drops, the retained
    /// bytes remain charged as evictable cache memory.
    #[cfg(any(feature = "cugraph", test))]
    pub(crate) fn try_reserve_evictable_bytes_from_grant(&self, grant: &MemoryGrant, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        if !Arc::ptr_eq(&self.inner, &grant.inner.governor) {
            return Ok(false);
        }
        let mut state = self.lock_state("try_reserve_evictable_bytes_from_grant")?;
        let Some(active) = state.grants.get(&grant.inner.grant_id) else {
            return Ok(false);
        };
        if bytes > active.budget_bytes {
            return Ok(false);
        }
        state.evictable_bytes = state.evictable_bytes.saturating_add(bytes);
        Ok(true)
    }

    /// Releases bytes previously charged to the evictable-cache ledger.
    pub fn release_evictable_bytes(&self, bytes: u64) -> Result<u64> {
        if bytes == 0 {
            return Ok(0);
        }
        let mut state = self.lock_state("release_evictable_bytes")?;
        let released = bytes.min(state.evictable_bytes);
        if released == 0 {
            return Ok(0);
        }
        state.evictable_bytes = state.evictable_bytes.saturating_sub(released);
        self.inner.available.notify_all();
        Ok(released)
    }

    /// Attempts to grant immediately. Returns `Ok(None)` when less than
    /// `min_bytes` is available.
    pub fn try_request(&self, request: GrantRequest) -> Result<Option<MemoryGrant>> {
        self.validate_request(&request)?;
        let mut state = self.lock_state("try_request")?;
        if let Some(grant) = self.try_grant_locked(&mut state, request) {
            return Ok(Some(grant));
        }
        let Some(pressure) = self.evictable_pressure_locked(&state, request) else {
            return Ok(None);
        };
        drop(state);
        self.notify_evictable_memory_observers(pressure)?;
        let mut state = self.lock_state("try_request")?;
        Ok(self.try_grant_locked(&mut state, request))
    }

    /// Blocks until at least `min_bytes` is available or `deadline` expires.
    pub fn request(&self, request: GrantRequest) -> Result<MemoryGrant> {
        self.validate_request(&request)?;
        let mut state = self.lock_state("request")?;
        let mut eviction_attempted = false;
        loop {
            if let Some(grant) = self.try_grant_locked(&mut state, request) {
                return Ok(grant);
            }
            if !eviction_attempted && let Some(pressure) = self.evictable_pressure_locked(&state, request) {
                drop(state);
                self.notify_evictable_memory_observers(pressure)?;
                state = self.lock_state("request")?;
                eviction_attempted = true;
                continue;
            }
            if let Some(deadline) = request.deadline {
                let now = Instant::now();
                if now >= deadline {
                    return Err(Self::timeout_error(&request, self.inner.capacity_bytes, &state));
                }
                let wait = deadline.saturating_duration_since(now);
                let (next_state, timeout) = self.inner.available.wait_timeout(state, wait).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.runtime.gpu.memory_governor.request",
                        "memory governor mutex was poisoned while waiting",
                    )
                })?;
                state = next_state;
                eviction_attempted = false;
                if timeout.timed_out() && Self::free_bytes_locked(self.inner.capacity_bytes, &state) < request.min_bytes
                {
                    return Err(Self::timeout_error(&request, self.inner.capacity_bytes, &state));
                }
            } else {
                state = self.inner.available.wait(state).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.runtime.gpu.memory_governor.request",
                        "memory governor mutex was poisoned while waiting",
                    )
                })?;
                eviction_attempted = false;
            }
        }
    }

    /// Returns a report of active soft grants.
    pub fn report(&self) -> Result<MemoryGovernorReport> {
        let state = self.lock_state("report")?;
        let grants = state
            .grants
            .iter()
            .map(|(grant_id, grant)| MemoryGovernorGrantRow {
                grant_id: *grant_id,
                admission_request_id: grant.admission_request_id,
                priority: grant.qos.priority().as_str(),
                budget_bytes: grant.budget_bytes,
            })
            .collect();
        Ok(MemoryGovernorReport {
            device_ordinal: self.inner.device_ordinal,
            capacity_bytes: self.inner.capacity_bytes,
            outstanding_bytes: state.outstanding_bytes,
            evictable_bytes: state.evictable_bytes,
            free_bytes: Self::free_bytes_locked(self.inner.capacity_bytes, &state),
            available_bytes: Self::grantable_bytes_locked(self.inner.capacity_bytes, &state),
            grants,
        })
    }

    fn validate_request(&self, request: &GrantRequest) -> Result<()> {
        if request.min_bytes == 0 {
            return Err(invalid_request("min_bytes", "zero", request));
        }
        if request.want_bytes < request.min_bytes {
            return Err(invalid_request("want_bytes", "below_min_bytes", request));
        }
        if request.min_bytes > self.inner.capacity_bytes {
            return Err(invalid_request("min_bytes", "above_capacity_bytes", request)
                .with_fact("capacity_bytes", self.inner.capacity_bytes));
        }
        Ok(())
    }

    fn lock_state(&self, action: &'static str) -> Result<std::sync::MutexGuard<'_, GovernorState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_governor",
                "memory governor mutex was poisoned",
            )
            .with_fact("action", action)
        })
    }

    fn try_grant_locked(&self, state: &mut GovernorState, request: GrantRequest) -> Option<MemoryGrant> {
        let free = Self::free_bytes_locked(self.inner.capacity_bytes, state);
        if free < request.min_bytes {
            return None;
        }
        let budget_bytes = request.want_bytes.min(free);
        let grant_id = state.next_grant_id;
        state.next_grant_id = state.next_grant_id.saturating_add(1);
        state.outstanding_bytes = state.outstanding_bytes.saturating_add(budget_bytes);
        state.grants.insert(
            grant_id,
            ActiveGrant {
                admission_request_id: request.admission_request_id,
                qos: request.qos,
                min_bytes: request.min_bytes,
                budget_bytes,
                live_reserved_bytes: 0,
                requested_shrink_target_bytes: NO_SHRINK_REQUEST,
            },
        );
        Some(MemoryGrant {
            inner: Arc::new(MemoryGrantInner {
                governor: Arc::clone(&self.inner),
                grant_id,
                admission_request_id: request.admission_request_id,
                qos: request.qos,
                min_bytes: request.min_bytes,
                budget_bytes: AtomicU64::new(budget_bytes),
                hard_cap_bytes: AtomicU64::new(0),
                live_reserved_bytes: AtomicU64::new(0),
                budget_change_sequence: AtomicU64::new(0),
                release_observers: Mutex::new(Vec::new()),
                budget_decrease_observers: Mutex::new(Vec::new()),
                budget_change_observers: Mutex::new(Vec::new()),
            }),
            resource_epoch: None,
        })
    }

    fn evictable_pressure_locked(
        &self,
        state: &GovernorState,
        request: GrantRequest,
    ) -> Option<EvictableMemoryPressure> {
        let free_bytes = Self::free_bytes_locked(self.inner.capacity_bytes, state);
        if free_bytes >= request.min_bytes || state.evictable_bytes == 0 {
            return None;
        }
        let grantable_bytes = Self::grantable_bytes_locked(self.inner.capacity_bytes, state);
        if grantable_bytes < request.min_bytes {
            return None;
        }
        // Cache eviction is deliberately admission-floor driven: reclaim enough
        // for progress, but do not evict warm cache entries merely to satisfy a
        // preferred grant size.
        Some(EvictableMemoryPressure {
            device_ordinal: self.inner.device_ordinal,
            admission_request_id: request.admission_request_id,
            requested_evict_bytes: request.min_bytes.saturating_sub(free_bytes),
            min_grant_bytes: request.min_bytes,
            want_grant_bytes: request.want_bytes,
            outstanding_bytes: state.outstanding_bytes,
            evictable_bytes: state.evictable_bytes,
            free_bytes,
        })
    }

    fn notify_evictable_memory_observers(&self, pressure: EvictableMemoryPressure) -> Result<()> {
        let observers = self
            .inner
            .evictable_memory_observers
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.runtime.gpu.memory_governor.notify_evictable_memory_observers",
                    "memory governor evictable observer mutex was poisoned",
                )
            })?
            .clone();
        let target_free_bytes = pressure.free_bytes.saturating_add(pressure.requested_evict_bytes);
        for observer in observers {
            let Some(next_pressure) = self.evictable_pressure_for_target(pressure, target_free_bytes)? else {
                break;
            };
            if let Err(error) = observer(next_pressure) {
                eprintln!(
                    "[nexus-query-engine] warning: evictable memory observer failed under grant pressure: {error}"
                );
            }
        }
        Ok(())
    }

    fn evictable_pressure_for_target(
        &self,
        original: EvictableMemoryPressure,
        target_free_bytes: u64,
    ) -> Result<Option<EvictableMemoryPressure>> {
        let state = self.lock_state("evictable_pressure_for_target")?;
        let free_bytes = Self::free_bytes_locked(self.inner.capacity_bytes, &state);
        let requested_evict_bytes = target_free_bytes.saturating_sub(free_bytes).min(state.evictable_bytes);
        if requested_evict_bytes == 0 {
            return Ok(None);
        }
        Ok(Some(EvictableMemoryPressure {
            requested_evict_bytes,
            outstanding_bytes: state.outstanding_bytes,
            evictable_bytes: state.evictable_bytes,
            free_bytes,
            ..original
        }))
    }

    fn grantable_bytes_locked(capacity_bytes: u64, state: &GovernorState) -> u64 {
        capacity_bytes.saturating_sub(state.outstanding_bytes)
    }

    fn free_bytes_locked(capacity_bytes: u64, state: &GovernorState) -> u64 {
        capacity_bytes.saturating_sub(state.outstanding_bytes.saturating_add(state.evictable_bytes))
    }

    fn timeout_error(request: &GrantRequest, capacity_bytes: u64, state: &GovernorState) -> QueryError {
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.runtime.gpu.memory_governor.request",
            "memory governor grant request timed out",
        )
        .with_fact("reason", "memory_grant_wait_timeout")
        .with_fact("admission_request_id", request.admission_request_id)
        .with_fact("min_bytes", request.min_bytes)
        .with_fact("want_bytes", request.want_bytes)
        .with_fact("capacity_bytes", capacity_bytes)
        .with_fact("outstanding_bytes", state.outstanding_bytes)
        .with_fact("evictable_bytes", state.evictable_bytes)
        .with_fact("free_bytes", Self::free_bytes_locked(capacity_bytes, state))
    }
}

impl MemoryGrant {
    pub(crate) fn with_resource_epoch(&self, resource_epoch: Arc<QueryServiceResourceEpoch>) -> Self {
        Self { inner: Arc::clone(&self.inner), resource_epoch: Some(resource_epoch) }
    }

    fn check_resource_epoch(&self, operation: &'static str) -> Result<()> {
        self.resource_epoch.as_ref().map_or(Ok(()), |epoch| epoch.check_open(operation))
    }

    /// Unique governor-local grant id.
    pub fn grant_id(&self) -> u64 {
        self.inner.grant_id
    }

    /// Service-local admission request that owns this grant.
    pub fn admission_request_id(&self) -> AdmissionRequestId {
        self.inner.admission_request_id
    }

    pub(crate) fn identity_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// QoS metadata attached to the grant request.
    pub fn qos(&self) -> QueryQos {
        self.inner.qos
    }

    /// Minimum budget floor assigned when this grant was admitted.
    pub fn min_bytes(&self) -> u64 {
        self.inner.min_bytes
    }

    /// Current grant budget.
    pub fn budget_bytes(&self) -> u64 {
        self.inner.budget_bytes.load(Ordering::Acquire)
    }

    /// Optional immutable ceiling above which adaptive growth must not move
    /// this grant.
    pub(crate) fn hard_cap_bytes(&self) -> Option<u64> {
        let cap = self.inner.hard_cap_bytes.load(Ordering::Acquire);
        (cap != 0).then_some(cap)
    }

    /// Computes and installs a hard growth ceiling under the governor lock.
    ///
    /// `compute` must be pure arithmetic over the current budget; it runs while
    /// the governor state lock is held so the installed ceiling cannot be based
    /// on a stale budget racing adaptive growth.
    ///
    /// The cap is intentionally one-way: callers set it when binding a native
    /// allocator domain with an immutable driver ceiling. Repeated calls return
    /// the existing effective ceiling so a context-acquisition retry cannot
    /// wedge a grant after a partial lower-level setup failure.
    pub(crate) fn install_hard_cap_bytes_for_current_budget(
        &self,
        compute: impl FnOnce(u64) -> Result<u64>,
    ) -> Result<(u64, u64)> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.install_hard_cap")?;
        let state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.install_hard_cap",
                "memory governor mutex was poisoned",
            )
        })?;
        let current_budget = state
            .grants
            .get(&self.inner.grant_id)
            .map(|active| active.budget_bytes)
            .unwrap_or_else(|| self.inner.budget_bytes.load(Ordering::Acquire));
        let existing = self.inner.hard_cap_bytes.load(Ordering::Acquire);
        if existing != 0 {
            return Ok((current_budget, existing));
        }

        let hard_cap_bytes = compute(current_budget)?;
        if hard_cap_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.gpu.memory_grant.install_hard_cap",
                "memory grant hard cap bytes must be greater than zero",
            )
            .with_fact("field", "hard_cap_bytes")
            .with_fact("reason", "zero"));
        }
        if hard_cap_bytes < current_budget {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.runtime.gpu.memory_grant.install_hard_cap",
                "memory grant hard cap cannot be below the current budget",
            )
            .with_fact("grant_id", self.grant_id())
            .with_fact("admission_request_id", self.admission_request_id())
            .with_fact("current_budget_bytes", current_budget)
            .with_fact("requested_hard_cap_bytes", hard_cap_bytes)
            .with_fact("reason", "hard_cap_below_current_budget"));
        }
        self.inner.hard_cap_bytes.store(hard_cap_bytes, Ordering::Release);
        Ok((current_budget, hard_cap_bytes))
    }

    /// Device-resident bytes currently protected by all reservation trackers
    /// sharing this grant.
    pub(crate) fn live_reserved_bytes(&self) -> u64 {
        self.inner.live_reserved_bytes.load(Ordering::Acquire)
    }

    /// Remaining bytes this grant can admit across all active reservation
    /// trackers sharing it.
    pub(crate) fn remaining_live_reservation_bytes(&self) -> Result<u64> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.remaining_live_reservation_bytes")?;
        let state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.remaining_live_reservation_bytes",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(active) = state.grants.get(&self.inner.grant_id) else {
            return Ok(0);
        };
        Ok(active.budget_bytes.saturating_sub(active.live_reserved_bytes))
    }

    /// Attempts to charge new device-resident bytes to this grant.
    ///
    /// This check is serialized with cooperative shrink floor reads so sibling
    /// trackers cannot jointly reserve more live bytes than the shared budget.
    pub(crate) fn try_reserve_live_bytes(&self, bytes: u64) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.try_reserve_live_bytes")?;
        if bytes == 0 {
            return Ok(true);
        }
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.try_reserve_live_bytes",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        let Some(next_live_reserved_bytes) = active.live_reserved_bytes.checked_add(bytes) else {
            return Ok(false);
        };
        if next_live_reserved_bytes > active.budget_bytes {
            return Ok(false);
        }
        active.live_reserved_bytes = next_live_reserved_bytes;
        self.inner.live_reserved_bytes.store(next_live_reserved_bytes, Ordering::Release);
        Ok(true)
    }

    /// Releases live bytes previously admitted by
    /// [`Self::try_reserve_live_bytes`].
    pub(crate) fn release_live_bytes(&self, bytes: u64) {
        self.record_live_reservation_delta(-i128::from(bytes));
    }

    /// Updates the grant-scoped live reservation footprint without an
    /// admission check.
    ///
    /// Positive deltas are used only to transfer already-admitted resident
    /// bytes into output-handle retention. New reservations must use
    /// [`Self::try_reserve_live_bytes`].
    pub(crate) fn record_live_reservation_delta(&self, delta_bytes: i128) {
        let mut state = match self.inner.governor.state.lock() {
            Ok(state) => state,
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to update memory grant live reservation bytes");
                return;
            },
        };
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            if delta_bytes < 0 {
                let released_bytes = (-delta_bytes) as u64;
                let _ = self.inner.live_reserved_bytes.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    Some(current.saturating_sub(released_bytes))
                });
            }
            return;
        };
        if delta_bytes >= 0 {
            active.live_reserved_bytes = active.live_reserved_bytes.saturating_add(delta_bytes as u64);
        } else {
            let released_bytes = (-delta_bytes) as u64;
            debug_assert!(
                active.live_reserved_bytes >= released_bytes,
                "memory grant live-reservation release exceeded tracked live bytes"
            );
            active.live_reserved_bytes = active.live_reserved_bytes.saturating_sub(released_bytes);
        }
        self.inner.live_reserved_bytes.store(active.live_reserved_bytes, Ordering::Release);
    }

    /// Retains grant-scoped live bytes after an executor-side reservation has
    /// been released. Clones share one retention and the final drop releases it.
    pub(crate) fn retain_live_reservation(&self, bytes: u64) -> Option<MemoryGrantLiveReservation> {
        if bytes == 0 {
            return None;
        }
        self.record_live_reservation_delta(i128::from(bytes));
        Some(MemoryGrantLiveReservation {
            inner: Arc::new(MemoryGrantLiveReservationInner { grant: self.clone(), bytes }),
        })
    }

    /// Records a cooperative shrink target without refunding governor capacity.
    ///
    /// This is the request half of chunk-boundary reclamation. The executing
    /// query must later acknowledge the request at a safe boundary via
    /// [`Self::try_apply_requested_shrink`], after resident work above the
    /// target has been released or replanned.
    pub fn request_shrink_to(&self, target_bytes: u64) -> Result<Option<u64>> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.request_shrink_to")?;
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.request_shrink_to",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(None);
        };
        let target_bytes = target_bytes.max(active.min_bytes);
        if target_bytes >= active.budget_bytes {
            return Ok(None);
        }

        let current = active.requested_shrink_target_bytes;
        let next = if current == NO_SHRINK_REQUEST { target_bytes } else { target_bytes.min(current) };
        active.requested_shrink_target_bytes = next;
        Ok(Some(next))
    }

    /// Returns the currently requested cooperative shrink target, if any.
    pub fn requested_shrink_target_bytes(&self) -> Result<Option<u64>> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.requested_shrink_target_bytes")?;
        let state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.requested_shrink_target_bytes",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(active) = state.grants.get(&self.inner.grant_id) else {
            return Ok(None);
        };
        let target_bytes = active.requested_shrink_target_bytes;
        if target_bytes == NO_SHRINK_REQUEST || target_bytes >= active.budget_bytes {
            Ok(None)
        } else {
            Ok(Some(target_bytes.max(active.min_bytes)))
        }
    }

    /// Clears a pending cooperative shrink request without changing the
    /// current grant budget.
    pub fn clear_requested_shrink(&self) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.clear_requested_shrink")?;
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.clear_requested_shrink",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        let had_request = active.requested_shrink_target_bytes != NO_SHRINK_REQUEST;
        active.requested_shrink_target_bytes = NO_SHRINK_REQUEST;
        Ok(had_request)
    }

    /// Attempts to increase this grant by `additional_bytes`.
    ///
    /// This is intended for adaptive execution. Strict bounded execution should
    /// treat the initial grant as a gate-first budget.
    pub fn try_grow(&self, additional_bytes: u64) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.try_grow")?;
        if additional_bytes == 0 {
            return Ok(true);
        }
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.try_grow",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(current_budget) = state.grants.get(&self.inner.grant_id).map(|active| active.budget_bytes) else {
            return Ok(false);
        };
        let additional_bytes = self.clamp_additional_growth(current_budget, additional_bytes);
        if additional_bytes == 0 {
            return Ok(false);
        }
        let mut available = MemoryGovernor::free_bytes_locked(self.inner.governor.capacity_bytes, &state);
        if available < additional_bytes
            && let Some(pressure) = self.evictable_pressure_for_growth_locked(&state, additional_bytes)
        {
            let governor = MemoryGovernor { inner: Arc::clone(&self.inner.governor) };
            drop(state);
            governor.notify_evictable_memory_observers(pressure)?;
            state = self.inner.governor.state.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.runtime.gpu.memory_grant.try_grow",
                    "memory governor mutex was poisoned after evictable pressure",
                )
            })?;
            available = MemoryGovernor::free_bytes_locked(self.inner.governor.capacity_bytes, &state);
        }
        let Some(current_budget) = state.grants.get(&self.inner.grant_id).map(|active| active.budget_bytes) else {
            return Ok(false);
        };
        let additional_bytes = self.clamp_additional_growth(current_budget, additional_bytes);
        if additional_bytes == 0 {
            return Ok(false);
        }
        if available < additional_bytes {
            return Ok(false);
        }
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        active.budget_bytes = active.budget_bytes.saturating_add(additional_bytes);
        let budget_bytes = active.budget_bytes;
        state.outstanding_bytes = state.outstanding_bytes.saturating_add(additional_bytes);
        self.inner.budget_bytes.store(budget_bytes, Ordering::Release);
        let change = self.next_budget_change(budget_bytes);
        drop(state);
        self.notify_budget_change_observers(change);
        Ok(true)
    }

    /// Attempts to grow this grant to at least `target_bytes`.
    pub fn try_grow_to(&self, target_bytes: u64) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.try_grow_to")?;
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.try_grow_to",
                "memory governor mutex was poisoned",
            )
        })?;
        if self.hard_cap_bytes().is_some_and(|cap| target_bytes > cap) {
            return Ok(false);
        }
        let Some(current_budget) = state.grants.get(&self.inner.grant_id).map(|active| active.budget_bytes) else {
            return Ok(false);
        };
        if target_bytes <= current_budget {
            return Ok(true);
        }
        let additional_bytes = target_bytes - current_budget;
        let mut available = MemoryGovernor::free_bytes_locked(self.inner.governor.capacity_bytes, &state);
        if available < additional_bytes
            && let Some(pressure) = self.evictable_pressure_for_growth_locked(&state, additional_bytes)
        {
            let governor = MemoryGovernor { inner: Arc::clone(&self.inner.governor) };
            drop(state);
            governor.notify_evictable_memory_observers(pressure)?;
            state = self.inner.governor.state.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.runtime.gpu.memory_grant.try_grow_to",
                    "memory governor mutex was poisoned after evictable pressure",
                )
            })?;
            available = MemoryGovernor::free_bytes_locked(self.inner.governor.capacity_bytes, &state);
        }
        if self.hard_cap_bytes().is_some_and(|cap| target_bytes > cap) {
            return Ok(false);
        }
        let Some(current_budget) = state.grants.get(&self.inner.grant_id).map(|active| active.budget_bytes) else {
            return Ok(false);
        };
        if target_bytes <= current_budget {
            return Ok(true);
        }
        let additional_bytes = target_bytes - current_budget;
        if available < additional_bytes {
            return Ok(false);
        }
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        active.budget_bytes = target_bytes;
        state.outstanding_bytes = state.outstanding_bytes.saturating_add(additional_bytes);
        self.inner.budget_bytes.store(target_bytes, Ordering::Release);
        let change = self.next_budget_change(target_bytes);
        drop(state);
        self.notify_budget_change_observers(change);
        Ok(true)
    }

    fn clamp_additional_growth(&self, current_budget: u64, additional_bytes: u64) -> u64 {
        match self.hard_cap_bytes() {
            Some(cap) => cap.saturating_sub(current_budget).min(additional_bytes),
            None => additional_bytes,
        }
    }

    fn evictable_pressure_for_growth_locked(
        &self,
        state: &GovernorState,
        additional_bytes: u64,
    ) -> Option<EvictableMemoryPressure> {
        let free_bytes = MemoryGovernor::free_bytes_locked(self.inner.governor.capacity_bytes, state);
        if free_bytes >= additional_bytes || state.evictable_bytes == 0 {
            return None;
        }
        let grantable_bytes = MemoryGovernor::grantable_bytes_locked(self.inner.governor.capacity_bytes, state);
        if grantable_bytes < additional_bytes {
            return None;
        }
        Some(EvictableMemoryPressure {
            device_ordinal: self.inner.governor.device_ordinal,
            admission_request_id: self.inner.admission_request_id,
            requested_evict_bytes: additional_bytes.saturating_sub(free_bytes),
            min_grant_bytes: additional_bytes,
            want_grant_bytes: additional_bytes,
            outstanding_bytes: state.outstanding_bytes,
            evictable_bytes: state.evictable_bytes,
            free_bytes,
        })
    }

    /// Attempts to reduce this grant to `target_bytes`, clamped to the
    /// admission minimum.
    ///
    /// This only refunds soft governor capacity. Callers must use it after the
    /// executor has reached a cooperative boundary and released/replanned any
    /// resident work above the target.
    pub fn try_shrink_to(&self, target_bytes: u64) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.try_shrink_to")?;
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.try_shrink_to",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        let current_budget = active.budget_bytes;
        let target_bytes = target_bytes.max(active.min_bytes).max(active.live_reserved_bytes);
        if target_bytes >= current_budget {
            return Ok(true);
        }
        let refunded_bytes = current_budget - target_bytes;
        active.budget_bytes = target_bytes;
        if active.requested_shrink_target_bytes != NO_SHRINK_REQUEST
            && target_bytes <= active.requested_shrink_target_bytes
        {
            active.requested_shrink_target_bytes = NO_SHRINK_REQUEST;
        }
        state.outstanding_bytes = state.outstanding_bytes.saturating_sub(refunded_bytes);
        self.inner.budget_bytes.store(target_bytes, Ordering::Release);
        let change = self.next_budget_change(target_bytes);
        self.inner.governor.available.notify_all();
        drop(state);
        self.notify_budget_change_observers(change);
        self.notify_budget_decrease_observers();
        Ok(true)
    }

    /// Applies a pending cooperative shrink request as far as
    /// `live_reserved_bytes` permits.
    ///
    /// The method never refunds below the live reservation footprint supplied
    /// by the executor, so it is safe to call at chunk boundaries before the
    /// requested target has been fully reached. It returns `true` when no
    /// shrink request remains pending.
    pub fn try_apply_requested_shrink(&self, live_reserved_bytes: u64) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.try_apply_requested_shrink")?;
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.try_apply_requested_shrink",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        let requested_target = active.requested_shrink_target_bytes;
        if requested_target == NO_SHRINK_REQUEST {
            return Ok(true);
        }
        if requested_target >= active.budget_bytes {
            active.requested_shrink_target_bytes = NO_SHRINK_REQUEST;
            return Ok(true);
        }
        let safe_target =
            requested_target.max(live_reserved_bytes).max(active.live_reserved_bytes).max(active.min_bytes);
        if safe_target >= active.budget_bytes {
            return Ok(false);
        }

        let current_budget = active.budget_bytes;
        let refunded_bytes = current_budget - safe_target;
        active.budget_bytes = safe_target;
        if safe_target <= requested_target {
            active.requested_shrink_target_bytes = NO_SHRINK_REQUEST;
        }
        let request_satisfied = active.requested_shrink_target_bytes == NO_SHRINK_REQUEST;
        state.outstanding_bytes = state.outstanding_bytes.saturating_sub(refunded_bytes);
        self.inner.budget_bytes.store(safe_target, Ordering::Release);
        let change = self.next_budget_change(safe_target);
        self.inner.governor.available.notify_all();
        drop(state);
        self.notify_budget_change_observers(change);
        self.notify_budget_decrease_observers();
        Ok(request_satisfied)
    }

    /// Adds a callback invoked when the final clone releases this grant.
    pub(crate) fn notify_on_release(&self, observer: ReleaseObserver) -> Result<()> {
        self.inner
            .release_observers
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.runtime.gpu.memory_grant.notify_on_release",
                    "memory grant release observer mutex was poisoned",
                )
            })?
            .push(observer);
        Ok(())
    }

    /// Adds a callback invoked after this grant's budget decreases.
    pub(crate) fn notify_on_budget_decrease(&self, observer: ReleaseObserver) -> Result<()> {
        self.inner
            .budget_decrease_observers
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.runtime.gpu.memory_grant.notify_on_budget_decrease",
                    "memory grant budget-decrease observer mutex was poisoned",
                )
            })?
            .push(observer);
        Ok(())
    }

    /// Adds a callback invoked after this grant's budget changes and returns
    /// the current sequenced budget state for observer initialization.
    pub(crate) fn notify_on_budget_change(&self, observer: BudgetChangeObserver) -> Result<BudgetChange> {
        let mut observers = self.inner.budget_change_observers.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.notify_on_budget_change",
                "memory grant budget-change observer mutex was poisoned",
            )
        })?;
        observers.push(observer);
        self.current_budget_change()
    }

    fn current_budget_change(&self) -> Result<BudgetChange> {
        let state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.current_budget_change",
                "memory governor mutex was poisoned",
            )
        })?;
        let budget_bytes = state
            .grants
            .get(&self.inner.grant_id)
            .map(|active| active.budget_bytes)
            .unwrap_or_else(|| self.inner.budget_bytes.load(Ordering::Acquire));
        let sequence = self.inner.budget_change_sequence.load(Ordering::Acquire);
        Ok(BudgetChange { budget_bytes, sequence })
    }

    fn next_budget_change(&self, budget_bytes: u64) -> BudgetChange {
        let sequence = self.inner.budget_change_sequence.fetch_add(1, Ordering::AcqRel) + 1;
        BudgetChange { budget_bytes, sequence }
    }

    fn notify_budget_decrease_observers(&self) {
        let observers = match self.inner.budget_decrease_observers.lock() {
            Ok(observers) => observers.clone(),
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to notify memory grant budget-decrease observers");
                Vec::new()
            },
        };
        for observer in observers {
            observer();
        }
    }

    fn notify_budget_change_observers(&self, change: BudgetChange) {
        let observers = match self.inner.budget_change_observers.lock() {
            Ok(observers) => observers.clone(),
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to notify memory grant budget-change observers");
                Vec::new()
            },
        };
        for observer in observers {
            observer(change);
        }
    }
}

impl Drop for MemoryGrantLiveReservationInner {
    fn drop(&mut self) {
        self.grant.record_live_reservation_delta(-i128::from(self.bytes));
    }
}

impl MemoryGrantLiveReservation {
    pub(crate) fn retain_derived_live_reservation(&self, bytes: u64) -> Result<Option<Self>> {
        let retained = self.inner.grant.retain_live_reservation(bytes);
        if retained.is_none() {
            return Ok(None);
        }
        let target_budget = self.inner.grant.live_reserved_bytes().max(self.inner.grant.min_bytes());
        if self.inner.grant.try_grow_to(target_budget)? {
            return Ok(retained);
        }
        drop(retained);
        Err(QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.runtime.gpu.memory_grant.retain_derived_live_reservation",
            "memory grant cannot grow to cover derived resident GPU frame",
        )
        .with_fact("requested_bytes", bytes)
        .with_fact("target_budget_bytes", target_budget)
        .with_fact("budget_bytes", self.inner.grant.budget_bytes())
        .with_fact("live_reserved_bytes", self.inner.grant.live_reserved_bytes()))
    }
}

impl std::fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("grant_id", &self.grant_id())
            .field("admission_request_id", &self.admission_request_id())
            .field("qos", &self.qos())
            .field("min_bytes", &self.min_bytes())
            .field("budget_bytes", &self.budget_bytes())
            .field("live_reserved_bytes", &self.live_reserved_bytes())
            .finish()
    }
}

impl Drop for MemoryGrantInner {
    fn drop(&mut self) {
        match self.governor.state.lock() {
            Ok(mut state) => {
                let released = state
                    .grants
                    .remove(&self.grant_id)
                    .map(|grant| grant.budget_bytes)
                    .unwrap_or_else(|| self.budget_bytes.load(Ordering::Acquire));
                state.outstanding_bytes = state.outstanding_bytes.saturating_sub(released);
                self.governor.available.notify_all();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to release memory governor grant");
            },
        }
        let observers = match self.release_observers.lock() {
            Ok(observers) => observers.clone(),
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to notify memory governor grant release observers");
                Vec::new()
            },
        };
        for observer in observers {
            observer();
        }
    }
}

fn invalid_request(field: &'static str, reason: &'static str, request: &GrantRequest) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.runtime.gpu.memory_governor.request",
        "invalid memory governor grant request",
    )
    .with_fact("field", field)
    .with_fact("reason", reason)
    .with_fact("admission_request_id", request.admission_request_id)
    .with_fact("min_bytes", request.min_bytes)
    .with_fact("want_bytes", request.want_bytes)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::mpsc, thread, time::Duration};

    use super::*;

    #[test]
    fn grants_never_exceed_capacity_and_release_on_drop() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let first = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 70)).unwrap().unwrap();
        let second = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 20, 70)).unwrap().unwrap();

        assert_eq!(first.budget_bytes(), 70);
        assert_eq!(second.budget_bytes(), 30);
        assert_eq!(governor.available_bytes().unwrap(), 0);
        assert!(governor.try_request(GrantRequest::new(AdmissionRequestId::new(3), 1, 1)).unwrap().is_none());

        drop(first);
        assert_eq!(governor.available_bytes().unwrap(), 70);
        drop(second);
        assert_eq!(governor.available_bytes().unwrap(), 100);
    }

    #[test]
    fn small_query_receives_want_not_capacity_partition() {
        let governor = MemoryGovernor::new(0, 96).unwrap();
        let small = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 8, 12)).unwrap().unwrap();

        assert_eq!(small.budget_bytes(), 12);
        assert_eq!(governor.available_bytes().unwrap(), 84);
    }

    #[test]
    fn try_request_returns_none_under_pressure() {
        let governor = MemoryGovernor::new(0, 64).unwrap();
        let _active = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 64, 64)).unwrap().unwrap();

        assert!(governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 1, 1)).unwrap().is_none());
    }

    #[test]
    fn try_grow_charges_available_capacity_only() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();
        let other = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 50, 50)).unwrap().unwrap();

        assert!(!grant.try_grow(20).unwrap());
        drop(other);
        assert!(grant.try_grow(20).unwrap());
        assert_eq!(grant.budget_bytes(), 60);
        assert_eq!(governor.available_bytes().unwrap(), 40);
    }

    #[test]
    fn hard_cap_rejects_current_budget_undercut() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 60)).unwrap().unwrap();

        let error = grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(50)).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reason"), Some("hard_cap_below_current_budget"));
        assert_eq!(grant.hard_cap_bytes(), None);
    }

    #[test]
    fn try_grow_clamps_to_hard_cap() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();

        assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(55)).unwrap(), (40, 55));
        assert!(grant.try_grow(40).unwrap());
        assert_eq!(grant.budget_bytes(), 55);
        assert_eq!(governor.available_bytes().unwrap(), 45);
        assert!(!grant.try_grow(1).unwrap());
        assert_eq!(grant.budget_bytes(), 55);
    }

    #[test]
    fn try_grow_to_returns_false_above_hard_cap() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();

        assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(60)).unwrap(), (40, 60));
        assert!(!grant.try_grow_to(90).unwrap());
        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(governor.available_bytes().unwrap(), 60);
        assert!(grant.try_grow_to(60).unwrap());
        assert_eq!(grant.budget_bytes(), 60);
        assert_eq!(governor.available_bytes().unwrap(), 40);
    }

    #[test]
    fn hard_cap_install_returns_existing_effective_cap() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();

        assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|budget| Ok(budget * 2)).unwrap(), (40, 80));
        assert!(grant.try_grow_to(60).unwrap());
        assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|budget| Ok(budget * 3)).unwrap(), (60, 80));
        assert_eq!(grant.hard_cap_bytes(), Some(80));
    }

    #[test]
    fn try_grow_to_rereads_budget_after_evictable_pressure_observer() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 40)).unwrap().unwrap();
        let other = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 40, 40)).unwrap().unwrap();
        assert!(governor.try_reserve_evictable_bytes(20).unwrap());

        let other_slot = Arc::new(Mutex::new(Some(other)));
        let observer_grant = grant.clone();
        let observer_governor = governor.clone();
        let observer_other_slot = Arc::clone(&other_slot);
        let (sender, receiver) = mpsc::channel();
        governor
            .register_evictable_memory_observer(Arc::new(move |pressure| {
                sender.send(pressure).unwrap();
                drop(observer_other_slot.lock().unwrap().take());
                observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)?;
                assert!(observer_grant.try_grow(30)?);
                Ok(pressure.requested_evict_bytes)
            }))
            .unwrap();

        assert!(grant.try_grow_to(60).unwrap());

        let pressure = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(pressure.requested_evict_bytes, 20);
        assert_eq!(grant.budget_bytes(), 70);
        assert_eq!(governor.report().unwrap().outstanding_bytes, 70);
    }

    #[test]
    fn retain_derived_live_reservation_fails_when_target_exceeds_hard_cap() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 40, 50)).unwrap().unwrap();
        assert_eq!(grant.install_hard_cap_bytes_for_current_budget(|_budget| Ok(60)).unwrap(), (50, 60));
        grant.record_live_reservation_delta(50);
        let retained = grant.retain_live_reservation(50).unwrap();
        grant.record_live_reservation_delta(-50);

        let error = match retained.retain_derived_live_reservation(20) {
            Ok(_) => panic!("derived retention above hard cap must fail"),
            Err(error) => error,
        };

        assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
        assert_eq!(error.fact_value("target_budget_bytes"), Some("70"));
        assert_eq!(error.fact_value("budget_bytes"), Some("50"));
        assert_eq!(grant.budget_bytes(), 50);
        assert_eq!(grant.live_reserved_bytes(), 50);
    }

    #[test]
    fn try_shrink_to_refunds_capacity_without_dropping_grant() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert!(grant.try_shrink_to(50).unwrap());
        assert_eq!(grant.budget_bytes(), 50);
        assert_eq!(governor.available_bytes().unwrap(), 50);

        let other = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 40, 40)).unwrap().unwrap();
        assert_eq!(other.budget_bytes(), 40);
        assert_eq!(governor.available_bytes().unwrap(), 10);
    }

    #[test]
    fn try_shrink_to_preserves_admission_floor() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert!(grant.try_shrink_to(10).unwrap());
        assert_eq!(grant.min_bytes(), 30);
        assert_eq!(grant.budget_bytes(), 30);
        assert_eq!(governor.available_bytes().unwrap(), 70);

        assert!(grant.try_shrink_to(0).unwrap());
        assert_eq!(grant.budget_bytes(), 30);
        assert_eq!(governor.available_bytes().unwrap(), 70);
    }

    #[test]
    fn try_shrink_to_does_not_grow_grant() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 40)).unwrap().unwrap();

        assert!(grant.try_shrink_to(90).unwrap());
        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(governor.available_bytes().unwrap(), 60);
    }

    #[test]
    fn shrink_request_does_not_refund_until_acknowledged() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
        assert_eq!(grant.budget_bytes(), 80);
        assert_eq!(governor.available_bytes().unwrap(), 20);

        let blocked = governor.try_request(GrantRequest::new(AdmissionRequestId::new(2), 30, 30)).unwrap();
        assert!(blocked.is_none());
    }

    #[test]
    fn shrink_request_clamps_to_floor_and_keeps_lowest_target() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(60).unwrap(), Some(60));
        assert_eq!(grant.request_shrink_to(10).unwrap(), Some(30));
        assert_eq!(grant.request_shrink_to(50).unwrap(), Some(30));
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(30));
    }

    #[test]
    fn apply_requested_shrink_refunds_only_to_live_reserved_bytes() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert!(!grant.try_apply_requested_shrink(55).unwrap());
        assert_eq!(grant.budget_bytes(), 55);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
        assert_eq!(governor.available_bytes().unwrap(), 45);

        assert!(grant.try_apply_requested_shrink(35).unwrap());
        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
        assert_eq!(governor.available_bytes().unwrap(), 60);
    }

    #[test]
    fn retained_live_reservation_blocks_shrink_until_final_handle_drop() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        grant.record_live_reservation_delta(50);
        let retained = grant.retain_live_reservation(50).unwrap();
        let retained_clone = retained.clone();
        grant.record_live_reservation_delta(-50);

        assert_eq!(grant.live_reserved_bytes(), 50);
        assert_eq!(grant.request_shrink_to(30).unwrap(), Some(30));
        assert!(!grant.try_apply_requested_shrink(grant.live_reserved_bytes()).unwrap());
        assert_eq!(grant.budget_bytes(), 50);

        drop(retained);
        assert_eq!(grant.live_reserved_bytes(), 50);
        drop(retained_clone);
        assert_eq!(grant.live_reserved_bytes(), 0);
        assert!(grant.try_apply_requested_shrink(grant.live_reserved_bytes()).unwrap());
        assert_eq!(grant.budget_bytes(), 30);
    }

    #[test]
    fn direct_shrink_clears_satisfied_request() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert!(grant.try_shrink_to(40).unwrap());

        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
    }

    #[test]
    fn direct_shrink_above_requested_target_keeps_request_pending() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert!(grant.try_shrink_to(60).unwrap());

        assert_eq!(grant.budget_bytes(), 60);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
    }

    #[test]
    fn clear_requested_shrink_rescinds_pending_request_without_refund() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert!(grant.clear_requested_shrink().unwrap());

        assert_eq!(grant.budget_bytes(), 80);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
        assert_eq!(governor.available_bytes().unwrap(), 20);
        assert!(!grant.clear_requested_shrink().unwrap());
    }

    #[test]
    fn budget_decrease_observer_runs_on_shrink_not_grow() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 40)).unwrap().unwrap();
        let observer_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        grant
            .notify_on_budget_decrease(Arc::new(move || {
                sender.send(observer_governor.available_bytes().unwrap()).unwrap();
            }))
            .unwrap();

        assert!(grant.try_grow_to(70).unwrap());
        assert!(receiver.try_recv().is_err());

        assert!(grant.try_shrink_to(50).unwrap());
        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 50);
    }

    #[test]
    fn budget_change_observer_runs_on_grow_and_shrink() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 40)).unwrap().unwrap();
        let (sender, receiver) = mpsc::channel();
        let initial =
            grant.notify_on_budget_change(Arc::new(move |change| sender.send(change.budget_bytes).unwrap())).unwrap();
        assert_eq!(initial.budget_bytes, 40);
        assert_eq!(initial.sequence, 0);

        assert!(grant.try_grow_to(70).unwrap());
        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 70);

        assert!(grant.try_shrink_to(50).unwrap());
        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 50);

        assert!(grant.try_grow(0).unwrap());
        assert!(grant.try_shrink_to(90).unwrap());
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn current_budget_change_snapshots_match_published_pairs_under_concurrency() -> Result<()> {
        fn locked_pair(grant: &MemoryGrant) -> (u64, u64) {
            let state = grant.inner.governor.state.lock().unwrap();
            let budget = state.grants.get(&grant.inner.grant_id).unwrap().budget_bytes;
            let sequence = grant.inner.budget_change_sequence.load(Ordering::Acquire);
            (budget, sequence)
        }

        let governor = MemoryGovernor::new(0, 200)?;
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 50, 100))?.unwrap();
        let published = Arc::new(Mutex::new(vec![locked_pair(&grant)]));
        let writer_grant = grant.clone();
        let writer_published = Arc::clone(&published);
        let writer = thread::spawn(move || -> Result<()> {
            for iteration in 0..10_000 {
                if iteration % 2 == 0 {
                    assert!(writer_grant.try_grow_to(150)?);
                } else {
                    assert!(writer_grant.try_shrink_to(100)?);
                }
                writer_published.lock().unwrap().push(locked_pair(&writer_grant));
                thread::yield_now();
            }
            Ok(())
        });

        let mut observed = Vec::new();
        while !writer.is_finished() {
            let change = grant.current_budget_change()?;
            observed.push((change.budget_bytes, change.sequence));
            thread::yield_now();
        }
        writer.join().unwrap()?;
        for _ in 0..100 {
            let change = grant.current_budget_change()?;
            observed.push((change.budget_bytes, change.sequence));
        }

        let published = published.lock().unwrap().iter().copied().collect::<HashSet<_>>();
        assert!(!observed.is_empty());
        for pair in observed {
            assert!(published.contains(&pair), "observed torn budget snapshot {pair:?}");
        }
        Ok(())
    }

    #[test]
    fn budget_decrease_observer_runs_only_when_requested_shrink_is_applied() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();
        let (sender, receiver) = mpsc::channel();
        grant.notify_on_budget_decrease(Arc::new(move || sender.send(()).unwrap())).unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert!(receiver.try_recv().is_err());

        assert!(!grant.try_apply_requested_shrink(55).unwrap());
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn shrink_request_after_satisfied_direct_shrink_is_not_recorded() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 80)).unwrap().unwrap();

        assert!(grant.try_shrink_to(40).unwrap());
        assert_eq!(grant.request_shrink_to(50).unwrap(), None);

        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
        assert!(grant.try_grow_to(70).unwrap());
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
    }

    #[test]
    fn blocked_request_wakes_after_shrink() {
        let governor = MemoryGovernor::new(0, 64).unwrap();
        let active = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 32, 64)).unwrap().unwrap();
        let waiting_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let grant = waiting_governor
                .request(GrantRequest::new(AdmissionRequestId::new(2), 32, 32).with_deadline(deadline))
                .unwrap();
            sender.send(grant.budget_bytes()).unwrap();
        });

        thread::sleep(Duration::from_millis(25));
        assert!(active.try_shrink_to(32).unwrap());

        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 32);
        waiter.join().unwrap();
        assert_eq!(governor.available_bytes().unwrap(), 32);
    }

    #[test]
    fn blocked_request_wakes_after_release_without_deadlock() {
        let governor = MemoryGovernor::new(0, 64).unwrap();
        let active = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 64, 64)).unwrap().unwrap();
        let waiting_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let grant = waiting_governor
                .request(GrantRequest::new(AdmissionRequestId::new(2), 32, 32).with_deadline(deadline))
                .unwrap();
            sender.send(grant.budget_bytes()).unwrap();
        });

        thread::sleep(Duration::from_millis(25));
        drop(active);
        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 32);
        waiter.join().unwrap();
        assert_eq!(governor.available_bytes().unwrap(), 64);
    }

    #[test]
    fn evictable_bytes_are_reported_as_grantable_but_not_free() {
        let governor = MemoryGovernor::new(0, 100).unwrap();

        assert!(governor.try_reserve_evictable_bytes(60).unwrap());
        assert_eq!(governor.evictable_bytes().unwrap(), 60);
        assert_eq!(governor.free_bytes().unwrap(), 40);
        assert_eq!(governor.available_bytes().unwrap(), 100);

        let report = governor.report().unwrap();
        assert_eq!(report.outstanding_bytes, 0);
        assert_eq!(report.evictable_bytes, 60);
        assert_eq!(report.free_bytes, 40);
        assert_eq!(report.available_bytes, 100);

        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 50)).unwrap().unwrap();
        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(governor.free_bytes().unwrap(), 0);
        assert_eq!(governor.available_bytes().unwrap(), 60);
        assert!(!governor.try_reserve_evictable_bytes(1).unwrap());
    }

    #[test]
    fn evictable_bytes_can_be_reserved_from_active_grant_budget() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 100, 100)).unwrap().unwrap();

        assert_eq!(governor.free_bytes().unwrap(), 0);
        assert!(!governor.try_reserve_evictable_bytes(100).unwrap());
        assert!(governor.grant_can_cover_evictable_bytes(&grant, 100).unwrap());
        assert!(governor.try_reserve_evictable_bytes_from_grant(&grant, 100).unwrap());

        let active_report = governor.report().unwrap();
        assert_eq!(active_report.outstanding_bytes, 100);
        assert_eq!(active_report.evictable_bytes, 100);
        assert_eq!(active_report.free_bytes, 0);

        drop(grant);
        let retained_report = governor.report().unwrap();
        assert_eq!(retained_report.outstanding_bytes, 0);
        assert_eq!(retained_report.evictable_bytes, 100);
        assert_eq!(retained_report.free_bytes, 0);
    }

    #[test]
    fn evictable_grant_transfer_respects_active_grant_budget() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(1), 30, 50)).unwrap().unwrap();

        assert!(!governor.grant_can_cover_evictable_bytes(&grant, 51).unwrap());
        assert!(!governor.try_reserve_evictable_bytes_from_grant(&grant, 51).unwrap());
        assert!(governor.grant_can_cover_evictable_bytes(&grant, 50).unwrap());
        assert!(governor.try_reserve_evictable_bytes_from_grant(&grant, 50).unwrap());
    }

    #[test]
    fn pressure_observer_releases_evictable_bytes_before_grant() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        assert!(governor.try_reserve_evictable_bytes(80).unwrap());
        let observer_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        governor
            .register_evictable_memory_observer(Arc::new(move |pressure| {
                let report = observer_governor.report().unwrap();
                sender.send((pressure, report.free_bytes)).unwrap();
                observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)
            }))
            .unwrap();

        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 50)).unwrap().unwrap();

        assert_eq!(grant.budget_bytes(), 40);
        let (pressure, observer_free_bytes) = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(pressure.device_ordinal, 0);
        assert_eq!(pressure.admission_request_id, AdmissionRequestId::new(7));
        assert_eq!(pressure.requested_evict_bytes, 20);
        assert_eq!(pressure.min_grant_bytes, 40);
        assert_eq!(pressure.want_grant_bytes, 50);
        assert_eq!(observer_free_bytes, 20);
        let report = governor.report().unwrap();
        assert_eq!(report.outstanding_bytes, 40);
        assert_eq!(report.evictable_bytes, 60);
        assert_eq!(report.free_bytes, 0);
        assert_eq!(report.available_bytes, 60);
    }

    #[test]
    fn grant_growth_releases_evictable_bytes_before_growing() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 20, 40)).unwrap().unwrap();
        assert!(governor.try_reserve_evictable_bytes(50).unwrap());
        let observer_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        governor
            .register_evictable_memory_observer(Arc::new(move |pressure| {
                sender.send(pressure).unwrap();
                observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)
            }))
            .unwrap();

        assert!(grant.try_grow(30).unwrap());

        let pressure = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(pressure.device_ordinal, 0);
        assert_eq!(pressure.admission_request_id, AdmissionRequestId::new(7));
        assert_eq!(pressure.requested_evict_bytes, 20);
        assert_eq!(pressure.min_grant_bytes, 30);
        assert_eq!(pressure.want_grant_bytes, 30);
        let report = governor.report().unwrap();
        assert_eq!(grant.budget_bytes(), 70);
        assert_eq!(report.outstanding_bytes, 70);
        assert_eq!(report.evictable_bytes, 30);
        assert_eq!(report.free_bytes, 0);
    }

    #[test]
    fn pressure_observer_cannot_fake_released_evictable_bytes() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        assert!(governor.try_reserve_evictable_bytes(80).unwrap());
        governor
            .register_evictable_memory_observer(Arc::new(move |pressure| Ok(pressure.requested_evict_bytes)))
            .unwrap();

        assert!(governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 50)).unwrap().is_none());
        let report = governor.report().unwrap();
        assert_eq!(report.outstanding_bytes, 0);
        assert_eq!(report.evictable_bytes, 80);
        assert_eq!(report.free_bytes, 20);
    }

    #[test]
    fn pressure_observer_error_does_not_skip_later_observer() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        assert!(governor.try_reserve_evictable_bytes(80).unwrap());
        let failing_observer_called = Arc::new(AtomicU64::new(0));
        let succeeding_observer_called = Arc::new(AtomicU64::new(0));
        let failing_count = Arc::clone(&failing_observer_called);
        governor
            .register_evictable_memory_observer(Arc::new(move |_pressure| {
                failing_count.fetch_add(1, Ordering::AcqRel);
                Err(QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.runtime.gpu.memory_governor.test_observer",
                    "synthetic observer failure",
                ))
            }))
            .unwrap();
        let observer_governor = governor.clone();
        let succeeding_count = Arc::clone(&succeeding_observer_called);
        governor
            .register_evictable_memory_observer(Arc::new(move |pressure| {
                succeeding_count.fetch_add(1, Ordering::AcqRel);
                observer_governor.release_evictable_bytes(pressure.requested_evict_bytes)
            }))
            .unwrap();

        let grant = governor.try_request(GrantRequest::new(AdmissionRequestId::new(7), 40, 50)).unwrap().unwrap();

        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(failing_observer_called.load(Ordering::Acquire), 1);
        assert_eq!(succeeding_observer_called.load(Ordering::Acquire), 1);
        let report = governor.report().unwrap();
        assert_eq!(report.outstanding_bytes, 40);
        assert_eq!(report.evictable_bytes, 60);
        assert_eq!(report.free_bytes, 0);
    }

    #[test]
    fn blocked_request_wakes_after_evictable_release() {
        let governor = MemoryGovernor::new(0, 64).unwrap();
        assert!(governor.try_reserve_evictable_bytes(64).unwrap());
        let waiting_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let grant = waiting_governor
                .request(GrantRequest::new(AdmissionRequestId::new(2), 32, 32).with_deadline(deadline))
                .unwrap();
            sender.send(grant.budget_bytes()).unwrap();
        });

        thread::sleep(Duration::from_millis(25));
        assert_eq!(governor.release_evictable_bytes(32).unwrap(), 32);

        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 32);
        waiter.join().unwrap();
        let report = governor.report().unwrap();
        assert_eq!(report.outstanding_bytes, 0);
        assert_eq!(report.evictable_bytes, 32);
        assert_eq!(report.free_bytes, 32);
    }

    #[test]
    fn report_is_separate_grant_surface() {
        let governor = MemoryGovernor::new(1, 100).unwrap();
        let _grant = governor
            .try_request(
                GrantRequest::new(AdmissionRequestId::new(7), 10, 25)
                    .with_qos(QueryQos::new(super::super::Priority::High)),
            )
            .unwrap()
            .unwrap();

        let report = governor.report().unwrap();
        assert_eq!(report.device_ordinal, 1);
        assert_eq!(report.capacity_bytes, 100);
        assert_eq!(report.outstanding_bytes, 25);
        assert_eq!(report.evictable_bytes, 0);
        assert_eq!(report.free_bytes, 75);
        assert_eq!(report.available_bytes, 75);
        assert_eq!(report.grants.len(), 1);
        assert_eq!(report.grants[0].priority, "high");
    }
}
