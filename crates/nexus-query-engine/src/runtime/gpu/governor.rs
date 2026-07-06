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
    runtime::gpu::QueryQos,
};

type ReleaseObserver = Arc<dyn Fn() + Send + Sync + 'static>;
const NO_SHRINK_REQUEST: u64 = 0;

/// Request for a variable per-query GPU memory grant.
#[derive(Debug, Clone, Copy)]
pub struct GrantRequest {
    /// Service-local query id requesting memory.
    pub query_id: u64,
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
    pub const fn new(query_id: u64, min_bytes: u64, want_bytes: u64) -> Self {
        Self { query_id, qos: QueryQos::new(super::Priority::Normal), min_bytes, want_bytes, deadline: None }
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

#[derive(Debug)]
struct MemoryGovernorInner {
    device_ordinal: usize,
    capacity_bytes: u64,
    state: Mutex<GovernorState>,
    available: Condvar,
}

#[derive(Debug)]
struct GovernorState {
    next_grant_id: u64,
    outstanding_bytes: u64,
    grants: BTreeMap<u64, ActiveGrant>,
}

#[derive(Debug, Clone, Copy)]
struct ActiveGrant {
    query_id: u64,
    qos: QueryQos,
    min_bytes: u64,
    budget_bytes: u64,
    requested_shrink_target_bytes: u64,
}

/// RAII query memory grant. Dropping the final clone returns its budget.
#[derive(Clone)]
pub struct MemoryGrant {
    inner: Arc<MemoryGrantInner>,
}

struct MemoryGrantInner {
    governor: Arc<MemoryGovernorInner>,
    grant_id: u64,
    query_id: u64,
    qos: QueryQos,
    min_bytes: u64,
    budget_bytes: AtomicU64,
    release_observers: Mutex<Vec<ReleaseObserver>>,
    budget_decrease_observers: Mutex<Vec<ReleaseObserver>>,
}

/// Point-in-time governor report. This is intentionally separate from strict
/// and adaptive execution reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryGovernorReport {
    pub device_ordinal: usize,
    pub capacity_bytes: u64,
    pub outstanding_bytes: u64,
    pub available_bytes: u64,
    pub grants: Vec<MemoryGovernorGrantRow>,
}

/// One active grant row in [`MemoryGovernorReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryGovernorGrantRow {
    pub grant_id: u64,
    pub query_id: u64,
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
                state: Mutex::new(GovernorState { next_grant_id: 1, outstanding_bytes: 0, grants: BTreeMap::new() }),
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

    /// Returns currently ungranted bytes.
    pub fn available_bytes(&self) -> Result<u64> {
        let state = self.lock_state("available_bytes")?;
        Ok(self.inner.capacity_bytes.saturating_sub(state.outstanding_bytes))
    }

    /// Attempts to grant immediately. Returns `Ok(None)` when less than
    /// `min_bytes` is available.
    pub fn try_request(&self, request: GrantRequest) -> Result<Option<MemoryGrant>> {
        self.validate_request(&request)?;
        let mut state = self.lock_state("try_request")?;
        Ok(self.try_grant_locked(&mut state, request))
    }

    /// Blocks until at least `min_bytes` is available or `deadline` expires.
    pub fn request(&self, request: GrantRequest) -> Result<MemoryGrant> {
        self.validate_request(&request)?;
        let mut state = self.lock_state("request")?;
        loop {
            if let Some(grant) = self.try_grant_locked(&mut state, request) {
                return Ok(grant);
            }
            if let Some(deadline) = request.deadline {
                let now = Instant::now();
                if now >= deadline {
                    return Err(Self::timeout_error(&request, self.inner.capacity_bytes, state.outstanding_bytes));
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
                if timeout.timed_out()
                    && self.inner.capacity_bytes.saturating_sub(state.outstanding_bytes) < request.min_bytes
                {
                    return Err(Self::timeout_error(&request, self.inner.capacity_bytes, state.outstanding_bytes));
                }
            } else {
                state = self.inner.available.wait(state).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.runtime.gpu.memory_governor.request",
                        "memory governor mutex was poisoned while waiting",
                    )
                })?;
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
                query_id: grant.query_id,
                priority: grant.qos.priority().as_str(),
                budget_bytes: grant.budget_bytes,
            })
            .collect();
        Ok(MemoryGovernorReport {
            device_ordinal: self.inner.device_ordinal,
            capacity_bytes: self.inner.capacity_bytes,
            outstanding_bytes: state.outstanding_bytes,
            available_bytes: self.inner.capacity_bytes.saturating_sub(state.outstanding_bytes),
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
        let available = self.inner.capacity_bytes.saturating_sub(state.outstanding_bytes);
        if available < request.min_bytes {
            return None;
        }
        let budget_bytes = request.want_bytes.min(available);
        let grant_id = state.next_grant_id;
        state.next_grant_id = state.next_grant_id.saturating_add(1);
        state.outstanding_bytes = state.outstanding_bytes.saturating_add(budget_bytes);
        state.grants.insert(
            grant_id,
            ActiveGrant {
                query_id: request.query_id,
                qos: request.qos,
                min_bytes: request.min_bytes,
                budget_bytes,
                requested_shrink_target_bytes: NO_SHRINK_REQUEST,
            },
        );
        Some(MemoryGrant {
            inner: Arc::new(MemoryGrantInner {
                governor: Arc::clone(&self.inner),
                grant_id,
                query_id: request.query_id,
                qos: request.qos,
                min_bytes: request.min_bytes,
                budget_bytes: AtomicU64::new(budget_bytes),
                release_observers: Mutex::new(Vec::new()),
                budget_decrease_observers: Mutex::new(Vec::new()),
            }),
        })
    }

    fn timeout_error(request: &GrantRequest, capacity_bytes: u64, outstanding_bytes: u64) -> QueryError {
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.runtime.gpu.memory_governor.request",
            "memory governor grant request timed out",
        )
        .with_fact("reason", "memory_grant_wait_timeout")
        .with_fact("query_id", request.query_id)
        .with_fact("min_bytes", request.min_bytes)
        .with_fact("want_bytes", request.want_bytes)
        .with_fact("capacity_bytes", capacity_bytes)
        .with_fact("outstanding_bytes", outstanding_bytes)
    }
}

impl MemoryGrant {
    /// Unique governor-local grant id.
    pub fn grant_id(&self) -> u64 {
        self.inner.grant_id
    }

    /// Query id that owns this grant.
    pub fn query_id(&self) -> u64 {
        self.inner.query_id
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

    /// Records a cooperative shrink target without refunding governor capacity.
    ///
    /// This is the request half of chunk-boundary reclamation. The executing
    /// query must later acknowledge the request at a safe boundary via
    /// [`Self::try_apply_requested_shrink`], after resident work above the
    /// target has been released or replanned.
    pub fn request_shrink_to(&self, target_bytes: u64) -> Result<Option<u64>> {
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

    /// Attempts to increase this grant by `additional_bytes`.
    ///
    /// This is intended for adaptive execution. Strict bounded execution should
    /// treat the initial grant as a gate-first budget.
    pub fn try_grow(&self, additional_bytes: u64) -> Result<bool> {
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
        let available = self.inner.governor.capacity_bytes.saturating_sub(state.outstanding_bytes);
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
        Ok(true)
    }

    /// Attempts to grow this grant to at least `target_bytes`.
    pub fn try_grow_to(&self, target_bytes: u64) -> Result<bool> {
        let mut state = self.inner.governor.state.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.runtime.gpu.memory_grant.try_grow_to",
                "memory governor mutex was poisoned",
            )
        })?;
        let Some(current_budget) = state.grants.get(&self.inner.grant_id).map(|active| active.budget_bytes) else {
            return Ok(false);
        };
        if target_bytes <= current_budget {
            return Ok(true);
        }
        let additional_bytes = target_bytes - current_budget;
        let available = self.inner.governor.capacity_bytes.saturating_sub(state.outstanding_bytes);
        if available < additional_bytes {
            return Ok(false);
        }
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        active.budget_bytes = target_bytes;
        state.outstanding_bytes = state.outstanding_bytes.saturating_add(additional_bytes);
        self.inner.budget_bytes.store(target_bytes, Ordering::Release);
        Ok(true)
    }

    /// Attempts to reduce this grant to `target_bytes`, clamped to the
    /// admission minimum.
    ///
    /// This only refunds soft governor capacity. Callers must use it after the
    /// executor has reached a cooperative boundary and released/replanned any
    /// resident work above the target.
    pub fn try_shrink_to(&self, target_bytes: u64) -> Result<bool> {
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
        let target_bytes = target_bytes.max(active.min_bytes);
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
        self.inner.governor.available.notify_all();
        drop(state);
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
        let safe_target = requested_target.max(live_reserved_bytes).max(active.min_bytes);
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
        self.inner.governor.available.notify_all();
        drop(state);
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
}

impl std::fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("grant_id", &self.grant_id())
            .field("query_id", &self.query_id())
            .field("qos", &self.qos())
            .field("min_bytes", &self.min_bytes())
            .field("budget_bytes", &self.budget_bytes())
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
    .with_fact("query_id", request.query_id)
    .with_fact("min_bytes", request.min_bytes)
    .with_fact("want_bytes", request.want_bytes)
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use super::*;

    #[test]
    fn grants_never_exceed_capacity_and_release_on_drop() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let first = governor.try_request(GrantRequest::new(1, 40, 70)).unwrap().unwrap();
        let second = governor.try_request(GrantRequest::new(2, 20, 70)).unwrap().unwrap();

        assert_eq!(first.budget_bytes(), 70);
        assert_eq!(second.budget_bytes(), 30);
        assert_eq!(governor.available_bytes().unwrap(), 0);
        assert!(governor.try_request(GrantRequest::new(3, 1, 1)).unwrap().is_none());

        drop(first);
        assert_eq!(governor.available_bytes().unwrap(), 70);
        drop(second);
        assert_eq!(governor.available_bytes().unwrap(), 100);
    }

    #[test]
    fn small_query_receives_want_not_capacity_partition() {
        let governor = MemoryGovernor::new(0, 96).unwrap();
        let small = governor.try_request(GrantRequest::new(1, 8, 12)).unwrap().unwrap();

        assert_eq!(small.budget_bytes(), 12);
        assert_eq!(governor.available_bytes().unwrap(), 84);
    }

    #[test]
    fn try_request_returns_none_under_pressure() {
        let governor = MemoryGovernor::new(0, 64).unwrap();
        let _active = governor.try_request(GrantRequest::new(1, 64, 64)).unwrap().unwrap();

        assert!(governor.try_request(GrantRequest::new(2, 1, 1)).unwrap().is_none());
    }

    #[test]
    fn try_grow_charges_available_capacity_only() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 40, 40)).unwrap().unwrap();
        let other = governor.try_request(GrantRequest::new(2, 50, 50)).unwrap().unwrap();

        assert!(!grant.try_grow(20).unwrap());
        drop(other);
        assert!(grant.try_grow(20).unwrap());
        assert_eq!(grant.budget_bytes(), 60);
        assert_eq!(governor.available_bytes().unwrap(), 40);
    }

    #[test]
    fn try_shrink_to_refunds_capacity_without_dropping_grant() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

        assert!(grant.try_shrink_to(50).unwrap());
        assert_eq!(grant.budget_bytes(), 50);
        assert_eq!(governor.available_bytes().unwrap(), 50);

        let other = governor.try_request(GrantRequest::new(2, 40, 40)).unwrap().unwrap();
        assert_eq!(other.budget_bytes(), 40);
        assert_eq!(governor.available_bytes().unwrap(), 10);
    }

    #[test]
    fn try_shrink_to_preserves_admission_floor() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

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
        let grant = governor.try_request(GrantRequest::new(1, 30, 40)).unwrap().unwrap();

        assert!(grant.try_shrink_to(90).unwrap());
        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(governor.available_bytes().unwrap(), 60);
    }

    #[test]
    fn shrink_request_does_not_refund_until_acknowledged() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
        assert_eq!(grant.budget_bytes(), 80);
        assert_eq!(governor.available_bytes().unwrap(), 20);

        let blocked = governor.try_request(GrantRequest::new(2, 30, 30)).unwrap();
        assert!(blocked.is_none());
    }

    #[test]
    fn shrink_request_clamps_to_floor_and_keeps_lowest_target() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(60).unwrap(), Some(60));
        assert_eq!(grant.request_shrink_to(10).unwrap(), Some(30));
        assert_eq!(grant.request_shrink_to(50).unwrap(), Some(30));
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(30));
    }

    #[test]
    fn apply_requested_shrink_refunds_only_to_live_reserved_bytes() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

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
    fn direct_shrink_clears_satisfied_request() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert!(grant.try_shrink_to(40).unwrap());

        assert_eq!(grant.budget_bytes(), 40);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), None);
    }

    #[test]
    fn direct_shrink_above_requested_target_keeps_request_pending() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

        assert_eq!(grant.request_shrink_to(40).unwrap(), Some(40));
        assert!(grant.try_shrink_to(60).unwrap());

        assert_eq!(grant.budget_bytes(), 60);
        assert_eq!(grant.requested_shrink_target_bytes().unwrap(), Some(40));
    }

    #[test]
    fn budget_decrease_observer_runs_on_shrink_not_grow() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 40)).unwrap().unwrap();
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
    fn budget_decrease_observer_runs_only_when_requested_shrink_is_applied() {
        let governor = MemoryGovernor::new(0, 100).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();
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
        let grant = governor.try_request(GrantRequest::new(1, 30, 80)).unwrap().unwrap();

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
        let active = governor.try_request(GrantRequest::new(1, 32, 64)).unwrap().unwrap();
        let waiting_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let grant = waiting_governor.request(GrantRequest::new(2, 32, 32).with_deadline(deadline)).unwrap();
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
        let active = governor.try_request(GrantRequest::new(1, 64, 64)).unwrap().unwrap();
        let waiting_governor = governor.clone();
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let grant = waiting_governor.request(GrantRequest::new(2, 32, 32).with_deadline(deadline)).unwrap();
            sender.send(grant.budget_bytes()).unwrap();
        });

        thread::sleep(Duration::from_millis(25));
        drop(active);
        assert_eq!(receiver.recv_timeout(Duration::from_secs(1)).unwrap(), 32);
        waiter.join().unwrap();
        assert_eq!(governor.available_bytes().unwrap(), 64);
    }

    #[test]
    fn report_is_separate_grant_surface() {
        let governor = MemoryGovernor::new(1, 100).unwrap();
        let _grant = governor
            .try_request(GrantRequest::new(7, 10, 25).with_qos(QueryQos::new(super::super::Priority::High)))
            .unwrap()
            .unwrap();

        let report = governor.report().unwrap();
        assert_eq!(report.device_ordinal, 1);
        assert_eq!(report.capacity_bytes, 100);
        assert_eq!(report.outstanding_bytes, 25);
        assert_eq!(report.available_bytes, 75);
        assert_eq!(report.grants.len(), 1);
        assert_eq!(report.grants[0].priority, "high");
    }
}
