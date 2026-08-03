//! Versioned per-device immutable-grant ledger.
//!
//! One [`DeviceLedger`] owns admission accounting for exactly one CUDA device:
//! an immutable [`DeviceResourceProfile`] plus a mutable, versioned state of
//! active attempt grants, one optional slot protection, device admission
//! health, and cache-internal evictable accounting (kept for the disabled
//! cache substrate; never part of the admission equation while cache caps are
//! zero).
//!
//! The admission equation is normative (phase-2 resource equations):
//!
//! ```text
//! active_attempts + 1 <= max_active_attempts
//! unprotected_empty = max_active - active_after - protected_slots  (checked)
//! protected_reserve = max(configured_floor, protected_liveness_bytes)
//! max_new_grant = managed_capacity - live_grants - protected_reserve
//!               - configured_floor * unprotected_empty
//! required_floor <= grant <= max_new_grant
//! ```
//!
//! All arithmetic is checked or saturating at validation boundaries and exact
//! inside the ledger transaction. Estimates may steer the grant within
//! `[required_floor, max_new_grant]` but never the fit proof.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

use super::DeviceResourceProfile;
use crate::{
    admission::AttemptRequirement,
    error::{QueryError, QueryErrorCode, Result},
    observability::AdmissionRequestId,
    runtime::context::QueryServiceResourceEpoch,
};

const OPERATION: &str = "nexus_query_engine.gpu.device_ledger";

/// Callback used by caches to release evictable device bytes under grant pressure.
///
/// The returned byte count is advisory diagnostics only; the ledger always
/// recomputes pressure from its own state after each observer callback.
pub type EvictableMemoryObserver = Arc<dyn Fn(EvictableMemoryPressure) -> Result<u64> + Send + Sync + 'static>;

/// Memory-pressure request delivered to evictable-cache observers.
///
/// Observers should evict up to `requested_evict_bytes` on this device and
/// return the number of bytes actually released from the ledger's evictable
/// accounting. They are invoked without holding the ledger state mutex.
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

/// One protected future slot for an overtake-limited waiter.
///
/// Protected bytes are neither a grant nor preemption: they withhold one
/// future slot plus subsequently freed managed bytes up to the waiter's
/// remaining floor deficit. At most one protection exists per device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotProtection {
    pub(crate) waiter_id: AdmissionRequestId,
    /// Freed managed bytes banked for this waiter since protection began;
    /// grows only up to the waiter's required floor.
    pub(crate) liveness_bytes: u64,
    /// The protected waiter's required floor (its remaining deficit target).
    pub(crate) required_floor_bytes: u64,
}

/// Atomic read of one device ledger at one observation sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerSnapshot {
    pub device_ordinal: usize,
    pub sequence: u64,
    pub managed_capacity_bytes: u64,
    pub configured_floor_bytes: u64,
    pub max_active_attempts: usize,
    pub active_attempts: usize,
    pub live_grants_bytes: u64,
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

    /// Bytes grantable to a new candidate with `required_floor_bytes`, using
    /// the normative equation. `None` when the candidate is not eligible.
    pub fn max_new_grant_bytes(&self, required_floor_bytes: u64, is_protected_waiter: bool) -> Option<u64> {
        let active_after = self.active_attempts.checked_add(1)?;
        let protected_slots = usize::from(self.protected_slot && !is_protected_waiter);
        let unprotected_empty = self.max_active_attempts.checked_sub(active_after)?.checked_sub(protected_slots)?;
        let protected_reserve = if protected_slots == 1 { self.protected_reserve_bytes } else { 0 };
        let empty_floor_reserve = self.configured_floor_bytes.checked_mul(u64::try_from(unprotected_empty).ok()?)?;
        let max_new_grant = self
            .managed_capacity_bytes
            .checked_sub(self.live_grants_bytes)?
            .checked_sub(protected_reserve)?
            .checked_sub(empty_floor_reserve)?;
        (required_floor_bytes <= max_new_grant).then_some(max_new_grant)
    }
}

/// Public diagnostic report for one device ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLedgerReport {
    pub device_ordinal: usize,
    pub sequence: u64,
    pub managed_capacity_bytes: u64,
    pub configured_floor_bytes: u64,
    pub max_active_attempts: usize,
    pub active_attempts: usize,
    pub live_grants_bytes: u64,
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
    /// Managed bytes not currently covered by live grants.
    pub available_bytes: u64,
    /// Immediately grantable bytes: managed minus live grants minus
    /// cache-internal evictable charges.
    pub free_bytes: u64,
    /// Cache-internal evictable accounting (P5 substrate; reported separately
    /// from the always-zero admission cache residency fields).
    pub evictable_bytes: u64,
    pub protected_slot: bool,
    pub protected_reserve_bytes: u64,
    pub cache_evictable_resident_bytes: u64,
    pub health: String,
    pub grants: Vec<DeviceLedgerGrantRow>,
}

/// One active grant row in [`DeviceLedgerReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLedgerGrantRow {
    pub grant_id: u64,
    pub admission_request_id: AdmissionRequestId,
    pub grant_bytes: u64,
    pub live_reserved_bytes: u64,
}

/// Per-device versioned immutable-grant ledger.
#[derive(Clone)]
pub struct DeviceLedger {
    inner: Arc<DeviceLedgerInner>,
}

struct DeviceLedgerInner {
    profile: DeviceResourceProfile,
    state: Mutex<DeviceLedgerState>,
    evictable_memory_observers: Mutex<Vec<EvictableMemoryObserver>>,
    release_hook: Mutex<Option<Arc<dyn Fn(usize) + Send + Sync + 'static>>>,
    available: Condvar,
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
    next_grant_id: u64,
    grants: BTreeMap<u64, ActiveGrant>,
    protection: Option<SlotProtection>,
    health: DeviceAdmissionHealth,
    evictable_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveGrant {
    admission_request_id: AdmissionRequestId,
    grant_bytes: u64,
    live_reserved_bytes: u64,
}

impl DeviceLedgerState {
    fn live_grants_bytes(&self) -> u64 {
        self.grants.values().fold(0u64, |sum, grant| sum.saturating_add(grant.grant_bytes))
    }

    fn live_reserved_bytes(&self) -> u64 {
        self.grants.values().fold(0u64, |sum, grant| sum.saturating_add(grant.live_reserved_bytes))
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
                    next_grant_id: 1,
                    grants: BTreeMap::new(),
                    protection: None,
                    health: DeviceAdmissionHealth::Open,
                    evictable_bytes: 0,
                }),
                evictable_memory_observers: Mutex::new(Vec::new()),
                release_hook: Mutex::new(None),
                available: Condvar::new(),
            }),
        }
    }

    /// Test constructor deriving a valid profile from raw capacity inputs.
    #[cfg(test)]
    pub(crate) fn for_test(device_ordinal: usize, managed_capacity_bytes: u64, max_active_attempts: usize) -> Self {
        let max_active_attempts = max_active_attempts.max(1);
        let source_work = 1usize;
        let slots = max_active_attempts.saturating_mul(2).max(max_active_attempts);
        let profile = DeviceResourceProfile::resolve(
            device_ordinal,
            managed_capacity_bytes,
            0,
            0,
            (managed_capacity_bytes / u64::try_from(max_active_attempts).unwrap_or(1).max(8)).max(1),
            max_active_attempts,
            source_work,
            slots,
            slots,
        )
        .expect("test ledger profile must resolve");
        Self::new(profile)
    }

    /// Test convenience: admit one attempt with a legacy-style band.
    #[cfg(test)]
    pub(crate) fn admit_for_test(
        &self,
        waiter_id: AdmissionRequestId,
        floor_bytes: u64,
        hint_bytes: u64,
    ) -> Result<Option<MemoryGrant>> {
        self.try_admit(
            waiter_id,
            &AttemptRequirement {
                configured_floor_bytes: floor_bytes,
                capability_floor_raise_bytes: 0,
                soft_working_set_hint_bytes: Some(hint_bytes),
                estimate_quality: crate::admission::EstimateQuality::Modeled,
                required_capabilities: Vec::new(),
            },
        )
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

    /// Resolves an attempt requirement against this immutable device profile.
    pub(crate) fn required_floor_bytes_for(&self, requirement: &AttemptRequirement) -> Result<u64> {
        let configured_floor_bytes = self.inner.profile.attempt_floor_bytes.max(requirement.configured_floor_bytes);
        configured_floor_bytes.checked_add(requirement.capability_floor_raise_bytes).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "attempt requirement floor overflows this device profile",
            )
            .with_fact("reason", "attempt_requirement_floor_overflow")
            .with_fact("device_ordinal", self.inner.profile.device_ordinal)
            .with_fact("configured_floor_bytes", configured_floor_bytes)
            .with_fact("capability_floor_raise_bytes", requirement.capability_floor_raise_bytes)
        })
    }

    /// Whether this device could ever reserve a protected slot for the
    /// resolved floor, ignoring currently live grants. The protected waiter
    /// itself occupies one slot, so only the other slots retain their
    /// baseline profile floors.
    pub(crate) fn can_ever_fit_required_floor_bytes(&self, required_floor_bytes: u64) -> Result<bool> {
        if required_floor_bytes == 0 {
            return Ok(false);
        }
        let empty_other_slots = self.inner.profile.max_active_attempts.checked_sub(1).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "device profile has no attempt slot for a protected request",
            )
            .with_fact("reason", "zero_max_active_attempts")
            .with_fact("device_ordinal", self.inner.profile.device_ordinal)
        })?;
        let empty_floor_reserve = self
            .inner
            .profile
            .attempt_floor_bytes
            .checked_mul(u64::try_from(empty_other_slots).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    OPERATION,
                    "device profile empty-slot reservation overflows u64",
                )
                .with_fact("reason", "empty_slot_reservation_overflow")
                .with_fact("device_ordinal", self.inner.profile.device_ordinal)
            })?)
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    OPERATION,
                    "device profile empty-slot reservation overflows u64",
                )
                .with_fact("reason", "empty_slot_reservation_overflow")
                .with_fact("device_ordinal", self.inner.profile.device_ordinal)
            })?;
        Ok(self
            .inner
            .profile
            .managed_capacity_bytes
            .checked_sub(empty_floor_reserve)
            .is_some_and(|maximum| required_floor_bytes <= maximum))
    }

    /// Candidate-local version of [`Self::can_ever_fit_required_floor_bytes`].
    pub(crate) fn can_ever_fit_requirement(&self, requirement: &AttemptRequirement) -> Result<bool> {
        self.can_ever_fit_required_floor_bytes(self.required_floor_bytes_for(requirement)?)
    }

    fn lock_state(&self, action: &'static str) -> Result<MutexGuard<'_, DeviceLedgerState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "device ledger mutex was poisoned")
                .with_fact("action", action)
                .with_fact("reason", "ledger_mutex_poisoned")
        })
    }

    /// One atomic ledger snapshot for placement, reports, and metrics.
    pub fn snapshot(&self) -> Result<LedgerSnapshot> {
        let state = self.lock_state("snapshot")?;
        Ok(self.snapshot_locked(&state))
    }

    fn snapshot_locked(&self, state: &DeviceLedgerState) -> LedgerSnapshot {
        let profile = &self.inner.profile;
        let configured_floor = profile.attempt_floor_bytes;
        let active_attempts = state.grants.len();
        // Admission reserves the complete root plus bounded auxiliary envelope
        // for every active attempt. Derive availability from that immutable
        // envelope under the ledger lock instead of reading mutable executor or
        // stream-pool counters independently.
        let committed_auxiliary_slots = active_attempts.saturating_mul(profile.source_work_per_attempt);
        LedgerSnapshot {
            device_ordinal: profile.device_ordinal,
            sequence: state.sequence,
            managed_capacity_bytes: profile.managed_capacity_bytes,
            configured_floor_bytes: configured_floor,
            max_active_attempts: profile.max_active_attempts,
            active_attempts,
            live_grants_bytes: state.live_grants_bytes(),
            live_reserved_bytes: state.live_reserved_bytes(),
            root_streams_available: profile.root_stream_count.saturating_sub(active_attempts),
            auxiliary_streams_available: profile.aux_source_work_capacity.saturating_sub(committed_auxiliary_slots),
            root_worker_slots_available: profile.root_lane_count.saturating_sub(active_attempts),
            auxiliary_worker_slots_available: profile
                .aux_source_work_capacity
                .saturating_sub(committed_auxiliary_slots),
            worker_depth: active_attempts.saturating_add(committed_auxiliary_slots),
            protected_slot: state.protection.is_some(),
            protected_reserve_bytes: state
                .protection
                .map_or(0, |protection| protection.liveness_bytes.max(configured_floor)),
            // P2 has no resident-cache owner, but any bounded evictable-cache
            // charge is already ledger-owned and must share this atomic
            // snapshot sequence with grants and reservations.
            cache_evictable_resident_bytes: state.evictable_bytes,
            cache_pinned_in_use_bytes: 0,
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
            })
            .collect();
        Ok(DeviceLedgerReport {
            device_ordinal: snapshot.device_ordinal,
            sequence: snapshot.sequence,
            managed_capacity_bytes: snapshot.managed_capacity_bytes,
            configured_floor_bytes: snapshot.configured_floor_bytes,
            max_active_attempts: snapshot.max_active_attempts,
            active_attempts: snapshot.active_attempts,
            live_grants_bytes: snapshot.live_grants_bytes,
            live_reserved_bytes: snapshot.live_reserved_bytes,
            root_streams_available: snapshot.root_streams_available,
            auxiliary_streams_available: snapshot.auxiliary_streams_available,
            root_worker_slots_available: snapshot.root_worker_slots_available,
            auxiliary_worker_slots_available: snapshot.auxiliary_worker_slots_available,
            worker_depth: snapshot.worker_depth,
            available_bytes: snapshot.managed_capacity_bytes.saturating_sub(snapshot.live_grants_bytes),
            free_bytes: snapshot
                .managed_capacity_bytes
                .saturating_sub(snapshot.live_grants_bytes)
                .saturating_sub(state.evictable_bytes),
            evictable_bytes: state.evictable_bytes,
            protected_slot: snapshot.protected_slot,
            protected_reserve_bytes: snapshot.protected_reserve_bytes,
            cache_evictable_resident_bytes: snapshot.cache_evictable_resident_bytes,
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
        state.sequence = state.sequence.saturating_add(1);
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
        state.sequence = state.sequence.saturating_add(1);
        Ok(true)
    }

    /// Returns whether the candidate with `required_floor_bytes` could be
    /// admitted right now (same equation as [`Self::try_admit`]).
    pub(crate) fn candidate_max_new_grant_bytes(
        &self,
        required_floor_bytes: u64,
        waiter_id: AdmissionRequestId,
    ) -> Result<Option<u64>> {
        let state = self.lock_state("candidate_max_new_grant_bytes")?;
        let snapshot = self.snapshot_locked(&state);
        let is_protected = state.protection.is_some_and(|protection| protection.waiter_id == waiter_id);
        Ok(snapshot.max_new_grant_bytes(required_floor_bytes, is_protected))
    }

    /// Returns the candidate maximum after resolving a zero request floor
    /// against this ledger's own immutable profile.
    pub(crate) fn candidate_max_new_grant_for_requirement(
        &self,
        requirement: &AttemptRequirement,
        waiter_id: AdmissionRequestId,
    ) -> Result<Option<u64>> {
        self.candidate_max_new_grant_bytes(self.required_floor_bytes_for(requirement)?, waiter_id)
    }

    /// Attempts one immutable grant admission for `waiter_id`.
    ///
    /// The grant equals the soft working-set hint clamped into
    /// `[required_floor, max_new_grant]`. Admitting the protected waiter
    /// consumes its protection atomically: the immutable grant replaces the
    /// protected reserve and is never subtracted twice. Cache-internal
    /// evictable bytes reduce the immediately grantable headroom; under
    /// pressure the registered cache observers reclaim them first (the P5
    /// substrate; always zero when caches are disabled).
    pub(crate) fn try_admit(
        &self,
        waiter_id: AdmissionRequestId,
        requirement: &AttemptRequirement,
    ) -> Result<Option<MemoryGrant>> {
        let required_floor = self.required_floor_bytes_for(requirement)?;
        if required_floor == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "attempt required floor must be greater than zero",
            )
            .with_fact("reason", "zero_required_floor")
            .with_fact("admission_request_id", waiter_id));
        }
        let mut state = self.lock_state("try_admit")?;
        if !matches!(state.health, DeviceAdmissionHealth::Open) {
            return Ok(None);
        }
        let is_protected = state.protection.is_some_and(|protection| protection.waiter_id == waiter_id);
        let snapshot = self.snapshot_locked(&state);
        let Some(max_new_grant) = snapshot.max_new_grant_bytes(required_floor, is_protected) else {
            return Ok(None);
        };
        let grantable_now = max_new_grant.saturating_sub(state.evictable_bytes);
        if grantable_now < required_floor {
            #[cfg(any(feature = "cugraph", test))]
            {
                let hint = requirement.soft_working_set_hint_bytes.unwrap_or(grantable_now);
                drop(state);
                self.reclaim_evictable_for_grant(waiter_id, required_floor, hint)?;
                state = self.lock_state("try_admit")?;
                if !matches!(state.health, DeviceAdmissionHealth::Open) {
                    return Ok(None);
                }
                let is_protected = state.protection.is_some_and(|protection| protection.waiter_id == waiter_id);
                let snapshot = self.snapshot_locked(&state);
                let Some(max_new_grant) = snapshot.max_new_grant_bytes(required_floor, is_protected) else {
                    return Ok(None);
                };
                let grantable_now = max_new_grant.saturating_sub(state.evictable_bytes);
                if grantable_now < required_floor {
                    return Ok(None);
                }
                return Ok(Some(self.commit_grant_locked(
                    &mut state,
                    waiter_id,
                    requirement,
                    required_floor,
                    grantable_now,
                    is_protected,
                )));
            }
            #[cfg(not(any(feature = "cugraph", test)))]
            {
                return Ok(None);
            }
        }
        Ok(Some(self.commit_grant_locked(
            &mut state,
            waiter_id,
            requirement,
            required_floor,
            grantable_now,
            is_protected,
        )))
    }

    fn commit_grant_locked(
        &self,
        state: &mut DeviceLedgerState,
        waiter_id: AdmissionRequestId,
        requirement: &AttemptRequirement,
        required_floor: u64,
        grant_ceiling: u64,
        is_protected: bool,
    ) -> MemoryGrant {
        // A request with no soft estimate has no estimate-derived bound: the
        // grant is the full candidate maximum, clamped below by its floor.
        let grant_bytes =
            requirement.soft_working_set_hint_bytes.unwrap_or(grant_ceiling).clamp(required_floor, grant_ceiling);
        if is_protected {
            state.protection = None;
        }
        let grant_id = state.next_grant_id;
        state.next_grant_id = state.next_grant_id.saturating_add(1);
        state
            .grants
            .insert(grant_id, ActiveGrant { admission_request_id: waiter_id, grant_bytes, live_reserved_bytes: 0 });
        state.sequence = state.sequence.saturating_add(1);
        MemoryGrant {
            inner: Arc::new(MemoryGrantInner {
                ledger: Arc::clone(&self.inner),
                grant_id,
                admission_request_id: waiter_id,
                grant_bytes,
                live_reserved_bytes: Default::default(),
                reservation_arena: Mutex::new(None),
            }),
            resource_epoch: None,
        }
    }

    /// Installs or refreshes the single slot protection for `waiter_id`.
    ///
    /// Returns `Ok(false)` without changing state when another waiter already
    /// holds this device's protection.
    pub(crate) fn protect_slot(&self, waiter_id: AdmissionRequestId, required_floor_bytes: u64) -> Result<bool> {
        if required_floor_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "slot protection requires a positive attempt floor",
            )
            .with_fact("reason", "zero_protection_floor")
            .with_fact("device_ordinal", self.inner.profile.device_ordinal)
            .with_fact("admission_request_id", waiter_id));
        }
        if !self.can_ever_fit_required_floor_bytes(required_floor_bytes)? {
            return Err(QueryError::new(
                QueryErrorCode::ServiceOverloaded,
                OPERATION,
                "slot protection target can never satisfy the attempt floor",
            )
            .with_fact("reason", "impossible_protection_target")
            .with_fact("device_ordinal", self.inner.profile.device_ordinal)
            .with_fact("required_floor_bytes", required_floor_bytes)
            .with_fact("admission_request_id", waiter_id));
        }
        let mut state = self.lock_state("protect_slot")?;
        if !matches!(state.health, DeviceAdmissionHealth::Open) {
            return Err(QueryError::new(
                QueryErrorCode::ServiceOverloaded,
                OPERATION,
                "slot protection target is closed for admission",
            )
            .with_fact("reason", "closed_protection_target")
            .with_fact("device_ordinal", self.inner.profile.device_ordinal)
            .with_fact("admission_request_id", waiter_id));
        }
        if state.protection.is_some_and(|protection| protection.waiter_id != waiter_id) {
            return Ok(false);
        }
        if state.protection.is_none() {
            state.protection = Some(SlotProtection { waiter_id, liveness_bytes: 0, required_floor_bytes });
            state.sequence = state.sequence.saturating_add(1);
        }
        Ok(true)
    }

    /// Ends this waiter's protection (admission, cancellation, deadline,
    /// impossibility, or retarget). Returns whether a protection was removed.
    pub(crate) fn end_protection(&self, waiter_id: AdmissionRequestId) -> Result<bool> {
        let mut state = self.lock_state("end_protection")?;
        if state.protection.is_some_and(|protection| protection.waiter_id == waiter_id) {
            state.protection = None;
            state.sequence = state.sequence.saturating_add(1);
            return Ok(true);
        }
        Ok(false)
    }

    /// Current protection state for queue/tests.
    pub(crate) fn protection(&self) -> Option<SlotProtection> {
        self.inner.state.lock().ok()?.protection
    }

    /// Installs the single queue-wakeup hook fired after any grant release.
    ///
    /// The hook runs after the ledger state lock is released. Callers must
    /// uphold the same invariant as the legacy release observers: a grant is
    /// never dropped while the admission mutex is held.
    pub(crate) fn set_release_hook(&self, hook: Arc<dyn Fn(usize) + Send + Sync + 'static>) -> Result<()> {
        let mut slot = self.inner.release_hook.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "device ledger release-hook mutex was poisoned")
                .with_fact("reason", "release_hook_mutex_poisoned")
        })?;
        *slot = Some(hook);
        Ok(())
    }

    /// Releases one grant, moving freed managed bytes into the waiter's
    /// protection liveness before any later admission observes them.
    fn release_grant(&self, grant_id: u64) {
        let hook = self.inner.release_hook.lock().ok().and_then(|slot| slot.clone());
        let Ok(mut state) = self.inner.state.lock() else {
            eprintln!("[nexus-query-engine] warning: device ledger release observed a poisoned mutex");
            return;
        };
        let Some(active) = state.grants.remove(&grant_id) else {
            return;
        };
        if let Some(protection) = state.protection.as_mut() {
            let deficit = protection.required_floor_bytes.saturating_sub(protection.liveness_bytes);
            let moved = active.grant_bytes.min(deficit);
            protection.liveness_bytes = protection.liveness_bytes.saturating_add(moved);
        }
        state.sequence = state.sequence.saturating_add(1);
        drop(state);
        self.inner.available.notify_all();
        if let Some(hook) = hook {
            hook(self.inner.profile.device_ordinal);
        }
    }

    // ---- Cache-internal evictable accounting (P5 substrate; never part of
    // the admission equation while cache caps are zero). ----

    /// Registers a synchronous evictable-cache pressure observer.
    pub(crate) fn register_evictable_memory_observer(&self, observer: EvictableMemoryObserver) -> Result<()> {
        self.inner
            .evictable_memory_observers
            .lock()
            .map_err(|_error| {
                QueryError::new(QueryErrorCode::Internal, OPERATION, "evictable observer mutex was poisoned")
                    .with_fact("reason", "evictable_observer_mutex_poisoned")
            })?
            .push(observer);
        Ok(())
    }

    /// Attempts to charge bytes retained by an evictable cache.
    pub(crate) fn try_reserve_evictable_bytes(&self, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        let mut state = self.lock_state("try_reserve_evictable_bytes")?;
        if self.free_bytes_locked(&state) < bytes {
            return Ok(false);
        }
        state.evictable_bytes = state.evictable_bytes.saturating_add(bytes);
        state.sequence = state.sequence.saturating_add(1);
        Ok(true)
    }

    /// Returns whether an active grant from this ledger can cover bytes being
    /// retained as evictable cache memory.
    #[cfg(any(feature = "cugraph", test))]
    pub(crate) fn grant_can_cover_evictable_bytes(&self, grant: &MemoryGrant, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        if !Arc::ptr_eq(&self.inner, &grant.inner.ledger) {
            return Ok(false);
        }
        let state = self.lock_state("grant_can_cover_evictable_bytes")?;
        let Some(active) = state.grants.get(&grant.inner.grant_id) else {
            return Ok(false);
        };
        Ok(bytes <= active.grant_bytes)
    }

    /// Charges evictable cache bytes already covered by an active grant.
    #[cfg(any(feature = "cugraph", test))]
    pub(crate) fn try_reserve_evictable_bytes_from_grant(&self, grant: &MemoryGrant, bytes: u64) -> Result<bool> {
        if bytes == 0 {
            return Ok(true);
        }
        if !Arc::ptr_eq(&self.inner, &grant.inner.ledger) {
            return Ok(false);
        }
        let mut state = self.lock_state("try_reserve_evictable_bytes_from_grant")?;
        let Some(active) = state.grants.get(&grant.inner.grant_id) else {
            return Ok(false);
        };
        if bytes > active.grant_bytes {
            return Ok(false);
        }
        state.evictable_bytes = state.evictable_bytes.saturating_add(bytes);
        state.sequence = state.sequence.saturating_add(1);
        Ok(true)
    }

    /// Releases bytes previously charged to the evictable-cache ledger.
    pub(crate) fn release_evictable_bytes(&self, bytes: u64) -> Result<u64> {
        if bytes == 0 {
            return Ok(0);
        }
        let mut state = self.lock_state("release_evictable_bytes")?;
        let released = bytes.min(state.evictable_bytes);
        if released == 0 {
            return Ok(0);
        }
        state.evictable_bytes = state.evictable_bytes.saturating_sub(released);
        state.sequence = state.sequence.saturating_add(1);
        drop(state);
        self.inner.available.notify_all();
        Ok(released)
    }

    /// Bytes currently charged to evictable caches.
    pub(crate) fn evictable_bytes(&self) -> Result<u64> {
        let state = self.lock_state("evictable_bytes")?;
        Ok(state.evictable_bytes)
    }

    /// Immediate unreserved bytes, excluding bytes held by evictable caches.
    pub(crate) fn free_bytes(&self) -> Result<u64> {
        let state = self.lock_state("free_bytes")?;
        Ok(self.free_bytes_locked(&state))
    }

    fn free_bytes_locked(&self, state: &DeviceLedgerState) -> u64 {
        self.inner
            .profile
            .managed_capacity_bytes
            .saturating_sub(state.live_grants_bytes().saturating_add(state.evictable_bytes))
    }

    /// Runs evictable-cache pressure observers when a grant of `min_bytes`
    /// cannot be satisfied immediately. Kept for the disabled cache
    /// substrate; admission itself never depends on cache bytes in P2.
    #[cfg(any(feature = "cugraph", test))]
    pub(crate) fn reclaim_evictable_for_grant(
        &self,
        admission_request_id: AdmissionRequestId,
        min_bytes: u64,
        want_bytes: u64,
    ) -> Result<()> {
        let (pressure, observers) = {
            let state = self.lock_state("reclaim_evictable_for_grant")?;
            let free_bytes = self.free_bytes_locked(&state);
            if free_bytes >= min_bytes || state.evictable_bytes == 0 {
                return Ok(());
            }
            let grantable = self.inner.profile.managed_capacity_bytes.saturating_sub(state.live_grants_bytes());
            if grantable < min_bytes {
                return Ok(());
            }
            let pressure = EvictableMemoryPressure {
                device_ordinal: self.inner.profile.device_ordinal,
                admission_request_id,
                requested_evict_bytes: min_bytes.saturating_sub(free_bytes),
                min_grant_bytes: min_bytes,
                want_grant_bytes: want_bytes,
                outstanding_bytes: state.live_grants_bytes(),
                evictable_bytes: state.evictable_bytes,
                free_bytes,
            };
            let observers = self
                .inner
                .evictable_memory_observers
                .lock()
                .map_err(|_error| {
                    QueryError::new(QueryErrorCode::Internal, OPERATION, "evictable observer mutex was poisoned")
                        .with_fact("reason", "evictable_observer_mutex_poisoned")
                })?
                .clone();
            (pressure, observers)
        };
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

    #[cfg(any(feature = "cugraph", test))]
    fn evictable_pressure_for_target(
        &self,
        original: EvictableMemoryPressure,
        target_free_bytes: u64,
    ) -> Result<Option<EvictableMemoryPressure>> {
        let state = self.lock_state("evictable_pressure_for_target")?;
        let free_bytes = self.free_bytes_locked(&state);
        let requested_evict_bytes = target_free_bytes.saturating_sub(free_bytes).min(state.evictable_bytes);
        if requested_evict_bytes == 0 {
            return Ok(None);
        }
        Ok(Some(EvictableMemoryPressure {
            requested_evict_bytes,
            outstanding_bytes: state.live_grants_bytes(),
            evictable_bytes: state.evictable_bytes,
            free_bytes,
            ..original
        }))
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
pub(crate) struct MemoryGrantLiveReservation {
    inner: Arc<MemoryGrantLiveReservationInner>,
}

struct MemoryGrantLiveReservationInner {
    grant: MemoryGrant,
    bytes: u64,
}

struct MemoryGrantInner {
    ledger: Arc<DeviceLedgerInner>,
    grant_id: u64,
    admission_request_id: AdmissionRequestId,
    grant_bytes: u64,
    live_reserved_bytes: std::sync::atomic::AtomicU64,
    /// One direct-context arena capability shared by every clone of this
    /// immutable grant.  Admission envelopes use their hard
    /// `AttemptAllocationDomain` instead and never populate this slot.
    reservation_arena: Mutex<Option<Arc<cudf::ReservationArena>>>,
}

impl Drop for MemoryGrantInner {
    fn drop(&mut self) {
        // Destroy the native arena before returning the immutable grant to the
        // ledger.  The arena is grant-local state, so this remains ordered even
        // when the last `MemoryGrant` clone is dropped from a context teardown.
        let arena = self.reservation_arena.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        drop(arena);
        DeviceLedger { inner: Arc::clone(&self.ledger) }.release_grant(self.grant_id);
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

    /// Immutable grant budget in bytes. This value never changes while the
    /// grant is live; v1 has no live grant growth, shrink, borrowing, or
    /// revocation.
    pub fn budget_bytes(&self) -> u64 {
        self.inner.grant_bytes
    }

    /// Device ordinal that owns this grant.
    pub fn device_ordinal(&self) -> usize {
        self.inner.ledger.profile.device_ordinal
    }

    /// Returns the one direct-context reservation arena capability for this
    /// grant, creating it once under the grant's shared ownership cell.
    ///
    /// Cloning a [`MemoryGrant`] clones that cell rather than creating another
    /// full-budget arena.  This seam is intentionally separate from admitted
    /// attempt domains, whose `AdmissionEnvelope` creates and owns the hard
    /// query domain directly.
    pub(crate) fn reservation_arena(&self, runtime: &cudf::Runtime) -> Result<Arc<cudf::ReservationArena>> {
        const OPERATION: &str = "nexus_query_engine.runtime.grant_reservation_arena";
        if runtime.device().ordinal() != self.device_ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "memory grant reservation arena runtime device does not match grant device",
            )
            .with_fact("grant_id", self.grant_id())
            .with_fact("admission_request_id", self.admission_request_id())
            .with_fact("grant_device_ordinal", self.device_ordinal())
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        let limit_bytes = usize::try_from(self.budget_bytes()).map_err(|_error| {
            QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, "memory grant budget exceeds platform usize range")
                .with_fact("grant_id", self.grant_id())
                .with_fact("admission_request_id", self.admission_request_id())
                .with_fact("budget_bytes", self.budget_bytes())
        })?;
        let mut slot = self.inner.reservation_arena.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "memory grant reservation arena mutex was poisoned")
                .with_fact("grant_id", self.grant_id())
                .with_fact("admission_request_id", self.admission_request_id())
        })?;
        if let Some(arena) = slot.as_ref() {
            validate_reservation_arena_identity(self, runtime, arena, OPERATION)?;
            return Ok(Arc::clone(arena));
        }

        let manager = runtime.device_memory_manager().map_err(|error| {
            QueryError::from_cudf(
                OPERATION,
                "failed to resolve device memory manager for grant-backed reservation arena",
                error,
            )
            .with_fact("grant_id", self.grant_id())
            .with_fact("admission_request_id", self.admission_request_id())
        })?;
        let arena = Arc::new(manager.reservation_arena(self.grant_id(), limit_bytes).map_err(|error| {
            QueryError::from_cudf(OPERATION, "failed to create grant-backed reservation arena", error)
                .with_fact("grant_id", self.grant_id())
                .with_fact("admission_request_id", self.admission_request_id())
                .with_fact("budget_bytes", self.budget_bytes())
        })?);
        validate_reservation_arena_identity(self, runtime, &arena, OPERATION)?;
        *slot = Some(Arc::clone(&arena));
        Ok(arena)
    }

    /// Device-resident bytes currently protected by all reservation trackers
    /// sharing this grant.
    pub(crate) fn live_reserved_bytes(&self) -> u64 {
        self.inner.live_reserved_bytes.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Remaining bytes this grant can admit across all active reservation
    /// trackers sharing it.
    pub(crate) fn remaining_live_reservation_bytes(&self) -> Result<u64> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.remaining_live_reservation_bytes")?;
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let state = ledger.lock_state("remaining_live_reservation_bytes")?;
        let Some(active) = state.grants.get(&self.inner.grant_id) else {
            return Ok(0);
        };
        Ok(active.grant_bytes.saturating_sub(active.live_reserved_bytes))
    }

    /// Attempts to charge new device-resident bytes to this grant.
    pub(crate) fn try_reserve_live_bytes(&self, bytes: u64) -> Result<bool> {
        self.check_resource_epoch("nexus_query_engine.runtime.gpu.memory_grant.try_reserve_live_bytes")?;
        if bytes == 0 {
            return Ok(true);
        }
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = ledger.lock_state("try_reserve_live_bytes")?;
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return Ok(false);
        };
        let Some(next_live_reserved_bytes) = active.live_reserved_bytes.checked_add(bytes) else {
            return Ok(false);
        };
        if next_live_reserved_bytes > active.grant_bytes {
            return Ok(false);
        }
        active.live_reserved_bytes = next_live_reserved_bytes;
        self.inner.live_reserved_bytes.store(next_live_reserved_bytes, std::sync::atomic::Ordering::Release);
        state.sequence = state.sequence.saturating_add(1);
        Ok(true)
    }

    /// Releases live bytes previously admitted by
    /// [`Self::try_reserve_live_bytes`].
    pub(crate) fn release_live_bytes(&self, bytes: u64) {
        self.record_live_reservation_delta(-i128::from(bytes));
    }

    /// Releases grant-scoped live reservation bytes already admitted by the
    /// owning reservation tracker or retention handle.
    ///
    /// New positive charges must use [`Self::try_reserve_live_bytes`], which
    /// proves that the immutable grant can cover them. Ownership transfer to
    /// a result handle is represented without a delta by
    /// [`Self::transfer_live_reservation`].
    pub(crate) fn record_live_reservation_delta(&self, delta_bytes: i128) {
        let ledger = DeviceLedger { inner: Arc::clone(&self.inner.ledger) };
        let mut state = match ledger.inner.state.lock() {
            Ok(state) => state,
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to update memory grant live reservation bytes");
                return;
            },
        };
        let Some(active) = state.grants.get_mut(&self.inner.grant_id) else {
            return;
        };
        let previous = active.live_reserved_bytes;
        let next = if delta_bytes.is_negative() {
            let release_bytes = u64::try_from(delta_bytes.unsigned_abs()).unwrap_or(u64::MAX);
            active.live_reserved_bytes.saturating_sub(release_bytes)
        } else {
            // This branch remains defensive for legacy internal callers, but
            // cannot grow beyond the immutable grant. New callers use the
            // checked reservation API above.
            active.live_reserved_bytes.saturating_add(delta_bytes as u64).min(active.grant_bytes)
        };
        if next == previous {
            return;
        }
        active.live_reserved_bytes = next;
        self.inner.live_reserved_bytes.store(next, std::sync::atomic::Ordering::Release);
        state.sequence = state.sequence.saturating_add(1);
    }

    /// Charges and retains newly escaping grant-scoped live bytes.
    ///
    /// Clones share one retention and the final drop releases it. Unlike an
    /// ownership transfer, this is a new charge and therefore must fit the
    /// immutable grant before the retention is created.
    pub(crate) fn retain_live_reservation(&self, bytes: u64) -> Result<Option<MemoryGrantLiveReservation>> {
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
    pub(crate) fn transfer_live_reservation(&self, bytes: u64) -> Option<MemoryGrantLiveReservation> {
        (bytes > 0).then(|| MemoryGrantLiveReservation {
            inner: Arc::new(MemoryGrantLiveReservationInner { grant: self.clone(), bytes }),
        })
    }
}

fn validate_reservation_arena_identity(
    grant: &MemoryGrant,
    runtime: &cudf::Runtime,
    arena: &cudf::ReservationArena,
    operation: &'static str,
) -> Result<()> {
    if arena.device().ordinal() != runtime.device().ordinal() {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvariantViolation,
            operation,
            "grant reservation arena device does not match execution runtime device",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("arena_device_ordinal", arena.device().ordinal())
        .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
    }
    let usage = arena.usage().map_err(|error| {
        QueryError::from_cudf(operation, "failed to verify grant reservation arena identity", error)
            .with_fact("grant_id", grant.grant_id())
    })?;
    if usage.arena_id != grant.grant_id() || usage.allocation_limit_bytes != grant.budget_bytes() {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvariantViolation,
            operation,
            "grant reservation arena does not match immutable grant",
        )
        .with_fact("grant_id", grant.grant_id())
        .with_fact("arena_id", usage.arena_id)
        .with_fact("grant_bytes", grant.budget_bytes())
        .with_fact("allocation_limit_bytes", usage.allocation_limit_bytes));
    }
    Ok(())
}

impl MemoryGrantLiveReservation {
    /// Bytes retained by this handle in the owning memory grant.
    pub(crate) fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Whether this retention consumes capacity from `grant`.
    pub(crate) fn belongs_to(&self, grant: &MemoryGrant) -> bool {
        self.inner.grant.identity_key() == grant.identity_key()
    }

    /// Retains a derived reservation for output bytes produced under the
    /// retained footprint.
    ///
    /// Immutable grants cannot grow to cover derived bytes, so the derived
    /// retention must fit inside the grant's remaining live-reservation
    /// capacity or the retention fails with a structured reservation error.
    pub(crate) fn retain_derived_live_reservation(&self, bytes: u64) -> Result<Option<Self>> {
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

impl Drop for MemoryGrantLiveReservationInner {
    fn drop(&mut self) {
        self.grant.record_live_reservation_delta(-i128::from(self.bytes));
    }
}

impl fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("grant_id", &self.inner.grant_id)
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
