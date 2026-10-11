//! Accounted cuGraph resident-cache subdomain.
//!
//! The cache has no local byte budget, copy limit, reservation-id range, or
//! build singleflight. Physical allocation belongs to the one device-scoped
//! [`cudf::CacheAllocationDomain`], while every retained graph owns exactly
//! one `CacheResidency` authority charged to the device ledger.

use std::{
    collections::HashMap,
    fmt,
    marker::PhantomData,
    mem,
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::Instant,
};

use cudf::runtime::{DeviceHealthStatus, device_health_snapshot, quarantine_device_resource};
use cugraph::{ParkedGraph, Transposed, Transposition};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{GraphResidencyOrientation, ResidentGraphKey},
};

use super::{CacheGraphActivity, CacheResidency, CacheResidencyChargeTransaction, CacheResidencyPin};
use crate::{native_error::cudf_query_error, runtime::gpu::DeviceLedger};

const OPERATION: &str = "query_engine.runtime.cache.resident_graph";

/// Point-in-time graph-cache state. Host singleflight counters deliberately do
/// not belong here: request coalescing is not resident GPU state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GpuResidentGraphCacheSnapshot {
    pub(crate) available_entries: usize,
    pub(crate) checked_out_entries: usize,
    pub(crate) closed: bool,
}

/// Result of attempting to retain a newly built graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum GpuResidentGraphCacheInsert {
    Inserted,
    AlreadyPresent,
    Closed,
}

/// One parked graph in either supported cuGraph orientation.
#[doc(hidden)]
pub enum GpuResidentGraphEntry {
    Forward(ParkedGraph<cugraph::Forward>),
    Transposed(ParkedGraph<Transposed>),
}

/// Orientation-specific conversion between an entry and a typed parked graph.
#[doc(hidden)]
pub trait GpuResidentGraphTransposition: Transposition + Send + Sized + 'static {
    fn orientation() -> GraphResidencyOrientation;
    fn into_entry(parked: ParkedGraph<Self>) -> GpuResidentGraphEntry;
    fn take_entry(entry: GpuResidentGraphEntry) -> Option<ParkedGraph<Self>>;
}

impl GpuResidentGraphTransposition for cugraph::Forward {
    fn orientation() -> GraphResidencyOrientation {
        GraphResidencyOrientation::Forward
    }

    fn into_entry(parked: ParkedGraph<Self>) -> GpuResidentGraphEntry {
        GpuResidentGraphEntry::Forward(parked)
    }

    fn take_entry(entry: GpuResidentGraphEntry) -> Option<ParkedGraph<Self>> {
        match entry {
            GpuResidentGraphEntry::Forward(parked) => Some(parked),
            GpuResidentGraphEntry::Transposed(_) => None,
        }
    }
}

impl GpuResidentGraphTransposition for Transposed {
    fn orientation() -> GraphResidencyOrientation {
        GraphResidencyOrientation::Transposed
    }

    fn into_entry(parked: ParkedGraph<Self>) -> GpuResidentGraphEntry {
        GpuResidentGraphEntry::Transposed(parked)
    }

    fn take_entry(entry: GpuResidentGraphEntry) -> Option<ParkedGraph<Self>> {
        match entry {
            GpuResidentGraphEntry::Transposed(parked) => Some(parked),
            GpuResidentGraphEntry::Forward(_) => None,
        }
    }
}

struct GpuResidentGraphCacheEntry {
    graph: Mutex<Option<GpuResidentGraphEntry>>,
    /// One graph-specific child tracker and exact runtime binding. It remains
    /// live through `ParkedGraph` destruction so teardown can prove no scoped
    /// allocation survives before releasing the ledger charge.
    allocation_scope: Arc<cudf::CacheAllocationScope>,
    retained_scope_usage: cudf::DeviceMemoryResourceUsage,
    residency: CacheResidency,
}

impl GpuResidentGraphCacheEntry {
    fn take_graph(&self) -> Result<GpuResidentGraphEntry> {
        self.graph.lock().map_err(|_error| resident_graph_cache_poisoned("take_graph"))?.take().ok_or_else(|| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "resident graph entry has no parked graph")
                .with_fact("reason", "resident_graph_entry_missing_parked_graph")
        })
    }

    fn restore_graph(&self, graph: GpuResidentGraphEntry) -> Result<()> {
        let mut parked = self.graph.lock().map_err(|_error| resident_graph_cache_poisoned("restore_graph"))?;
        if parked.replace(graph).is_some() {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident graph entry already holds a parked graph",
            )
            .with_fact("reason", "resident_graph_entry_already_parked"));
        }
        Ok(())
    }
}

struct GpuResidentGraphPreparationPinState {
    id: u64,
    pin: CacheResidencyPin,
}

enum GpuResidentGraphCheckedOutState {
    /// The checkout guard owns the pin and has not escaped the parked graph.
    Active(Arc<GpuResidentGraphCacheEntry>),
    /// A caller extracted the parked graph and dropped its checkout guard.
    /// Retaining the pin is intentional: the cache cannot observe later
    /// graph destruction, so releasing the ledger charge would undercount.
    /// Both fields are keep-alive only: `_pin` retains the ledger charge and
    /// `_entry` keeps the entry's residency alive until the cache state drops.
    Escaped { _entry: Arc<GpuResidentGraphCacheEntry>, _pin: CacheResidencyPin },
}

#[cfg_attr(not(test), derive(Default))]
struct GpuResidentGraphCacheState {
    available: HashMap<ResidentGraphKey, Arc<GpuResidentGraphCacheEntry>>,
    checked_out: HashMap<ResidentGraphKey, GpuResidentGraphCheckedOutState>,
    preparation_pins: HashMap<ResidentGraphKey, GpuResidentGraphPreparationPinState>,
    #[cfg(test)]
    next_preparation_pin_id: u64,
    destructions_in_progress: usize,
    lookup_hits: u64,
    lookup_misses: u64,
    insertions: u64,
    duplicate_inserts: u64,
    closed: bool,
}

#[cfg(test)]
impl Default for GpuResidentGraphCacheState {
    fn default() -> Self {
        Self {
            available: HashMap::new(),
            checked_out: HashMap::new(),
            preparation_pins: HashMap::new(),
            #[cfg(test)]
            next_preparation_pin_id: 1,
            destructions_in_progress: 0,
            lookup_hits: 0,
            lookup_misses: 0,
            insertions: 0,
            duplicate_inserts: 0,
            closed: false,
        }
    }
}

impl GpuResidentGraphCacheState {
    #[cfg(test)]
    fn next_preparation_pin_id(&mut self) -> u64 {
        let next = self.next_preparation_pin_id;
        self.next_preparation_pin_id = self.next_preparation_pin_id.wrapping_add(1);
        if self.next_preparation_pin_id == 0 {
            self.next_preparation_pin_id = 1;
        }
        next
    }
}

struct GpuResidentGraphCacheInner {
    ledger: DeviceLedger,
    allocation_domain: cudf::CacheAllocationDomain,
    state: Mutex<GpuResidentGraphCacheState>,
    changed: Condvar,
}

/// Per-device cuGraph resident-cache subdomain.
///
/// A cache instance owns the one shared device allocation domain. Each graph
/// entry retains its own child-tracked scope and therefore its own exact
/// runtime binding; source and graph allocations still share the domain's
/// aggregate hard cap.
#[doc(hidden)]
pub struct GpuResidentGraphCache {
    inner: Arc<GpuResidentGraphCacheInner>,
}

impl Clone for GpuResidentGraphCache {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl fmt::Debug for GpuResidentGraphCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = self.snapshot().ok();
        f.debug_struct("GpuResidentGraphCache")
            .field("device_ordinal", &self.inner.ledger.device_ordinal())
            .field("cache_allocation_domain_id", &self.cache_allocation_domain_id().get())
            .field("snapshot", &snapshot)
            .finish()
    }
}

impl GpuResidentGraphCache {
    /// Attaches this graph subdomain to the device's one shared cache domain.
    pub(crate) fn new(ledger: DeviceLedger, domain: cudf::CacheAllocationDomain) -> Result<Self> {
        let device_ordinal = ledger.device_ordinal();
        if domain.device().ordinal() != device_ordinal {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "graph cache allocation domain does not match its ledger device",
            )
            .with_fact("reason", "graph_cache_domain_device_mismatch")
            .with_fact("ledger_device_ordinal", device_ordinal)
            .with_fact("domain_device_ordinal", domain.device().ordinal()));
        }
        if domain.capacity_bytes() != ledger.profile().cache_cap_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "graph cache allocation domain capacity does not match the ledger cache cap",
            )
            .with_fact("reason", "graph_cache_domain_capacity_mismatch")
            .with_fact("domain_capacity_bytes", domain.capacity_bytes())
            .with_fact("ledger_cache_cap_bytes", ledger.profile().cache_cap_bytes));
        }
        Ok(Self {
            inner: Arc::new(GpuResidentGraphCacheInner {
                ledger,
                allocation_domain: domain,
                state: Mutex::new(GpuResidentGraphCacheState::default()),
                changed: Condvar::new(),
            }),
        })
    }

    /// Stable identity of the shared source/graph cache allocation domain.
    pub(crate) fn cache_allocation_domain_id(&self) -> cudf::CacheAllocationDomainId {
        self.inner.allocation_domain.id()
    }

    /// Shared physical allocation domain used by both resident cache subdomains.
    #[cfg(test)]
    pub(crate) fn cache_allocation_domain(&self) -> &cudf::CacheAllocationDomain {
        &self.inner.allocation_domain
    }

    /// CUDA device on which every graph scope in this cache is created.
    #[doc(hidden)]
    pub fn device(&self) -> cudf::DeviceId {
        self.inner.allocation_domain.device()
    }

    /// The largest conservative graph reservation still permitted by the
    /// ledger-owned cache share (`K`). This is derived from the ledger
    /// snapshot; the graph cache never keeps a second byte counter. Live
    /// query grants never participate: cache eviction owns only `K`, and the
    /// query-share bound independently keeps the combined commitment within
    /// `M`, so either admission order promises identical capacity.
    #[doc(hidden)]
    pub fn available_reservation_bytes(&self) -> Result<u64> {
        let snapshot = self.inner.ledger.snapshot()?;
        let charged_cache_bytes =
            snapshot.cache_evictable_resident_bytes.saturating_add(snapshot.cache_pinned_in_use_bytes);
        Ok(self.inner.ledger.profile().cache_cap_bytes.saturating_sub(charged_cache_bytes))
    }

    /// A scope that does not return to its retained baseline may still own
    /// allocations beyond the cache entry's charge. Stop new admission while
    /// the execution reservation remains fail-closed in the ledger.
    #[doc(hidden)]
    pub fn close_admission_for_scope_invariant(&self) -> Result<()> {
        self.inner.ledger.close_admission_if_open("resident_graph_scope_usage_mismatch").map(|_| ())
    }

    /// A scope-local stream fence failed, so the cache cannot prove that a
    /// transient execution reservation is physically retired before reuse.
    #[doc(hidden)]
    pub fn close_admission_for_scope_retirement_fence(&self) -> Result<()> {
        self.inner.ledger.close_admission_if_open("resident_graph_scope_retirement_fence_failed").map(|_| ())
    }

    /// Pre-reserves a graph's measured retention bytes before graph construction
    /// or ownership transfer. The caller must prove candidate teardown through
    /// [`GpuResidentGraphInsertTransaction::abort_after_candidate_destroyed`]
    /// before the precharge can return; an unproven drop instead closes
    /// admission and retains it.
    #[doc(hidden)]
    pub fn begin_insert(&self, retained_device_bytes: u64) -> Result<Option<GpuResidentGraphInsertTransaction>> {
        if self.lock()?.closed {
            return Ok(None);
        }
        let Some(charge) = CacheResidency::begin_attempt_owned(self.inner.ledger.clone(), retained_device_bytes)?
        else {
            return Ok(None);
        };
        let allocation_scope = self.inner.allocation_domain.allocation_scope().map_err(|error| {
            cudf_query_error(OPERATION, "failed to create a graph-specific cache allocation scope", error)
                .with_fact("device_ordinal", self.inner.ledger.device_ordinal())
                .with_fact("cache_allocation_domain_id", self.cache_allocation_domain_id())
        })?;
        if allocation_scope.device() != self.device() {
            return Err(QueryError::new(
                QueryErrorCode::CudfInvariantViolation,
                OPERATION,
                "resident graph allocation scope does not match its shared cache domain device",
            )
            .with_fact("reason", "resident_graph_scope_device_mismatch")
            .with_fact("cache_allocation_domain_id", self.cache_allocation_domain_id())
            .with_fact("scope_device_ordinal", allocation_scope.device().ordinal())
            .with_fact("cache_device_ordinal", self.device().ordinal()));
        }
        if self.lock()?.closed {
            return Ok(None);
        }
        Ok(Some(GpuResidentGraphInsertTransaction {
            cache: Arc::clone(&self.inner),
            allocation_scope: Arc::new(allocation_scope),
            retained_scope_usage: None,
            charge: Some(charge),
        }))
    }

    /// Pins an available entry before source-side graph preparation commits to
    /// a cache hit. Once [`Self::close`] begins, no final pin is admitted.
    #[cfg(test)]
    pub(crate) fn try_pin_available_key(
        &self,
        resident_key: &ResidentGraphKey,
    ) -> Result<Option<GpuResidentGraphPreparationPin>> {
        let mut state = self.lock()?;
        if state.closed || state.preparation_pins.contains_key(resident_key) {
            return Ok(None);
        }
        let Some(entry) = state.available.get(resident_key) else {
            return Ok(None);
        };
        let Some(pin) = entry.residency.try_pin()? else {
            return Ok(None);
        };
        let id = state.next_preparation_pin_id();
        state.preparation_pins.insert(resident_key.clone(), GpuResidentGraphPreparationPinState { id, pin });
        Ok(Some(GpuResidentGraphPreparationPin {
            cache: Arc::clone(&self.inner),
            resident_key: resident_key.clone(),
            id,
            active: true,
        }))
    }

    /// Checks out one exact-orientation resident graph. The returned guard
    /// keeps the shared residency pinned until it is returned or safely
    /// destroyed; it never creates a cache-local byte charge.
    #[doc(hidden)]
    pub fn checkout<T>(
        &self,
        resident_key: &ResidentGraphKey,
        preparation_pin: Option<&mut GpuResidentGraphPreparationPin>,
    ) -> Result<Option<GpuResidentGraphCheckout<T>>>
    where
        T: GpuResidentGraphTransposition,
    {
        if resident_key.orientation != T::orientation() {
            return Ok(None);
        }
        let mut state = self.lock()?;
        if state.closed {
            return Err(resident_graph_cache_closed("checkout"));
        }
        let Some(entry) = state.available.get(resident_key).cloned() else {
            state.lookup_misses = state.lookup_misses.saturating_add(1);
            return Ok(None);
        };
        let pin = match preparation_pin {
            Some(preparation_pin) => {
                if !preparation_pin.belongs_to(&self.inner, resident_key) {
                    state.lookup_misses = state.lookup_misses.saturating_add(1);
                    return Ok(None);
                }
                let Some(prepared) = state.preparation_pins.remove(resident_key) else {
                    state.lookup_misses = state.lookup_misses.saturating_add(1);
                    return Ok(None);
                };
                if prepared.id != preparation_pin.id {
                    state.preparation_pins.insert(resident_key.clone(), prepared);
                    state.lookup_misses = state.lookup_misses.saturating_add(1);
                    return Ok(None);
                }
                preparation_pin.active = false;
                prepared.pin
            },
            None => {
                if state.preparation_pins.contains_key(resident_key) {
                    state.lookup_misses = state.lookup_misses.saturating_add(1);
                    return Ok(None);
                }
                let Some(pin) = entry.residency.try_pin()? else {
                    state.lookup_misses = state.lookup_misses.saturating_add(1);
                    return Ok(None);
                };
                pin
            },
        };
        let graph = entry.take_graph()?;
        let Some(parked) = T::take_entry(graph) else {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident graph entry orientation does not match its canonical key",
            )
            .with_fact("reason", "resident_graph_orientation_mismatch"));
        };
        let removed = state.available.remove(resident_key);
        debug_assert!(removed.is_some());
        state.checked_out.insert(resident_key.clone(), GpuResidentGraphCheckedOutState::Active(Arc::clone(&entry)));
        state.lookup_hits = state.lookup_hits.saturating_add(1);
        Ok(Some(GpuResidentGraphCheckout {
            cache: Arc::clone(&self.inner),
            resident_key: resident_key.clone(),
            entry,
            pin: Some(pin),
            parked: Some(parked),
            completed: false,
            _transposition: PhantomData,
        }))
    }

    /// Begins shutdown. Existing pins may finish, but no final preparation pin,
    /// checkout, or insertion can start after this point.
    #[doc(hidden)]
    pub fn close(&self) -> Result<()> {
        let mut state = self.lock()?;
        state.closed = true;
        self.inner.changed.notify_all();
        Ok(())
    }

    /// Closes and drains all available graphs before `deadline`. Checked-out or
    /// already-pinned entries retain their charges until their owners return;
    /// expiry therefore reports a timeout with accounting intact.
    pub(crate) fn close_and_drain_until(&self, deadline: Instant) -> Result<()> {
        self.close()?;
        let mut primary_error = None;
        loop {
            let entries = {
                let mut state = self.lock()?;
                let keys = state
                    .available
                    .iter()
                    .filter(|(key, _entry)| !state.preparation_pins.contains_key(*key))
                    .map(|(key, _entry)| key.clone())
                    .collect::<Vec<_>>();
                let entries = keys.into_iter().filter_map(|key| state.available.remove(&key)).collect::<Vec<_>>();
                state.destructions_in_progress = state.destructions_in_progress.saturating_add(entries.len());
                entries
            };
            for entry in entries {
                let outcome = entry.take_graph().and_then(|graph| self.destroy_entry_after_graph_drop(entry, graph));
                self.finish_destruction();
                match outcome {
                    Ok(_) => {},
                    Err(error) if primary_error.is_none() => primary_error = Some(error),
                    Err(_) => {},
                }
            }
            let mut state = self.lock()?;
            if state.available.is_empty()
                && state.checked_out.is_empty()
                && state.preparation_pins.is_empty()
                && state.destructions_in_progress == 0
            {
                return primary_error.map_or(Ok(()), Err);
            }
            if Instant::now() >= deadline {
                return Err(resident_graph_cache_drain_timeout());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let (next, _timeout) = self.inner.changed.wait_timeout(state, remaining).map_err(|_error| {
                QueryError::new(QueryErrorCode::Internal, OPERATION, "resident graph cache drain mutex was poisoned")
                    .with_fact("reason", "resident_graph_cache_drain_mutex_poisoned")
            })?;
            state = next;
            // Discard the re-acquired guard; the next iteration locks the cache state again.
            drop(state);
        }
    }

    pub(crate) fn snapshot(&self) -> Result<GpuResidentGraphCacheSnapshot> {
        let state = self.lock()?;
        Ok(GpuResidentGraphCacheSnapshot {
            available_entries: state.available.len(),
            checked_out_entries: state.checked_out.len(),
            closed: state.closed,
        })
    }

    pub(crate) fn activity(&self) -> Result<CacheGraphActivity> {
        let state = self.lock()?;
        Ok(CacheGraphActivity {
            available_entries: state.available.len(),
            checked_out_entries: state.checked_out.len(),
            preparation_pins: state.preparation_pins.len(),
            lookup_hits: state.lookup_hits,
            lookup_misses: state.lookup_misses,
            insertions: state.insertions,
            duplicate_inserts: state.duplicate_inserts,
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, GpuResidentGraphCacheState>> {
        self.inner.state.lock().map_err(|_error| resident_graph_cache_poisoned("lock"))
    }

    fn destroy_entry_after_graph_drop(
        &self,
        entry: Arc<GpuResidentGraphCacheEntry>,
        graph: GpuResidentGraphEntry,
    ) -> Result<bool> {
        destroy_entry_after_graph_drop(&self.inner, entry, graph)
    }

    fn finish_destruction(&self) {
        finish_destruction(&self.inner);
    }
}

/// Linear reservation used while a caller constructs or promotes a graph into
/// the shared cache allocation domain.
#[must_use = "an insert transaction must fence candidate teardown before returning its ledger precharge"]
#[doc(hidden)]
pub struct GpuResidentGraphInsertTransaction {
    cache: Arc<GpuResidentGraphCacheInner>,
    allocation_scope: Arc<cudf::CacheAllocationScope>,
    retained_scope_usage: Option<cudf::DeviceMemoryResourceUsage>,
    charge: Option<CacheResidencyChargeTransaction>,
}

impl GpuResidentGraphInsertTransaction {
    #[cfg(all(test, feature = "cugraph"))]
    #[doc(hidden)]
    pub fn bytes(&self) -> u64 {
        self.charge.as_ref().map_or(0, CacheResidencyChargeTransaction::bytes)
    }

    /// Exact runtime binding for this prospective graph. Retain the matching
    /// scope with the parked graph; sibling cache work is deliberately absent
    /// from this runtime's usage counter.
    #[doc(hidden)]
    pub fn runtime(&self) -> Arc<cudf::Runtime> {
        Arc::new(self.allocation_scope.runtime().clone())
    }

    /// Returns the scope that the caller passes to `begin_cache_scope_allocator_correlation`
    /// for this prospective cache entry. The caller must finish that correlation window
    /// only after all scope-owned temporaries have been destroyed, while this transaction
    /// still keeps the scope alive.
    #[doc(hidden)]
    pub fn allocation_scope(&self) -> Arc<cudf::CacheAllocationScope> {
        Arc::clone(&self.allocation_scope)
    }

    /// Returns the exact high-water mark from this candidate graph's private
    /// allocation scope. The caller must first finish the scope's work so the
    /// value covers every allocation from the physical graph build.
    #[doc(hidden)]
    pub fn scope_peak_bytes(&self) -> Result<u64> {
        self.scope_usage().map(|usage| usage.peak_bytes)
    }

    /// Reconciles the conservative pre-reservation to this graph scope's live
    /// bytes. The scope excludes sibling source/graph cache mutations, so this
    /// remains exact even while the shared allocation domain is under churn.
    #[doc(hidden)]
    pub fn shrink_to_scope_usage(&mut self) -> Result<bool> {
        self.synchronize_scope()?;
        let usage = self.scope_usage()?;
        let shrunk = self
            .charge
            .as_mut()
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    OPERATION,
                    "resident graph insertion transaction was already consumed",
                )
                .with_fact("reason", "resident_graph_insert_transaction_consumed")
            })?
            .shrink_to(usage.current_bytes)?;
        if shrunk {
            self.retained_scope_usage = Some(usage);
        }
        Ok(shrunk)
    }

    #[doc(hidden)]
    pub fn insert_if_absent<T>(
        mut self,
        resident_key: ResidentGraphKey,
        parked: ParkedGraph<T>,
    ) -> Result<GpuResidentGraphCacheInsert>
    where
        T: GpuResidentGraphTransposition,
    {
        let expected_device = self.allocation_scope.device().ordinal();
        let graph_device = parked.device();
        if graph_device != i32::try_from(expected_device).unwrap_or(i32::MAX) {
            // Destroy the native candidate before proving its scope empty and returning the precharge.
            drop(parked);
            let error = QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "parked graph device does not match the shared cache allocation domain",
            )
            .with_fact("reason", "resident_graph_device_mismatch")
            .with_fact("graph_device_ordinal", graph_device)
            .with_fact("cache_device_ordinal", expected_device);
            self.abort_after_candidate_destroyed()?;
            return Err(error);
        }
        if resident_key.orientation != T::orientation() {
            // Destroy the native candidate before proving its scope empty and returning the precharge.
            drop(parked);
            let error = QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "parked graph orientation does not match the resident graph key",
            )
            .with_fact("reason", "resident_graph_key_orientation_mismatch");
            self.abort_after_candidate_destroyed()?;
            return Err(error);
        }
        let Some(retained_scope_usage) = self.retained_scope_usage else {
            // Destroy the native candidate before proving its scope empty and returning the precharge.
            drop(parked);
            let error = QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "resident graph insertion was not reconciled to its allocation scope",
            )
            .with_fact("reason", "resident_graph_scope_usage_not_reconciled");
            self.abort_after_candidate_destroyed()?;
            return Err(error);
        };
        let current_scope_usage = match self.scope_usage() {
            Ok(usage) => usage,
            Err(error) => {
                // Destroy the native candidate before its unmeasurable scope is retained fail-closed.
                drop(parked);
                let _ = self.abort_after_candidate_destroyed();
                return Err(error);
            },
        };
        if current_scope_usage.current_bytes != retained_scope_usage.current_bytes
            || current_scope_usage.current_aligned_bytes != retained_scope_usage.current_aligned_bytes
            || current_scope_usage.current_allocations != retained_scope_usage.current_allocations
        {
            // Destroy the native candidate before proving its scope empty and returning the precharge.
            drop(parked);
            let error = scope_usage_changed_after_reconciliation(
                retained_scope_usage,
                current_scope_usage,
                self.cache.allocation_domain.id(),
            );
            self.abort_after_candidate_destroyed()?;
            return Err(error);
        }
        let mut state = self.cache.state.lock().map_err(|_error| resident_graph_cache_poisoned("insert"))?;
        if state.closed {
            // Do not hold the cache mutex across native graph destruction or scope teardown.
            drop(state);
            // Destroy the native candidate before proving its scope empty and returning the precharge.
            drop(parked);
            self.abort_after_candidate_destroyed()?;
            return Ok(GpuResidentGraphCacheInsert::Closed);
        }
        if state.available.contains_key(&resident_key) || state.checked_out.contains_key(&resident_key) {
            state.duplicate_inserts = state.duplicate_inserts.saturating_add(1);
            // Do not hold the cache mutex across native graph destruction or scope teardown.
            drop(state);
            // Destroy the native candidate before proving its scope empty and returning the precharge.
            drop(parked);
            self.abort_after_candidate_destroyed()?;
            return Ok(GpuResidentGraphCacheInsert::AlreadyPresent);
        }
        let charge = self.charge.take().expect("resident graph insertion transaction is consumed once");
        let entry = Arc::new(GpuResidentGraphCacheEntry {
            graph: Mutex::new(Some(T::into_entry(parked))),
            allocation_scope: Arc::clone(&self.allocation_scope),
            retained_scope_usage,
            residency: charge.commit_evictable(),
        });
        state.available.insert(resident_key, entry);
        state.insertions = state.insertions.saturating_add(1);
        self.cache.changed.notify_all();
        Ok(GpuResidentGraphCacheInsert::Inserted)
    }

    fn scope_usage(&self) -> Result<cudf::DeviceMemoryResourceUsage> {
        self.allocation_scope.usage().map_err(|error| {
            cudf_query_error(OPERATION, "failed to read resident graph allocation-scope usage", error)
                .with_fact("cache_allocation_domain_id", self.allocation_scope.domain_id())
        })
    }

    /// The graph-construction stream is unique to this scope. It must retire
    /// before a logical usage snapshot can authorize release of any cache
    /// precharge; accounting counters alone are not a CUDA completion proof.
    fn synchronize_scope(&self) -> Result<()> {
        self.allocation_scope.synchronize().map_err(|error| {
            cudf_query_error(OPERATION, "failed to fence resident graph allocation scope before reconciliation", error)
                .with_fact("reason", "resident_graph_scope_retirement_fence_failed")
                .with_fact("cache_allocation_domain_id", self.allocation_scope.domain_id())
        })
    }

    /// Releases a prospective graph-cache precharge only after all candidate
    /// objects and companion runtime handles have been destroyed.  The graph
    /// builder must call this on every non-commit path; dropping the
    /// transaction without this proof instead closes admission and retains the
    /// charge.
    #[doc(hidden)]
    pub fn abort_after_candidate_destroyed(mut self) -> Result<()> {
        let Some(charge) = self.charge.take() else {
            return Ok(());
        };
        if let Err(error) = self.synchronize_scope() {
            charge.retain_charge_after_unsafe_teardown();
            self.cache.ledger.close_admission_if_open("resident_graph_scope_retirement_fence_failed")?;
            return Err(error);
        }
        let device = self.allocation_scope.device();
        let health = device_health_snapshot(device);
        let scope_usage = self.scope_usage();
        let usage_is_empty = scope_usage.as_ref().is_ok_and(|usage| {
            usage.current_bytes == 0 && usage.current_aligned_bytes == 0 && usage.current_allocations == 0
        });
        if health.status() != DeviceHealthStatus::Healthy || !usage_is_empty {
            charge.retain_charge_after_unsafe_teardown();
            return self.fail_closed_candidate_teardown(
                "resident graph candidate teardown could not prove an empty healthy allocation scope",
                scope_usage.ok(),
            );
        }
        if let Err(error) = charge.release_after_proven_empty() {
            let _ = self.cache.ledger.close_admission_if_open("resident_graph_candidate_charge_release_failed");
            return Err(error);
        }
        Ok(())
    }

    fn fail_closed_candidate_teardown(
        &self,
        message: &'static str,
        scope_usage: Option<cudf::DeviceMemoryResourceUsage>,
    ) -> Result<()> {
        self.cache.ledger.close_admission_if_open("resident_graph_candidate_teardown_unproven")?;
        let mut error = QueryError::new(QueryErrorCode::CudfPersistentFailure, OPERATION, message)
            .with_fact("reason", "resident_graph_candidate_teardown_unproven")
            .with_fact("device_ordinal", self.allocation_scope.device().ordinal())
            .with_fact("cache_allocation_domain_id", self.allocation_scope.domain_id());
        if let Some(usage) = scope_usage {
            error = error
                .with_fact("scope_current_bytes", usage.current_bytes)
                .with_fact("scope_current_aligned_bytes", usage.current_aligned_bytes)
                .with_fact("scope_current_allocations", usage.current_allocations);
        }
        Err(error)
    }
}

impl Drop for GpuResidentGraphInsertTransaction {
    fn drop(&mut self) {
        let Some(charge) = self.charge.take() else {
            return;
        };
        // This destructor runs after later graph-builder locals have dropped.
        // It may therefore return the precharge only when the exact scope
        // stream fences successfully and reports no surviving allocation;
        // every other path retains the charge and stops admission.
        let scope_fenced = self.allocation_scope.synchronize().is_ok();
        let scope_empty = self.allocation_scope.usage().is_ok_and(|usage| {
            usage.current_bytes == 0 && usage.current_aligned_bytes == 0 && usage.current_allocations == 0
        });
        let healthy = device_health_snapshot(self.allocation_scope.device()).status() == DeviceHealthStatus::Healthy;
        if scope_fenced && scope_empty && healthy {
            if charge.release_after_proven_empty().is_err() {
                let _ =
                    self.cache.ledger.close_admission_if_open("resident_graph_insert_transaction_drop_release_failed");
            }
            return;
        }
        charge.retain_charge_after_unsafe_teardown();
        let _ = self
            .cache
            .ledger
            .close_admission_if_open("resident_graph_insert_transaction_dropped_without_teardown_proof");
    }
}

/// A preparation pin that reserves an exact resident graph for its eventual
/// checkout owner. Dropping it returns the entry to evictable state.
#[must_use = "dropping the preparation pin releases its shared residency pin"]
#[doc(hidden)]
pub struct GpuResidentGraphPreparationPin {
    cache: Arc<GpuResidentGraphCacheInner>,
    resident_key: ResidentGraphKey,
    id: u64,
    active: bool,
}

impl GpuResidentGraphPreparationPin {
    fn belongs_to(&self, cache: &Arc<GpuResidentGraphCacheInner>, resident_key: &ResidentGraphKey) -> bool {
        self.active && Arc::ptr_eq(&self.cache, cache) && self.resident_key == *resident_key
    }
}

impl Drop for GpuResidentGraphPreparationPin {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Ok(mut state) = self.cache.state.lock()
            && state.preparation_pins.get(&self.resident_key).is_some_and(|pin| pin.id == self.id)
        {
            state.preparation_pins.remove(&self.resident_key);
            self.cache.changed.notify_all();
        }
    }
}

/// Exclusive typed checkout of a parked resident graph.
#[doc(hidden)]
pub struct GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    cache: Arc<GpuResidentGraphCacheInner>,
    resident_key: ResidentGraphKey,
    entry: Arc<GpuResidentGraphCacheEntry>,
    pin: Option<CacheResidencyPin>,
    parked: Option<ParkedGraph<T>>,
    completed: bool,
    _transposition: PhantomData<T>,
}

impl<T> GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    /// Keeps the exact cache allocation scope alive while an exclusive
    /// checkout contributes allocator-event evidence for one attempt.
    #[doc(hidden)]
    pub fn allocation_scope(&self) -> Arc<cudf::CacheAllocationScope> {
        Arc::clone(&self.entry.allocation_scope)
    }

    /// Returns the exact high-water mark for this resident graph's exclusive
    /// cache scope. The checkout remains pinned while it is measured, so no
    /// other graph execution can contribute to this scope's report.
    #[doc(hidden)]
    pub fn scope_peak_bytes(&self) -> Result<u64> {
        self.entry.allocation_scope.usage().map(|usage| usage.peak_bytes).map_err(|error| {
            cudf_query_error(OPERATION, "failed to read checked-out resident graph allocation-scope peak", error)
                .with_fact("cache_allocation_domain_id", self.entry.allocation_scope.domain_id())
        })
    }

    /// Fences the exact private stream that executed this checkout. The cache
    /// must retain transient ledger ownership until this returns successfully.
    #[doc(hidden)]
    pub fn synchronize_scope(&self) -> Result<()> {
        self.entry.allocation_scope.synchronize().map_err(|error| {
            cudf_query_error(OPERATION, "failed to fence checked-out resident graph allocation scope", error)
                .with_fact("reason", "resident_graph_scope_retirement_fence_failed")
                .with_fact("cache_allocation_domain_id", self.entry.allocation_scope.domain_id())
        })
    }

    /// Requires cached execution to return the graph-specific allocation scope
    /// to the exact retained baseline before its transient ledger reservation
    /// may be released.
    pub(crate) fn verify_retained_scope_usage(&self) -> Result<()> {
        let current = self.entry.allocation_scope.usage().map_err(|error| {
            cudf_query_error(OPERATION, "failed to read checked-out resident graph allocation-scope usage", error)
                .with_fact("cache_allocation_domain_id", self.entry.allocation_scope.domain_id())
        })?;
        let retained = self.entry.retained_scope_usage;
        if current.current_bytes != retained.current_bytes
            || current.current_aligned_bytes != retained.current_aligned_bytes
            || current.current_allocations != retained.current_allocations
        {
            return Err(scope_usage_changed_after_reconciliation(
                retained,
                current,
                self.entry.allocation_scope.domain_id(),
            ));
        }
        Ok(())
    }

    /// Proves that a re-parked graph may return to the available cache before
    /// its checkout correlation window closes. The private-stream fence must
    /// already have completed; this checks both the retained allocation
    /// baseline and current device health while the checkout is still
    /// exclusive.
    #[doc(hidden)]
    pub fn verify_reparked_scope_for_return(&self) -> Result<()> {
        self.verify_retained_scope_usage()?;
        let device = self.entry.allocation_scope.device();
        if device_health_snapshot(device).status() != DeviceHealthStatus::Healthy {
            return Err(QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                OPERATION,
                "resident graph allocation scope device is unhealthy after re-park",
            )
            .with_fact("reason", "resident_graph_scope_device_unhealthy_after_repark")
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("cache_allocation_domain_id", self.entry.allocation_scope.domain_id()));
        }
        Ok(())
    }

    /// Extracts the parked graph for a cuGraph execution. The caller must
    /// return it with [`Self::return_parked`]; dropping the checkout after an
    /// extraction conservatively leaves the shared charge pinned, so the cache
    /// drain cannot finish.
    #[doc(hidden)]
    pub fn take_parked(&mut self) -> ParkedGraph<T> {
        self.parked.take().expect("resident graph checkout already consumed")
    }

    #[doc(hidden)]
    pub fn return_parked(mut self, parked: ParkedGraph<T>) -> Result<()> {
        self.complete_with_parked(parked)
    }

    fn complete_with_parked(&mut self, parked: ParkedGraph<T>) -> Result<()> {
        let expected_device = self.entry.allocation_scope.device().ordinal();
        let graph_device = parked.device();
        if graph_device != i32::try_from(expected_device).unwrap_or(i32::MAX) {
            // Destroy the native graph before escape_without_graph may release the pin on error.
            drop(parked);
            self.escape_without_graph()?;
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "returned parked graph device does not match the shared cache allocation domain",
            )
            .with_fact("reason", "resident_graph_return_device_mismatch")
            .with_fact("graph_device_ordinal", graph_device)
            .with_fact("cache_device_ordinal", expected_device));
        }
        let closed = {
            let mut state = self.cache.state.lock().map_err(|_error| resident_graph_cache_poisoned("return"))?;
            match state.checked_out.remove(&self.resident_key) {
                Some(GpuResidentGraphCheckedOutState::Active(entry)) if Arc::ptr_eq(&entry, &self.entry) => {},
                Some(other) => {
                    state.checked_out.insert(self.resident_key.clone(), other);
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        OPERATION,
                        "resident graph checkout ownership does not match cache state",
                    )
                    .with_fact("reason", "resident_graph_checkout_owner_mismatch"));
                },
                None => {
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        OPERATION,
                        "resident graph checkout is not registered",
                    )
                    .with_fact("reason", "resident_graph_checkout_missing"));
                },
            }
            let closed = state.closed;
            if !closed {
                self.entry.restore_graph(T::into_entry(parked))?;
                state.available.insert(self.resident_key.clone(), Arc::clone(&self.entry));
                self.cache.changed.notify_all();
                self.completed = true;
                let pin = self.pin.take();
                // Release the cache state first: the pin drop can run ledger transition observers.
                drop(state);
                drop(pin);
                return Ok(());
            }
            state.destructions_in_progress = state.destructions_in_progress.saturating_add(1);
            self.cache.changed.notify_all();
            closed
        };
        self.completed = true;
        let pin = self.pin.take();
        // Unpin the entry: destroy_entry_after_graph_drop begins eviction only when unpinned.
        drop(pin);
        debug_assert!(closed);
        let outcome = destroy_entry_after_graph_drop(&self.cache, Arc::clone(&self.entry), T::into_entry(parked));
        finish_destruction(&self.cache);
        outcome?;
        Ok(())
    }

    fn escape_without_graph(&mut self) -> Result<()> {
        let Some(pin) = self.pin.take() else {
            return Ok(());
        };
        let mut state = self.cache.state.lock().map_err(|_error| resident_graph_cache_poisoned("escape"))?;
        match state.checked_out.remove(&self.resident_key) {
            Some(GpuResidentGraphCheckedOutState::Active(entry)) if Arc::ptr_eq(&entry, &self.entry) => {
                state.checked_out.insert(
                    self.resident_key.clone(),
                    GpuResidentGraphCheckedOutState::Escaped { _entry: entry, _pin: pin },
                );
            },
            Some(previous) => {
                state.checked_out.insert(self.resident_key.clone(), previous);
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    OPERATION,
                    "resident graph escaped checkout does not match cache state",
                )
                .with_fact("reason", "resident_graph_escape_conflict"));
            },
            None => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    OPERATION,
                    "resident graph escaped checkout is not registered",
                )
                .with_fact("reason", "resident_graph_escape_missing"));
            },
        }
        self.completed = true;
        self.cache.changed.notify_all();
        Ok(())
    }
}

impl<T> Drop for GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if let Some(parked) = self.parked.take() {
            if self.complete_with_parked(parked).is_err() {
                let _ = self.escape_without_graph();
            }
        } else {
            let _ = self.escape_without_graph();
        }
    }
}

fn destroy_entry_after_graph_drop(
    cache: &Arc<GpuResidentGraphCacheInner>,
    entry: Arc<GpuResidentGraphCacheEntry>,
    graph: GpuResidentGraphEntry,
) -> Result<bool> {
    let Some(destruction) = entry.residency.begin_eviction()? else {
        return Err(QueryError::new(
            QueryErrorCode::Internal,
            OPERATION,
            "resident graph destruction was requested for a pinned or destroyed entry",
        )
        .with_fact("reason", "resident_graph_destroy_state_invalid"));
    };
    let device = entry.allocation_scope.device();
    let before = device_health_snapshot(device);
    if before.status() != DeviceHealthStatus::Healthy {
        mem::forget(graph);
        let reason = "device was already unhealthy before resident graph teardown";
        quarantine_device_resource(device, OPERATION, &reason);
        destruction.retain_charge_after_failed_destruction()?;
        return record_teardown_quarantine(cache, entry.residency.bytes(), reason).map(|_| false);
    }

    // `ParkedGraph::drop` synchronizes its park event before freeing native
    // graph state. Only after that destructor returns may the ledger charge be
    // committed as released.
    drop(graph);
    if entry.allocation_scope.synchronize().is_err() {
        destruction.retain_charge_after_failed_destruction()?;
        return record_teardown_quarantine(
            cache,
            entry.residency.bytes(),
            "resident graph allocation scope retirement fence failed after graph destruction",
        )
        .map(|_| false);
    }
    let after = device_health_snapshot(device);
    if after.status() != DeviceHealthStatus::Healthy
        || after.quarantined_teardown_count() > before.quarantined_teardown_count()
    {
        destruction.retain_charge_after_failed_destruction()?;
        return record_teardown_quarantine(
            cache,
            entry.residency.bytes(),
            "resident graph teardown quarantined its device",
        )
        .map(|_| false);
    }
    let scope_usage = match entry.allocation_scope.usage() {
        Ok(usage) => usage,
        Err(_error) => {
            destruction.retain_charge_after_failed_destruction()?;
            return record_teardown_quarantine(
                cache,
                entry.residency.bytes(),
                "resident graph allocation scope could not be measured after graph destruction",
            )
            .map(|_| false);
        },
    };
    if scope_usage.current_bytes != 0 || scope_usage.current_aligned_bytes != 0 || scope_usage.current_allocations != 0
    {
        destruction.retain_charge_after_failed_destruction()?;
        return record_teardown_quarantine(
            cache,
            entry.residency.bytes(),
            "resident graph allocation scope retained bytes after graph destruction",
        )
        .map(|_| false);
    }
    destruction.commit_after_destroy()?;
    Ok(true)
}

fn scope_usage_changed_after_reconciliation(
    retained: cudf::DeviceMemoryResourceUsage,
    current: cudf::DeviceMemoryResourceUsage,
    allocation_domain_id: cudf::CacheAllocationDomainId,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::CudfInvariantViolation,
        OPERATION,
        "resident graph allocation scope did not return to its retained baseline",
    )
    .with_fact("reason", "resident_graph_scope_usage_mismatch")
    .with_fact("cache_allocation_domain_id", allocation_domain_id)
    .with_fact("retained_current_bytes", retained.current_bytes)
    .with_fact("current_bytes", current.current_bytes)
    .with_fact("retained_current_aligned_bytes", retained.current_aligned_bytes)
    .with_fact("current_aligned_bytes", current.current_aligned_bytes)
    .with_fact("retained_current_allocations", retained.current_allocations)
    .with_fact("current_allocations", current.current_allocations)
}

fn record_teardown_quarantine(cache: &Arc<GpuResidentGraphCacheInner>, bytes: u64, reason: &'static str) -> Result<()> {
    cache.ledger.close_admission(reason)?;
    cache.changed.notify_all();
    Err(QueryError::new(
        QueryErrorCode::CudfPersistentFailure,
        OPERATION,
        "resident graph teardown was quarantined; the ledger charge remains retained",
    )
    .with_fact("reason", "resident_graph_teardown_quarantined")
    .with_fact("device_ordinal", cache.ledger.device_ordinal())
    .with_fact("retained_device_bytes", bytes))
}

fn finish_destruction(cache: &Arc<GpuResidentGraphCacheInner>) {
    if let Ok(mut state) = cache.state.lock() {
        state.destructions_in_progress = state.destructions_in_progress.saturating_sub(1);
        cache.changed.notify_all();
    }
}

fn resident_graph_cache_closed(action: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, OPERATION, "resident graph cache is closed")
        .with_fact("reason", "resident_graph_cache_closed")
        .with_fact("action", action)
}

fn resident_graph_cache_drain_timeout() -> QueryError {
    QueryError::new(QueryErrorCode::ServiceOverloaded, OPERATION, "timed out draining resident graph cache")
        .with_fact("reason", "resident_graph_cache_drain_timeout")
        .with_cleanup_deadline_elapsed()
}

fn resident_graph_cache_poisoned(action: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Internal, OPERATION, "resident graph cache mutex was poisoned")
        .with_fact("reason", "resident_graph_cache_mutex_poisoned")
        .with_fact("action", action)
}

#[cfg(test)]
mod tests;
