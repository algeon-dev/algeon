//! Accounted cuGraph resident-cache subdomain.
//!
//! The cache has no local byte budget, copy limit, reservation-id range, or
//! build singleflight. Physical allocation belongs to the one device-scoped
//! [`cudf::CacheAllocationDomain`], while every retained graph owns exactly
//! one [`super::CacheResidency`] authority charged to the device ledger.

use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::Instant,
};

use cugraph::{ParkedGraph, Transposed, Transposition};

use super::{CacheResidency, CacheResidencyChargeTransaction, CacheResidencyPin};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{GraphResidencyOrientation, ResidentGraphKey},
    runtime::gpu::DeviceLedger,
};

const OPERATION: &str = "nexus_query_engine.runtime.cache.resident_graph";

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
pub(crate) enum GpuResidentGraphCacheInsert {
    Inserted,
    AlreadyPresent,
    CapacitySkipped,
    Closed,
}

/// One parked graph in either supported cuGraph orientation.
pub(crate) enum GpuResidentGraphEntry {
    Forward(ParkedGraph<cugraph::Forward>),
    Transposed(ParkedGraph<Transposed>),
}

/// Orientation-specific conversion between an entry and a typed parked graph.
pub(crate) trait GpuResidentGraphTransposition: Transposition + Send + Sized + 'static {
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
    Escaped { entry: Arc<GpuResidentGraphCacheEntry>, pin: CacheResidencyPin },
}

struct GpuResidentGraphCacheState {
    available: HashMap<ResidentGraphKey, Arc<GpuResidentGraphCacheEntry>>,
    checked_out: HashMap<ResidentGraphKey, GpuResidentGraphCheckedOutState>,
    preparation_pins: HashMap<ResidentGraphKey, GpuResidentGraphPreparationPinState>,
    next_preparation_pin_id: u64,
    destructions_in_progress: usize,
    closed: bool,
}

impl Default for GpuResidentGraphCacheState {
    fn default() -> Self {
        Self {
            available: HashMap::new(),
            checked_out: HashMap::new(),
            preparation_pins: HashMap::new(),
            next_preparation_pin_id: 1,
            destructions_in_progress: 0,
            closed: false,
        }
    }
}

impl GpuResidentGraphCacheState {
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
    allocation_lease: cudf::CacheAllocationLease,
    runtime: Arc<cudf::Runtime>,
    state: Mutex<GpuResidentGraphCacheState>,
    changed: Condvar,
}

/// Per-device cuGraph resident-cache subdomain.
///
/// A cache instance owns one runtime lease from the shared device cache domain.
/// Graphs are therefore always resumed on the same exact runtime binding, and
/// source and graph cache allocations still share the domain's aggregate cap.
pub(crate) struct GpuResidentGraphCache {
    inner: Arc<GpuResidentGraphCacheInner>,
}

impl Clone for GpuResidentGraphCache {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl std::fmt::Debug for GpuResidentGraphCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        let allocation_lease = domain.lease_runtime().map_err(|error| {
            QueryError::from_cudf(
                OPERATION,
                "failed to lease the shared cache allocation domain for resident graphs",
                error,
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("cache_allocation_domain_id", domain.id())
        })?;
        let runtime = Arc::new(allocation_lease.runtime().clone());
        if runtime.device().ordinal() != device_ordinal {
            return Err(QueryError::new(
                QueryErrorCode::CudfInvariantViolation,
                OPERATION,
                "graph cache runtime does not match its ledger device",
            )
            .with_fact("reason", "graph_cache_runtime_device_mismatch")
            .with_fact("ledger_device_ordinal", device_ordinal)
            .with_fact("runtime_device_ordinal", runtime.device().ordinal()));
        }
        Ok(Self {
            inner: Arc::new(GpuResidentGraphCacheInner {
                ledger,
                allocation_lease,
                runtime,
                state: Mutex::new(GpuResidentGraphCacheState::default()),
                changed: Condvar::new(),
            }),
        })
    }

    /// Stable identity of the shared source/graph cache allocation domain.
    pub(crate) fn cache_allocation_domain_id(&self) -> cudf::CacheAllocationDomainId {
        self.inner.allocation_lease.domain_id()
    }

    /// Shared physical allocation domain used by both resident cache subdomains.
    pub(crate) fn cache_allocation_domain(&self) -> &cudf::CacheAllocationDomain {
        self.inner.allocation_lease.domain()
    }

    /// The one graph-cache runtime binding. Graph builders and consumers must
    /// use this runtime, not a query runtime or a per-entry companion runtime.
    pub(crate) fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.inner.runtime
    }

    /// Samples aggregate live bytes from the shared physical allocation domain.
    pub(crate) fn allocation_usage(&self) -> Result<cudf::DeviceMemoryResourceUsage> {
        self.cache_allocation_domain().usage().map_err(|error| {
            QueryError::from_cudf(OPERATION, "failed to sample shared cache allocation-domain usage", error)
                .with_fact("cache_allocation_domain_id", self.cache_allocation_domain_id())
        })
    }

    /// The largest conservative graph reservation still permitted by the one
    /// ledger-owned aggregate cap. This is derived from the ledger snapshot;
    /// the graph cache never keeps a second byte counter.
    pub(crate) fn available_reservation_bytes(&self) -> Result<u64> {
        let snapshot = self.inner.ledger.snapshot()?;
        let charged_cache_bytes =
            snapshot.cache_evictable_resident_bytes.saturating_add(snapshot.cache_pinned_in_use_bytes);
        let cache_remaining = self.inner.ledger.profile().cache_cap_bytes.saturating_sub(charged_cache_bytes);
        let managed_remaining = snapshot
            .managed_capacity_bytes
            .saturating_sub(snapshot.live_grants_bytes)
            .saturating_sub(charged_cache_bytes);
        Ok(cache_remaining.min(managed_remaining))
    }

    /// Pre-reserves a graph's measured retention bytes before graph construction
    /// or ownership transfer. Dropping the returned transaction rolls the exact
    /// charge back unless it is committed by [`GpuResidentGraphInsertTransaction::insert_if_absent`].
    pub(crate) fn begin_insert(&self, retained_device_bytes: u64) -> Result<Option<GpuResidentGraphInsertTransaction>> {
        if self.lock()?.closed {
            return Ok(None);
        }
        let Some(charge) = CacheResidency::begin_attempt_owned(self.inner.ledger.clone(), retained_device_bytes)?
        else {
            return Ok(None);
        };
        if self.lock()?.closed {
            drop(charge);
            return Ok(None);
        }
        Ok(Some(GpuResidentGraphInsertTransaction { cache: Arc::clone(&self.inner), charge: Some(charge) }))
    }

    /// Reports whether this device cache contains the canonical graph key.
    /// This is host-only placement evidence and never allocates or pins.
    pub(crate) fn contains_resident_key(&self, resident_key: &ResidentGraphKey) -> Result<bool> {
        let state = self.lock()?;
        Ok(state.available.contains_key(resident_key) || state.checked_out.contains_key(resident_key))
    }

    /// Pins an available entry before source-side graph preparation commits to
    /// a cache hit. Once [`Self::close`] begins, no final pin is admitted.
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
    pub(crate) fn checkout<T>(
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
            return Ok(None);
        };
        let pin = match preparation_pin {
            Some(preparation_pin) => {
                if !preparation_pin.belongs_to(&self.inner, resident_key) {
                    return Ok(None);
                }
                let Some(prepared) = state.preparation_pins.remove(resident_key) else {
                    return Ok(None);
                };
                if prepared.id != preparation_pin.id {
                    state.preparation_pins.insert(resident_key.clone(), prepared);
                    return Ok(None);
                }
                preparation_pin.active = false;
                prepared.pin
            },
            None => {
                if state.preparation_pins.contains_key(resident_key) {
                    return Ok(None);
                }
                let Some(pin) = entry.residency.try_pin()? else {
                    return Ok(None);
                };
                pin
            },
        };
        let graph = match entry.take_graph() {
            Ok(graph) => graph,
            Err(error) => {
                drop(pin);
                return Err(error);
            },
        };
        let Some(parked) = T::take_entry(graph) else {
            drop(pin);
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

    /// Evicts one idle resident graph. Pinned graphs are refused without
    /// changing their charge. GPU destruction happens before the matching
    /// ledger release transition.
    pub(crate) fn evict(&self, resident_key: &ResidentGraphKey) -> Result<bool> {
        let entry = {
            let mut state = self.lock()?;
            let Some(entry) = state.available.get(resident_key).cloned() else {
                return Ok(false);
            };
            let Some(_destruction) = entry.residency.begin_eviction()? else {
                return Ok(false);
            };
            // The token is intentionally recreated after the graph has been
            // detached below. Dropping this probe aborts the gate and avoids
            // holding a cache mutex during native destruction.
            drop(_destruction);
            state.available.remove(resident_key);
            entry
        };
        let graph = entry.take_graph()?;
        self.destroy_entry_after_graph_drop(entry, graph)
    }

    /// Begins shutdown. Existing pins may finish, but no final preparation pin,
    /// checkout, or insertion can start after this point.
    pub(crate) fn close(&self) -> Result<()> {
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
#[must_use = "dropping an insert transaction rolls its ledger reservation back"]
pub(crate) struct GpuResidentGraphInsertTransaction {
    cache: Arc<GpuResidentGraphCacheInner>,
    charge: Option<CacheResidencyChargeTransaction>,
}

impl GpuResidentGraphInsertTransaction {
    pub(crate) fn bytes(&self) -> u64 {
        self.charge.as_ref().map_or(0, CacheResidencyChargeTransaction::bytes)
    }

    /// Runtime bound to the shared cache allocation domain. Build the graph on
    /// this runtime before committing it, never on a private companion runtime.
    pub(crate) fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.cache.runtime
    }

    /// Measures the graph's retained allocation delta from the shared cache
    /// domain and reconciles a conservative pre-reservation down to that exact
    /// value. A larger-than-reserved delta returns `false`; the caller must
    /// destroy the graph before this transaction drops and rolls its charge
    /// back, so the ledger never observes an undercharged retained graph.
    pub(crate) fn shrink_to_domain_delta(&mut self, before: cudf::DeviceMemoryResourceUsage) -> Result<bool> {
        let after = self.cache.allocation_lease.domain().usage().map_err(|error| {
            QueryError::from_cudf(OPERATION, "failed to measure resident graph allocation-domain delta", error)
                .with_fact("cache_allocation_domain_id", self.cache.allocation_lease.domain_id())
        })?;
        let retained_bytes = after.current_bytes.saturating_sub(before.current_bytes);
        self.charge
            .as_mut()
            .ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    OPERATION,
                    "resident graph insertion transaction was already consumed",
                )
                .with_fact("reason", "resident_graph_insert_transaction_consumed")
            })?
            .shrink_to(retained_bytes)
    }

    pub(crate) fn insert_if_absent<T>(
        mut self,
        resident_key: ResidentGraphKey,
        parked: ParkedGraph<T>,
    ) -> Result<GpuResidentGraphCacheInsert>
    where
        T: GpuResidentGraphTransposition,
    {
        let expected_device = self.cache.runtime.device().ordinal();
        let graph_device = parked.device();
        if graph_device != i32::try_from(expected_device).unwrap_or(i32::MAX) {
            drop(parked);
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "parked graph device does not match the shared cache allocation domain",
            )
            .with_fact("reason", "resident_graph_device_mismatch")
            .with_fact("graph_device_ordinal", graph_device)
            .with_fact("cache_device_ordinal", expected_device));
        }
        if resident_key.orientation != T::orientation() {
            drop(parked);
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "parked graph orientation does not match the resident graph key",
            )
            .with_fact("reason", "resident_graph_key_orientation_mismatch"));
        }
        let mut state = self.cache.state.lock().map_err(|_error| resident_graph_cache_poisoned("insert"))?;
        if state.closed {
            drop(state);
            drop(parked);
            return Ok(GpuResidentGraphCacheInsert::Closed);
        }
        if state.available.contains_key(&resident_key) || state.checked_out.contains_key(&resident_key) {
            drop(state);
            drop(parked);
            return Ok(GpuResidentGraphCacheInsert::AlreadyPresent);
        }
        let charge = self.charge.take().expect("resident graph insertion transaction is consumed once");
        let entry = Arc::new(GpuResidentGraphCacheEntry {
            graph: Mutex::new(Some(T::into_entry(parked))),
            residency: charge.commit_evictable(),
        });
        state.available.insert(resident_key, entry);
        self.cache.changed.notify_all();
        Ok(GpuResidentGraphCacheInsert::Inserted)
    }
}

/// A preparation pin that reserves an exact resident graph for its eventual
/// checkout owner. Dropping it returns the entry to evictable state.
#[must_use = "dropping the preparation pin releases its shared residency pin"]
pub(crate) struct GpuResidentGraphPreparationPin {
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
pub(crate) struct GpuResidentGraphCheckout<T>
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
    pub(crate) fn retained_device_bytes(&self) -> u64 {
        self.entry.residency.bytes()
    }

    /// Extracts the parked graph for a cuGraph execution. The caller must
    /// return it with [`Self::return_parked`]; dropping the checkout after an
    /// extraction conservatively leaves the shared charge pinned and prevents
    /// shutdown from claiming the bytes were drained.
    pub(crate) fn take_parked(&mut self) -> ParkedGraph<T> {
        self.parked.take().expect("resident graph checkout already consumed")
    }

    pub(crate) fn return_parked(mut self, parked: ParkedGraph<T>) -> Result<()> {
        self.complete_with_parked(parked)
    }

    fn complete_with_parked(&mut self, parked: ParkedGraph<T>) -> Result<()> {
        let expected_device = self.cache.runtime.device().ordinal();
        let graph_device = parked.device();
        if graph_device != i32::try_from(expected_device).unwrap_or(i32::MAX) {
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
                state
                    .checked_out
                    .insert(self.resident_key.clone(), GpuResidentGraphCheckedOutState::Escaped { entry, pin });
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
    let device = cache.runtime.device();
    let before = cudf::runtime::device_health_snapshot(device);
    if before.status() != cudf::runtime::DeviceHealthStatus::Healthy {
        std::mem::forget(graph);
        let reason = "device was already unhealthy before resident graph teardown";
        cudf::runtime::quarantine_device_resource(device, OPERATION, &reason);
        destruction.retain_charge_after_failed_destruction()?;
        return record_teardown_quarantine(cache, entry.residency.bytes(), reason).map(|_| false);
    }

    // `ParkedGraph::drop` synchronizes its park event before freeing native
    // graph state. Only after that destructor returns may the ledger charge be
    // committed as released.
    drop(graph);
    let after = cudf::runtime::device_health_snapshot(device);
    if after.status() != cudf::runtime::DeviceHealthStatus::Healthy
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
    destruction.commit_after_destroy()?;
    Ok(true)
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
#[path = "resident_graph_tests.rs"]
mod tests;
