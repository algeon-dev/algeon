//! Runtime-owned resident cuGraph cache and singleflight coordination.

#[cfg(feature = "cugraph")]
use std::sync::TryLockError;
#[cfg(feature = "cugraph")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Condvar, Mutex, Weak},
    time::{Duration, Instant},
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::ResidentGraphKey,
    runtime::{
        ExecutionContext,
        gpu::ledger::{DeviceLedger, EvictableMemoryPressure, MemoryGrant},
    },
};

/// Default bound on same-graph replicas retained across service stream-pool
/// runtimes.
pub const DEFAULT_RESIDENT_GRAPH_CACHE_MAX_COPIES_PER_KEY: usize = 4;

/// Cache companions use a process-local high id range so their durable stream
/// reservations remain distinguishable from short-lived query reservations.
#[cfg(feature = "cugraph")]
static NEXT_RESIDENT_GRAPH_CACHE_RESERVATION_ID: AtomicU64 = AtomicU64::new(1 << 63);

/// Point-in-time resident graph cache state and monotonic activity counters.
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuResidentGraphCacheSnapshot {
    /// Available plus checked-out entries.
    pub entries: usize,
    /// Device bytes retained across all entries.
    pub resident_bytes: u64,
    /// Device bytes held by checked-out entries.
    pub in_use_bytes: u64,
    /// Successful exact-runtime checkouts since cache creation.
    pub hits: u64,
    /// Cold graph executions elected since cache creation.
    pub misses: u64,
    /// Available entries removed under capacity or governor pressure.
    pub evictions: u64,
    /// Graph-build generations currently in flight.
    pub builds_in_flight: usize,
    /// Follower waits joined since cache creation.
    pub build_waits: u64,
}

/// Result of inserting a parked resident graph.
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuResidentGraphCacheInsert {
    /// The parked graph was retained by the cache.
    Inserted,
    /// An equivalent entry was already available for the same runtime.
    AlreadyPresent,
    /// The retained-byte cap could not admit this graph.
    CapacitySkipped,
    /// The per-resident-key replica cap would be exceeded.
    CopyLimitSkipped,
}

#[cfg(feature = "cugraph")]
pub(crate) enum GpuResidentGraphEntry {
    Forward(cugraph::ParkedGraph<cugraph::Forward>),
    Transposed(cugraph::ParkedGraph<cugraph::Transposed>),
}

#[cfg(feature = "cugraph")]
pub(crate) trait GpuResidentGraphTransposition: cugraph::Transposition + Send + Sized + 'static {
    fn orientation() -> crate::plan::GraphResidencyOrientation;
    fn into_entry(parked: cugraph::ParkedGraph<Self>) -> GpuResidentGraphEntry;
    fn take_entry(entry: GpuResidentGraphEntry) -> Option<cugraph::ParkedGraph<Self>>;
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphTransposition for cugraph::Forward {
    fn orientation() -> crate::plan::GraphResidencyOrientation {
        crate::plan::GraphResidencyOrientation::Forward
    }

    fn into_entry(parked: cugraph::ParkedGraph<Self>) -> GpuResidentGraphEntry {
        GpuResidentGraphEntry::Forward(parked)
    }

    fn take_entry(entry: GpuResidentGraphEntry) -> Option<cugraph::ParkedGraph<Self>> {
        match entry {
            GpuResidentGraphEntry::Forward(parked) => Some(parked),
            GpuResidentGraphEntry::Transposed(_) => None,
        }
    }
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphTransposition for cugraph::Transposed {
    fn orientation() -> crate::plan::GraphResidencyOrientation {
        crate::plan::GraphResidencyOrientation::Transposed
    }

    fn into_entry(parked: cugraph::ParkedGraph<Self>) -> GpuResidentGraphEntry {
        GpuResidentGraphEntry::Transposed(parked)
    }

    fn take_entry(entry: GpuResidentGraphEntry) -> Option<cugraph::ParkedGraph<Self>> {
        match entry {
            GpuResidentGraphEntry::Transposed(parked) => Some(parked),
            GpuResidentGraphEntry::Forward(_) => None,
        }
    }
}

#[cfg(feature = "cugraph")]
#[derive(Clone, PartialEq, Eq, Hash)]
struct GpuResidentGraphCacheEntryKey {
    resident_key: ResidentGraphKey,
    runtime_key: u64,
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphCacheEntryKey {
    fn for_context(resident_key: ResidentGraphKey, context: &ExecutionContext) -> Result<Self> {
        let runtime =
            context.resident_graph_cache_owner_runtime("nexus_query_engine.gpu_resident_graph_cache.entry_key")?;
        Ok(Self { resident_key, runtime_key: runtime.affinity_id().get() })
    }
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCacheEntry {
    graph: GpuResidentGraphEntry,
    retained_device_bytes: u64,
    last_access_epoch: u64,
    runtime_lease: cudf::runtime::RuntimeLeaseAnchor,
}

#[cfg(feature = "cugraph")]
#[derive(Default)]
struct GpuResidentGraphCacheEviction {
    entries: Vec<GpuResidentGraphCacheEntry>,
    device_bytes: u64,
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCheckedOut {
    retained_device_bytes: u64,
    runtime_lease: cudf::runtime::RuntimeLeaseAnchor,
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCompanionRuntime {
    // The companion stream must remain attached to its cache-owned reservation
    // for as long as cache entries can retain allocations made on that stream.
    stream_reservation: cudf::StreamReservationLease,
    pooled_runtime_lease: cudf::runtime::RuntimeLeaseAnchor,
    companion: Weak<cudf::Runtime>,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Default)]
struct GpuResidentGraphBuildFlightState {
    finished: bool,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Default)]
struct GpuResidentGraphBuildFlight {
    state: Mutex<GpuResidentGraphBuildFlightState>,
    changed: Condvar,
}

#[cfg(feature = "cugraph")]
#[derive(Default)]
struct GpuResidentGraphCacheState {
    available: HashMap<GpuResidentGraphCacheEntryKey, GpuResidentGraphCacheEntry>,
    checked_out: HashMap<GpuResidentGraphCacheEntryKey, GpuResidentGraphCheckedOut>,
    preparation_pins: HashMap<GpuResidentGraphCacheEntryKey, u64>,
    companion_runtimes: HashMap<u64, GpuResidentGraphCompanionRuntime>,
    builds: HashMap<ResidentGraphKey, Arc<GpuResidentGraphBuildFlight>>,
    retained_device_bytes: u64,
    next_access_epoch: u64,
    next_preparation_pin_id: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    build_waits: u64,
    closed: bool,
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCacheInner {
    max_retained_device_bytes: u64,
    max_copies_per_key: usize,
    governor: Option<DeviceLedger>,
    state: Mutex<GpuResidentGraphCacheState>,
    #[cfg(test)]
    close_and_drain_delay: Mutex<Option<Duration>>,
}

#[cfg(feature = "cugraph")]
impl Drop for GpuResidentGraphCacheInner {
    fn drop(&mut self) {
        let Some(governor) = &self.governor else {
            return;
        };
        let Ok(state) = self.state.get_mut() else {
            return;
        };
        let retained_device_bytes = state.retained_device_bytes;
        if retained_device_bytes > 0 {
            let _ = governor.release_evictable_bytes(retained_device_bytes);
            state.retained_device_bytes = 0;
        }
    }
}

/// Per-device cache for cuGraph graphs parked between query executions.
///
/// Entries are keyed by a device-agnostic [`ResidentGraphKey`] plus the
/// service stream-pool runtime identity that owns the parked graph's cuDF
/// runtime-bound handle leases. Checked-out graphs are exclusive: a hit moves
/// the parked graph out of the available map until the caller parks it again
/// or drops the checkout guard, at which point the governor evictable claim is
/// released.
#[cfg(feature = "cugraph")]
pub struct GpuResidentGraphCache {
    inner: Arc<GpuResidentGraphCacheInner>,
}

#[cfg(feature = "cugraph")]
impl std::fmt::Debug for GpuResidentGraphCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.inner.state.lock().ok();
        f.debug_struct("GpuResidentGraphCache")
            .field("max_retained_device_bytes", &self.inner.max_retained_device_bytes)
            .field("max_copies_per_key", &self.inner.max_copies_per_key)
            .field("governor_backed", &self.is_governor_backed())
            .field("available_entry_count", &state.as_ref().map(|state| state.available.len()))
            .field("checked_out_entry_count", &state.as_ref().map(|state| state.checked_out.len()))
            .field("retained_device_bytes", &state.as_ref().map(|state| state.retained_device_bytes))
            .finish()
    }
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphCache {
    /// Create an empty resident graph cache with a retained-device-byte cap.
    pub fn new(max_retained_device_bytes: u64) -> Self {
        Self::new_with_governor(max_retained_device_bytes, DEFAULT_RESIDENT_GRAPH_CACHE_MAX_COPIES_PER_KEY, None)
    }

    /// Create an empty resident graph cache charged to a device memory
    /// governor's evictable ledger.
    pub fn new_governor_backed(max_retained_device_bytes: u64, governor: DeviceLedger) -> Self {
        Self::new_with_governor(
            max_retained_device_bytes,
            DEFAULT_RESIDENT_GRAPH_CACHE_MAX_COPIES_PER_KEY,
            Some(governor),
        )
    }

    /// Override the per-resident-key copy cap.
    pub fn with_max_copies_per_key(self, max_copies_per_key: usize) -> Self {
        Self::new_with_governor(self.inner.max_retained_device_bytes, max_copies_per_key, self.inner.governor.clone())
    }

    fn new_with_governor(
        max_retained_device_bytes: u64,
        max_copies_per_key: usize,
        governor: Option<DeviceLedger>,
    ) -> Self {
        Self {
            inner: Arc::new(GpuResidentGraphCacheInner {
                max_retained_device_bytes,
                max_copies_per_key,
                governor,
                state: Mutex::new(GpuResidentGraphCacheState::default()),
                #[cfg(test)]
                close_and_drain_delay: Mutex::new(None),
            }),
        }
    }

    /// Maximum retained device bytes allowed by this cache.
    pub fn max_retained_device_bytes(&self) -> u64 {
        self.inner.max_retained_device_bytes
    }

    /// Maximum same-graph replicas retained across runtime identities.
    pub fn max_copies_per_key(&self) -> usize {
        self.inner.max_copies_per_key
    }

    /// Whether retained bytes are charged to a memory governor.
    pub fn is_governor_backed(&self) -> bool {
        self.inner.governor.is_some()
    }

    /// Register this cache as a synchronous evictable-memory observer.
    pub fn register_governor_pressure_observer(cache: &Arc<Self>) -> Result<()> {
        let Some(governor) = cache.inner.governor.as_ref().cloned() else {
            return Ok(());
        };
        let weak_cache: Weak<Self> = Arc::downgrade(cache);
        governor.register_evictable_memory_observer(Arc::new(move |pressure| {
            let Some(cache) = weak_cache.upgrade() else {
                return Ok(0);
            };
            cache.handle_memory_pressure(pressure)
        }))
    }

    /// Number of available parked graph entries.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.available.len())
    }

    /// Whether no parked graph entry is currently available.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.lock()?.available.is_empty())
    }

    /// Number of graphs currently checked out by running queries.
    pub fn checked_out_len(&self) -> Result<usize> {
        Ok(self.lock()?.checked_out.len())
    }

    /// Retained device bytes across available and checked-out parked graphs.
    pub fn retained_device_bytes(&self) -> Result<u64> {
        Ok(self.lock()?.retained_device_bytes)
    }

    /// Returns current state and monotonic cache activity counters.
    pub fn snapshot(&self) -> Result<GpuResidentGraphCacheSnapshot> {
        let state = self.lock()?;
        Ok(GpuResidentGraphCacheSnapshot {
            entries: state.available.len().saturating_add(state.checked_out.len()),
            resident_bytes: state.retained_device_bytes,
            in_use_bytes: state
                .checked_out
                .values()
                .map(|entry| entry.retained_device_bytes)
                .fold(0u64, u64::saturating_add),
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            builds_in_flight: state.builds.len(),
            build_waits: state.build_waits,
        })
    }

    /// Whether this device cache currently has a live copy for `resident_key`.
    ///
    /// This is a host-side placement hint only: it must not perform CUDA work,
    /// call the memory governor, or acquire locks beyond the cache mutex because
    /// query-service admission may call it while holding the admission mutex.
    pub fn contains_resident_key(&self, resident_key: &ResidentGraphKey) -> Result<bool> {
        let state = self.lock()?;
        Ok(state
            .available
            .iter()
            .any(|(key, entry)| &key.resident_key == resident_key && entry.runtime_lease.is_current())
            || state
                .checked_out
                .iter()
                .any(|(key, entry)| &key.resident_key == resident_key && entry.runtime_lease.is_current()))
    }

    /// Pin an available graph for an imminent exact-runtime checkout.
    ///
    /// Unlike [`Self::contains_resident_key`], this is an exact-runtime lookup
    /// used by execution to decide whether source-side graph preparation can be
    /// skipped safely. Pressure eviction skips the pinned entry until checkout
    /// consumes the pin or the returned guard drops.
    pub(crate) fn try_pin_available_key_for_context(
        &self,
        resident_key: &ResidentGraphKey,
        context: &ExecutionContext,
    ) -> Result<Option<GpuResidentGraphPreparationPin>> {
        let context_runtime =
            context.resident_graph_cache_owner_runtime("nexus_query_engine.gpu_resident_graph_cache.prepare_pin")?;
        self.try_pin_available_key_for_runtime(resident_key, &context_runtime)
    }

    pub(crate) fn try_pin_available_key_for_runtime(
        &self,
        resident_key: &ResidentGraphKey,
        runtime: &Arc<cudf::Runtime>,
    ) -> Result<Option<GpuResidentGraphPreparationPin>> {
        let entry_key = GpuResidentGraphCacheEntryKey {
            resident_key: resident_key.clone(),
            runtime_key: runtime.affinity_id().get(),
        };
        let mut state = self.lock()?;
        if state.closed {
            return Ok(None);
        }
        let available =
            state.available.get(&entry_key).is_some_and(|entry| runtime_lease_matches(&entry.runtime_lease, runtime));
        if !available || state.preparation_pins.contains_key(&entry_key) {
            return Ok(None);
        }
        let pin_id = state.next_preparation_pin_id();
        state.preparation_pins.insert(entry_key.clone(), pin_id);
        Ok(Some(GpuResidentGraphPreparationPin { cache: Arc::clone(&self.inner), entry_key, pin_id }))
    }

    pub(crate) fn join_build(&self, resident_key: &ResidentGraphKey) -> Result<GpuResidentGraphBuildJoin> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(resident_graph_cache_closed("nexus_query_engine.gpu_resident_graph_cache.join_build"));
        }
        if let Some(flight) = state.builds.get(resident_key).cloned() {
            state.build_waits = state.build_waits.saturating_add(1);
            return Ok(GpuResidentGraphBuildJoin::Follower(GpuResidentGraphBuildFollower { flight }));
        }
        let flight = Arc::new(GpuResidentGraphBuildFlight::default());
        state.builds.insert(resident_key.clone(), Arc::clone(&flight));
        state.misses = state.misses.saturating_add(1);
        Ok(GpuResidentGraphBuildJoin::Leader(Box::new(GpuResidentGraphBuildLeader {
            cache: Arc::clone(&self.inner),
            resident_key: resident_key.clone(),
            flight,
            finished: false,
        })))
    }

    pub(crate) fn companion_runtime_for_context(
        &self,
        context: &ExecutionContext,
        operation: &'static str,
    ) -> Result<Arc<cudf::Runtime>> {
        let pooled_runtime = context.resident_graph_cache_owner_runtime(operation)?;
        let runtime_key = pooled_runtime.affinity_id().get();
        let device_ordinal = pooled_runtime.device().ordinal();
        {
            let state = self.lock()?;
            if state.closed {
                return Err(resident_graph_cache_closed(operation));
            }
            if let Some(companion) = state.companion_runtimes.get(&runtime_key)
                && runtime_lease_matches(&companion.pooled_runtime_lease, &pooled_runtime)
                && let Some(companion) = companion.companion.upgrade()
            {
                return Ok(companion);
            }
        }
        // A parked graph outlives the admitted attempt that populated it.
        // Reusing that attempt's stream would leave cache-owned allocations in
        // its query-memory domain and quarantine the device at release.
        let companion_runtime = cudf::Runtime::builder()
            .with_device(pooled_runtime.device())
            .with_device_memory_manager()
            .build()
            .map_err(|error| {
                QueryError::from_cudf(
                    operation,
                    "failed to build stable companion runtime for resident graph cache",
                    error,
                )
                .with_fact("device_ordinal", device_ordinal)
            })?
            .inherit_affinity_from(pooled_runtime.as_ref())
            .map_err(|error| {
                QueryError::from_cudf(
                    operation,
                    "failed to inherit base runtime affinity for resident graph companion runtime",
                    error,
                )
                .with_fact("device_ordinal", device_ordinal)
            })?
            .inherit_retained_lifetime_from(pooled_runtime.as_ref())
            .map_err(|error| {
                QueryError::from_cudf(
                    operation,
                    "failed to retain service lifetime for resident graph companion runtime",
                    error,
                )
                .with_fact("device_ordinal", device_ordinal)
            })?;
        let companion = Arc::new(companion_runtime);
        let reservation_id = NEXT_RESIDENT_GRAPH_CACHE_RESERVATION_ID.fetch_add(1, Ordering::Relaxed);
        let stream_reservation = companion.stream_reservation(reservation_id, usize::MAX).map_err(|error| {
            QueryError::from_cudf(
                operation,
                "failed to attach a cache-owned stream reservation to the resident graph companion runtime",
                error,
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("reservation_id", reservation_id)
        })?;
        let mut state = self.lock()?;
        if state.closed {
            return Err(resident_graph_cache_closed(operation));
        }
        if let Some(existing) = state.companion_runtimes.get(&runtime_key)
            && runtime_lease_matches(&existing.pooled_runtime_lease, &pooled_runtime)
            && let Some(existing) = existing.companion.upgrade()
        {
            return Ok(existing);
        }
        state.companion_runtimes.insert(
            runtime_key,
            GpuResidentGraphCompanionRuntime {
                stream_reservation,
                pooled_runtime_lease: pooled_runtime.lease_anchor(),
                companion: Arc::downgrade(&companion),
            },
        );
        Ok(companion)
    }

    /// Snapshot live allocations routed through a cache companion stream.
    ///
    /// Native cuGraph allocations resolve through the device manager's current
    /// resource, which charges them to the stream-bound reservation rather
    /// than to a runtime's explicit output allocator. The cache uses this
    /// before/after delta around graph construction so retained graph state is
    /// charged to the durable cache rather than to a query attempt.
    pub(crate) fn companion_runtime_live_bytes(
        &self,
        companion_runtime: &Arc<cudf::Runtime>,
        operation: &'static str,
    ) -> Result<u64> {
        let runtime_key = companion_runtime.affinity_id().get();
        let stream_reservation = {
            let state = self.lock()?;
            let Some(companion) = state.companion_runtimes.get(&runtime_key) else {
                return Err(QueryError::new(
                    QueryErrorCode::CudfInvariantViolation,
                    operation,
                    "resident graph cache companion is missing its stream-reservation accounting lease",
                )
                .with_fact("reason", "resident_graph_companion_reservation_missing")
                .with_fact("device_ordinal", companion_runtime.device().ordinal()));
            };
            let Some(registered_runtime) = companion.companion.upgrade() else {
                return Err(QueryError::new(
                    QueryErrorCode::CudfInvariantViolation,
                    operation,
                    "resident graph cache companion runtime expired before its allocation usage could be sampled",
                )
                .with_fact("reason", "resident_graph_companion_runtime_expired")
                .with_fact("device_ordinal", companion_runtime.device().ordinal()));
            };
            if !Arc::ptr_eq(&registered_runtime, companion_runtime) {
                return Err(QueryError::new(
                    QueryErrorCode::CudfInvariantViolation,
                    operation,
                    "resident graph cache companion runtime does not match its allocation accounting lease",
                )
                .with_fact("reason", "resident_graph_companion_reservation_mismatch")
                .with_fact("device_ordinal", companion_runtime.device().ordinal()));
            }
            companion.stream_reservation.clone()
        };
        stream_reservation.usage().map(|usage| usage.current_bytes).map_err(|error| {
            QueryError::from_cudf(
                operation,
                "failed to snapshot resident graph cache companion stream reservation",
                error,
            )
            .with_fact("device_ordinal", companion_runtime.device().ordinal())
        })
    }

    pub(crate) fn checkout<T>(
        &self,
        resident_key: &ResidentGraphKey,
        context: &ExecutionContext,
        preparation_pin_id: Option<u64>,
    ) -> Result<Option<GpuResidentGraphCheckout<T>>>
    where
        T: GpuResidentGraphTransposition,
    {
        let entry_key = GpuResidentGraphCacheEntryKey::for_context(resident_key.clone(), context)?;
        let mut state = self.lock()?;
        if state.closed {
            return Err(resident_graph_cache_closed("nexus_query_engine.gpu_resident_graph_cache.checkout"));
        }
        if !state.available.contains_key(&entry_key)
            || !state.consume_checkout_preparation_pin(&entry_key, preparation_pin_id)
        {
            return Ok(None);
        }
        let Some(entry) = state.available.remove(&entry_key) else {
            unreachable!("available resident graph entry checked immediately before removal")
        };
        let context_runtime =
            context.resident_graph_cache_owner_runtime("nexus_query_engine.gpu_resident_graph_cache.checkout")?;
        let retained_device_bytes = entry.retained_device_bytes;
        if !runtime_lease_matches(&entry.runtime_lease, &context_runtime) {
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(retained_device_bytes);
            drop(state);
            let released = self.release_governor_evictable_bytes(retained_device_bytes);
            drop(entry);
            released?;
            return Ok(None);
        }
        let Some(parked) = T::take_entry(entry.graph) else {
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(retained_device_bytes);
            drop(state);
            self.release_governor_evictable_bytes(retained_device_bytes)?;
            return Ok(None);
        };
        state.checked_out.insert(
            entry_key.clone(),
            GpuResidentGraphCheckedOut { retained_device_bytes, runtime_lease: entry.runtime_lease },
        );
        state.hits = state.hits.saturating_add(1);
        Ok(Some(GpuResidentGraphCheckout {
            cache: Arc::clone(&self.inner),
            entry_key,
            retained_device_bytes,
            parked: Some(parked),
            completed: false,
            _transposition: PhantomData,
        }))
    }

    pub(crate) fn insert_if_absent<T>(
        &self,
        resident_key: ResidentGraphKey,
        context: &ExecutionContext,
        parked: cugraph::ParkedGraph<T>,
        retained_device_bytes: u64,
    ) -> Result<GpuResidentGraphCacheInsert>
    where
        T: GpuResidentGraphTransposition,
    {
        let context_runtime =
            context.resident_graph_cache_owner_runtime("nexus_query_engine.gpu_resident_graph_cache.insert")?;
        let entry_key = GpuResidentGraphCacheEntryKey::for_context(resident_key, context)?;
        if retained_device_bytes > self.inner.max_retained_device_bytes {
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let mut state = self.lock()?;
        if state.closed {
            return Err(resident_graph_cache_closed("nexus_query_engine.gpu_resident_graph_cache.insert"));
        }
        if state.available.contains_key(&entry_key) || state.checked_out.contains_key(&entry_key) {
            return Ok(GpuResidentGraphCacheInsert::AlreadyPresent);
        }
        if state.copy_count_for_resident_key(&entry_key.resident_key) >= self.inner.max_copies_per_key {
            return Ok(GpuResidentGraphCacheInsert::CopyLimitSkipped);
        }
        let memory_grant = context.memory_grant()?;
        if !self.governor_can_admit_after_cache_eviction(&state, retained_device_bytes, memory_grant.as_ref())? {
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let eviction = state.evict_until_fits(self.inner.max_retained_device_bytes, retained_device_bytes);
        let evicted_device_bytes = eviction.device_bytes;
        // These governor ledger calls must not notify eviction observers while
        // holding the governor state lock; observers may call back into this
        // cache and take the cache mutex.
        if let Err(error) = self.release_governor_evictable_bytes(evicted_device_bytes) {
            drop(state);
            drop(eviction);
            return Err(error);
        }
        if state.retained_device_bytes.saturating_add(retained_device_bytes) > self.inner.max_retained_device_bytes {
            drop(state);
            drop(eviction);
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let reserved = match self.reserve_governor_evictable_bytes(retained_device_bytes, memory_grant.as_ref()) {
            Ok(reserved) => reserved,
            Err(error) => {
                drop(state);
                drop(eviction);
                return Err(error);
            },
        };
        if !reserved {
            drop(state);
            drop(eviction);
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let last_access_epoch = state.next_epoch();
        state.available.insert(
            entry_key,
            GpuResidentGraphCacheEntry {
                graph: T::into_entry(parked),
                retained_device_bytes,
                last_access_epoch,
                runtime_lease: context_runtime.lease_anchor(),
            },
        );
        state.retained_device_bytes = state.retained_device_bytes.saturating_add(retained_device_bytes);
        drop(state);
        drop(eviction);
        Ok(GpuResidentGraphCacheInsert::Inserted)
    }

    /// Close the cache and release every cache-owned parked graph.
    ///
    /// Checked-out graphs are owned by their checkout/live graph values and
    /// are intentionally not detached here. Their runtime-bound handle leases
    /// keep the service resource epoch counted until those GPU objects drop.
    #[cfg(test)]
    pub(crate) fn close_and_drain(&self) -> Result<()> {
        let (entries, flights, released_device_bytes) = {
            let mut state = self.lock()?;
            state.closed = true;
            state.preparation_pins.clear();
            let entries = state.available.drain().map(|(_key, entry)| entry).collect::<Vec<_>>();
            let released_device_bytes =
                entries.iter().map(|entry| entry.retained_device_bytes).fold(0u64, u64::saturating_add);
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(released_device_bytes);
            let flights = state.builds.drain().map(|(_key, flight)| flight).collect::<Vec<_>>();
            (entries, flights, released_device_bytes)
        };

        for flight in flights {
            if let Ok(mut state) = flight.state.lock() {
                state.finished = true;
                flight.changed.notify_all();
            }
        }
        // Parked graphs own the counted runtime/epoch token. Destroy them
        // before clearing companion lookup metadata.
        drop(entries);
        let release_result = self.release_governor_evictable_bytes(released_device_bytes);
        self.lock()?.companion_runtimes.clear();
        release_result.map(|_| ())
    }

    /// Close and drain under one server-owned absolute shutdown deadline.
    pub(crate) fn close_and_drain_until(&self, deadline: Instant) -> Result<()> {
        const OPERATION: &str = "nexus_query_engine.gpu_resident_graph_cache.close_and_drain";
        let (entries, flights, released_device_bytes) = {
            let mut state = self.lock_until(deadline, OPERATION)?;
            state.closed = true;
            state.preparation_pins.clear();
            let entries = state.available.drain().map(|(_key, entry)| entry).collect::<Vec<_>>();
            let released_device_bytes =
                entries.iter().map(|entry| entry.retained_device_bytes).fold(0u64, u64::saturating_add);
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(released_device_bytes);
            let flights = state.builds.drain().map(|(_key, flight)| flight).collect::<Vec<_>>();
            (entries, flights, released_device_bytes)
        };

        let mut timed_out = Instant::now() >= deadline;
        for flight in flights {
            loop {
                match flight.state.try_lock() {
                    Ok(mut state) => {
                        state.finished = true;
                        flight.changed.notify_all();
                        break;
                    },
                    Err(TryLockError::Poisoned(_)) => break,
                    Err(TryLockError::WouldBlock) if Instant::now() >= deadline => {
                        timed_out = true;
                        break;
                    },
                    Err(TryLockError::WouldBlock) => std::thread::yield_now(),
                }
            }
        }
        #[cfg(test)]
        if let Some(delay) = *self.inner.close_and_drain_delay.lock().unwrap() {
            std::thread::sleep(delay);
        }
        drop(entries);
        let release_result = self.release_governor_evictable_bytes(released_device_bytes);
        match self.lock_until(deadline, OPERATION) {
            Ok(mut state) => state.companion_runtimes.clear(),
            Err(error) if error.fact_value("reason") == Some("resident_graph_cache_drain_timeout") => {
                timed_out = true;
            },
            Err(error) => return Err(error),
        }
        if timed_out || Instant::now() >= deadline {
            return Err(resident_graph_cache_drain_timeout(OPERATION));
        }
        release_result.map(|_| ())
    }

    #[cfg(test)]
    fn set_close_and_drain_delay_for_test(&self, delay: Duration) {
        *self.inner.close_and_drain_delay.lock().unwrap() = Some(delay);
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, GpuResidentGraphCacheState>> {
        self.inner.state.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_resident_graph_cache",
                "native GPU resident graph cache lock was poisoned",
            )
        })
    }

    fn lock_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, GpuResidentGraphCacheState>> {
        loop {
            match self.inner.state.try_lock() {
                Ok(state) => return Ok(state),
                Err(TryLockError::Poisoned(_)) => return self.lock(),
                Err(TryLockError::WouldBlock) if Instant::now() >= deadline => {
                    return Err(resident_graph_cache_drain_timeout(operation));
                },
                Err(TryLockError::WouldBlock) => std::thread::yield_now(),
            }
        }
    }

    fn handle_memory_pressure(&self, pressure: EvictableMemoryPressure) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(0);
        };
        if pressure.device_ordinal != governor.device_ordinal() {
            return Ok(0);
        }
        let mut state = self.lock()?;
        let eviction = state.evict_at_least(pressure.requested_evict_bytes);
        let evicted_device_bytes = eviction.device_bytes;
        drop(state);
        drop(eviction);
        self.release_governor_evictable_bytes(evicted_device_bytes)?;
        Ok(evicted_device_bytes)
    }

    fn governor_can_admit_after_cache_eviction(
        &self,
        state: &GpuResidentGraphCacheState,
        retained_device_bytes: u64,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        let evictable_for_capacity =
            state.evictable_bytes_until_fits(self.inner.max_retained_device_bytes, retained_device_bytes);
        let free_bytes = governor.free_bytes()?;
        if free_bytes.saturating_add(evictable_for_capacity) >= retained_device_bytes {
            return Ok(true);
        }
        match memory_grant {
            Some(memory_grant) => governor.grant_can_cover_evictable_bytes(memory_grant, retained_device_bytes),
            None => Ok(false),
        }
    }

    fn reserve_governor_evictable_bytes(
        &self,
        retained_device_bytes: u64,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        if governor.try_reserve_evictable_bytes(retained_device_bytes)? {
            return Ok(true);
        }
        match memory_grant {
            Some(memory_grant) => governor.try_reserve_evictable_bytes_from_grant(memory_grant, retained_device_bytes),
            None => Ok(false),
        }
    }

    fn release_governor_evictable_bytes(&self, released_device_bytes: u64) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(released_device_bytes);
        };
        governor.release_evictable_bytes(released_device_bytes)
    }
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphCacheState {
    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_access_epoch;
        self.next_access_epoch = self.next_access_epoch.saturating_add(1);
        epoch
    }

    fn copy_count_for_resident_key(&self, resident_key: &ResidentGraphKey) -> usize {
        self.available.keys().filter(|key| &key.resident_key == resident_key).count()
            + self.checked_out.keys().filter(|key| &key.resident_key == resident_key).count()
    }

    fn evict_until_fits(
        &mut self,
        max_retained_device_bytes: u64,
        incoming_retained_device_bytes: u64,
    ) -> GpuResidentGraphCacheEviction {
        let mut eviction = GpuResidentGraphCacheEviction::default();
        while self.retained_device_bytes.saturating_add(incoming_retained_device_bytes) > max_retained_device_bytes {
            let Some(evict_key) = self.lru_available_key() else {
                break;
            };
            if let Some(evicted) = self.available.remove(&evict_key) {
                let retained_device_bytes = evicted.retained_device_bytes;
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(retained_device_bytes);
                eviction.device_bytes = eviction.device_bytes.saturating_add(retained_device_bytes);
                eviction.entries.push(evicted);
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        eviction
    }

    fn evictable_bytes_until_fits(&self, max_retained_device_bytes: u64, incoming_retained_device_bytes: u64) -> u64 {
        let mut retained_device_bytes = self.retained_device_bytes;
        let mut evictable_device_bytes = 0u64;
        let mut candidates = self
            .available
            .iter()
            .filter(|(key, _entry)| !self.preparation_pins.contains_key(*key))
            .map(|(_key, entry)| (entry.last_access_epoch, entry.retained_device_bytes))
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(last_access_epoch, _bytes)| *last_access_epoch);
        for (_last_access_epoch, candidate_bytes) in candidates {
            if retained_device_bytes.saturating_add(incoming_retained_device_bytes) <= max_retained_device_bytes {
                break;
            }
            retained_device_bytes = retained_device_bytes.saturating_sub(candidate_bytes);
            evictable_device_bytes = evictable_device_bytes.saturating_add(candidate_bytes);
        }
        evictable_device_bytes
    }

    fn evict_at_least(&mut self, target_device_bytes: u64) -> GpuResidentGraphCacheEviction {
        let mut eviction = GpuResidentGraphCacheEviction::default();
        while eviction.device_bytes < target_device_bytes {
            let Some(evict_key) = self.lru_available_key() else {
                break;
            };
            if let Some(evicted) = self.available.remove(&evict_key) {
                let retained_device_bytes = evicted.retained_device_bytes;
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(retained_device_bytes);
                eviction.device_bytes = eviction.device_bytes.saturating_add(retained_device_bytes);
                eviction.entries.push(evicted);
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        eviction
    }

    fn lru_available_key(&self) -> Option<GpuResidentGraphCacheEntryKey> {
        self.available
            .iter()
            .filter(|(key, _entry)| !self.preparation_pins.contains_key(*key))
            .min_by_key(|(_key, entry)| entry.last_access_epoch)
            .map(|(key, _entry)| key.clone())
    }

    fn next_preparation_pin_id(&mut self) -> u64 {
        self.next_preparation_pin_id = self.next_preparation_pin_id.wrapping_add(1);
        if self.next_preparation_pin_id == 0 {
            self.next_preparation_pin_id = 1;
        }
        self.next_preparation_pin_id
    }

    fn consume_checkout_preparation_pin(
        &mut self,
        entry_key: &GpuResidentGraphCacheEntryKey,
        preparation_pin_id: Option<u64>,
    ) -> bool {
        match (self.preparation_pins.get(entry_key).copied(), preparation_pin_id) {
            (None, None) => true,
            (Some(expected), Some(actual)) if expected == actual => {
                self.preparation_pins.remove(entry_key);
                true
            },
            _ => false,
        }
    }
}

#[cfg(feature = "cugraph")]
fn runtime_lease_matches(lease: &cudf::runtime::RuntimeLeaseAnchor, runtime: &Arc<cudf::Runtime>) -> bool {
    runtime.matches_lease_anchor(lease)
}

#[cfg(feature = "cugraph")]
fn resident_graph_cache_closed(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "resident graph cache is closed")
        .with_fact("reason", "resident_graph_cache_closed")
}

#[cfg(feature = "cugraph")]
fn resident_graph_cache_drain_timeout(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::ServiceOverloaded, operation, "timed out draining the resident graph cache")
        .with_fact("reason", "resident_graph_cache_drain_timeout")
        .with_cleanup_deadline_elapsed()
}

#[cfg(feature = "cugraph")]
pub(crate) struct GpuResidentGraphPreparationPin {
    cache: Arc<GpuResidentGraphCacheInner>,
    entry_key: GpuResidentGraphCacheEntryKey,
    pin_id: u64,
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphPreparationPin {
    pub(crate) fn pin_id(&self) -> u64 {
        self.pin_id
    }

    pub(crate) fn resident_key(&self) -> &ResidentGraphKey {
        &self.entry_key.resident_key
    }
}

#[cfg(feature = "cugraph")]
pub(crate) enum GpuResidentGraphBuildJoin {
    Leader(Box<GpuResidentGraphBuildLeader>),
    Follower(GpuResidentGraphBuildFollower),
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuResidentGraphBuildWait {
    LeaderFinished,
    DeadlineExpired,
}

#[cfg(feature = "cugraph")]
pub(crate) struct GpuResidentGraphBuildLeader {
    cache: Arc<GpuResidentGraphCacheInner>,
    resident_key: ResidentGraphKey,
    flight: Arc<GpuResidentGraphBuildFlight>,
    finished: bool,
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphBuildLeader {
    fn finish_inner(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if let Ok(mut state) = self.cache.state.lock()
            && state.builds.get(&self.resident_key).is_some_and(|flight| Arc::ptr_eq(flight, &self.flight))
        {
            state.builds.remove(&self.resident_key);
        }
        if let Ok(mut state) = self.flight.state.lock() {
            state.finished = true;
            self.flight.changed.notify_all();
        }
    }
}

#[cfg(feature = "cugraph")]
impl Drop for GpuResidentGraphBuildLeader {
    fn drop(&mut self) {
        self.finish_inner();
    }
}

#[cfg(feature = "cugraph")]
pub(crate) struct GpuResidentGraphBuildFollower {
    flight: Arc<GpuResidentGraphBuildFlight>,
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphBuildFollower {
    pub(crate) fn wait(
        self,
        cancellation: &crate::runtime::ExecutionCancellation,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<GpuResidentGraphBuildWait> {
        const POLL: Duration = Duration::from_millis(10);
        let mut state = self.flight.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, operation, "resident graph build flight mutex was poisoned")
        })?;
        while !state.finished {
            cancellation.check(operation)?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(GpuResidentGraphBuildWait::DeadlineExpired);
            }
            let (next, _timeout) = self.flight.changed.wait_timeout(state, remaining.min(POLL)).map_err(|_error| {
                QueryError::new(QueryErrorCode::Internal, operation, "resident graph build flight mutex was poisoned")
            })?;
            state = next;
        }
        Ok(GpuResidentGraphBuildWait::LeaderFinished)
    }
}

#[cfg(feature = "cugraph")]
impl Drop for GpuResidentGraphPreparationPin {
    fn drop(&mut self) {
        if let Ok(mut state) = self.cache.state.lock()
            && state.preparation_pins.get(&self.entry_key) == Some(&self.pin_id)
        {
            state.preparation_pins.remove(&self.entry_key);
        }
    }
}

/// Exclusive checked-out parked graph.
#[cfg(feature = "cugraph")]
pub(crate) struct GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    cache: Arc<GpuResidentGraphCacheInner>,
    entry_key: GpuResidentGraphCacheEntryKey,
    retained_device_bytes: u64,
    parked: Option<cugraph::ParkedGraph<T>>,
    completed: bool,
    _transposition: PhantomData<T>,
}

#[cfg(feature = "cugraph")]
impl<T> GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    pub(crate) fn retained_device_bytes(&self) -> u64 {
        self.retained_device_bytes
    }

    pub(crate) fn take_parked(&mut self) -> cugraph::ParkedGraph<T> {
        self.parked.take().expect("resident graph checkout already consumed")
    }

    pub(crate) fn return_parked(mut self, parked: cugraph::ParkedGraph<T>) -> Result<()> {
        let mut state = self.cache.state.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_resident_graph_cache.return",
                "native GPU resident graph cache lock was poisoned",
            )
        })?;
        let Some(checked_out) = state.checked_out.remove(&self.entry_key) else {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_resident_graph_cache.return",
                "resident graph checkout was not registered",
            ));
        };
        if state.closed {
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(checked_out.retained_device_bytes);
            let governor = self.cache.governor.clone();
            self.completed = true;
            drop(state);
            drop(parked);
            if let Some(governor) = governor {
                governor.release_evictable_bytes(checked_out.retained_device_bytes)?;
            }
            return Ok(());
        }
        let last_access_epoch = state.next_epoch();
        state.available.insert(
            self.entry_key.clone(),
            GpuResidentGraphCacheEntry {
                graph: T::into_entry(parked),
                retained_device_bytes: checked_out.retained_device_bytes,
                last_access_epoch,
                runtime_lease: checked_out.runtime_lease,
            },
        );
        self.completed = true;
        Ok(())
    }
}

#[cfg(feature = "cugraph")]
impl<T> Drop for GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let Ok(mut state) = self.cache.state.lock() else {
            return;
        };
        let released =
            state.checked_out.remove(&self.entry_key).map_or(0, |checked_out| checked_out.retained_device_bytes);
        if released == 0 {
            return;
        }
        state.retained_device_bytes = state.retained_device_bytes.saturating_sub(released);
        let governor = self.cache.governor.clone();
        drop(state);
        if let Some(governor) = governor {
            let _ = governor.release_evictable_bytes(released);
        }
    }
}

// Cache behavior is retained as private P5 substrate. P2 test targets do not
// construct durable cache instances.
