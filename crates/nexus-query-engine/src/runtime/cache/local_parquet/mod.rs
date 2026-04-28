//! GPU-resident cache for decoded local-Parquet source results.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, TryLockError, Weak},
    time::Instant,
};

use arrow::datatypes::SchemaRef;

use crate::{
    error::{QueryError, Result},
    runtime::{
        ExecutionContext,
        gpu::governor::{EvictableMemoryPressure, MemoryGovernor},
        resident::{GpuLocalParquetCachedResult, GpuResultHandle},
    },
};

mod fingerprint;
mod key;

pub use key::GpuLocalParquetSourceCacheKey;

/// Result of attempting to retain a GPU source chunk in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLocalParquetSourceCacheInsert {
    /// Entry was inserted and may have evicted older entries.
    Inserted,
    /// Equivalent entry was already present; the existing handle is retained.
    AlreadyPresent,
    /// Entry was too large or protected entries prevented enough eviction.
    CapacitySkipped,
}

/// Accounting returned by a GPU-local Parquet source cache insert attempt.
///
/// `evicted_device_bytes` reports cache entries removed to make room, while
/// `retained_device_bytes` reports how many bytes the inserted or existing
/// entry contributes to cache residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuLocalParquetSourceCacheInsertResult {
    /// Whether the cache inserted the entry, found it already present, or
    /// skipped insertion because capacity could not be made available.
    pub(crate) outcome: GpuLocalParquetSourceCacheInsert,
    /// Device bytes released from evicted cache entries during this attempt.
    pub(crate) evicted_device_bytes: u64,
    /// Device bytes retained by the inserted or already-present entry.
    pub(crate) retained_device_bytes: u64,
}

#[derive(Clone)]
struct GpuLocalParquetSourceCacheEntry {
    value: GpuLocalParquetCachedResult,
    retained_device_bytes: u64,
    last_access_epoch: u64,
    in_use_count: u64,
    generation: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct GpuLocalParquetSourceCacheEntryKey {
    source_key: GpuLocalParquetSourceCacheKey,
    runtime_key: u64,
}

impl GpuLocalParquetSourceCacheEntryKey {
    fn new(source_key: GpuLocalParquetSourceCacheKey, runtime: &cudf::Runtime) -> Self {
        Self { source_key, runtime_key: runtime.affinity_id().get() }
    }

    fn for_context(source_key: &GpuLocalParquetSourceCacheKey, context: &ExecutionContext) -> Result<Self> {
        let runtime =
            context.local_parquet_cache_owner_runtime("nexus_query_engine.gpu_local_parquet_source_cache.entry_key")?;
        Ok(Self::new(source_key.clone(), &runtime))
    }
}

struct GpuLocalParquetCompanionRuntime {
    pooled_runtime_lease: cudf::runtime::RuntimeLeaseAnchor,
    companion: Weak<cudf::Runtime>,
}

struct GpuLocalParquetPendingInsert {
    retained_device_bytes: u64,
    generation: u64,
}

#[derive(Default)]
struct GpuLocalParquetSourceCacheState {
    cache: HashMap<GpuLocalParquetSourceCacheEntryKey, GpuLocalParquetSourceCacheEntry>,
    pending: HashMap<GpuLocalParquetSourceCacheEntryKey, GpuLocalParquetPendingInsert>,
    companion_runtimes: HashMap<u64, GpuLocalParquetCompanionRuntime>,
    retained_device_bytes: u64,
    next_access_epoch: u64,
    next_entry_generation: u64,
    closed: bool,
}

struct GpuLocalParquetSourceCacheInner {
    max_retained_device_bytes: u64,
    governor: Option<MemoryGovernor>,
    state: Mutex<GpuLocalParquetSourceCacheState>,
    #[cfg(test)]
    close_and_drain_delay: Mutex<Option<std::time::Duration>>,
}

impl Drop for GpuLocalParquetSourceCacheInner {
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

#[derive(Clone)]
pub(crate) struct GpuLocalParquetSourceCachePin {
    _inner: Arc<GpuLocalParquetSourceCachePinInner>,
}

struct GpuLocalParquetSourceCachePinInner {
    cache: Arc<GpuLocalParquetSourceCacheInner>,
    entry_key: GpuLocalParquetSourceCacheEntryKey,
    generation: u64,
}

impl Drop for GpuLocalParquetSourceCachePinInner {
    fn drop(&mut self) {
        if let Ok(mut state) = self.cache.state.lock()
            && let Some(entry) = state.cache.get_mut(&self.entry_key)
            && entry.generation == self.generation
        {
            entry.in_use_count = entry.in_use_count.saturating_sub(1);
        }
    }
}

/// LRU cache for GPU-resident local Parquet source results.
///
/// Capacity and retained sizes are device bytes. Cached handles remain
/// runtime/device-affine, and runtime identity is part of the internal cache
/// key. A lookup with a runtime that has not decoded the source yet misses
/// without evicting entries for other runtimes, while a schema mismatch is
/// treated as an invalid plan.
pub struct GpuLocalParquetSourceCache {
    inner: Arc<GpuLocalParquetSourceCacheInner>,
}

pub(crate) struct GpuLocalParquetSourceCachePinnedInsert {
    pub(crate) outcome: GpuLocalParquetSourceCacheInsert,
    pub(crate) handle: Option<GpuResultHandle>,
}

impl std::fmt::Debug for GpuLocalParquetSourceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.inner.state.lock().ok();
        f.debug_struct("GpuLocalParquetSourceCache")
            .field("max_retained_device_bytes", &self.inner.max_retained_device_bytes)
            .field("governor_backed", &self.is_governor_backed())
            .field("entry_count", &state.as_ref().map(|state| state.cache.len()))
            .field("retained_device_bytes", &state.as_ref().map(|state| state.retained_device_bytes))
            .finish()
    }
}

impl GpuLocalParquetSourceCache {
    /// Create an empty source cache with a retained-device-byte capacity.
    pub fn new(max_retained_device_bytes: u64) -> Self {
        Self::new_with_governor(max_retained_device_bytes, None)
    }

    /// Create an empty source cache whose retained bytes are charged to a
    /// device memory governor as evictable cache residency.
    pub fn new_governor_backed(max_retained_device_bytes: u64, governor: MemoryGovernor) -> Self {
        Self::new_with_governor(max_retained_device_bytes, Some(governor))
    }

    fn new_with_governor(max_retained_device_bytes: u64, governor: Option<MemoryGovernor>) -> Self {
        Self {
            inner: Arc::new(GpuLocalParquetSourceCacheInner {
                max_retained_device_bytes,
                governor,
                state: Mutex::new(GpuLocalParquetSourceCacheState::default()),
                #[cfg(test)]
                close_and_drain_delay: Mutex::new(None),
            }),
        }
    }

    /// Maximum retained device bytes allowed by this cache.
    pub fn max_retained_device_bytes(&self) -> u64 {
        self.inner.max_retained_device_bytes
    }

    /// Whether retained bytes are owned by the per-device memory governor's
    /// evictable ledger instead of by per-query reservation baselines.
    pub fn is_governor_backed(&self) -> bool {
        self.inner.governor.is_some()
    }

    /// Register this cache as a synchronous evictable-memory observer on its
    /// backing governor.
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

    /// Number of retained source-result entries.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.cache.len())
    }

    /// Whether the cache currently holds no source-result entries.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Observed retained device bytes across all cache entries.
    pub fn retained_device_bytes(&self) -> Result<u64> {
        Ok(self.lock()?.retained_device_bytes)
    }

    /// Close this cache, destroy every available entry, and release its
    /// governor evictable-byte ledger. Repeated calls are idempotent.
    #[cfg(test)]
    pub(crate) fn close_and_drain(&self) -> Result<()> {
        let mut state = self.lock()?;
        state.closed = true;
        let entries = std::mem::take(&mut state.cache);
        state.pending.clear();
        state.companion_runtimes.clear();
        let released_device_bytes = state.retained_device_bytes;
        state.retained_device_bytes = 0;
        drop(state);
        drop(entries);
        self.release_governor_evictable_bytes(released_device_bytes)?;
        Ok(())
    }

    /// Close and drain under one server-owned absolute shutdown deadline.
    pub(crate) fn close_and_drain_until(&self, deadline: Instant) -> Result<()> {
        const OPERATION: &str = "nexus_query_engine.gpu_local_parquet_source_cache.close_and_drain";
        let mut state = self.lock_until(deadline, OPERATION)?;
        state.closed = true;
        let entries = std::mem::take(&mut state.cache);
        state.pending.clear();
        state.companion_runtimes.clear();
        let released_device_bytes = state.retained_device_bytes;
        state.retained_device_bytes = 0;
        drop(state);

        #[cfg(test)]
        if let Some(delay) = *self.inner.close_and_drain_delay.lock().unwrap() {
            std::thread::sleep(delay);
        }
        drop(entries);
        let release_result = self.release_governor_evictable_bytes(released_device_bytes);
        if Instant::now() >= deadline {
            return Err(local_source_cache_drain_timeout(OPERATION));
        }
        release_result?;
        Ok(())
    }

    #[cfg(test)]
    fn set_close_and_drain_delay_for_test(&self, delay: std::time::Duration) {
        *self.inner.close_and_drain_delay.lock().unwrap() = Some(delay);
    }

    pub(crate) fn evict_at_least_protecting(
        &self,
        target_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Result<u64> {
        let mut state = self.lock()?;
        self.evict_at_least_locked(&mut state, target_device_bytes, protected_keys)
    }

    pub(crate) fn get(
        &self,
        key: &GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        schema: &SchemaRef,
    ) -> Result<Option<(GpuResultHandle, u64)>> {
        let entry_key = GpuLocalParquetSourceCacheEntryKey::for_context(key, context)?;
        let mut state = self.lock()?;
        if state.closed {
            return Err(local_parquet_source_cache_closed("nexus_query_engine.gpu_local_parquet_source_cache.get"));
        }
        let access_epoch = state.next_epoch();
        let Some(entry) = state.cache.get_mut(&entry_key) else {
            return Ok(None);
        };
        if !entry.value.can_reuse_for_source_cache(context, schema)? {
            let released = if entry.in_use_count == 0 { state.remove_entry(&entry_key) } else { 0 };
            drop(state);
            self.release_governor_evictable_bytes(released)?;
            return Ok(None);
        }
        entry.in_use_count = entry.in_use_count.saturating_add(1);
        entry.last_access_epoch = access_epoch;
        let pin = GpuLocalParquetSourceCachePin {
            _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                cache: Arc::clone(&self.inner),
                entry_key: entry_key.clone(),
                generation: entry.generation,
            }),
        };
        let value = entry.value.clone();
        let retained_device_bytes = entry.retained_device_bytes;
        drop(state);
        let handle = value.checkout(context, pin)?;
        Ok(Some((handle, retained_device_bytes)))
    }

    pub(crate) fn insert_if_absent_pinned(
        &self,
        context: &ExecutionContext,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
    ) -> Result<GpuLocalParquetSourceCachePinnedInsert> {
        let (insert, handle) = self.insert_if_absent_protecting_internal(
            key,
            context,
            handle,
            retained_device_bytes,
            &HashSet::new(),
            true,
        )?;
        Ok(GpuLocalParquetSourceCachePinnedInsert { outcome: insert.outcome, handle })
    }

    pub(crate) fn insert_if_absent_protecting(
        &self,
        context: &ExecutionContext,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Result<GpuLocalParquetSourceCacheInsertResult> {
        Ok(self
            .insert_if_absent_protecting_internal(key, context, handle, retained_device_bytes, protected_keys, false)?
            .0)
    }

    fn insert_if_absent_protecting_internal(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
        pin_returned_handle: bool,
    ) -> Result<(GpuLocalParquetSourceCacheInsertResult, Option<GpuResultHandle>)> {
        let entry_key = GpuLocalParquetSourceCacheEntryKey::for_context(&key, context)?;
        let mut state = self.lock()?;
        if state.closed {
            return Err(local_parquet_source_cache_closed("nexus_query_engine.gpu_local_parquet_source_cache.insert"));
        }
        let retained_before = state.retained_device_bytes;
        let access_epoch = state.next_epoch();
        if let Some(entry) = state.cache.get_mut(&entry_key) {
            let reusable = !pin_returned_handle || entry.value.can_reuse_for_source_cache_handle(&handle)?;
            if !reusable {
                return Ok((
                    GpuLocalParquetSourceCacheInsertResult {
                        outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                        evicted_device_bytes: 0,
                        retained_device_bytes: retained_before,
                    },
                    None,
                ));
            }
            entry.last_access_epoch = access_epoch;
            if pin_returned_handle {
                entry.in_use_count = entry.in_use_count.saturating_add(1);
                let pin = GpuLocalParquetSourceCachePin {
                    _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                        cache: Arc::clone(&self.inner),
                        entry_key: entry_key.clone(),
                        generation: entry.generation,
                    }),
                };
                let value = entry.value.clone();
                drop(state);
                return Ok((
                    GpuLocalParquetSourceCacheInsertResult {
                        outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                        evicted_device_bytes: 0,
                        retained_device_bytes: retained_before,
                    },
                    Some(value.checkout(context, pin)?),
                ));
            }
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                    evicted_device_bytes: 0,
                    retained_device_bytes: retained_before,
                },
                None,
            ));
        }
        if state.pending.contains_key(&entry_key) {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                    evicted_device_bytes: 0,
                    retained_device_bytes: retained_before,
                },
                pin_returned_handle.then_some(handle),
            ));
        }
        if handle.has_live_reservation() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.insert",
                "cannot retain a source cache entry that carries a query live-reservation",
            )
            .with_fact("reason", "query_live_reservation_in_cache_handle"));
        }
        if handle.has_source_cache_pin() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.insert",
                "cannot retain a source cache entry that already carries a source-cache pin",
            )
            .with_fact("reason", "source_cache_pin_in_cache_handle"));
        }
        if retained_device_bytes > self.inner.max_retained_device_bytes {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes: 0,
                    retained_device_bytes: retained_before,
                },
                None,
            ));
        }
        if !self.governor_can_admit_after_cache_eviction(&state, retained_device_bytes, protected_keys)? {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes: 0,
                    retained_device_bytes: state.retained_device_bytes,
                },
                None,
            ));
        }
        let evicted_device_bytes =
            state.evict_until_fits(self.inner.max_retained_device_bytes, retained_device_bytes, protected_keys);
        self.release_governor_evictable_bytes(evicted_device_bytes)?;
        if state.retained_device_bytes.saturating_add(retained_device_bytes) > self.inner.max_retained_device_bytes {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes,
                    retained_device_bytes: state.retained_device_bytes,
                },
                None,
            ));
        }
        if !self.reserve_governor_evictable_bytes(retained_device_bytes)? {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes,
                    retained_device_bytes: state.retained_device_bytes,
                },
                None,
            ));
        }
        let generation = state.next_generation();
        state.pending.insert(entry_key.clone(), GpuLocalParquetPendingInsert { retained_device_bytes, generation });
        state.retained_device_bytes = state.retained_device_bytes.saturating_add(retained_device_bytes);
        drop(state);

        let cache_runtime = match self
            .companion_runtime_for_context(context, "nexus_query_engine.gpu_local_parquet_source_cache.insert")
        {
            Ok(runtime) => runtime,
            Err(error) => {
                self.rollback_pending_insert(&entry_key, generation, retained_device_bytes)?;
                return Err(error);
            },
        };
        let value = match handle.copy_for_local_parquet_cache(cache_runtime) {
            Ok(value) => value,
            Err(error) => {
                self.rollback_pending_insert(&entry_key, generation, retained_device_bytes)?;
                if error.code() == crate::error::QueryErrorCode::CudfResourceExhausted {
                    return Ok((
                        GpuLocalParquetSourceCacheInsertResult {
                            outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                            evicted_device_bytes,
                            retained_device_bytes: self.retained_device_bytes()?,
                        },
                        None,
                    ));
                }
                return Err(error);
            },
        };
        let copied_device_bytes = match value.device_allocated_size_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.rollback_pending_insert(&entry_key, generation, retained_device_bytes)?;
                drop(value);
                return Err(error);
            },
        };
        if copied_device_bytes > retained_device_bytes {
            self.rollback_pending_insert(&entry_key, generation, retained_device_bytes)?;
            drop(value);
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes,
                    retained_device_bytes: self.retained_device_bytes()?,
                },
                None,
            ));
        }
        if copied_device_bytes < retained_device_bytes {
            self.shrink_pending_insert(&entry_key, generation, retained_device_bytes, copied_device_bytes)?;
        }
        let retained_device_bytes = copied_device_bytes;

        let mut state = self.lock()?;
        if state.closed {
            drop(state);
            drop(value);
            return Err(local_parquet_source_cache_closed("nexus_query_engine.gpu_local_parquet_source_cache.insert"));
        }
        let pending_matches = state.pending.get(&entry_key).is_some_and(|pending| pending.generation == generation);
        if !pending_matches {
            drop(state);
            drop(value);
            return Err(QueryError::new(
                crate::error::QueryErrorCode::Internal,
                "nexus_query_engine.gpu_local_parquet_source_cache.insert",
                "local Parquet cache pending insertion disappeared before commit",
            )
            .with_fact("reason", "local_parquet_cache_pending_insert_lost"));
        }
        state.pending.remove(&entry_key);
        let pin = pin_returned_handle.then(|| GpuLocalParquetSourceCachePin {
            _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                cache: Arc::clone(&self.inner),
                entry_key: entry_key.clone(),
                generation,
            }),
        });
        state.cache.insert(
            entry_key,
            GpuLocalParquetSourceCacheEntry {
                value: value.clone(),
                retained_device_bytes,
                last_access_epoch: access_epoch,
                in_use_count: usize::from(pin.is_some()) as u64,
                generation,
            },
        );
        let retained_after = state.retained_device_bytes;
        drop(state);
        let returned_handle = match pin {
            Some(pin) => Some(value.checkout(context, pin)?),
            None => None,
        };
        Ok((
            GpuLocalParquetSourceCacheInsertResult {
                outcome: GpuLocalParquetSourceCacheInsert::Inserted,
                evicted_device_bytes,
                retained_device_bytes: retained_after,
            },
            returned_handle,
        ))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, GpuLocalParquetSourceCacheState>> {
        self.inner.state.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache",
                "native GPU local Parquet source cache lock was poisoned",
            )
        })
    }

    fn companion_runtime_for_context(
        &self,
        context: &ExecutionContext,
        operation: &'static str,
    ) -> Result<Arc<cudf::Runtime>> {
        let pooled_runtime = context.local_parquet_cache_owner_runtime(operation)?;
        let runtime_key = pooled_runtime.affinity_id().get();
        let device_ordinal = pooled_runtime.device().ordinal();
        {
            let state = self.lock()?;
            if state.closed {
                return Err(local_parquet_source_cache_closed(operation));
            }
            if let Some(companion) = state.companion_runtimes.get(&runtime_key)
                && pooled_runtime.matches_lease_anchor(&companion.pooled_runtime_lease)
                && let Some(companion) = companion.companion.upgrade()
            {
                return Ok(companion);
            }
        }

        // A distinct stream is deliberate: cudf only deep-copies a frame when
        // its target execution binding differs. Sharing the pooled stream here
        // would leave column provenance retaining the inserting query.
        let companion = cudf::Runtime::builder()
            .with_device(pooled_runtime.device())
            .with_device_memory_manager()
            .build()
            .map_err(|error| {
                QueryError::from_cudf(operation, "failed to build local Parquet cache companion runtime", error)
                    .with_fact("device_ordinal", device_ordinal)
            })?
            .inherit_affinity_from(pooled_runtime.as_ref())
            .map_err(|error| {
                QueryError::from_cudf(operation, "failed to inherit local Parquet cache runtime affinity", error)
                    .with_fact("device_ordinal", device_ordinal)
            })?
            .inherit_retained_lifetime_from(pooled_runtime.as_ref())
            .map_err(|error| {
                QueryError::from_cudf(operation, "failed to retain service lifetime for local Parquet cache", error)
                    .with_fact("device_ordinal", device_ordinal)
            })?;
        let companion = Arc::new(companion);

        let mut state = self.lock()?;
        if state.closed {
            return Err(local_parquet_source_cache_closed(operation));
        }
        if let Some(existing) = state.companion_runtimes.get(&runtime_key)
            && pooled_runtime.matches_lease_anchor(&existing.pooled_runtime_lease)
            && let Some(existing) = existing.companion.upgrade()
        {
            return Ok(existing);
        }
        state.companion_runtimes.insert(
            runtime_key,
            GpuLocalParquetCompanionRuntime {
                pooled_runtime_lease: pooled_runtime.lease_anchor(),
                companion: Arc::downgrade(&companion),
            },
        );
        Ok(companion)
    }

    fn rollback_pending_insert(
        &self,
        entry_key: &GpuLocalParquetSourceCacheEntryKey,
        generation: u64,
        retained_device_bytes: u64,
    ) -> Result<()> {
        let released = {
            let mut state = self.lock()?;
            let matches = state.pending.get(entry_key).is_some_and(|pending| pending.generation == generation);
            if !matches {
                0
            } else {
                let pending = state.pending.remove(entry_key).expect("matching pending insertion is present");
                debug_assert_eq!(pending.retained_device_bytes, retained_device_bytes);
                state.retained_device_bytes = state.retained_device_bytes.saturating_sub(pending.retained_device_bytes);
                pending.retained_device_bytes
            }
        };
        self.release_governor_evictable_bytes(released)?;
        Ok(())
    }

    fn shrink_pending_insert(
        &self,
        entry_key: &GpuLocalParquetSourceCacheEntryKey,
        generation: u64,
        reserved_device_bytes: u64,
        copied_device_bytes: u64,
    ) -> Result<()> {
        debug_assert!(copied_device_bytes < reserved_device_bytes);
        let released = {
            let mut state = self.lock()?;
            let Some(pending) = state.pending.get_mut(entry_key) else {
                return Err(local_parquet_source_cache_closed(
                    "nexus_query_engine.gpu_local_parquet_source_cache.reconcile_insert",
                ));
            };
            if pending.generation != generation {
                return Err(QueryError::new(
                    crate::error::QueryErrorCode::Internal,
                    "nexus_query_engine.gpu_local_parquet_source_cache.reconcile_insert",
                    "local Parquet cache pending insertion generation changed during copy",
                )
                .with_fact("reason", "local_parquet_cache_pending_generation_changed"));
            }
            let released = reserved_device_bytes.saturating_sub(copied_device_bytes);
            pending.retained_device_bytes = copied_device_bytes;
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(released);
            released
        };
        self.release_governor_evictable_bytes(released)?;
        Ok(())
    }

    fn lock_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, GpuLocalParquetSourceCacheState>> {
        loop {
            match self.inner.state.try_lock() {
                Ok(state) => return Ok(state),
                Err(TryLockError::Poisoned(_)) => return self.lock(),
                Err(TryLockError::WouldBlock) if Instant::now() >= deadline => {
                    return Err(local_source_cache_drain_timeout(operation));
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
        if state.closed {
            return Ok(0);
        }
        self.evict_at_least_locked(&mut state, pressure.requested_evict_bytes, &HashSet::new())
    }

    fn evict_at_least_locked(
        &self,
        state: &mut GpuLocalParquetSourceCacheState,
        target_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Result<u64> {
        let evicted_device_bytes = state.evict_at_least(target_device_bytes, protected_keys);
        self.release_governor_evictable_bytes(evicted_device_bytes)?;
        Ok(evicted_device_bytes)
    }

    fn governor_can_admit_after_cache_eviction(
        &self,
        state: &GpuLocalParquetSourceCacheState,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        let evictable_for_capacity = state.evictable_bytes_until_fits(
            self.inner.max_retained_device_bytes,
            retained_device_bytes,
            protected_keys,
        );
        let free_bytes = governor.free_bytes()?;
        Ok(free_bytes.saturating_add(evictable_for_capacity) >= retained_device_bytes)
    }

    fn reserve_governor_evictable_bytes(&self, retained_device_bytes: u64) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        governor.try_reserve_evictable_bytes(retained_device_bytes)
    }

    fn release_governor_evictable_bytes(&self, released_device_bytes: u64) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(released_device_bytes);
        };
        governor.release_evictable_bytes(released_device_bytes)
    }
}

fn local_parquet_source_cache_closed(operation: &'static str) -> QueryError {
    QueryError::new(crate::error::QueryErrorCode::Cancelled, operation, "GPU local Parquet source cache is closed")
        .with_fact("reason", "local_parquet_source_cache_closed")
}

fn local_source_cache_drain_timeout(operation: &'static str) -> QueryError {
    QueryError::new(
        crate::error::QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out draining the GPU local Parquet source cache",
    )
    .with_fact("reason", "local_source_cache_drain_timeout")
    .with_cleanup_deadline_elapsed()
}

impl GpuLocalParquetSourceCacheState {
    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_access_epoch;
        self.next_access_epoch = self.next_access_epoch.saturating_add(1);
        epoch
    }

    fn next_generation(&mut self) -> u64 {
        let generation = self.next_entry_generation;
        self.next_entry_generation = self.next_entry_generation.saturating_add(1);
        generation
    }

    fn evict_until_fits(
        &mut self,
        max_retained_device_bytes: u64,
        incoming_retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> u64 {
        let mut evicted_device_bytes = 0u64;
        while self.retained_device_bytes.saturating_add(incoming_retained_device_bytes) > max_retained_device_bytes {
            let Some(evict_key) = self.lru_unprotected_key(protected_keys) else {
                break;
            };
            if let Some(evicted) = self.cache.remove(&evict_key) {
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(evicted.retained_device_bytes);
                evicted_device_bytes = evicted_device_bytes.saturating_add(evicted.retained_device_bytes);
            }
        }
        evicted_device_bytes
    }

    fn evictable_bytes_until_fits(
        &self,
        max_retained_device_bytes: u64,
        incoming_retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> u64 {
        let mut retained_device_bytes = self.retained_device_bytes;
        let mut evictable_device_bytes = 0u64;
        let mut candidates = self
            .cache
            .iter()
            .filter(|(key, entry)| entry.in_use_count == 0 && !protected_keys.contains(&key.source_key))
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

    fn evict_at_least(
        &mut self,
        target_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> u64 {
        let mut evicted_device_bytes = 0u64;
        while evicted_device_bytes < target_device_bytes {
            let Some(evict_key) = self.lru_unprotected_key(protected_keys) else {
                break;
            };
            if let Some(evicted) = self.cache.remove(&evict_key) {
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(evicted.retained_device_bytes);
                evicted_device_bytes = evicted_device_bytes.saturating_add(evicted.retained_device_bytes);
            }
        }
        evicted_device_bytes
    }

    fn remove_entry(&mut self, entry_key: &GpuLocalParquetSourceCacheEntryKey) -> u64 {
        let Some(removed) = self.cache.remove(entry_key) else {
            return 0;
        };
        self.retained_device_bytes = self.retained_device_bytes.saturating_sub(removed.retained_device_bytes);
        removed.retained_device_bytes
    }

    fn lru_unprotected_key(
        &self,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Option<GpuLocalParquetSourceCacheEntryKey> {
        self.cache
            .iter()
            .filter(|(key, entry)| entry.in_use_count == 0 && !protected_keys.contains(&key.source_key))
            .min_by_key(|(_key, entry)| entry.last_access_epoch)
            .map(|(key, _entry)| key.clone())
    }
}

impl GpuLocalParquetCachedResult {
    fn can_reuse_for_source_cache(&self, context: &ExecutionContext, schema: &SchemaRef) -> Result<bool> {
        if self.schema().as_ref() != schema.as_ref() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.reuse",
                "cached local Parquet source schema does not match the requested source schema",
            )
            .with_fact("reason", "schema_mismatch")
            .with_fact("cached_fields", self.schema().fields().len())
            .with_fact("requested_fields", schema.fields().len()));
        }
        let context_runtime =
            context.local_parquet_cache_owner_runtime("nexus_query_engine.gpu_local_parquet_source_cache.reuse")?;
        if !cudf::Runtime::shares_affinity(self.runtime(), &context_runtime) {
            return Ok(false);
        }
        if self.device_placement() != context.device_placement()
            || self.placement().device != context.runtime()?.device()
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn can_reuse_for_source_cache_handle(&self, requested: &GpuResultHandle) -> Result<bool> {
        if self.schema().as_ref() != requested.schema().as_ref() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.reuse",
                "cached local Parquet source schema does not match the requested source schema",
            )
            .with_fact("reason", "schema_mismatch")
            .with_fact("cached_fields", self.schema().fields().len())
            .with_fact("requested_fields", requested.schema().fields().len()));
        }
        if self.runtime().placement().workflow_id != requested.runtime().placement().workflow_id {
            return Ok(false);
        }
        if self.device_placement() != requested.device_placement()
            || self.placement().device != requested.placement().device
        {
            return Ok(false);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use cudf::DataFrame;

    use super::*;
    use crate::{
        runtime::{
            gpu::governor::MemoryGovernor,
            resident::{GpuDataFrame, GpuResultHandle},
        },
        source::LocalParquetSource,
    };

    struct CacheFixture {
        cache: Arc<GpuLocalParquetSourceCache>,
        context: ExecutionContext,
        governor: MemoryGovernor,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        batch: RecordBatch,
        retained_device_bytes: u64,
    }

    fn cache_fixture() -> CacheFixture {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let context = ExecutionContext::builder().with_shared_runtime(Arc::clone(&runtime)).build().unwrap();
        let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
        let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(32 * 1024 * 1024, governor.clone()));
        GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
        let frame = DataFrame::from_record_batch(runtime.as_ref(), &batch).unwrap();
        let retained_device_bytes = GpuDataFrame::new(frame.clone()).device_allocated_size_bytes().unwrap();
        let handle = GpuResultHandle::new(&context, GpuDataFrame::new(frame), Arc::clone(&schema)).unwrap();
        let source =
            LocalParquetSource::new("/tmp/datafusion-nexus-cache-close.parquet", schema, vec!["value".to_string()]);
        let key = GpuLocalParquetSourceCacheKey::from_source(&source);
        CacheFixture { cache, context, governor, key, handle, batch, retained_device_bytes }
    }

    #[test]
    fn populated_cache_close_is_idempotent_releases_governor_and_rejects_reuse() {
        let fixture = cache_fixture();
        let retained_bytes = fixture.retained_device_bytes;
        let reinsertion_handle = fixture.handle.clone();
        let inserted = fixture
            .cache
            .insert_if_absent_protecting(
                &fixture.context,
                fixture.key.clone(),
                fixture.handle,
                retained_bytes,
                &HashSet::new(),
            )
            .unwrap();
        assert_eq!(inserted.outcome, GpuLocalParquetSourceCacheInsert::Inserted);
        assert_eq!(fixture.cache.len().unwrap(), 1);
        assert_eq!(fixture.cache.retained_device_bytes().unwrap(), retained_bytes);
        assert_eq!(fixture.governor.evictable_bytes().unwrap(), retained_bytes);

        fixture.cache.close_and_drain_until(Instant::now() + std::time::Duration::from_secs(1)).unwrap();
        fixture.cache.close_and_drain().unwrap();
        assert!(fixture.cache.is_empty().unwrap());
        assert_eq!(fixture.cache.retained_device_bytes().unwrap(), 0);
        assert_eq!(fixture.governor.evictable_bytes().unwrap(), 0);

        let get_error = match fixture.cache.get(&fixture.key, &fixture.context, &fixture.batch.schema()) {
            Ok(_) => panic!("closed source cache must reject lookup"),
            Err(error) => error,
        };
        assert_eq!(get_error.fact_value("reason"), Some("local_parquet_source_cache_closed"));
        let insert_error = fixture
            .cache
            .insert_if_absent_protecting(
                &fixture.context,
                fixture.key,
                reinsertion_handle,
                retained_bytes,
                &HashSet::new(),
            )
            .unwrap_err();
        assert_eq!(insert_error.fact_value("reason"), Some("local_parquet_source_cache_closed"));
    }

    #[test]
    fn escaped_pinned_handle_survives_cache_close_without_retaining_governor_charge() {
        let fixture = cache_fixture();
        let retained_bytes = fixture.retained_device_bytes;
        let inserted = fixture
            .cache
            .insert_if_absent_pinned(&fixture.context, fixture.key, fixture.handle, retained_bytes)
            .unwrap();
        assert_eq!(inserted.outcome, GpuLocalParquetSourceCacheInsert::Inserted);
        let escaped = inserted.handle.expect("pinned insertion returns an escaped handle");
        assert_eq!(fixture.governor.evictable_bytes().unwrap(), retained_bytes);

        fixture.cache.close_and_drain().unwrap();
        assert!(fixture.cache.is_empty().unwrap());
        assert_eq!(fixture.governor.evictable_bytes().unwrap(), 0);
        assert_eq!(escaped.to_host_record_batch().unwrap(), fixture.batch);
        drop(escaped);
        fixture.cache.close_and_drain().unwrap();
    }

    #[test]
    fn deadline_overrun_during_entry_drop_reports_local_source_cache_timeout_after_cleanup() {
        let fixture = cache_fixture();
        let retained_bytes = fixture.retained_device_bytes;
        fixture
            .cache
            .insert_if_absent_protecting(&fixture.context, fixture.key, fixture.handle, retained_bytes, &HashSet::new())
            .unwrap();
        fixture.cache.set_close_and_drain_delay_for_test(std::time::Duration::from_millis(40));

        let error =
            fixture.cache.close_and_drain_until(Instant::now() + std::time::Duration::from_millis(10)).unwrap_err();

        assert_eq!(error.fact_value("reason"), Some("local_source_cache_drain_timeout"));
        assert!(fixture.cache.is_empty().unwrap());
        assert_eq!(fixture.governor.evictable_bytes().unwrap(), 0);
    }

    #[test]
    fn deadline_aware_cache_lock_acquisition_times_out_without_starting_new_grace() {
        let fixture = cache_fixture();
        let cache = Arc::clone(&fixture.cache);
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _state = cache.inner.state.lock().unwrap();
            locked_tx.send(()).unwrap();
            release_rx.recv().unwrap()
        });
        locked_rx.recv().unwrap();

        let error =
            fixture.cache.close_and_drain_until(Instant::now() + std::time::Duration::from_millis(10)).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("local_source_cache_drain_timeout"));
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        fixture.cache.close_and_drain().unwrap();
    }
}
