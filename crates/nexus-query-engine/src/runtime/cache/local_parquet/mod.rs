//! GPU-resident cache for decoded local-Parquet source results.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, Weak},
};

use arrow::datatypes::SchemaRef;

use crate::{
    error::{QueryError, Result},
    runtime::{
        ExecutionContext,
        gpu::governor::{EvictableMemoryPressure, MemoryGovernor, MemoryGrant},
        resident::GpuResultHandle,
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
    handle: GpuResultHandle,
    retained_device_bytes: u64,
    last_access_epoch: u64,
    in_use_count: u64,
    generation: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct GpuLocalParquetSourceCacheEntryKey {
    source_key: GpuLocalParquetSourceCacheKey,
    runtime_key: usize,
}

impl GpuLocalParquetSourceCacheEntryKey {
    fn new(source_key: GpuLocalParquetSourceCacheKey, runtime: &Arc<cudf::Runtime>) -> Self {
        Self { source_key, runtime_key: Arc::as_ptr(runtime) as usize }
    }

    fn for_context(source_key: &GpuLocalParquetSourceCacheKey, context: &ExecutionContext) -> Self {
        Self::new(source_key.clone(), &context.shared_runtime())
    }

    fn for_handle(source_key: GpuLocalParquetSourceCacheKey, handle: &GpuResultHandle) -> Self {
        Self::new(source_key, handle.runtime())
    }
}

#[derive(Default)]
struct GpuLocalParquetSourceCacheState {
    cache: HashMap<GpuLocalParquetSourceCacheEntryKey, GpuLocalParquetSourceCacheEntry>,
    retained_device_bytes: u64,
    next_access_epoch: u64,
    next_entry_generation: u64,
}

struct GpuLocalParquetSourceCacheInner {
    max_retained_device_bytes: u64,
    governor: Option<MemoryGovernor>,
    state: Mutex<GpuLocalParquetSourceCacheState>,
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
        let entry_key = GpuLocalParquetSourceCacheEntryKey::for_context(key, context);
        let mut state = self.lock()?;
        let access_epoch = state.next_epoch();
        let Some(entry) = state.cache.get_mut(&entry_key) else {
            return Ok(None);
        };
        if !entry.handle.can_reuse_for_source_cache(context, schema)? {
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
        let handle = entry.handle.clone().with_source_cache_pin(pin);
        let retained_device_bytes = entry.retained_device_bytes;
        Ok(Some((handle, retained_device_bytes)))
    }

    pub(crate) fn insert_if_absent_from_grant_pinned(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<GpuLocalParquetSourceCachePinnedInsert> {
        let (insert, handle) = self.insert_if_absent_protecting_internal(
            key,
            handle,
            retained_device_bytes,
            &HashSet::new(),
            memory_grant,
            true,
        )?;
        Ok(GpuLocalParquetSourceCachePinnedInsert { outcome: insert.outcome, handle })
    }

    pub(crate) fn insert_if_absent_protecting(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<GpuLocalParquetSourceCacheInsertResult> {
        Ok(self
            .insert_if_absent_protecting_internal(
                key,
                handle,
                retained_device_bytes,
                protected_keys,
                memory_grant,
                false,
            )?
            .0)
    }

    fn insert_if_absent_protecting_internal(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
        memory_grant: Option<&MemoryGrant>,
        pin_returned_handle: bool,
    ) -> Result<(GpuLocalParquetSourceCacheInsertResult, Option<GpuResultHandle>)> {
        let entry_key = GpuLocalParquetSourceCacheEntryKey::for_handle(key, &handle);
        let mut state = self.lock()?;
        let retained_before = state.retained_device_bytes;
        let access_epoch = state.next_epoch();
        if let Some(entry) = state.cache.get_mut(&entry_key) {
            let reusable = !pin_returned_handle || entry.handle.can_reuse_for_source_cache_handle(&handle)?;
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
            let handle = if pin_returned_handle {
                entry.in_use_count = entry.in_use_count.saturating_add(1);
                Some(entry.handle.clone().with_source_cache_pin(GpuLocalParquetSourceCachePin {
                    _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                        cache: Arc::clone(&self.inner),
                        entry_key: entry_key.clone(),
                        generation: entry.generation,
                    }),
                }))
            } else {
                None
            };
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                    evicted_device_bytes: 0,
                    retained_device_bytes: retained_before,
                },
                handle,
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
        if !self.governor_can_admit_after_cache_eviction(&state, retained_device_bytes, protected_keys, memory_grant)? {
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
        if !self.reserve_governor_evictable_bytes(retained_device_bytes, memory_grant)? {
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
        let pin = pin_returned_handle.then(|| GpuLocalParquetSourceCachePin {
            _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                cache: Arc::clone(&self.inner),
                entry_key: entry_key.clone(),
                generation,
            }),
        });
        let returned_handle = pin.clone().map(|pin| handle.clone().with_source_cache_pin(pin));
        state.cache.insert(
            entry_key,
            GpuLocalParquetSourceCacheEntry {
                handle,
                retained_device_bytes,
                last_access_epoch: access_epoch,
                in_use_count: if pin.is_some() { 1 } else { 0 },
                generation,
            },
        );
        state.retained_device_bytes = state.retained_device_bytes.saturating_add(retained_device_bytes);
        Ok((
            GpuLocalParquetSourceCacheInsertResult {
                outcome: GpuLocalParquetSourceCacheInsert::Inserted,
                evicted_device_bytes,
                retained_device_bytes: state.retained_device_bytes,
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

    fn handle_memory_pressure(&self, pressure: EvictableMemoryPressure) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(0);
        };
        if pressure.device_ordinal != governor.device_ordinal() {
            return Ok(0);
        }
        let mut state = self.lock()?;
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
        memory_grant: Option<&MemoryGrant>,
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

impl GpuResultHandle {
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
        let context_runtime = context.shared_runtime();
        if !Arc::ptr_eq(self.runtime(), &context_runtime) {
            return Ok(false);
        }
        if self.device_placement() != context.device_placement()
            || self.placement().device != context.runtime().device()
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn can_reuse_for_source_cache_handle(&self, requested: &Self) -> Result<bool> {
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
        if !Arc::ptr_eq(self.runtime(), requested.runtime()) {
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
