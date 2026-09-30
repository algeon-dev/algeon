//! GPU-resident decoded local-Parquet source cache.
//!
//! A promotion always copies the attempt-owned frame into the one shared
//! device cache allocation domain. The cache pre-reserves its ledger charge
//! before that copy, so an allocation failure or cancellation cannot leave a
//! copied allocation uncharged. The source handle is borrowed throughout: the
//! attempt remains its sole owner when retention is skipped or fails.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt, mem,
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::Duration,
};

use arrow::datatypes::SchemaRef;
use query_planning::error::{QueryError, QueryErrorCode, Result};

pub(crate) use crate::runtime::source_identity::GpuLocalParquetSourceCacheKey;
use crate::runtime::{
    ExecutionContext,
    cache::{
        CacheResidency, CacheResidencyChargeTransaction, CacheResidencyDestruction, CacheResidencyPin,
        CacheResidencyState, CacheSourceActivity, CacheSourceEventSnapshot, cache_observation_fingerprint,
    },
    gpu::DeviceLedger,
    resident::{GpuLocalParquetCachedResult, GpuResultHandle},
};

const OPERATION: &str = "query_engine.gpu_local_parquet_source_cache";
/// Cache events are only a bounded diagnostic projection.  Cache residency
/// remains exclusively owned by the device ledger and allocation domain.
const MAX_SOURCE_EVENT_SNAPSHOTS: usize = 256;

/// Result of attempting to promote an attempt-owned decoded source frame.
///
/// Allocation failure is deliberately distinct from [`Self::CapacitySkipped`]:
/// the former means capacity was reserved before the copy but the cache-domain
/// allocator could not complete it; the latter performs no device copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuLocalParquetSourceCachePromotion {
    /// A new cache-owned copy was retained.
    Inserted,
    /// An equivalent entry was already present or being promoted by another
    /// attempt. The caller's source frame remains attempt-owned.
    AlreadyPresent,
    /// The aggregate cache cap could not reserve capacity, even after all
    /// evictable LRU entries were retired.
    CapacitySkipped,
    /// Cache-domain allocation failed after the pre-reservation. The original
    /// attempt-owned frame remains the only usable source result.
    AllocationFailed,
}

/// Stable facts for one source-cache promotion attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct GpuLocalParquetSourceCachePromotionResult {
    pub(crate) outcome: GpuLocalParquetSourceCachePromotion,
    pub(crate) retained_device_bytes: u64,
    pub(crate) evicted_device_bytes: u64,
    pub(crate) allocation_failure_code: Option<QueryErrorCode>,
}

impl GpuLocalParquetSourceCachePromotionResult {
    fn inserted(retained_device_bytes: u64, evicted_device_bytes: u64) -> Self {
        Self {
            outcome: GpuLocalParquetSourceCachePromotion::Inserted,
            retained_device_bytes,
            evicted_device_bytes,
            allocation_failure_code: None,
        }
    }

    fn already_present(retained_device_bytes: u64) -> Self {
        Self {
            outcome: GpuLocalParquetSourceCachePromotion::AlreadyPresent,
            retained_device_bytes,
            evicted_device_bytes: 0,
            allocation_failure_code: None,
        }
    }

    fn capacity_skipped(evicted_device_bytes: u64) -> Self {
        Self {
            outcome: GpuLocalParquetSourceCachePromotion::CapacitySkipped,
            retained_device_bytes: 0,
            evicted_device_bytes,
            allocation_failure_code: None,
        }
    }

    fn allocation_failed(evicted_device_bytes: u64, code: QueryErrorCode) -> Self {
        Self {
            outcome: GpuLocalParquetSourceCachePromotion::AllocationFailed,
            retained_device_bytes: 0,
            evicted_device_bytes,
            allocation_failure_code: Some(code),
        }
    }
}

struct GpuLocalParquetSourceCacheEntry {
    value: Arc<GpuLocalParquetCachedResult>,
    residency: CacheResidency,
    last_access_epoch: u64,
}

#[derive(Default)]
struct GpuLocalParquetSourceCacheState {
    entries: HashMap<GpuLocalParquetSourceCacheKey, GpuLocalParquetSourceCacheEntry>,
    /// Source reads currently building a candidate value.  This is separate
    /// from `promotions`: it covers the decode itself so waiters do not each
    /// perform the same cache-cold Parquet read before discovering a shared
    /// entry at promotion time.
    loads: HashMap<GpuLocalParquetSourceCacheKey, u64>,
    promotions: HashSet<GpuLocalParquetSourceCacheKey>,
    next_access_epoch: u64,
    next_load_id: u64,
    lookup_hits: u64,
    lookup_misses: u64,
    promotions_inserted: u64,
    promotions_coalesced: u64,
    promotions_capacity_skipped: u64,
    promotions_allocation_failed: u64,
    promoted_device_bytes: u64,
    eviction_batches: u64,
    evicted_device_bytes: u64,
    next_event_sequence: u64,
    recent_events: VecDeque<CacheSourceEventSnapshot>,
    closed: bool,
}

impl GpuLocalParquetSourceCacheState {
    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_access_epoch;
        self.next_access_epoch = self.next_access_epoch.saturating_add(1);
        epoch
    }

    fn record_event(
        &mut self,
        device_ordinal: usize,
        allocation_domain_id: u64,
        correlation_fingerprint: Option<String>,
        key: &GpuLocalParquetSourceCacheKey,
        action: &'static str,
        retained_device_bytes: u64,
    ) {
        let event = CacheSourceEventSnapshot {
            sequence: self.next_event_sequence,
            device_ordinal,
            allocation_domain_id,
            correlation_fingerprint,
            cache_key_fingerprint: cache_observation_fingerprint(key.event_fingerprint_source()),
            action,
            retained_device_bytes,
        };
        self.next_event_sequence = self.next_event_sequence.saturating_add(1);
        if self.recent_events.len() == MAX_SOURCE_EVENT_SNAPSHOTS {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(event);
    }
}

struct GpuLocalParquetSourceCacheInner {
    allocation_lease: Arc<cudf::CacheAllocationLease>,
    ledger: DeviceLedger,
    state: Mutex<GpuLocalParquetSourceCacheState>,
    load_complete: Condvar,
}

/// The result of claiming one cache-key read before decoded source work
/// starts.  Only this cache can construct or complete a claim, so a stale
/// caller cannot clear a newer load for the same source key.
#[doc(hidden)]
pub struct GpuLocalParquetSourceCacheLoadClaim {
    cache: Arc<GpuLocalParquetSourceCacheInner>,
    key: GpuLocalParquetSourceCacheKey,
    id: u64,
    active: bool,
}

impl GpuLocalParquetSourceCacheLoadClaim {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for GpuLocalParquetSourceCacheLoadClaim {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Ok(mut state) = self.cache.state.lock() else {
            // A poisoned source-cache state already rejects future callers;
            // no best-effort wakeup can restore a trustworthy claim map.
            return;
        };
        if state.loads.get(&self.key).copied() != Some(self.id) {
            return;
        }
        state.loads.remove(&self.key);
        state.record_event(
            self.cache.ledger.device_ordinal(),
            self.cache.allocation_lease.domain_id().get(),
            None,
            &self.key,
            "load_abandoned",
            0,
        );
        // Do not hold the cache state mutex while followers retry and elect a
        // new leader after an unwind or another abnormal leader exit.
        drop(state);
        self.cache.load_complete.notify_all();
    }
}

/// A cache hit is returned as an ordinary query-affine result handle.  A
/// cold leader owns the matching claim until it promotes or abandons the
/// decoded frame; followers wait for that outcome and then retry lookup.
#[doc(hidden)]
pub enum GpuLocalParquetSourceCacheLoad {
    Hit(GpuResultHandle),
    Leader(GpuLocalParquetSourceCacheLoadClaim),
}

/// Pin held by every cache checkout and every transitive derived frame.
///
/// The shared residency pin is deliberately non-cloneable. This wrapper makes
/// a cloned result retain the same one lease; the state machine sees one final
/// release only after the final result clone is gone.
#[derive(Clone)]
pub(crate) struct GpuLocalParquetSourceCachePin {
    _inner: Arc<CacheResidencyPin>,
}

impl GpuLocalParquetSourceCachePin {
    fn new(pin: CacheResidencyPin) -> Self {
        Self { _inner: Arc::new(pin) }
    }
}

/// Per-device cache of decoded local-Parquet results.
///
/// Construction requires a device cache allocation lease and its matching
/// ledger. There is no unaccounted constructor and no independent source-cache
/// capacity.
#[doc(hidden)]
pub struct GpuLocalParquetSourceCache {
    inner: Arc<GpuLocalParquetSourceCacheInner>,
}

impl Clone for GpuLocalParquetSourceCache {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl fmt::Debug for GpuLocalParquetSourceCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.inner.state.lock().ok();
        f.debug_struct("GpuLocalParquetSourceCache")
            .field("cache_domain_id", &self.inner.allocation_lease.domain_id())
            .field("device_ordinal", &self.inner.ledger.device_ordinal())
            .field("entry_count", &state.as_ref().map(|state| state.entries.len()))
            .field("promotion_count", &state.as_ref().map(|state| state.promotions.len()))
            .finish()
    }
}

impl GpuLocalParquetSourceCache {
    /// Creates one accounted source-cache subdomain on an already-created
    /// cache allocation domain.
    pub(crate) fn new(allocation_lease: cudf::CacheAllocationLease, ledger: DeviceLedger) -> Result<Self> {
        let domain = allocation_lease.domain();
        if domain.device().ordinal() != ledger.device_ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "source-cache allocation domain device does not match its ledger",
            )
            .with_fact("cache_domain_device", domain.device().ordinal())
            .with_fact("ledger_device", ledger.device_ordinal()));
        }
        if domain.capacity_bytes() != ledger.profile().cache_cap_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "source-cache allocation domain capacity does not match the aggregate ledger cache cap",
            )
            .with_fact("cache_domain_capacity_bytes", domain.capacity_bytes())
            .with_fact("ledger_cache_cap_bytes", ledger.profile().cache_cap_bytes));
        }
        if domain.capacity_bytes() == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "zero-capacity source-cache allocation domains are forbidden",
            )
            .with_fact("reason", "zero_cache_capacity"));
        }
        Ok(Self {
            inner: Arc::new(GpuLocalParquetSourceCacheInner {
                allocation_lease: Arc::new(allocation_lease),
                ledger,
                state: Mutex::new(GpuLocalParquetSourceCacheState::default()),
                load_complete: Condvar::new(),
            }),
        })
    }

    /// Stable allocation-domain identity used by this cache.
    #[cfg(all(test, feature = "cugraph"))]
    pub(crate) fn allocation_domain_id(&self) -> cudf::CacheAllocationDomainId {
        self.inner.allocation_lease.domain_id()
    }

    /// Number of retained entries. A pending promotion is not an entry.
    #[cfg(test)]
    #[doc(hidden)]
    pub(crate) fn len(&self) -> Result<usize> {
        Ok(self.lock("len")?.entries.len())
    }

    /// Exact cache-owned bytes currently represented by completed entries.
    #[cfg(test)]
    pub(crate) fn retained_device_bytes(&self) -> Result<u64> {
        self.lock("retained_device_bytes")?.entries.values().try_fold(0u64, |sum, entry| {
            sum.checked_add(entry.residency.bytes()).ok_or_else(|| {
                QueryError::new(QueryErrorCode::Internal, OPERATION, "source-cache retained byte total overflowed")
                    .with_fact("reason", "source_cache_retained_bytes_overflow")
            })
        })
    }

    /// Cumulative activity plus the live key count.  These counters describe
    /// cache work only; byte residency remains the authoritative ledger and
    /// allocation-domain observations published by the service.
    pub(crate) fn activity(&self) -> Result<CacheSourceActivity> {
        let state = self.lock("activity")?;
        Ok(CacheSourceActivity {
            entry_count: state.entries.len(),
            promotions_in_progress: state.promotions.len(),
            lookup_hits: state.lookup_hits,
            lookup_misses: state.lookup_misses,
            promotions_inserted: state.promotions_inserted,
            promotions_coalesced: state.promotions_coalesced,
            promotions_capacity_skipped: state.promotions_capacity_skipped,
            promotions_allocation_failed: state.promotions_allocation_failed,
            promoted_device_bytes: state.promoted_device_bytes,
            eviction_batches: state.eviction_batches,
            evicted_device_bytes: state.evicted_device_bytes,
        })
    }

    /// Returns the bounded cache-owner event projection for this ordinal.
    /// Sequence numbers are local to the allocation domain and let an
    /// observer retain the after-snapshot delta without reconstructing cache
    /// behavior from aggregate counters.
    pub(crate) fn event_snapshots(&self) -> Result<Vec<CacheSourceEventSnapshot>> {
        Ok(self.lock("event_snapshots")?.recent_events.iter().cloned().collect())
    }

    /// Atomically checks out a cached source frame or claims the one cold
    /// read permitted for `key` in this ordinal-local cache domain.
    ///
    /// This is intentionally above `promote`: coalescing only completed
    /// promotions prevents duplicate cache charges, but it still lets every
    /// concurrent caller decode the same Parquet bytes.  A load claim closes
    /// that gap and gives the source path real build-wait behavior.
    #[doc(hidden)]
    pub fn checkout_or_claim(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        schema: &SchemaRef,
    ) -> Result<GpuLocalParquetSourceCacheLoad> {
        const OPERATION: &str = "query_engine.gpu_local_parquet_source_cache.checkout_or_claim";
        context.check_cancellation(OPERATION)?;
        let correlation_fingerprint =
            context.query_id().map(|query_id| cache_observation_fingerprint(query_id.as_str()));
        let mut state = self.lock("checkout_or_claim")?;
        let mut counted_lookup = false;
        let mut counted_follower = false;
        loop {
            if state.closed {
                return Err(source_cache_closed(OPERATION));
            }
            let access_epoch = state.next_epoch();
            if state.entries.contains_key(&key) {
                let hit = {
                    let entry = state.entries.get_mut(&key).expect("entry existence was checked above");
                    if entry.value.schema().as_ref() != schema.as_ref() {
                        return Err(QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            OPERATION,
                            "cached local-Parquet schema does not match the requested source schema",
                        )
                        .with_fact("reason", "schema_mismatch")
                        .with_fact("cached_fields", entry.value.schema().fields().len())
                        .with_fact("requested_fields", schema.fields().len()));
                    }
                    match entry.residency.try_pin()? {
                        Some(pin) => {
                            entry.last_access_epoch = access_epoch;
                            Some((GpuLocalParquetSourceCachePin::new(pin), Arc::clone(&entry.value)))
                        },
                        None => None,
                    }
                };
                let Some((pin, value)) = hit else {
                    if !counted_lookup {
                        state.lookup_misses = state.lookup_misses.saturating_add(1);
                        counted_lookup = true;
                    }
                    continue;
                };
                if !counted_lookup {
                    state.lookup_hits = state.lookup_hits.saturating_add(1);
                }
                state.record_event(
                    self.inner.ledger.device_ordinal(),
                    self.inner.allocation_lease.domain_id().get(),
                    correlation_fingerprint,
                    &key,
                    "load_hit",
                    0,
                );
                // Release the state mutex before checkout so a cache-copy or
                // cancellation callback cannot extend this ownership window.
                drop(state);
                let handle = value.checkout(context, pin)?;
                return Ok(GpuLocalParquetSourceCacheLoad::Hit(handle));
            }

            if !counted_lookup {
                state.lookup_misses = state.lookup_misses.saturating_add(1);
                counted_lookup = true;
            }
            if state.loads.contains_key(&key) {
                if !counted_follower {
                    state.promotions_coalesced = state.promotions_coalesced.saturating_add(1);
                    state.record_event(
                        self.inner.ledger.device_ordinal(),
                        self.inner.allocation_lease.domain_id().get(),
                        correlation_fingerprint.clone(),
                        &key,
                        "load_follower",
                        0,
                    );
                    counted_follower = true;
                }
                let (next_state, _) =
                    self.inner.load_complete.wait_timeout(state, Duration::from_millis(10)).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            OPERATION,
                            "source-cache load wait mutex was poisoned",
                        )
                        .with_fact("reason", "source_cache_load_wait_mutex_poisoned")
                    })?;
                state = next_state;
                context.check_cancellation(OPERATION)?;
                continue;
            }

            let id = state.next_load_id;
            state.next_load_id = state.next_load_id.saturating_add(1);
            let previous = state.loads.insert(key.clone(), id);
            debug_assert!(previous.is_none(), "a cache-key load claim is unique per ordinal domain");
            state.record_event(
                self.inner.ledger.device_ordinal(),
                self.inner.allocation_lease.domain_id().get(),
                correlation_fingerprint,
                &key,
                "load_leader",
                0,
            );
            return Ok(GpuLocalParquetSourceCacheLoad::Leader(GpuLocalParquetSourceCacheLoadClaim {
                cache: Arc::clone(&self.inner),
                key,
                id,
                active: true,
            }));
        }
    }

    /// Promotes a successfully decoded leader frame, then wakes every
    /// same-key follower.  The wake happens on all promotion outcomes: a
    /// skipped/failed promotion elects at most one next leader rather than
    /// leaving waiters blocked or charging duplicates.
    #[doc(hidden)]
    pub fn promote_load(
        &self,
        mut claim: GpuLocalParquetSourceCacheLoadClaim,
        context: &ExecutionContext,
        source: &GpuResultHandle,
    ) -> Result<GpuLocalParquetSourceCachePromotionResult> {
        let promotion = self.promote(claim.key.clone(), context, source);
        self.complete_load(&claim)?;
        claim.disarm();
        promotion
    }

    /// Releases a load claim when the source decode itself failed before a
    /// cache promotion could be attempted.
    #[doc(hidden)]
    pub fn abandon_load(&self, mut claim: GpuLocalParquetSourceCacheLoadClaim) -> Result<()> {
        self.complete_load(&claim)?;
        claim.disarm();
        Ok(())
    }

    /// Looks up and pins a cache entry before copying it into `context`.
    #[cfg(test)]
    pub(crate) fn get(
        &self,
        key: &GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        schema: &SchemaRef,
    ) -> Result<Option<(GpuResultHandle, u64)>> {
        context.check_cancellation("query_engine.gpu_local_parquet_source_cache.get")?;
        let (pin, value, retained_device_bytes) = {
            let mut state = self.lock("get")?;
            if state.closed {
                return Err(source_cache_closed("query_engine.gpu_local_parquet_source_cache.get"));
            }
            let access_epoch = state.next_epoch();
            let lookup = {
                let Some(entry) = state.entries.get_mut(key) else {
                    state.lookup_misses = state.lookup_misses.saturating_add(1);
                    return Ok(None);
                };
                if entry.value.schema().as_ref() != schema.as_ref() {
                    return Err(QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.gpu_local_parquet_source_cache.get",
                        "cached local-Parquet schema does not match the requested source schema",
                    )
                    .with_fact("reason", "schema_mismatch")
                    .with_fact("cached_fields", entry.value.schema().fields().len())
                    .with_fact("requested_fields", schema.fields().len()));
                }
                let Some(pin) = entry.residency.try_pin()? else {
                    state.lookup_misses = state.lookup_misses.saturating_add(1);
                    return Ok(None);
                };
                entry.last_access_epoch = access_epoch;
                (pin, Arc::clone(&entry.value), entry.residency.bytes())
            };
            state.lookup_hits = state.lookup_hits.saturating_add(1);
            (GpuLocalParquetSourceCachePin::new(lookup.0), lookup.1, lookup.2)
        };
        match value.checkout(context, pin) {
            Ok(handle) => Ok(Some((handle, retained_device_bytes))),
            Err(error) => Err(error),
        }
    }

    fn complete_load(&self, claim: &GpuLocalParquetSourceCacheLoadClaim) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &claim.cache) {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.gpu_local_parquet_source_cache.complete_load",
                "source-cache load claim belongs to another cache domain",
            )
            .with_fact("reason", "source_cache_load_claim_foreign"));
        }
        let mut state = self.lock("complete_load")?;
        let Some(id) = state.loads.get(&claim.key).copied() else {
            if state.closed {
                // `close_and_drain` retired every outstanding claim and woke
                // its waiters; a late leader completion is therefore benign.
                return Ok(());
            }
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.gpu_local_parquet_source_cache.complete_load",
                "source-cache load claim was not active at completion",
            )
            .with_fact("reason", "source_cache_load_claim_missing"));
        };
        if id != claim.id {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.gpu_local_parquet_source_cache.complete_load",
                "source-cache load claim identity changed before completion",
            )
            .with_fact("reason", "source_cache_load_claim_changed"));
        }
        state.loads.remove(&claim.key);
        // Release the mutex before waking waiters so they can immediately retry their lookup.
        drop(state);
        self.inner.load_complete.notify_all();
        Ok(())
    }

    /// Promotes a decoded attempt-owned source frame by making a cache-domain
    /// copy. An active attempt allocation domain is the source of this copy,
    /// never a rejection condition.
    pub(crate) fn promote(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        source: &GpuResultHandle,
    ) -> Result<GpuLocalParquetSourceCachePromotionResult> {
        const PROMOTION_OPERATION: &str = "query_engine.gpu_local_parquet_source_cache.promote";
        context.check_cancellation(PROMOTION_OPERATION)?;
        let reservation_bytes = source.device_allocated_size_bytes()?;

        {
            let mut state = self.lock("promote.claim")?;
            if state.closed {
                return Err(source_cache_closed(PROMOTION_OPERATION));
            }
            if let Some(retained_device_bytes) = state.entries.get(&key).map(|entry| entry.residency.bytes()) {
                state.promotions_coalesced = state.promotions_coalesced.saturating_add(1);
                state.record_event(
                    self.inner.ledger.device_ordinal(),
                    self.inner.allocation_lease.domain_id().get(),
                    context.query_id().map(|query_id| cache_observation_fingerprint(query_id.as_str())),
                    &key,
                    "promotion_already_present",
                    retained_device_bytes,
                );
                return Ok(GpuLocalParquetSourceCachePromotionResult::already_present(retained_device_bytes));
            }
            if !state.promotions.insert(key.clone()) {
                state.promotions_coalesced = state.promotions_coalesced.saturating_add(1);
                state.record_event(
                    self.inner.ledger.device_ordinal(),
                    self.inner.allocation_lease.domain_id().get(),
                    context.query_id().map(|query_id| cache_observation_fingerprint(query_id.as_str())),
                    &key,
                    "promotion_already_present",
                    0,
                );
                return Ok(GpuLocalParquetSourceCachePromotionResult::already_present(0));
            }
        }

        let result = self.promote_claimed(key.clone(), context, source, reservation_bytes);
        if result.is_err() {
            self.clear_promotion(&key)?;
            let mut state = self.lock("promote.error_event")?;
            state.record_event(
                self.inner.ledger.device_ordinal(),
                self.inner.allocation_lease.domain_id().get(),
                context.query_id().map(|query_id| cache_observation_fingerprint(query_id.as_str())),
                &key,
                "promotion_error",
                0,
            );
        } else if let Ok(promotion) = result.as_ref() {
            self.record_promotion(&key, context, *promotion)?;
        }
        result
    }

    fn record_promotion(
        &self,
        key: &GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        promotion: GpuLocalParquetSourceCachePromotionResult,
    ) -> Result<()> {
        let mut state = self.lock("record_promotion")?;
        let action = match promotion.outcome {
            GpuLocalParquetSourceCachePromotion::Inserted => {
                state.promotions_inserted = state.promotions_inserted.saturating_add(1);
                state.promoted_device_bytes =
                    state.promoted_device_bytes.saturating_add(promotion.retained_device_bytes);
                "promotion_inserted"
            },
            GpuLocalParquetSourceCachePromotion::AlreadyPresent => {
                // The early-return paths count these while the key's state is
                // still observable, so this branch is unreachable today.
                "promotion_already_present"
            },
            GpuLocalParquetSourceCachePromotion::CapacitySkipped => {
                state.promotions_capacity_skipped = state.promotions_capacity_skipped.saturating_add(1);
                "promotion_capacity_skipped"
            },
            GpuLocalParquetSourceCachePromotion::AllocationFailed => {
                state.promotions_allocation_failed = state.promotions_allocation_failed.saturating_add(1);
                "promotion_allocation_failed"
            },
        };
        if promotion.evicted_device_bytes != 0 {
            state.eviction_batches = state.eviction_batches.saturating_add(1);
            state.evicted_device_bytes = state.evicted_device_bytes.saturating_add(promotion.evicted_device_bytes);
        }
        state.record_event(
            self.inner.ledger.device_ordinal(),
            self.inner.allocation_lease.domain_id().get(),
            context.query_id().map(|query_id| cache_observation_fingerprint(query_id.as_str())),
            key,
            action,
            promotion.retained_device_bytes,
        );
        Ok(())
    }

    fn promote_claimed(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        source: &GpuResultHandle,
        reservation_bytes: u64,
    ) -> Result<GpuLocalParquetSourceCachePromotionResult> {
        self.promote_claimed_with_copy(key, context, reservation_bytes, || {
            source.copy_for_local_parquet_cache(Arc::clone(&self.inner.allocation_lease))
        })
    }

    /// Completes an already-claimed promotion around the concrete cache copy.
    ///
    /// Keeping the copy as the sole operation between the two cancellation
    /// checks makes the post-copy cleanup path testable without adding cache
    /// test state or changing the production ownership protocol.
    fn promote_claimed_with_copy(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        reservation_bytes: u64,
        copy: impl FnOnce() -> Result<GpuLocalParquetCachedResult>,
    ) -> Result<GpuLocalParquetSourceCachePromotionResult> {
        const PROMOTION_OPERATION: &str = "query_engine.gpu_local_parquet_source_cache.promote";
        let mut evicted_device_bytes = 0u64;
        let mut transaction = loop {
            if let Some(transaction) =
                CacheResidency::begin_attempt_owned(self.inner.ledger.clone(), reservation_bytes)?
            {
                break transaction;
            }
            let Some(evicted) = self.evict_one_lru()? else {
                self.clear_promotion(&key)?;
                return Ok(GpuLocalParquetSourceCachePromotionResult::capacity_skipped(evicted_device_bytes));
            };
            evicted_device_bytes = evicted_device_bytes.checked_add(evicted).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    PROMOTION_OPERATION,
                    "source-cache eviction byte total overflowed",
                )
                .with_fact("reason", "source_cache_evicted_bytes_overflow")
            })?;
        };

        if let Err(error) = context.check_cancellation(PROMOTION_OPERATION) {
            self.clear_promotion(&key)?;
            return Err(error);
        }
        let value = match copy() {
            Ok(value) => value,
            Err(error)
                if matches!(
                    error.code(),
                    QueryErrorCode::QueryMemoryLimitExceeded
                        | QueryErrorCode::DeviceOutOfMemory
                        | QueryErrorCode::DeviceAllocationFailed
                        | QueryErrorCode::CudfResourceExhausted
                ) =>
            {
                self.clear_promotion(&key)?;
                return Ok(GpuLocalParquetSourceCachePromotionResult::allocation_failed(
                    evicted_device_bytes,
                    error.code(),
                ));
            },
            Err(error) => return Err(error),
        };
        if let Err(error) = context.check_cancellation(PROMOTION_OPERATION) {
            self.discard_uncommitted_value(value, transaction)?;
            self.clear_promotion(&key)?;
            return Err(error);
        }
        let copied_device_bytes = match value.device_allocated_size_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.discard_uncommitted_value(value, transaction)?;
                self.clear_promotion(&key)?;
                return Err(error);
            },
        };
        let shrunk = match transaction.shrink_to(copied_device_bytes) {
            Ok(shrunk) => shrunk,
            Err(error) => {
                self.discard_uncommitted_value(value, transaction)?;
                self.clear_promotion(&key)?;
                return Err(error);
            },
        };
        if !shrunk {
            self.discard_uncommitted_value(value, transaction)?;
            self.clear_promotion(&key)?;
            return Ok(GpuLocalParquetSourceCachePromotionResult::allocation_failed(
                evicted_device_bytes,
                QueryErrorCode::CudfResourceExhausted,
            ));
        }

        let residency = transaction.commit_evictable();
        let mut state = match self.lock("promote.commit") {
            Ok(state) => state,
            Err(error) => {
                self.destroy_committed_value(value, residency)?;
                self.clear_promotion(&key)?;
                return Err(error);
            },
        };
        if state.closed {
            drop(state);
            self.destroy_committed_value(value, residency)?;
            self.clear_promotion(&key)?;
            return Err(source_cache_closed(PROMOTION_OPERATION));
        }
        let pending = state.promotions.remove(&key);
        if !pending || state.entries.contains_key(&key) {
            drop(state);
            self.destroy_committed_value(value, residency)?;
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                PROMOTION_OPERATION,
                "source-cache promotion ownership was lost before commit",
            )
            .with_fact("reason", "source_cache_promotion_lost"));
        }
        let access_epoch = state.next_epoch();
        state.entries.insert(
            key,
            GpuLocalParquetSourceCacheEntry { value: Arc::new(value), residency, last_access_epoch: access_epoch },
        );
        Ok(GpuLocalParquetSourceCachePromotionResult::inserted(copied_device_bytes, evicted_device_bytes))
    }

    /// Drains all cache entries after ensuring no checkout remains pinned.
    ///
    /// Refuses to close a pinned source cache rather than dropping its map
    /// ownership and relying on a later pin release to discover teardown.
    pub(crate) fn close_and_drain(&self) -> Result<()> {
        let entries = {
            let mut state = self.lock("close_and_drain")?;
            if state.closed {
                return Ok(());
            }
            for entry in state.entries.values() {
                if entry.residency.state()? != CacheResidencyState::Evictable {
                    return Err(QueryError::new(
                        QueryErrorCode::ServiceOverloaded,
                        "query_engine.gpu_local_parquet_source_cache.close_and_drain",
                        "cannot close a source cache while an entry is pinned",
                    )
                    .with_fact("reason", "source_cache_close_pinned"));
                }
            }
            state.closed = true;
            state.loads.clear();
            state.promotions.clear();
            mem::take(&mut state.entries)
        };
        self.inner.load_complete.notify_all();
        for (_key, entry) in entries {
            let Some(destruction) = entry.residency.begin_eviction()? else {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.gpu_local_parquet_source_cache.close_and_drain",
                    "source-cache entry changed while closing",
                )
                .with_fact("reason", "source_cache_close_entry_state_changed"));
            };
            self.destroy_entry(entry.value, destruction)?;
        }
        Ok(())
    }

    fn evict_one_lru(&self) -> Result<Option<u64>> {
        let (entry, destruction) = {
            let mut state = self.lock("evict_one_lru")?;
            let mut candidate = None;
            for (key, entry) in &state.entries {
                if entry.residency.state()? == CacheResidencyState::Evictable
                    && candidate.as_ref().is_none_or(|(_key, oldest_epoch): &(GpuLocalParquetSourceCacheKey, u64)| {
                        entry.last_access_epoch < *oldest_epoch
                    })
                {
                    candidate = Some((key.clone(), entry.last_access_epoch));
                }
            }
            let candidate = candidate.map(|(key, _epoch)| key);
            let Some(key) = candidate else {
                return Ok(None);
            };
            let entry = state.entries.remove(&key).ok_or_else(|| {
                QueryError::new(QueryErrorCode::Internal, OPERATION, "selected source-cache LRU entry disappeared")
                    .with_fact("reason", "source_cache_lru_entry_missing")
            })?;
            let Some(destruction) = entry.residency.begin_eviction()? else {
                state.entries.insert(key, entry);
                return Ok(None);
            };
            (entry, destruction)
        };
        let bytes = entry.residency.bytes();
        self.destroy_entry(entry.value, destruction)?;
        Ok(Some(bytes))
    }

    fn destroy_committed_value(&self, value: GpuLocalParquetCachedResult, residency: CacheResidency) -> Result<()> {
        let Some(destruction) = residency.begin_eviction()? else {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "newly committed source-cache entry could not enter destruction",
            )
            .with_fact("reason", "source_cache_committed_entry_not_evictable"));
        };
        self.destroy_entry(Arc::new(value), destruction)
    }

    fn discard_uncommitted_value(
        &self,
        value: GpuLocalParquetCachedResult,
        transaction: CacheResidencyChargeTransaction,
    ) -> Result<()> {
        let destroy = value.synchronize_for_destruction();
        drop(value);
        match destroy {
            Ok(()) => transaction.rollback(),
            Err(error) => self.seal_failed_uncommitted_destruction(transaction, error),
        }
    }

    fn destroy_entry(
        &self,
        value: Arc<GpuLocalParquetCachedResult>,
        destruction: CacheResidencyDestruction,
    ) -> Result<()> {
        let destroy = value.synchronize_for_destruction();
        drop(value);
        match destroy {
            Ok(()) => destruction.commit_after_destroy(),
            Err(error) => {
                destruction.retain_charge_after_failed_destruction()?;
                self.inner.ledger.close_admission_if_open("source_cache_destruction_sync_failed")?;
                Err(error)
            },
        }
    }

    fn seal_failed_uncommitted_destruction(
        &self,
        transaction: CacheResidencyChargeTransaction,
        error: QueryError,
    ) -> Result<()> {
        let residency = transaction.commit_evictable();
        let destruction = residency.begin_eviction()?.ok_or_else(|| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "uncommitted source cache value could not be sealed")
                .with_fact("reason", "source_cache_uncommitted_destruction_lost")
        })?;
        destruction.retain_charge_after_failed_destruction()?;
        self.inner.ledger.close_admission_if_open("source_cache_destruction_sync_failed")?;
        Err(error)
    }

    fn clear_promotion(&self, key: &GpuLocalParquetSourceCacheKey) -> Result<()> {
        self.lock("clear_promotion")?.promotions.remove(key);
        Ok(())
    }

    fn lock(&self, action: &'static str) -> Result<MutexGuard<'_, GpuLocalParquetSourceCacheState>> {
        self.inner.state.lock().map_err(|_error| {
            QueryError::new(QueryErrorCode::Internal, OPERATION, "source-cache state mutex was poisoned")
                .with_fact("action", action)
                .with_fact("reason", "source_cache_mutex_poisoned")
        })
    }
}

fn source_cache_closed(operation: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::Cancelled, operation, "GPU local-Parquet source cache is closed")
        .with_fact("reason", "local_parquet_source_cache_closed")
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    };
    use cudf::{CacheAllocationDomain, DataFrame};
    use query_planning::source::LocalParquetSource;

    use super::*;
    use crate::runtime::{
        ExecutionContext,
        backend_metrics::BackendMetricsStore,
        gpu::{DeviceCapabilities, DeviceResourceEnvelope, DeviceResourceProfile},
        resident::GpuDataFrame,
    };

    fn source_fixture() -> (Arc<ExecutionContext>, GpuResultHandle, SchemaRef, GpuLocalParquetSourceCacheKey, u64) {
        let runtime = Arc::new(cudf::Runtime::new().expect("test CUDA runtime"));
        let context = Arc::new(
            ExecutionContext::builder()
                .with_shared_runtime(Arc::clone(&runtime))
                .build()
                .expect("test execution context"),
        );
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from_iter_values(0..4096))])
            .expect("test Arrow batch");
        let frame = DataFrame::from_record_batch(runtime.as_ref(), &batch).expect("test GPU frame");
        let bytes = GpuDataFrame::new(frame.clone()).device_allocated_size_bytes().expect("test source size");
        let handle = GpuResultHandle::new(context.as_ref(), GpuDataFrame::new(frame), Arc::clone(&schema))
            .expect("test GPU result handle");
        let source = LocalParquetSource::new("/tmp/algeon-source-cache.parquet", schema, vec!["value".into()]);
        (context, handle, batch.schema(), GpuLocalParquetSourceCacheKey::from_source(&source), bytes)
    }

    fn ledger(device_ordinal: usize, cache_cap_bytes: u64) -> DeviceLedger {
        let managed_capacity_bytes = cache_cap_bytes.saturating_mul(4).saturating_add(1024);
        let profile = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal,
            configured_limit_bytes: managed_capacity_bytes,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 1,
            max_active_attempts: 1,
            source_work_per_attempt: 1,
            stream_pool_size: 2,
            worker_job_capacity: 2,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes,
        })
        .expect("source-cache test ledger profile");
        DeviceLedger::new(profile)
    }

    fn assert_metrics_project_the_ledger(metrics: &BackendMetricsStore, ledger: &DeviceLedger) {
        let ledger_snapshot = ledger.snapshot().expect("ledger snapshot");
        let device = metrics.snapshot().device(ledger.device_ordinal()).expect("configured metrics device");
        assert_eq!(device.ledger_sequence, ledger_snapshot.sequence);
        assert_eq!(device.admission.cache_evictable_resident_bytes, ledger_snapshot.cache_evictable_resident_bytes);
        assert_eq!(device.admission.cache_pinned_in_use_bytes, ledger_snapshot.cache_pinned_in_use_bytes);
    }

    #[test]
    fn concurrent_identical_promotions_create_one_entry_and_one_charge() {
        let (context, source, schema, key, bytes) = source_fixture();
        let cache_cap_bytes = bytes.saturating_mul(2).saturating_add(1024);
        let domain = CacheAllocationDomain::new(source.runtime().device(), cache_cap_bytes).expect("cache domain");
        let ledger = ledger(source.runtime().device().ordinal(), cache_cap_bytes);
        let metrics = BackendMetricsStore::new(1, &[ledger.device_ordinal()]).expect("backend metrics store");
        let metrics_observer = metrics.clone();
        ledger
            .register_transition_observer(Arc::new(move |transition| {
                metrics_observer.record_ledger_transition(transition)
            }))
            .expect("register metrics projection observer");
        let cache = Arc::new(
            GpuLocalParquetSourceCache::new(domain.lease_runtime().expect("source lease"), ledger.clone()).unwrap(),
        );
        let start = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let cache = Arc::clone(&cache);
            let context = Arc::clone(&context);
            let source = source.clone();
            let key = key.clone();
            let start = Arc::clone(&start);
            workers.push(thread::spawn(move || {
                start.wait();
                cache.promote(key, context.as_ref(), &source).expect("promotion result")
            }));
        }
        start.wait();
        let outcomes = workers.into_iter().map(|worker| worker.join().expect("promotion worker")).collect::<Vec<_>>();
        assert_eq!(
            outcomes.iter().filter(|outcome| outcome.outcome == GpuLocalParquetSourceCachePromotion::Inserted).count(),
            1
        );
        assert_eq!(cache.len().unwrap(), 1);
        assert_eq!(cache.retained_device_bytes().unwrap(), bytes);
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.cache_evictable_resident_bytes, bytes);
        assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
        assert_metrics_project_the_ledger(&metrics, &ledger);

        let hit = cache.get(&key, context.as_ref(), &schema).unwrap().expect("cache hit").0;
        assert_eq!(hit.to_host_record_batch().unwrap().num_rows(), 4096);
        assert_eq!(ledger.snapshot().unwrap().cache_pinned_in_use_bytes, bytes);
        assert_metrics_project_the_ledger(&metrics, &ledger);
        drop(hit);
        cache.close_and_drain().unwrap();
        assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, 0);
        assert_metrics_project_the_ledger(&metrics, &ledger);
    }

    #[test]
    fn same_key_load_waiter_reuses_the_leader_entry_without_a_second_charge() {
        let (context, source, schema, key, bytes) = source_fixture();
        let cache_cap_bytes = bytes.saturating_mul(2).saturating_add(1024);
        let domain = CacheAllocationDomain::new(source.runtime().device(), cache_cap_bytes).expect("cache domain");
        let ledger = ledger(source.runtime().device().ordinal(), cache_cap_bytes);
        let cache = Arc::new(
            GpuLocalParquetSourceCache::new(domain.lease_runtime().expect("source lease"), ledger.clone()).unwrap(),
        );
        let claim = match cache.checkout_or_claim(key.clone(), context.as_ref(), &schema).expect("cold load claim") {
            GpuLocalParquetSourceCacheLoad::Leader(claim) => claim,
            GpuLocalParquetSourceCacheLoad::Hit(_) => panic!("fresh cache must elect a load leader"),
        };
        let waiter_cache = Arc::clone(&cache);
        let waiter_context = Arc::clone(&context);
        let waiter_schema = Arc::clone(&schema);
        let waiter_key = key;
        let waiter =
            thread::spawn(move || waiter_cache.checkout_or_claim(waiter_key, waiter_context.as_ref(), &waiter_schema));

        let mut follower_registered = false;
        for _ in 0..100_000 {
            if cache.lock("test load waiter").unwrap().promotions_coalesced == 1 {
                follower_registered = true;
                break;
            }
            thread::yield_now();
        }
        assert!(follower_registered, "same-key follower must wait behind the elected source-cache leader");

        let promotion = cache.promote_load(claim, context.as_ref(), &source).expect("leader promotion");
        assert_eq!(promotion.outcome, GpuLocalParquetSourceCachePromotion::Inserted);
        let follower = waiter.join().expect("same-key waiter thread").expect("same-key waiter result");
        let GpuLocalParquetSourceCacheLoad::Hit(follower) = follower else {
            panic!("same-key waiter must reuse the leader entry");
        };
        assert_eq!(follower.to_host_record_batch().unwrap().num_rows(), 4096);
        drop(follower);
        assert_eq!(cache.len().unwrap(), 1);
        assert_eq!(cache.retained_device_bytes().unwrap(), bytes);
        assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, bytes);
        cache.close_and_drain().unwrap();
    }

    #[test]
    fn dropped_same_key_load_claim_wakes_a_follower_to_become_the_next_leader() {
        let (context, source, schema, key, bytes) = source_fixture();
        let cache_cap_bytes = bytes.saturating_mul(2).saturating_add(1024);
        let domain = CacheAllocationDomain::new(source.runtime().device(), cache_cap_bytes).expect("cache domain");
        let ledger = ledger(source.runtime().device().ordinal(), cache_cap_bytes);
        let cache =
            Arc::new(GpuLocalParquetSourceCache::new(domain.lease_runtime().expect("source lease"), ledger).unwrap());
        let claim = match cache.checkout_or_claim(key.clone(), context.as_ref(), &schema).expect("cold load claim") {
            GpuLocalParquetSourceCacheLoad::Leader(claim) => claim,
            GpuLocalParquetSourceCacheLoad::Hit(_) => panic!("fresh cache must elect a load leader"),
        };
        let waiter_cache = Arc::clone(&cache);
        let waiter_context = Arc::clone(&context);
        let waiter_schema = Arc::clone(&schema);
        let waiter_key = key;
        let waiter =
            thread::spawn(move || waiter_cache.checkout_or_claim(waiter_key, waiter_context.as_ref(), &waiter_schema));

        let mut follower_registered = false;
        for _ in 0..100_000 {
            if cache.lock("test abandoned load waiter").unwrap().promotions_coalesced == 1 {
                follower_registered = true;
                break;
            }
            thread::yield_now();
        }
        assert!(follower_registered, "same-key follower must wait behind the elected source-cache leader");

        drop(claim);
        let follower = waiter.join().expect("same-key waiter thread").expect("same-key waiter result");
        assert!(matches!(follower, GpuLocalParquetSourceCacheLoad::Leader(_)));
        drop(follower);

        let state = cache.lock("test abandoned load state").unwrap();
        assert!(state.loads.is_empty(), "the RAII claim drop must retire the key for later leaders");
        assert!(state.recent_events.iter().any(|event| event.action == "load_abandoned"));
    }

    #[test]
    fn allocator_failure_is_not_laundered_into_capacity_skip() {
        let (context, source, _schema, key, bytes) = source_fixture();
        let cache_cap_bytes = bytes.saturating_add(1);
        let domain = CacheAllocationDomain::new(source.runtime().device(), cache_cap_bytes).expect("cache domain");
        let occupancy_lease = domain.lease_runtime().expect("occupancy lease");
        let occupied = source.frame().copy_into_runtime(occupancy_lease.runtime()).expect("occupy cache domain");
        let ledger = ledger(source.runtime().device().ordinal(), cache_cap_bytes);
        let cache =
            GpuLocalParquetSourceCache::new(domain.lease_runtime().expect("source lease"), ledger.clone()).unwrap();

        let outcome = cache.promote(key, context.as_ref(), &source).expect("allocation failure is a cache outcome");
        assert_eq!(outcome.outcome, GpuLocalParquetSourceCachePromotion::AllocationFailed);
        assert_eq!(outcome.allocation_failure_code, Some(QueryErrorCode::DeviceAllocationFailed));
        assert_eq!(cache.len().unwrap(), 0);
        assert_eq!(ledger.snapshot().unwrap().cache_evictable_resident_bytes, 0);
        assert_eq!(source.to_host_record_batch().unwrap().num_rows(), 4096, "attempt retains its source owner");
        drop(occupied);
        drop(occupancy_lease);
    }

    #[test]
    fn cancelled_promotion_keeps_the_attempt_as_the_only_owner() {
        let (context, source, _schema, key, bytes) = source_fixture();
        let cache_cap_bytes = bytes.saturating_mul(2).saturating_add(1024);
        let domain = CacheAllocationDomain::new(source.runtime().device(), cache_cap_bytes).expect("cache domain");
        let ledger = ledger(source.runtime().device().ordinal(), cache_cap_bytes);
        let cache =
            GpuLocalParquetSourceCache::new(domain.lease_runtime().expect("source lease"), ledger.clone()).unwrap();

        context.cancellation().cancel();
        let error = cache.promote(key, context.as_ref(), &source).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(cache.len().unwrap(), 0);
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.cache_evictable_resident_bytes, 0);
        assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
        assert_eq!(source.to_host_record_batch().unwrap().num_rows(), 4096);
    }

    #[test]
    fn cancellation_after_cache_copy_destroys_the_copy_before_rolling_back_the_charge() {
        let (context, source, _schema, key, bytes) = source_fixture();
        let cache_cap_bytes = bytes.saturating_mul(2).saturating_add(1024);
        let domain = CacheAllocationDomain::new(source.runtime().device(), cache_cap_bytes).expect("cache domain");
        let ledger = ledger(source.runtime().device().ordinal(), cache_cap_bytes);
        let cache =
            GpuLocalParquetSourceCache::new(domain.lease_runtime().expect("source lease"), ledger.clone()).unwrap();
        cache.lock("test promotion claim").unwrap().promotions.insert(key.clone());

        let error = cache
            .promote_claimed_with_copy(key, context.as_ref(), bytes, || {
                let value = source.copy_for_local_parquet_cache(Arc::clone(&cache.inner.allocation_lease))?;
                context.cancellation().cancel();
                Ok(value)
            })
            .unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(cache.len().unwrap(), 0);
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.cache_evictable_resident_bytes, 0);
        assert_eq!(snapshot.cache_pinned_in_use_bytes, 0);
        assert_eq!(
            source.to_host_record_batch().unwrap().num_rows(),
            4096,
            "the attempt keeps the only live source frame"
        );
        assert!(cache.lock("test promotion claim").unwrap().promotions.is_empty());
    }

    #[test]
    fn source_copy_rejects_a_cache_domain_on_another_device() {
        let (context, source, _schema, _key, bytes) = source_fixture();
        let device_count = cudf::DeviceId::count().expect("CUDA device count");
        if device_count < 2 {
            return;
        }
        let source_device = source.runtime().device();
        let other_device =
            cudf::DeviceId::new(if source_device.ordinal() == 0 { 1 } else { 0 }).expect("another visible CUDA device");
        let domain =
            CacheAllocationDomain::new(other_device, bytes.saturating_add(1024)).expect("other-device cache domain");

        let error = match source
            .copy_for_local_parquet_cache(Arc::new(domain.lease_runtime().expect("other-device cache lease")))
        {
            Ok(_) => panic!("source copy must reject a cache domain on another device"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(context.runtime().unwrap().device(), source_device);
    }
}
