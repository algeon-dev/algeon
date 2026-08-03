//! GPU-resident decoded local-Parquet source cache.
//!
//! A promotion always copies the attempt-owned frame into the one shared
//! device cache allocation domain. The cache pre-reserves its ledger charge
//! before that copy, so an allocation failure or cancellation cannot leave a
//! copied allocation uncharged. The source handle is borrowed throughout: the
//! attempt remains its sole owner when retention is skipped or fails.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, MutexGuard},
};

use arrow::datatypes::SchemaRef;

pub(crate) use crate::runtime::source_identity::GpuLocalParquetSourceCacheKey;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::{
        ExecutionContext,
        cache::{CacheResidency, CacheResidencyChargeTransaction, CacheResidencyDestruction, CacheResidencyPin},
        gpu::DeviceLedger,
        resident::{GpuLocalParquetCachedResult, GpuResultHandle},
    },
};

const OPERATION: &str = "nexus_query_engine.gpu_local_parquet_source_cache";

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
pub(crate) struct GpuLocalParquetSourceCachePromotionResult {
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
    promotions: HashSet<GpuLocalParquetSourceCacheKey>,
    next_access_epoch: u64,
    closed: bool,
}

impl GpuLocalParquetSourceCacheState {
    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_access_epoch;
        self.next_access_epoch = self.next_access_epoch.saturating_add(1);
        epoch
    }
}

struct GpuLocalParquetSourceCacheInner {
    allocation_lease: Arc<cudf::CacheAllocationLease>,
    ledger: DeviceLedger,
    state: Mutex<GpuLocalParquetSourceCacheState>,
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
/// Construction requires the P5.3 device cache allocation lease and the
/// matching P5 ledger. There is no unaccounted constructor and no independent
/// source-cache capacity.
pub(crate) struct GpuLocalParquetSourceCache {
    inner: Arc<GpuLocalParquetSourceCacheInner>,
}

impl Clone for GpuLocalParquetSourceCache {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl std::fmt::Debug for GpuLocalParquetSourceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    /// P5.3 cache allocation domain.
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
            }),
        })
    }

    /// Stable P5.3 allocation-domain identity used by this cache.
    pub(crate) fn allocation_domain_id(&self) -> cudf::CacheAllocationDomainId {
        self.inner.allocation_lease.domain_id()
    }

    /// Number of retained entries. A pending promotion is not an entry.
    pub(crate) fn len(&self) -> Result<usize> {
        Ok(self.lock("len")?.entries.len())
    }

    /// Exact cache-owned bytes currently represented by completed entries.
    pub(crate) fn retained_device_bytes(&self) -> Result<u64> {
        self.lock("retained_device_bytes")?.entries.values().try_fold(0u64, |sum, entry| {
            sum.checked_add(entry.residency.bytes()).ok_or_else(|| {
                QueryError::new(QueryErrorCode::Internal, OPERATION, "source-cache retained byte total overflowed")
                    .with_fact("reason", "source_cache_retained_bytes_overflow")
            })
        })
    }

    /// Looks up and pins a cache entry before copying it into `context`.
    pub(crate) fn get(
        &self,
        key: &GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        schema: &SchemaRef,
    ) -> Result<Option<(GpuResultHandle, u64)>> {
        context.check_cancellation("nexus_query_engine.gpu_local_parquet_source_cache.get")?;
        let (pin, value, retained_device_bytes) = {
            let mut state = self.lock("get")?;
            if state.closed {
                return Err(source_cache_closed("nexus_query_engine.gpu_local_parquet_source_cache.get"));
            }
            let access_epoch = state.next_epoch();
            let Some(entry) = state.entries.get_mut(key) else {
                return Ok(None);
            };
            if entry.value.schema().as_ref() != schema.as_ref() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu_local_parquet_source_cache.get",
                    "cached local-Parquet schema does not match the requested source schema",
                )
                .with_fact("reason", "schema_mismatch")
                .with_fact("cached_fields", entry.value.schema().fields().len())
                .with_fact("requested_fields", schema.fields().len()));
            }
            let Some(pin) = entry.residency.try_pin()? else {
                return Ok(None);
            };
            entry.last_access_epoch = access_epoch;
            (GpuLocalParquetSourceCachePin::new(pin), Arc::clone(&entry.value), entry.residency.bytes())
        };
        match value.checkout(context, pin) {
            Ok(handle) => Ok(Some((handle, retained_device_bytes))),
            Err(error) => Err(error),
        }
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
        const PROMOTION_OPERATION: &str = "nexus_query_engine.gpu_local_parquet_source_cache.promote";
        context.check_cancellation(PROMOTION_OPERATION)?;
        let reservation_bytes = source.device_allocated_size_bytes()?;

        {
            let mut state = self.lock("promote.claim")?;
            if state.closed {
                return Err(source_cache_closed(PROMOTION_OPERATION));
            }
            if let Some(entry) = state.entries.get(&key) {
                return Ok(GpuLocalParquetSourceCachePromotionResult::already_present(entry.residency.bytes()));
            }
            if !state.promotions.insert(key.clone()) {
                return Ok(GpuLocalParquetSourceCachePromotionResult::already_present(0));
            }
        }

        let result = self.promote_claimed(key.clone(), context, source, reservation_bytes);
        if result.is_err() {
            self.clear_promotion(&key)?;
        }
        result
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
        const PROMOTION_OPERATION: &str = "nexus_query_engine.gpu_local_parquet_source_cache.promote";
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
            Err(error) if error.code() == QueryErrorCode::CudfResourceExhausted => {
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
    /// P5.5 refuses to close a pinned source cache rather than dropping its
    /// map ownership and relying on a later pin release to discover teardown.
    /// The P5.6 shared close path will extend this to final-pin coordination.
    pub(crate) fn close_and_drain(&self) -> Result<()> {
        let entries = {
            let mut state = self.lock("close_and_drain")?;
            if state.closed {
                return Ok(());
            }
            for entry in state.entries.values() {
                if entry.residency.state()? != crate::runtime::cache::CacheResidencyState::Evictable {
                    return Err(QueryError::new(
                        QueryErrorCode::ServiceOverloaded,
                        "nexus_query_engine.gpu_local_parquet_source_cache.close_and_drain",
                        "cannot close a source cache while an entry is pinned",
                    )
                    .with_fact("reason", "source_cache_close_pinned"));
                }
            }
            state.closed = true;
            state.promotions.clear();
            std::mem::take(&mut state.entries)
        };
        for (_key, entry) in entries {
            let Some(destruction) = entry.residency.begin_eviction()? else {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.gpu_local_parquet_source_cache.close_and_drain",
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
                if entry.residency.state()? == crate::runtime::cache::CacheResidencyState::Evictable
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

    use super::*;
    use crate::{
        runtime::{
            ExecutionContext,
            backend_metrics::BackendMetricsStore,
            gpu::{DeviceCapabilities, DeviceResourceProfile},
            resident::GpuDataFrame,
        },
        source::LocalParquetSource,
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
        let source = LocalParquetSource::new("/tmp/nexus-p5-source-cache.parquet", schema, vec!["value".into()]);
        (context, handle, batch.schema(), GpuLocalParquetSourceCacheKey::from_source(&source), bytes)
    }

    fn ledger(device_ordinal: usize, cache_cap_bytes: u64) -> DeviceLedger {
        let managed_capacity_bytes = cache_cap_bytes.saturating_mul(4).saturating_add(1024);
        let profile = DeviceResourceProfile::resolve_with_capabilities_and_cache_cap(
            device_ordinal,
            managed_capacity_bytes,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            DeviceCapabilities::default(),
            cache_cap_bytes,
        )
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
        assert_eq!(outcome.allocation_failure_code, Some(QueryErrorCode::CudfResourceExhausted));
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
