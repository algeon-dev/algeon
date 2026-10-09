use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    source::{ParquetSourceLocation, ResolvedParquetSource},
};
use query_runtime::runtime::{CacheFillBytePermit, S3ObjectStoreKey};

use super::{
    s3::{CacheMaintenanceLocation, cache_maintenance_location, object_store_bucket_fact},
    *,
};

/// Handle to one object fetch running on the object-store control runtime. `finish` blocks the
/// calling thread until the fetch has produced its result.
pub(in crate::exec::source) struct PendingCacheMaintenanceFetch {
    receiver: std::sync::mpsc::Receiver<Result<CacheMaintenanceObject>>,
}

/// A fetched object together with its host byte reservation.
///
/// `permit` is charged for the payload and must live until the bytes are published to the cache
/// file, because the payload is resident host memory until then. `release_after_publish`
/// returns the reservation; dropping the object without it also returns it.
pub(in crate::exec::source::lakehouse_cache) struct CacheMaintenanceObject {
    pub(in crate::exec::source::lakehouse_cache) bytes: bytes::Bytes,
    pub(in crate::exec::source::lakehouse_cache) permit: Option<CacheFillBytePermit>,
    pub(in crate::exec::source::lakehouse_cache) actual_bytes: u64,
}

impl PendingCacheMaintenanceFetch {
    /// Waits for the fetch. The result is the fetch error (object store failure, payload over the
    /// remaining fill budget) or `Internal` if the worker dropped its sender without a result.
    pub(super) fn finish(self) -> Result<CacheMaintenanceObject> {
        self.receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.lakehouse_cache.maintenance.receive",
                "object-store cache maintenance worker stopped before returning an object",
            )
            .with_source(error)
        })?
    }
}

impl CacheMaintenanceObject {
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn release_after_publish(&mut self) {
        self.permit.take();
    }
}

/// Starts fetching one S3 object for a cache fill and returns before the fetch completes.
///
/// Admission happens on the calling thread, in this order: the cancellation check, a host byte
/// reservation from the cache-fill budget, then a cache-fill operation slot. Each wait is bounded
/// by the cache-fill acquire timeout and ends in `ServiceOverloaded` on expiry; the byte
/// reservation is released if the slot wait fails. The fetch itself runs on the object-store
/// control runtime. The slot is held until the result has been sent, and the byte reservation
/// travels with the object until `release_after_publish`.
///
/// `estimated_fetch_bytes` sizes the reservation. When it is unknown the whole budget is reserved,
/// so an unsized fetch runs alone against the byte budget.
pub(super) fn start_cache_maintenance_fetch(
    context: &ExecutionContext,
    location: &ParquetSourceLocation,
    resolved_source: &ResolvedParquetSource,
    estimated_fetch_bytes: Option<u64>,
) -> Result<PendingCacheMaintenanceFetch> {
    context.check_cancellation("query_engine.lakehouse_cache.maintenance")?;
    let maintenance_location = cache_maintenance_location(location, resolved_source)?;
    let cache_fill_budget = context.lakehouse_cache_fill_budget();
    let cache_fill_budget_bytes = cache_fill_budget.budget_bytes();
    let cache_fill_acquire_timeout = context.lakehouse_cache_fill_acquire_timeout();
    let reserve_bytes = cache_fill_reserve_bytes(
        cache_fill_budget_bytes,
        context.source_scheduler_policy().max_source_reads_in_flight(),
        estimated_fetch_bytes,
    );
    let mut permit = cache_fill_budget
        .acquire_with_timeout(
            context.cancellation(),
            reserve_bytes,
            cache_fill_acquire_timeout,
            "query_engine.lakehouse_cache.maintenance.fill_budget",
        )
        .map_err(|error| attach_cache_maintenance_budget_facts(error, &maintenance_location, reserve_bytes))?
        .ok_or_else(|| cache_fill_budget_timeout_error(&maintenance_location, reserve_bytes))?;
    let operation_permit = context
        .acquire_lakehouse_cache_fill_operation_with_timeout(
            cache_fill_acquire_timeout,
            "query_engine.lakehouse_cache.maintenance.fetch_concurrency",
        )
        .map_err(|error| attach_cache_maintenance_budget_facts(error, &maintenance_location, reserve_bytes))?
        .ok_or_else(|| cache_fill_concurrency_timeout_error(&maintenance_location, reserve_bytes))?;
    let store = context.s3_object_store(S3ObjectStoreKey {
        bucket: maintenance_location.bucket.clone(),
        region: maintenance_location.region.clone(),
        storage_profile_id: maintenance_location.storage_profile_id.clone(),
    })?;
    let (sender, receiver) = std::sync::mpsc::channel();
    context.spawn_cache_maintenance_fetch(store, maintenance_location.key, move |fetch_result| {
        let result = fetch_result
            .bytes
            .map_err(|error| {
                error
                    .with_fact_if_absent("cache_maintenance_substrate", "object_store")
                    .with_fact_if_absent("cache_maintenance_scheme", "s3")
                    .with_fact_if_absent("bucket", object_store_bucket_fact(maintenance_location.bucket.as_str()))
                    .with_fact_if_absent("region", maintenance_location.region.as_str())
            })
            .and_then(|bytes| {
                let actual_bytes = u64::try_from(bytes.len()).map_err(|_error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.lakehouse_cache.maintenance.object_size",
                        "object-store cache maintenance payload length exceeds u64 range",
                    )
                })?;
                account_cache_fill_payload_bytes(&mut permit, actual_bytes).map_err(|error| {
                    error
                        .with_fact_if_absent("cache_maintenance_substrate", "object_store")
                        .with_fact_if_absent("cache_maintenance_scheme", "s3")
                        .with_fact_if_absent("bucket", object_store_bucket_fact(maintenance_location.bucket.as_str()))
                        .with_fact_if_absent("actual_bytes", actual_bytes)
                })?;
                Ok(CacheMaintenanceObject { bytes, permit: Some(permit), actual_bytes })
            });
        let _ = sender.send(result);
        // Naming the permit here moves it into this closure, which keeps the operation slot held
        // for the whole fetch. It is released after the result is sent.
        drop(operation_permit);
    });
    Ok(PendingCacheMaintenanceFetch { receiver })
}

/// Bytes to reserve before a fetch: the estimate, or the whole budget when unknown, and at least
/// one. `_max_source_reads_in_flight` is accepted and ignored.
fn cache_fill_reserve_bytes(
    budget_bytes: u64,
    _max_source_reads_in_flight: usize,
    estimated_fetch_bytes: Option<u64>,
) -> u64 {
    estimated_fetch_bytes.unwrap_or(budget_bytes).max(1)
}

/// Charges the fetched payload to the host cache-fill budget. Growth never
/// waits, since two fills each holding part of the budget would deadlock, so a
/// payload the remaining budget cannot cover is refused. The caller then
/// drops the payload with its permit, so unaccounted bytes are never published.
fn account_cache_fill_payload_bytes(permit: &mut CacheFillBytePermit, actual_bytes: u64) -> Result<()> {
    if permit.try_grow(actual_bytes.max(1))? {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "query_engine.lakehouse_cache.maintenance.fill_budget",
        "cache maintenance payload exceeds the available host cache-fill budget",
    )
    .with_fact("reason", "cache_fill_budget_exhausted")
    .with_fact("reserved_bytes", permit.bytes()))
}

fn attach_cache_maintenance_budget_facts(
    error: QueryError,
    maintenance_location: &CacheMaintenanceLocation,
    reserve_bytes: u64,
) -> QueryError {
    error
        .with_fact_if_absent("cache_maintenance_substrate", "object_store")
        .with_fact_if_absent("cache_maintenance_scheme", "s3")
        .with_fact_if_absent("bucket", object_store_bucket_fact(maintenance_location.bucket.as_str()))
        .with_fact_if_absent("reserved_bytes", reserve_bytes)
}

fn cache_fill_budget_timeout_error(maintenance_location: &CacheMaintenanceLocation, reserve_bytes: u64) -> QueryError {
    attach_cache_maintenance_budget_facts(
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.lakehouse_cache.maintenance.fill_budget",
            "timed out waiting for cache maintenance byte-budget capacity",
        ),
        maintenance_location,
        reserve_bytes,
    )
}

fn cache_fill_concurrency_timeout_error(
    maintenance_location: &CacheMaintenanceLocation,
    reserve_bytes: u64,
) -> QueryError {
    attach_cache_maintenance_budget_facts(
        QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.lakehouse_cache.maintenance.fetch_concurrency",
            "timed out waiting for object-store cache maintenance operation capacity",
        ),
        maintenance_location,
        reserve_bytes,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use query_runtime::runtime::{ExecutionCancellation, LakehouseCacheFillConfig, StandaloneConfig};

    use super::*;

    #[test]
    fn a_payload_the_fill_budget_cannot_cover_is_refused_and_its_reservation_returns() {
        let context = ExecutionContext::standalone(StandaloneConfig {
            lakehouse_cache_fill: LakehouseCacheFillConfig {
                budget_bytes: Some(10),
                ..LakehouseCacheFillConfig::default()
            },
            ..StandaloneConfig::default()
        })
        .unwrap();
        let budget = context.lakehouse_cache_fill_budget();
        let acquire =
            |bytes| budget.acquire_with_timeout(&ExecutionCancellation::new(), bytes, Duration::ZERO, "test").unwrap();
        let held = acquire(6).expect("the first fill reserves 6 of 10 bytes");
        let mut refused = acquire(2).expect("the second fill reserves 2 bytes");

        // Five payload bytes need three more than the 2-byte reservation, and only two are free.
        let error = account_cache_fill_payload_bytes(&mut refused, 5).expect_err("growth beyond the budget is refused");
        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("reason"), Some("cache_fill_budget_exhausted"));
        drop(refused);
        assert!(acquire(5).is_none(), "the refused fill must not leave its payload charged");
        let mut accepted = acquire(2).expect("the refused fill returned its reservation");

        account_cache_fill_payload_bytes(&mut accepted, 4).expect("a payload within the free budget is charged");
        assert_eq!(accepted.bytes(), 4);
        drop((held, accepted));
        assert!(acquire(10).is_some(), "both fills return their whole charge");
    }

    #[test]
    fn unknown_fetch_size_reserves_full_budget() {
        assert_eq!(cache_fill_reserve_bytes(100, 4, None), 100);
        assert_eq!(cache_fill_reserve_bytes(3, 4, None), 3);
        assert_eq!(cache_fill_reserve_bytes(100, 0, None), 100);
    }

    #[test]
    fn known_fetch_size_uses_estimate() {
        assert_eq!(cache_fill_reserve_bytes(100, 4, Some(7)), 7);
        assert_eq!(cache_fill_reserve_bytes(100, 4, Some(0)), 1);
    }
}
