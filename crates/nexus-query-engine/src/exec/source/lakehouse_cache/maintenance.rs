use super::{
    s3::{CacheMaintenanceLocation, cache_maintenance_location, object_store_bucket_fact},
    *,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::S3ObjectStoreKey,
};

pub(in crate::exec::source) struct PendingCacheMaintenanceFetch {
    receiver: std::sync::mpsc::Receiver<Result<CacheMaintenanceObject>>,
}

pub(in crate::exec::source::lakehouse_cache) struct CacheMaintenanceObject {
    pub(in crate::exec::source::lakehouse_cache) bytes: bytes::Bytes,
    pub(in crate::exec::source::lakehouse_cache) permit: Option<crate::runtime::CacheFillBytePermit>,
    pub(in crate::exec::source::lakehouse_cache) actual_bytes: u64,
}

impl PendingCacheMaintenanceFetch {
    pub(super) fn finish(self) -> Result<CacheMaintenanceObject> {
        self.receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.lakehouse_cache.maintenance.receive",
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

pub(super) fn start_cache_maintenance_fetch(
    context: &ExecutionContext,
    location: &crate::source::ParquetSourceLocation,
    resolved_source: &crate::source::ResolvedParquetSource,
    estimated_fetch_bytes: Option<u64>,
) -> Result<PendingCacheMaintenanceFetch> {
    context.check_cancellation("nexus_query_engine.lakehouse_cache.maintenance")?;
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
            "nexus_query_engine.lakehouse_cache.maintenance.fill_budget",
        )
        .map_err(|error| attach_cache_maintenance_budget_facts(error, &maintenance_location, reserve_bytes))?
        .ok_or_else(|| cache_fill_budget_timeout_error(&maintenance_location, reserve_bytes))?;
    let operation_permit = context
        .acquire_lakehouse_cache_fill_operation_with_timeout(
            cache_fill_acquire_timeout,
            "nexus_query_engine.lakehouse_cache.maintenance.fetch_concurrency",
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
                        "nexus_query_engine.lakehouse_cache.maintenance.object_size",
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
        drop(operation_permit);
    });
    Ok(PendingCacheMaintenanceFetch { receiver })
}

fn cache_fill_reserve_bytes(
    budget_bytes: u64,
    _max_source_reads_in_flight: usize,
    estimated_fetch_bytes: Option<u64>,
) -> u64 {
    estimated_fetch_bytes.unwrap_or(budget_bytes).max(1)
}

fn account_cache_fill_payload_bytes(permit: &mut crate::runtime::CacheFillBytePermit, actual_bytes: u64) -> Result<()> {
    let charge_bytes = actual_bytes.max(1);
    permit.grow_to(charge_bytes)?;
    Ok(())
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
            "nexus_query_engine.lakehouse_cache.maintenance.fill_budget",
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
            "nexus_query_engine.lakehouse_cache.maintenance.fetch_concurrency",
            "timed out waiting for object-store cache maintenance operation capacity",
        ),
        maintenance_location,
        reserve_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
