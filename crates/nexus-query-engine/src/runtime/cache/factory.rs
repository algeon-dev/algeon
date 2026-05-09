//! Backend-owned, device-scoped allocation domain for resident caches.
//!
//! A factory is created from one device ledger and owns the sole physical
//! [`cudf::CacheAllocationDomain`] for that device.  Source and graph cache
//! subdomains receive independent leases from that domain; they never create
//! a domain of their own.

use std::sync::Arc;

#[cfg(feature = "cugraph")]
use super::resident_graph::GpuResidentGraphCache;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::{
        cache::{DeviceCacheReport, local_parquet::GpuLocalParquetSourceCache},
        gpu::DeviceLedger,
    },
};

/// One per-device cache allocation owner.
///
/// `None` is returned when the ledger profile has a zero cache cap.  Thus the
/// disabled baseline creates no CUDA allocator domain at all.
#[derive(Debug, Clone)]
pub(crate) struct DeviceCacheAllocationFactory {
    ledger: DeviceLedger,
    domain: Arc<cudf::CacheAllocationDomain>,
}

impl DeviceCacheAllocationFactory {
    /// Create the one physical cache domain for `ledger`, if caching is enabled.
    pub(crate) fn new(ledger: DeviceLedger) -> Result<Option<Self>> {
        let capacity = ledger.profile().cache_cap_bytes;
        if capacity == 0 {
            return Ok(None);
        }
        let device = cudf::DeviceId::new(ledger.device_ordinal()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.cache.factory",
                "failed to resolve the ledger device for the cache allocation domain",
                error,
            )
            .with_fact("device_ordinal", ledger.device_ordinal())
        })?;
        let domain = cudf::CacheAllocationDomain::new(device, capacity).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.cache.factory",
                "failed to create the device cache allocation domain",
                error,
            )
            .with_fact("device_ordinal", ledger.device_ordinal())
            .with_fact("cache_cap_bytes", capacity)
        })?;
        if domain.device().ordinal() != ledger.device_ordinal() || domain.capacity_bytes() != capacity {
            return Err(QueryError::new(
                QueryErrorCode::CudfInvariantViolation,
                "nexus_query_engine.runtime.cache.factory",
                "cache allocation domain does not match its ledger",
            )
            .with_fact("reason", "cache_domain_ledger_mismatch")
            .with_fact("device_ordinal", ledger.device_ordinal())
            .with_fact("domain_device_ordinal", domain.device().ordinal())
            .with_fact("cache_cap_bytes", capacity)
            .with_fact("domain_capacity_bytes", domain.capacity_bytes()));
        }
        Ok(Some(Self { ledger, domain: Arc::new(domain) }))
    }

    /// Lease the shared domain for the source-cache subdomain.
    pub(crate) fn source_cache(&self) -> Result<GpuLocalParquetSourceCache> {
        let lease = self.domain.lease_runtime().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.runtime.cache.factory.source",
                "failed to lease the shared cache allocation domain for source cache",
                error,
            )
        })?;
        GpuLocalParquetSourceCache::new(lease, self.ledger.clone())
    }

    /// Lease the shared domain for the graph-cache subdomain.
    #[cfg(feature = "cugraph")]
    pub(crate) fn graph_cache(&self) -> Result<GpuResidentGraphCache> {
        GpuResidentGraphCache::new(self.ledger.clone(), (*self.domain).clone())
    }

    /// Samples the one physical allocation domain shared by both cache
    /// subdomains.  Callers add their own activity observations; this owner
    /// intentionally has no duplicate resident-byte counter.
    pub(crate) fn report(&self) -> Result<DeviceCacheReport> {
        const MAX_COHERENCE_ATTEMPTS: usize = 4;
        for _ in 0..MAX_COHERENCE_ATTEMPTS {
            let before = self.ledger.report()?;
            let usage = self.domain.usage().map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.runtime.cache.factory.report",
                    "failed to sample the shared cache allocation domain",
                    error,
                )
                .with_fact("device_ordinal", self.ledger.device_ordinal())
                .with_fact("cache_allocation_domain_id", self.domain.id())
            })?;
            let after = self.ledger.report()?;
            let cache_residency_unchanged = before.cache_evictable_resident_bytes
                == after.cache_evictable_resident_bytes
                && before.cache_pinned_in_use_bytes == after.cache_pinned_in_use_bytes;
            if before.sequence != after.sequence && !cache_residency_unchanged {
                continue;
            }
            return Ok(DeviceCacheReport {
                device_ordinal: self.ledger.device_ordinal(),
                allocation_domain_id: self.domain.id().get(),
                cache_cap_bytes: self.domain.capacity_bytes(),
                ledger_sequence: after.sequence,
                ledger_evictable_resident_bytes: after.cache_evictable_resident_bytes,
                ledger_pinned_in_use_bytes: after.cache_pinned_in_use_bytes,
                allocation_limit_bytes: usage.allocation_limit_bytes,
                physical_current_bytes: usage.current_bytes,
                physical_current_aligned_bytes: usage.current_aligned_bytes,
                physical_peak_bytes: usage.peak_bytes,
                physical_total_bytes: usage.total_bytes,
                physical_total_allocations: usage.total_allocations,
                source: Default::default(),
                graph: Default::default(),
            });
        }
        Err(QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.runtime.cache.factory.report",
            "cache residency changed throughout the paired physical usage sample",
        )
        .with_fact("reason", "cache_report_residency_unstable")
        .with_fact("device_ordinal", self.ledger.device_ordinal())
        .with_fact("cache_allocation_domain_id", self.domain.id())
        .with_fact("max_coherence_attempts", MAX_COHERENCE_ATTEMPTS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_cap_does_not_create_a_domain() {
        let ledger = DeviceLedger::for_test(0, 1024, 1);
        assert!(DeviceCacheAllocationFactory::new(ledger).unwrap().is_none());
    }
}
