//! Service-owned runtime resources.

use std::{collections::HashMap, sync::Arc, time::Instant};

use query_planning::error::{QueryError, QueryErrorCode, Result};

#[cfg(feature = "cugraph")]
use crate::runtime::cache::resident_graph::GpuResidentGraphCache;
use crate::runtime::{
    GpuExecutionPool, GpuExecutionQueryTracker,
    cache::{
        CacheSourceEventSnapshot, DeviceCacheAllocationFactory, DeviceCacheReport,
        local_parquet::GpuLocalParquetSourceCache,
    },
    context::DeviceRuntime,
};

/// Cache resources owned by one installed backend resource generation.
///
/// The factory is deliberately retained beside the concrete cache subdomains:
/// it owns the one physical cuDF allocation domain for an ordinal.  Keeping it
/// here means cache allocator lifetime follows the backend resource generation.
/// Neither a query controller nor an individual execution context owns that
/// lifetime.
#[derive(Debug, Clone)]
pub(super) struct DeviceResidentCaches {
    devices: HashMap<usize, DeviceResidentCache>,
}

#[derive(Debug, Clone)]
struct DeviceResidentCache {
    source: GpuLocalParquetSourceCache,
    #[cfg(feature = "cugraph")]
    graph: GpuResidentGraphCache,
    // Declare this last: every cache lease and cached allocation must be
    // destroyed before the domain owner becomes droppable. The factory is
    // retained only to keep the one physical cuDF allocation domain for the
    // ordinal alive, so it is never read directly.
    allocation_factory: DeviceCacheAllocationFactory,
}

impl DeviceResidentCaches {
    pub(super) fn install(devices: &super::DeviceRegistry) -> Result<Self> {
        let mut installed = HashMap::new();
        for context in devices.contexts() {
            let Some(allocation_factory) = DeviceCacheAllocationFactory::new(context.ledger().as_ref().clone())? else {
                continue;
            };
            let source = allocation_factory.source_cache()?;
            #[cfg(feature = "cugraph")]
            let graph = allocation_factory.graph_cache()?;
            let previous = installed.insert(
                context.device().ordinal(),
                DeviceResidentCache {
                    source,
                    #[cfg(feature = "cugraph")]
                    graph,
                    allocation_factory,
                },
            );
            debug_assert!(previous.is_none(), "a registry installs each CUDA ordinal only once");
        }
        Ok(Self { devices: installed })
    }

    pub(super) fn source_for_device(&self, device_ordinal: usize) -> Option<GpuLocalParquetSourceCache> {
        self.devices.get(&device_ordinal).map(|cache| cache.source.clone())
    }

    #[cfg(feature = "cugraph")]
    pub(super) fn graph_for_device(&self, device_ordinal: usize) -> Option<GpuResidentGraphCache> {
        self.devices.get(&device_ordinal).map(|cache| cache.graph.clone())
    }

    /// Samples each enabled device's one physical cache allocator alongside
    /// activity observed from its two subdomains.  Logical resident bytes stay
    /// in `DeviceLedgerReport`; this is the independent physical comparison
    /// surface used by service operators and cache qualification tools.
    pub(super) fn reports(&self) -> Result<Vec<DeviceCacheReport>> {
        self.devices
            .values()
            .map(|cache| {
                let mut report = cache.allocation_factory.report()?;
                report.source = cache.source.activity()?;
                #[cfg(feature = "cugraph")]
                {
                    report.graph = cache.graph.activity()?;
                }
                Ok(report)
            })
            .collect()
    }

    /// Returns the bounded source-cache events emitted by the ordinal-local
    /// cache owners.  These are diagnostic observations only; the paired
    /// ledger and allocation-domain reports remain byte-residency authority.
    pub(super) fn source_event_snapshots(&self) -> Result<Vec<CacheSourceEventSnapshot>> {
        let mut events = self
            .devices
            .values()
            .map(|cache| cache.source.event_snapshots())
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        events.sort_by_key(|event| (event.device_ordinal, event.allocation_domain_id, event.sequence));
        Ok(events)
    }

    pub(super) fn close_and_drain_until(&self, _deadline: Instant) -> Result<()> {
        // Close every graph cache before draining any, so a drain on one
        // device does not wait while another device still accepts new
        // resident graphs. Each source cache closes and drains in one call.
        #[cfg(feature = "cugraph")]
        for cache in self.devices.values() {
            cache.graph.close()?;
        }
        #[cfg(feature = "cugraph")]
        for cache in self.devices.values() {
            cache.graph.close_and_drain_until(_deadline)?;
        }
        for cache in self.devices.values() {
            cache.source.close_and_drain()?;
        }
        Ok(())
    }
}

/// The installed GPU resources of one service generation.
///
/// Field order is the drop order and is load-bearing: anything that leases
/// from a pool or cache is declared before it. The construction journal in
/// `backend_core` holds clones of the pools and contexts as well, so dropping
/// this value does not by itself destroy them; it only releases the service's
/// references.
#[derive(Debug, Clone)]
pub(super) struct QueryServiceResources {
    /// The execution-context scope of each installed device. Declared first so
    /// the pool and resident-cache clones it holds release before the caches
    /// and device contexts below.
    pub(super) device_runtimes: HashMap<usize, Arc<DeviceRuntime>>,
    pub(super) devices: super::DeviceRegistry,
    pub(super) gpu_execution_pools: HashMap<usize, GpuExecutionPool>,
    /// Process-owned resident-cache allocator domains and cache subdomains.
    /// This is declared before the registry so its leases and cached values are
    /// released before device contexts and their memory managers are dropped.
    pub(super) resident_caches: DeviceResidentCaches,
    pub(super) gpu_execution_query_tracker: GpuExecutionQueryTracker,
}

impl QueryServiceResources {
    pub(super) fn install_resident_caches(devices: &super::DeviceRegistry) -> Result<DeviceResidentCaches> {
        DeviceResidentCaches::install(devices)
    }

    /// Stops new stream leases on every device without waiting. Visits every
    /// pool even after one fails; see `CleanupErrorPriority` for which error
    /// is returned.
    pub(super) fn close_stream_pools(&self) -> Result<()> {
        let operation = "query_engine.query_service.close_stream_pools";
        let mut errors = CleanupErrorPriority::default();
        for device in self.devices.contexts() {
            if let Err(error) = device.stream_pool().close(operation) {
                errors.observe(error);
            }
        }
        errors.into_result()
    }

    pub(super) fn close_and_drain_stream_pools_until(&self, deadline: Instant) -> Result<()> {
        let operation = "query_engine.query_service.close_and_drain_stream_pools";
        let mut errors = CleanupErrorPriority::default();
        for device in self.devices.contexts() {
            if let Err(error) = device.stream_pool().close_and_wait_until(deadline, operation) {
                errors.observe(error);
            }
        }
        errors.into_result()
    }

    pub(super) fn close_and_drain_gpu_execution_pools_until(&self, deadline: Instant) -> Result<()> {
        let mut errors = CleanupErrorPriority::default();
        for pool in self.gpu_execution_pools.values() {
            if let Err(error) = pool.close_and_wait_until(deadline) {
                errors.observe(error);
            }
        }
        errors.into_result()
    }

    pub(super) fn close_and_drain_resident_caches_until(&self, deadline: Instant) -> Result<()> {
        self.resident_caches.close_and_drain_until(deadline)
    }

    pub(super) fn device_runtime(&self, device_ordinal: usize) -> Result<Arc<DeviceRuntime>> {
        self.device_runtimes.get(&device_ordinal).cloned().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.query_service.device_runtime",
                "query service runtime resources are not configured for the selected device",
            )
            .with_fact("device_ordinal", device_ordinal)
        })
    }

    pub(super) fn gpu_execution_pool_for_device(&self, device_ordinal: usize) -> Result<GpuExecutionPool> {
        self.gpu_execution_pools.get(&device_ordinal).cloned().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.query_service.gpu_execution_pool",
                "query service GPU execution pool is not configured for the selected device",
            )
            .with_fact("device_ordinal", device_ordinal)
        })
    }
}

/// Preserves phase-total cleanup while ensuring an invariant/native failure
/// cannot be hidden by an earlier caller-deadline expiration.
#[derive(Debug, Default)]
struct CleanupErrorPriority {
    first_deadline: Option<QueryError>,
    first_permanent: Option<QueryError>,
}

impl CleanupErrorPriority {
    fn observe(&mut self, error: QueryError) {
        let slot =
            if error.is_cleanup_deadline_elapsed() { &mut self.first_deadline } else { &mut self.first_permanent };
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    fn into_result(self) -> Result<()> {
        self.first_permanent.or(self.first_deadline).map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod cleanup_error_priority_tests {
    use super::*;

    #[test]
    fn later_permanent_failure_wins_over_first_deadline() {
        let deadline = QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "query_engine.query_service.test_cleanup_priority",
            "injected cleanup deadline",
        )
        .with_fact("reason", "deadline")
        .with_cleanup_deadline_elapsed();
        let permanent = QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.query_service.test_cleanup_priority",
            "injected permanent cleanup failure",
        )
        .with_fact("reason", "permanent");
        let mut errors = CleanupErrorPriority::default();

        errors.observe(deadline);
        errors.observe(permanent);

        let error = errors.into_result().unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("permanent"));
    }

    #[test]
    fn first_permanent_and_first_deadline_are_stable_within_their_class() {
        let permanent = |reason| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.test_cleanup_priority",
                "injected permanent cleanup failure",
            )
            .with_fact("reason", reason)
        };
        let mut errors = CleanupErrorPriority::default();

        errors.observe(permanent("first"));
        errors.observe(permanent("second"));

        let error = errors.into_result().unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("first"));
    }
}
