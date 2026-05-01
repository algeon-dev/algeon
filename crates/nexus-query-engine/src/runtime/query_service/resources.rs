//! Service-owned runtime resources.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct QueryServiceResources {
    pub(super) devices: super::DeviceRegistry,
    pub(super) gpu_execution_pools: HashMap<usize, GpuExecutionPool>,
    pub(super) gpu_execution_query_tracker: GpuExecutionQueryTracker,
    pub(super) lakehouse_cache_fill_budget: CacheFillByteBudget,
    pub(super) lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    pub(super) object_store_control_runtime: Arc<tokio::runtime::Runtime>,
    pub(super) source_object_stores: Arc<SourceObjectStoreRegistry>,
    pub(super) deployment_io_capability: DeploymentIoCapability,
    /// Declared last so construction rollback drops every created GPU
    /// resource before ownership can become available to a replacement.
    pub(super) owner_lease: Option<LiveBackendLease>,
}

impl QueryServiceResources {
    pub(super) fn close_stream_pools(&self) -> Result<()> {
        let operation = "nexus_query_engine.query_service.close_stream_pools";
        let mut errors = CleanupErrorPriority::default();
        for device in self.devices.contexts() {
            if let Err(error) = device.stream_pool().close(operation) {
                errors.observe(error);
            }
        }
        errors.into_result()
    }

    pub(super) fn close_and_drain_stream_pools_until(&self, deadline: Instant) -> Result<()> {
        let operation = "nexus_query_engine.query_service.close_and_drain_stream_pools";
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

    pub(super) fn gpu_execution_pool_for_device(&self, device_ordinal: usize) -> Result<GpuExecutionPool> {
        self.gpu_execution_pools.get(&device_ordinal).cloned().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.gpu_execution_pool",
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
            "nexus_query_engine.query_service.test_cleanup_priority",
            "injected cleanup deadline",
        )
        .with_fact("reason", "deadline")
        .with_cleanup_deadline_elapsed();
        let permanent = QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.query_service.test_cleanup_priority",
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
                "nexus_query_engine.query_service.test_cleanup_priority",
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

pub(super) fn gpu_execution_worker_count_per_device(_config: &QueryExecutionConfig) -> usize {
    // Backend-owned profiles determine runtime worker capacities. Direct
    // construction has no scalar compatibility override; this value is only
    // a defensive fallback for an empty registry profile.
    1
}
