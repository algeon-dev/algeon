//! Service-owned runtime resources and per-device cache construction.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct QueryServiceResources {
    pub(super) devices: super::DeviceRegistry,
    pub(super) gpu_execution_pools: HashMap<usize, GpuExecutionPool>,
    pub(super) gpu_execution_query_tracker: GpuExecutionQueryTracker,
    pub(super) _governor_claim: Option<GovernorClaimLease>,
    pub(super) lakehouse_cache_fill_budget: CacheFillByteBudget,
    pub(super) lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    pub(super) object_store_control_runtime: Arc<tokio::runtime::Runtime>,
    pub(super) source_object_stores: Arc<SourceObjectStoreRegistry>,
    pub(super) deployment_io_capability: DeploymentIoCapability,
    pub(super) local_parquet_source_caches: HashMap<usize, Arc<GpuLocalParquetSourceCache>>,
    #[cfg(feature = "cugraph")]
    pub(super) resident_graph_caches: HashMap<usize, Arc<crate::runtime::cache::resident_graph::GpuResidentGraphCache>>,
}

#[cfg(feature = "cugraph")]
pub(super) struct WarmGraphReservation {
    pub(super) stream_lease: StreamLease,
    pub(super) preparation_pin: crate::runtime::cache::resident_graph::GpuResidentGraphPreparationPin,
}

#[cfg(feature = "cugraph")]
impl std::fmt::Debug for WarmGraphReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WarmGraphReservation")
            .field("device_ordinal", &self.stream_lease.device().ordinal())
            .field("resident_key", self.preparation_pin.resident_key())
            .finish()
    }
}

impl QueryServiceResources {
    pub(super) fn close_stream_pools(&self) -> Result<()> {
        let operation = "nexus_query_engine.query_service.close_stream_pools";
        let mut first_error = None;
        for device in self.devices.contexts() {
            if let Err(error) = device.stream_pool().close(operation)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(super) fn close_and_drain_stream_pools_until(&self, deadline: Instant) -> Result<()> {
        let operation = "nexus_query_engine.query_service.close_and_drain_stream_pools";
        let mut first_error = None;
        for device in self.devices.contexts() {
            if let Err(error) = device.stream_pool().close_and_wait_until(deadline, operation)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(super) fn close_and_drain_local_parquet_source_caches_until(&self, deadline: Instant) -> Result<()> {
        let mut first_error = None;
        for cache in self.local_parquet_source_caches.values() {
            if let Err(error) = cache.close_and_drain_until(deadline)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    #[cfg(feature = "cugraph")]
    pub(super) fn close_and_drain_resident_graph_caches_until(&self, deadline: Instant) -> Result<()> {
        let mut first_error = None;
        for cache in self.resident_graph_caches.values() {
            if let Err(error) = cache.close_and_drain_until(deadline)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
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

    pub(super) fn local_parquet_source_cache_for_device(
        &self,
        device_ordinal: usize,
    ) -> Option<Arc<GpuLocalParquetSourceCache>> {
        self.local_parquet_source_caches.get(&device_ordinal).map(Arc::clone)
    }

    #[cfg(feature = "cugraph")]
    pub(super) fn resident_graph_cache_for_device(
        &self,
        device_ordinal: usize,
    ) -> Option<Arc<crate::runtime::cache::resident_graph::GpuResidentGraphCache>> {
        self.resident_graph_caches.get(&device_ordinal).map(Arc::clone)
    }

    #[cfg(feature = "cugraph")]
    pub(super) fn try_reserve_warm_graph(
        &self,
        device: &DeviceContext,
        resident_graph_key: &crate::plan::ResidentGraphKey,
    ) -> Result<Option<WarmGraphReservation>> {
        let Some(cache) = self.resident_graph_cache_for_device(device.device().ordinal()) else {
            return Ok(None);
        };
        device
            .stream_pool()
            .try_acquire_matching("nexus_query_engine.query_service.warm_graph_reservation", |runtime| {
                cache.try_pin_available_key_for_runtime(resident_graph_key, runtime)
            })
            .map(|reservation| {
                reservation
                    .map(|(stream_lease, preparation_pin)| WarmGraphReservation { stream_lease, preparation_pin })
            })
    }
}

pub(super) fn local_parquet_source_caches_for_devices(
    devices: &super::DeviceRegistry,
    max_retained_bytes: Option<u64>,
) -> Result<HashMap<usize, Arc<GpuLocalParquetSourceCache>>> {
    let Some(max_retained_bytes) = max_retained_bytes else {
        return Ok(HashMap::new());
    };
    let mut caches = HashMap::new();
    for device in devices.contexts() {
        let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(
            max_retained_bytes,
            device.memory_governor().clone(),
        ));
        GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache)?;
        caches.insert(device.device().ordinal(), cache);
    }
    Ok(caches)
}

#[cfg(feature = "cugraph")]
pub(super) fn resident_graph_caches_for_devices(
    devices: &super::DeviceRegistry,
    max_retained_bytes: Option<u64>,
    max_copies_per_key: usize,
) -> Result<HashMap<usize, Arc<crate::runtime::cache::resident_graph::GpuResidentGraphCache>>> {
    let Some(max_retained_bytes) = max_retained_bytes else {
        return Ok(HashMap::new());
    };
    let mut caches = HashMap::new();
    for device in devices.contexts() {
        let cache = Arc::new(
            crate::runtime::cache::resident_graph::GpuResidentGraphCache::new_governor_backed(
                max_retained_bytes,
                device.memory_governor().clone(),
            )
            .with_max_copies_per_key(max_copies_per_key),
        );
        crate::runtime::cache::resident_graph::GpuResidentGraphCache::register_governor_pressure_observer(&cache)?;
        caches.insert(device.device().ordinal(), cache);
    }
    Ok(caches)
}

#[cfg(test)]
pub(super) fn local_parquet_source_caches_for_ordinals(
    device_ordinals: &[usize],
    max_retained_bytes: Option<u64>,
) -> HashMap<usize, Arc<GpuLocalParquetSourceCache>> {
    let Some(max_retained_bytes) = max_retained_bytes else {
        return HashMap::new();
    };
    device_ordinals
        .iter()
        .copied()
        .map(|ordinal| (ordinal, Arc::new(GpuLocalParquetSourceCache::new(max_retained_bytes))))
        .collect()
}

#[cfg(all(test, feature = "cugraph"))]
pub(super) fn resident_graph_caches_for_ordinals(
    device_ordinals: &[usize],
    max_retained_bytes: Option<u64>,
    max_copies_per_key: usize,
) -> HashMap<usize, Arc<crate::runtime::cache::resident_graph::GpuResidentGraphCache>> {
    let Some(max_retained_bytes) = max_retained_bytes else {
        return HashMap::new();
    };
    device_ordinals
        .iter()
        .copied()
        .map(|ordinal| {
            (
                ordinal,
                Arc::new(
                    crate::runtime::cache::resident_graph::GpuResidentGraphCache::new(max_retained_bytes)
                        .with_max_copies_per_key(max_copies_per_key),
                ),
            )
        })
        .collect()
}

#[cfg(feature = "cugraph")]
pub(super) fn resident_graph_cache_max_copies_per_key(config: &QueryServiceConfig) -> usize {
    config.max_queries_per_device.unwrap_or(config.max_concurrent_queries).min(config.max_concurrent_queries)
}

pub(super) fn gpu_execution_worker_count_per_device(config: &QueryServiceConfig) -> usize {
    config.max_queries_per_device.unwrap_or(config.max_concurrent_queries).min(config.max_concurrent_queries)
}
