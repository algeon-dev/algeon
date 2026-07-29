//! Service diagnostics: governor, cache, telemetry, and stream-pool waits.

use super::*;

impl QueryService {
    pub(crate) fn backend_device_startup_facts(
        &self,
    ) -> Vec<(
        crate::runtime::DeviceFacts,
        u64,
        cudf::DeviceMemoryManagerInstallationState,
        crate::runtime::gpu::registry::EngineDeviceHealthIdentity,
    )> {
        self.inner
            .resources
            .as_ref()
            .map(|resources| {
                resources
                    .devices
                    .contexts()
                    .map(|device| {
                        (
                            device.facts().clone(),
                            device.capacity_bytes(),
                            device.memory_manager_installation(),
                            device.engine_health_identity(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn memory_governor_reports(&self) -> Result<Vec<MemoryGovernorReport>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(vec![self.inner.fallback_memory_governor.report()?]);
        };
        resources.devices.contexts().map(|context| context.memory_governor().report()).collect()
    }

    /// Returns process-wide active-query slot counts for each service device.
    pub fn device_admission_snapshots(&self) -> Vec<QueryServiceDeviceAdmissionSnapshot> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        resources
            .devices
            .contexts()
            .map(|context| QueryServiceDeviceAdmissionSnapshot {
                device_ordinal: context.device().ordinal(),
                process_active_query_slots: context.active_queries(),
            })
            .collect()
    }

    /// Returns one resident graph cache row per configured service device,
    /// including disabled all-zero rows.
    pub fn resident_graph_cache_snapshots(&self) -> Result<Vec<QueryServiceGraphCacheSnapshot>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(vec![QueryServiceGraphCacheSnapshot {
                device_ordinal: self.inner.config.device_ordinal,
                enabled: self.inner.config.resident_graph_cache_max_retained_bytes.is_some(),
                ..QueryServiceGraphCacheSnapshot::default()
            }]);
        };
        resources
            .devices
            .contexts()
            .map(|context| {
                let device_ordinal = context.device().ordinal();
                #[cfg(feature = "cugraph")]
                if let Some(cache) = resources.resident_graph_cache_for_device(device_ordinal) {
                    let snapshot = cache.snapshot()?;
                    return Ok(QueryServiceGraphCacheSnapshot {
                        device_ordinal,
                        enabled: true,
                        entries: snapshot.entries,
                        resident_bytes: snapshot.resident_bytes,
                        in_use_bytes: snapshot.in_use_bytes,
                        hits: snapshot.hits,
                        misses: snapshot.misses,
                        evictions: snapshot.evictions,
                        builds_in_flight: snapshot.builds_in_flight,
                        build_waits: snapshot.build_waits,
                    });
                }
                Ok(QueryServiceGraphCacheSnapshot { device_ordinal, ..QueryServiceGraphCacheSnapshot::default() })
            })
            .collect()
    }

    /// Samples current CUDA and optional NVML telemetry for each service device.
    ///
    /// Telemetry is observability-only and is never consulted by admission or
    /// placement. A failed substrate is represented by status fields in each
    /// snapshot rather than failing diagnostics as a whole.
    pub fn device_telemetry_snapshots(&self) -> Vec<DeviceTelemetrySnapshot> {
        self.device_telemetry_snapshots_with(&NexusNvmlTelemetry)
    }

    pub(in crate::runtime::query_service::service) fn device_telemetry_snapshots_with(
        &self,
        telemetry: &dyn GpuTelemetry,
    ) -> Vec<DeviceTelemetrySnapshot> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        gpu::inventory::device_telemetry_snapshots_for_ordinals(telemetry, resources.devices.device_ordinals())
    }

    /// Returns one device memory allocation-failure report for each installed service device.
    pub fn device_memory_failure_reports(&self) -> Vec<DeviceMemoryFailureReport> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        resources.devices.memory_failure_reports()
    }

    /// Returns one dedicated GPU execution-pool report per service device.
    pub fn gpu_execution_pool_reports(&self) -> Vec<GpuExecutionPoolSnapshot> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        resources
            .devices
            .device_ordinals()
            .iter()
            .filter_map(|ordinal| resources.gpu_execution_pools.get(ordinal).map(GpuExecutionPool::snapshot))
            .collect()
    }

    /// Returns the exact current and peak query concurrency across all service GPU pools.
    pub fn gpu_execution_query_report(&self) -> GpuExecutionQuerySnapshot {
        self.inner
            .resources
            .as_ref()
            .map(|resources| resources.gpu_execution_query_tracker.snapshot())
            .unwrap_or(GpuExecutionQuerySnapshot { active_query_jobs: 0, peak_active_query_jobs: 0 })
    }

    /// Blocks until every created service runtime has completed background
    /// stream cleanup and returned to its per-device pool.
    ///
    /// This provides an explicit quiescence boundary for warmup/capture and
    /// graceful drain. It does not require unused lazy pool slots to be
    /// created. Active queries cause the call to wait until `timeout` expires.
    pub fn wait_for_stream_pools_idle(&self, timeout: Duration) -> Result<()> {
        let Some(resources) = &self.inner.resources else {
            return Ok(());
        };
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        for context in resources.devices.contexts() {
            context
                .stream_pool()
                .wait_until_idle(deadline, "nexus_query_engine.query_service.wait_for_stream_pools_idle")?;
        }
        Ok(())
    }

    /// Blocks until each pool has only its declared long-lived runtime leases
    /// checked out and every other created runtime has completed cleanup.
    ///
    /// `expected_leased_runtimes` contains `(device_ordinal, count)` pairs.
    /// Omitted configured devices expect zero leases. Duplicate or unknown
    /// ordinals are rejected instead of being silently merged.
    pub fn wait_for_stream_pools_quiescent(
        &self,
        expected_leased_runtimes: &[(usize, usize)],
        timeout: Duration,
    ) -> Result<()> {
        let operation = "nexus_query_engine.query_service.wait_for_stream_pools_quiescent";
        let Some(resources) = &self.inner.resources else {
            if expected_leased_runtimes.iter().all(|(_device_ordinal, expected)| *expected == 0) {
                return Ok(());
            }
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "expected stream-pool leases require query service runtime resources",
            )
            .with_fact("reason", "runtime_resources_unavailable"));
        };
        let mut expected_by_device = BTreeMap::new();
        for &(device_ordinal, expected) in expected_leased_runtimes {
            if expected_by_device.insert(device_ordinal, expected).is_some() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    operation,
                    "expected stream-pool lease counts contain a duplicate device ordinal",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("reason", "duplicate_device_ordinal"));
            }
            resources.devices.context_by_ordinal(device_ordinal)?;
        }
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        for context in resources.devices.contexts() {
            let expected = expected_by_device.remove(&context.device().ordinal()).unwrap_or(0);
            context.stream_pool().wait_until_quiescent(expected, deadline, operation)?;
        }
        Ok(())
    }

    /// Returns the service configuration.
    pub fn config(&self) -> &QueryServiceConfig {
        &self.inner.config
    }

    /// Returns tenant ids in their admission rotation order.
    pub fn tenant_order(&self) -> &[TenantId] {
        &self.inner.tenant_order
    }

    /// Returns a point-in-time admission snapshot for service reports.
    pub fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        let mut state = self.lock_admission()?;
        let expired = self.sweep_expired_queued_requests_locked(&mut state);
        let queued_by_tenant = self
            .inner
            .tenant_order
            .iter()
            .map(|tenant_id| TenantQueueDepth {
                tenant_id: tenant_id.clone(),
                queued_queries: state.queued.iter().filter(|request| &request.tenant_id == tenant_id).count(),
            })
            .collect();
        let snapshot = QueryServiceAdmissionSnapshot {
            active_queries: state.active_queries,
            queued_queries: state.queued.len(),
            queued_by_tenant,
            next_tenant_id: self.inner.tenant_order.get(state.next_tenant_index).cloned(),
        };
        drop(state);
        finish_expired_queued_requests(expired);
        Ok(snapshot)
    }
}
