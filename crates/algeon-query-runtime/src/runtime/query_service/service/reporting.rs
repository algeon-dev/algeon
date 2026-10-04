//! Service diagnostics: governor, cache, telemetry, and stream-pool waits.

#[cfg(test)]
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use gpu::inventory::device_telemetry_snapshots_for_ordinals;
use query_planning::error::Result;
#[cfg(test)]
use query_planning::error::{QueryError, QueryErrorCode};

use super::{super::config::QueryExecutionConfig, QueryService};
use crate::runtime::{
    DeviceFacts, DeviceResourceProfile,
    cache::{CacheSourceEventSnapshot, DeviceCacheReport},
    gpu::{
        self, AlgeonNvmlTelemetry, DeviceLedgerReport, DeviceMemoryFailureReport, DeviceTelemetrySnapshot,
        GpuTelemetry, registry::EngineDeviceHealthIdentity,
    },
    gpu_execution_pool::{GpuExecutionPool, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot},
    query_service::{BackendDiagnostics, QueryServiceAdmissionSnapshot},
};

impl QueryService {
    pub(crate) fn backend_device_startup_facts(
        &self,
    ) -> Vec<(
        DeviceFacts,
        u64,
        cudf::DeviceMemoryManagerInstallationState,
        EngineDeviceHealthIdentity,
        DeviceResourceProfile,
    )> {
        let devices = &self.inner.resources.devices;
        devices
            .contexts()
            .map(|device| {
                (
                    device.facts().clone(),
                    device.capacity_bytes(),
                    device.memory_manager_installation(),
                    device.engine_health_identity(),
                    devices
                        .resource_profile(device.device().ordinal())
                        .cloned()
                        .expect("installed device has its immutable resource profile"),
                )
            })
            .collect()
    }

    pub(crate) fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        self.inner.admission.device_ledger_reports()
    }

    /// Returns physical allocation-domain usage and cache activity for every
    /// cache-enabled device.  It is diagnostic-only and never participates in
    /// admission; `device_ledger_reports` remains the ownership authority.
    pub(crate) fn device_cache_reports(&self) -> Result<Vec<DeviceCacheReport>> {
        self.inner.resources.resident_caches.reports()
    }

    /// Returns bounded source-cache state-machine observations across enabled
    /// device domains. They are correlation diagnostics, never admission or
    /// residency accounting inputs.
    pub(crate) fn device_cache_source_event_snapshots(&self) -> Result<Vec<CacheSourceEventSnapshot>> {
        self.inner.resources.resident_caches.source_event_snapshots()
    }

    /// Samples current CUDA and optional NVML telemetry for each service
    /// device whose immutable profile enables telemetry.
    ///
    /// Telemetry is observability-only and is never consulted by admission or
    /// placement. A failed substrate is represented by status fields in each
    /// snapshot rather than failing diagnostics as a whole.
    pub fn device_telemetry_snapshots(&self) -> Vec<DeviceTelemetrySnapshot> {
        self.device_telemetry_snapshots_with(&AlgeonNvmlTelemetry)
    }

    pub(in crate::runtime::query_service::service) fn device_telemetry_snapshots_with(
        &self,
        telemetry: &dyn GpuTelemetry,
    ) -> Vec<DeviceTelemetrySnapshot> {
        device_telemetry_snapshots_for_ordinals(telemetry, self.inner.resources.devices.telemetry_device_ordinals())
    }

    /// Returns one device memory allocation-failure report for each installed service device.
    pub(crate) fn device_memory_failure_reports(&self) -> Vec<DeviceMemoryFailureReport> {
        self.inner.resources.devices.memory_failure_reports()
    }

    /// Returns one dedicated GPU execution-pool report per service device.
    pub(crate) fn gpu_execution_pool_reports(&self) -> Vec<GpuExecutionPoolSnapshot> {
        let resources = &self.inner.resources;
        resources
            .devices
            .device_ordinals()
            .iter()
            .filter_map(|ordinal| resources.gpu_execution_pools.get(ordinal).map(GpuExecutionPool::snapshot))
            .collect()
    }

    /// Returns the exact current and peak query concurrency across all service GPU pools.
    pub(crate) fn gpu_execution_query_report(&self) -> GpuExecutionQuerySnapshot {
        self.inner.resources.gpu_execution_query_tracker.snapshot()
    }

    /// Blocks until every created service runtime has completed background
    /// stream cleanup and returned to its per-device pool.
    ///
    /// This provides an explicit quiescence boundary for warmup/capture and
    /// graceful drain. It does not require unused lazy pool slots to be
    /// created. Active queries and rejected provisional envelopes whose
    /// rollback still retains a lane or immutable grant cause the call to wait
    /// until `timeout` expires.
    #[cfg(test)]
    pub(crate) fn wait_for_stream_pools_idle(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        for context in self.inner.resources.devices.contexts() {
            context.stream_pool().wait_until_idle(deadline, "query_engine.query_service.wait_for_stream_pools_idle")?;
        }
        self.inner
            .admission
            .wait_for_lifecycle_quiescence_until(deadline, "query_engine.query_service.wait_for_stream_pools_idle")
    }

    /// Blocks until each pool has only its declared long-lived runtime leases
    /// checked out and every other created runtime has completed cleanup.
    ///
    /// `expected_leased_runtimes` contains `(device_ordinal, count)` pairs.
    /// Omitted configured devices expect zero leases. Duplicate or unknown
    /// ordinals are rejected instead of being silently merged.
    #[cfg(test)]
    pub(crate) fn wait_for_stream_pools_quiescent(
        &self,
        expected_leased_runtimes: &[(usize, usize)],
        timeout: Duration,
    ) -> Result<()> {
        let operation = "query_engine.query_service.wait_for_stream_pools_quiescent";
        let resources = &self.inner.resources;
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

    /// Returns the execution policy copied into admitted query contexts.
    pub fn config(&self) -> &QueryExecutionConfig {
        &self.inner.config
    }

    /// Reads every service report once.
    pub fn diagnostics(&self) -> Result<BackendDiagnostics> {
        Ok(BackendDiagnostics {
            admission: self.admission_snapshot()?,
            ledgers: self.device_ledger_reports()?,
            caches: self.device_cache_reports()?,
            cache_source_events: self.device_cache_source_event_snapshots()?,
            memory_failures: self.device_memory_failure_reports(),
            execution_pools: self.gpu_execution_pool_reports(),
            execution_queries: self.gpu_execution_query_report(),
        })
    }

    /// Returns a point-in-time admission snapshot for service reports.
    pub(crate) fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        self.inner.admission.admission_snapshot()
    }
}
