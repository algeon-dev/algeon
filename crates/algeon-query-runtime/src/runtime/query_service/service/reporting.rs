//! Service diagnostics: governor, cache, telemetry, and stream-pool waits.

#[cfg(test)]
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use query_planning::error::Result;
#[cfg(test)]
use query_planning::error::{QueryError, QueryErrorCode};

#[cfg(test)]
use super::super::admission::AdmissionState;
use super::{
    super::config::{QueryExecutionConfig, QueueBounds},
    QueryService,
};
use crate::runtime::{
    gpu::{
        self, AlgeonNvmlTelemetry, DeviceLedgerReport, DeviceMemoryFailureReport, DeviceTelemetrySnapshot, GpuTelemetry,
    },
    gpu_execution_pool::{GpuExecutionPool, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot},
    query_service::{QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot},
};

impl QueryService {
    #[doc(hidden)]
    pub fn backend_device_startup_facts(
        &self,
    ) -> Vec<(
        crate::runtime::DeviceFacts,
        u64,
        cudf::DeviceMemoryManagerInstallationState,
        crate::runtime::gpu::registry::EngineDeviceHealthIdentity,
        crate::runtime::DeviceResourceProfile,
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
                            resources
                                .devices
                                .resource_profile(device.device().ordinal())
                                .cloned()
                                .expect("installed device has its immutable resource profile"),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        let Some(resources) = &self.inner.resources else {
            return self
                .inner
                .fallback_device_ledger
                .as_ref()
                .map(|ledger| ledger.report())
                .transpose()
                .map(|report| report.into_iter().collect());
        };
        resources.devices.contexts().map(|context| context.ledger().report()).collect()
    }

    /// Returns physical allocation-domain usage and cache activity for every
    /// cache-enabled device.  It is diagnostic-only and never participates in
    /// admission; `device_ledger_reports` remains the ownership authority.
    pub fn device_cache_reports(&self) -> Result<Vec<crate::runtime::cache::DeviceCacheReport>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(Vec::new());
        };
        resources.resident_caches.reports()
    }

    /// Returns bounded source-cache state-machine observations across enabled
    /// device domains. They are correlation diagnostics, never admission or
    /// residency accounting inputs.
    pub fn device_cache_source_event_snapshots(&self) -> Result<Vec<crate::runtime::cache::CacheSourceEventSnapshot>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(Vec::new());
        };
        resources.resident_caches.source_event_snapshots()
    }

    /// Returns process-wide active-query slot counts for each service device.
    pub fn device_admission_snapshots(&self) -> Vec<QueryServiceDeviceAdmissionSnapshot> {
        let Some(resources) = &self.inner.resources else {
            return self
                .inner
                .fallback_device_ledger
                .as_ref()
                .map(|ledger| {
                    let snapshot = ledger.snapshot().ok();
                    vec![QueryServiceDeviceAdmissionSnapshot {
                        device_ordinal: ledger.device_ordinal(),
                        process_active_query_slots: snapshot.as_ref().map_or(0, |snapshot| snapshot.active_attempts),
                        has_protection: snapshot.as_ref().is_some_and(|snapshot| snapshot.protected_slot),
                    }]
                })
                .unwrap_or_default();
        };
        resources
            .devices
            .contexts()
            .map(|context| {
                let device_ordinal = context.device().ordinal();
                let snapshot = context.ledger().snapshot();
                QueryServiceDeviceAdmissionSnapshot {
                    device_ordinal,
                    process_active_query_slots: snapshot.as_ref().map_or(0, |snapshot| snapshot.active_attempts),
                    has_protection: snapshot.as_ref().is_ok_and(|snapshot| snapshot.protected_slot),
                }
            })
            .collect()
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
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        gpu::inventory::device_telemetry_snapshots_for_ordinals(
            telemetry,
            resources.devices.telemetry_device_ordinals(),
        )
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
    /// created. Active queries and rejected provisional envelopes whose
    /// rollback still retains a lane or immutable grant cause the call to wait
    /// until `timeout` expires.
    #[cfg(test)]
    pub fn wait_for_stream_pools_idle(&self, timeout: Duration) -> Result<()> {
        let Some(resources) = &self.inner.resources else {
            return Ok(());
        };
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        for context in resources.devices.contexts() {
            context.stream_pool().wait_until_idle(deadline, "query_engine.query_service.wait_for_stream_pools_idle")?;
        }
        self.wait_for_release_commits_until(deadline)
    }

    /// Waits for the release or provisional-rollback callback that follows
    /// stream-pool cleanup.
    ///
    /// A returned root stream can make a pool look idle before its cleanup
    /// completion commits the report and drops the immutable grant. The same
    /// ordering applies when an unpublished envelope rolls back. Capacity
    /// remains unavailable until that callback clears the admission counters,
    /// so the public idle fence must include both phases.
    #[cfg(test)]
    fn wait_for_release_commits_until(&self, deadline: Instant) -> Result<()> {
        let operation = "query_engine.query_service.wait_for_stream_pools_idle";
        let mut state = match crate::runtime::deadline::mutex_lock_until(&self.inner.admission, deadline) {
            Ok(state) => state,
            Err(crate::runtime::deadline::DeadlineMutexLockError::Poisoned) => {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission mutex was poisoned while waiting for stream cleanup",
                )
                .with_fact("reason", "admission_mutex_poisoned"));
            },
            Err(crate::runtime::deadline::DeadlineMutexLockError::TimedOut) => {
                return Err(stream_pool_release_commit_timeout(operation, None));
            },
        };
        loop {
            if state.lifecycle.is_quiescent() {
                return Ok(());
            }
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return Err(stream_pool_release_commit_timeout(operation, Some(&state)));
            };
            let (next, timeout) = self.inner.admission_available.wait_timeout(state, remaining).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission condition variable was poisoned while waiting for stream cleanup",
                )
                .with_fact("reason", "admission_condition_poisoned")
            })?;
            state = next;
            if timeout.timed_out() && !state.lifecycle.is_quiescent() {
                return Err(stream_pool_release_commit_timeout(operation, Some(&state)));
            }
        }
    }

    /// Blocks until each pool has only its declared long-lived runtime leases
    /// checked out and every other created runtime has completed cleanup.
    ///
    /// `expected_leased_runtimes` contains `(device_ordinal, count)` pairs.
    /// Omitted configured devices expect zero leases. Duplicate or unknown
    /// ordinals are rejected instead of being silently merged.
    #[cfg(test)]
    pub fn wait_for_stream_pools_quiescent(
        &self,
        expected_leased_runtimes: &[(usize, usize)],
        timeout: Duration,
    ) -> Result<()> {
        let operation = "query_engine.query_service.wait_for_stream_pools_quiescent";
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

    /// Returns the execution policy copied into admitted query contexts.
    pub fn config(&self) -> &QueryExecutionConfig {
        &self.inner.config
    }

    /// Returns the controller-owned bounded admission queue policy.
    pub fn queue_bounds(&self) -> &QueueBounds {
        &self.inner.queue_bounds
    }

    /// Returns a point-in-time admission snapshot for service reports.
    pub fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        let mut state = self.lock_admission()?;
        let expired = self.sweep_expired_queued_requests_locked(&mut state);
        // A ledger grant becomes provisional before the complete physical
        // envelope can publish, so provisional transactions stay hidden.
        // A published handle in terminal cleanup still owns its immutable
        // grant, however; include it until the report commit and mechanical
        // capacity release complete.
        let lifecycle = state.lifecycle.snapshot();
        let active_queries = lifecycle.active.saturating_add(lifecycle.published_cleanup);
        let snapshot = QueryServiceAdmissionSnapshot {
            active_queries,
            queued_queries: state.queued.len(),
            queue_high_watermark: state.queue_high_watermark(),
            protected_devices: self
                .device_ledger_snapshots()
                .map(|snapshots| {
                    snapshots
                        .iter()
                        .filter(|snapshot| snapshot.protected_slot)
                        .map(|snapshot| snapshot.device_ordinal)
                        .collect()
                })
                .unwrap_or_default(),
        };
        // Queue-sweep completion may reenter service callers, so release the admission lock first.
        drop(state);
        expired.finish();
        Ok(snapshot)
    }
}

#[cfg(test)]
fn stream_pool_release_commit_timeout(operation: &'static str, state: Option<&AdmissionState>) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for query-service stream cleanup to commit capacity release",
    )
    .with_fact("reason", "stream_pool_idle_timeout")
    .with_fact("cleanup_stage", "release_commit");
    if let Some(state) = state {
        let lifecycle = state.lifecycle.snapshot();
        error = error
            .with_fact("active_queries", lifecycle.active)
            .with_fact("admitting_queries", lifecycle.provisional)
            .with_fact("cleanup_in_progress", lifecycle.cleanup);
    }
    error
}

impl QueryService {
    /// Test hook: closes new admission on one device ledger, driving the
    /// queue's protection retarget path without a real device fault.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn close_device_admission_for_test(
        &self,
        device_ordinal: usize,
        reason: &str,
    ) -> Result<()> {
        self.ledger_for_ordinal(device_ordinal)?.close_admission(reason)?;
        self.inner.notify_admission_available();
        Ok(())
    }
}

impl QueryService {
    /// Largest device managed capacity represented by this query service.
    pub fn max_device_capacity_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || {
                self.inner
                    .fallback_device_ledger
                    .as_ref()
                    .expect("admission-only service has a fallback device ledger")
                    .managed_capacity_bytes()
            },
            |resources| {
                resources
                    .devices
                    .contexts()
                    .map(|context| context.capacity_bytes())
                    .max()
                    .expect("runtime service has an installed device")
            },
        )
    }

    /// Largest physical device-total byte count represented by this query service.
    ///
    /// Admission-only services have no CUDA inventory, so they fall back to the
    /// configured synthetic-ledger capacity used by deterministic CPU-only tests.
    pub fn max_device_total_memory_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || self.max_device_capacity_bytes(),
            |resources| {
                resources
                    .devices
                    .contexts()
                    .map(|context| context.facts().total_bytes)
                    .max()
                    .expect("runtime service has an installed device")
            },
        )
    }

    /// One atomic snapshot per device ledger, for placement, reports, and
    /// metrics. Admission-only services expose their synthetic ledger.
    pub(in crate::runtime::query_service) fn device_ledger_snapshots(
        &self,
    ) -> Result<Vec<crate::runtime::gpu::LedgerSnapshot>> {
        let Some(resources) = &self.inner.resources else {
            return self
                .inner
                .fallback_device_ledger
                .as_ref()
                .map(|ledger| ledger.snapshot())
                .transpose()
                .map(|snapshot| snapshot.into_iter().collect());
        };
        resources.devices.contexts().map(|context| context.ledger().snapshot()).collect()
    }
}
