//! Per-device service runtime context.

use std::{sync, sync::Arc};

use cudf::runtime::{DeviceHealthStatus, device_health_snapshot};
use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    DeviceFacts, DeviceLedger, DeviceMemoryFailureReport, DeviceMemoryFailureTracker, MemoryGrant,
    registry::EngineDeviceHealthIdentity,
};
use crate::{
    observability::{AdmissionRequestId, AttemptId},
    runtime::{ServiceStreamPool, gpu::lanes::DeviceLanePermits},
};

/// Service runtime state for one CUDA device ordinal.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct DeviceContext {
    inner: Arc<DeviceContextInner>,
}

#[derive(Debug)]
struct DeviceContextInner {
    device: cudf::DeviceId,
    stream_pool: ServiceStreamPool,
    // Keep-alive ownership guard: the manager owns the device's installed
    // reservation-aware memory resource (whose `Drop` tears it down), so the
    // context must hold it for the whole service lifetime; nothing reads it.
    _mr_manager: cudf::DeviceMemoryManager,
    memory_manager_installation: cudf::DeviceMemoryManagerInstallationState,
    engine_health_identity: EngineDeviceHealthIdentity,
    memory_failure_tracker: DeviceMemoryFailureTracker,
    ledger: Arc<DeviceLedger>,
    facts: DeviceFacts,
    lane_permits: Arc<DeviceLanePermits>,
}

impl DeviceContext {
    /// Creates a per-device service context.
    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "service construction inputs are distinct admission and ownership values"
    )]
    pub fn new(
        device: cudf::DeviceId,
        stream_pool: ServiceStreamPool,
        mr_manager: cudf::DeviceMemoryManager,
        memory_manager_installation: cudf::DeviceMemoryManagerInstallationState,
        engine_health_identity: EngineDeviceHealthIdentity,
        memory_failure_tracker: DeviceMemoryFailureTracker,
        ledger: Arc<DeviceLedger>,
        facts: DeviceFacts,
        max_active_attempts: usize,
    ) -> Result<Self> {
        if device.ordinal() != facts.device {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device context facts must describe the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("facts_device_ordinal", facts.device));
        }
        if stream_pool.device() != device {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device context stream pool must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("pool_device_ordinal", stream_pool.device().ordinal()));
        }
        if mr_manager.device() != device {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device context memory manager must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("memory_manager_device_ordinal", mr_manager.device().ordinal()));
        }
        if memory_manager_installation.device_ordinal != device.ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device memory manager installation record must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("installation_device_ordinal", memory_manager_installation.device_ordinal));
        }
        if memory_manager_installation.installation_generation
            != Some(engine_health_identity.memory_manager_installation_generation)
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "engine health identity must bind the installed memory-manager generation",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("reason", "engine_health_installation_generation_mismatch")
            .with_fact(
                "engine_memory_manager_installation_generation",
                engine_health_identity.memory_manager_installation_generation,
            )
            .with_fact(
                "memory_manager_installation_generation",
                format!("{:?}", memory_manager_installation.installation_generation),
            ));
        }
        if memory_failure_tracker.device_ordinal() != device.ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device context memory failure tracker must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("memory_failure_tracker_device_ordinal", memory_failure_tracker.device_ordinal()));
        }
        if ledger.device_ordinal() != device.ordinal() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device context memory ledger must match the context device",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("device_ledger_ordinal", ledger.device_ordinal()));
        }
        if max_active_attempts == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device active-attempt limit must be greater than zero",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("field", "max_active_attempts")
            .with_fact("reason", "zero"));
        }
        if max_active_attempts != ledger.profile().max_active_attempts {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_context",
                "device active-attempt limit must match the ledger profile",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("field", "max_active_attempts")
            .with_fact("reason", "active_limit_mismatches_ledger_profile")
            .with_fact("max_active_attempts", max_active_attempts)
            .with_fact("ledger_max_active_attempts", ledger.profile().max_active_attempts));
        }
        let lane_permits =
            sync::Arc::new(DeviceLanePermits::new(device.ordinal(), ledger.profile().max_active_attempts));
        Ok(Self {
            inner: Arc::new(DeviceContextInner {
                device,
                stream_pool,
                _mr_manager: mr_manager,
                memory_manager_installation,
                engine_health_identity,
                memory_failure_tracker,
                ledger,
                facts,
                lane_permits,
            }),
        })
    }

    /// CUDA device served by this context.
    pub fn device(&self) -> cudf::DeviceId {
        self.inner.device
    }

    /// Service stream pool for this device.
    pub fn stream_pool(&self) -> &ServiceStreamPool {
        &self.inner.stream_pool
    }

    /// Immutable process-installation record captured before this device was
    /// published in the backend core.
    pub fn memory_manager_installation(&self) -> cudf::DeviceMemoryManagerInstallationState {
        self.inner.memory_manager_installation
    }

    #[doc(hidden)]
    pub fn engine_health_identity(&self) -> EngineDeviceHealthIdentity {
        self.inner.engine_health_identity
    }

    /// Signal-only allocation-failure counters from the device memory manager callback.
    pub fn memory_failure_report(&self) -> DeviceMemoryFailureReport {
        self.inner.memory_failure_tracker.report()
    }

    /// Per-device versioned immutable-grant ledger.
    #[doc(hidden)]
    pub fn ledger(&self) -> &Arc<DeviceLedger> {
        &self.inner.ledger
    }

    /// Per-device attempt lane array (`A_d` lanes, one per admitted attempt).
    #[doc(hidden)]
    pub fn lane_permits(&self) -> &Arc<DeviceLanePermits> {
        &self.inner.lane_permits
    }

    /// Stable inventory facts captured when the registry was installed.
    pub fn facts(&self) -> &DeviceFacts {
        &self.inner.facts
    }

    /// Device managed capacity used for granted accounting.
    pub fn capacity_bytes(&self) -> u64 {
        self.inner.ledger.managed_capacity_bytes()
    }

    /// Folds a persistent native CUDA fault into the owning admission ledger.
    ///
    /// The ledger remains the scheduler's only health authority. This method
    /// is its device-context boundary: it samples the sibling runtime's
    /// process-lifetime fault state and synchronously closes this device's
    /// ledger before placement can issue another grant.
    #[doc(hidden)]
    pub fn synchronize_native_admission_health(&self) -> Result<bool> {
        let health = device_health_snapshot(self.inner.device);
        if health.status() != DeviceHealthStatus::Unhealthy {
            return Ok(false);
        }
        self.inner.ledger.close_admission_if_open(format!(
            "persistent CUDA fault on device {} (fault_count={}, first_fault={:?})",
            self.inner.device.ordinal(),
            health.fault_count(),
            health.first_fault()
        ))
    }

    /// Attempts one immutable grant admission against this device's ledger.
    ///
    /// The ledger transaction owns both the active-attempt count and the
    /// grant bytes; there is no separate slot to acquire.
    #[doc(hidden)]
    pub fn try_admit_attempt(
        &self,
        waiter_id: AdmissionRequestId,
        attempt_id: AttemptId,
        requested_query_cap_bytes: Option<u64>,
    ) -> Result<Option<MemoryGrant>> {
        if self.synchronize_native_admission_health()? {
            return Ok(None);
        }
        self.inner.ledger.try_admit_for_attempt(waiter_id, attempt_id, requested_query_cap_bytes)
    }
}
