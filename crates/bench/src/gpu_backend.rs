#[cfg(feature = "iceberg")]
use std::time::Duration;

use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    error::Result,
    planner::{AlgeonNativeOptimizerConfig, DataFusionCpuExecutionPolicy},
};
#[cfg(feature = "iceberg")]
use query_engine::{
    error::{QueryError, QueryErrorCode, Result as QueryEngineResult},
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendDeviceProfile, GpuBackendSession, KvikioCompatModeStatus,
        NvidiaFsProbeStatus, QueryExecutionConfig, QueueBounds,
    },
};

#[cfg(feature = "iceberg")]
const OPERATION: &str = "bench.gpu_backend";

pub(crate) fn build_installed_gpu_backend(
    native_optimizer_config: AlgeonNativeOptimizerConfig,
    device_memory_limit: GpuDeviceMemoryLimit,
    source_work_per_attempt: usize,
    max_active_attempts: usize,
    datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy,
) -> Result<AlgeonGpuBackend> {
    let native_optimizer_config =
        native_optimizer_config.with_datafusion_cpu_execution_policy(datafusion_cpu_execution_policy);
    let device_profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(device_memory_limit)
        .with_cache_cap_bytes(0)
        .with_source_work_per_attempt(source_work_per_attempt)
        .with_derived_installation_bounds(max_active_attempts);
    let device_config = AlgeonGpuDeviceConfig::new(device_profile, GpuMemoryOwnership::WholeDeviceExclusive);
    AlgeonGpuBackend::builder()
        .with_optimizer_config(native_optimizer_config)
        .devices([device_config])
        .queue_policy(AlgeonGpuQueuePolicy::default())
        .build()
}

/// Process-owned GPU resources for benchmark and stress applications.
///
/// Sessions are deliberately weak capabilities. Keeping this owner alive is
/// what keeps the installed runtime, allocator, pools, and caches available.
#[derive(Debug)]
#[cfg(feature = "iceberg")]
pub(crate) struct GpuBackendOwner {
    core: GpuBackendCore,
    session: GpuBackendSession,
}

#[cfg(feature = "iceberg")]
impl GpuBackendOwner {
    pub(crate) fn device_config(
        device_ordinal: usize,
        capacity_bytes: u64,
        max_active_attempts: usize,
        source_reads: usize,
    ) -> GpuBackendDeviceConfig {
        GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(device_ordinal)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: capacity_bytes })
                .with_source_work_per_attempt(source_reads.max(1))
                .with_derived_installation_bounds(max_active_attempts),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )
    }

    pub(crate) fn build(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        devices: Vec<GpuBackendDeviceConfig>,
    ) -> QueryEngineResult<Self> {
        let device_count = devices.len();
        if device_count == 0 {
            return Err(
                invalid_config("device_ordinals", "must contain at least one device").with_fact("reason", "empty")
            );
        }
        let controller = GpuBackendControllerDraft::new(config, queue_bounds);
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(devices, controller, deployment)).build()?;
        let session = core.session_capability();
        Ok(Self { core, session })
    }

    pub(crate) fn session(&self) -> GpuBackendSession {
        self.session.clone()
    }

    pub(crate) fn close_and_wait(&self, timeout: Duration) -> QueryEngineResult<()> {
        self.core.close_and_wait(timeout).map(|_| ())
    }
}

#[cfg(feature = "iceberg")]
fn invalid_config(field: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message).with_fact("field", field)
}
