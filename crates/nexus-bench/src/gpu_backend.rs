use std::time::Duration;

use datafusion_nexus::{
    backend::{
        NexusGpuBackend, NexusGpuDeviceConfig, NexusGpuDeviceInstallationBounds, NexusGpuDeviceProfile,
        NexusGpuQueuePolicy,
    },
    planner::{FinalPlanRequirement, NexusNativeOptimizerConfig},
};
use nexus_query_engine::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendSession, GpuDeviceMemoryLimit, GpuMemoryOwnership, KvikioCompatModeStatus,
        NvidiaFsProbeStatus, QueryExecutionConfig, QueueBounds,
    },
};

const OPERATION: &str = "nexus_bench.gpu_backend";

pub(crate) fn build_installed_gpu_backend(
    native_optimizer_config: NexusNativeOptimizerConfig,
    device_memory_limit: GpuDeviceMemoryLimit,
    source_work_per_attempt: usize,
    max_active_attempts: usize,
    final_plan_requirement: Option<FinalPlanRequirement>,
) -> datafusion_nexus::error::Result<NexusGpuBackend> {
    let native_optimizer_config = final_plan_requirement
        .into_iter()
        .fold(native_optimizer_config, |config, requirement| config.with_final_plan_requirement(requirement));
    let stream_pool_size =
        max_active_attempts.saturating_mul(source_work_per_attempt.max(1).saturating_add(1)).max(max_active_attempts);
    let device_profile = NexusGpuDeviceProfile::new(0)
        .with_memory_limit(device_memory_limit)
        .with_cache_cap_bytes(0)
        .with_source_work_per_attempt(source_work_per_attempt)
        .with_max_active_attempts(max_active_attempts)
        .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(
            stream_pool_size,
            max_active_attempts,
            max_active_attempts,
        ));
    let device_config = NexusGpuDeviceConfig::new(device_profile, GpuMemoryOwnership::WholeDeviceExclusive);
    NexusGpuBackend::builder()
        .with_optimizer_config(native_optimizer_config)
        .devices([device_config])
        .queue_policy(NexusGpuQueuePolicy::default())
        .build()
}

/// Process-owned GPU resources for benchmark and stress applications.
///
/// Sessions are deliberately weak capabilities. Keeping this owner alive is
/// what keeps the installed runtime, allocator, pools, and caches available.
#[derive(Debug)]
pub(crate) struct GpuBackendOwner {
    core: GpuBackendCore,
    session: GpuBackendSession,
}

impl GpuBackendOwner {
    pub(crate) fn device_config(
        device_ordinal: usize,
        capacity_bytes: u64,
        max_active_attempts: usize,
        source_reads: usize,
    ) -> GpuBackendDeviceConfig {
        let streams =
            max_active_attempts.saturating_mul(source_reads.max(1).saturating_add(1)).max(max_active_attempts);
        GpuBackendDeviceConfig::new(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: capacity_bytes })
            .with_max_active_attempts(max_active_attempts)
            .with_installation_bounds(streams, max_active_attempts, max_active_attempts)
    }

    pub(crate) fn build(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        devices: Vec<GpuBackendDeviceConfig>,
    ) -> Result<Self> {
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

    pub(crate) fn close_and_wait(&self, timeout: Duration) -> Result<()> {
        self.core.close_and_wait(timeout).map(|_| ())
    }
}

fn invalid_config(field: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message).with_fact("field", field)
}
