use std::time::Duration;

use nexus_query_engine::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendSession, GpuDeviceMemoryLimit, GpuMemoryOwnership, KvikioCompatModeStatus,
        NvidiaFsProbeStatus, QueryExecutionConfig, QueueBounds,
    },
};

const OPERATION: &str = "nexus_bench.gpu_backend";

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
        max_active_queries: usize,
        source_reads: usize,
    ) -> GpuBackendDeviceConfig {
        let streams = max_active_queries.saturating_mul(source_reads.max(1).saturating_add(1)).max(max_active_queries);
        GpuBackendDeviceConfig::new(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: capacity_bytes })
            .with_max_active_queries(max_active_queries)
            .with_installation_bounds(streams, max_active_queries, max_active_queries)
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
