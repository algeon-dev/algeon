use std::time::Duration;

use nexus_query_engine::{
    error::Result,
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendSession, GpuBackendTenantDraft, GpuDeviceMemoryLimit,
        GpuMemoryOwnership, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryServiceConfig, TenantId,
    },
};

#[derive(Debug)]
pub(crate) struct ToolGpuBackend {
    core: GpuBackendCore,
    session: GpuBackendSession,
    tenant_id: TenantId,
}

impl ToolGpuBackend {
    pub(crate) fn build(
        config: QueryServiceConfig,
        memory_limit: GpuDeviceMemoryLimit,
        minimum: GpuBackendMemoryShare,
        preferred: GpuBackendMemoryShare,
        tenant_id: impl Into<TenantId>,
    ) -> Result<Self> {
        let tenant_id = tenant_id.into();
        let max_active_queries = config.max_concurrent_queries;
        let source_reads = config.source_scheduler_policy.max_source_reads_in_flight().max(1);
        let stream_pool_size = config.stream_pool_size.unwrap_or_else(|| {
            max_active_queries.saturating_mul(source_reads.saturating_add(1)).max(max_active_queries)
        });
        let controller = GpuBackendControllerDraft::from_legacy_policy(
            config,
            vec![GpuBackendTenantDraft::new(tenant_id.clone(), minimum, preferred)],
        );
        let device = GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)
            .with_memory_limit(memory_limit)
            .with_max_active_queries(max_active_queries)
            .with_installation_bounds(stream_pool_size, max_active_queries, max_active_queries);
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller, deployment)).build()?;
        let session = core.session_capability();
        Ok(Self { core, session, tenant_id })
    }

    pub(crate) fn session(&self) -> GpuBackendSession {
        self.session.clone()
    }

    pub(crate) fn tenant_id(&self) -> TenantId {
        self.tenant_id.clone()
    }

    pub(crate) fn close_and_wait(&self, timeout: Duration) -> Result<()> {
        self.core.close_and_wait(timeout).map(|_| ())
    }
}
