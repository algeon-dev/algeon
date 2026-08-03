use std::time::Duration;

use nexus_query_engine::{
    error::Result,
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendSession, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryExecutionConfig,
        QueueBounds,
    },
};

#[derive(Debug)]
pub(crate) struct ToolGpuBackend {
    core: GpuBackendCore,
    session: GpuBackendSession,
}

impl ToolGpuBackend {
    pub(crate) fn build(config: QueryExecutionConfig, device: GpuBackendDeviceConfig) -> Result<Self> {
        let controller = GpuBackendControllerDraft::new(config, QueueBounds::default());
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller, deployment)).build()?;
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
