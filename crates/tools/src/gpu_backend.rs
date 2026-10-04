use query_planning::error::Result;
use query_runtime::runtime::{
    GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft, GpuBackendDeviceConfig, GpuBackendSession,
    QueryExecutionConfig, QueueBounds,
};

#[derive(Debug)]
pub(crate) struct ToolGpuBackend {
    _core: GpuBackendCore,
    session: GpuBackendSession,
}

impl ToolGpuBackend {
    pub(crate) fn build(config: QueryExecutionConfig, device: GpuBackendDeviceConfig) -> Result<Self> {
        let controller = GpuBackendControllerDraft::new(config, QueueBounds::default());
        let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller)).build()?;
        let session = core.session_capability();
        Ok(Self { _core: core, session })
    }

    pub(crate) fn session(&self) -> GpuBackendSession {
        self.session.clone()
    }
}
