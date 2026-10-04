use query_planning::error::Result;
use query_runtime::runtime::{
    GpuBackendConfig, GpuBackendCore, GpuBackendDeviceConfig, GpuBackendSession, QueryExecutionConfig, QueueBounds,
};

#[derive(Debug)]
pub(crate) struct ToolGpuBackend {
    _core: GpuBackendCore,
    session: GpuBackendSession,
}

impl ToolGpuBackend {
    pub(crate) fn build(config: QueryExecutionConfig, device: GpuBackendDeviceConfig) -> Result<Self> {
        let core =
            GpuBackendCore::new(GpuBackendConfig::new(vec![device], config, QueueBounds::default()).validate()?)?;
        let session = core.session_capability();
        Ok(Self { _core: core, session })
    }

    pub(crate) fn session(&self) -> GpuBackendSession {
        self.session.clone()
    }
}
