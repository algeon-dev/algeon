use algeon_datafusion::backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership};
use datafusion::execution::SessionStateBuilder;

#[test]
fn backend_session_does_not_register_session_config_extension() {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();

    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("algeon_datafusion.")));
}
