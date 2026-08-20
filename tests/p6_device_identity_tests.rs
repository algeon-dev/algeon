use std::time::Duration;

use datafusion_nexus::{
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile},
    native_exec::NexusNativeExecutionConfig,
};

const REQUIRED_DEVICE_COUNT: usize = 4;
const BACKEND_MEMORY_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;
const ATTEMPT_MEMORY_LIMIT_BYTES: u64 = 256 * 1024 * 1024;

fn device_profile(device_ordinal: usize) -> NexusGpuDeviceProfile {
    NexusGpuDeviceProfile::new(device_ordinal)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: BACKEND_MEMORY_LIMIT_BYTES })
        .with_max_active_attempts(1)
        .with_installation_bounds(3, 1, 1)
}

#[test]
#[ignore = "requires at least 4 GPUs"]
fn public_backend_startup_records_capture_four_device_identities() {
    let device_count = cudf::DeviceId::count().expect("CUDA device enumeration succeeds");
    assert!(device_count >= REQUIRED_DEVICE_COUNT, "identity coverage requires at least four CUDA devices");

    let device_ordinals = 0..REQUIRED_DEVICE_COUNT;
    let backend = NexusGpuBackend::builder()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(
            ATTEMPT_MEMORY_LIMIT_BYTES,
            64 * 1024 * 1024,
            1,
            3,
            1,
        ))
        .device_profiles(device_ordinals.clone().map(device_profile))
        .device_memory_ownership(
            device_ordinals.clone().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)),
        )
        .build()
        .expect("public backend builds across the qualifying devices");

    let records = backend.device_startup_records().to_vec();
    let close = backend.close_and_wait(Duration::from_secs(10));

    assert_eq!(records.len(), REQUIRED_DEVICE_COUNT);
    for (device_ordinal, record) in records.iter().enumerate() {
        assert_eq!(record.device_ordinal, device_ordinal);
        assert_eq!(record.facts.device, device_ordinal);
        assert!(record.model_name.as_deref().is_some_and(|model_name| !model_name.is_empty()));
        assert_eq!(record.model_name, record.facts.model_name);
        assert_eq!(record.uuid, record.facts.uuid);
        assert!(record.uuid.as_deref().is_some_and(|uuid| uuid.starts_with("GPU-")));
        assert!(record.physical_total_bytes.is_some_and(|bytes| bytes > 0));
        assert_eq!(record.physical_total_bytes, record.facts.physical_total_bytes);
    }

    let close = close.expect("public backend closes after identity inspection");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}
