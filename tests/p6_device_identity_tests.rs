use std::time::Duration;

use datafusion_nexus::{
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile},
    native_exec::NexusNativeExecutionConfig,
};

const QUALIFYING_DEVICE_COUNT: usize = 4;
const L40S_MODEL_NAME: &str = "NVIDIA L40S";
const L40S_PHYSICAL_MEMORY_BYTES: u64 = 46_068 * 1024 * 1024;
const BACKEND_MEMORY_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;
const ATTEMPT_MEMORY_LIMIT_BYTES: u64 = 256 * 1024 * 1024;

fn qualifying_device_profile(device_ordinal: usize) -> NexusGpuDeviceProfile {
    NexusGpuDeviceProfile::new(device_ordinal)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: BACKEND_MEMORY_LIMIT_BYTES })
        .with_max_active_attempts(1)
        .with_installation_bounds(3, 1, 1)
}

#[test]
fn public_backend_startup_records_capture_all_qualifying_l40s_identities() {
    let device_count = cudf::DeviceId::count().expect("qualifying-host CUDA device enumeration succeeds");
    assert_eq!(device_count, QUALIFYING_DEVICE_COUNT, "P6.2 requires exactly four qualifying L40S devices");

    let device_ordinals = 0..QUALIFYING_DEVICE_COUNT;
    let backend = NexusGpuBackend::builder()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(
            ATTEMPT_MEMORY_LIMIT_BYTES,
            64 * 1024 * 1024,
            1,
            3,
            1,
        ))
        .device_profiles(device_ordinals.clone().map(qualifying_device_profile))
        .device_memory_ownership(
            device_ordinals.clone().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)),
        )
        .build()
        .expect("public backend builds across the qualifying devices");

    let records = backend.device_startup_records().to_vec();
    let close = backend.close_and_wait(Duration::from_secs(10));

    assert_eq!(records.len(), QUALIFYING_DEVICE_COUNT);
    for (device_ordinal, record) in records.iter().enumerate() {
        assert_eq!(record.device_ordinal, device_ordinal);
        assert_eq!(record.facts.device, device_ordinal);
        assert_eq!(record.model_name.as_deref(), Some(L40S_MODEL_NAME));
        assert_eq!(record.facts.model_name.as_deref(), Some(L40S_MODEL_NAME));
        assert_eq!(record.uuid, record.facts.uuid);
        assert!(record.uuid.as_deref().is_some_and(|uuid| uuid.starts_with("GPU-")));
        assert_eq!(record.physical_total_bytes, Some(L40S_PHYSICAL_MEMORY_BYTES));
        assert_eq!(record.physical_total_bytes, record.facts.physical_total_bytes);
    }

    let close = close.expect("public backend closes after identity inspection");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}
