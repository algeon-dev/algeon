use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    error::Result,
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
};

/// Device budget, in bytes, that benchmark profiles carry when no budget
/// override is configured. It equals the fallback capacity of
/// [`GpuDeviceMemoryLimit::default`].
pub(crate) const DEFAULT_NATIVE_DEVICE_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

pub(crate) fn build_installed_gpu_backend(
    native_optimizer_config: AlgeonNativeOptimizerConfig,
    device_memory_limit: GpuDeviceMemoryLimit,
    source_work_per_attempt: usize,
    max_active_attempts: usize,
    execution_mode: AlgeonGpuExecutionMode,
) -> Result<AlgeonGpuBackend> {
    let native_optimizer_config = native_optimizer_config.with_execution_mode(execution_mode);
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
