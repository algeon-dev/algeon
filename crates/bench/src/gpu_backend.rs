use std::time::Duration;

use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    error::Result,
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
};
use datafusion::error::DataFusionError;
use workloads::WorkloadResult;

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

/// The backend a workload report executes on: one attempt at a time, with the
/// optimizer's source-read concurrency per attempt.
pub(crate) fn build_report_gpu_backend(
    optimizer_config: AlgeonNativeOptimizerConfig,
    device_memory_limit: GpuDeviceMemoryLimit,
) -> WorkloadResult<AlgeonGpuBackend> {
    let source_work_per_attempt =
        optimizer_config.native_execution_config().source_scheduler_policy().max_source_reads_in_flight();
    let execution_mode = optimizer_config.execution_mode();
    build_installed_gpu_backend(optimizer_config, device_memory_limit, source_work_per_attempt, 1, execution_mode)
        .map_err(|error| DataFusionError::External(Box::new(error)).into())
}

pub(crate) fn close_report_gpu_backend(backend: AlgeonGpuBackend) -> WorkloadResult<()> {
    backend
        .close_and_wait(Duration::from_secs(30))
        .map(|_| ())
        .map_err(|error| DataFusionError::External(Box::new(error)).into())
}

#[cfg(test)]
mod tests;
