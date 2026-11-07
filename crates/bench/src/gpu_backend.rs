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
mod tests {
    mod gpu_exclusive {
        use std::{
            fs,
            time::{SystemTime, UNIX_EPOCH},
        };

        use algeon_datafusion::planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig};
        use datafusion::{
            execution::SessionStateBuilder,
            prelude::{ParquetReadOptions, SessionContext},
        };
        use query_runtime::runtime::GpuDeviceMemoryLimit;
        use workloads::tpcds::write_tiny_tpcds_parquet_dataset;

        use crate::gpu_backend::{build_report_gpu_backend, close_report_gpu_backend};

        #[tokio::test]
        async fn report_backend_preserves_native_required_rejection_of_partial_plans() {
            let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
            let root = std::env::temp_dir().join(format!("algeon-report-required-{id}"));
            write_tiny_tpcds_parquet_dataset(&root).unwrap();
            let config =
                AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired);
            let backend =
                build_report_gpu_backend(config, GpuDeviceMemoryLimit::Explicit { limit_bytes: 512 * 1024 * 1024 })
                    .unwrap();
            let result = {
                let state = backend.install_on(SessionStateBuilder::new_with_default_features()).unwrap().build();
                let ctx = SessionContext::new_with_state(state);
                ctx.register_parquet(
                    "customer",
                    root.join("customer.parquet").to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await
                .unwrap();
                ctx.sql("SELECT c_customer_sk, repeat(c_first_name, 2) FROM customer")
                    .await
                    .unwrap()
                    .create_physical_plan()
                    .await
            };
            close_report_gpu_backend(backend).unwrap();
            fs::remove_dir_all(root).unwrap();
            assert!(
                result.is_err(),
                "a report requiring native execution must reject a CPU projection over a GPU scan"
            );
        }
    }
}
