use std::time::Duration;

pub use query_engine::{
    admission, attempt, error, exec, expr, metrics, native_error, observability, parquet_stats, plan, runtime, source,
};

use crate::{
    error::Result,
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryExecutionResult},
    plan::QueryPlan,
    runtime::{
        DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus, GpuBackendDeviceConfig,
        GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuDeviceMemoryLimit, GpuMemoryOwnership,
        KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryAdmissionRequest, QueryExecutionConfig, QueryService,
        QueueBounds, resident::GpuResidentInputMap,
    },
};

fn try_guaranteed_runtime_test_service_with_concurrency(max_active_attempts: usize) -> Result<QueryService> {
    const CAPACITY_BYTES: u64 = 64 * 1024 * 1024;

    let max_active_attempts = max_active_attempts.max(1);
    let default_query_limit_bytes =
        (CAPACITY_BYTES / u64::try_from(max_active_attempts.saturating_mul(8)).unwrap_or(u64::MAX)).max(1);
    let profile = GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: CAPACITY_BYTES })
            .with_default_query_limit_bytes(default_query_limit_bytes)
            .with_source_work_per_attempt(2)
            .with_max_active_attempts(max_active_attempts)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(
                max_active_attempts.saturating_mul(3),
                max_active_attempts,
                max_active_attempts,
            )),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    QueryService::new_with_device_profiles_and_deployment_for_test(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        vec![profile],
        deployment,
    )
}

fn try_guaranteed_runtime_test_service() -> Result<QueryService> {
    try_guaranteed_runtime_test_service_with_concurrency(1)
}

fn execute_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: Option<&GpuResidentInputMap>,
) -> Result<QueryExecutionResult> {
    let mut request = QueryExecutionRequest::new(plan);
    if let Some(gpu_inputs) = gpu_inputs {
        request = request.with_gpu_inputs(gpu_inputs);
    }
    let outcome = execute_query_plan(context, request);
    let (result, metrics, _observation) = outcome.into_parts();
    result.map(|output| QueryExecutionResult::new(output, metrics))
}

fn runtime_test_service_with_profiles(
    config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    device_profiles: Vec<GpuBackendDeviceConfig>,
) -> Result<QueryService> {
    QueryService::new_with_device_profiles_and_deployment_for_test(
        config,
        queue_bounds,
        device_profiles,
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        ),
    )
}

include!("cross_attempt_gpu_resident.rs");
include!("deferred_resident.rs");
