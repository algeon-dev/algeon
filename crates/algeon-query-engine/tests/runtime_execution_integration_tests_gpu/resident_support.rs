use std::time::Duration;

use query_engine::{
    exec,
    exec::{QueryExecutionRequest, execute_query_plan, output::QueryExecutionResult},
};
use query_planning::{error::Result, plan::QueryPlan};
use query_runtime::runtime::{
    ExecutionContext, GpuBackendDeviceConfig, GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile,
    GpuDeviceMemoryLimit, GpuMemoryOwnership, QueryExecutionConfig, QueryService, QueueBounds,
    resident::GpuResidentInputMap,
};

pub(super) fn try_guaranteed_runtime_test_service_with_concurrency(max_active_attempts: usize) -> Result<QueryService> {
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
                // One worker is reserved for lifecycle jobs.
                max_active_attempts + 1,
                max_active_attempts,
            )),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    QueryService::new_with_device_profiles_for_test(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(500) },
        vec![profile],
    )
}

pub(super) fn try_guaranteed_runtime_test_service() -> Result<QueryService> {
    try_guaranteed_runtime_test_service_with_concurrency(1)
}

pub(super) fn execute_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: Option<&GpuResidentInputMap>,
) -> Result<QueryExecutionResult> {
    let admitted =
        exec::admit_query_plan(plan, context.statistical_aggregate_execution_mode(), context.planning_policy())?;
    let mut request = QueryExecutionRequest::admitted(&admitted);
    if let Some(gpu_inputs) = gpu_inputs {
        request = request.with_gpu_inputs(gpu_inputs);
    }
    let outcome = execute_query_plan(context, request);
    let (result, metrics, _observation) = outcome.into_parts();
    result.map(|output| QueryExecutionResult::new(output, metrics.snapshot()))
}

#[cfg(feature = "cugraph")]
pub(super) fn runtime_test_service_with_profiles(
    config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    device_profiles: Vec<GpuBackendDeviceConfig>,
) -> Result<QueryService> {
    QueryService::new_with_device_profiles_for_test(config, queue_bounds, device_profiles)
}
