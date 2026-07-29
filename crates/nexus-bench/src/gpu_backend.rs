use std::time::Duration;

use nexus_query_engine::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendSession, GpuBackendTenantDraft, GpuDeviceMemoryLimit,
        GpuMemoryOwnership, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryServiceConfig, QueryServiceTenantConfig,
    },
};

const OPERATION: &str = "nexus_bench.gpu_backend";

/// Process-owned GPU resources for benchmark and stress applications.
///
/// Sessions are deliberately weak capabilities. Keeping this owner alive is
/// what keeps the installed runtime, allocator, pools, and caches available.
#[derive(Debug)]
pub(crate) struct GpuBackendOwner {
    core: GpuBackendCore,
    session: GpuBackendSession,
}

impl GpuBackendOwner {
    pub(crate) fn build(
        config: QueryServiceConfig,
        device_ordinals: Vec<usize>,
        tenants: Vec<QueryServiceTenantConfig>,
    ) -> Result<Self> {
        let device_count = device_ordinals.len();
        if device_count == 0 {
            return Err(
                invalid_config("device_ordinals", "must contain at least one device").with_fact("reason", "empty")
            );
        }
        if config.max_concurrent_queries < device_count {
            return Err(invalid_config("max_concurrent_queries", "must be at least the number of configured devices")
                .with_fact("reason", "below_device_count")
                .with_fact("max_concurrent_queries", config.max_concurrent_queries)
                .with_fact("device_count", device_count));
        }

        let base_active = config.max_concurrent_queries / device_count;
        let remainder = config.max_concurrent_queries % device_count;
        let active_limits =
            (0..device_count).map(|index| base_active + usize::from(index < remainder)).collect::<Vec<_>>();
        if let Some(per_device_cap) = config.max_queries_per_device
            && active_limits.iter().any(|active| *active > per_device_cap)
        {
            return Err(invalid_config(
                "max_queries_per_device",
                "cannot cover global concurrency across the configured devices",
            )
            .with_fact("reason", "global_concurrency_exceeds_device_caps")
            .with_fact("max_concurrent_queries", config.max_concurrent_queries)
            .with_fact("max_queries_per_device", per_device_cap)
            .with_fact("device_count", device_count));
        }

        let compatibility_capacity = tenants
            .iter()
            .map(|tenant| tenant.device_budget_bytes)
            .max()
            .unwrap_or(0)
            .saturating_mul(u64::try_from(config.max_concurrent_queries).unwrap_or(u64::MAX));
        let capacity_bytes = config.device_capacity_bytes.unwrap_or(compatibility_capacity);
        if capacity_bytes == 0 {
            return Err(invalid_config("device_capacity_bytes", "resolved device capacity must be positive")
                .with_fact("reason", "zero"));
        }

        let tenant_drafts = tenants
            .into_iter()
            .map(|tenant| {
                GpuBackendTenantDraft::new(
                    tenant.tenant_id,
                    GpuBackendMemoryShare::Bytes(tenant.min_device_budget_bytes),
                    GpuBackendMemoryShare::Bytes(tenant.device_budget_bytes),
                )
            })
            .collect();
        let source_reads = config.source_scheduler_policy.max_source_reads_in_flight().max(1);
        let configured_stream_pool_size = config.stream_pool_size;
        let controller = GpuBackendControllerDraft::from_legacy_policy(config, tenant_drafts);
        let devices = device_ordinals
            .into_iter()
            .zip(active_limits)
            .map(|(device_ordinal, active_limit)| {
                let stream_pool_size = configured_stream_pool_size
                    .unwrap_or_else(|| active_limit.saturating_mul(source_reads.saturating_add(1)).max(active_limit));
                GpuBackendDeviceConfig::new(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)
                    .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: capacity_bytes })
                    .with_max_active_queries(active_limit)
                    .with_installation_bounds(stream_pool_size, active_limit, active_limit)
            })
            .collect();
        let deployment = DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        );
        let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(devices, controller, deployment)).build()?;
        let session = core.session_capability();
        Ok(Self { core, session })
    }

    pub(crate) fn session(&self) -> GpuBackendSession {
        self.session.clone()
    }

    pub(crate) fn close_and_wait(&self, timeout: Duration) -> Result<()> {
        self.core.close_and_wait(timeout).map(|_| ())
    }
}

fn invalid_config(field: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message).with_fact("field", field)
}
