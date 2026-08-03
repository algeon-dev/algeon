//! Process-scoped GPU backend facade for embedded DataFusion sessions.

#[cfg(test)]
use std::sync::{Mutex, MutexGuard};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use datafusion::execution::SessionStateBuilder;
pub use nexus_query_engine::runtime::GpuMemoryOwnership;
#[cfg(test)]
use nexus_query_engine::runtime::QueryServiceTenantConfig;
use nexus_query_engine::{
    mode::NativePipelineExecutionMode,
    runtime::{
        DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
        GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendSession, GpuBackendTenantDraft, GpuDeviceMemoryLimit,
        GpuMemoryFraction, KvikioCompatModeStatus, MemoryControlPolicy, NvidiaFsProbeStatus, QueryAdmissionRequest,
        QueryGrantRequest, QueryMemoryDomainPolicy, QueryPlacementPolicy, QueryServiceConfig, SlotGuarantee, TenantId,
    },
};

#[cfg(test)]
use crate::native_exec::admission::{TestAdmissionController, TestAdmissionPermit};
use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::{NexusNativeExecutionConfig, QueryAdmissionSource},
    planner::{FinalPlanRequirement, NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::{NexusSessionStateBuilderExt, install_nexus_native_rule_with_session_config},
    session_config::DataFusionNexusSessionConfig,
};

const DEFAULT_BACKEND_TENANT_ID: &str = "default";
const DEFAULT_ADMISSION_QUEUE_MULTIPLIER: usize = 64;
const DEFAULT_GRANT_MIN: MemoryShare = MemoryShare::Fraction(1.0 / 8.0);
const DEFAULT_GRANT_WANT: MemoryShare = MemoryShare::PolicyDefault;

/// Process-scoped native GPU backend shared by one or more DataFusion sessions.
#[derive(Debug, Clone)]
pub struct NexusGpuBackend {
    core: GpuBackendCore,
    session: GpuBackendSession,
    tenant_id: TenantId,
    admission_policy: NexusAdmissionPolicy,
    optimizer_config: NexusNativeOptimizerConfig,
    device_ordinals: Vec<usize>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
}

/// Backend memory policy.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NexusMemoryPolicy {
    /// Auto-size bounded execution from selected device capacity.
    BoundedAuto { device_ordinals: Vec<usize>, budget_fraction: Option<f64> },
    /// Explicit bounded execution budget and source chunk controls.
    Bounded {
        device_ordinals: Vec<usize>,
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    },
    /// No native bounded-memory controls. Backend admission currently requires bounded memory.
    Unbounded { device_ordinals: Vec<usize> },
}

impl Default for NexusMemoryPolicy {
    fn default() -> Self {
        Self::BoundedAuto { device_ordinals: vec![0], budget_fraction: None }
    }
}

/// Backend admission policy.
#[derive(Debug, Clone, PartialEq)]
pub struct NexusAdmissionPolicy {
    max_concurrent_queries: usize,
    max_queries_per_device: Option<usize>,
    placement_policy: QueryPlacementPolicy,
    grant_min: Option<MemoryShare>,
    grant_want: Option<MemoryShare>,
    slot_guarantee: SlotGuarantee,
    max_queued: Option<usize>,
    wait_timeout: Duration,
}

/// Device-memory share used in backend admission grant bands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryShare {
    /// Fraction of the resolved backend device capacity.
    Fraction(f64),
    /// Explicit byte count.
    Bytes(u64),
    /// Use the backend policy-derived default.
    ///
    /// This is valid for [`GrantBand::want`]. It is rejected anywhere a
    /// concrete byte/fraction share is required.
    PolicyDefault,
}

/// Minimum and preferred device-memory grant for each admitted query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrantBand {
    pub min: MemoryShare,
    /// Preferred grant. Use [`MemoryShare::PolicyDefault`] to keep the
    /// backend-derived default.
    pub want: MemoryShare,
}

impl NexusAdmissionPolicy {
    pub fn with_max_concurrent_queries(mut self, max_concurrent_queries: usize) -> Self {
        self.max_concurrent_queries = max_concurrent_queries;
        self
    }

    pub fn with_max_queries_per_device(mut self, max_queries_per_device: Option<usize>) -> Self {
        self.max_queries_per_device = max_queries_per_device;
        self
    }

    pub fn with_placement_policy(mut self, placement_policy: QueryPlacementPolicy) -> Self {
        self.placement_policy = placement_policy;
        self
    }

    pub fn with_grant_band(mut self, grant: GrantBand) -> Self {
        self.grant_min = Some(grant.min);
        self.grant_want = if grant.want == MemoryShare::PolicyDefault { None } else { Some(grant.want) };
        self
    }

    /// Override only the minimum per-query grant.
    ///
    /// If `want` is not set explicitly, the backend derives it from the final
    /// slot guarantee, execution mode, device count, and this minimum.
    pub fn with_grant_min(mut self, min: MemoryShare) -> Self {
        self.grant_min = Some(min);
        self
    }

    /// Override only the preferred per-query grant.
    ///
    /// The default minimum remains in force unless [`Self::with_grant_min`] or
    /// [`Self::with_grant_band`] is also used.
    pub fn with_grant_want(mut self, want: MemoryShare) -> Self {
        self.grant_want = if want == MemoryShare::PolicyDefault { None } else { Some(want) };
        self
    }

    pub fn with_slot_guarantee(mut self, slot_guarantee: SlotGuarantee) -> Self {
        self.slot_guarantee = slot_guarantee;
        self
    }

    pub fn with_max_queued(mut self, max_queued: usize) -> Self {
        self.max_queued = Some(max_queued);
        self
    }

    pub fn with_wait_timeout(mut self, wait_timeout: Duration) -> Self {
        self.wait_timeout = wait_timeout;
        self
    }

    pub fn max_concurrent_queries(&self) -> usize {
        self.max_concurrent_queries
    }

    pub fn max_queries_per_device(&self) -> Option<usize> {
        self.max_queries_per_device
    }

    pub fn placement_policy(&self) -> QueryPlacementPolicy {
        self.placement_policy
    }

    pub fn grant(&self) -> GrantBand {
        GrantBand {
            min: self.grant_min.unwrap_or(DEFAULT_GRANT_MIN),
            want: self.grant_want.unwrap_or(DEFAULT_GRANT_WANT),
        }
    }

    pub fn slot_guarantee(&self) -> SlotGuarantee {
        self.slot_guarantee
    }

    pub fn max_queued(&self) -> usize {
        self.max_queued.unwrap_or_else(|| admission_queue_len(self.max_concurrent_queries))
    }

    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout
    }
}

impl Default for NexusAdmissionPolicy {
    fn default() -> Self {
        Self {
            max_concurrent_queries: 2,
            max_queries_per_device: None,
            placement_policy: QueryPlacementPolicy::default(),
            grant_min: None,
            grant_want: None,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            max_queued: None,
            wait_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for GrantBand {
    fn default() -> Self {
        Self { min: DEFAULT_GRANT_MIN, want: DEFAULT_GRANT_WANT }
    }
}

/// Builder for [`NexusGpuBackend`].
#[derive(Debug, Clone)]
pub struct NexusGpuBackendBuilder {
    memory_policy: NexusMemoryPolicy,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    admission_policy: NexusAdmissionPolicy,
    final_plan_requirements: Vec<FinalPlanRequirement>,
    query_service_local_parquet_source_cache_max_retained_bytes: Option<u64>,
    #[cfg(feature = "cugraph")]
    query_service_resident_graph_cache_max_retained_bytes: Option<u64>,
    query_service_query_memory_domain: Option<QueryMemoryDomainPolicy>,
    tenant_id: TenantId,
    attempt_completion_queue_capacity: Option<usize>,
}

impl NexusGpuBackend {
    pub fn builder() -> NexusGpuBackendBuilder {
        NexusGpuBackendBuilder::default()
    }

    /// Install this backend's native optimizer into a DataFusion session builder.
    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }

    pub fn install_on_with_cugraph_sql(
        &self,
        builder: SessionStateBuilder,
        cugraph_config: crate::cugraph_sql::CugraphSqlConfig,
    ) -> Result<SessionStateBuilder> {
        use crate::session::NexusSessionStateBuilderExt;

        let builder = self.install_on(builder)?;
        Ok(builder.with_cugraph_sql(
            cugraph_config.with_native_execution_config(self.optimizer_config.native_execution_config().clone()),
        ))
    }

    /// Opaque identity shared by every clone and installed session.
    pub fn identity(&self) -> nexus_query_engine::runtime::GpuBackendCoreIdentity {
        self.core.identity()
    }

    /// Immutable per-device construction records shared by every clone and
    /// session installed from this backend.
    pub fn device_startup_records(&self) -> &[nexus_query_engine::runtime::GpuBackendDeviceStartupRecord] {
        self.core.device_startup_records()
    }

    pub fn close(&self) -> bool {
        self.core.close()
    }

    pub fn close_and_wait(
        &self,
        timeout: Duration,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::GpuBackendCloseSnapshot> {
        self.core.close_and_wait(timeout)
    }

    pub fn close_and_wait_until(
        &self,
        deadline: std::time::Instant,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::GpuBackendCloseSnapshot> {
        self.core.close_and_wait_until(deadline)
    }

    pub fn admission_snapshot(
        &self,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryServiceAdmissionSnapshot> {
        self.session.admission_snapshot()
    }

    pub fn controller_config(&self) -> nexus_query_engine::error::Result<QueryServiceConfig> {
        self.session.config()
    }

    pub fn tenant_budget(&self) -> nexus_query_engine::error::Result<u64> {
        self.session.tenant_budget(&self.tenant_id)
    }

    pub fn device_admission_snapshots(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::QueryServiceDeviceAdmissionSnapshot>> {
        self.session.device_admission_snapshots()
    }

    pub fn memory_governor_reports(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::MemoryGovernorReport>> {
        self.session.memory_governor_reports()
    }

    pub fn device_memory_failure_reports(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::DeviceMemoryFailureReport>> {
        self.session.device_memory_failure_reports()
    }

    pub fn device_telemetry_snapshots(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::DeviceTelemetrySnapshot>> {
        self.session.device_telemetry_snapshots()
    }

    pub fn gpu_execution_pool_reports(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::GpuExecutionPoolSnapshot>> {
        self.session.gpu_execution_pool_reports()
    }

    pub fn gpu_execution_query_report(
        &self,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::GpuExecutionQuerySnapshot> {
        self.session.gpu_execution_query_report()
    }

    pub fn resident_graph_cache_snapshots(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::QueryServiceGraphCacheSnapshot>> {
        self.session.resident_graph_cache_snapshots()
    }

    pub fn admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.session.admit_query(self.tenant_id.clone())
    }

    pub fn try_admit_query(
        &self,
    ) -> nexus_query_engine::error::Result<Option<nexus_query_engine::runtime::QueryHandle>> {
        self.session.try_admit_query_with_request(self.tenant_id.clone(), QueryAdmissionRequest::new())
    }

    pub async fn admit_async(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.session.admit_async_with_request(self.tenant_id.clone(), QueryAdmissionRequest::new()).await
    }

    pub fn same_backend(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }

    pub(crate) fn session_capability(&self) -> &GpuBackendSession {
        &self.session
    }

    pub fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    pub(crate) fn admission_policy(&self) -> &NexusAdmissionPolicy {
        &self.admission_policy
    }

    pub fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        &self.optimizer_config
    }

    /// Exact-request BFS coalescing counters shared by sessions from this backend.
    #[cfg(feature = "cugraph")]
    pub fn cugraph_bfs_singleflight_snapshot(&self) -> crate::native_exec::graph::GraphBfsSingleflightSnapshot {
        self.optimizer_config.native_execution_config().graph_bfs_singleflight_snapshot()
    }

    /// Enables or disables exact-request BFS coalescing for subsequent executions.
    #[cfg(feature = "cugraph")]
    pub fn set_cugraph_bfs_singleflight_enabled(&self, enabled: bool) {
        self.optimizer_config.native_execution_config().set_graph_bfs_singleflight_enabled(enabled);
    }

    pub fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    /// Exact deployment ownership declaration for every selected device.
    pub fn device_memory_ownership(&self) -> &[(usize, GpuMemoryOwnership)] {
        &self.device_memory_ownership
    }

    #[cfg(feature = "server")]
    pub(crate) fn from_optimizer_config_for_admission(
        optimizer_config: NexusNativeOptimizerConfig,
        device_ordinals: Vec<usize>,
        device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
    ) -> Result<Option<Self>> {
        Self::from_optimizer_config_inner(
            optimizer_config,
            device_ordinals,
            device_memory_ownership,
            admission_policy,
            tenant_id,
        )
    }

    #[cfg(feature = "server")]
    fn from_optimizer_config_inner(
        optimizer_config: NexusNativeOptimizerConfig,
        device_ordinals: Vec<usize>,
        device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
    ) -> Result<Option<Self>> {
        let Some(resolved) = ResolvedBackendMemoryPolicy::from_execution_config(
            device_ordinals,
            optimizer_config.native_execution_config().clone(),
        )?
        else {
            return Ok(None);
        };
        Self::build_bounded(optimizer_config, resolved, device_memory_ownership, admission_policy, tenant_id, None)
            .map(Some)
    }

    fn build_bounded(
        optimizer_config: NexusNativeOptimizerConfig,
        resolved: ResolvedBackendMemoryPolicy,
        device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
        attempt_completion_queue_capacity: Option<usize>,
    ) -> Result<Self> {
        let ownership_by_ordinal =
            validate_device_memory_ownership(&resolved.device_ordinals, &device_memory_ownership)?;
        let device_memory_ownership =
            ownership_by_ordinal.iter().map(|(&ordinal, &ownership)| (ordinal, ownership)).collect();
        validate_admission_policy(&admission_policy, &resolved.device_ordinals)?;
        let execution_config = resolved.execution_config.clone();
        let service_config = backend_query_service_config(
            &execution_config,
            &admission_policy,
            &resolved,
            admission_policy.max_concurrent_queries,
        );
        let tenant = GpuBackendTenantDraft::new(
            tenant_id.clone(),
            gpu_backend_memory_share("grant.min", admission_policy.grant().min, false)?,
            gpu_backend_memory_share("grant.want", admission_policy.grant().want, true)?,
        );
        let controller = GpuBackendControllerDraft::from_legacy_policy(service_config, vec![tenant]);
        let active_limits = distributed_active_query_limits(&admission_policy, resolved.device_ordinals.len())?;
        let source_reads = execution_config.source_scheduler_policy().max_source_reads_in_flight();
        let devices = resolved
            .device_ordinals
            .iter()
            .copied()
            .zip(active_limits)
            .map(|(device_ordinal, active_limit)| {
                let stream_pool_size = backend_stream_pool_size(active_limit, source_reads);
                GpuBackendDeviceConfig::new(device_ordinal, ownership_by_ordinal[&device_ordinal])
                    .with_memory_limit(resolved.device_memory_limit)
                    .with_max_active_queries(active_limit)
                    .with_installation_bounds(stream_pool_size, active_limit, active_limit)
            })
            .collect::<Vec<_>>();
        let draft = GpuBackendCoreDraft::new(devices, controller, nongds_deployment());
        let draft = match attempt_completion_queue_capacity {
            Some(capacity) => draft.with_attempt_completion_queue_capacity(capacity),
            None => draft,
        };
        let core = GpuBackendCore::builder(draft).build().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.gpu_core",
                "failed to initialize ownership-enforcing GPU backend core",
                error,
            )
        })?;
        let session = core.session_capability();
        let grant_band = session.requested_grant_band(&tenant_id, QueryGrantRequest::new()).map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.grant_projection",
                "failed to read the resolved backend grant projection",
                error,
            )
        })?;
        let min_budget = grant_band.min_bytes;
        let projection_device = core
            .device_startup_records()
            .iter()
            .max_by_key(|record| (record.capacity_bytes, std::cmp::Reverse(record.device_ordinal)))
            .map(|record| record.device_ordinal)
            .ok_or_else(|| invalid_backend_config("device_ordinals", "backend core returned no startup records"))?;
        let mut execution_config = execution_config
            .with_device_ordinal(match admission_policy.placement_policy {
                QueryPlacementPolicy::Pinned { device_ordinal } => device_ordinal,
                _ => projection_device,
            })
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: min_budget,
                max_source_chunk_bytes: resolved.max_source_chunk_bytes,
                max_row_groups_per_chunk: resolved.max_row_groups_per_chunk,
                max_retry_attempts: resolved.max_retry_attempts,
                min_retry_chunk_row_groups: resolved.min_retry_chunk_row_groups,
            });
        let device_ordinals = resolved.device_ordinals.clone();
        let max_device_total_memory_bytes = session.max_device_total_memory_bytes().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_projection",
                "failed to read backend device facts",
                error,
            )
        })?;
        execution_config = execution_config.with_query_admission_source(QueryAdmissionSource::new(
            session.clone(),
            tenant_id.clone(),
            max_device_total_memory_bytes,
        ));
        let optimizer_config = optimizer_config.with_native_execution_config(execution_config);
        Ok(NexusGpuBackend {
            core,
            session,
            tenant_id,
            admission_policy,
            optimizer_config,
            device_ordinals,
            device_memory_ownership,
        })
    }
}

fn install_backend_optimizer_config(
    builder: SessionStateBuilder,
    optimizer_config: &NexusNativeOptimizerConfig,
) -> Result<SessionStateBuilder> {
    let builder = builder.try_with_cuvs_sql().map_err(|error| {
        Error::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.backend.install_on",
            "failed to install the Nexus cuVS query planner",
        )
        .with_source(error)
    })?;
    let builder = {
        let mut builder = builder;
        reject_preexisting_cugraph_sql(&mut builder)?;
        builder
    };
    let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let session_config =
        DataFusionNexusSessionConfig::from_optimizer_config(optimizer_config).frozen_for_backend_admission();
    install_nexus_native_rule_with_session_config(builder, rule, session_config).map_err(|error| {
        Error::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.backend.install_on",
            "pre-existing datafusion-nexus session config conflicts with backend admission",
        )
        .with_fact("reason", "session_config_conflicts_with_backend_admission")
        .with_source(error)
    })
}

/// Pure CUDA-free projection used only by root unit tests.
///
/// This is intentionally not a `NexusGpuBackend`: it owns no engine resources,
/// cannot create a runtime context, and exists only to exercise product-policy
/// lowering before the ownership-enforcing core is constructed.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct TestBackendControllerProjection {
    controller: TestAdmissionController,
    controller_config: QueryServiceConfig,
    tenant_id: TenantId,
    admission_policy: NexusAdmissionPolicy,
    optimizer_config: NexusNativeOptimizerConfig,
    device_ordinals: Vec<usize>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
}

#[cfg(test)]
impl TestBackendControllerProjection {
    pub(crate) fn controller_config(&self) -> nexus_query_engine::error::Result<QueryServiceConfig> {
        Ok(self.controller_config.clone())
    }

    pub(crate) fn tenant_budget(&self) -> nexus_query_engine::error::Result<u64> {
        self.controller.requested_grant_band(&self.tenant_id, QueryGrantRequest::new()).map(|grant| grant.want_bytes)
    }

    pub(crate) fn admission_policy(&self) -> &NexusAdmissionPolicy {
        &self.admission_policy
    }

    pub(crate) fn admit(&self) -> nexus_query_engine::error::Result<TestAdmissionPermit> {
        self.controller.admit_for_projection(&self.tenant_id)
    }

    pub(crate) fn admitted_grant_bytes(&self) -> nexus_query_engine::error::Result<u64> {
        self.controller.admitted_grant_bytes(&self.tenant_id)
    }

    pub(crate) fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        &self.optimizer_config
    }

    pub(crate) fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    pub(crate) fn device_memory_ownership(&self) -> &[(usize, GpuMemoryOwnership)] {
        &self.device_memory_ownership
    }

    pub(crate) fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }

    fn install_on_with_cugraph_sql(
        &self,
        builder: SessionStateBuilder,
        cugraph_config: crate::cugraph_sql::CugraphSqlConfig,
    ) -> Result<SessionStateBuilder> {
        use crate::session::NexusSessionStateBuilderExt;

        let builder = self.install_on(builder)?;
        Ok(builder.with_cugraph_sql(
            cugraph_config.with_native_execution_config(self.optimizer_config.native_execution_config().clone()),
        ))
    }
}

/// Root-unit-test fixture for paths that genuinely execute an installed GPU
/// backend. The process lease remains production-exclusive; tests serialize
/// construction and deterministically close the backend before releasing the
/// fixture lock.
#[cfg(test)]
pub(crate) struct InstalledTestGpuBackend {
    backend: Option<NexusGpuBackend>,
    _serial: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl InstalledTestGpuBackend {
    pub(crate) fn new() -> Self {
        static SERIAL: Mutex<()> = Mutex::new(());

        let serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 256 * 1024 * 1024,
                max_source_chunk_bytes: 64 * 1024 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
            .admission(NexusAdmissionPolicy::default())
            .build()
            .expect("root GPU unit test backend must build");
        Self { backend: Some(backend), _serial: serial }
    }

    pub(crate) fn backend(&self) -> &NexusGpuBackend {
        self.backend.as_ref().expect("test backend remains live until fixture drop")
    }

    pub(crate) fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        self.backend().optimizer_config()
    }

    pub(crate) fn admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.backend().admit_query()
    }

    pub(crate) fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        self.backend().install_on(builder)
    }
}

#[cfg(test)]
impl Drop for InstalledTestGpuBackend {
    fn drop(&mut self) {
        let Some(backend) = self.backend.take() else {
            return;
        };
        let close = backend.close_and_wait(Duration::from_secs(10));
        drop(backend);
        if let Err(error) = close
            && !std::thread::panicking()
        {
            panic!("root GPU unit test backend failed to close: {error}");
        }
    }
}

impl Default for NexusGpuBackendBuilder {
    fn default() -> Self {
        Self {
            memory_policy: NexusMemoryPolicy::default(),
            device_memory_ownership: Vec::new(),
            admission_policy: NexusAdmissionPolicy::default(),
            final_plan_requirements: Vec::new(),
            query_service_local_parquet_source_cache_max_retained_bytes: None,
            #[cfg(feature = "cugraph")]
            query_service_resident_graph_cache_max_retained_bytes: None,
            query_service_query_memory_domain: None,
            tenant_id: TenantId::new(DEFAULT_BACKEND_TENANT_ID),
            attempt_completion_queue_capacity: None,
        }
    }
}

impl NexusGpuBackendBuilder {
    /// Retain bounded reports whose completion receivers were abandoned.
    pub fn with_attempt_completion_queue_capacity(mut self, capacity: usize) -> Self {
        self.attempt_completion_queue_capacity = Some(capacity);
        self
    }

    /// Require each completed backend-installed physical plan to satisfy this
    /// native execution constraint.
    pub fn with_final_plan_requirement(mut self, requirement: FinalPlanRequirement) -> Self {
        if !self.final_plan_requirements.contains(&requirement) {
            self.final_plan_requirements.push(requirement);
        }
        self
    }

    /// Reject backend-installed physical plans that retain DataFusion CPU execution.
    pub fn with_no_datafusion_cpu_requirement(self) -> Self {
        self.with_final_plan_requirement(FinalPlanRequirement::NoDataFusionCpu)
    }

    pub fn memory_policy(mut self, memory_policy: NexusMemoryPolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    /// Declare deployment ownership for exactly the devices selected by the
    /// memory policy. Missing, extra, and duplicate ordinals are rejected
    /// before the engine begins backend construction.
    pub fn device_memory_ownership(
        mut self,
        device_memory_ownership: impl IntoIterator<Item = (usize, GpuMemoryOwnership)>,
    ) -> Self {
        self.device_memory_ownership = device_memory_ownership.into_iter().collect();
        self
    }

    pub fn admission(mut self, admission_policy: NexusAdmissionPolicy) -> Self {
        self.admission_policy = admission_policy;
        self
    }

    /// Configure the service-owned decoded local-Parquet source cache for
    /// backend-created query-service execution.
    ///
    /// `None` disables the cache. `Some(0)` is rejected during build.
    pub fn query_service_local_parquet_source_cache_max_retained_bytes(
        mut self,
        max_retained_bytes: Option<u64>,
    ) -> Self {
        self.query_service_local_parquet_source_cache_max_retained_bytes = max_retained_bytes;
        self
    }

    /// Configure the service-owned parked cuGraph cache for backend-created
    /// query-service execution.
    ///
    /// `None` disables the cache. `Some(0)` is rejected during build.
    #[cfg(feature = "cugraph")]
    pub fn query_service_resident_graph_cache_max_retained_bytes(mut self, max_retained_bytes: Option<u64>) -> Self {
        self.query_service_resident_graph_cache_max_retained_bytes = max_retained_bytes;
        self
    }

    /// Configure per-query allocator domains for backend-created query-service execution.
    ///
    /// `None` preserves reservation arenas. Durable source and graph caches
    /// remain an explicit incompatible combination until their retained values
    /// are moved to cache-owned allocator resources.
    pub fn query_service_query_memory_domain(mut self, policy: Option<QueryMemoryDomainPolicy>) -> Self {
        self.query_service_query_memory_domain = policy;
        self
    }

    pub fn tenant_id(mut self, tenant_id: impl Into<TenantId>) -> Self {
        self.tenant_id = tenant_id.into();
        self
    }

    pub fn build(self) -> Result<NexusGpuBackend> {
        self.build_inner()
    }

    #[cfg(test)]
    pub(crate) fn build_test_controller_projection(self) -> Result<TestBackendControllerProjection> {
        let mut resolved = ResolvedBackendMemoryPolicy::resolve(&self.memory_policy)?;
        resolved.execution_config =
            resolved.execution_config.with_query_service_local_parquet_source_cache_max_retained_bytes(
                self.query_service_local_parquet_source_cache_max_retained_bytes,
            );
        #[cfg(feature = "cugraph")]
        {
            resolved.execution_config =
                resolved.execution_config.with_query_service_resident_graph_cache_max_retained_bytes(
                    self.query_service_resident_graph_cache_max_retained_bytes,
                );
        }
        resolved.execution_config =
            resolved.execution_config.with_query_service_query_memory_domain(self.query_service_query_memory_domain);
        let optimizer_config = self.final_plan_requirements.into_iter().fold(
            NexusNativeOptimizerConfig::default().with_native_execution_config(resolved.execution_config.clone()),
            |optimizer_config, requirement| optimizer_config.with_final_plan_requirement(requirement),
        );
        build_test_controller_projection(
            optimizer_config,
            resolved,
            self.device_memory_ownership,
            self.admission_policy,
            self.tenant_id,
        )
    }

    fn build_inner(self) -> Result<NexusGpuBackend> {
        let mut resolved = ResolvedBackendMemoryPolicy::resolve(&self.memory_policy)?;
        resolved.execution_config =
            resolved.execution_config.with_query_service_local_parquet_source_cache_max_retained_bytes(
                self.query_service_local_parquet_source_cache_max_retained_bytes,
            );
        #[cfg(feature = "cugraph")]
        {
            resolved.execution_config =
                resolved.execution_config.with_query_service_resident_graph_cache_max_retained_bytes(
                    self.query_service_resident_graph_cache_max_retained_bytes,
                );
        }
        resolved.execution_config =
            resolved.execution_config.with_query_service_query_memory_domain(self.query_service_query_memory_domain);
        let optimizer_config = self.final_plan_requirements.into_iter().fold(
            NexusNativeOptimizerConfig::default().with_native_execution_config(resolved.execution_config.clone()),
            |optimizer_config, requirement| optimizer_config.with_final_plan_requirement(requirement),
        );
        NexusGpuBackend::build_bounded(
            optimizer_config,
            resolved,
            self.device_memory_ownership,
            self.admission_policy,
            self.tenant_id,
            self.attempt_completion_queue_capacity,
        )
    }
}

#[derive(Debug)]
struct ResolvedBackendMemoryPolicy {
    device_ordinals: Vec<usize>,
    device_memory_limit: GpuDeviceMemoryLimit,
    execution_config: NexusNativeExecutionConfig,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
}

impl ResolvedBackendMemoryPolicy {
    #[cfg(feature = "server")]
    fn from_execution_config(
        device_ordinals: Vec<usize>,
        execution_config: NexusNativeExecutionConfig,
    ) -> Result<Option<Self>> {
        validate_device_ordinals(&device_ordinals)?;
        let MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = execution_config.memory_control_policy()
        else {
            return Ok(None);
        };
        validate_positive_u64("device_budget_bytes", device_budget_bytes)?;
        validate_positive_u64("max_source_chunk_bytes", max_source_chunk_bytes)?;
        validate_positive_usize("max_row_groups_per_chunk", max_row_groups_per_chunk)?;
        validate_positive_usize("max_retry_attempts", max_retry_attempts)?;
        validate_positive_usize("min_retry_chunk_row_groups", min_retry_chunk_row_groups)?;
        Ok(Some(Self {
            execution_config: execution_config.with_device_ordinal(device_ordinals[0]),
            device_ordinals,
            device_memory_limit: GpuDeviceMemoryLimit::Explicit { limit_bytes: device_budget_bytes },
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        }))
    }

    fn resolve(policy: &NexusMemoryPolicy) -> Result<Self> {
        match policy {
            NexusMemoryPolicy::BoundedAuto { device_ordinals, budget_fraction } => {
                validate_device_ordinals(device_ordinals)?;
                let fraction = budget_fraction
                    .map(|fraction| {
                        GpuMemoryFraction::new(fraction).map_err(|error| {
                            Error::from_native(
                                "datafusion_nexus.backend.config",
                                "invalid automatic backend memory fraction",
                                error,
                            )
                        })
                    })
                    .transpose()?;
                let first_device = device_ordinals[0];
                // This value is only a pure construction placeholder for
                // policy fields. The engine resolves every device after the
                // exclusive claim and replaces it before publication.
                let budget = 1;
                Ok(Self {
                    device_ordinals: device_ordinals.clone(),
                    device_memory_limit: GpuDeviceMemoryLimit::Automatic {
                        headroom_bytes: NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES,
                        fallback_bytes: NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
                        fraction,
                    },
                    execution_config: NexusNativeExecutionConfig::bounded(
                        budget,
                        NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                        NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                        NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                        NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
                    )
                    .with_device_ordinal(first_device),
                    max_source_chunk_bytes: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                    max_row_groups_per_chunk: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                    max_retry_attempts: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                    min_retry_chunk_row_groups: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
                })
            },
            NexusMemoryPolicy::Bounded {
                device_ordinals,
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            } => {
                validate_device_ordinals(device_ordinals)?;
                validate_positive_u64("device_budget_bytes", *device_budget_bytes)?;
                validate_positive_u64("max_source_chunk_bytes", *max_source_chunk_bytes)?;
                validate_positive_usize("max_row_groups_per_chunk", *max_row_groups_per_chunk)?;
                validate_positive_usize("max_retry_attempts", *max_retry_attempts)?;
                validate_positive_usize("min_retry_chunk_row_groups", *min_retry_chunk_row_groups)?;
                let execution_config = NexusNativeExecutionConfig::bounded(
                    *device_budget_bytes,
                    *max_source_chunk_bytes,
                    *max_row_groups_per_chunk,
                    *max_retry_attempts,
                    *min_retry_chunk_row_groups,
                )
                .with_device_ordinal(device_ordinals[0]);
                Ok(Self {
                    device_ordinals: device_ordinals.clone(),
                    device_memory_limit: GpuDeviceMemoryLimit::Explicit { limit_bytes: *device_budget_bytes },
                    execution_config,
                    max_source_chunk_bytes: *max_source_chunk_bytes,
                    max_row_groups_per_chunk: *max_row_groups_per_chunk,
                    max_retry_attempts: *max_retry_attempts,
                    min_retry_chunk_row_groups: *min_retry_chunk_row_groups,
                })
            },
            NexusMemoryPolicy::Unbounded { device_ordinals } => {
                validate_device_ordinals(device_ordinals)?;
                Err(invalid_backend_config("memory_policy", "NexusGpuBackend admission requires bounded memory policy"))
            },
        }
    }
}

#[cfg(test)]
fn build_test_controller_projection(
    optimizer_config: NexusNativeOptimizerConfig,
    resolved: ResolvedBackendMemoryPolicy,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    admission_policy: NexusAdmissionPolicy,
    tenant_id: TenantId,
) -> Result<TestBackendControllerProjection> {
    let ownership_by_ordinal = validate_device_memory_ownership(&resolved.device_ordinals, &device_memory_ownership)?;
    let device_memory_ownership =
        ownership_by_ordinal.iter().map(|(&ordinal, &ownership)| (ordinal, ownership)).collect();
    validate_admission_policy(&admission_policy, &resolved.device_ordinals)?;
    let execution_config = resolved.execution_config.clone();
    let mut controller_config = backend_query_service_config(
        &execution_config,
        &admission_policy,
        &resolved,
        admission_policy.max_concurrent_queries,
    );
    let tenant_draft = GpuBackendTenantDraft::new(
        tenant_id.clone(),
        gpu_backend_memory_share("grant.min", admission_policy.grant().min, false)?,
        gpu_backend_memory_share("grant.want", admission_policy.grant().want, true)?,
    );
    let controller_draft = GpuBackendControllerDraft::from_legacy_policy(controller_config.clone(), vec![tenant_draft]);
    let active_limits = distributed_active_query_limits(&admission_policy, resolved.device_ordinals.len())?;
    let source_reads = execution_config.source_scheduler_policy().max_source_reads_in_flight();
    let devices = resolved
        .device_ordinals
        .iter()
        .copied()
        .zip(active_limits)
        .map(|(device_ordinal, active_limit)| {
            GpuBackendDeviceConfig::new(device_ordinal, ownership_by_ordinal[&device_ordinal])
                .with_memory_limit(resolved.device_memory_limit)
                .with_max_active_queries(active_limit)
                .with_installation_bounds(
                    backend_stream_pool_size(active_limit, source_reads),
                    active_limit,
                    active_limit,
                )
        })
        .collect::<Vec<_>>();
    GpuBackendCoreDraft::new(devices, controller_draft, nongds_deployment()).validate().map_err(|error| {
        Error::from_native(
            "datafusion_nexus.backend.gpu_core",
            "invalid ownership-enforcing GPU backend core configuration",
            error,
        )
    })?;

    let capacity_bytes = match resolved.device_memory_limit {
        GpuDeviceMemoryLimit::Explicit { limit_bytes } => limit_bytes,
        GpuDeviceMemoryLimit::Automatic { fallback_bytes, fraction, .. } => {
            fraction.map_or(fallback_bytes, |value| ((fallback_bytes as f64 * value.value()).floor() as u64).max(1))
        },
        _ => {
            return Err(invalid_backend_config(
                "device_memory_limit",
                "test projection does not support this device memory limit",
            ));
        },
    };
    let grant_band = admission_policy.resolve_grant_band_bytes(
        capacity_bytes,
        resolved.device_ordinals.len(),
        execution_config.native_pipeline_execution_mode(),
    )?;
    validate_grant_band_slot_guarantee(
        capacity_bytes,
        resolved.device_ordinals.len(),
        admission_policy.max_concurrent_queries,
        admission_policy.slot_guarantee,
        grant_band,
        admission_policy.max_queries_per_device,
        admission_policy.placement_policy,
    )?;
    controller_config.device_capacity_bytes = Some(capacity_bytes);
    controller_config.stream_pool_size = Some(backend_stream_pool_size(
        admission_policy.max_concurrent_queries,
        execution_config.source_scheduler_policy().max_source_reads_in_flight(),
    ));
    let controller = TestAdmissionController::new(
        controller_config.clone(),
        vec![QueryServiceTenantConfig::with_grant_range(
            tenant_id.clone(),
            grant_band.min_bytes,
            grant_band.want_bytes,
        )],
    )
    .map_err(|error| {
        Error::from_native(
            "datafusion_nexus.backend.test_controller",
            "failed to initialize CUDA-free test admission controller",
            error,
        )
    })?;
    let projection_device = match admission_policy.placement_policy {
        QueryPlacementPolicy::Pinned { device_ordinal } => device_ordinal,
        _ => resolved.device_ordinals[0],
    };
    let execution_config = execution_config
        .with_device_ordinal(projection_device)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: grant_band.min_bytes,
            max_source_chunk_bytes: resolved.max_source_chunk_bytes,
            max_row_groups_per_chunk: resolved.max_row_groups_per_chunk,
            max_retry_attempts: resolved.max_retry_attempts,
            min_retry_chunk_row_groups: resolved.min_retry_chunk_row_groups,
        })
        .with_query_admission_source(controller.source(tenant_id.clone()));
    let optimizer_config = optimizer_config.with_native_execution_config(execution_config);
    Ok(TestBackendControllerProjection {
        controller,
        controller_config,
        tenant_id,
        admission_policy,
        optimizer_config,
        device_ordinals: resolved.device_ordinals,
        device_memory_ownership,
    })
}

#[cfg(all(test, feature = "server"))]
pub(crate) fn test_controller_projection_from_optimizer_config(
    optimizer_config: NexusNativeOptimizerConfig,
    device_ordinals: Vec<usize>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    admission_policy: NexusAdmissionPolicy,
    tenant_id: TenantId,
) -> Result<Option<TestBackendControllerProjection>> {
    let Some(resolved) = ResolvedBackendMemoryPolicy::from_execution_config(
        device_ordinals,
        optimizer_config.native_execution_config().clone(),
    )?
    else {
        return Ok(None);
    };
    build_test_controller_projection(optimizer_config, resolved, device_memory_ownership, admission_policy, tenant_id)
        .map(Some)
}

fn backend_query_service_config(
    execution_config: &NexusNativeExecutionConfig,
    admission_policy: &NexusAdmissionPolicy,
    resolved: &ResolvedBackendMemoryPolicy,
    max_concurrent_queries: usize,
) -> QueryServiceConfig {
    QueryServiceConfig {
        device_ordinal: match admission_policy.placement_policy() {
            QueryPlacementPolicy::Pinned { device_ordinal } => device_ordinal,
            QueryPlacementPolicy::LeastLoaded | QueryPlacementPolicy::BestFit => resolved.device_ordinals[0],
            _ => resolved.device_ordinals[0],
        },
        lakehouse_cache_fill: execution_config.lakehouse_cache_fill_config().clone(),
        source_storage_profiles: execution_config.source_storage_profiles().to_vec(),
        boundary_policy: execution_config.boundary_policy(),
        source_scheduler_policy: execution_config.source_scheduler_policy(),
        source_chunk_planning_mode: execution_config.source_chunk_planning_mode(),
        native_pipeline_execution_mode: execution_config.native_pipeline_execution_mode(),
        stddev_samp_execution_mode: execution_config.stddev_samp_execution_mode(),
        dynamic_filter_execution_options: execution_config.dynamic_filter_execution_options(),
        keyed_hash_join_strategy_options: execution_config.keyed_hash_join_strategy_options(),
        aggregate_strategy_execution_options: execution_config.aggregate_strategy_execution_options(),
        source_chunk_planning_options: execution_config.source_chunk_planning_options(),
        graph_memory_calibration_override: execution_config.graph_memory_calibration_override(),
        local_parquet_source_cache_max_retained_bytes: execution_config
            .query_service_local_parquet_source_cache_max_retained_bytes(),
        resident_graph_cache_max_retained_bytes: execution_config
            .query_service_resident_graph_cache_max_retained_bytes(),
        query_memory_domain: execution_config.query_service_query_memory_domain(),
        device_memory_manager_base: cudf::DeviceMemoryManagerBase::default(),
        stream_pool_size: None,
        device_capacity_bytes: None,
        max_concurrent_queries,
        max_queries_per_device: admission_policy.max_queries_per_device(),
        placement_policy: admission_policy.placement_policy(),
        slot_guarantee: admission_policy.slot_guarantee(),
        max_admission_queue_len: admission_policy.max_queued(),
        admission_wait_timeout: admission_policy.wait_timeout(),
        max_source_chunk_bytes: resolved.max_source_chunk_bytes,
        max_row_groups_per_chunk: resolved.max_row_groups_per_chunk,
        max_retry_attempts: resolved.max_retry_attempts,
        min_retry_chunk_row_groups: resolved.min_retry_chunk_row_groups,
    }
}

fn backend_stream_pool_size(max_concurrent_queries: usize, max_source_reads_in_flight: usize) -> usize {
    let per_query_streams = 1usize.saturating_add(max_source_reads_in_flight.max(1));
    max_concurrent_queries.saturating_mul(per_query_streams).max(max_concurrent_queries)
}

fn gpu_backend_memory_share(
    field: &'static str,
    share: MemoryShare,
    allow_default: bool,
) -> Result<GpuBackendMemoryShare> {
    match share {
        MemoryShare::Bytes(bytes) => {
            validate_positive_u64(field, bytes)?;
            Ok(GpuBackendMemoryShare::Bytes(bytes))
        },
        MemoryShare::Fraction(fraction) => {
            GpuMemoryFraction::new(fraction).map(GpuBackendMemoryShare::Fraction).map_err(|error| {
                Error::from_native("datafusion_nexus.backend.config", "invalid backend memory share", error)
                    .with_fact("field", field)
            })
        },
        MemoryShare::PolicyDefault if allow_default => Ok(GpuBackendMemoryShare::PolicyDefault),
        MemoryShare::PolicyDefault => Err(invalid_backend_config(field, "policy default is not a concrete share")),
    }
}

fn distributed_active_query_limits(policy: &NexusAdmissionPolicy, device_count: usize) -> Result<Vec<usize>> {
    validate_positive_usize("device_count", device_count)?;
    if policy.max_concurrent_queries < device_count {
        return Err(invalid_backend_config(
            "max_concurrent_queries",
            "must be at least the number of configured devices",
        )
        .with_fact("reason", "below_device_count")
        .with_fact("max_concurrent_queries", policy.max_concurrent_queries)
        .with_fact("device_count", device_count));
    }
    let target = policy.max_concurrent_queries;
    let base = target / device_count;
    let remainder = target % device_count;
    let limits = (0..device_count).map(|index| base + usize::from(index < remainder)).collect::<Vec<_>>();
    if let Some(cap) = policy.max_queries_per_device
        && limits.iter().any(|limit| *limit > cap)
    {
        return Err(invalid_backend_config(
            "max_queries_per_device",
            "cannot cover the requested global concurrency across configured devices",
        )
        .with_fact("device_count", device_count)
        .with_fact("max_concurrent_queries", policy.max_concurrent_queries)
        .with_fact("max_queries_per_device", cap));
    }
    Ok(limits)
}

fn admission_queue_len(max_concurrent_queries: usize) -> usize {
    max_concurrent_queries.saturating_mul(DEFAULT_ADMISSION_QUEUE_MULTIPLIER).max(DEFAULT_ADMISSION_QUEUE_MULTIPLIER)
}

fn validate_admission_policy(policy: &NexusAdmissionPolicy, device_ordinals: &[usize]) -> Result<()> {
    validate_positive_usize("max_concurrent_queries", policy.max_concurrent_queries)?;
    if policy.max_concurrent_queries < device_ordinals.len() {
        return Err(invalid_backend_config(
            "max_concurrent_queries",
            "must be at least the number of configured devices",
        )
        .with_fact("reason", "below_device_count")
        .with_fact("max_concurrent_queries", policy.max_concurrent_queries)
        .with_fact("device_count", device_ordinals.len()));
    }
    if let Some(max_queries_per_device) = policy.max_queries_per_device {
        validate_positive_usize("max_queries_per_device", max_queries_per_device)?;
    }
    if let QueryPlacementPolicy::Pinned { device_ordinal } = policy.placement_policy
        && !device_ordinals.contains(&device_ordinal)
    {
        return Err(invalid_backend_config("placement_policy", "pinned device must be one of the configured devices")
            .with_fact("pinned_device_ordinal", device_ordinal)
            .with_fact("device_ordinals", device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")));
    }
    validate_positive_usize("max_queued", policy.max_queued())?;
    if policy.wait_timeout.is_zero() {
        return Err(invalid_backend_config("wait_timeout", "must be greater than zero"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedGrantBand {
    pub(crate) min_bytes: u64,
    pub(crate) want_bytes: u64,
}

impl NexusAdmissionPolicy {
    pub(crate) fn resolve_grant_band_bytes(
        &self,
        capacity_bytes: u64,
        device_count: usize,
        native_pipeline_execution_mode: NativePipelineExecutionMode,
    ) -> Result<ResolvedGrantBand> {
        let min_bytes = resolve_memory_share("grant.min", self.grant().min, capacity_bytes)?;
        let want_bytes = match self.grant_want {
            Some(want) => resolve_memory_share("grant.want", want, capacity_bytes)?,
            None => default_grant_want_bytes(
                capacity_bytes,
                min_bytes,
                device_count,
                self.max_concurrent_queries,
                self.max_queries_per_device,
                self.placement_policy,
                self.slot_guarantee,
                native_pipeline_execution_mode,
            )?,
        };
        if want_bytes < min_bytes {
            return Err(invalid_backend_config("grant.want", "must be at least grant.min"));
        }
        Ok(ResolvedGrantBand { min_bytes, want_bytes })
    }
}

fn default_grant_want_bytes(
    capacity_bytes: u64,
    min_bytes: u64,
    device_count: usize,
    max_concurrent_queries: usize,
    max_queries_per_device: Option<usize>,
    placement_policy: QueryPlacementPolicy,
    slot_guarantee: SlotGuarantee,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
) -> Result<u64> {
    validate_positive_u64("device_capacity_bytes", capacity_bytes)?;
    if slot_guarantee == SlotGuarantee::Optimistic
        && native_pipeline_execution_mode == NativePipelineExecutionMode::StrictBounded
    {
        let effective_devices = placement_eligible_device_count(device_count, placement_policy);
        let slots_per_device = max_queries_per_device.map_or_else(
            || max_concurrent_queries.div_ceil(effective_devices).max(1),
            |cap| max_concurrent_queries.div_ceil(effective_devices).max(1).min(cap),
        );
        let reserved_for_arrivals =
            min_bytes.saturating_mul(u64::try_from(slots_per_device.saturating_sub(1)).unwrap_or(u64::MAX));
        return Ok(capacity_bytes.saturating_sub(reserved_for_arrivals).max(min_bytes));
    }
    Ok(capacity_bytes)
}

fn resolve_memory_share(field: &'static str, share: MemoryShare, capacity_bytes: u64) -> Result<u64> {
    match share {
        MemoryShare::Bytes(bytes) => {
            validate_positive_u64(field, bytes)?;
            Ok(bytes)
        },
        MemoryShare::Fraction(fraction) => {
            if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) || fraction == 0.0 {
                return Err(invalid_backend_config(field, "fraction must be greater than zero and at most one"));
            }
            let bytes = (capacity_bytes as f64 * fraction).floor() as u64;
            validate_positive_u64(field, bytes)?;
            Ok(bytes)
        },
        MemoryShare::PolicyDefault => {
            Err(invalid_backend_config(field, "policy default is not a concrete memory share"))
        },
    }
}

fn validate_grant_band_slot_guarantee(
    capacity_bytes: u64,
    device_count: usize,
    max_concurrent_queries: usize,
    slot_guarantee: SlotGuarantee,
    grant_band: ResolvedGrantBand,
    max_queries_per_device: Option<usize>,
    placement_policy: QueryPlacementPolicy,
) -> Result<()> {
    if slot_guarantee != SlotGuarantee::GuaranteedMinimum {
        return Ok(());
    }
    let placement_eligible_device_count = placement_eligible_device_count(device_count, placement_policy);
    if let Some(max_queries_per_device) = max_queries_per_device {
        let max_slots_by_device_cap = u64::try_from(max_queries_per_device)
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::try_from(placement_eligible_device_count).unwrap_or(u64::MAX));
        if max_slots_by_device_cap < u64::try_from(max_concurrent_queries).unwrap_or(u64::MAX) {
            return Err(invalid_backend_config(
                "max_queries_per_device",
                "must cover max_concurrent_queries across placement-eligible devices when slot_guarantee is guaranteed_minimum",
            )
            .with_fact("bound_field", "max_concurrent_queries")
            .with_fact("max_queries_per_device", max_queries_per_device)
            .with_fact("placement_eligible_device_count", placement_eligible_device_count)
            .with_fact("max_concurrent_queries", max_concurrent_queries)
            .with_fact("slot_guarantee", slot_guarantee_backend_value(slot_guarantee)));
        }
    }
    let slots_per_device = capacity_bytes / grant_band.min_bytes;
    let slots_per_device = max_queries_per_device
        .map_or(slots_per_device, |cap| slots_per_device.min(u64::try_from(cap).unwrap_or(u64::MAX)));
    let guaranteed_min_slots =
        slots_per_device.saturating_mul(u64::try_from(placement_eligible_device_count).unwrap_or(u64::MAX));
    if guaranteed_min_slots < u64::try_from(max_concurrent_queries).unwrap_or(u64::MAX) {
        return Err(invalid_backend_config(
            "grant.min",
            "must fit max_concurrent_queries when slot_guarantee is guaranteed_minimum",
        ));
    }
    Ok(())
}

fn placement_eligible_device_count(device_count: usize, placement_policy: QueryPlacementPolicy) -> usize {
    match placement_policy {
        QueryPlacementPolicy::LeastLoaded | QueryPlacementPolicy::BestFit => device_count.max(1),
        QueryPlacementPolicy::Pinned { .. } => 1,
        _ => device_count.max(1),
    }
}

fn slot_guarantee_backend_value(slot_guarantee: SlotGuarantee) -> &'static str {
    match slot_guarantee {
        SlotGuarantee::GuaranteedMinimum => "guaranteed_minimum",
        SlotGuarantee::Optimistic => "optimistic",
    }
}

fn validate_device_ordinals(device_ordinals: &[usize]) -> Result<()> {
    if device_ordinals.is_empty() {
        return Err(invalid_backend_config("device_ordinals", "must not be empty"));
    }
    let mut seen = BTreeSet::new();
    for ordinal in device_ordinals {
        if !seen.insert(*ordinal) {
            return Err(invalid_backend_config("device_ordinals", "must not contain duplicates"));
        }
    }
    Ok(())
}

fn validate_device_memory_ownership(
    device_ordinals: &[usize],
    declarations: &[(usize, GpuMemoryOwnership)],
) -> Result<BTreeMap<usize, GpuMemoryOwnership>> {
    let selected = device_ordinals.iter().copied().collect::<BTreeSet<_>>();
    let mut ownership_by_ordinal = BTreeMap::new();
    for (device_ordinal, ownership) in declarations.iter().copied() {
        if ownership_by_ordinal.insert(device_ordinal, ownership).is_some() {
            return Err(invalid_backend_config(
                "device_memory_ownership",
                "must contain exactly one declaration for each selected device",
            )
            .with_fact("reason", "duplicate_device")
            .with_fact("device_ordinal", device_ordinal));
        }
    }

    let declared = ownership_by_ordinal.keys().copied().collect::<BTreeSet<_>>();
    let missing = selected.difference(&declared).copied().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(invalid_backend_config(
            "device_memory_ownership",
            "must declare ownership for every selected device",
        )
        .with_fact("reason", "missing_devices")
        .with_fact("device_ordinals", format_device_ordinals(&missing)));
    }
    let extra = declared.difference(&selected).copied().collect::<Vec<_>>();
    if !extra.is_empty() {
        return Err(invalid_backend_config(
            "device_memory_ownership",
            "must not declare ownership for an unselected device",
        )
        .with_fact("reason", "extra_devices")
        .with_fact("device_ordinals", format_device_ordinals(&extra)));
    }
    Ok(ownership_by_ordinal)
}

fn format_device_ordinals(device_ordinals: &[usize]) -> String {
    device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn validate_budget_fraction(fraction: f64) -> Result<()> {
    if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) || fraction == 0.0 {
        return Err(invalid_backend_config("budget_fraction", "must be greater than zero and at most one"));
    }
    Ok(())
}

fn validate_positive_u64(field: &'static str, value: u64) -> Result<()> {
    if value == 0 {
        return Err(invalid_backend_config(field, "must be greater than zero"));
    }
    Ok(())
}

fn validate_positive_usize(field: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(invalid_backend_config(field, "must be greater than zero"));
    }
    Ok(())
}

fn invalid_backend_config(field: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.backend.config", message).with_fact("field", field)
}

fn reject_preexisting_cugraph_sql(builder: &mut SessionStateBuilder) -> Result<()> {
    let Some(functions) = builder.table_functions().as_ref() else {
        return Ok(());
    };
    let Some(function_name) =
        functions.keys().filter(|name| is_cugraph_sql_table_function_name(name.as_str())).min().cloned()
    else {
        return Ok(());
    };
    Err(Error::new(
        ErrorCode::InvalidRequest,
        "datafusion_nexus.backend.install_on",
        "cuGraph SQL must be registered through backend admission",
    )
    .with_fact("reason", "preexisting_cugraph_sql_without_backend_admission")
    .with_fact("table_function", function_name)
    .with_fact("remediation", "use install_on_with_cugraph_sql or call with_cugraph_sql after install_on"))
}

fn is_cugraph_sql_table_function_name(name: &str) -> bool {
    use crate::cugraph_sql::CugraphAlgorithm;

    CugraphAlgorithm::all().any(|algorithm| algorithm.function_name() == name)
}

fn nongds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
