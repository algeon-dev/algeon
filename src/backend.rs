//! Process-scoped GPU backend facade for embedded DataFusion sessions.

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use datafusion::execution::SessionStateBuilder;
use nexus_query_engine::{
    DeploymentIoCapability, GdsDriverProbeStatus, KvikioCompatModeStatus, MemoryControlPolicy,
    NativePipelineExecutionMode, NvidiaFsProbeStatus, QueryPlacementPolicy, QueryService, QueryServiceConfig,
    QueryServiceTenantConfig, SlotGuarantee, TenantId,
};

use crate::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, Error, ErrorCode, GpuFallbackPolicy,
    NativeCudfExecutionConfig, Result, native_exec::QueryAdmissionSource,
    session::install_cudf_native_rule_with_session_config, session_config::DataFusionNexusSessionConfig,
};

const DEFAULT_BACKEND_TENANT_ID: &str = "default";
const DEFAULT_ADMISSION_QUEUE_MULTIPLIER: usize = 64;
const DEFAULT_GRANT_MIN: MemoryShare = MemoryShare::Fraction(1.0 / 8.0);
const DEFAULT_GRANT_WANT: MemoryShare = MemoryShare::PolicyDefault;

/// Process-scoped native GPU backend shared by one or more DataFusion sessions.
#[derive(Debug, Clone)]
pub struct NexusGpuBackend {
    query_service: QueryService,
    tenant_id: TenantId,
    optimizer_config: CudfOptimizerConfig,
    device_ordinals: Vec<usize>,
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
    fallback_policy: GpuFallbackPolicy,
    memory_policy: NexusMemoryPolicy,
    admission_policy: NexusAdmissionPolicy,
    tenant_id: TenantId,
}

impl NexusGpuBackend {
    pub fn builder() -> NexusGpuBackendBuilder {
        NexusGpuBackendBuilder::default()
    }

    /// Install this backend's native optimizer into a DataFusion session builder.
    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        #[cfg(feature = "cugraph")]
        let builder = {
            let mut builder = builder;
            reject_preexisting_cugraph_sql(&mut builder)?;
            builder
        };
        let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(self.optimizer_config.clone()));
        let session_config =
            DataFusionNexusSessionConfig::from_optimizer_config(&self.optimizer_config).frozen_for_backend_admission();
        install_cudf_native_rule_with_session_config(builder, rule, session_config).map_err(|error| {
            Error::new(
                ErrorCode::InvalidRequest,
                "datafusion_nexus.backend.install_on",
                "pre-existing datafusion-nexus session config conflicts with backend admission",
            )
            .with_fact("reason", "session_config_conflicts_with_backend_admission")
            .with_source(error)
        })
    }

    #[cfg(feature = "cugraph")]
    pub fn install_on_with_cugraph_sql(
        &self,
        builder: SessionStateBuilder,
        cugraph_config: crate::graph_sql::CugraphSqlConfig,
    ) -> Result<SessionStateBuilder> {
        use crate::session::CudfSessionStateBuilderExt;

        let builder = self.install_on(builder)?;
        Ok(builder.with_cugraph_sql(
            cugraph_config.with_native_execution_config(self.optimizer_config.native_execution_config().clone()),
        ))
    }

    /// Shared query-service admission owner used by sessions installed from this backend.
    pub fn query_service(&self) -> &QueryService {
        &self.query_service
    }

    pub fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    pub fn optimizer_config(&self) -> &CudfOptimizerConfig {
        &self.optimizer_config
    }

    pub fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    #[cfg(feature = "server")]
    pub(crate) fn from_optimizer_config_for_admission(
        optimizer_config: CudfOptimizerConfig,
        device_ordinals: Vec<usize>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
    ) -> Result<Option<Self>> {
        Self::from_optimizer_config_inner(optimizer_config, device_ordinals, admission_policy, tenant_id, false)
    }

    #[cfg(all(test, feature = "server"))]
    pub(crate) fn from_optimizer_config_for_admission_only(
        optimizer_config: CudfOptimizerConfig,
        device_ordinals: Vec<usize>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
    ) -> Result<Option<Self>> {
        Self::from_optimizer_config_inner(optimizer_config, device_ordinals, admission_policy, tenant_id, true)
    }

    #[cfg(feature = "server")]
    fn from_optimizer_config_inner(
        optimizer_config: CudfOptimizerConfig,
        device_ordinals: Vec<usize>,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
        admission_only: bool,
    ) -> Result<Option<Self>> {
        let Some(resolved) = ResolvedBackendMemoryPolicy::from_execution_config(
            device_ordinals,
            optimizer_config.native_execution_config().clone(),
        )?
        else {
            return Ok(None);
        };
        Self::build_bounded(optimizer_config, resolved, admission_policy, tenant_id, admission_only).map(Some)
    }

    fn build_bounded(
        optimizer_config: CudfOptimizerConfig,
        resolved: ResolvedBackendMemoryPolicy,
        admission_policy: NexusAdmissionPolicy,
        tenant_id: TenantId,
        admission_only: bool,
    ) -> Result<Self> {
        validate_admission_policy(&admission_policy, &resolved.device_ordinals)?;
        let grant_band = admission_policy.resolve_grant_band_bytes(
            resolved.planning_device_budget_bytes,
            resolved.device_ordinals.len(),
            resolved.execution_config.native_pipeline_execution_mode(),
        )?;
        validate_grant_band_slot_guarantee(
            resolved.planning_device_budget_bytes,
            resolved.device_ordinals.len(),
            admission_policy.max_concurrent_queries,
            admission_policy.slot_guarantee,
            grant_band,
            admission_policy.max_queries_per_device,
            admission_policy.placement_policy,
        )?;
        let min_budget = grant_band.min_bytes;
        let mut execution_config =
            resolved.execution_config.clone().with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: min_budget,
                max_source_chunk_bytes: resolved.max_source_chunk_bytes,
                max_row_groups_per_chunk: resolved.max_row_groups_per_chunk,
                max_retry_attempts: resolved.max_retry_attempts,
                min_retry_chunk_row_groups: resolved.min_retry_chunk_row_groups,
            });

        let service_config = backend_query_service_config(
            &execution_config,
            &admission_policy,
            &resolved,
            admission_policy.max_concurrent_queries,
        );
        let tenant_config =
            QueryServiceTenantConfig::with_grant_range(tenant_id.clone(), min_budget, grant_band.want_bytes);
        let device_ordinals = resolved.device_ordinals.clone();
        let query_service = if admission_only {
            QueryService::new_admission_only(service_config, vec![tenant_config], nongds_deployment())
        } else {
            QueryService::new_with_device_ordinals(
                service_config,
                resolved.device_ordinals.clone(),
                vec![tenant_config],
            )
        }
        .map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.query_service",
                "failed to initialize backend query service admission",
                error,
            )
        })?;
        execution_config = execution_config
            .with_query_admission_source(QueryAdmissionSource::new(query_service.clone(), tenant_id.clone()));
        let optimizer_config = optimizer_config.with_native_execution_config(execution_config);
        Ok(NexusGpuBackend { query_service, tenant_id, optimizer_config, device_ordinals })
    }
}

impl Default for NexusGpuBackendBuilder {
    fn default() -> Self {
        Self {
            fallback_policy: GpuFallbackPolicy::default(),
            memory_policy: NexusMemoryPolicy::default(),
            admission_policy: NexusAdmissionPolicy::default(),
            tenant_id: TenantId::new(DEFAULT_BACKEND_TENANT_ID),
        }
    }
}

impl NexusGpuBackendBuilder {
    pub fn fallback_policy(mut self, fallback_policy: GpuFallbackPolicy) -> Self {
        self.fallback_policy = fallback_policy;
        self
    }

    pub fn memory_policy(mut self, memory_policy: NexusMemoryPolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    pub fn admission(mut self, admission_policy: NexusAdmissionPolicy) -> Self {
        self.admission_policy = admission_policy;
        self
    }

    pub fn tenant_id(mut self, tenant_id: impl Into<TenantId>) -> Self {
        self.tenant_id = tenant_id.into();
        self
    }

    pub fn build(self) -> Result<NexusGpuBackend> {
        self.build_inner(false)
    }

    #[cfg(test)]
    pub(crate) fn build_admission_only(self) -> Result<NexusGpuBackend> {
        self.build_inner(true)
    }

    fn build_inner(self, admission_only: bool) -> Result<NexusGpuBackend> {
        let resolved = ResolvedBackendMemoryPolicy::resolve(&self.memory_policy)?;
        let optimizer_config = CudfOptimizerConfig::new(self.fallback_policy)
            .with_native_execution_config(resolved.execution_config.clone());
        NexusGpuBackend::build_bounded(
            optimizer_config,
            resolved,
            self.admission_policy,
            self.tenant_id,
            admission_only,
        )
    }
}

#[derive(Debug)]
struct ResolvedBackendMemoryPolicy {
    device_ordinals: Vec<usize>,
    planning_device_budget_bytes: u64,
    execution_config: NativeCudfExecutionConfig,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
}

impl ResolvedBackendMemoryPolicy {
    #[cfg(feature = "server")]
    fn from_execution_config(
        device_ordinals: Vec<usize>,
        execution_config: NativeCudfExecutionConfig,
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
            planning_device_budget_bytes: device_budget_bytes,
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
                if let Some(fraction) = budget_fraction {
                    validate_budget_fraction(*fraction)?;
                }
                let first_device = device_ordinals[0];
                let budget = budget_fraction.map_or_else(
                    || NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device(first_device),
                    |fraction| {
                        NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device_with_fraction(
                            first_device,
                            fraction,
                        )
                    },
                );
                Ok(Self {
                    device_ordinals: device_ordinals.clone(),
                    planning_device_budget_bytes: budget,
                    execution_config: NativeCudfExecutionConfig::bounded(
                        budget,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                        NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
                    )
                    .with_device_ordinal(first_device),
                    max_source_chunk_bytes: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                    max_row_groups_per_chunk: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                    max_retry_attempts: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                    min_retry_chunk_row_groups: NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
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
                let execution_config = NativeCudfExecutionConfig::bounded(
                    *device_budget_bytes,
                    *max_source_chunk_bytes,
                    *max_row_groups_per_chunk,
                    *max_retry_attempts,
                    *min_retry_chunk_row_groups,
                )
                .with_device_ordinal(device_ordinals[0]);
                Ok(Self {
                    device_ordinals: device_ordinals.clone(),
                    planning_device_budget_bytes: *device_budget_bytes,
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

fn backend_query_service_config(
    execution_config: &NativeCudfExecutionConfig,
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
        source_fetch: execution_config.source_fetch_config().clone(),
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
        resident_graph_cache_max_retained_bytes: None,
        stream_pool_size: Some(backend_stream_pool_size(
            max_concurrent_queries,
            execution_config.source_scheduler_policy().max_source_reads_in_flight(),
        )),
        device_capacity_bytes: Some(resolved.planning_device_budget_bytes),
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

fn admission_queue_len(max_concurrent_queries: usize) -> usize {
    max_concurrent_queries.saturating_mul(DEFAULT_ADMISSION_QUEUE_MULTIPLIER).max(DEFAULT_ADMISSION_QUEUE_MULTIPLIER)
}

fn validate_admission_policy(policy: &NexusAdmissionPolicy, device_ordinals: &[usize]) -> Result<()> {
    validate_positive_usize("max_concurrent_queries", policy.max_concurrent_queries)?;
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

#[cfg(feature = "cugraph")]
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

#[cfg(feature = "cugraph")]
fn is_cugraph_sql_table_function_name(name: &str) -> bool {
    use crate::graph_sql::{
        CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_VALIDATE_CALL_FUNCTION,
        CugraphAlgorithm,
    };

    CugraphAlgorithm::all().any(|algorithm| algorithm.function_name() == name)
        || matches!(
            name,
            CUGRAPH_LIST_ALGORITHMS_FUNCTION | CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION | CUGRAPH_VALIDATE_CALL_FUNCTION
        )
}

fn nongds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[cfg(test)]
mod tests {
    use datafusion::{
        common::config::{ConfigOptions, ExtensionOptions},
        execution::SessionStateBuilder,
        prelude::SessionConfig,
    };

    use super::*;
    use crate::session_config::DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS;

    #[test]
    fn backend_admission_only_lowers_to_query_service_and_planning_floor() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(2))
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().config().max_concurrent_queries, 2);
        assert_eq!(backend.query_service().config().slot_guarantee, SlotGuarantee::GuaranteedMinimum);
        assert_eq!(backend.device_ordinals(), &[0, 1]);
        assert_eq!(backend.query_service().config().device_capacity_bytes, Some(4096));
        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 4096);
        let handle = backend
            .query_service()
            .try_admit_query(backend.tenant_id().clone())
            .unwrap()
            .expect("first query should admit");
        assert_eq!(handle.device_budget_bytes(), 3584);
        assert_eq!(
            backend.optimizer_config().native_execution_config().memory_control_policy(),
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 512,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            }
        );
    }

    #[test]
    fn backend_propagates_query_service_local_parquet_source_cache_capacity() {
        let execution_config = NativeCudfExecutionConfig::bounded(4096, 1024, 1, 3, 1)
            .with_query_service_local_parquet_source_cache_max_retained_bytes(Some(2048));
        let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
            .with_native_execution_config(execution_config.clone());
        let resolved = ResolvedBackendMemoryPolicy {
            device_ordinals: vec![0, 1],
            planning_device_budget_bytes: 4096,
            execution_config,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        };

        let backend = NexusGpuBackend::build_bounded(
            optimizer_config,
            resolved,
            NexusAdmissionPolicy::default().with_max_concurrent_queries(2),
            TenantId::new("tenant"),
            true,
        )
        .unwrap();

        assert_eq!(backend.query_service().config().local_parquet_source_cache_max_retained_bytes, Some(2048));
        assert_eq!(
            backend
                .optimizer_config()
                .native_execution_config()
                .query_service_local_parquet_source_cache_max_retained_bytes(),
            Some(2048)
        );
    }

    #[test]
    fn backend_rejects_zero_query_service_local_parquet_source_cache_capacity() {
        let execution_config = NativeCudfExecutionConfig::bounded(4096, 1024, 1, 3, 1)
            .with_query_service_local_parquet_source_cache_max_retained_bytes(Some(0));
        let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
            .with_native_execution_config(execution_config.clone());
        let resolved = ResolvedBackendMemoryPolicy {
            device_ordinals: vec![0],
            planning_device_budget_bytes: 4096,
            execution_config,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        };

        let error = NexusGpuBackend::build_bounded(
            optimizer_config,
            resolved,
            NexusAdmissionPolicy::default().with_max_concurrent_queries(1),
            TenantId::new("tenant"),
            true,
        )
        .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("field"), Some("local_parquet_source_cache_max_retained_bytes"));
    }

    #[test]
    fn backend_rejects_invalid_admission_policy() {
        let error = NexusGpuBackend::builder()
            .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(0))
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("max_concurrent_queries"));
    }

    #[test]
    fn backend_admission_policy_propagates_placement_and_per_device_cap() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(2)
                    .with_max_queries_per_device(Some(2))
                    .with_placement_policy(QueryPlacementPolicy::BestFit),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().config().max_queries_per_device, Some(2));
        assert_eq!(backend.query_service().config().placement_policy, QueryPlacementPolicy::BestFit);
    }

    #[test]
    fn backend_admission_only_pinned_placement_sets_service_device_ordinal() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(1)
                    .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 1 }),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().config().device_ordinal, 1);
        assert_eq!(
            backend.query_service().config().placement_policy,
            QueryPlacementPolicy::Pinned { device_ordinal: 1 }
        );
    }

    #[test]
    fn backend_rejects_pinned_placement_outside_configured_devices() {
        let error = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(1)
                    .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 1 }),
            )
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("placement_policy"));
        assert_eq!(error.fact_value("pinned_device_ordinal"), Some("1"));
    }

    #[test]
    fn backend_rejects_invalid_grant_band() {
        let error = NexusGpuBackend::builder()
            .admission(
                NexusAdmissionPolicy::default()
                    .with_grant_band(GrantBand { min: MemoryShare::Bytes(2048), want: MemoryShare::Bytes(1024) }),
            )
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("grant.want"));
    }

    #[test]
    fn backend_rejects_policy_default_grant_min() {
        let error = NexusGpuBackend::builder()
            .admission(
                NexusAdmissionPolicy::default()
                    .with_grant_band(GrantBand { min: MemoryShare::PolicyDefault, want: MemoryShare::PolicyDefault }),
            )
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("grant.min"));
    }

    #[test]
    fn backend_rejects_guaranteed_minimum_grant_floor_that_cannot_fill_slots() {
        let error = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 100,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(2)
                    .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) }),
            )
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("grant.min"));
    }

    #[test]
    fn backend_optimistic_slot_guarantee_allows_floor_above_empty_slot_reservation() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 100,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(2)
                    .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) })
                    .with_slot_guarantee(SlotGuarantee::Optimistic),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().config().slot_guarantee, SlotGuarantee::Optimistic);
        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 60);
    }

    #[test]
    fn backend_default_optimistic_strict_want_keeps_per_device_arrival_cushion() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 800,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(3)
                    .with_slot_guarantee(SlotGuarantee::Optimistic),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 600);
    }

    #[test]
    fn backend_default_optimistic_strict_want_uses_effective_slots_per_device() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 800,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(5)
                    .with_slot_guarantee(SlotGuarantee::Optimistic),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 600);
    }

    #[test]
    fn backend_default_optimistic_strict_want_uses_pinned_device_slots() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 800,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(3)
                    .with_slot_guarantee(SlotGuarantee::Optimistic)
                    .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 0 }),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 600);
    }

    #[test]
    fn backend_default_optimistic_strict_want_respects_per_device_cap() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 800,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(5)
                    .with_max_queries_per_device(Some(2))
                    .with_slot_guarantee(SlotGuarantee::Optimistic),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 700);
    }

    #[test]
    fn backend_policy_default_grant_want_round_trips_without_pinning_full_capacity() {
        let policy = NexusAdmissionPolicy::default()
            .with_max_concurrent_queries(5)
            .with_slot_guarantee(SlotGuarantee::Optimistic);
        assert_eq!(policy.grant().want, MemoryShare::PolicyDefault);
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 800,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(policy.max_concurrent_queries())
                    .with_slot_guarantee(policy.slot_guarantee())
                    .with_grant_band(policy.grant()),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 400);
    }

    #[cfg(feature = "server")]
    #[test]
    fn backend_default_optimistic_adaptive_want_uses_full_capacity() {
        let adaptive = CudfOptimizerConfig::default().with_native_execution_config(
            NativeCudfExecutionConfig::bounded(800, 64, 1, 1, 1)
                .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive),
        );
        let adaptive = NexusGpuBackend::from_optimizer_config_for_admission_only(
            adaptive,
            vec![0],
            NexusAdmissionPolicy::default()
                .with_max_concurrent_queries(3)
                .with_slot_guarantee(SlotGuarantee::Optimistic),
            TenantId::new("default"),
        )
        .unwrap()
        .unwrap();

        assert_eq!(adaptive.query_service().tenant_budget(adaptive.tenant_id()).unwrap(), 800);
    }

    #[test]
    fn backend_explicit_grant_want_overrides_optimistic_strict_default() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 800,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(3)
                    .with_slot_guarantee(SlotGuarantee::Optimistic)
                    .with_grant_want(MemoryShare::Bytes(700)),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 700);
    }

    #[test]
    fn backend_guaranteed_minimum_floor_counts_selected_devices() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 200,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(2)
                    .with_grant_band(GrantBand { min: MemoryShare::Bytes(100), want: MemoryShare::Bytes(100) }),
            )
            .build_admission_only()
            .unwrap();

        assert_eq!(backend.device_ordinals(), &[0, 1]);
        assert_eq!(backend.query_service().tenant_budget(backend.tenant_id()).unwrap(), 100);
    }

    #[test]
    fn backend_rejects_guaranteed_minimum_floor_that_only_fits_as_aggregate_bytes() {
        let error = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 100,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(3)
                    .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) }),
            )
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("grant.min"));
    }

    #[test]
    fn backend_rejects_pinned_guaranteed_minimum_floor_that_needs_unpinned_device() {
        let error = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 100,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(2)
                    .with_placement_policy(QueryPlacementPolicy::Pinned { device_ordinal: 0 })
                    .with_grant_band(GrantBand { min: MemoryShare::Bytes(60), want: MemoryShare::Bytes(60) }),
            )
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("grant.min"));
    }

    #[test]
    fn backend_rejects_guaranteed_minimum_floor_that_needs_more_slots_than_per_device_cap() {
        let error = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0, 1],
                device_budget_bytes: 100,
                max_source_chunk_bytes: 64,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .admission(
                NexusAdmissionPolicy::default()
                    .with_max_concurrent_queries(3)
                    .with_max_queries_per_device(Some(1))
                    .with_grant_band(GrantBand { min: MemoryShare::Bytes(50), want: MemoryShare::Bytes(50) }),
            )
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("field"), Some("max_queries_per_device"));
        assert_eq!(error.fact_value("bound_field"), Some("max_concurrent_queries"));
        assert_eq!(error.fact_value("max_queries_per_device"), Some("1"));
        assert_eq!(error.fact_value("placement_eligible_device_count"), Some("2"));
        assert_eq!(error.fact_value("max_concurrent_queries"), Some("3"));
    }

    #[test]
    fn backend_rejects_duplicate_devices() {
        let error = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::BoundedAuto { device_ordinals: vec![0, 0], budget_fraction: None })
            .build_admission_only()
            .unwrap_err();

        assert_eq!(error.fact_value("field"), Some("device_ordinals"));
    }

    #[test]
    fn backend_install_freezes_native_session_keys() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();
        let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
        let extension = state
            .config_options()
            .extensions
            .get::<DataFusionNexusSessionConfig>()
            .expect("backend install must register datafusion-nexus session config");
        assert!(extension.is_backend_admission_frozen());

        for option in DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS {
            let mut extension = extension.clone();
            if option.key.starts_with("request.") {
                extension.set(option.key, "1").unwrap();
                continue;
            }
            let error = extension.set(option.key, "1").unwrap_err();
            assert!(error.to_string().contains(&option.full_key()), "{} should reject", option.full_key());
            assert!(error.to_string().contains("query-service admission"));
        }
    }

    #[test]
    fn backend_install_rejects_conflicting_preexisting_session_extension() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();
        let mut config = SessionConfig::new();
        config
            .options_mut()
            .extensions
            .insert(DataFusionNexusSessionConfig::from_optimizer_config(&CudfOptimizerConfig::default()));

        let error = backend.install_on(SessionStateBuilder::new().with_config(config)).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("reason"), Some("session_config_conflicts_with_backend_admission"));
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn backend_install_rejects_preexisting_cugraph_sql_registration() {
        use crate::{CugraphSqlConfig, session::CudfSessionStateBuilderExt};

        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();

        let builder = SessionStateBuilder::new().with_cugraph_sql(CugraphSqlConfig::default());
        let error = backend.install_on(builder).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidRequest);
        assert_eq!(error.fact_value("reason"), Some("preexisting_cugraph_sql_without_backend_admission"));
        assert!(error.fact_value("table_function").is_some_and(|name| name.starts_with("cugraph_")));
    }

    #[test]
    fn backend_freeze_rejects_direct_session_config_mutation_at_planning_time() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();
        let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
        let mut extension = state
            .config_options()
            .extensions
            .get::<DataFusionNexusSessionConfig>()
            .expect("backend install must register datafusion-nexus session config")
            .clone();
        extension.native_device_budget_bytes += 1;
        let mut options = ConfigOptions::default();
        options.extensions.insert(extension);

        let error = backend.optimizer_config().with_datafusion_config_options(&options).unwrap_err();

        assert!(error.to_string().contains("native.device_budget_bytes"));
        assert!(error.to_string().contains("query-service admission"));
    }

    #[test]
    fn backend_admission_rejects_non_frozen_session_extension() {
        let backend = NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 4096,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .build_admission_only()
            .unwrap();
        let mut options = ConfigOptions::default();
        options.extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(backend.optimizer_config()));

        let error = backend.optimizer_config().with_datafusion_config_options(&options).unwrap_err();

        assert!(error.to_string().contains("session config must be frozen"));
        assert!(error.to_string().contains("query-service admission"));
    }
}
