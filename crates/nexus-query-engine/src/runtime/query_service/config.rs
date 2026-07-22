use super::*;
use crate::graph_memory::GraphMemoryCalibrationOverride;

/// Stable tenant identifier used by service-level admission and future caches.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TenantId(String);

impl TenantId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.0.trim().is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "tenant id must not be empty",
            )
            .with_fact("field", "tenant_id")
            .with_fact("reason", "empty"));
        }
        Ok(())
    }
}

impl From<&str> for TenantId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for TenantId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-tenant device-memory slice used to build bounded query contexts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryServiceTenantConfig {
    /// Tenant admitted by the query service.
    pub tenant_id: TenantId,
    /// Preferred per-query device budget for this tenant's query handles, in bytes.
    pub device_budget_bytes: u64,
    /// Minimum device budget required to admit this tenant's query handles, in bytes.
    pub min_device_budget_bytes: u64,
}

impl QueryServiceTenantConfig {
    /// Creates a tenant config with a fixed bounded device-memory grant.
    pub fn new(tenant_id: impl Into<TenantId>, device_budget_bytes: u64) -> Self {
        Self { tenant_id: tenant_id.into(), device_budget_bytes, min_device_budget_bytes: device_budget_bytes }
    }

    /// Creates a tenant config with separate minimum and preferred grants.
    pub fn with_grant_range(
        tenant_id: impl Into<TenantId>,
        min_device_budget_bytes: u64,
        device_budget_bytes: u64,
    ) -> Self {
        Self { tenant_id: tenant_id.into(), device_budget_bytes, min_device_budget_bytes }
    }

    pub(super) fn validate(&self) -> Result<()> {
        self.tenant_id.validate()?;
        if self.min_device_budget_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "tenant minimum device budget bytes must be greater than zero",
            )
            .with_fact("tenant_id", self.tenant_id.as_str())
            .with_fact("field", "min_device_budget_bytes")
            .with_fact("reason", "zero"));
        }
        if self.device_budget_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "tenant device budget bytes must be greater than zero",
            )
            .with_fact("tenant_id", self.tenant_id.as_str())
            .with_fact("field", "device_budget_bytes")
            .with_fact("reason", "zero"));
        }
        if self.device_budget_bytes < self.min_device_budget_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "tenant preferred device budget must be at least the minimum device budget",
            )
            .with_fact("tenant_id", self.tenant_id.as_str())
            .with_fact("field", "device_budget_bytes")
            .with_fact("bound_field", "min_device_budget_bytes")
            .with_fact("reason", "device_budget_below_minimum"));
        }
        Ok(())
    }
}

/// Policy for reserving future admission-slot minimum grants.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SlotGuarantee {
    /// Keep enough capacity free for every remaining slot to receive the
    /// largest configured tenant minimum. This preserves the historical
    /// fixed-slice safety property.
    #[default]
    GuaranteedMinimum,
    /// Grant from currently available memory without reserving minimum budgets
    /// for empty slots. Later arrivals wait for their minimum if necessary.
    Optimistic,
}

impl SlotGuarantee {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::GuaranteedMinimum => "guaranteed_minimum",
            Self::Optimistic => "optimistic",
        }
    }
}

/// Opt-in policy for binding service contexts to cudf-rs query memory domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryMemoryDomainPolicy {
    /// Domain ceiling as a multiple of the grant budget observed when the
    /// service context is first acquired, capped by the CUDA device capacity.
    pub ceiling_multiplier: u64,
}

impl QueryMemoryDomainPolicy {
    pub const DEFAULT_CEILING_MULTIPLIER: u64 = 2;

    /// Creates the default query-memory-domain policy.
    pub const fn new() -> Self {
        Self { ceiling_multiplier: Self::DEFAULT_CEILING_MULTIPLIER }
    }

    /// Sets the immutable domain ceiling multiplier.
    #[must_use]
    pub const fn with_ceiling_multiplier(mut self, ceiling_multiplier: u64) -> Self {
        self.ceiling_multiplier = ceiling_multiplier;
        self
    }

    pub(super) fn validate(self) -> Result<()> {
        validate_nonzero_config_field("query_memory_domain.ceiling_multiplier", self.ceiling_multiplier == 0)
    }

    pub(crate) fn ceiling_bytes(self, grant_bytes: u64, device_capacity_bytes: u64) -> Result<u64> {
        let multiplied = grant_bytes.saturating_mul(self.ceiling_multiplier);
        let ceiling_bytes = multiplied.min(device_capacity_bytes);
        if ceiling_bytes < grant_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.query_memory_domain",
                "query memory domain ceiling cannot be below the current grant budget",
            )
            .with_fact("grant_bytes", grant_bytes)
            .with_fact("device_capacity_bytes", device_capacity_bytes)
            .with_fact("ceiling_multiplier", self.ceiling_multiplier)
            .with_fact("ceiling_bytes", ceiling_bytes)
            .with_fact("reason", "ceiling_below_grant"));
        }
        Ok(ceiling_bytes)
    }
}

impl Default for QueryMemoryDomainPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Service-level configuration for multi-tenant native query admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryServiceConfig {
    /// CUDA device ordinal used by contexts created through this service.
    pub device_ordinal: usize,
    /// Lakehouse cache-fill policy and shared resource limits.
    pub lakehouse_cache_fill: LakehouseCacheFillConfig,
    /// Source object-store profiles available to service-owned execution contexts.
    pub source_storage_profiles: Vec<SourceStorageProfile>,
    /// Source/build/probe/result boundary policy copied into tenant contexts.
    pub boundary_policy: ExecutionBoundaryPolicy,
    /// Bounded source scheduling policy copied into tenant contexts.
    pub source_scheduler_policy: SourceSchedulerPolicy,
    /// Source chunk planning mode copied into tenant contexts.
    pub source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Strict bounded or adaptive execution mode copied into tenant contexts.
    pub native_pipeline_execution_mode: NativePipelineExecutionMode,
    /// Floating statistical aggregate execution contract copied into tenant
    /// contexts. The field keeps its legacy STDDEV_SAMP name for source
    /// compatibility.
    pub stddev_samp_execution_mode: StddevSampExecutionMode,
    /// Dynamic-filter execution policy assigned to native plans using service handles.
    pub dynamic_filter_execution_options: DynamicFilterExecutionOptions,
    /// Keyed-hash join strategy policy assigned to native plans using service handles.
    pub keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions,
    /// Grouped aggregate strategy execution policy copied into tenant contexts.
    pub aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert local-throughput source chunk planning policy copied into tenant contexts.
    pub source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Optional service-scoped graph calibration metadata copied into tenant contexts.
    pub graph_memory_calibration_override: GraphMemoryCalibrationOverride,
    /// Optional per-device retained-byte cap for service-owned decoded local-Parquet source caches.
    ///
    /// `None` disables the cache. When enabled, each configured CUDA device gets
    /// one cache with runtime-aware entries, so cross-runtime lookups cannot
    /// evict otherwise reusable entries and the cap remains per device. Until
    /// cross-runtime source-frame reuse is modeled, one source can retain a
    /// separate entry per warmed service stream-pool runtime.
    pub local_parquet_source_cache_max_retained_bytes: Option<u64>,
    /// Optional per-device retained-byte cap for service-owned parked cuGraph caches.
    ///
    /// `None` disables resident graph caching. When enabled, entries are scoped
    /// per CUDA device and keyed internally by graph residency key plus service
    /// stream-pool runtime identity, matching the cugraph-rs parked-graph v1
    /// same-runtime resume contract.
    pub resident_graph_cache_max_retained_bytes: Option<u64>,
    /// Optional per-query allocator domain policy for service contexts.
    ///
    /// `None` preserves the legacy grant-backed reservation arena path. When
    /// enabled, strict and adaptive contexts both use hard-limited query-domain
    /// members for their boundary budget tokens.
    pub query_memory_domain: Option<QueryMemoryDomainPolicy>,
    /// Process-lifetime physical allocator installed on each service device.
    ///
    /// The RMM pool form amortizes CUDA allocator calls for cache-backed
    /// serving, but cannot be combined with private query memory domains.
    pub device_memory_manager_base: cudf::DeviceMemoryManagerBase,
    /// Optional service stream pool size. `None` uses the SM-aware default.
    pub stream_pool_size: Option<usize>,
    /// Soft memory-governor capacity for the configured CUDA device.
    ///
    /// `None` preserves fixed-slice callers by sizing capacity from tenant
    /// grants and `max_concurrent_queries`.
    pub device_capacity_bytes: Option<u64>,
    /// Maximum number of admitted query handles active at one time.
    pub max_concurrent_queries: usize,
    /// Optional maximum number of admitted query handles active on one CUDA device.
    pub max_queries_per_device: Option<usize>,
    /// Device placement policy for runtime-backed admission.
    pub placement_policy: QueryPlacementPolicy,
    /// Whether empty admission slots reserve future minimum grants.
    pub slot_guarantee: SlotGuarantee,
    /// Maximum number of queued query admissions across all tenants.
    pub max_admission_queue_len: usize,
    /// Maximum time a query admission may wait before returning `service_overloaded`.
    pub admission_wait_timeout: Duration,
    /// Source chunk byte target copied into per-tenant bounded contexts.
    pub max_source_chunk_bytes: u64,
    /// Source row-group target copied into per-tenant bounded contexts.
    pub max_row_groups_per_chunk: usize,
    /// Bounded retry-attempt cap copied into per-tenant bounded contexts.
    pub max_retry_attempts: usize,
    /// Bounded retry row-group floor copied into per-tenant bounded contexts.
    pub min_retry_chunk_row_groups: usize,
}

impl QueryServiceConfig {
    pub(super) fn validate(&self, deployment: &DeploymentIoCapability) -> Result<()> {
        let _ = deployment;
        self.lakehouse_cache_fill.validate()?;
        self.boundary_policy.validate(false)?;
        self.source_scheduler_policy.validate()?;
        self.dynamic_filter_execution_options.validate()?;
        self.keyed_hash_join_strategy_options.validate()?;
        self.aggregate_strategy_execution_options.validate()?;
        self.source_chunk_planning_options.validate()?;
        if let Some(max_retained_bytes) = self.local_parquet_source_cache_max_retained_bytes {
            validate_nonzero_config_field("local_parquet_source_cache_max_retained_bytes", max_retained_bytes == 0)?;
        }
        if let Some(max_retained_bytes) = self.resident_graph_cache_max_retained_bytes {
            validate_nonzero_config_field("resident_graph_cache_max_retained_bytes", max_retained_bytes == 0)?;
            #[cfg(not(feature = "cugraph"))]
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.config.validate",
                    "resident graph cache requires the cugraph feature",
                )
                .with_fact("field", "resident_graph_cache_max_retained_bytes")
                .with_fact("reason", "cugraph_feature_disabled"));
            }
        }
        if let cudf::DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size } =
            self.device_memory_manager_base
        {
            validate_nonzero_config_field("device_memory_manager_base.initial_pool_size", initial_pool_size == 0)?;
            if maximum_pool_size.is_some_and(|maximum| maximum < initial_pool_size) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.config.validate",
                    "device memory manager pool maximum must be at least its initial size",
                )
                .with_fact("field", "device_memory_manager_base.maximum_pool_size")
                .with_fact("initial_pool_size", initial_pool_size)
                .with_fact("maximum_pool_size", maximum_pool_size.unwrap_or_default())
                .with_fact("reason", "pool_maximum_below_initial"));
            }
        }
        if let Some(policy) = self.query_memory_domain {
            policy.validate()?;
            if matches!(self.device_memory_manager_base, cudf::DeviceMemoryManagerBase::Pool { .. }) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.config.validate",
                    "query memory domains require the CUDA async device memory manager base",
                )
                .with_fact("field", "query_memory_domain")
                .with_fact("conflicting_field", "device_memory_manager_base")
                .with_fact("reason", "query_memory_domain_requires_cuda_async_manager"));
            }
            if self.local_parquet_source_cache_max_retained_bytes.is_some() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.config.validate",
                    "query memory domains cannot be combined with the service source cache until shared-cache allocator routing is modeled",
                )
                .with_fact("field", "query_memory_domain")
                .with_fact("conflicting_field", "local_parquet_source_cache_max_retained_bytes")
                .with_fact("reason", "query_memory_domain_incompatible_with_source_cache"));
            }
            if self.resident_graph_cache_max_retained_bytes.is_some() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.config.validate",
                    "query memory domains cannot be combined with the service graph cache until shared-cache allocator routing is modeled",
                )
                .with_fact("field", "query_memory_domain")
                .with_fact("conflicting_field", "resident_graph_cache_max_retained_bytes")
                .with_fact("reason", "query_memory_domain_incompatible_with_graph_cache"));
            }
        }
        if let Some(stream_pool_size) = self.stream_pool_size
            && stream_pool_size == 0
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.config.validate",
                "service stream pool size must be greater than zero",
            )
            .with_fact("field", "stream_pool_size")
            .with_fact("reason", "zero"));
        }
        validate_nonzero_config_field("max_concurrent_queries", self.max_concurrent_queries == 0)?;
        if let Some(max_queries_per_device) = self.max_queries_per_device {
            validate_nonzero_config_field("max_queries_per_device", max_queries_per_device == 0)?;
        }
        validate_nonzero_config_field("max_admission_queue_len", self.max_admission_queue_len == 0)?;
        if let Some(device_capacity_bytes) = self.device_capacity_bytes {
            validate_nonzero_config_field("device_capacity_bytes", device_capacity_bytes == 0)?;
        }
        validate_nonzero_config_field("max_source_chunk_bytes", self.max_source_chunk_bytes == 0)?;
        validate_nonzero_config_field("max_row_groups_per_chunk", self.max_row_groups_per_chunk == 0)?;
        validate_nonzero_config_field("max_retry_attempts", self.max_retry_attempts == 0)?;
        validate_nonzero_config_field("min_retry_chunk_row_groups", self.min_retry_chunk_row_groups == 0)?;
        if let Some(stream_pool_size) = self.stream_pool_size {
            validate_stream_pool_admission_capacity(
                stream_pool_size,
                self.max_concurrent_queries,
                self.source_scheduler_policy.max_source_reads_in_flight(),
            )?;
        }
        if self.admission_wait_timeout.is_zero() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.config.validate",
                "admission wait timeout must be greater than zero",
            )
            .with_fact("field", "admission_wait_timeout")
            .with_fact("reason", "zero"));
        }
        Ok(())
    }

    pub(super) fn memory_policy_for_tenant(&self, device_budget_bytes: u64) -> MemoryControlPolicy {
        MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes: self.max_source_chunk_bytes,
            max_row_groups_per_chunk: self.max_row_groups_per_chunk,
            max_retry_attempts: self.max_retry_attempts,
            min_retry_chunk_row_groups: self.min_retry_chunk_row_groups,
        }
    }
}

impl Default for QueryServiceConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            lakehouse_cache_fill: LakehouseCacheFillConfig::default(),
            source_storage_profiles: Vec::new(),
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            dynamic_filter_execution_options: DynamicFilterExecutionOptions::default(),
            keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            graph_memory_calibration_override: GraphMemoryCalibrationOverride::default(),
            local_parquet_source_cache_max_retained_bytes: None,
            resident_graph_cache_max_retained_bytes: None,
            query_memory_domain: None,
            device_memory_manager_base: cudf::DeviceMemoryManagerBase::default(),
            stream_pool_size: None,
            device_capacity_bytes: None,
            max_concurrent_queries: 64,
            max_queries_per_device: None,
            placement_policy: QueryPlacementPolicy::default(),
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            max_admission_queue_len: 64,
            admission_wait_timeout: Duration::from_secs(30),
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    }
}

fn validate_nonzero_config_field(field: &'static str, is_zero: bool) -> Result<()> {
    if is_zero {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.config.validate",
            "query service config field must be greater than zero",
        )
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(())
}

pub(super) fn admission_capacity_bounds(
    resources: Option<&QueryServiceResources>,
    fallback_capacity_bytes: u64,
    placement_policy: QueryPlacementPolicy,
) -> (u64, u64) {
    let Some(resources) = resources else {
        return (fallback_capacity_bytes, fallback_capacity_bytes);
    };
    let mut max_device_capacity = 0u64;
    let mut total_capacity = 0u64;
    for context in resources.devices.contexts() {
        if !placement_policy_admits_ordinal(placement_policy, context.device().ordinal()) {
            continue;
        }
        let capacity = context.capacity_bytes();
        max_device_capacity = max_device_capacity.max(capacity);
        total_capacity = total_capacity.saturating_add(capacity);
    }
    (max_device_capacity, total_capacity)
}

pub(super) fn guaranteed_min_slot_capacity(
    resources: Option<&QueryServiceResources>,
    fallback_capacity_bytes: u64,
    min_bytes: u64,
    max_queries_per_device: Option<usize>,
    placement_policy: QueryPlacementPolicy,
) -> usize {
    let Some(resources) = resources else {
        return guaranteed_min_slot_capacity_for_capacities_with_cap(
            [fallback_capacity_bytes],
            min_bytes,
            max_queries_per_device,
        );
    };
    guaranteed_min_slot_capacity_for_capacities_with_cap(
        resources
            .devices
            .contexts()
            .filter(|context| placement_policy_admits_ordinal(placement_policy, context.device().ordinal()))
            .map(|context| context.capacity_bytes()),
        min_bytes,
        max_queries_per_device,
    )
}

#[cfg(test)]
pub(super) fn guaranteed_min_slot_capacity_for_capacities(
    capacities: impl IntoIterator<Item = u64>,
    min_bytes: u64,
) -> usize {
    guaranteed_min_slot_capacity_for_capacities_with_cap(capacities, min_bytes, None)
}

#[cfg(test)]
pub(super) fn guaranteed_min_slot_capacity_for_ordinal_capacities_with_policy(
    capacities: impl IntoIterator<Item = (usize, u64)>,
    min_bytes: u64,
    max_queries_per_device: Option<usize>,
    placement_policy: QueryPlacementPolicy,
) -> usize {
    guaranteed_min_slot_capacity_for_capacities_with_cap(
        capacities.into_iter().filter_map(|(ordinal, capacity)| {
            placement_policy_admits_ordinal(placement_policy, ordinal).then_some(capacity)
        }),
        min_bytes,
        max_queries_per_device,
    )
}

pub(super) fn guaranteed_min_slot_capacity_for_capacities_with_cap(
    capacities: impl IntoIterator<Item = u64>,
    min_bytes: u64,
    max_queries_per_device: Option<usize>,
) -> usize {
    if min_bytes == 0 {
        return usize::MAX;
    }
    capacities
        .into_iter()
        .map(|capacity| {
            let memory_slots = usize::try_from(capacity / min_bytes).unwrap_or(usize::MAX);
            max_queries_per_device.map_or(memory_slots, |cap| memory_slots.min(cap))
        })
        .fold(0usize, usize::saturating_add)
}

pub(super) const fn placement_policy_admits_ordinal(policy: QueryPlacementPolicy, ordinal: usize) -> bool {
    match policy {
        QueryPlacementPolicy::LeastLoaded | QueryPlacementPolicy::BestFit => true,
        QueryPlacementPolicy::Pinned { device_ordinal } => device_ordinal == ordinal,
    }
}

pub(super) fn minimum_stream_pool_size(max_concurrent_queries: usize, max_source_reads_in_flight: usize) -> usize {
    let streams_per_query = 1usize.saturating_add(max_source_reads_in_flight.max(1));
    max_concurrent_queries.saturating_mul(streams_per_query).max(max_concurrent_queries)
}

pub(super) fn validate_stream_pool_admission_capacity(
    stream_pool_size: usize,
    max_concurrent_queries: usize,
    max_source_reads_in_flight: usize,
) -> Result<()> {
    let minimum_stream_pool_size = minimum_stream_pool_size(max_concurrent_queries, max_source_reads_in_flight);
    if stream_pool_size < minimum_stream_pool_size {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.config.validate",
            "service stream pool size must cover root and source-read streams for every admitted query",
        )
        .with_fact("field", "stream_pool_size")
        .with_fact("stream_pool_size", stream_pool_size)
        .with_fact("max_concurrent_queries", max_concurrent_queries)
        .with_fact("max_source_reads_in_flight", max_source_reads_in_flight)
        .with_fact("minimum_stream_pool_size", minimum_stream_pool_size)
        .with_fact("reason", "stream_pool_smaller_than_admission_limit"));
    }
    Ok(())
}

pub(super) fn default_stream_pool_size_for_ordinals(device_ordinals: &[usize]) -> Result<usize> {
    let ordinals = if device_ordinals.is_empty() {
        let device_count = super::NexusGpuInventory.device_count()?;
        (0..device_count).collect::<Vec<_>>()
    } else {
        device_ordinals.to_vec()
    };
    let mut default_size = 0;
    for ordinal in ordinals {
        let device = cudf::DeviceId::new(ordinal).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.query_service.device",
                "failed to resolve query service CUDA device",
                error,
            )
            .with_fact("device_ordinal", ordinal)
        })?;
        default_size = default_size.max(super::default_service_stream_pool_size_for_device(device));
    }
    Ok(default_size)
}
