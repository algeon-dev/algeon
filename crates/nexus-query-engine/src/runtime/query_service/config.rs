use super::*;

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

/// Service-level configuration for multi-tenant native query admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryServiceConfig {
    /// CUDA device ordinal used by contexts created through this service.
    pub device_ordinal: usize,
    /// Source IO policy and shared gate sizes.
    pub source_fetch: SourceFetchConfig,
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
    /// Optional service stream pool size. `None` uses the SM-aware default.
    pub stream_pool_size: Option<usize>,
    /// Soft memory-governor capacity for the configured CUDA device.
    ///
    /// `None` preserves fixed-slice callers by sizing capacity from tenant
    /// grants and `max_concurrent_queries`.
    pub device_capacity_bytes: Option<u64>,
    /// Maximum number of admitted query handles active at one time.
    pub max_concurrent_queries: usize,
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
        self.source_fetch.validate_for_deployment(deployment)?;
        self.boundary_policy.validate(false)?;
        self.source_scheduler_policy.validate()?;
        self.dynamic_filter_execution_options.validate()?;
        self.keyed_hash_join_strategy_options.validate()?;
        self.aggregate_strategy_execution_options.validate()?;
        self.source_chunk_planning_options.validate()?;
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
            source_fetch: SourceFetchConfig::default(),
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
            stream_pool_size: None,
            device_capacity_bytes: None,
            max_concurrent_queries: SourceFetchConfig::DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            max_admission_queue_len: SourceFetchConfig::DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS,
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
) -> (u64, u64) {
    let Some(resources) = resources else {
        return (fallback_capacity_bytes, fallback_capacity_bytes);
    };
    let mut max_device_capacity = 0u64;
    let mut total_capacity = 0u64;
    for context in resources.devices.contexts() {
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
) -> usize {
    let Some(resources) = resources else {
        return guaranteed_min_slot_capacity_for_capacities([fallback_capacity_bytes], min_bytes);
    };
    guaranteed_min_slot_capacity_for_capacities(
        resources.devices.contexts().map(|context| context.capacity_bytes()),
        min_bytes,
    )
}

pub(super) fn guaranteed_min_slot_capacity_for_capacities(
    capacities: impl IntoIterator<Item = u64>,
    min_bytes: u64,
) -> usize {
    if min_bytes == 0 {
        return usize::MAX;
    }
    capacities
        .into_iter()
        .map(|capacity| usize::try_from(capacity / min_bytes).unwrap_or(usize::MAX))
        .fold(0usize, usize::saturating_add)
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
        let device_count = super::CudfInventory.device_count()?;
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
