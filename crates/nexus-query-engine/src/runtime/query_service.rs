//! Multi-tenant service admission harness for native execution.

use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::{sync::Notify, time};

use super::{
    DeploymentIoCapability, ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext, ExecutionContextParts,
    GdsConcurrencyGate, GpuInventory, GrantRequest, HostFetchBudget, LeastLoadedPlacement, MemoryControlPolicy,
    MemoryGovernor, MemoryGovernorReport, MemoryGrant, MemoryResourcePolicy, PlacementPolicy, PlacementRequest,
    Priority, QueryQos, SourceChunkPlanningMode, SourceFetchConcurrencyGate, SourceFetchConfig,
    SourceObjectStoreRegistry, SourceSchedulerPolicy, SourceStorageProfile, default_host_fetch_budget_bytes,
    default_lakehouse_cache_fill_runtime, default_source_fetch_runtime, validate_service_stream_pool_device,
};
use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    error::{QueryError, QueryErrorCode, Result},
    metrics::QueryId,
    mode::{NativePipelineExecutionMode, StddevSampExecutionMode},
};

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

    fn validate(&self) -> Result<()> {
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

    fn validate(&self) -> Result<()> {
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
    fn validate(&self, deployment: &DeploymentIoCapability) -> Result<()> {
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

    fn memory_policy_for_tenant(&self, device_budget_bytes: u64) -> MemoryControlPolicy {
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

fn minimum_stream_pool_size(max_concurrent_queries: usize, max_source_reads_in_flight: usize) -> usize {
    let streams_per_query = 1usize.saturating_add(max_source_reads_in_flight.max(1));
    max_concurrent_queries.saturating_mul(streams_per_query).max(max_concurrent_queries)
}

fn validate_stream_pool_admission_capacity(
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

fn default_stream_pool_size_for_ordinals(device_ordinals: &[usize]) -> Result<usize> {
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

/// Service-level entry point for tenant-aware native query admission.
#[derive(Clone, Debug)]
pub struct QueryService {
    inner: Arc<QueryServiceInner>,
}

/// Point-in-time service admission state for fairness and overload reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryServiceAdmissionSnapshot {
    /// Number of currently admitted query handles.
    pub active_queries: usize,
    /// Number of admissions currently waiting in the service queue.
    pub queued_queries: usize,
    /// Queue depth by configured tenant, in admission rotation order.
    pub queued_by_tenant: Vec<TenantQueueDepth>,
    /// Tenant that will be considered first on the next admission decision.
    pub next_tenant_id: Option<TenantId>,
}

/// Per-tenant queue depth in a [`QueryServiceAdmissionSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantQueueDepth {
    /// Tenant identifier.
    pub tenant_id: TenantId,
    /// Number of queued admission requests for this tenant.
    pub queued_queries: usize,
}

/// Cancel-safe queued admission request.
///
/// Dropping this value before admission removes the queued request from the
/// service queue. This lets async servers wait without parking a blocking
/// thread inside [`QueryService::admit_query`].
#[derive(Debug)]
pub struct QueryAdmissionWaiter {
    service: QueryService,
    tenant_id: TenantId,
    request_id: u64,
    requested_at: Instant,
    deadline: Instant,
    grant_policy: TenantGrantPolicy,
    queued: bool,
}

#[derive(Debug)]
struct QueryServiceInner {
    config: QueryServiceConfig,
    tenant_grants: HashMap<TenantId, TenantGrantPolicy>,
    tenant_order: Vec<TenantId>,
    max_tenant_min_bytes: u64,
    fallback_memory_governor: MemoryGovernor,
    resources: Option<QueryServiceResources>,
    admission: Mutex<AdmissionState>,
    admission_available: Condvar,
    admission_notify: Notify,
    next_query_id: AtomicU64,
    next_admission_sequence: AtomicU64,
}

impl QueryServiceInner {
    fn notify_admission_available(&self) {
        self.admission_available.notify_all();
        self.admission_notify.notify_waiters();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TenantGrantPolicy {
    min_bytes: u64,
    want_bytes: u64,
}

#[derive(Debug, Clone)]
struct QueryServiceResources {
    devices: super::DeviceRegistry,
    host_fetch_budget: HostFetchBudget,
    source_fetch_concurrency: SourceFetchConcurrencyGate,
    lakehouse_cache_fill_budget: HostFetchBudget,
    lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate,
    gds_concurrency: GdsConcurrencyGate,
    source_fetch_runtime: Arc<tokio::runtime::Runtime>,
    lakehouse_cache_fill_runtime: Arc<tokio::runtime::Runtime>,
    source_object_stores: Arc<SourceObjectStoreRegistry>,
    deployment_io_capability: DeploymentIoCapability,
}

#[derive(Debug, Default)]
struct AdmissionState {
    active_queries: usize,
    queued: VecDeque<AdmissionRequest>,
    active_adaptive_grants: HashMap<u64, ActiveAdaptiveGrant>,
    next_tenant_index: usize,
    closed: bool,
}

#[derive(Debug, Clone)]
struct ActiveAdaptiveGrant {
    device_ordinal: usize,
    target_bytes: u64,
    memory_grant: MemoryGrant,
}

#[derive(Debug, Clone)]
struct AdmissionRequest {
    request_id: u64,
    tenant_id: TenantId,
    grant_policy: TenantGrantPolicy,
    qos: QueryQos,
}

#[derive(Debug)]
struct PlacedQuery {
    device_ordinal: usize,
    memory_grant: MemoryGrant,
}

#[derive(Debug)]
struct QueryHandleLifetime {
    service: Arc<QueryServiceInner>,
    tenant_id: TenantId,
    query_id: u64,
    release_requested: AtomicBool,
    released: AtomicBool,
    active_contexts: AtomicUsize,
    root_context: Mutex<Option<ExecutionContext>>,
    memory_grant: Mutex<Option<MemoryGrant>>,
}

#[derive(Debug)]
struct QueryContextLifetimeGuard {
    lifetime: Arc<QueryHandleLifetime>,
}

impl QueryHandleLifetime {
    fn new(
        service: Arc<QueryServiceInner>,
        tenant_id: TenantId,
        query_id: u64,
        memory_grant: MemoryGrant,
    ) -> Arc<Self> {
        Arc::new(Self {
            service,
            tenant_id,
            query_id,
            release_requested: AtomicBool::new(false),
            released: AtomicBool::new(false),
            active_contexts: AtomicUsize::new(0),
            root_context: Mutex::new(None),
            memory_grant: Mutex::new(Some(memory_grant)),
        })
    }

    fn memory_grant(&self) -> Result<MemoryGrant> {
        self.memory_grant
            .lock()
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.context",
                    "query service memory grant mutex was poisoned",
                )
            })?
            .as_ref()
            .cloned()
            .ok_or_else(|| self.released_error())
    }

    fn retain_context(self: &Arc<Self>) -> Result<QueryContextLifetimeGuard> {
        if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
            return Err(self.released_error());
        }
        self.active_contexts.fetch_add(1, Ordering::AcqRel);
        if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
            let guard = QueryContextLifetimeGuard { lifetime: Arc::clone(self) };
            drop(guard);
            return Err(self.released_error());
        }
        Ok(QueryContextLifetimeGuard { lifetime: Arc::clone(self) })
    }

    fn request_release(&self) {
        self.release_requested.store(true, Ordering::Release);
        self.release_if_ready();
    }

    fn release_if_ready(&self) {
        if !self.release_requested.load(Ordering::Acquire) || self.active_contexts.load(Ordering::Acquire) != 0 {
            return;
        }
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        match self.service.admission.lock() {
            Ok(mut state) => {
                state.active_queries = state.active_queries.saturating_sub(1);
                state.active_adaptive_grants.remove(&self.query_id);
                self.service.notify_admission_available();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to release poisoned query service admission state");
            },
        }
        match self.root_context.lock() {
            Ok(mut context) => {
                context.take();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to drop poisoned query service root context");
            },
        }
        match self.memory_grant.lock() {
            Ok(mut memory_grant) => {
                memory_grant.take();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to drop poisoned query service memory grant");
            },
        }
    }

    fn released_error(&self) -> QueryError {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.context",
            "query service handle was released before context acquisition",
        )
        .with_fact("tenant_id", self.tenant_id.as_str())
        .with_fact("query_id", self.query_id)
        .with_fact("reason", "query_handle_released")
    }
}

impl Drop for QueryContextLifetimeGuard {
    fn drop(&mut self) {
        self.lifetime.active_contexts.fetch_sub(1, Ordering::AcqRel);
        self.lifetime.release_if_ready();
    }
}

impl QueryService {
    /// Creates a service entry point and initializes service-owned runtime gates.
    pub fn new(config: QueryServiceConfig, tenants: Vec<QueryServiceTenantConfig>) -> Result<Self> {
        let device_ordinals = vec![config.device_ordinal];
        Self::new_with_device_ordinals(config, device_ordinals, tenants)
    }

    /// Creates a service entry point over an explicit set of CUDA device ordinals.
    pub fn new_with_device_ordinals(
        config: QueryServiceConfig,
        device_ordinals: Vec<usize>,
        tenants: Vec<QueryServiceTenantConfig>,
    ) -> Result<Self> {
        let deployment = DeploymentIoCapability::cached_current_host();
        Self::new_with_device_ordinals_and_deployment(config, device_ordinals, tenants, deployment)
    }

    /// Creates a service entry point with caller-supplied deployment diagnostics.
    pub fn new_with_deployment(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let device_ordinals = vec![config.device_ordinal];
        Self::new_with_device_ordinals_and_deployment(config, device_ordinals, tenants, deployment_io_capability)
    }

    /// Creates a service entry point over explicit devices with caller-supplied deployment diagnostics.
    pub fn new_with_device_ordinals_and_deployment(
        config: QueryServiceConfig,
        device_ordinals: Vec<usize>,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let default_stream_pool_size = default_stream_pool_size_for_ordinals(&device_ordinals)?;
        let minimum_stream_pool_size = minimum_stream_pool_size(
            config.max_concurrent_queries,
            config.source_scheduler_policy.max_source_reads_in_flight(),
        );
        let stream_pool_size =
            config.stream_pool_size.unwrap_or_else(|| default_stream_pool_size.max(minimum_stream_pool_size));
        validate_stream_pool_admission_capacity(
            stream_pool_size,
            config.max_concurrent_queries,
            config.source_scheduler_policy.max_source_reads_in_flight(),
        )?;
        let mut registry_config = super::DeviceRegistryConfig::new(device_ordinals, stream_pool_size);
        if let Some(device_capacity_bytes) = config.device_capacity_bytes {
            registry_config = registry_config.with_device_capacity_bytes(device_capacity_bytes);
        }
        let resources = QueryServiceResources {
            devices: super::DeviceRegistry::install(&super::CudfInventory, &registry_config)?,
            host_fetch_budget: HostFetchBudget::new(
                config.source_fetch.host_fetch_budget_bytes.unwrap_or_else(default_host_fetch_budget_bytes),
            )?,
            source_fetch_concurrency: SourceFetchConcurrencyGate::new(
                config.source_fetch.host_fetch_max_concurrent_ops,
            )?,
            lakehouse_cache_fill_budget: HostFetchBudget::new(
                config.source_fetch.lakehouse_cache_fill_budget_bytes.unwrap_or_else(default_host_fetch_budget_bytes),
            )?,
            lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate::new(
                config.source_fetch.lakehouse_cache_fill_max_concurrent_ops,
            )?,
            gds_concurrency: GdsConcurrencyGate::new(config.source_fetch.gds_max_concurrent_ops)?,
            source_fetch_runtime: default_source_fetch_runtime()?,
            lakehouse_cache_fill_runtime: default_lakehouse_cache_fill_runtime()?,
            source_object_stores: Arc::new(SourceObjectStoreRegistry::with_profiles(
                config.source_storage_profiles.clone(),
            )?),
            deployment_io_capability,
        };
        Self::build(config, tenants, Some(resources))
    }

    /// Creates a service entry point without CUDA/runtime resources.
    ///
    /// This constructor is intended for admission-only embedders and tests that
    /// need the service queue, tenant fairness, and memory-governor ledger
    /// without touching a GPU. Admitted handles from such a service can report
    /// their grants and release admission normally, but
    /// [`QueryHandle::acquire_context`] returns an error because no runtime
    /// resources were installed.
    pub fn new_admission_only(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        Self::build_with_deployment(config, tenants, None, deployment_io_capability)
    }

    #[cfg(test)]
    fn new_for_admission_tests(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        Self::new_admission_only(config, tenants, deployment_io_capability)
    }

    fn build(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        resources: Option<QueryServiceResources>,
    ) -> Result<Self> {
        let deployment =
            resources.as_ref().map(|resources| &resources.deployment_io_capability).cloned().unwrap_or_else(|| {
                DeploymentIoCapability::from_probe_inputs(
                    super::NvidiaFsProbeStatus::DriverMissing,
                    super::KvikioCompatModeStatus::Unset,
                    super::GdsDriverProbeStatus::ProbeUnimplemented,
                )
            });
        Self::build_with_deployment(config, tenants, resources, deployment)
    }

    fn build_with_deployment(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        resources: Option<QueryServiceResources>,
        deployment: DeploymentIoCapability,
    ) -> Result<Self> {
        config.validate(&deployment)?;
        if tenants.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "query service requires at least one tenant",
            )
            .with_fact("field", "tenants")
            .with_fact("reason", "empty"));
        }

        let mut tenant_grants = HashMap::with_capacity(tenants.len());
        let mut tenant_order = Vec::with_capacity(tenants.len());
        let mut max_tenant_want_bytes = 0u64;
        let mut max_tenant_min_bytes = 0u64;
        for tenant in tenants {
            tenant.validate()?;
            max_tenant_want_bytes = max_tenant_want_bytes.max(tenant.device_budget_bytes);
            max_tenant_min_bytes = max_tenant_min_bytes.max(tenant.min_device_budget_bytes);
            if tenant_grants
                .insert(
                    tenant.tenant_id.clone(),
                    TenantGrantPolicy {
                        min_bytes: tenant.min_device_budget_bytes,
                        want_bytes: tenant.device_budget_bytes,
                    },
                )
                .is_some()
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.tenant.validate",
                    "query service tenant ids must be unique",
                )
                .with_fact("tenant_id", tenant.tenant_id.as_str())
                .with_fact("reason", "duplicate"));
            }
            tenant_order.push(tenant.tenant_id);
        }
        let max_concurrent_queries_u64 = u64::try_from(config.max_concurrent_queries).unwrap_or(u64::MAX);
        let compatibility_capacity = max_tenant_want_bytes.saturating_mul(max_concurrent_queries_u64).max(1);
        let device_capacity_bytes = config.device_capacity_bytes.unwrap_or(compatibility_capacity);
        if device_capacity_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.config.validate",
                "query service device capacity bytes must be greater than zero",
            )
            .with_fact("field", "device_capacity_bytes")
            .with_fact("reason", "zero"));
        }
        for (tenant_id, grant) in &tenant_grants {
            if grant.min_bytes > device_capacity_bytes {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.tenant.validate",
                    "tenant minimum device budget exceeds memory governor capacity",
                )
                .with_fact("tenant_id", tenant_id.as_str())
                .with_fact("field", "min_device_budget_bytes")
                .with_fact("reason", "min_device_budget_above_capacity")
                .with_fact("device_capacity_bytes", device_capacity_bytes));
            }
        }
        let fallback_memory_governor = MemoryGovernor::new(config.device_ordinal, device_capacity_bytes)?;

        Ok(Self {
            inner: Arc::new(QueryServiceInner {
                config,
                tenant_grants,
                tenant_order,
                max_tenant_min_bytes,
                fallback_memory_governor,
                resources,
                admission: Mutex::new(AdmissionState::default()),
                admission_available: Condvar::new(),
                admission_notify: Notify::new(),
                next_query_id: AtomicU64::new(1),
                next_admission_sequence: AtomicU64::new(1),
            }),
        })
    }

    /// Admits a query for `tenant_id` under round-robin tenant fairness.
    ///
    /// The call waits only up to [`QueryServiceConfig::admission_wait_timeout`].
    /// Queue overflow and wait timeout return `service_overloaded` instead of
    /// parking the caller indefinitely.
    pub fn admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<QueryHandle> {
        self.admit_query_with_qos(tenant_id, QueryQos::default())
    }

    /// Admits a query with caller-supplied QoS metadata.
    ///
    /// Priority affects queued admission order only; it does not preempt active
    /// CUDA work or change memory-governor accounting.
    pub fn admit_query_with_qos(&self, tenant_id: impl Into<TenantId>, qos: QueryQos) -> Result<QueryHandle> {
        let tenant_id = tenant_id.into();
        let grant_policy = self.tenant_grant_policy(&tenant_id)?;
        let request_id = self.inner.next_query_id.fetch_add(1, Ordering::Relaxed);
        let requested_at = Instant::now();
        let request = AdmissionRequest { request_id, tenant_id: tenant_id.clone(), grant_policy, qos };
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);

        let mut state = self.lock_admission()?;
        if state.closed {
            return Err(service_closed_error(&tenant_id));
        }
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            return Err(service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            ));
        }
        state.queued.push_back(request);

        loop {
            if state.closed {
                remove_queued_request(&mut state, request_id);
                self.inner.notify_admission_available();
                return Err(service_closed_error(&tenant_id));
            }
            if self.request_is_admissible(&state, request_id)?
                && let Some(position) = state.queued.iter().position(|queued| queued.request_id == request_id)
            {
                let Some(placed_query) = self.try_place_query(
                    &tenant_id,
                    request_id,
                    grant_policy,
                    qos,
                    Some(deadline),
                    state.active_queries,
                )?
                else {
                    let now = Instant::now();
                    if now >= deadline {
                        remove_queued_request(&mut state, request_id);
                        self.inner.notify_admission_available();
                        return Err(service_overloaded_error(
                            "memory_grant_wait_timeout",
                            &tenant_id,
                            state.queued.len(),
                            self.inner.config.max_admission_queue_len,
                        ));
                    }
                    let wait = deadline.saturating_duration_since(now);
                    let (next_state, timeout) =
                        self.inner.admission_available.wait_timeout(state, wait).map_err(|_error| {
                            QueryError::new(
                                QueryErrorCode::Internal,
                                "nexus_query_engine.query_service.admission",
                                "query service admission mutex was poisoned while waiting",
                            )
                        })?;
                    state = next_state;
                    if timeout.timed_out() {
                        remove_queued_request(&mut state, request_id);
                        self.inner.notify_admission_available();
                        return Err(service_overloaded_error(
                            "memory_grant_wait_timeout",
                            &tenant_id,
                            state.queued.len(),
                            self.inner.config.max_admission_queue_len,
                        ));
                    }
                    continue;
                };
                let admitted = state.queued.remove(position).expect("admissible request exists");
                self.attach_admission_release_wakeup(&placed_query.memory_grant)?;
                state.active_queries = state.active_queries.saturating_add(1);
                self.register_active_adaptive_grant(
                    &mut state,
                    request_id,
                    placed_query.device_ordinal,
                    grant_policy,
                    &placed_query.memory_grant,
                );
                self.rotate_after_admission(&mut state, &admitted.tenant_id);
                if state.queued.is_empty() {
                    self.grow_active_adaptive_grants_locked(&mut state);
                }
                self.inner.notify_admission_available();
                let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
                let service = Arc::clone(&self.inner);
                let device_budget_bytes = placed_query.memory_grant.budget_bytes();
                let device_ordinal = placed_query.device_ordinal;
                return Ok(QueryHandle {
                    lifetime: QueryHandleLifetime::new(
                        Arc::clone(&service),
                        tenant_id.clone(),
                        request_id,
                        placed_query.memory_grant,
                    ),
                    service,
                    tenant_id,
                    query_id: request_id,
                    admission_sequence,
                    admission_wait: requested_at.elapsed(),
                    device_budget_bytes,
                    device_ordinal,
                });
            }

            let now = Instant::now();
            if now >= deadline {
                remove_queued_request(&mut state, request_id);
                self.inner.notify_admission_available();
                return Err(service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                ));
            }
            let wait = deadline.saturating_duration_since(now);
            let (next_state, timeout) = self.inner.admission_available.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.admission",
                    "query service admission mutex was poisoned while waiting",
                )
            })?;
            state = next_state;
            if timeout.timed_out() {
                remove_queued_request(&mut state, request_id);
                self.inner.notify_admission_available();
                return Err(service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                ));
            }
        }
    }

    /// Enqueues a query admission request without blocking the caller.
    ///
    /// Call [`QueryAdmissionWaiter::try_admit`] until it returns a handle. If
    /// the waiter is dropped first, the queued request is removed.
    pub fn enqueue_admission(&self, tenant_id: impl Into<TenantId>) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_qos(tenant_id, QueryQos::default())
    }

    /// Admits a query asynchronously without parking a blocking thread.
    ///
    /// This is the async counterpart to [`Self::admit_query`]. It uses the
    /// same queue, tenant fairness, timeout, and overload error taxonomy, but
    /// waits on an event-driven wakeup instead of polling
    /// [`QueryAdmissionWaiter::try_admit`].
    pub async fn admit_async(&self, tenant_id: impl Into<TenantId>) -> Result<QueryHandle> {
        self.admit_async_with_qos(tenant_id, QueryQos::default()).await
    }

    /// Admits a query asynchronously with caller-supplied QoS metadata.
    pub async fn admit_async_with_qos(&self, tenant_id: impl Into<TenantId>, qos: QueryQos) -> Result<QueryHandle> {
        let tenant_id = tenant_id.into();
        if let Some(handle) = self.try_admit_query_with_qos(tenant_id.clone(), qos)? {
            return Ok(handle);
        }
        let mut waiter = self.enqueue_admission_with_qos(tenant_id, qos)?;
        loop {
            let notified = self.inner.admission_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(handle) = waiter.try_admit()? {
                return Ok(handle);
            }
            let remaining_wait = waiter.remaining_wait();
            if remaining_wait.is_zero() {
                continue;
            }
            let _ = time::timeout(remaining_wait, &mut notified).await;
        }
    }

    /// Enqueues a query admission request with caller-supplied QoS metadata.
    pub fn enqueue_admission_with_qos(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
    ) -> Result<QueryAdmissionWaiter> {
        let tenant_id = tenant_id.into();
        let grant_policy = self.tenant_grant_policy(&tenant_id)?;
        let request_id = self.inner.next_query_id.fetch_add(1, Ordering::Relaxed);
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let mut state = self.lock_admission()?;
        if state.closed {
            return Err(service_closed_error(&tenant_id));
        }
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            return Err(service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            ));
        }
        state.queued.push_back(AdmissionRequest { request_id, tenant_id: tenant_id.clone(), grant_policy, qos });
        self.inner.notify_admission_available();
        Ok(QueryAdmissionWaiter {
            service: self.clone(),
            tenant_id,
            request_id,
            requested_at,
            deadline,
            grant_policy,
            queued: true,
        })
    }

    /// Attempts to admit a query without queueing or waiting.
    ///
    /// Returns `Ok(None)` when all admission slots are active or when queued
    /// requests already exist and this caller must not bypass them.
    pub fn try_admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_qos(tenant_id, QueryQos::default())
    }

    /// Attempts to admit a query with caller-supplied QoS metadata without
    /// queueing or waiting.
    pub fn try_admit_query_with_qos(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
    ) -> Result<Option<QueryHandle>> {
        let tenant_id = tenant_id.into();
        let grant_policy = self.tenant_grant_policy(&tenant_id)?;
        let mut state = self.lock_admission()?;
        if state.closed {
            return Err(service_closed_error(&tenant_id));
        }
        if state.active_queries >= self.inner.config.max_concurrent_queries || !state.queued.is_empty() {
            return Ok(None);
        }
        let query_id = self.inner.next_query_id.fetch_add(1, Ordering::Relaxed);
        let Some(placed_query) =
            self.try_place_query(&tenant_id, query_id, grant_policy, qos, None, state.active_queries)?
        else {
            return Ok(None);
        };
        self.attach_admission_release_wakeup(&placed_query.memory_grant)?;
        state.active_queries = state.active_queries.saturating_add(1);
        self.register_active_adaptive_grant(
            &mut state,
            query_id,
            placed_query.device_ordinal,
            grant_policy,
            &placed_query.memory_grant,
        );
        self.rotate_after_admission(&mut state, &tenant_id);
        let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
        let service = Arc::clone(&self.inner);
        let device_budget_bytes = placed_query.memory_grant.budget_bytes();
        let device_ordinal = placed_query.device_ordinal;
        Ok(Some(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&service),
                tenant_id.clone(),
                query_id,
                placed_query.memory_grant,
            ),
            service,
            tenant_id,
            query_id,
            admission_sequence,
            admission_wait: Duration::ZERO,
            device_budget_bytes,
            device_ordinal,
        }))
    }

    /// Closes admission and wakes every queued waiter.
    ///
    /// Active query handles keep their slots until dropped or finished.
    pub fn close(&self) -> Result<()> {
        let mut state = self.lock_admission()?;
        state.closed = true;
        self.inner.notify_admission_available();
        Ok(())
    }

    /// Returns the configured preferred device budget for a tenant.
    pub fn tenant_budget(&self, tenant_id: &TenantId) -> Result<u64> {
        self.tenant_grant_policy(tenant_id).map(|grant| grant.want_bytes)
    }

    fn tenant_grant_policy(&self, tenant_id: &TenantId) -> Result<TenantGrantPolicy> {
        self.inner.tenant_grants.get(tenant_id).copied().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.admission",
                "unknown tenant id for query service admission",
            )
            .with_fact("tenant_id", tenant_id.as_str())
        })
    }

    fn remaining_slots_after_current(&self, active_queries: usize) -> usize {
        self.inner.config.max_concurrent_queries.saturating_sub(active_queries.saturating_add(1))
    }

    fn reserved_min_bytes_for_remaining_slots(&self, active_queries: usize) -> u64 {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return 0;
        }
        self.inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(self.remaining_slots_after_current(active_queries)).unwrap_or(u64::MAX))
    }

    fn reserved_min_bytes_for_empty_slots(&self, active_queries: usize) -> u64 {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return 0;
        }
        let empty_slots = self.inner.config.max_concurrent_queries.saturating_sub(active_queries);
        self.inner.max_tenant_min_bytes.saturating_mul(u64::try_from(empty_slots).unwrap_or(u64::MAX))
    }

    fn reserved_min_bytes_for_selected_device(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
        active_queries: usize,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let remaining_slots = self.remaining_slots_after_current(active_queries);
        if remaining_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            let slots = context.available_bytes()? / self.inner.max_tenant_min_bytes;
            other_device_min_slots =
                other_device_min_slots.saturating_add(usize::try_from(slots).unwrap_or(usize::MAX));
            if other_device_min_slots >= remaining_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = remaining_slots.saturating_sub(other_device_min_slots);
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    fn reserved_min_bytes_for_selected_device_empty_slots(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
        active_queries: usize,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let empty_slots = self.inner.config.max_concurrent_queries.saturating_sub(active_queries);
        if empty_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            let slots = context.available_bytes()? / self.inner.max_tenant_min_bytes;
            other_device_min_slots =
                other_device_min_slots.saturating_add(usize::try_from(slots).unwrap_or(usize::MAX));
            if other_device_min_slots >= empty_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = empty_slots.saturating_sub(other_device_min_slots);
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    fn grant_want_bytes(
        &self,
        available_bytes: u64,
        grant_policy: TenantGrantPolicy,
        reserved_for_remaining_slots: u64,
    ) -> u64 {
        if available_bytes <= reserved_for_remaining_slots {
            grant_policy.min_bytes
        } else {
            grant_policy.want_bytes.min(available_bytes - reserved_for_remaining_slots).max(grant_policy.min_bytes)
        }
    }

    fn try_grant_from_governor(
        &self,
        governor: &MemoryGovernor,
        query_id: u64,
        grant_policy: TenantGrantPolicy,
        qos: QueryQos,
        deadline: Option<Instant>,
        active_queries: usize,
    ) -> Result<Option<MemoryGrant>> {
        let available_bytes = governor.available_bytes()?;
        if available_bytes < grant_policy.min_bytes {
            return Ok(None);
        }
        let want_bytes = self.grant_want_bytes(
            available_bytes,
            grant_policy,
            self.reserved_min_bytes_for_remaining_slots(active_queries),
        );
        let mut request = GrantRequest::new(query_id, grant_policy.min_bytes, want_bytes);
        request = request.with_qos(qos);
        if let Some(deadline) = deadline {
            request = request.with_deadline(deadline);
        }
        governor.try_request(request)
    }

    fn try_place_query(
        &self,
        tenant_id: &TenantId,
        query_id: u64,
        grant_policy: TenantGrantPolicy,
        qos: QueryQos,
        deadline: Option<Instant>,
        active_queries: usize,
    ) -> Result<Option<PlacedQuery>> {
        let Some(resources) = &self.inner.resources else {
            let Some(memory_grant) = self.try_grant_from_governor(
                &self.inner.fallback_memory_governor,
                query_id,
                grant_policy,
                qos,
                deadline,
                active_queries,
            )?
            else {
                return Ok(None);
            };
            return Ok(Some(PlacedQuery { device_ordinal: self.inner.config.device_ordinal, memory_grant }));
        };

        let placement_request = PlacementRequest::with_grant_range(grant_policy.min_bytes, grant_policy.want_bytes)
            .with_tenant_id(tenant_id.as_str());
        let mut any_device_can_fit_minimum = false;
        for device in LeastLoadedPlacement.candidates(&resources.devices, &placement_request)? {
            let context = resources.devices.context(device)?;
            if context.capacity_bytes() >= grant_policy.min_bytes {
                any_device_can_fit_minimum = true;
            }
            let available_bytes = context.available_bytes()?;
            if available_bytes < grant_policy.min_bytes {
                continue;
            }
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, device, active_queries)?;
            let want_bytes = self.grant_want_bytes(available_bytes, grant_policy, reserved_for_remaining_slots);
            let mut request = GrantRequest::new(query_id, grant_policy.min_bytes, want_bytes);
            request = request.with_qos(qos);
            if let Some(deadline) = deadline {
                request = request.with_deadline(deadline);
            }
            if let Some(memory_grant) = context.try_grant_query(request)? {
                return Ok(Some(PlacedQuery { device_ordinal: device.ordinal(), memory_grant }));
            }
        }
        if any_device_can_fit_minimum {
            return Ok(None);
        }
        Err(QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.placement",
            "no configured CUDA device has enough capacity for the query minimum budget",
        )
        .with_fact("tenant_id", tenant_id.as_str())
        .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
        .with_fact("reason", "query_min_budget_exceeds_device_capacity"))
    }

    fn attach_admission_release_wakeup(&self, grant: &MemoryGrant) -> Result<()> {
        let service = self.clone();
        grant.notify_on_release(Arc::new(move || {
            let mut guard = match service.inner.admission.lock() {
                Ok(guard) => guard,
                Err(_error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to lock query service admission for grant release wakeup"
                    );
                    return;
                },
            };
            if guard.queued.is_empty() {
                service.grow_active_adaptive_grants_locked(&mut guard);
            }
            service.inner.notify_admission_available();
        }))
    }

    fn register_active_adaptive_grant(
        &self,
        state: &mut AdmissionState,
        query_id: u64,
        device_ordinal: usize,
        grant_policy: TenantGrantPolicy,
        memory_grant: &MemoryGrant,
    ) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        if memory_grant.budget_bytes() >= grant_policy.want_bytes {
            return;
        }
        state.active_adaptive_grants.insert(
            query_id,
            ActiveAdaptiveGrant {
                device_ordinal,
                target_bytes: grant_policy.want_bytes,
                memory_grant: memory_grant.clone(),
            },
        );
    }

    fn adaptive_growth_available_bytes(&self, state: &AdmissionState, device_ordinal: usize) -> Result<u64> {
        let Some(resources) = &self.inner.resources else {
            let available = self.inner.fallback_memory_governor.available_bytes()?;
            return Ok(available.saturating_sub(self.reserved_min_bytes_for_empty_slots(state.active_queries)));
        };
        let context = resources.devices.context_by_ordinal(device_ordinal)?;
        let available = context.available_bytes()?;
        let reserved =
            self.reserved_min_bytes_for_selected_device_empty_slots(resources, context.device(), state.active_queries)?;
        Ok(available.saturating_sub(reserved))
    }

    fn grow_active_adaptive_grants_locked(&self, state: &mut AdmissionState) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        let query_ids = state.active_adaptive_grants.keys().copied().collect::<Vec<_>>();
        for query_id in query_ids {
            let Some(active) = state.active_adaptive_grants.get(&query_id).cloned() else {
                continue;
            };
            let current_bytes = active.memory_grant.budget_bytes();
            if current_bytes >= active.target_bytes {
                state.active_adaptive_grants.remove(&query_id);
                continue;
            }
            let available_bytes = match self.adaptive_growth_available_bytes(state, active.device_ordinal) {
                Ok(bytes) => bytes,
                Err(error) => {
                    eprintln!("[nexus-query-engine] warning: failed to compute adaptive grant growth: {error}");
                    continue;
                },
            };
            if available_bytes == 0 {
                continue;
            }
            let grow_bytes = active.target_bytes.saturating_sub(current_bytes).min(available_bytes);
            match active.memory_grant.try_grow(grow_bytes) {
                Ok(true) => {
                    if active.memory_grant.budget_bytes() >= active.target_bytes {
                        state.active_adaptive_grants.remove(&query_id);
                    }
                },
                Ok(false) => continue,
                Err(error) => {
                    eprintln!("[nexus-query-engine] warning: failed to grow adaptive grant: {error}");
                    continue;
                },
            }
        }
    }

    /// Returns the service memory-governor report.
    pub fn memory_governor_report(&self) -> Result<MemoryGovernorReport> {
        if let Some(resources) = &self.inner.resources {
            return resources.devices.context_by_ordinal(self.inner.config.device_ordinal)?.memory_governor().report();
        }
        self.inner.fallback_memory_governor.report()
    }

    /// Returns one memory-governor report for each installed service device.
    pub fn memory_governor_reports(&self) -> Result<Vec<MemoryGovernorReport>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(vec![self.inner.fallback_memory_governor.report()?]);
        };
        resources.devices.contexts().map(|context| context.memory_governor().report()).collect()
    }

    /// Returns the service configuration.
    pub fn config(&self) -> &QueryServiceConfig {
        &self.inner.config
    }

    /// Returns tenant ids in their admission rotation order.
    pub fn tenant_order(&self) -> &[TenantId] {
        &self.inner.tenant_order
    }

    /// Returns a point-in-time admission snapshot for service reports.
    pub fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        let state = self.lock_admission()?;
        let queued_by_tenant = self
            .inner
            .tenant_order
            .iter()
            .map(|tenant_id| TenantQueueDepth {
                tenant_id: tenant_id.clone(),
                queued_queries: state.queued.iter().filter(|request| &request.tenant_id == tenant_id).count(),
            })
            .collect();
        Ok(QueryServiceAdmissionSnapshot {
            active_queries: state.active_queries,
            queued_queries: state.queued.len(),
            queued_by_tenant,
            next_tenant_id: self.inner.tenant_order.get(state.next_tenant_index).cloned(),
        })
    }

    fn lock_admission(&self) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        self.inner.admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission",
                "query service admission mutex was poisoned",
            )
        })
    }

    fn request_is_admissible(&self, state: &AdmissionState, request_id: u64) -> Result<bool> {
        if state.active_queries >= self.inner.config.max_concurrent_queries {
            return Ok(false);
        }
        let Some(position) = self.select_next_request_position(state)? else {
            return Ok(false);
        };
        Ok(state.queued[position].request_id == request_id)
    }

    fn select_next_request_position(&self, state: &AdmissionState) -> Result<Option<usize>> {
        if state.queued.is_empty() {
            return Ok(None);
        }

        let mut first_candidate = None;
        for priority in Priority::DESCENDING {
            for offset in 0..self.inner.tenant_order.len() {
                let tenant_index = (state.next_tenant_index + offset) % self.inner.tenant_order.len();
                let tenant_id = &self.inner.tenant_order[tenant_index];
                for position in state.queued.iter().enumerate().filter_map(|(position, request)| {
                    (&request.tenant_id == tenant_id && request.qos.priority() == priority).then_some(position)
                }) {
                    first_candidate.get_or_insert(position);
                    if self.request_can_receive_minimum_grant(&state.queued[position])? {
                        return Ok(Some(position));
                    }
                }
            }
        }

        Ok(first_candidate.or(Some(0)))
    }

    fn request_can_receive_minimum_grant(&self, request: &AdmissionRequest) -> Result<bool> {
        let Some(resources) = &self.inner.resources else {
            return Ok(self.inner.fallback_memory_governor.available_bytes()? >= request.grant_policy.min_bytes);
        };
        for context in resources.devices.contexts() {
            if context.capacity_bytes() >= request.grant_policy.min_bytes
                && context.available_bytes()? >= request.grant_policy.min_bytes
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn rotate_after_admission(&self, state: &mut AdmissionState, tenant_id: &TenantId) {
        if let Some(index) = self.inner.tenant_order.iter().position(|candidate| candidate == tenant_id) {
            state.next_tenant_index = (index + 1) % self.inner.tenant_order.len();
        }
    }
}

impl QueryAdmissionWaiter {
    /// Attempts to complete this queued admission without blocking.
    ///
    /// Returns `Ok(None)` while the request is still queued behind active work
    /// or tenant-fairness rotation. Queue timeout and service close remove the
    /// request and return a structured error.
    pub fn try_admit(&mut self) -> Result<Option<QueryHandle>> {
        if !self.queued {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.admission",
                "query admission waiter has already completed",
            )
            .with_fact("tenant_id", self.tenant_id.as_str())
            .with_fact("query_id", self.request_id)
            .with_fact("reason", "admission_waiter_completed"));
        }

        let mut state = self.service.lock_admission()?;
        if state.closed {
            remove_queued_request(&mut state, self.request_id);
            self.queued = false;
            self.service.inner.notify_admission_available();
            return Err(service_closed_error(&self.tenant_id));
        }
        if self.service.request_is_admissible(&state, self.request_id)?
            && let Some(position) = state.queued.iter().position(|queued| queued.request_id == self.request_id)
        {
            let Some(placed_query) = self.service.try_place_query(
                &self.tenant_id,
                self.request_id,
                self.grant_policy,
                state.queued[position].qos,
                Some(self.deadline),
                state.active_queries,
            )?
            else {
                if Instant::now() >= self.deadline {
                    remove_queued_request(&mut state, self.request_id);
                    self.queued = false;
                    self.service.inner.notify_admission_available();
                    return Err(service_overloaded_error(
                        "memory_grant_wait_timeout",
                        &self.tenant_id,
                        state.queued.len(),
                        self.service.inner.config.max_admission_queue_len,
                    ));
                }
                return Ok(None);
            };
            let admitted = state.queued.remove(position).expect("admissible request exists");
            self.service.attach_admission_release_wakeup(&placed_query.memory_grant)?;
            state.active_queries = state.active_queries.saturating_add(1);
            self.service.register_active_adaptive_grant(
                &mut state,
                self.request_id,
                placed_query.device_ordinal,
                self.grant_policy,
                &placed_query.memory_grant,
            );
            self.service.rotate_after_admission(&mut state, &admitted.tenant_id);
            self.queued = false;
            if state.queued.is_empty() {
                self.service.grow_active_adaptive_grants_locked(&mut state);
            }
            self.service.inner.notify_admission_available();
            let admission_sequence = self.service.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
            let service = Arc::clone(&self.service.inner);
            let device_budget_bytes = placed_query.memory_grant.budget_bytes();
            let device_ordinal = placed_query.device_ordinal;
            return Ok(Some(QueryHandle {
                lifetime: QueryHandleLifetime::new(
                    Arc::clone(&service),
                    self.tenant_id.clone(),
                    self.request_id,
                    placed_query.memory_grant,
                ),
                service,
                tenant_id: self.tenant_id.clone(),
                query_id: self.request_id,
                admission_sequence,
                admission_wait: self.requested_at.elapsed(),
                device_budget_bytes,
                device_ordinal,
            }));
        }

        if Instant::now() >= self.deadline {
            remove_queued_request(&mut state, self.request_id);
            self.queued = false;
            self.service.inner.notify_admission_available();
            return Err(service_overloaded_error(
                "admission_wait_timeout",
                &self.tenant_id,
                state.queued.len(),
                self.service.inner.config.max_admission_queue_len,
            ));
        }

        Ok(None)
    }

    /// Remaining time before this queued admission reaches the service timeout.
    pub fn remaining_wait(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

impl Drop for QueryAdmissionWaiter {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        match self.service.inner.admission.lock() {
            Ok(mut state) => {
                remove_queued_request(&mut state, self.request_id);
                self.service.inner.notify_admission_available();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to remove queued query service admission request");
            },
        }
    }
}

fn remove_queued_request(state: &mut AdmissionState, request_id: u64) {
    if let Some(position) = state.queued.iter().position(|request| request.request_id == request_id) {
        state.queued.remove(position);
    }
}

fn service_overloaded_error(
    reason: &'static str,
    tenant_id: &TenantId,
    queued_queries: usize,
    max_admission_queue_len: usize,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "nexus_query_engine.query_service.admission",
        "query service admission is overloaded",
    )
    .with_fact("reason", reason)
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact("queued_queries", queued_queries)
    .with_fact("max_admission_queue_len", max_admission_queue_len)
}

fn service_closed_error(tenant_id: &TenantId) -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        "nexus_query_engine.query_service.admission",
        "query service admission is closed",
    )
    .with_fact("reason", "admission_closed")
    .with_fact("tenant_id", tenant_id.as_str())
}

/// RAII admission handle for one tenant query.
#[derive(Debug)]
pub struct QueryHandle {
    service: Arc<QueryServiceInner>,
    tenant_id: TenantId,
    query_id: u64,
    admission_sequence: u64,
    admission_wait: Duration,
    device_budget_bytes: u64,
    device_ordinal: usize,
    lifetime: Arc<QueryHandleLifetime>,
}

impl QueryHandle {
    /// Stable service-local query id assigned at admission time.
    pub fn query_id(&self) -> u64 {
        self.query_id
    }

    /// Monotonic service-local admission sequence assigned when this handle was admitted.
    pub fn admission_sequence(&self) -> u64 {
        self.admission_sequence
    }

    /// Time spent waiting in the service admission queue.
    pub fn admission_wait(&self) -> Duration {
        self.admission_wait
    }

    /// Tenant that owns this admitted query.
    pub fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    /// Device-memory budget slice assigned to this query.
    pub fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }

    /// CUDA device ordinal selected for this query.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Bounded memory policy this handle will use for native execution.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.service.config.memory_policy_for_tenant(self.device_budget_bytes)
    }

    /// Boundary policy copied into the tenant execution context.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.service.config.boundary_policy
    }

    /// Source scheduler policy copied into the tenant execution context.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.service.config.source_scheduler_policy
    }

    /// Source chunk planning mode copied into the tenant execution context.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.service.config.source_chunk_planning_mode
    }

    /// Source IO resource policy assigned to this admitted query.
    pub fn source_fetch_config(&self) -> SourceFetchConfig {
        self.service.config.source_fetch.clone()
    }

    /// Native bounded-pipeline mode copied into the tenant execution context.
    pub fn native_pipeline_execution_mode(&self) -> NativePipelineExecutionMode {
        self.service.config.native_pipeline_execution_mode
    }

    /// Floating statistical aggregate execution contract assigned to this admitted query.
    pub fn statistical_aggregate_execution_mode(&self) -> StddevSampExecutionMode {
        self.service.config.stddev_samp_execution_mode
    }

    /// Floating statistical aggregate contract returned through the legacy
    /// STDDEV_SAMP accessor.
    pub fn stddev_samp_execution_mode(&self) -> StddevSampExecutionMode {
        self.statistical_aggregate_execution_mode()
    }

    /// Dynamic-filter execution policy assigned to this admitted query.
    pub fn dynamic_filter_execution_options(&self) -> DynamicFilterExecutionOptions {
        self.service.config.dynamic_filter_execution_options
    }

    /// Keyed-hash join strategy policy assigned to this admitted query.
    pub fn keyed_hash_join_strategy_options(&self) -> KeyedHashJoinStrategyOptions {
        self.service.config.keyed_hash_join_strategy_options
    }

    /// Grouped aggregate strategy execution policy assigned to this admitted query.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.service.config.aggregate_strategy_execution_options
    }

    /// Expert source chunk planning policy assigned to this admitted query.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.service.config.source_chunk_planning_options
    }

    /// Acquires a tenant-bound execution context from the service stream pool.
    pub fn acquire_context(&self) -> Result<TenantQueryContext> {
        self.acquire_context_with_cancellation(ExecutionCancellation::new())
    }

    /// Acquires a tenant-bound execution context that observes caller-supplied
    /// cooperative cancellation.
    pub fn acquire_context_with_cancellation(&self, cancellation: ExecutionCancellation) -> Result<TenantQueryContext> {
        let lifetime_guard = self.lifetime.retain_context()?;
        let Some(resources) = &self.service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.context",
                "query service was constructed without runtime resources",
            ));
        };
        let device = resources.devices.context_by_ordinal(self.device_ordinal)?;
        let mut root_context = self.lifetime.root_context.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.context",
                "query service root context mutex was poisoned",
            )
        })?;
        if let Some(context) = root_context.as_ref() {
            return Ok(TenantQueryContext { context: context.clone(), _lifetime_guard: lifetime_guard });
        }
        let stream_lease = device
            .stream_pool()
            .acquire_with_cancellation(&cancellation, "nexus_query_engine.query_service.context.stream_acquire")?;
        validate_service_stream_pool_device(device.stream_pool(), stream_lease.runtime().device())?;
        let stream_lease_guard = Arc::new(stream_lease);
        let memory_grant = self.lifetime.memory_grant()?;
        let context = ExecutionContext::from_parts(ExecutionContextParts {
            query_id: Some(QueryId::new(format!("service-{}", self.query_id))),
            trace_id: None,
            runtime: stream_lease_guard.shared_runtime(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy(),
            memory_grant: Some(memory_grant),
            boundary_policy: self.service.config.boundary_policy,
            source_scheduler_policy: self.service.config.source_scheduler_policy,
            source_chunk_planning_mode: self.service.config.source_chunk_planning_mode,
            source_fetch_config: self.service.config.source_fetch.clone(),
            service_stream_pool: device.stream_pool().clone(),
            host_fetch_budget: resources.host_fetch_budget.clone(),
            source_fetch_concurrency: resources.source_fetch_concurrency.clone(),
            lakehouse_cache_fill_budget: resources.lakehouse_cache_fill_budget.clone(),
            lakehouse_cache_fill_concurrency: resources.lakehouse_cache_fill_concurrency.clone(),
            lakehouse_cache_fill_events: super::LakehouseCacheFillEventSink::default(),
            gds_concurrency: resources.gds_concurrency.clone(),
            source_fetch_runtime: Arc::clone(&resources.source_fetch_runtime),
            lakehouse_cache_fill_runtime: Arc::clone(&resources.lakehouse_cache_fill_runtime),
            source_object_stores: Arc::clone(&resources.source_object_stores),
            deployment_io_capability: resources.deployment_io_capability.clone(),
            cancellation,
            native_pipeline_execution_mode: self.service.config.native_pipeline_execution_mode,
            stddev_samp_execution_mode: self.service.config.stddev_samp_execution_mode,
            aggregate_strategy_execution_options: self.service.config.aggregate_strategy_execution_options,
            source_chunk_planning_options: self.service.config.source_chunk_planning_options,
            adaptive_memory_history: Some(Arc::new(AdaptiveMemoryHistory::default())),
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
            stream_lease_guard: Some(stream_lease_guard),
            query_service_admitted: true,
        });
        *root_context = Some(context.clone());
        Ok(TenantQueryContext { context, _lifetime_guard: lifetime_guard })
    }

    /// Releases the admission slot before dropping the handle.
    pub fn finish(&self) {
        self.lifetime.request_release();
    }

    fn release_once(&self) {
        self.lifetime.request_release();
    }
}

impl Drop for QueryHandle {
    fn drop(&mut self) {
        self.release_once();
    }
}

/// Execution context borrowed from a [`QueryHandle`].
#[derive(Debug)]
pub struct TenantQueryContext {
    context: ExecutionContext,
    _lifetime_guard: QueryContextLifetimeGuard,
}

impl TenantQueryContext {
    /// Native execution context for this tenant query.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus};

    fn non_gds_deployment() -> DeploymentIoCapability {
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        )
    }

    fn test_service(max_concurrent_queries: usize, max_admission_queue_len: usize) -> QueryService {
        let config = QueryServiceConfig {
            max_concurrent_queries,
            max_admission_queue_len,
            admission_wait_timeout: Duration::from_millis(500),
            ..QueryServiceConfig::default()
        };
        QueryService::new_for_admission_tests(
            config,
            vec![
                QueryServiceTenantConfig::new("large", 8 * 1024 * 1024),
                QueryServiceTenantConfig::new("small", 1024 * 1024),
            ],
            non_gds_deployment(),
        )
        .unwrap()
    }

    fn runtime_test_service(max_concurrent_queries: usize, max_admission_queue_len: usize) -> QueryService {
        let config = QueryServiceConfig {
            max_concurrent_queries,
            max_admission_queue_len,
            admission_wait_timeout: Duration::from_millis(500),
            ..QueryServiceConfig::default()
        };
        QueryService::new_with_deployment(
            config,
            vec![
                QueryServiceTenantConfig::new("large", 8 * 1024 * 1024),
                QueryServiceTenantConfig::new("small", 1024 * 1024),
            ],
            non_gds_deployment(),
        )
        .unwrap()
    }

    fn wait_for_queued_len(service: &QueryService, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let state = service.inner.admission.lock().unwrap();
            if state.queued.len() >= expected {
                return;
            }
            drop(state);
            assert!(Instant::now() < deadline, "timed out waiting for queued admissions");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn wait_for_exact_queued_len(service: &QueryService, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let queued = service.admission_snapshot().unwrap().queued_queries;
            if queued == expected {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {expected} queued admissions; last queued={queued}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn query_service_assigns_fixed_per_tenant_bounded_budget() {
        let service = test_service(2, 8);

        let large = service.admit_query("large").unwrap();
        let small = service.admit_query("small").unwrap();

        assert_eq!(large.device_budget_bytes(), 8 * 1024 * 1024);
        assert_eq!(small.device_budget_bytes(), 1024 * 1024);
        assert_eq!(large.device_ordinal(), 0);
        assert_eq!(small.device_ordinal(), 0);
        assert_eq!(
            small.memory_control_policy(),
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 1024 * 1024,
                max_source_chunk_bytes: 64 * 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            }
        );
        assert_eq!(large.admission_sequence(), 1);
        assert_eq!(small.admission_sequence(), 2);
        assert!(large.admission_wait() <= Duration::from_secs(1));
        assert!(small.admission_wait() <= Duration::from_secs(1));
    }

    #[test]
    fn query_service_admission_test_config_can_record_non_default_device_without_runtime_resources() {
        let config = QueryServiceConfig { device_ordinal: 2, ..QueryServiceConfig::default() };
        let service = QueryService::new_admission_only(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap();

        let handle = service.admit_query("tenant").unwrap();

        assert_eq!(handle.device_ordinal(), 2);
        let error = handle.acquire_context().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    }

    #[test]
    fn query_service_reports_overload_when_queue_is_full() {
        let service = test_service(1, 1);
        let active = service.admit_query("large").unwrap();
        let queued_service = service.clone();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let (finish_sender, finish_receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            let handle = queued_service.admit_query("large").unwrap();
            finish_receiver.recv().unwrap();
            drop(handle);
        });
        started_receiver.recv().unwrap();
        wait_for_queued_len(&service, 1);

        let error = service.admit_query("small").unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.kind(), crate::error::QueryErrorKind::ResourceExhausted);
        assert_eq!(error.status(), crate::error::QueryErrorStatus::Temporary);
        assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));
        assert_eq!(error.fact_value("tenant_id"), Some("small"));

        drop(active);
        finish_sender.send(()).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn query_service_reports_admission_snapshot_by_tenant_order() {
        let service = test_service(1, 8);
        let active = service.admit_query("large").unwrap();
        let queued_service = service.clone();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            let _handle = queued_service.admit_query("small").unwrap();
        });
        started_receiver.recv().unwrap();
        wait_for_queued_len(&service, 1);

        let snapshot = service.admission_snapshot().unwrap();
        assert_eq!(snapshot.active_queries, 1);
        assert_eq!(snapshot.queued_queries, 1);
        assert_eq!(snapshot.next_tenant_id.as_ref().map(TenantId::as_str), Some("small"));
        assert_eq!(snapshot.queued_by_tenant.len(), 2);
        assert_eq!(snapshot.queued_by_tenant[0].tenant_id.as_str(), "large");
        assert_eq!(snapshot.queued_by_tenant[0].queued_queries, 0);
        assert_eq!(snapshot.queued_by_tenant[1].tenant_id.as_str(), "small");
        assert_eq!(snapshot.queued_by_tenant[1].queued_queries, 1);

        drop(active);
        waiter.join().unwrap();
    }

    #[test]
    fn query_service_handle_exposes_configured_execution_policies() {
        let dynamic_filter_execution_options = DynamicFilterExecutionOptions::default().with_max_build_rows(10_000);
        let keyed_hash_join_strategy_options =
            KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(11);
        let aggregate_strategy_execution_options =
            AggregateStrategyExecutionOptions::default().with_encoded_key_partial_min_input_rows(22);
        let source_chunk_planning_options =
            SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(33);
        let config = QueryServiceConfig {
            source_scheduler_policy: SourceSchedulerPolicy::new(4),
            source_chunk_planning_mode: SourceChunkPlanningMode::LocalParquetThroughput,
            native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
            stddev_samp_execution_mode: StddevSampExecutionMode::GpuNativeTolerant,
            dynamic_filter_execution_options,
            keyed_hash_join_strategy_options,
            aggregate_strategy_execution_options,
            source_chunk_planning_options,
            ..QueryServiceConfig::default()
        };
        let service = QueryService::new_for_admission_tests(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap();

        let handle = service.admit_query("tenant").unwrap();

        assert_eq!(handle.boundary_policy(), ExecutionBoundaryPolicy::default());
        assert_eq!(handle.source_scheduler_policy(), SourceSchedulerPolicy::new(4));
        assert_eq!(handle.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
        assert_eq!(handle.native_pipeline_execution_mode(), NativePipelineExecutionMode::Adaptive);
        assert_eq!(handle.stddev_samp_execution_mode(), StddevSampExecutionMode::GpuNativeTolerant);
        assert_eq!(handle.dynamic_filter_execution_options(), dynamic_filter_execution_options);
        assert_eq!(handle.keyed_hash_join_strategy_options(), keyed_hash_join_strategy_options);
        assert_eq!(handle.aggregate_strategy_execution_options(), aggregate_strategy_execution_options);
        assert_eq!(handle.source_chunk_planning_options(), source_chunk_planning_options);
    }

    #[test]
    fn query_service_rejects_inconsistent_dynamic_filter_execution_policy() {
        let config = QueryServiceConfig {
            dynamic_filter_execution_options: DynamicFilterExecutionOptions::default()
                .with_max_build_rows(10)
                .with_large_local_parquet_range_producer_rows(11),
            ..QueryServiceConfig::default()
        };

        let error = QueryService::new_for_admission_tests(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("large_local_parquet_range_producer_rows"));
        assert_eq!(error.fact_value("bound_field"), Some("max_build_rows"));
        assert_eq!(error.fact_value("reason"), Some("range_producer_rows_exceeds_runtime_build_cap"));
    }

    #[test]
    fn query_service_rejects_inconsistent_keyed_hash_join_strategy_policy() {
        let config = QueryServiceConfig {
            keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default()
                .with_large_keyed_hash_build_row_threshold(10),
            ..QueryServiceConfig::default()
        };

        let error = QueryService::new_for_admission_tests(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("large_field"), Some("large_keyed_hash_build_row_threshold"));
        assert_eq!(error.fact_value("small_field"), Some("small_reusable_hash_build_row_threshold"));
        assert_eq!(error.fact_value("reason"), Some("large_threshold_below_small_threshold"));
    }

    #[test]
    fn query_service_rejects_invalid_aggregate_strategy_execution_policy() {
        let config = QueryServiceConfig {
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default()
                .with_prepared_sort_groupby_low_cardinality_threshold_ppm(1_000_001),
            ..QueryServiceConfig::default()
        };

        let error = QueryService::new_for_admission_tests(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("prepared_sort_groupby_low_cardinality_threshold_ppm"));
        assert_eq!(error.fact_value("reason"), Some("ratio_ppm_above_one_hundred_percent"));
    }

    #[test]
    fn query_service_rejects_invalid_source_chunk_planning_policy() {
        let config = QueryServiceConfig {
            source_chunk_planning_options: SourceChunkPlanningOptions::default()
                .with_local_parquet_throughput_max_row_groups_per_chunk(0),
            ..QueryServiceConfig::default()
        };

        let error = QueryService::new_for_admission_tests(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_max_row_groups_per_chunk"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn query_service_resources_use_independent_lakehouse_cache_fill_limits() {
        let config = QueryServiceConfig {
            source_fetch: SourceFetchConfig {
                host_fetch_budget_bytes: Some(4096),
                host_fetch_acquire_timeout: Duration::from_secs(9),
                host_fetch_max_concurrent_ops: 5,
                lakehouse_cache_fill_budget_bytes: Some(2048),
                lakehouse_cache_fill_max_concurrent_ops: 2,
                ..SourceFetchConfig::default()
            },
            ..QueryServiceConfig::default()
        };
        let service = QueryService::new_with_deployment(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap();
        let resources = service.inner.resources.as_ref().unwrap();

        assert_eq!(resources.host_fetch_budget.budget_bytes(), 4096);
        assert_eq!(resources.source_fetch_concurrency.max_concurrent_ops(), 5);
        assert_eq!(resources.lakehouse_cache_fill_budget.budget_bytes(), 2048);
        assert_eq!(resources.lakehouse_cache_fill_concurrency.max_concurrent_ops(), 2);

        let handle = service.admit_query("tenant").unwrap();
        let tenant_context = handle.acquire_context().unwrap();
        let context = tenant_context.context();
        assert_eq!(context.host_fetch_budget().budget_bytes(), 4096);
        assert_eq!(context.host_fetch_acquire_timeout(), Duration::from_secs(9));
        assert_eq!(context.host_fetch_max_concurrent_ops(), 5);
        assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
        assert_eq!(context.lakehouse_cache_fill_max_concurrent_ops(), 2);

        context.lakehouse_cache_fill_events().record(super::super::LakehouseCacheFillEvent {
            outcome: super::super::LakehouseCacheFillOutcome::Completed,
            bytes: Some(128),
            elapsed: Duration::from_millis(1),
            cache_path_hash: "cache-path-hash".to_string(),
            stable_id_hash: "stable-id-hash".to_string(),
            file_index: None,
            chunk_index: None,
            error: None,
        });
        assert_eq!(context.lakehouse_cache_fill_event_snapshots().len(), 1);

        let other_handle = service.admit_query("tenant").unwrap();
        let other_tenant_context = other_handle.acquire_context().unwrap();
        assert!(other_tenant_context.context().lakehouse_cache_fill_event_snapshots().is_empty());
    }

    #[test]
    fn query_service_released_handle_rejects_context_acquisition() {
        let service = test_service(1, 8);
        let handle = service.admit_query("small").unwrap();
        handle.finish();

        let error = handle.acquire_context().unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("tenant_id"), Some("small"));
        assert_eq!(error.fact_value("reason"), Some("query_handle_released"));
    }

    #[test]
    fn query_service_context_keeps_admission_until_context_drop() {
        let service = runtime_test_service(1, 8);
        let handle = service.admit_query("large").unwrap();
        let tenant_context = handle.acquire_context().unwrap();
        drop(handle);

        assert!(service.try_admit_query("small").unwrap().is_none());
        drop(tenant_context);
        assert!(service.try_admit_query("small").unwrap().is_some());
    }

    #[test]
    fn query_service_handle_reuses_one_root_context() {
        let service = runtime_test_service(1, 8);
        let handle = service.admit_query("large").unwrap();
        let first = handle.acquire_context().unwrap();
        let second = handle.acquire_context().unwrap();

        assert!(Arc::ptr_eq(&first.context().shared_runtime(), &second.context().shared_runtime()));
    }

    #[test]
    fn query_service_rejects_stream_pool_smaller_than_admission_limit() {
        let config = QueryServiceConfig {
            max_concurrent_queries: 2,
            stream_pool_size: Some(1),
            ..QueryServiceConfig::default()
        };

        let error = QueryService::new_for_admission_tests(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("stream_pool_size"));
        assert_eq!(error.fact_value("reason"), Some("stream_pool_smaller_than_admission_limit"));
    }

    #[test]
    fn query_service_rejects_stream_pool_that_covers_only_root_streams() {
        let config = QueryServiceConfig {
            max_concurrent_queries: 2,
            source_scheduler_policy: SourceSchedulerPolicy::new(1),
            stream_pool_size: Some(2),
            ..QueryServiceConfig::default()
        };

        let error = QueryService::new_for_admission_tests(
            config,
            vec![QueryServiceTenantConfig::new("tenant", 1024 * 1024)],
            non_gds_deployment(),
        )
        .unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("stream_pool_size"));
        assert_eq!(error.fact_value("max_source_reads_in_flight"), Some("1"));
        assert_eq!(error.fact_value("minimum_stream_pool_size"), Some("4"));
    }

    #[test]
    fn query_service_try_admit_does_not_queue_when_saturated() {
        let service = test_service(1, 8);
        let active = service.try_admit_query("large").unwrap().expect("first query should admit");

        assert!(service.try_admit_query("small").unwrap().is_none());
        let snapshot = service.admission_snapshot().unwrap();
        assert_eq!(snapshot.active_queries, 1);
        assert_eq!(snapshot.queued_queries, 0);

        drop(active);
        assert!(service.try_admit_query("small").unwrap().is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn query_service_admit_async_wakes_on_release() {
        let service = test_service(1, 8);
        let active = service.admit_query("large").unwrap();
        let queued_service = service.clone();
        let queued = tokio::spawn(async move { queued_service.admit_async("small").await });
        wait_for_queued_len(&service, 1);

        drop(active);
        let handle = time::timeout(Duration::from_millis(200), queued).await.unwrap().unwrap().unwrap();

        assert_eq!(handle.tenant_id().as_str(), "small");
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn query_service_admit_async_notifies_after_queue_head_admits() {
        let service = test_service(3, 8);
        let active_1 = service.admit_query("large").unwrap();
        let active_2 = service.admit_query("large").unwrap();
        let active_3 = service.admit_query("large").unwrap();
        let first_service = service.clone();
        let second_service = service.clone();
        let first = tokio::spawn(async move { first_service.admit_async("small").await });
        let second = tokio::spawn(async move { second_service.admit_async("small").await });
        wait_for_queued_len(&service, 2);

        drop(active_1);
        drop(active_2);
        drop(active_3);
        let first = time::timeout(Duration::from_millis(200), first).await.unwrap().unwrap().unwrap();
        let second = time::timeout(Duration::from_millis(200), second).await.unwrap().unwrap().unwrap();

        assert_eq!(first.tenant_id().as_str(), "small");
        assert_eq!(second.tenant_id().as_str(), "small");
        assert_eq!(service.admission_snapshot().unwrap().active_queries, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn query_service_cancelled_admit_async_removes_queued_request() {
        let service = test_service(1, 8);
        let active = service.admit_query("large").unwrap();
        let queued_service = service.clone();
        let queued = tokio::spawn(async move { queued_service.admit_async("small").await });
        wait_for_queued_len(&service, 1);

        queued.abort();
        assert!(queued.await.unwrap_err().is_cancelled());
        wait_for_exact_queued_len(&service, 0);
        drop(active);

        assert!(service.try_admit_query("small").unwrap().is_some());
    }

    #[test]
    fn query_service_concurrent_budget_sum_never_exceeds_capacity() {
        let capacity_bytes = 12 * 1024 * 1024;
        let max_concurrent_queries = 4;
        let per_query_budget = capacity_bytes / u64::try_from(max_concurrent_queries).unwrap();
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(capacity_bytes),
                max_concurrent_queries,
                max_admission_queue_len: 8,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::new("tenant", per_query_budget)],
            non_gds_deployment(),
        )
        .unwrap();

        let handles = (0..max_concurrent_queries)
            .map(|_| service.try_admit_query("tenant").unwrap().expect("capacity should admit query"))
            .collect::<Vec<_>>();

        let admitted_budget_sum: u64 = handles.iter().map(QueryHandle::device_budget_bytes).sum();
        assert_eq!(admitted_budget_sum, capacity_bytes);
        assert!(service.try_admit_query("tenant").unwrap().is_none());
    }

    #[test]
    fn query_service_uses_variable_memory_governor_grants() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 4,
                max_admission_queue_len: 8,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 60)],
            non_gds_deployment(),
        )
        .unwrap();

        let first = service.try_admit_query("tenant").unwrap().expect("first query should admit");
        let second = service.try_admit_query("tenant").unwrap().expect("second query should preserve remaining minima");
        let third = service.try_admit_query("tenant").unwrap().expect("third query should admit minimum grant");
        let fourth = service.try_admit_query("tenant").unwrap().expect("fourth query should admit minimum grant");

        assert_eq!(first.device_budget_bytes(), 60);
        assert_eq!(second.device_budget_bytes(), 20);
        assert_eq!(third.device_budget_bytes(), 10);
        assert_eq!(fourth.device_budget_bytes(), 10);
        assert!(service.try_admit_query("tenant").unwrap().is_none());
        let report = service.memory_governor_report().unwrap();
        assert_eq!(report.capacity_bytes, 100);
        assert_eq!(report.outstanding_bytes, 100);
        assert_eq!(report.available_bytes, 0);

        drop(first);
        assert_eq!(service.memory_governor_report().unwrap().available_bytes, 60);
    }

    #[test]
    fn adaptive_query_service_grows_partial_active_grant_on_release() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 95)],
            non_gds_deployment(),
        )
        .unwrap();

        let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
        let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
        let second_grant = second.lifetime.memory_grant().unwrap();

        assert_eq!(first.device_budget_bytes(), 90);
        assert_eq!(second.device_budget_bytes(), 10);
        assert_eq!(second_grant.budget_bytes(), 10);

        drop(first);

        assert_eq!(second_grant.budget_bytes(), 90);
        let report = service.memory_governor_report().unwrap();
        assert_eq!(report.outstanding_bytes, 90);
        assert_eq!(report.available_bytes, 10);
    }

    #[test]
    fn strict_query_service_does_not_grow_partial_active_grant_on_release() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                native_pipeline_execution_mode: NativePipelineExecutionMode::StrictBounded,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 95)],
            non_gds_deployment(),
        )
        .unwrap();

        let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
        let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
        let second_grant = second.lifetime.memory_grant().unwrap();

        drop(first);

        assert_eq!(second_grant.budget_bytes(), 10);
        let report = service.memory_governor_report().unwrap();
        assert_eq!(report.outstanding_bytes, 10);
        assert_eq!(report.available_bytes, 90);
    }

    #[test]
    fn adaptive_query_service_preserves_queued_minimum_before_growing_active_grants() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 80)],
            non_gds_deployment(),
        )
        .unwrap();

        let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
        let second = service.try_admit_query("tenant").unwrap().expect("second query should admit remainder");
        let second_grant = second.lifetime.memory_grant().unwrap();
        let mut queued = service.enqueue_admission("tenant").unwrap();

        drop(first);

        assert_eq!(second_grant.budget_bytes(), 20);
        let admitted = queued.try_admit().unwrap().expect("queued query should receive released capacity");
        assert_eq!(admitted.device_budget_bytes(), 80);
        assert_eq!(second_grant.budget_bytes(), 20);
    }

    #[test]
    fn adaptive_query_service_grows_after_queue_drains_when_capacity_remains() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
                ..QueryServiceConfig::default()
            },
            vec![
                QueryServiceTenantConfig::with_grant_range("wide", 10, 80),
                QueryServiceTenantConfig::new("small", 10),
            ],
            non_gds_deployment(),
        )
        .unwrap();

        let first = service.try_admit_query("wide").unwrap().expect("first wide query should admit want");
        let second = service.try_admit_query("wide").unwrap().expect("second wide query should admit remainder");
        let second_grant = second.lifetime.memory_grant().unwrap();
        let mut queued = service.enqueue_admission("small").unwrap();

        drop(first);

        assert_eq!(second_grant.budget_bytes(), 20);
        let admitted = queued.try_admit().unwrap().expect("small queued query should receive its minimum");
        assert_eq!(admitted.device_budget_bytes(), 10);
        assert_eq!(second_grant.budget_bytes(), 80);
    }

    #[test]
    fn adaptive_blocking_query_service_grows_after_queue_drains_when_capacity_remains() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                admission_wait_timeout: Duration::from_secs(1),
                native_pipeline_execution_mode: NativePipelineExecutionMode::Adaptive,
                ..QueryServiceConfig::default()
            },
            vec![
                QueryServiceTenantConfig::with_grant_range("wide", 10, 80),
                QueryServiceTenantConfig::new("small", 10),
            ],
            non_gds_deployment(),
        )
        .unwrap();

        let first = service.try_admit_query("wide").unwrap().expect("first wide query should admit want");
        let second = service.try_admit_query("wide").unwrap().expect("second wide query should admit remainder");
        let second_grant = second.lifetime.memory_grant().unwrap();
        let queued_service = service.clone();
        let queued = std::thread::spawn(move || queued_service.admit_query("small").unwrap());

        wait_for_queued_len(&service, 1);
        drop(first);

        let admitted = queued.join().expect("blocking admission thread should not panic");
        assert_eq!(admitted.device_budget_bytes(), 10);
        assert_eq!(second_grant.budget_bytes(), 80);
    }

    #[test]
    fn query_service_optimistic_slot_guarantee_does_not_reserve_empty_slot_minima() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 4,
                slot_guarantee: SlotGuarantee::Optimistic,
                max_admission_queue_len: 8,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range("tenant", 10, 60)],
            non_gds_deployment(),
        )
        .unwrap();

        let first = service.try_admit_query("tenant").unwrap().expect("first query should admit want");
        let second = service.try_admit_query("tenant").unwrap().expect("second query should consume remaining memory");

        assert_eq!(first.device_budget_bytes(), 60);
        assert_eq!(second.device_budget_bytes(), 40);
        assert!(service.try_admit_query("tenant").unwrap().is_none());
        assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
    }

    #[test]
    fn query_service_reserves_remaining_slots_for_largest_tenant_minimum() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::with_grant_range("low", 10, 100), QueryServiceTenantConfig::new("high", 60)],
            non_gds_deployment(),
        )
        .unwrap();

        let low = service.try_admit_query("low").unwrap().expect("low-min tenant should admit");
        let high = service.try_admit_query("high").unwrap().expect("high-min tenant should retain capacity");

        assert_eq!(low.device_budget_bytes(), 40);
        assert_eq!(high.device_budget_bytes(), 60);
        assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
    }

    #[test]
    fn query_service_waiting_on_memory_grant_wakes_after_grant_release() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                admission_wait_timeout: Duration::from_secs(2),
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::new("tenant", 100)],
            non_gds_deployment(),
        )
        .unwrap();
        let active = service.admit_query("tenant").unwrap();
        let queued_service = service.clone();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            queued_service.admit_query("tenant").unwrap()
        });
        started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        wait_for_queued_len(&service, 1);
        assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);

        drop(active);
        let admitted = waiter.join().unwrap();
        assert_eq!(admitted.device_budget_bytes(), 100);
        assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);
        drop(admitted);
        assert_eq!(service.memory_governor_report().unwrap().available_bytes, 100);
    }

    #[test]
    fn query_service_waiting_on_memory_grant_wakes_after_retained_grant_release() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                admission_wait_timeout: Duration::from_secs(2),
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::new("tenant", 100)],
            non_gds_deployment(),
        )
        .unwrap();
        let active = service.admit_query("tenant").unwrap();
        let retained_grant = active.lifetime.memory_grant().unwrap();
        drop(active);
        assert_eq!(service.memory_governor_report().unwrap().available_bytes, 0);

        let queued_service = service.clone();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            queued_service.admit_query("tenant").unwrap()
        });
        started_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        wait_for_queued_len(&service, 1);

        drop(retained_grant);
        let admitted = waiter.join().unwrap();
        assert_eq!(admitted.device_budget_bytes(), 100);
    }

    #[test]
    fn query_admission_waiter_times_out_while_selected_but_memory_blocked() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                admission_wait_timeout: Duration::from_millis(20),
                ..QueryServiceConfig::default()
            },
            vec![QueryServiceTenantConfig::new("tenant", 100)],
            non_gds_deployment(),
        )
        .unwrap();
        let _active = service.admit_query("tenant").unwrap();
        let mut waiter = service.enqueue_admission("tenant").unwrap();
        std::thread::sleep(Duration::from_millis(30));

        let error = waiter.try_admit().unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("reason"), Some("memory_grant_wait_timeout"));
    }

    #[test]
    fn query_service_work_conserving_admission_skips_memory_blocked_head() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 2,
                max_admission_queue_len: 8,
                ..QueryServiceConfig::default()
            },
            vec![
                QueryServiceTenantConfig::new("holder", 90),
                QueryServiceTenantConfig::new("large", 100),
                QueryServiceTenantConfig::new("small", 10),
            ],
            non_gds_deployment(),
        )
        .unwrap();
        let _active = service.admit_query("holder").unwrap();
        let mut large = service.enqueue_admission("large").unwrap();
        let mut small = service.enqueue_admission("small").unwrap();

        assert!(large.try_admit().unwrap().is_none());
        let admitted_small = small.try_admit().unwrap().expect("small query should pass memory-blocked large query");

        assert_eq!(admitted_small.tenant_id().as_str(), "small");
        assert_eq!(admitted_small.device_budget_bytes(), 10);
        assert_eq!(admitted_small.admission_sequence(), 2);
        let snapshot = service.admission_snapshot().unwrap();
        assert_eq!(snapshot.active_queries, 2);
        assert_eq!(snapshot.queued_queries, 1);
        assert_eq!(snapshot.queued_by_tenant[1].tenant_id.as_str(), "large");
        assert_eq!(snapshot.queued_by_tenant[1].queued_queries, 1);
    }

    #[test]
    fn query_service_admission_priority_passes_lower_priority_queue_head() {
        let service = QueryService::new_for_admission_tests(
            QueryServiceConfig {
                device_capacity_bytes: Some(100),
                max_concurrent_queries: 1,
                max_admission_queue_len: 8,
                ..QueryServiceConfig::default()
            },
            vec![
                QueryServiceTenantConfig::new("holder", 100),
                QueryServiceTenantConfig::new("normal", 10),
                QueryServiceTenantConfig::new("high", 10),
            ],
            non_gds_deployment(),
        )
        .unwrap();
        let active = service.admit_query("holder").unwrap();
        let mut normal = service.enqueue_admission("normal").unwrap();
        let mut high = service.enqueue_admission_with_qos("high", QueryQos::new(Priority::High)).unwrap();
        drop(active);

        assert!(normal.try_admit().unwrap().is_none());
        let admitted_high = high.try_admit().unwrap().expect("high-priority query should pass normal-priority head");

        assert_eq!(admitted_high.tenant_id().as_str(), "high");
        assert_eq!(admitted_high.admission_sequence(), 2);
        let report = service.memory_governor_report().unwrap();
        assert_eq!(report.grants.len(), 1);
        assert_eq!(report.grants[0].priority, "high");
    }

    #[test]
    fn query_service_enqueued_waiter_counts_against_queue_limit_until_drop() {
        let service = test_service(1, 1);
        let _active = service.admit_query("large").unwrap();
        let waiter = service.enqueue_admission("small").unwrap();

        let error = service.enqueue_admission("large").unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.fact_value("reason"), Some("admission_queue_full"));

        drop(waiter);
        assert!(service.enqueue_admission("large").is_ok());
    }

    #[test]
    fn query_service_close_wakes_waiting_admission() {
        let service = test_service(1, 8);
        let _active = service.admit_query("large").unwrap();
        let queued_service = service.clone();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_sender.send(()).unwrap();
            queued_service.admit_query("small").unwrap_err()
        });
        started_receiver.recv().unwrap();
        wait_for_queued_len(&service, 1);

        service.close().unwrap();
        let error = waiter.join().unwrap();

        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("admission_closed"));
    }

    #[test]
    fn query_service_rotates_admission_by_tenant_when_queries_wait() {
        let service = test_service(1, 8);
        let first = service.admit_query("large").unwrap();
        let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();

        for tenant in ["large", "large", "small"] {
            let service = service.clone();
            let ready_sender = ready_sender.clone();
            let result_sender = result_sender.clone();
            std::thread::spawn(move || {
                ready_sender.send(tenant.to_string()).unwrap();
                let handle = service.admit_query(tenant).unwrap();
                result_sender.send(handle.tenant_id().to_string()).unwrap();
                std::thread::sleep(Duration::from_millis(20));
                drop(handle);
            });
        }
        for _ in 0..3 {
            ready_receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        }
        wait_for_queued_len(&service, 3);

        drop(first);
        assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "small");
        assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "large");
        assert_eq!(result_receiver.recv_timeout(Duration::from_secs(1)).unwrap(), "large");
    }
}
