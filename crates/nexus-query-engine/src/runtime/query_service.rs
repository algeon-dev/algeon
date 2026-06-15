//! Multi-tenant service admission harness for native execution.

use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use super::{
    DeploymentIoCapability, ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext, ExecutionContextParts,
    GdsConcurrencyGate, HostFetchBudget, MemoryControlPolicy, MemoryResourcePolicy, ServiceStreamPool,
    SourceChunkPlanningMode, SourceFetchConcurrencyGate, SourceFetchConfig, SourceObjectStoreRegistry,
    SourceSchedulerPolicy, SourceStorageProfile, default_host_fetch_budget_bytes, default_lakehouse_cache_fill_runtime,
    default_source_fetch_runtime, validate_service_stream_pool_device,
};
use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    error::{QueryError, QueryErrorCode, Result},
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
    /// Per-query device budget for this tenant's query handles, in bytes.
    pub device_budget_bytes: u64,
}

impl QueryServiceTenantConfig {
    /// Creates a tenant config with a fixed bounded device-memory slice.
    pub fn new(tenant_id: impl Into<TenantId>, device_budget_bytes: u64) -> Self {
        Self { tenant_id: tenant_id.into(), device_budget_bytes }
    }

    fn validate(&self) -> Result<()> {
        self.tenant_id.validate()?;
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
        Ok(())
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
    /// Maximum number of admitted query handles active at one time.
    pub max_concurrent_queries: usize,
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
        validate_nonzero_config_field("max_source_chunk_bytes", self.max_source_chunk_bytes == 0)?;
        validate_nonzero_config_field("max_row_groups_per_chunk", self.max_row_groups_per_chunk == 0)?;
        validate_nonzero_config_field("max_retry_attempts", self.max_retry_attempts == 0)?;
        validate_nonzero_config_field("min_retry_chunk_row_groups", self.min_retry_chunk_row_groups == 0)?;
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
            max_concurrent_queries: SourceFetchConfig::DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS,
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

#[derive(Debug)]
struct QueryServiceInner {
    config: QueryServiceConfig,
    tenant_budgets: HashMap<TenantId, u64>,
    tenant_order: Vec<TenantId>,
    resources: Option<QueryServiceResources>,
    admission: Mutex<AdmissionState>,
    admission_available: Condvar,
    next_query_id: AtomicU64,
    next_admission_sequence: AtomicU64,
}

#[derive(Debug, Clone)]
struct QueryServiceResources {
    service_stream_pool: ServiceStreamPool,
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
    next_tenant_index: usize,
}

#[derive(Debug, Clone)]
struct AdmissionRequest {
    request_id: u64,
    tenant_id: TenantId,
}

impl QueryService {
    /// Creates a service entry point and initializes service-owned runtime gates.
    pub fn new(config: QueryServiceConfig, tenants: Vec<QueryServiceTenantConfig>) -> Result<Self> {
        let deployment = DeploymentIoCapability::cached_current_host();
        Self::new_with_deployment(config, tenants, deployment)
    }

    /// Creates a service entry point with caller-supplied deployment diagnostics.
    pub fn new_with_deployment(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let device = cudf::DeviceId::new(config.device_ordinal).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.query_service.device",
                "failed to resolve query service CUDA device",
                error,
            )
        })?;
        let stream_pool_size =
            config.stream_pool_size.unwrap_or_else(|| super::default_service_stream_pool_size_for_device(device));
        let resources = QueryServiceResources {
            service_stream_pool: ServiceStreamPool::new(device, stream_pool_size)?,
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
            source_fetch_runtime: default_source_fetch_runtime(),
            lakehouse_cache_fill_runtime: default_lakehouse_cache_fill_runtime(),
            source_object_stores: Arc::new(SourceObjectStoreRegistry::with_profiles(
                config.source_storage_profiles.clone(),
            )?),
            deployment_io_capability,
        };
        Self::build(config, tenants, Some(resources))
    }

    #[cfg(test)]
    fn new_for_admission_tests(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        config.validate(&deployment_io_capability)?;
        Self::build(config, tenants, None)
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

        let mut tenant_budgets = HashMap::with_capacity(tenants.len());
        let mut tenant_order = Vec::with_capacity(tenants.len());
        for tenant in tenants {
            tenant.validate()?;
            if tenant_budgets.insert(tenant.tenant_id.clone(), tenant.device_budget_bytes).is_some() {
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

        Ok(Self {
            inner: Arc::new(QueryServiceInner {
                config,
                tenant_budgets,
                tenant_order,
                resources,
                admission: Mutex::new(AdmissionState::default()),
                admission_available: Condvar::new(),
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
        let tenant_id = tenant_id.into();
        let device_budget_bytes = self.tenant_budget(&tenant_id)?;
        let request_id = self.inner.next_query_id.fetch_add(1, Ordering::Relaxed);
        let requested_at = Instant::now();
        let request = AdmissionRequest { request_id, tenant_id: tenant_id.clone() };
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);

        let mut state = self.lock_admission()?;
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
            if self.request_is_admissible(&state, request_id)
                && let Some(position) = state.queued.iter().position(|queued| queued.request_id == request_id)
            {
                let admitted = state.queued.remove(position).expect("admissible request exists");
                state.active_queries = state.active_queries.saturating_add(1);
                self.rotate_after_admission(&mut state, &admitted.tenant_id);
                let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
                return Ok(QueryHandle {
                    service: Arc::clone(&self.inner),
                    tenant_id,
                    query_id: request_id,
                    admission_sequence,
                    admission_wait: requested_at.elapsed(),
                    device_budget_bytes,
                    released: AtomicBool::new(false),
                });
            }

            let now = Instant::now();
            if now >= deadline {
                remove_queued_request(&mut state, request_id);
                self.inner.admission_available.notify_all();
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
                self.inner.admission_available.notify_all();
                return Err(service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                ));
            }
        }
    }

    /// Returns the configured device budget for a tenant.
    pub fn tenant_budget(&self, tenant_id: &TenantId) -> Result<u64> {
        self.inner.tenant_budgets.get(tenant_id).copied().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.admission",
                "unknown tenant id for query service admission",
            )
            .with_fact("tenant_id", tenant_id.as_str())
        })
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

    fn request_is_admissible(&self, state: &AdmissionState, request_id: u64) -> bool {
        if state.active_queries >= self.inner.config.max_concurrent_queries {
            return false;
        }
        let Some(selected) = self.select_next_request(state) else {
            return false;
        };
        selected.request_id == request_id
    }

    fn select_next_request<'a>(&self, state: &'a AdmissionState) -> Option<&'a AdmissionRequest> {
        if state.queued.is_empty() {
            return None;
        }
        for offset in 0..self.inner.tenant_order.len() {
            let tenant_index = (state.next_tenant_index + offset) % self.inner.tenant_order.len();
            let tenant_id = &self.inner.tenant_order[tenant_index];
            if let Some(request) = state.queued.iter().find(|request| &request.tenant_id == tenant_id) {
                return Some(request);
            }
        }
        state.queued.front()
    }

    fn rotate_after_admission(&self, state: &mut AdmissionState, tenant_id: &TenantId) {
        if let Some(index) = self.inner.tenant_order.iter().position(|candidate| candidate == tenant_id) {
            state.next_tenant_index = (index + 1) % self.inner.tenant_order.len();
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

/// RAII admission handle for one tenant query.
#[derive(Debug)]
pub struct QueryHandle {
    service: Arc<QueryServiceInner>,
    tenant_id: TenantId,
    query_id: u64,
    admission_sequence: u64,
    admission_wait: Duration,
    device_budget_bytes: u64,
    released: AtomicBool,
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
        if self.released.load(Ordering::Acquire) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.context",
                "query service handle was released before context acquisition",
            )
            .with_fact("tenant_id", self.tenant_id.as_str())
            .with_fact("query_id", self.query_id)
            .with_fact("reason", "query_handle_released"));
        }
        let Some(resources) = &self.service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.context",
                "query service was constructed without runtime resources",
            ));
        };
        let stream_lease = resources.service_stream_pool.acquire()?;
        validate_service_stream_pool_device(&resources.service_stream_pool, stream_lease.runtime().device())?;
        let stream_lease_guard = Arc::new(stream_lease);
        let context = ExecutionContext::from_parts(ExecutionContextParts {
            runtime: stream_lease_guard.shared_runtime(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy(),
            boundary_policy: self.service.config.boundary_policy,
            source_scheduler_policy: self.service.config.source_scheduler_policy,
            source_chunk_planning_mode: self.service.config.source_chunk_planning_mode,
            source_fetch_config: self.service.config.source_fetch.clone(),
            service_stream_pool: resources.service_stream_pool.clone(),
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
            cancellation: ExecutionCancellation::new(),
            native_pipeline_execution_mode: self.service.config.native_pipeline_execution_mode,
            stddev_samp_execution_mode: self.service.config.stddev_samp_execution_mode,
            aggregate_strategy_execution_options: self.service.config.aggregate_strategy_execution_options,
            source_chunk_planning_options: self.service.config.source_chunk_planning_options,
            adaptive_memory_history: Some(Arc::new(AdaptiveMemoryHistory::default())),
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
            stream_lease_guard: Some(stream_lease_guard),
        });
        Ok(TenantQueryContext { context })
    }

    /// Releases the admission slot before dropping the handle.
    pub fn finish(&self) {
        self.release_once();
    }

    fn release_once(&self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        match self.service.admission.lock() {
            Ok(mut state) => {
                state.active_queries = state.active_queries.saturating_sub(1);
                self.service.admission_available.notify_all();
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to release poisoned query service admission state");
            },
        }
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

    #[test]
    fn query_service_assigns_fixed_per_tenant_bounded_budget() {
        let service = test_service(2, 8);

        let large = service.admit_query("large").unwrap();
        let small = service.admit_query("small").unwrap();

        assert_eq!(large.device_budget_bytes(), 8 * 1024 * 1024);
        assert_eq!(small.device_budget_bytes(), 1024 * 1024);
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
