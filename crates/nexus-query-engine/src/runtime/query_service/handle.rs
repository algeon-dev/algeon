use super::{super::create_grant_reservation_arena, *};

#[derive(Debug)]
pub(super) struct QueryHandleLifetime {
    pub(super) service: Arc<QueryServiceInner>,
    pub(super) tenant_id: TenantId,
    pub(super) query_id: u64,
    release_requested: AtomicBool,
    released: AtomicBool,
    active_contexts: AtomicUsize,
    root_context: Mutex<Option<ExecutionContext>>,
    memory_grant: Mutex<Option<MemoryGrant>>,
    device_slot: Mutex<Option<DeviceActiveQuerySlot>>,
}

#[derive(Debug)]
struct QueryContextLifetimeGuard {
    pub(super) lifetime: Arc<QueryHandleLifetime>,
}

impl QueryHandleLifetime {
    pub(super) fn new(
        service: Arc<QueryServiceInner>,
        tenant_id: TenantId,
        query_id: u64,
        memory_grant: MemoryGrant,
        device_slot: Option<DeviceActiveQuerySlot>,
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
            device_slot: Mutex::new(device_slot),
        })
    }

    pub(super) fn memory_grant(&self) -> Result<MemoryGrant> {
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
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to release poisoned query service admission state");
            },
        }
        self.shrink_memory_grant_to_live_floor();
        if let Some(device_ordinal) = self.release_device_slot() {
            notify_shared_device_admission_available(device_ordinal);
        }
        self.service.notify_admission_available();
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

    fn release_device_slot(&self) -> Option<usize> {
        match self.device_slot.lock() {
            Ok(mut slot) => slot.take().map(|slot| {
                let device_ordinal = slot.device_ordinal();
                drop(slot);
                device_ordinal
            }),
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to drop poisoned query service device slot");
                None
            },
        }
    }

    fn shrink_memory_grant_to_live_floor(&self) {
        let grant = match self.memory_grant.lock() {
            Ok(memory_grant) => memory_grant.as_ref().cloned(),
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to inspect poisoned query service memory grant");
                None
            },
        };
        let Some(grant) = grant else {
            return;
        };
        let live_reserved_bytes = grant.live_reserved_bytes();
        if live_reserved_bytes == 0 {
            return;
        }
        if let Err(error) = grant.try_shrink_to(live_reserved_bytes) {
            eprintln!("[nexus-query-engine] warning: failed to shrink query service memory grant on release: {error}");
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

/// RAII admission handle for one tenant query.
#[derive(Debug)]
pub struct QueryHandle {
    pub(super) service: Arc<QueryServiceInner>,
    pub(super) tenant_id: TenantId,
    pub(super) query_id: u64,
    pub(super) admission_sequence: u64,
    pub(super) admission_wait: Duration,
    pub(super) device_budget_bytes: u64,
    pub(super) device_ordinal: usize,
    pub(super) lifetime: Arc<QueryHandleLifetime>,
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
        let stream_acquire_deadline =
            Instant::now().checked_add(self.service.config.admission_wait_timeout).unwrap_or_else(Instant::now);
        let stream_lease = device.stream_pool().acquire_with_cancellation_deadline(
            &cancellation,
            stream_acquire_deadline,
            "nexus_query_engine.query_service.context.stream_acquire",
        )?;
        validate_service_stream_pool_device(device.stream_pool(), stream_lease.runtime().device())?;
        let stream_lease_guard = Arc::new(stream_lease);
        let memory_grant = self.lifetime.memory_grant()?;
        let runtime = stream_lease_guard.shared_runtime();
        let grant_reservation_arena = create_grant_reservation_arena(runtime.as_ref(), Some(&memory_grant))?;
        let context = ExecutionContext::from_parts(ExecutionContextParts {
            query_id: Some(QueryId::new(format!("service-{}", self.query_id))),
            trace_id: None,
            runtime,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy(),
            memory_grant: Some(memory_grant),
            grant_reservation_arena,
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
            graph_memory_calibration_override: self.service.config.graph_memory_calibration_override,
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
