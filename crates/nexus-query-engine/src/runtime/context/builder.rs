//! Construction and policy validation for native execution contexts.

use std::{
    error::Error as StdError,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Duration,
};

use super::{
    super::{
        CacheFillByteBudget, DeploymentIoCapability, ExecutionBoundaryPolicy, GpuExecutionPool,
        LakehouseCacheFillConfig, MemoryControlPolicy, MemoryGrant, MemoryResourcePolicy, ServiceStreamPool,
        SourceChunkPlanningMode, SourceStorageProfile,
        cache_events::LakehouseCacheFillEventSink,
        cache_fill::{CacheFillConcurrencyGate, default_lakehouse_cache_fill_budget_bytes},
        memory_domain::create_grant_reservation_arena,
        object_store::{SourceObjectStoreRegistry, default_object_store_control_runtime},
        service_stream::{default_service_stream_pool_size_for_device, validate_service_stream_pool_size},
    },
    ExecutionCancellation, ExecutionContext, ExecutionContextParts, RuntimeObservedExecutionContext,
    SourceSchedulerPolicy,
};
use crate::{
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    graph_memory::GraphMemoryCalibrationOverride,
    metrics::{QueryId, TraceId},
    mode::StddevSampExecutionMode,
    observability::{
        RuntimeObservationConfig, RuntimeObservationHandle, RuntimeObservationSession, RuntimeQueryId,
        RuntimeTerminalOutcome, RuntimeTraceId,
    },
};

/// Builder for [`ExecutionContext`].
///
/// The builder separates runtime construction policy from execution policy. If
/// a runtime is supplied, it is used as-is and `device_ordinal` /
/// `memory_policy` no longer construct CUDA state. If no runtime is supplied,
/// the builder creates a cudf runtime from the requested device ordinal and
/// memory-resource policy.
#[derive(Debug, Default)]
pub(crate) struct ExecutionContextBuilder {
    /// Optional caller-provided diagnostic query id.
    query_id: Option<QueryId>,
    /// Optional caller-provided parent trace id.
    trace_id: Option<TraceId>,
    /// Query-local bounded runtime observation policy.
    runtime_observation_config: RuntimeObservationConfig,
    /// Prebuilt runtime to use instead of constructing one.
    runtime: Option<Arc<cudf::Runtime>>,
    /// Optional server-generation gate for file-backed native I/O.
    io_lifecycle: Option<crate::runtime::IoLifecycle>,
    /// Optional caller-owned dedicated GPU execution pool.
    gpu_execution_pool: Option<GpuExecutionPool>,
    /// CUDA device ordinal used only when this builder constructs the runtime.
    device_ordinal: Option<usize>,
    /// Memory resource used only when this builder constructs the runtime.
    memory_policy: MemoryResourcePolicy,
    /// Engine-level bounded/unbounded memory contract.
    memory_control_policy: MemoryControlPolicy,
    /// Optional soft governor grant backing bounded execution.
    memory_grant: Option<MemoryGrant>,
    /// Source/build/probe/result boundary contract.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Source read scheduling contract.
    source_scheduler_policy: SourceSchedulerPolicy,
    source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Optional service-level lakehouse cache-fill configuration.
    lakehouse_cache_fill_config: Option<LakehouseCacheFillConfig>,
    /// Source object-store profiles available to execution-time fetches.
    source_storage_profiles: Vec<SourceStorageProfile>,
    /// Optional caller-owned service stream pool.
    service_stream_pool: Option<ServiceStreamPool>,
    /// Pool size for the default process-global service stream pool.
    service_stream_pool_size: Option<usize>,
    /// Optional caller-owned lakehouse cache-fill byte budget.
    lakehouse_cache_fill_budget: Option<CacheFillByteBudget>,
    /// Background lakehouse cache-fill budget size used when constructing the default fill budget.
    lakehouse_cache_fill_budget_bytes: Option<u64>,
    /// Max background lakehouse complete-object cache-fill operations allowed concurrently.
    lakehouse_cache_fill_max_concurrent_ops: Option<usize>,
    /// Optional caller-supplied deployment IO capability diagnostic.
    deployment_io_capability: Option<DeploymentIoCapability>,
    /// Cooperative cancellation flag observed by native execution.
    cancellation: ExecutionCancellation,
    /// Execution contract for floating statistical aggregates.
    stddev_samp_execution_mode: StddevSampExecutionMode,
    /// Runtime threshold policy for grouped aggregate strategy selection.
    aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert policy for local-throughput source chunk planning.
    source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Optional graph calibration metadata supplied by fixture/config.
    graph_memory_calibration_override: GraphMemoryCalibrationOverride,
}

/// Builder that owns a caller-supplied runtime observation session.
///
/// Attaching the unique observation owner moves construction onto this type so
/// every consuming error path can return that owner to the caller.
#[derive(Debug)]
pub struct RuntimeObservedExecutionContextBuilder {
    builder: ExecutionContextBuilder,
    observation: RuntimeObservationSession,
}

/// Recoverable failure to build a runtime-observed execution context.
///
/// The native [`QueryError`] remains available for structured classification,
/// while [`Self::into_parts`] returns the unique observation owner so retained
/// records can still be inspected and sealed.
#[derive(Debug)]
pub struct RuntimeObservationBuildError {
    error: Option<Box<QueryError>>,
    observation: Option<RuntimeObservationSession>,
}

impl RuntimeObservationBuildError {
    fn new(error: QueryError, observation: RuntimeObservationSession) -> Self {
        Self { error: Some(Box::new(error)), observation: Some(observation) }
    }

    /// Returns the structured native build error.
    pub fn query_error(&self) -> &QueryError {
        self.error.as_deref().expect("runtime observation build error always contains its query error")
    }

    /// Returns the recovered unique observation owner.
    pub fn observation(&self) -> &RuntimeObservationSession {
        self.observation.as_ref().expect("runtime observation build error always contains its observation owner")
    }

    /// Separates the structured error from the recovered observation owner.
    pub fn into_parts(mut self) -> (QueryError, RuntimeObservationSession) {
        let error = self.error.take().expect("runtime observation build error always contains its query error");
        let observation =
            self.observation.take().expect("runtime observation build error always contains its observation owner");
        (*error, observation)
    }
}

impl fmt::Display for RuntimeObservationBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.query_error(), formatter)
    }
}

impl StdError for RuntimeObservationBuildError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.query_error())
    }
}

impl Drop for RuntimeObservationBuildError {
    fn drop(&mut self) {
        let Some(observation) = self.observation.take() else {
            return;
        };
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _ = observation.seal(RuntimeTerminalOutcome::Dropped);
        }));
    }
}

impl ExecutionContextBuilder {
    /// Creates a builder with default runtime and execution policies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses an already-constructed cudf runtime.
    ///
    /// This marks the memory-resource policy as
    /// [`MemoryResourcePolicy::ExternalRuntime`] because allocator/device setup
    /// has happened outside the query engine.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Uses an already-shared cudf runtime.
    ///
    /// Prefer this over building separate runtimes when an adapter needs one
    /// CUDA stream, allocator, or device policy across multiple executions.
    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Attaches a server-generation lifecycle to file-backed native I/O.
    pub fn with_io_lifecycle(mut self, lifecycle: crate::runtime::IoLifecycle) -> Self {
        self.io_lifecycle = Some(lifecycle);
        self
    }

    /// Uses a caller-owned dedicated executor for synchronous GPU work.
    pub(crate) fn with_gpu_execution_pool(mut self, pool: GpuExecutionPool) -> Self {
        self.gpu_execution_pool = Some(pool);
        self
    }

    /// Supplies the query-wide diagnostic correlation id.
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    /// Supplies the optional parent trace id.
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Enables or disables bounded runtime observation for the constructed context.
    pub fn with_runtime_observation_config(mut self, config: RuntimeObservationConfig) -> Self {
        self.runtime_observation_config = config;
        self
    }

    /// Supplies a caller-created query session and enters the owner-preserving build API.
    pub fn with_runtime_observation_session(
        mut self,
        session: RuntimeObservationSession,
    ) -> RuntimeObservedExecutionContextBuilder {
        self.runtime_observation_config = session.config();
        RuntimeObservedExecutionContextBuilder { builder: self, observation: session }
    }

    /// Selects the CUDA device ordinal for runtime construction.
    ///
    /// Ignored when a runtime is supplied with [`Self::with_runtime`] or
    /// [`Self::with_shared_runtime`].
    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    /// Selects the memory resource used when constructing a runtime.
    ///
    /// Ignored for an injected runtime except for the recorded
    /// `ExternalRuntime` diagnostic state.
    pub fn with_memory_policy(mut self, memory_policy: MemoryResourcePolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    /// Sets the engine-level memory control contract.
    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    /// Attaches a soft governor grant to bounded execution contexts built here.
    pub fn with_memory_grant(mut self, memory_grant: MemoryGrant) -> Self {
        self.memory_grant = Some(memory_grant);
        self
    }

    /// Sets the execution boundary policy.
    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    /// Sets bounded source scheduling controls.
    pub fn with_source_scheduler_policy(mut self, source_scheduler_policy: SourceSchedulerPolicy) -> Self {
        self.source_scheduler_policy = source_scheduler_policy;
        self
    }

    /// Set bounded source chunk planning mode.
    pub fn with_source_chunk_planning_mode(mut self, mode: SourceChunkPlanningMode) -> Self {
        self.source_chunk_planning_mode = mode;
        self
    }

    /// Sets the service-level lakehouse cache-fill resource configuration.
    pub fn with_lakehouse_cache_fill_config(mut self, config: LakehouseCacheFillConfig) -> Self {
        self.lakehouse_cache_fill_config = Some(config);
        self
    }

    /// Registers source object-store profiles used by remote reads and cache fills.
    pub fn with_source_storage_profiles(mut self, profiles: impl IntoIterator<Item = SourceStorageProfile>) -> Self {
        self.source_storage_profiles = profiles.into_iter().collect();
        self
    }

    /// Uses a caller-owned service stream pool for bounded source-read workers.
    pub(crate) fn with_service_stream_pool(mut self, service_stream_pool: ServiceStreamPool) -> Self {
        self.service_stream_pool = Some(service_stream_pool);
        self
    }

    /// Sets the size of the default process-global service stream pool.
    ///
    /// Ignored when a pool is supplied with [`Self::with_service_stream_pool`].
    pub fn with_service_stream_pool_size(mut self, pool_size: usize) -> Self {
        self.service_stream_pool_size = Some(pool_size);
        self
    }

    /// Uses a caller-owned lakehouse cache-fill byte budget.
    pub fn with_lakehouse_cache_fill_budget(mut self, budget: CacheFillByteBudget) -> Self {
        self.lakehouse_cache_fill_budget = Some(budget);
        self
    }

    /// Sets the service-wide background lakehouse cache-fill byte budget.
    pub fn with_lakehouse_cache_fill_budget_bytes(mut self, budget_bytes: u64) -> Self {
        self.lakehouse_cache_fill_budget_bytes = Some(budget_bytes);
        self
    }

    /// Sets the bounded wait for blocking lakehouse cache maintenance resources.
    pub fn with_lakehouse_cache_fill_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.lakehouse_cache_fill_config.get_or_insert_with(Default::default).acquire_timeout = timeout;
        self
    }

    /// Sets the service-wide background lakehouse cache-fill operation cap.
    pub fn with_lakehouse_cache_fill_max_concurrent_ops(mut self, max_concurrent_ops: usize) -> Self {
        self.lakehouse_cache_fill_max_concurrent_ops = Some(max_concurrent_ops);
        self
    }

    /// Supplies deployment IO capability diagnostics instead of probing the current host.
    pub fn with_deployment_io_capability(mut self, capability: DeploymentIoCapability) -> Self {
        self.deployment_io_capability = Some(capability);
        self
    }

    /// Sets the cooperative cancellation flag for native execution.
    pub fn with_cancellation(mut self, cancellation: ExecutionCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Selects the execution contract for floating statistical aggregates.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StddevSampExecutionMode) -> Self {
        self.stddev_samp_execution_mode = mode;
        self
    }

    /// Selects the execution contract through the legacy STDDEV_SAMP builder.
    pub fn with_stddev_samp_execution_mode(self, mode: StddevSampExecutionMode) -> Self {
        self.with_statistical_aggregate_execution_mode(mode)
    }

    /// Sets grouped aggregate strategy execution policy.
    pub fn with_aggregate_strategy_execution_options(mut self, options: AggregateStrategyExecutionOptions) -> Self {
        self.aggregate_strategy_execution_options = options;
        self
    }

    /// Sets expert local-throughput source chunk planning policy.
    pub fn with_source_chunk_planning_options(mut self, options: SourceChunkPlanningOptions) -> Self {
        self.source_chunk_planning_options = options;
        self
    }

    /// Supplies fixture/config graph calibration metadata.
    pub fn with_graph_memory_calibration_override(mut self, override_: GraphMemoryCalibrationOverride) -> Self {
        self.graph_memory_calibration_override = override_;
        self
    }

    /// Validates policies, constructs or records the runtime, and returns a context.
    pub fn build(self) -> Result<ExecutionContext> {
        if self.runtime_observation_config.is_enabled() {
            return Err(runtime_observation_owner_required_error());
        }
        self.build_context(RuntimeObservationHandle::disabled())
    }

    /// Builds a cloneable execution context paired with its unique observation owner.
    pub fn build_with_runtime_observation(
        self,
    ) -> std::result::Result<RuntimeObservedExecutionContext, RuntimeObservationBuildError> {
        let observation = RuntimeObservationSession::new(
            RuntimeQueryId::from(self.query_id.clone().unwrap_or_default()),
            self.trace_id.as_ref().map(|trace_id| RuntimeTraceId::new(trace_id.to_string())),
            self.runtime_observation_config,
        );
        self.build_with_observation(observation)
    }

    fn build_with_observation(
        self,
        observation: RuntimeObservationSession,
    ) -> std::result::Result<RuntimeObservedExecutionContext, RuntimeObservationBuildError> {
        let handle = observation.handle();
        match self.build_context(handle) {
            Ok(context) => Ok(RuntimeObservedExecutionContext::new(context, observation)),
            Err(error) => Err(RuntimeObservationBuildError::new(error, observation)),
        }
    }

    fn adopt_or_validate_runtime_observation_identity(
        &mut self,
        observation: &RuntimeObservationSession,
    ) -> Result<()> {
        let Some(observation_query_id) = observation.query_id() else {
            return Ok(());
        };
        match &self.query_id {
            Some(query_id) if query_id.as_str() != observation_query_id.as_str() => {
                return Err(runtime_observation_identity_error(
                    "runtime_observation_query_id_mismatch",
                    "query_id",
                    query_id.as_str(),
                    observation_query_id.as_str(),
                ));
            },
            Some(_) => {},
            None => self.query_id = Some(QueryId::new(observation_query_id.as_str())),
        }

        match (&self.trace_id, observation.trace_id()) {
            (Some(trace_id), Some(observation_trace_id)) if trace_id.as_str() != observation_trace_id.as_str() => {
                Err(runtime_observation_identity_error(
                    "runtime_observation_trace_id_mismatch",
                    "trace_id",
                    trace_id.as_str(),
                    observation_trace_id.as_str(),
                ))
            },
            (Some(trace_id), None) => Err(runtime_observation_identity_error(
                "runtime_observation_trace_id_mismatch",
                "trace_id",
                trace_id.as_str(),
                "none",
            )),
            (Some(_), Some(_)) | (None, None) => Ok(()),
            (None, Some(observation_trace_id)) => {
                self.trace_id = Some(TraceId::new(observation_trace_id.as_str()));
                Ok(())
            },
        }
    }

    fn build_context(self, runtime_observation: RuntimeObservationHandle) -> Result<ExecutionContext> {
        self.memory_control_policy.validate()?;
        self.source_scheduler_policy.validate()?;
        self.aggregate_strategy_execution_options.validate()?;
        self.source_chunk_planning_options.validate()?;
        if let Some(service_stream_pool_size) = self.service_stream_pool_size {
            validate_service_stream_pool_size(service_stream_pool_size)?;
        }
        let deployment_io_capability =
            self.deployment_io_capability.unwrap_or_else(DeploymentIoCapability::cached_current_host);
        let mut lakehouse_cache_fill_config = self.lakehouse_cache_fill_config.unwrap_or_default();
        if let Some(lakehouse_cache_fill_budget_bytes) = self.lakehouse_cache_fill_budget_bytes {
            lakehouse_cache_fill_config.budget_bytes = Some(lakehouse_cache_fill_budget_bytes);
        }
        if let Some(lakehouse_cache_fill_max_concurrent_ops) = self.lakehouse_cache_fill_max_concurrent_ops {
            lakehouse_cache_fill_config.max_concurrent_ops = lakehouse_cache_fill_max_concurrent_ops;
        }
        lakehouse_cache_fill_config.validate()?;
        let lakehouse_cache_fill_budget = match self.lakehouse_cache_fill_budget {
            Some(budget) => budget,
            None => CacheFillByteBudget::new(
                lakehouse_cache_fill_config.budget_bytes.unwrap_or_else(default_lakehouse_cache_fill_budget_bytes),
            )?,
        };
        let lakehouse_cache_fill_concurrency =
            CacheFillConcurrencyGate::new(lakehouse_cache_fill_config.max_concurrent_ops)?;
        let object_store_control_runtime = default_object_store_control_runtime()?;
        let source_object_stores = Arc::new(SourceObjectStoreRegistry::with_profiles(self.source_storage_profiles)?);
        let query_id = self.query_id;
        let trace_id = self.trace_id;
        let gpu_execution_pool = self.gpu_execution_pool;
        if let Some(runtime) = self.runtime {
            let service_stream_pool_size = self
                .service_stream_pool_size
                .unwrap_or_else(|| default_service_stream_pool_size_for_device(runtime.device()));
            let service_stream_pool = match self.service_stream_pool {
                Some(pool) => {
                    validate_service_stream_pool_device(&pool, runtime.device())?;
                    pool
                },
                None => ServiceStreamPool::for_device(runtime.device(), service_stream_pool_size)?,
            };
            let gpu_execution_pool =
                resolve_gpu_execution_pool(gpu_execution_pool, runtime.device(), service_stream_pool.pool_size())?;
            let grant_reservation_arena = create_grant_reservation_arena(runtime.as_ref(), self.memory_grant.as_ref())?;
            return Ok(ExecutionContext::from_parts(ExecutionContextParts {
                service_epoch: None,
                query_epoch: None,
                query_anchor: None,
                io_lifecycle: self.io_lifecycle,
                query_id,
                trace_id,
                runtime_observation,
                runtime_observation_owner: None,
                runtime_execution_scope: None,
                runtime_stream_lease_scope: crate::observability::RuntimeStreamLeaseScope::default(),
                runtime,
                gpu_execution_pool,
                memory_policy: MemoryResourcePolicy::ExternalRuntime,
                memory_control_policy: self.memory_control_policy,
                memory_grant: self.memory_grant.clone(),
                grant_reservation_arena,
                grant_memory_domain: None,
                attempt_allocation_domain: None,
                attempt_auxiliary_slots: None,
                admitted_capability_evidence: None,
                attempt_member_lease: None,
                boundary_policy: self.boundary_policy,
                source_scheduler_policy: self.source_scheduler_policy,
                source_chunk_planning_mode: self.source_chunk_planning_mode,
                lakehouse_cache_fill_config,
                service_stream_pool,
                lakehouse_cache_fill_budget,
                lakehouse_cache_fill_concurrency,
                lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
                object_store_control_runtime,
                source_object_stores,
                deployment_io_capability,
                cancellation: self.cancellation,
                stddev_samp_execution_mode: self.stddev_samp_execution_mode,
                aggregate_strategy_execution_options: self.aggregate_strategy_execution_options,
                source_chunk_planning_options: self.source_chunk_planning_options,
                graph_memory_calibration_override: self.graph_memory_calibration_override,
                stream_lease_guard: None,
                auxiliary_stream_permit_guard: None,
                query_service_admitted: false,
            }));
        }

        let mut builder = cudf::Runtime::builder();
        if let Some(device_ordinal) = self.device_ordinal {
            let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
                QueryError::from_cudf("nexus_query_engine.runtime.device", "failed to resolve CUDA device", error)
            })?;
            builder = builder.with_device(device);
        }

        let memory_policy = match self.memory_policy {
            MemoryResourcePolicy::ExternalRuntime => MemoryResourcePolicy::DefaultCuda,
            other => other,
        };

        match memory_policy {
            MemoryResourcePolicy::DefaultCuda | MemoryResourcePolicy::ExternalRuntime => {},
            MemoryResourcePolicy::Pool { initial_pool_size, maximum_pool_size } => {
                builder = builder.with_pool_memory_resource(initial_pool_size, maximum_pool_size);
            },
        }

        let runtime = builder.build().map_err(|error| {
            QueryError::from_cudf("nexus_query_engine.runtime.build", "failed to initialize cudf runtime", error)
        })?;
        let service_stream_pool_size = self
            .service_stream_pool_size
            .unwrap_or_else(|| default_service_stream_pool_size_for_device(runtime.device()));
        let service_stream_pool = match self.service_stream_pool {
            Some(pool) => {
                validate_service_stream_pool_device(&pool, runtime.device())?;
                pool
            },
            None => ServiceStreamPool::for_device(runtime.device(), service_stream_pool_size)?,
        };
        let gpu_execution_pool =
            resolve_gpu_execution_pool(gpu_execution_pool, runtime.device(), service_stream_pool.pool_size())?;
        let runtime = Arc::new(runtime);
        let grant_reservation_arena = create_grant_reservation_arena(runtime.as_ref(), self.memory_grant.as_ref())?;
        Ok(ExecutionContext::from_parts(ExecutionContextParts {
            service_epoch: None,
            query_epoch: None,
            query_anchor: None,
            io_lifecycle: self.io_lifecycle,
            query_id,
            trace_id,
            runtime_observation,
            runtime_observation_owner: None,
            runtime_execution_scope: None,
            runtime_stream_lease_scope: crate::observability::RuntimeStreamLeaseScope::default(),
            runtime,
            gpu_execution_pool,
            memory_policy,
            memory_control_policy: self.memory_control_policy,
            memory_grant: self.memory_grant,
            grant_reservation_arena,
            grant_memory_domain: None,
            attempt_allocation_domain: None,
            attempt_auxiliary_slots: None,
            admitted_capability_evidence: None,
            attempt_member_lease: None,
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_chunk_planning_mode: self.source_chunk_planning_mode,
            lakehouse_cache_fill_config,
            service_stream_pool,
            lakehouse_cache_fill_budget,
            lakehouse_cache_fill_concurrency,
            lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            object_store_control_runtime,
            source_object_stores,
            deployment_io_capability,
            cancellation: self.cancellation,
            stddev_samp_execution_mode: self.stddev_samp_execution_mode,
            aggregate_strategy_execution_options: self.aggregate_strategy_execution_options,
            source_chunk_planning_options: self.source_chunk_planning_options,
            graph_memory_calibration_override: self.graph_memory_calibration_override,
            stream_lease_guard: None,
            auxiliary_stream_permit_guard: None,
            query_service_admitted: false,
        }))
    }
}

impl RuntimeObservedExecutionContextBuilder {
    /// Rejects the owner-discarding build method and returns the supplied owner.
    pub fn build(self) -> std::result::Result<ExecutionContext, RuntimeObservationBuildError> {
        Err(RuntimeObservationBuildError::new(runtime_observation_owner_required_error(), self.observation))
    }

    /// Builds the context while returning the supplied owner on every failure.
    pub fn build_with_runtime_observation(
        mut self,
    ) -> std::result::Result<RuntimeObservedExecutionContext, RuntimeObservationBuildError> {
        if let Err(error) = self.builder.adopt_or_validate_runtime_observation_identity(&self.observation) {
            return Err(RuntimeObservationBuildError::new(error, self.observation));
        }
        self.builder.build_with_observation(self.observation)
    }
}

fn runtime_observation_owner_required_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.runtime_observation.owner",
        "enabled runtime observation requires an explicit owner lifetime",
    )
    .with_fact("reason", "runtime_observation_owner_required")
    .with_fact("required_builder_method", "build_with_runtime_observation")
}

fn runtime_observation_identity_error(
    reason: &'static str,
    identity_kind: &'static str,
    context_value: &str,
    observation_value: &str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.runtime_observation.identity",
        "execution context identity conflicts with the supplied runtime observation session",
    )
    .with_fact("reason", reason)
    .with_fact("identity_kind", identity_kind)
    .with_fact("context_identity", context_value)
    .with_fact("observation_identity", observation_value)
}

fn resolve_gpu_execution_pool(
    pool: Option<GpuExecutionPool>,
    runtime_device: cudf::DeviceId,
    default_worker_count: usize,
) -> Result<GpuExecutionPool> {
    let pool = match pool {
        Some(pool) => pool,
        None => GpuExecutionPool::shared_for_device(runtime_device.ordinal(), default_worker_count)?,
    };
    let snapshot = pool.snapshot();
    if snapshot.device_ordinal != runtime_device.ordinal() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_execution_pool.device",
            "GPU execution pool device must match the execution runtime device",
        )
        .with_fact("pool_device_ordinal", snapshot.device_ordinal)
        .with_fact("runtime_device_ordinal", runtime_device.ordinal()));
    }
    Ok(pool)
}

pub(in crate::runtime) fn validate_service_stream_pool_device(
    pool: &ServiceStreamPool,
    runtime_device: cudf::DeviceId,
) -> Result<()> {
    if pool.device() != runtime_device {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.service_stream_pool.device",
            "service stream pool device must match the execution runtime device",
        )
        .with_fact("pool_device_ordinal", pool.device().ordinal())
        .with_fact("runtime_device_ordinal", runtime_device.ordinal()));
    }
    Ok(())
}
