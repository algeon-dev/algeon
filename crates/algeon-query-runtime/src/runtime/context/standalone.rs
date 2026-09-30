//! Standalone execution contexts, built without query-service admission.

use std::{
    error::Error as StdError,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use cudf::Runtime as CudfRuntime;
use query_planning::{
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
};

use super::{
    super::{
        CacheFillByteBudget, GpuExecutionPool, LakehouseCacheFillConfig, MemoryControlPolicy, MemoryResourcePolicy,
        ServiceStreamPool, SourceChunkPlanningMode, SourceSchedulerPolicy, SourceStorageProfile,
        cache_events::LakehouseCacheFillEventSink,
        cache_fill::{CacheFillConcurrencyGate, default_lakehouse_cache_fill_budget_bytes},
        object_store::{SourceObjectStoreRegistry, default_object_store_control_runtime},
        service_stream::{default_service_stream_pool_size_for_device, validate_service_stream_pool_size},
    },
    DeviceRuntime, ExecutionCancellation, ExecutionContext, PlanScope, QueryScope, StreamBinding,
    scope::StandaloneQuery,
    validate_service_stream_pool_device,
};
use crate::{
    metrics::{QueryId, TraceId},
    native_error::cudf_query_error,
    observability::{
        RuntimeObservationHandle, RuntimeObservationOwner, RuntimeObservationSession, RuntimeStreamLeaseScope,
        RuntimeTerminalOutcome,
    },
};

/// Settings for a standalone [`ExecutionContext`]: one that no query service
/// admitted, so it carries no grant, allocation domain, or attempt identity.
///
/// Without `runtime`, construction builds a cudf runtime on the default CUDA
/// device from `memory_policy`; with it, `memory_policy` is unused.
#[derive(Debug, Default)]
pub struct StandaloneConfig {
    /// Runtime shared with the caller; its stream, device, and allocator run
    /// every GPU operation of the context.
    pub runtime: Option<Arc<CudfRuntime>>,
    /// Memory resource of a runtime built by construction.
    pub memory_policy: MemoryResourcePolicy,
    pub memory_control_policy: MemoryControlPolicy,
    pub source_scheduler_policy: SourceSchedulerPolicy,
    pub source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Expert policy for local-throughput source chunk planning.
    pub source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Object-store profiles used by remote reads and cache fills.
    pub source_storage_profiles: Vec<SourceStorageProfile>,
    /// Pool for bounded source-read workers. `None` uses the process-global
    /// pool of `service_stream_pool_size` streams on the runtime's device.
    pub service_stream_pool: Option<ServiceStreamPool>,
    /// Size of the process-global pool; unused when `service_stream_pool` is set.
    pub service_stream_pool_size: Option<usize>,
    /// Background lakehouse cache-fill byte budget, operation cap, and acquire timeout.
    pub lakehouse_cache_fill: LakehouseCacheFillConfig,
    pub cancellation: ExecutionCancellation,
    /// Query-wide diagnostic correlation id.
    pub query_id: Option<QueryId>,
    /// Parent trace id.
    pub trace_id: Option<TraceId>,
}

/// Standalone execution context paired with its runtime observation owner.
///
/// The context retains the owner internally so native execution can select the
/// correct query or plan completion boundary. Callers that need to inspect or
/// finish the standalone observation retain the same shared owner here.
#[derive(Debug)]
pub struct RuntimeObservedExecutionContext {
    context: ExecutionContext,
    observation: RuntimeObservationOwner,
}

impl RuntimeObservedExecutionContext {
    /// Builds a standalone context recorded by `observation`.
    ///
    /// The session's query and trace ids fill identities `config` leaves unset
    /// and must equal the ones it sets. Every failure returns the session.
    pub fn standalone(
        mut config: StandaloneConfig,
        observation: RuntimeObservationSession,
    ) -> std::result::Result<Self, RuntimeObservationBuildError> {
        if let Err(error) = adopt_or_validate_runtime_observation_identity(&mut config, &observation) {
            return Err(RuntimeObservationBuildError::new(error, observation));
        }
        let scopes = match StandaloneScopes::new(config) {
            Ok(scopes) => scopes,
            Err(error) => return Err(RuntimeObservationBuildError::new(error, observation)),
        };
        let handle = observation.handle();
        let owner = RuntimeObservationOwner::from_session(observation);
        Ok(Self { context: scopes.into_context(handle, Some(owner.clone())), observation: owner })
    }

    /// Returns the execution context carrying the observation lifecycle state.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }

    /// Returns the owner-only terminal-seal capability.
    pub fn observation(&self) -> &RuntimeObservationOwner {
        &self.observation
    }

    /// Returns the execution context together with its shared owner.
    pub fn into_parts(self) -> (ExecutionContext, RuntimeObservationOwner) {
        (self.context, self.observation)
    }
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

impl ExecutionContext {
    /// Builds a standalone context without runtime observation.
    ///
    /// Production contexts come from query-service admission; tests and
    /// tools use this to run native plans directly.
    pub fn standalone(config: StandaloneConfig) -> Result<Self> {
        Ok(StandaloneScopes::new(config)?.into_context(RuntimeObservationHandle::disabled(), None))
    }
}

fn adopt_or_validate_runtime_observation_identity(
    config: &mut StandaloneConfig,
    observation: &RuntimeObservationSession,
) -> Result<()> {
    let Some(observation_query_id) = observation.query_id() else {
        return Ok(());
    };
    match &config.query_id {
        Some(query_id) if query_id.as_str() != observation_query_id.as_str() => {
            return Err(runtime_observation_identity_error(
                "runtime_observation_query_id_mismatch",
                "query_id",
                query_id.as_str(),
                observation_query_id.as_str(),
            ));
        },
        Some(_) => {},
        None => config.query_id = Some(QueryId::new(observation_query_id.as_str())),
    }

    match (&config.trace_id, observation.trace_id()) {
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
            config.trace_id = Some(TraceId::new(observation_trace_id.as_str()));
            Ok(())
        },
    }
}

/// A standalone context's validated scopes, before its observation is bound.
struct StandaloneScopes {
    device: DeviceRuntime,
    runtime: Arc<CudfRuntime>,
    memory_control_policy: MemoryControlPolicy,
    cancellation: ExecutionCancellation,
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
}

impl StandaloneScopes {
    fn new(config: StandaloneConfig) -> Result<Self> {
        let StandaloneConfig {
            runtime,
            memory_policy,
            memory_control_policy,
            source_scheduler_policy,
            source_chunk_planning_mode,
            source_chunk_planning_options,
            source_storage_profiles,
            service_stream_pool,
            service_stream_pool_size,
            lakehouse_cache_fill: lakehouse_cache_fill_config,
            cancellation,
            query_id,
            trace_id,
        } = config;
        memory_control_policy.validate()?;
        source_scheduler_policy.validate()?;
        source_chunk_planning_options.validate()?;
        if let Some(service_stream_pool_size) = service_stream_pool_size {
            validate_service_stream_pool_size(service_stream_pool_size)?;
        }
        lakehouse_cache_fill_config.validate()?;
        let lakehouse_cache_fill_budget = CacheFillByteBudget::new(
            lakehouse_cache_fill_config.budget_bytes.unwrap_or_else(default_lakehouse_cache_fill_budget_bytes),
        )?;
        let lakehouse_cache_fill_concurrency =
            CacheFillConcurrencyGate::new(lakehouse_cache_fill_config.max_concurrent_ops)?;
        let object_store_control_runtime = default_object_store_control_runtime()?;
        let source_object_stores = Arc::new(SourceObjectStoreRegistry::with_profiles(source_storage_profiles)?);
        let runtime = match runtime {
            Some(runtime) => runtime,
            None => Arc::new(build_runtime(memory_policy)?),
        };
        let service_stream_pool = match service_stream_pool {
            Some(pool) => {
                validate_service_stream_pool_device(&pool, runtime.device())?;
                pool
            },
            None => ServiceStreamPool::for_device(
                runtime.device(),
                service_stream_pool_size
                    .unwrap_or_else(|| default_service_stream_pool_size_for_device(runtime.device())),
            )?,
        };
        let gpu_execution_pool =
            GpuExecutionPool::shared_for_device(runtime.device().ordinal(), service_stream_pool.pool_size())?;
        Ok(Self {
            device: DeviceRuntime {
                gpu_execution_pool,
                service_stream_pool,
                resident_source_cache: None,
                #[cfg(feature = "cugraph")]
                resident_graph_cache: None,
                lakehouse_cache_fill_config,
                lakehouse_cache_fill_budget,
                lakehouse_cache_fill_concurrency,
                object_store_control_runtime,
                source_object_stores,
                source_scheduler_policy,
                source_chunk_planning_mode,
                source_chunk_planning_options,
                statistical_aggregate_execution_mode: Default::default(),
                aggregate_strategy_execution_options: Default::default(),
            },
            runtime,
            memory_control_policy,
            cancellation,
            query_id,
            trace_id,
        })
    }

    fn into_context(
        self,
        runtime_observation: RuntimeObservationHandle,
        observation_owner: Option<RuntimeObservationOwner>,
    ) -> ExecutionContext {
        ExecutionContext {
            plan: Arc::new(PlanScope::root(
                self.query_id,
                self.trace_id,
                runtime_observation,
                RuntimeStreamLeaseScope::default(),
            )),
            query: QueryScope::Standalone(Arc::new(StandaloneQuery {
                memory_control_policy: self.memory_control_policy,
                cancellation: self.cancellation,
                observation_owner,
                lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            })),
            stream: Arc::new(StreamBinding::Standalone { runtime: self.runtime }),
            device: Arc::new(self.device),
        }
    }
}

fn build_runtime(memory_policy: MemoryResourcePolicy) -> Result<CudfRuntime> {
    let builder = match memory_policy {
        MemoryResourcePolicy::DefaultCuda | MemoryResourcePolicy::ExternalRuntime => CudfRuntime::builder(),
        MemoryResourcePolicy::Pool { initial_pool_size, maximum_pool_size } => {
            CudfRuntime::builder().with_pool_memory_resource(initial_pool_size, maximum_pool_size)
        },
    };
    builder
        .build()
        .map_err(|error| cudf_query_error("query_engine.runtime.build", "failed to initialize cudf runtime", error))
}

fn runtime_observation_identity_error(
    reason: &'static str,
    identity_kind: &'static str,
    context_value: &str,
    observation_value: &str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.runtime_observation.identity",
        "execution context identity conflicts with the supplied runtime observation session",
    )
    .with_fact("reason", reason)
    .with_fact("identity_kind", identity_kind)
    .with_fact("context_identity", context_value)
    .with_fact("observation_identity", observation_value)
}
