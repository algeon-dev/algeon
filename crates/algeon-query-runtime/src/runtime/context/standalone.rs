//! Standalone execution contexts, built without query-service admission.

use std::{
    error::Error as StdError,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use cudf::{QueryMemoryDomainOptions, Runtime as CudfRuntime};
use query_planning::{
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
};

use super::{
    super::{
        CacheFillByteBudget, GpuExecutionPool, LakehouseCacheFillConfig, MemoryControlPolicy, MemoryResourcePolicy,
        QueryExecutionConfig, ServiceStreamPool, SourceChunkPlanningMode, SourceSchedulerPolicy, SourceStorageProfile,
        cache_fill::{CacheFillConcurrencyGate, default_lakehouse_cache_fill_budget_bytes},
        gpu_execution_pool::RESERVED_LIFECYCLE_WORKERS,
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
/// admitted, so it carries no admission grant or attempt identity. Its device
/// budget is enforced by one immutable query allocation domain.
///
/// Without `runtime`, construction builds a cudf runtime on the default CUDA
/// device from `memory_policy`; with it, `memory_policy` is unused. Context
/// operations use the device manager's routing allocator under their domain
/// member, including when the supplied runtime owns a separate pool.
#[derive(Debug)]
pub struct StandaloneConfig {
    /// Runtime supplying the context's device, stream, and retained provenance.
    ///
    /// Context derivation preserves its device, stream and lifetime owners while
    /// replacing the effective allocator with the device manager's member
    /// routing resource. The original runtime and its allocator remain usable
    /// outside the context. A raw stream cannot belong to independent live
    /// domains; reuse one context or provide another stream for another query.
    pub runtime: Option<Arc<CudfRuntime>>,
    /// Memory resource of a runtime built by construction.
    pub memory_policy: MemoryResourcePolicy,
    /// Device budget and source chunking. The default budget is
    /// [`Self::DEFAULT_DEVICE_BUDGET_BYTES`] with the query service's default
    /// source chunking.
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

impl StandaloneConfig {
    /// Device budget of a standalone context that names none: large enough
    /// that a test fixture is not limited by it.
    pub const DEFAULT_DEVICE_BUDGET_BYTES: u64 = 64 * 1024 * 1024 * 1024;
}

impl Default for StandaloneConfig {
    fn default() -> Self {
        Self {
            runtime: None,
            memory_policy: MemoryResourcePolicy::default(),
            memory_control_policy: QueryExecutionConfig::default()
                .memory_policy_for_admitted_query(Self::DEFAULT_DEVICE_BUDGET_BYTES),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            source_storage_profiles: Vec::new(),
            service_stream_pool: None,
            service_stream_pool_size: None,
            lakehouse_cache_fill: LakehouseCacheFillConfig::default(),
            cancellation: ExecutionCancellation::default(),
            query_id: None,
            trace_id: None,
        }
    }
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
    query: StandaloneQuery,
    stream: StreamBinding,
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
        // One execution worker per service stream, beside the reserved
        // lifecycle workers.
        let gpu_execution_pool = GpuExecutionPool::shared_for_device(
            runtime.device().ordinal(),
            service_stream_pool.pool_size() + RESERVED_LIFECYCLE_WORKERS,
        )?;
        let operation = "query_engine.runtime.standalone.create_domain";
        let cap = usize::try_from(memory_control_policy.device_budget_bytes).map_err(|_| {
            QueryError::new(QueryErrorCode::InvalidPlan, operation, "standalone query cap exceeds usize")
        })?;
        static NEXT_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);
        let domain = runtime
            .device_memory_manager()
            .and_then(|manager| {
                manager.query_memory_domain(
                    NEXT_DOMAIN_ID.fetch_add(1, Ordering::Relaxed),
                    QueryMemoryDomainOptions::new(cap).with_ceiling_bytes(cap),
                )
            })
            .map_err(|error| cudf_query_error(operation, "failed to create standalone query domain", error))?;
        let query = StandaloneQuery {
            allocation_domain: domain,
            next_member_id: AtomicU64::new(1),
            memory_control_policy,
            cancellation,
            observation_owner: None,
        };
        let stream = query.bind_root(runtime)?;
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
                planning_policy: Default::default(),
            },
            query,
            stream,
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
            query: QueryScope::Standalone(Arc::new(StandaloneQuery { observation_owner, ..self.query })),
            stream: Arc::new(self.stream),
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use cudf::{
        CudfAllocationAttribution, ExecutionBudgetRequest, QueryMemoryDomain, Runtime, allocation_attribution,
        testing::RawHandleTestingExt,
    };

    use super::{ExecutionContext, QueryScope, StandaloneConfig};

    struct RetainedProbe(Arc<AtomicBool>);

    impl Drop for RetainedProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn domain(context: &ExecutionContext) -> QueryMemoryDomain {
        match &context.query {
            QueryScope::Standalone(query) => query.allocation_domain.clone(),
            QueryScope::Admitted(_) => panic!("the fixture creates a standalone context"),
        }
    }

    fn context(runtime: Arc<Runtime>, cap: u64) -> ExecutionContext {
        let mut config = StandaloneConfig { runtime: Some(runtime), ..StandaloneConfig::default() };
        config.memory_control_policy.device_budget_bytes = cap;
        ExecutionContext::standalone(config).unwrap()
    }

    #[test]
    fn standalone_pool_allocations_share_a_capped_member_and_outputs_retain_it() {
        let released = Arc::new(AtomicBool::new(false));
        let supplied = Arc::new(
            Runtime::builder()
                .with_pool_memory_resource(1024 * 1024, Some(16 * 1024 * 1024))
                .build()
                .unwrap()
                .with_retained_lifetime(Arc::new(RetainedProbe(Arc::clone(&released))))
                .unwrap(),
        );
        let context = context(Arc::clone(&supplied), 4096);
        let domain = domain(&context);
        let runtime = context.shared_runtime().unwrap();
        assert_eq!(runtime.device(), supplied.device());
        assert_eq!(runtime.stream().testing_raw_handle_value(), supplied.stream().testing_raw_handle_value());
        assert_ne!(runtime.binding_id(), supplied.binding_id());
        assert_eq!(
            runtime.memory_resource().testing_raw_handle_value(),
            runtime.device_memory_manager().unwrap().resource().testing_raw_handle_value()
        );
        assert_ne!(
            runtime.memory_resource().testing_raw_handle_value(),
            supplied.memory_resource().testing_raw_handle_value()
        );
        assert_eq!(context.attempt_allocation_domain_id(), None);
        assert!(context.memory_grant().unwrap().is_none());

        let first = context.execution_budget_token(ExecutionBudgetRequest::new(1)).unwrap();
        let second = context.execution_budget_token(ExecutionBudgetRequest::new(8192)).unwrap();
        assert_eq!(
            first.reservation_usage().unwrap().reservation_id,
            second.reservation_usage().unwrap().reservation_id
        );
        assert_eq!(first.member_limit_bytes().unwrap(), 4096);
        assert_eq!(second.member_limit_bytes().unwrap(), 4096);
        let output = runtime.series(&[7i32; 128]).unwrap();
        assert!(first.reservation_usage().unwrap().routed_total_allocations > 0);
        let error = runtime.series(&[9i32; 2048]).unwrap_err();
        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::ConfirmedQueryLimit);
        drop(first);
        drop(second);
        drop(runtime);
        drop(supplied);
        drop(context);

        assert_eq!(domain.usage().unwrap().active_members, 1);
        assert!(!released.load(Ordering::SeqCst));
        assert_eq!(output.to_vec_strict::<i32>().unwrap(), vec![7; 128]);
        drop(output);
        assert_eq!(domain.usage().unwrap().active_members, 0);
        assert!(released.load(Ordering::SeqCst));
    }

    #[test]
    fn standalone_work_stream_shares_the_aggregate_cap_and_detaches_with_its_context() {
        let context = context(Arc::new(Runtime::builder().build().unwrap()), 4096);
        let domain = domain(&context);
        let lease = context.service_stream_pool().acquire().unwrap();
        let work = context.with_stream_lease(&lease).unwrap();
        let root_runtime = context.shared_runtime().unwrap();
        let work_runtime = work.shared_runtime().unwrap();
        assert_ne!(root_runtime.stream().testing_raw_handle_value(), work_runtime.stream().testing_raw_handle_value());
        let root = root_runtime.series(&[3i32; 512]).unwrap();
        let output = work_runtime.series(&[5i32; 512]).unwrap();
        assert_eq!(domain.usage().unwrap().active_members, 2);
        let token = work.execution_budget_token(ExecutionBudgetRequest::new(1)).unwrap();
        assert_eq!(token.member_limit_bytes().unwrap(), 4096);
        assert!(token.reservation_usage().unwrap().routed_total_allocations > 0);
        let error = work_runtime.series(&[6i32; 128]).unwrap_err();
        assert_eq!(allocation_attribution(&error), CudfAllocationAttribution::ConfirmedQueryLimit);
        drop(token);
        drop(work_runtime);
        drop(work);
        let usage = domain.usage().unwrap();
        assert_eq!(usage.active_members, 1);
        assert_eq!(usage.current_allocations, 2, "the detached member's output stays charged to the domain");

        // The pooled stream joins the domain again while that output lives.
        let rebound = context.with_stream_lease(&lease).unwrap();
        assert_eq!(domain.usage().unwrap().active_members, 2);
        drop(rebound);
        assert_eq!(output.to_vec_strict::<i32>().unwrap(), vec![5; 512]);
        drop(output);
        assert_eq!(domain.usage().unwrap().current_allocations, 1);
        drop(lease);
        drop(root);
        drop(root_runtime);
        drop(context);
        assert_eq!(domain.usage().unwrap().active_members, 0);
    }
}
