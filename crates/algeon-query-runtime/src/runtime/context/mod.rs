//! Native execution context state and query-local runtime controls.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use cudf::{DeviceId, Runtime as CudfRuntime};
use query_planning::{
    admission::AttemptCapabilityEvidence,
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    mode::StatisticalAggregateExecutionMode,
    plan::PlanFingerprint,
    policy::PlanningPolicy,
};

use super::{
    GpuExecutionPool, GpuExecutorLease, MemoryControlPolicy, MemoryGrant, ServiceStreamPool, SourceChunkPlanningMode,
    SourceSchedulerPolicy,
    attempt_domain::AttemptStreamProviderRoute,
    cache::local_parquet::GpuLocalParquetSourceCache,
    gpu::{GrantIdentity, ledger::ResidentOwnerLease},
    io_lifecycle::{IoLifecycle, IoLifecycleLease},
    resident::{GpuResidentInputMap, GpuResultHandle},
    service_stream::StreamCleanupOutcome,
};
#[cfg(feature = "cugraph")]
use super::{allocator_correlation::CacheScopeCorrelationGuard, cache::resident_graph::GpuResidentGraphCache};
use crate::{
    metrics::{QueryId, TraceId},
    observability::{
        AttemptId, PlanExecutionId, RuntimeExecutionObservationScope, RuntimeObservationHandle,
        RuntimePlanExecutionLifecycle, RuntimeQueryLifecycle, RuntimeStreamLeaseScope, RuntimeTerminalOutcome,
        RuntimeTraceId, SealedRuntimeObservationSnapshot,
    },
};

mod admitted;
mod auxiliary_slots;
mod cancellation;
mod epoch;
mod lakehouse;
mod leases;
mod scope;
#[cfg(any(test, feature = "test-hooks"))]
mod standalone;
mod stream_pool;

pub(in crate::runtime) use admitted::AdmittedPlanIdentities;
pub(crate) use auxiliary_slots::{AttemptAuxiliaryPermit, AttemptAuxiliarySlots};
pub use cancellation::ExecutionCancellation;
pub(crate) use epoch::{
    QueryResourceAnchor, QueryResourceEpoch, QueryServiceEpoch, QueryServiceEpochUser, QueryServiceResourceEpoch,
};
use scope::QueryScope;
pub(in crate::runtime) use scope::{AdmittedAttempt, DeviceRuntime, PlanScope, StreamBinding};
#[cfg(any(test, feature = "test-hooks"))]
pub use standalone::{RuntimeObservationBuildError, RuntimeObservedExecutionContext, StandaloneConfig};
pub(crate) use stream_pool::QueryServiceStreamPool;

/// Immutable execution configuration shared by native query execution.
///
/// The context owns or shares the [`cudf::Runtime`] used for all device work in
/// a query. The runtime determines the CUDA device, stream, and memory resource;
/// the query engine layers its own chunk-planning, reservation, retry, and
/// boundary policies on top of that runtime. Cloning the context is cheap and
/// keeps the same runtime/cache handles.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    // Fields drop in declaration order, which attempt teardown depends on: the
    // plan scope holds the outward query anchor, and the query scope's grant
    // and domain clones go before the stream binding, whose root lease drop
    // runs the attempt's terminal cleanup (see `StreamBinding`).
    plan: Arc<PlanScope>,
    query: QueryScope,
    stream: Arc<StreamBinding>,
    device: Arc<DeviceRuntime>,
}

#[derive(Debug)]
pub(super) struct ActivatedDeferredResidentPhase {
    consumer_execution_scope_fingerprint: PlanFingerprint,
    inputs: Vec<ActivatedDeferredResidentInput>,
    capability_evidence: AttemptCapabilityEvidence,
    grant_identity: GrantIdentity,
}

#[derive(Debug)]
pub(super) struct ActivatedDeferredResidentInput {
    producer_execution_scope_fingerprint: PlanFingerprint,
    input_name: String,
    input_schema: SchemaRef,
    input_memory_identity: cudf::DataFrameMemoryIdentity,
    input_row_count: u64,
}

impl ActivatedDeferredResidentPhase {
    fn validate_inputs(&self, inputs: &GpuResidentInputMap, grant: &MemoryGrant) -> Result<()> {
        for activated in &self.inputs {
            let input = inputs.get(&activated.input_name).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InternalInvariantViolation,
                    "query_engine.runtime.context",
                    "the activated deferred resident input is missing",
                )
                .with_fact("reason", "deferred_resident_input_missing")
                .with_fact("input_name", &activated.input_name)
            })?;
            activated.validate(input, grant, self)?;
        }
        Ok(())
    }
}

impl ActivatedDeferredResidentInput {
    fn validate(
        &self,
        input: &GpuResultHandle,
        grant: &MemoryGrant,
        phase: &ActivatedDeferredResidentPhase,
    ) -> Result<()> {
        let row_count = u64::try_from(input.frame().num_rows()).map_err(|_| {
            QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.context",
                "the materialized graph input row count does not fit the admission contract",
            )
            .with_fact("reason", "deferred_graph_input_row_count_overflow")
        })?;
        let physical_bytes = input.device_allocated_size_bytes()?;
        let retained_by_grant = input.retained_owner_identity_for(grant).is_some();
        if phase.grant_identity != grant.identity()
            || input.schema() != &self.input_schema
            || input.frame().memory_identity() != self.input_memory_identity
            || input.producer_execution_scope() != Some(self.producer_execution_scope_fingerprint)
            || row_count != self.input_row_count
            || (physical_bytes > 0 && !retained_by_grant)
        {
            return Err(QueryError::new(
                QueryErrorCode::InternalInvariantViolation,
                "query_engine.runtime.context",
                "the GPU-resident graph input differs from the frame that activated its admitted phase",
            )
            .with_fact("reason", "deferred_graph_input_identity_mismatch")
            .with_fact("input_name", &self.input_name)
            .with_fact("input_row_count", row_count)
            .with_fact("activated_row_count", self.input_row_count)
            .with_fact("input_device_bytes", physical_bytes));
        }
        Ok(())
    }
}

/// One native execution observation boundary selected from context ownership.
///
/// This type exposes plan recording and terminal completion without exposing
/// the observation owner stored by the execution context.
#[doc(hidden)]
#[derive(Debug)]
pub struct RuntimeExecutionObservationLifecycle {
    inner: RuntimeExecutionObservationLifecycleInner,
}

#[derive(Debug)]
enum RuntimeExecutionObservationLifecycleInner {
    Query(RuntimeQueryLifecycle),
    Plan(RuntimePlanExecutionLifecycle),
}

impl RuntimeExecutionObservationLifecycle {
    fn for_context(context: &ExecutionContext, plan_execution_id: PlanExecutionId) -> Self {
        // An admitted attempt seals once for the attempt, so each plan
        // finishes only its own execution; a standalone owner seals the query.
        let inner = match &context.query {
            QueryScope::Admitted(admitted) => RuntimeExecutionObservationLifecycleInner::Plan(
                RuntimePlanExecutionLifecycle::from_attempt(&admitted.attempt, plan_execution_id),
            ),
            QueryScope::Standalone(query) => match &query.observation_owner {
                Some(owner) => RuntimeExecutionObservationLifecycleInner::Query(RuntimeQueryLifecycle::from_owner(
                    owner.clone(),
                    plan_execution_id,
                )),
                None => RuntimeExecutionObservationLifecycleInner::Plan(RuntimePlanExecutionLifecycle::from_handle(
                    context.plan.runtime_observation.clone(),
                    plan_execution_id,
                )),
            },
        };
        Self { inner }
    }

    /// Returns the record-only scope for one native plan invocation.
    pub fn execution_scope(&self) -> RuntimeExecutionObservationScope {
        match &self.inner {
            RuntimeExecutionObservationLifecycleInner::Query(lifecycle) => lifecycle.execution_scope(),
            RuntimeExecutionObservationLifecycleInner::Plan(lifecycle) => lifecycle.execution_scope(),
        }
    }

    /// Completes this plan or standalone query according to context ownership.
    pub fn finish(self, outcome: RuntimeTerminalOutcome) -> Option<SealedRuntimeObservationSnapshot> {
        match self.inner {
            RuntimeExecutionObservationLifecycleInner::Query(lifecycle) => lifecycle.finish(outcome),
            RuntimeExecutionObservationLifecycleInner::Plan(lifecycle) => {
                lifecycle.finish(outcome);
                None
            },
        }
    }
}

impl ExecutionContext {
    /// Assembles the root context of an admitted attempt.
    pub(in crate::runtime) fn admitted_root(
        device: Arc<DeviceRuntime>,
        attempt: AdmittedAttempt,
        stream: StreamBinding,
        plan: PlanScope,
    ) -> Self {
        Self { plan: Arc::new(plan), query: QueryScope::Admitted(Arc::new(attempt)), stream: Arc::new(stream), device }
    }

    fn admitted(&self) -> Option<&AdmittedAttempt> {
        match &self.query {
            QueryScope::Admitted(attempt) => Some(attempt),
            QueryScope::Standalone(_) => None,
        }
    }

    /// Returns a clone whose plan scope `update` changed.
    fn with_plan(&self, update: impl FnOnce(&mut PlanScope)) -> Self {
        let mut plan = PlanScope::clone(&self.plan);
        update(&mut plan);
        Self { plan: Arc::new(plan), ..self.clone() }
    }

    /// Returns a clone that runs on `stream`, keeping every other scope.
    fn with_stream(&self, stream: StreamBinding) -> Self {
        Self { stream: Arc::new(stream), ..self.clone() }
    }

    /// Attaches one counted query owner to a clone of the internal root template.
    pub(super) fn with_query_resource_anchor(&self, anchor: Arc<QueryResourceAnchor>) -> Self {
        debug_assert!(
            self.admitted().is_some_and(|attempt| anchor.belongs_to(&attempt.query_epoch)),
            "query resource anchor must belong to the context query epoch"
        );
        self.with_plan(|plan| plan.query_anchor = Some(anchor))
    }

    fn acquire_resource_epoch(&self, operation: &'static str) -> Result<Option<Arc<QueryServiceResourceEpoch>>> {
        let Some(attempt) = self.admitted() else {
            return Ok(None);
        };
        QueryServiceResourceEpoch::acquire(&attempt.service_epoch, self.plan.query_anchor.as_ref(), operation).map(Some)
    }

    /// Returns the cudf runtime used for all GPU operations.
    pub fn runtime(&self) -> Result<CudfRuntime> {
        let operation = "query_engine.runtime.context.runtime";
        // Validate the query/service epoch at acquisition time, but do not
        // embed it in GPU allocations. Escaping allocations are owned by a
        // ResidentOwnerLease; unowned residual allocations must reach final
        // validation so cleanup can fail closed instead of keeping the query
        // epoch alive forever and preventing B-to-R handoff.
        drop(self.acquire_resource_epoch(operation)?);
        Ok(self.stream.runtime().as_ref().clone())
    }

    /// Acquire permission for one file-backed native I/O call.
    ///
    /// Standalone library contexts without a server lifecycle remain
    /// unrestricted. Server contexts carry one generation gate and reject
    /// acquisitions after graceful shutdown begins.
    pub fn acquire_io_lease(&self, operation: &'static str) -> Result<Option<IoLifecycleLease>> {
        self.plan.io_lifecycle.as_ref().map(|lifecycle| lifecycle.acquire(operation)).transpose()
    }

    /// Attach a server-generation I/O lifecycle to this context clone.
    pub fn with_io_lifecycle(self, lifecycle: IoLifecycle) -> Self {
        self.with_plan(|plan| plan.io_lifecycle = Some(lifecycle))
    }

    /// Returns a shared handle to the runtime.
    ///
    /// The caller receives the same runtime object, not a new runtime with
    /// copied configuration.
    pub fn shared_runtime(&self) -> Result<Arc<CudfRuntime>> {
        self.runtime().map(Arc::new)
    }

    /// Returns an opaque lifecycle-bound executor for synchronous GPU work on
    /// this context's device.
    ///
    /// The lease's query-execution job follows this context's own claim on
    /// the attempt: a root context runs it as root-class work, while an
    /// auxiliary fragment context -- a second native island of the same
    /// attempt -- runs it as auxiliary work backed by the attempt auxiliary
    /// permit that context already holds. Root capacity is one job per
    /// admitted attempt, so a second island charging it would fail where it
    /// has a reserved auxiliary slot of its own.
    pub fn gpu_executor(&self) -> Result<GpuExecutorLease> {
        self.gpu_execution_pool().map(if matches!(*self.stream, StreamBinding::AttemptWork { owned: Some(_), .. }) {
            GpuExecutorLease::from_pool_for_auxiliary_fragment
        } else {
            GpuExecutorLease::from_pool
        })
    }

    pub(crate) fn gpu_execution_pool(&self) -> Result<GpuExecutionPool> {
        let resource_epoch = self.acquire_resource_epoch("query_engine.runtime.context.gpu_execution_pool")?;
        Ok(match resource_epoch {
            Some(resource_epoch) => self.device.gpu_execution_pool.with_resource_epoch(resource_epoch),
            None => self.device.gpu_execution_pool.clone(),
        })
    }

    /// Returns the admitted attempt's fork-stream route, when this context
    /// carries one. Each lease derived from it holds one auxiliary fork of the
    /// route alone, up to [`AttemptStreamProviderRoute::lane_capacity`];
    /// direct contexts have no route and run serially.
    #[doc(hidden)]
    pub fn attempt_stream_provider_route(&self) -> Option<Arc<AttemptStreamProviderRoute>> {
        self.stream.route().cloned()
    }

    /// Returns the query-wide diagnostic correlation id.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.plan.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.plan.trace_id.as_ref()
    }

    /// Returns the engine-minted attempt identity carried by an admitted context.
    ///
    /// This remains absent for direct contexts because they are not part of a
    /// public execution-attempt lifecycle.
    #[doc(hidden)]
    pub fn attempt_id(&self) -> Option<AttemptId> {
        self.admitted().map(|attempt| attempt.attempt_id)
    }

    /// Opens one exact allocator-event window for an exclusive resident graph
    /// checkout. The caller must finish the returned guard after cache-runtime
    /// temporaries are destroyed and before returning the checkout.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn begin_cache_scope_allocator_correlation(
        &self,
        scope: Arc<cudf::CacheAllocationScope>,
    ) -> Option<CacheScopeCorrelationGuard> {
        self.admitted()?.cache_scope_allocator_correlation.begin(scope)
    }

    /// Returns the shared recording handle without terminal-seal authority.
    pub fn runtime_observation(&self) -> &RuntimeObservationHandle {
        &self.plan.runtime_observation
    }

    /// Opens one native execution boundary without exposing the context's
    /// observation owner.
    #[doc(hidden)]
    pub fn begin_runtime_execution_observation(
        &self,
        plan_execution_id: PlanExecutionId,
    ) -> RuntimeExecutionObservationLifecycle {
        RuntimeExecutionObservationLifecycle::for_context(self, plan_execution_id)
    }

    /// Returns the active native plan-execution scope.
    pub fn runtime_execution_scope(&self) -> Option<&RuntimeExecutionObservationScope> {
        self.plan.runtime_execution_scope.as_ref()
    }

    /// Returns a clone carrying one invocation's recording handle.
    ///
    /// Seal authority remains with the execution lifecycle that supplied the
    /// handle. Identity is validated before the clone can reach workers.
    #[doc(hidden)]
    pub fn with_runtime_observation_handle(&self, observation: RuntimeObservationHandle) -> Result<Self> {
        self.validate_observation_identity(&observation)?;
        Ok(self.with_plan(|plan| plan.runtime_observation = observation))
    }

    /// Returns a clone scoped to one native plan invocation.
    #[doc(hidden)]
    pub fn with_runtime_execution_scope(&self, scope: RuntimeExecutionObservationScope) -> Result<Self> {
        self.validate_observation_identity(scope.observation())?;
        Ok(self.with_plan(|plan| {
            plan.runtime_observation = scope.observation().clone();
            plan.runtime_stream_lease_scope.plan_execution_id = Some(scope.plan_execution_id().clone());
            plan.runtime_execution_scope = Some(scope);
        }))
    }

    fn validate_observation_identity(&self, observation: &RuntimeObservationHandle) -> Result<()> {
        if let Some(query_id) = &self.plan.query_id {
            validate_runtime_observation_query_id(observation, query_id)?;
        }
        if let Some(trace_id) = &self.plan.trace_id {
            validate_runtime_observation_trace_id(observation, trace_id)?;
        }
        Ok(())
    }

    /// Returns a clone carrying explicit GPU job/worker correlation.
    pub fn with_runtime_gpu_stream_scope(&self, scope: RuntimeStreamLeaseScope) -> Self {
        self.with_plan(|plan| {
            plan.runtime_stream_lease_scope.gpu_job_id = scope.gpu_job_id;
            plan.runtime_stream_lease_scope.gpu_worker_id = scope.gpu_worker_id;
        })
    }

    /// Returns a cloned context with a diagnostic query id consistent with observation identity.
    pub fn with_query_id(&self, query_id: impl Into<QueryId>) -> Result<Self> {
        let query_id = query_id.into();
        validate_runtime_observation_query_id(&self.plan.runtime_observation, &query_id)?;
        Ok(self.with_plan(|plan| plan.query_id = Some(query_id)))
    }

    /// Returns a cloned context with a trace id consistent with observation identity.
    pub fn with_trace_id(&self, trace_id: impl Into<TraceId>) -> Result<Self> {
        let trace_id = trace_id.into();
        validate_runtime_observation_trace_id(&self.plan.runtime_observation, &trace_id)?;
        Ok(self.with_plan(|plan| plan.trace_id = Some(trace_id)))
    }

    /// Returns whether this context is backed by a QueryService admission slot.
    pub fn is_query_service_admitted(&self) -> bool {
        self.admitted().is_some()
    }

    /// Returns the backend-owned local-Parquet cache for this context's
    /// ordinal, if its device profile enabled resident caching.
    #[doc(hidden)]
    pub fn resident_source_cache(&self) -> Option<GpuLocalParquetSourceCache> {
        self.device.resident_source_cache.clone()
    }

    /// Returns the backend-owned resident graph cache for this context's
    /// ordinal, if its device profile enabled resident caching.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn resident_graph_cache(&self) -> Option<GpuResidentGraphCache> {
        self.device.resident_graph_cache.clone()
    }

    pub(super) fn install_root_stream_cleanup_completion(
        &self,
        before_quiescence: impl FnOnce(StreamCleanupOutcome) + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> bool {
        let StreamBinding::AttemptRoot { lease, .. } = &*self.stream else {
            return false;
        };
        lease.install_cleanup_completion(before_quiescence, after_quiescence)
    }

    pub(super) fn has_root_stream_lease(&self) -> bool {
        matches!(*self.stream, StreamBinding::AttemptRoot { .. })
    }

    /// Returns the cooperative cancellation flag for this execution.
    pub fn cancellation(&self) -> &ExecutionCancellation {
        match &self.query {
            QueryScope::Standalone(query) => &query.cancellation,
            QueryScope::Admitted(attempt) => &attempt.cancellation,
        }
    }

    /// Fails with a structured cancellation error if the execution was cancelled.
    pub fn check_cancellation(&self, operation: &'static str) -> Result<()> {
        self.check_service_epoch(operation)?;
        self.cancellation().check(operation)
    }

    fn check_service_epoch(&self, operation: &'static str) -> Result<()> {
        let Some(attempt) = self.admitted() else {
            return Ok(());
        };
        attempt.service_epoch.check_open(operation)?;
        attempt.query_epoch.check_open(operation)
    }

    /// CUDA device that runs this context's work.
    pub fn device(&self) -> DeviceId {
        self.stream.runtime().device()
    }

    /// Returns the engine-level memory control policy.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        match &self.query {
            QueryScope::Standalone(query) => query.memory_control_policy,
            QueryScope::Admitted(attempt) => attempt.memory_control_policy,
        }
    }

    /// Returns the soft governor grant carried by this context, when present.
    pub fn memory_grant(&self) -> Result<Option<MemoryGrant>> {
        let Some(attempt) = self.admitted() else {
            return Ok(None);
        };
        self.admitted_memory_grant(attempt, "query_engine.runtime.context.memory_grant").map(Some)
    }

    fn admitted_memory_grant(&self, attempt: &AdmittedAttempt, operation: &'static str) -> Result<MemoryGrant> {
        let resource_epoch = self.acquire_resource_epoch(operation)?;
        Ok(match resource_epoch {
            Some(resource_epoch) => attempt.memory_grant.with_resource_epoch(resource_epoch),
            None => attempt.memory_grant.clone(),
        })
    }

    #[doc(hidden)]
    pub fn retain_resident_output_owner(&self, bytes: u64) -> Result<Option<ResidentOwnerLease>> {
        match self.memory_grant()? {
            Some(grant) => grant.retain_resident_owner(bytes),
            None => Ok(None),
        }
    }

    /// Shares this stream's member for a budgeted execution stage.
    ///
    /// The stage request cannot change the immutable query-domain cap.
    pub fn execution_budget_token(&self, request: cudf::ExecutionBudgetRequest) -> Result<cudf::ExecutionBudgetToken> {
        let runtime = self.runtime()?;
        Ok(cudf::ExecutionBudgetToken::from_shared_lease(&runtime, request, self.stream.member().clone()))
    }

    /// Returns the immutable domain id for this admitted attempt.
    pub fn attempt_allocation_domain_id(&self) -> Option<u64> {
        self.admitted().map(|attempt| attempt.allocation_domain.domain_id())
    }

    /// Immutable grant identity owned by this admitted attempt domain.
    #[cfg(test)]
    pub(crate) fn attempt_allocation_grant_identity(&self) -> Option<GrantIdentity> {
        self.admitted().map(|attempt| attempt.allocation_domain.grant_identity())
    }

    /// Samples the allocation-domain current and peak usage.
    pub fn attempt_allocation_domain_usage(&self) -> Result<Option<cudf::QueryMemoryDomainUsage>> {
        self.admitted().map(|attempt| attempt.allocation_domain.usage()).transpose()
    }

    /// Returns the device budget used by reservation tracking.
    pub fn device_budget_bytes(&self) -> u64 {
        self.memory_control_policy().device_budget_bytes
    }

    /// Returns source scheduling controls for bounded source execution.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.device.source_scheduler_policy
    }

    /// Returns the source chunk planning mode.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.device.source_chunk_planning_mode
    }
    /// Returns the selected execution contract for floating statistical aggregates.
    pub fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.device.statistical_aggregate_execution_mode
    }

    /// Returns whether this query preserves DataFusion input order for floating
    /// statistical aggregates.
    ///
    /// Native execution derives this per query from the aggregate execution
    /// mode and the plan shape, then carries it on every context clone so
    /// source-read workers on spawned threads observe the same scope as the
    /// caller thread.
    pub fn datafusion_order_tracking(&self) -> bool {
        self.plan.datafusion_order_tracking
    }

    /// Attaches the per-query DataFusion exact-order scope to this context clone.
    pub fn with_datafusion_order_tracking(self, enabled: bool) -> Self {
        self.with_plan(|plan| plan.datafusion_order_tracking = enabled)
    }

    /// Returns grouped aggregate strategy execution policy.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.device.aggregate_strategy_execution_options
    }

    /// Join and dynamic-filter thresholds a plan executed without an
    /// admission artifact is admitted under.
    pub fn planning_policy(&self) -> PlanningPolicy {
        self.device.planning_policy
    }

    /// Returns expert local-throughput source chunk planning policy.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.device.source_chunk_planning_options
    }
}

/// The service memory policy an admitted attempt runs under: its device budget
/// becomes the grant's.
pub(in crate::runtime) fn memory_control_policy_for_grant(
    policy: MemoryControlPolicy,
    grant: &MemoryGrant,
) -> MemoryControlPolicy {
    MemoryControlPolicy { device_budget_bytes: grant.budget_bytes(), ..policy }
}

fn validate_runtime_observation_query_id(observation: &RuntimeObservationHandle, query_id: &QueryId) -> Result<()> {
    let Some(observation_query_id) = observation.query_id() else {
        return Ok(());
    };
    if observation_query_id.as_str() == query_id.as_str() {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.runtime_observation.identity",
        "execution context query id conflicts with its runtime observation session",
    )
    .with_fact("reason", "runtime_observation_query_id_mismatch")
    .with_fact("context_query_id", query_id)
    .with_fact("observation_query_id", observation_query_id))
}

fn validate_runtime_observation_trace_id(observation: &RuntimeObservationHandle, trace_id: &TraceId) -> Result<()> {
    if !observation.is_enabled() {
        return Ok(());
    }
    let observation_trace_id = observation.trace_id().map(RuntimeTraceId::as_str);
    if observation_trace_id == Some(trace_id.as_str()) {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.runtime_observation.identity",
        "execution context trace id conflicts with its runtime observation session",
    )
    .with_fact("reason", "runtime_observation_trace_id_mismatch")
    .with_fact("context_trace_id", trace_id)
    .with_fact("observation_trace_id", observation_trace_id.unwrap_or("none")))
}

pub(in crate::runtime) fn validate_service_stream_pool_device(
    pool: &ServiceStreamPool,
    runtime_device: DeviceId,
) -> Result<()> {
    if pool.device() != runtime_device {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.service_stream_pool.device",
            "service stream pool device must match the execution runtime device",
        )
        .with_fact("pool_device_ordinal", pool.device().ordinal())
        .with_fact("runtime_device_ordinal", runtime_device.ordinal()));
    }
    Ok(())
}
