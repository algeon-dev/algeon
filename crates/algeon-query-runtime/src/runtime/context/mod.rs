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
};
use tokio::runtime::Runtime as TokioRuntime;

use super::{
    CacheFillByteBudget, GpuExecutionPool, GpuExecutorLease, LakehouseCacheFillConfig, MemoryControlPolicy,
    MemoryGrant, ServiceStreamPool, SourceChunkPlanningMode, SourceSchedulerPolicy, StreamLease,
    allocator_correlation::AttemptCacheScopeAllocatorCorrelation,
    attempt_domain::{AttemptAllocationDomain, AttemptStreamProviderRoute},
    cache::local_parquet::GpuLocalParquetSourceCache,
    cache_events::LakehouseCacheFillEventSink,
    cache_fill::CacheFillConcurrencyGate,
    default_service_stream_pool_size_for_device,
    gpu::{GrantIdentity, ledger::ResidentOwnerLease},
    io_lifecycle::{IoLifecycle, IoLifecycleLease},
    object_store::{SourceObjectStoreRegistry, default_object_store_control_runtime},
    resident::{GpuResidentInputMap, GpuResultHandle},
    service_stream::StreamCleanupOutcome,
};
#[cfg(feature = "cugraph")]
use super::{allocator_correlation::CacheScopeCorrelationGuard, cache::resident_graph::GpuResidentGraphCache};
use crate::{
    metrics::{QueryId, TraceId},
    native_error::cudf_query_error,
    observability::{
        AttemptId, PlanExecutionId, RuntimeExecutionObservationScope, RuntimeObservationHandle,
        RuntimeObservationOwner, RuntimePlanExecutionLifecycle, RuntimeQueryLifecycle, RuntimeStreamLeaseScope,
        RuntimeTerminalOutcome, RuntimeTraceId, SealedRuntimeObservationSnapshot,
    },
};

mod admitted;
mod auxiliary_slots;
mod builder;
mod cancellation;
mod epoch;
mod lakehouse;
mod leases;
mod stream_pool;

pub(crate) use auxiliary_slots::{AttemptAuxiliaryPermit, AttemptAuxiliarySlots};
pub(crate) use builder::ExecutionContextBuilder;
pub(super) use builder::validate_service_stream_pool_device;
pub use builder::{RuntimeObservationBuildError, RuntimeObservedExecutionContextBuilder};
pub use cancellation::ExecutionCancellation;
pub(crate) use epoch::{
    QueryResourceAnchor, QueryResourceEpoch, QueryServiceEpoch, QueryServiceEpochUser, QueryServiceResourceEpoch,
};
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
    /// Revocable service lifetime for runtime/resource acquisition.
    service_epoch: Option<QueryServiceEpoch>,
    /// Revocable lifetime for this admitted query's outward resources.
    query_epoch: Option<QueryResourceEpoch>,
    /// Counted query owner attached only to outward contexts, never the root template.
    query_anchor: Option<Arc<QueryResourceAnchor>>,
    /// Optional server-generation gate for file-backed native I/O.
    io_lifecycle: Option<IoLifecycle>,
    /// Optional caller-provided query-wide diagnostic correlation id.
    query_id: Option<QueryId>,
    /// Optional parent trace id used to group related query executions.
    trace_id: Option<TraceId>,
    /// Engine-minted identity for the physical attempt that admitted this context.
    ///
    /// Direct contexts are not attempt-bound and therefore leave this absent.
    attempt_id: Option<AttemptId>,
    /// Attempt-local carrier for exact resident-cache allocator failures.
    /// Direct contexts carry an inert value because they have no sealed
    /// attempt report that could truthfully own the evidence.
    cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation,
    /// Shared query-local bounded runtime observation session.
    runtime_observation: RuntimeObservationHandle,
    /// Shared terminal owner carried by query-service contexts.
    runtime_observation_owner: Option<RuntimeObservationOwner>,
    /// Current plan-execution scope installed while native execution runs.
    runtime_execution_scope: Option<RuntimeExecutionObservationScope>,
    /// Explicit plan/GPU correlation inherited by service-stream acquisitions.
    runtime_stream_lease_scope: RuntimeStreamLeaseScope,
    /// Shared cudf runtime used for GPU work.
    ///
    /// Runtime ownership is explicit: callers may inject an `Arc` to share a
    /// stream, allocator, and device across query plans, or allow the builder to
    /// create a fresh runtime from the configured device/memory-resource policy.
    runtime: Arc<CudfRuntime>,
    /// Dedicated executor used for synchronous GPU work on this context's device.
    gpu_execution_pool: GpuExecutionPool,
    /// Engine-level memory control contract for bounded execution.
    ///
    /// Bounded fields drive chunk planning, reservation admission, and
    /// retry/fan-in decisions. Source chunk byte and row-group settings are
    /// planning thresholds where row-group metadata permits exact splitting.
    memory_control_policy: MemoryControlPolicy,
    /// Optional native query memory domain that mirrors the current grant budget.
    grant_memory_domain: Option<Arc<cudf::QueryMemoryDomain>>,
    /// Authoritative domain for this service-admitted attempt.
    attempt_allocation_domain: Option<AttemptAllocationDomain>,
    /// Optional soft per-query memory grant backing bounded execution.
    ///
    /// This field is declared after native grant-backed resources so those
    /// resources are dropped before the final grant clone returns capacity to
    /// the ledger.
    memory_grant: Option<MemoryGrant>,
    /// Attempt-owned auxiliary stream allowance installed by query-service admission.
    attempt_auxiliary_slots: Option<Arc<AttemptAuxiliarySlots>>,
    /// Immutable post-grant evidence shared by this context and its query handle.
    /// Direct contexts intentionally carry no evidence.
    admitted_capability_evidence: Option<Arc<AttemptCapabilityEvidence>>,
    activated_deferred_resident_phase: Option<Arc<ActivatedDeferredResidentPhase>>,
    /// Alias removal must happen before the member lease can detach the raw
    /// stream from the attempt domain, and before the shared route can
    /// unregister its native provider mapping.
    _attempt_stream_provider_alias: Option<Arc<cudf::AttemptStreamProviderAlias>>,
    /// This context's shared member lease on its execution stream. One
    /// member per stream for the attempt; per-stage tokens clone it.
    attempt_member_lease: Option<cudf::StreamReservationLease>,
    /// Shared attempt fork-stream routing owner. Aliases are retained by
    /// derived contexts until their corresponding stream lease drains.
    attempt_stream_provider_route: Option<Arc<AttemptStreamProviderRoute>>,
    /// Source scheduling policy for bounded local Parquet reads.
    source_scheduler_policy: SourceSchedulerPolicy,
    /// Source chunk planning mode for bounded local Parquet reads.
    source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Backend-owned source cache for this context's device, when the
    /// ordinal profile opted into resident caching.
    resident_source_cache: Option<GpuLocalParquetSourceCache>,
    /// Backend-owned graph cache for this context's device, when the ordinal
    /// profile opted into resident caching.
    #[cfg(feature = "cugraph")]
    resident_graph_cache: Option<GpuResidentGraphCache>,
    /// Service-level lakehouse cache-fill resource configuration.
    lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    /// Service-level pool used by bounded source-read workers.
    service_stream_pool: ServiceStreamPool,
    /// Service-wide byte budget for complete-object lakehouse cache fills.
    lakehouse_cache_fill_budget: CacheFillByteBudget,
    /// Service-wide operation-count gate for complete-object lakehouse cache fills.
    lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    /// Structured completion/failure events from asynchronous lakehouse cache fills.
    lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    /// Service-scoped async runtime used for object-store control-plane work and cache maintenance.
    object_store_control_runtime: Arc<TokioRuntime>,
    /// Cached object-store clients used by metadata and cache-maintenance work.
    source_object_stores: Arc<SourceObjectStoreRegistry>,
    /// Cooperative cancellation flag observed at native execution boundaries.
    cancellation: ExecutionCancellation,
    /// Selected execution contract for floating statistical aggregates.
    statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    /// Per-query DataFusion exact-order scope for floating statistical aggregates.
    ///
    /// Source-read workers run on spawned OS threads that cannot observe the
    /// caller thread's tracking state, so this flag travels with the context
    /// clone handed to the worker instead of living in a thread-local.
    datafusion_order_tracking: bool,
    /// Runtime threshold policy for grouped aggregate strategy selection.
    aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert policy for local-throughput source chunk planning.
    source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Optional owned stream lease kept alive by contexts derived from it.
    stream_lease_guard: Option<Arc<StreamLease>>,
    /// Auxiliary envelope ownership paired with an owned auxiliary stream.
    auxiliary_stream_permit_guard: Option<Arc<AttemptAuxiliaryPermit>>,
    /// Whether this context was created from a QueryService admission handle.
    query_service_admitted: bool,
}

#[derive(Debug)]
pub(super) struct ActivatedDeferredResidentPhase {
    consumer_execution_scope_fingerprint: String,
    inputs: Vec<ActivatedDeferredResidentInput>,
    capability_evidence: AttemptCapabilityEvidence,
    grant_identity: GrantIdentity,
}

#[derive(Debug)]
pub(super) struct ActivatedDeferredResidentInput {
    producer_execution_scope_fingerprint: String,
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
                    QueryErrorCode::UnsupportedPlanShape,
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
                QueryErrorCode::UnsupportedPlanShape,
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
            || input.producer_execution_scope() != Some(&self.producer_execution_scope_fingerprint)
            || row_count != self.input_row_count
            || (physical_bytes > 0 && !retained_by_grant)
        {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
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

/// Execution context tied to an owned CUDA stream lease.
///
/// This is the preferred API when a caller needs GPU work to run on a leased
/// service stream. Holding this value keeps the underlying stream lease alive
/// for at least as long as the borrowed context can be used.
#[derive(Debug)]
pub(crate) struct LeasedExecutionContext {
    context: ExecutionContext,
}

/// Root execution context paired with its runtime observation owner.
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
    pub(super) fn new(context: ExecutionContext, observation: RuntimeObservationOwner) -> Self {
        Self { context, observation }
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
        let inner = match &context.runtime_observation_owner {
            Some(owner) if context.query_service_admitted => RuntimeExecutionObservationLifecycleInner::Plan(
                RuntimePlanExecutionLifecycle::from_owner(owner.clone(), plan_execution_id),
            ),
            Some(owner) => RuntimeExecutionObservationLifecycleInner::Query(RuntimeQueryLifecycle::from_owner(
                owner.clone(),
                plan_execution_id,
            )),
            None => RuntimeExecutionObservationLifecycleInner::Plan(RuntimePlanExecutionLifecycle::from_handle(
                context.runtime_observation.clone(),
                plan_execution_id,
            )),
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

impl LeasedExecutionContext {
    fn new(context: ExecutionContext) -> Self {
        Self { context }
    }

    /// Consumes the wrapper and keeps the owned stream lease on the context.
    pub(crate) fn into_context(self) -> ExecutionContext {
        self.context
    }
}

#[derive(Debug)]
pub(super) struct ExecutionContextParts {
    pub(super) service_epoch: Option<QueryServiceEpoch>,
    pub(super) query_epoch: Option<QueryResourceEpoch>,
    pub(super) query_anchor: Option<Arc<QueryResourceAnchor>>,
    pub(super) io_lifecycle: Option<IoLifecycle>,
    pub(super) query_id: Option<QueryId>,
    pub(super) trace_id: Option<TraceId>,
    pub(super) attempt_id: Option<AttemptId>,
    pub(super) cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation,
    pub(super) runtime_observation: RuntimeObservationHandle,
    pub(super) runtime_observation_owner: Option<RuntimeObservationOwner>,
    pub(super) runtime_execution_scope: Option<RuntimeExecutionObservationScope>,
    pub(super) runtime_stream_lease_scope: RuntimeStreamLeaseScope,
    pub(super) runtime: Arc<CudfRuntime>,
    pub(super) gpu_execution_pool: GpuExecutionPool,
    pub(super) memory_control_policy: MemoryControlPolicy,
    pub(super) memory_grant: Option<MemoryGrant>,
    pub(super) grant_memory_domain: Option<Arc<cudf::QueryMemoryDomain>>,
    pub(super) attempt_allocation_domain: Option<AttemptAllocationDomain>,
    pub(super) attempt_auxiliary_slots: Option<Arc<AttemptAuxiliarySlots>>,
    pub(super) admitted_capability_evidence: Option<Arc<AttemptCapabilityEvidence>>,
    pub(super) activated_deferred_resident_phase: Option<Arc<ActivatedDeferredResidentPhase>>,
    pub(super) attempt_stream_provider_alias: Option<Arc<cudf::AttemptStreamProviderAlias>>,
    pub(super) attempt_member_lease: Option<cudf::StreamReservationLease>,
    pub(super) attempt_stream_provider_route: Option<Arc<AttemptStreamProviderRoute>>,
    pub(super) source_scheduler_policy: SourceSchedulerPolicy,
    pub(super) source_chunk_planning_mode: SourceChunkPlanningMode,
    pub(super) resident_source_cache: Option<GpuLocalParquetSourceCache>,
    #[cfg(feature = "cugraph")]
    pub(super) resident_graph_cache: Option<GpuResidentGraphCache>,
    pub(super) lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    pub(super) service_stream_pool: ServiceStreamPool,
    pub(super) lakehouse_cache_fill_budget: CacheFillByteBudget,
    pub(super) lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    pub(super) lakehouse_cache_fill_events: LakehouseCacheFillEventSink,
    pub(super) object_store_control_runtime: Arc<TokioRuntime>,
    pub(super) source_object_stores: Arc<SourceObjectStoreRegistry>,
    pub(super) cancellation: ExecutionCancellation,
    pub(super) statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    pub(super) datafusion_order_tracking: bool,
    pub(super) aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    pub(super) source_chunk_planning_options: SourceChunkPlanningOptions,
    pub(super) stream_lease_guard: Option<Arc<StreamLease>>,
    pub(super) auxiliary_stream_permit_guard: Option<Arc<AttemptAuxiliaryPermit>>,
    pub(super) query_service_admitted: bool,
}

impl ExecutionContext {
    /// Starts a builder for an [`ExecutionContext`].
    // Only tests build contexts this way; production contexts come from query-service admission.
    pub fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    /// Builds a context around a shared cudf runtime.
    ///
    /// Use this when multiple plans or adapter layers must share the same CUDA
    /// stream, device placement, or allocator policy.
    // Only tests build contexts this way; production contexts come from query-service admission.
    pub fn from_shared_runtime(runtime: Arc<CudfRuntime>) -> Result<Self> {
        let service_stream_pool =
            ServiceStreamPool::new(runtime.device(), default_service_stream_pool_size_for_device(runtime.device()))
                .expect("default service stream pool size is non-zero");
        let lakehouse_cache_fill_config = LakehouseCacheFillConfig::default();
        let gpu_execution_pool =
            GpuExecutionPool::shared_for_device(runtime.device().ordinal(), service_stream_pool.pool_size())?;
        Ok(Self::from_parts(ExecutionContextParts {
            service_epoch: None,
            query_epoch: None,
            query_anchor: None,
            io_lifecycle: None,
            query_id: None,
            trace_id: None,
            attempt_id: None,
            cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation::default(),
            runtime_observation: RuntimeObservationHandle::disabled(),
            runtime_observation_owner: None,
            runtime_execution_scope: None,
            runtime_stream_lease_scope: RuntimeStreamLeaseScope::default(),
            runtime,
            gpu_execution_pool,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            memory_grant: None,
            grant_memory_domain: None,
            attempt_allocation_domain: None,
            attempt_auxiliary_slots: None,
            admitted_capability_evidence: None,
            activated_deferred_resident_phase: None,
            attempt_member_lease: None,
            attempt_stream_provider_route: None,
            attempt_stream_provider_alias: None,
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            resident_source_cache: None,
            #[cfg(feature = "cugraph")]
            resident_graph_cache: None,
            lakehouse_cache_fill_config,
            service_stream_pool,
            lakehouse_cache_fill_budget: CacheFillByteBudget::default(),
            lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate::new(
                LakehouseCacheFillConfig::DEFAULT_MAX_CONCURRENT_OPS,
            )
            .expect("default lakehouse cache fill concurrency is non-zero"),
            lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            object_store_control_runtime: default_object_store_control_runtime()?,
            source_object_stores: Arc::new(SourceObjectStoreRegistry::new()),
            cancellation: ExecutionCancellation::new(),
            statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::default(),
            datafusion_order_tracking: false,
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            stream_lease_guard: None,
            auxiliary_stream_permit_guard: None,
            query_service_admitted: false,
        }))
    }

    pub(super) fn from_parts(parts: ExecutionContextParts) -> Self {
        Self {
            service_epoch: parts.service_epoch,
            query_epoch: parts.query_epoch,
            query_anchor: parts.query_anchor,
            io_lifecycle: parts.io_lifecycle,
            query_id: parts.query_id,
            trace_id: parts.trace_id,
            attempt_id: parts.attempt_id,
            cache_scope_allocator_correlation: parts.cache_scope_allocator_correlation,
            runtime_observation: parts.runtime_observation,
            runtime_observation_owner: parts.runtime_observation_owner,
            runtime_execution_scope: parts.runtime_execution_scope,
            runtime_stream_lease_scope: parts.runtime_stream_lease_scope,
            runtime: parts.runtime,
            gpu_execution_pool: parts.gpu_execution_pool,
            stream_lease_guard: parts.stream_lease_guard,
            memory_control_policy: parts.memory_control_policy,
            memory_grant: parts.memory_grant,
            grant_memory_domain: parts.grant_memory_domain,
            attempt_allocation_domain: parts.attempt_allocation_domain,
            attempt_auxiliary_slots: parts.attempt_auxiliary_slots,
            admitted_capability_evidence: parts.admitted_capability_evidence,
            activated_deferred_resident_phase: parts.activated_deferred_resident_phase,
            attempt_member_lease: parts.attempt_member_lease,
            attempt_stream_provider_route: parts.attempt_stream_provider_route,
            _attempt_stream_provider_alias: parts.attempt_stream_provider_alias,
            source_scheduler_policy: parts.source_scheduler_policy,
            source_chunk_planning_mode: parts.source_chunk_planning_mode,
            resident_source_cache: parts.resident_source_cache,
            #[cfg(feature = "cugraph")]
            resident_graph_cache: parts.resident_graph_cache,
            lakehouse_cache_fill_config: parts.lakehouse_cache_fill_config,
            service_stream_pool: parts.service_stream_pool,
            lakehouse_cache_fill_budget: parts.lakehouse_cache_fill_budget,
            lakehouse_cache_fill_concurrency: parts.lakehouse_cache_fill_concurrency,
            lakehouse_cache_fill_events: parts.lakehouse_cache_fill_events,
            object_store_control_runtime: parts.object_store_control_runtime,
            source_object_stores: parts.source_object_stores,
            cancellation: parts.cancellation,
            statistical_aggregate_execution_mode: parts.statistical_aggregate_execution_mode,
            datafusion_order_tracking: parts.datafusion_order_tracking,
            aggregate_strategy_execution_options: parts.aggregate_strategy_execution_options,
            source_chunk_planning_options: parts.source_chunk_planning_options,
            auxiliary_stream_permit_guard: parts.auxiliary_stream_permit_guard,
            query_service_admitted: parts.query_service_admitted,
        }
    }

    /// Attaches one counted query owner to a clone of the internal root template.
    pub(super) fn with_query_resource_anchor(&self, anchor: Arc<QueryResourceAnchor>) -> Self {
        debug_assert!(
            self.query_epoch.as_ref().is_some_and(|epoch| anchor.belongs_to(epoch)),
            "query resource anchor must belong to the context query epoch"
        );
        let mut context = self.clone();
        context.query_anchor = Some(anchor);
        context
    }

    fn acquire_resource_epoch(&self, operation: &'static str) -> Result<Option<Arc<QueryServiceResourceEpoch>>> {
        let Some(service_epoch) = &self.service_epoch else {
            if let Some(query_epoch) = &self.query_epoch {
                query_epoch.check_open(operation)?;
            }
            return Ok(None);
        };
        QueryServiceResourceEpoch::acquire(service_epoch, self.query_anchor.as_ref(), operation).map(Some)
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
        Ok(self.runtime.as_ref().clone())
    }

    /// Acquire permission for one file-backed native I/O call.
    ///
    /// Standalone library contexts without a server lifecycle remain
    /// unrestricted. Server contexts carry one generation gate and reject
    /// acquisitions after graceful shutdown begins.
    pub fn acquire_io_lease(&self, operation: &'static str) -> Result<Option<IoLifecycleLease>> {
        self.io_lifecycle.as_ref().map(|lifecycle| lifecycle.acquire(operation)).transpose()
    }

    /// Attach a server-generation I/O lifecycle to this context clone.
    pub fn with_io_lifecycle(mut self, lifecycle: IoLifecycle) -> Self {
        self.io_lifecycle = Some(lifecycle);
        self
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
        self.gpu_execution_pool().map(if self.auxiliary_stream_permit_guard.is_some() {
            GpuExecutorLease::from_pool_for_auxiliary_fragment
        } else {
            GpuExecutorLease::from_pool
        })
    }

    pub(crate) fn gpu_execution_pool(&self) -> Result<GpuExecutionPool> {
        let resource_epoch = self.acquire_resource_epoch("query_engine.runtime.context.gpu_execution_pool")?;
        Ok(match resource_epoch {
            Some(resource_epoch) => self.gpu_execution_pool.with_resource_epoch(resource_epoch),
            None => self.gpu_execution_pool.clone(),
        })
    }

    /// Returns the admitted attempt's fork-stream route, when this context
    /// carries one. Each lease derived from it holds one auxiliary fork of the
    /// route alone, up to [`AttemptStreamProviderRoute::lane_capacity`];
    /// direct contexts have no route and run serially.
    #[doc(hidden)]
    pub fn attempt_stream_provider_route(&self) -> Option<Arc<AttemptStreamProviderRoute>> {
        self.attempt_stream_provider_route.as_ref().map(Arc::clone)
    }

    /// Returns the query-wide diagnostic correlation id.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Returns the engine-minted attempt identity carried by an admitted context.
    ///
    /// This remains absent for direct contexts because they are not part of a
    /// public execution-attempt lifecycle.
    #[doc(hidden)]
    pub fn attempt_id(&self) -> Option<AttemptId> {
        self.attempt_id
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
        self.cache_scope_allocator_correlation.begin(scope)
    }

    /// Returns the shared recording handle without terminal-seal authority.
    pub fn runtime_observation(&self) -> &RuntimeObservationHandle {
        &self.runtime_observation
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
        self.runtime_execution_scope.as_ref()
    }

    /// Returns a clone carrying one invocation's recording handle.
    ///
    /// Seal authority remains with the execution lifecycle that supplied the
    /// handle. Identity is validated before the clone can reach workers.
    #[doc(hidden)]
    pub fn with_runtime_observation_handle(&self, observation: RuntimeObservationHandle) -> Result<Self> {
        if let Some(query_id) = &self.query_id {
            validate_runtime_observation_query_id(&observation, query_id)?;
        }
        if let Some(trace_id) = &self.trace_id {
            validate_runtime_observation_trace_id(&observation, trace_id)?;
        }
        let mut context = self.clone();
        context.runtime_observation = observation;
        Ok(context)
    }

    /// Returns a clone scoped to one native plan invocation.
    #[doc(hidden)]
    pub fn with_runtime_execution_scope(&self, scope: RuntimeExecutionObservationScope) -> Result<Self> {
        let mut context = self.with_runtime_observation_handle(scope.observation().clone())?;
        context.runtime_stream_lease_scope.plan_execution_id = Some(scope.plan_execution_id().clone());
        context.runtime_execution_scope = Some(scope);
        Ok(context)
    }

    /// Returns a clone carrying explicit GPU job/worker correlation.
    pub fn with_runtime_gpu_stream_scope(&self, scope: RuntimeStreamLeaseScope) -> Self {
        let mut context = self.clone();
        context.runtime_stream_lease_scope.gpu_job_id = scope.gpu_job_id;
        context.runtime_stream_lease_scope.gpu_worker_id = scope.gpu_worker_id;
        context
    }

    /// Returns a cloned context with a diagnostic query id consistent with observation identity.
    pub fn with_query_id(&self, query_id: impl Into<QueryId>) -> Result<Self> {
        let mut context = self.clone();
        let query_id = query_id.into();
        validate_runtime_observation_query_id(&context.runtime_observation, &query_id)?;
        context.query_id = Some(query_id);
        Ok(context)
    }

    /// Returns a cloned context with a trace id consistent with observation identity.
    pub fn with_trace_id(&self, trace_id: impl Into<TraceId>) -> Result<Self> {
        let mut context = self.clone();
        let trace_id = trace_id.into();
        validate_runtime_observation_trace_id(&context.runtime_observation, &trace_id)?;
        context.trace_id = Some(trace_id);
        Ok(context)
    }
    /// Returns a context for an internal fragment that must bypass bounded admission.
    ///
    /// Query-local and service-scoped state is preserved; only the engine-level
    /// memory-control policy is relaxed. This is intentionally narrower than
    /// rebuilding from the shared cudf runtime, which would drop cancellation,
    /// source IO profiles, cache handles, and event sinks.
    // Only tests call this downgrade; production contexts come from query-service admission.
    #[cfg(test)]
    pub(crate) fn for_unbounded_fragment(&self) -> Self {
        let mut context = self.clone();
        context.memory_control_policy = MemoryControlPolicy::Unbounded;
        context
    }

    /// Returns whether this context is backed by a QueryService admission slot.
    pub fn is_query_service_admitted(&self) -> bool {
        self.query_service_admitted
    }

    /// Returns the backend-owned local-Parquet cache for this context's
    /// ordinal, if its device profile enabled resident caching.
    #[doc(hidden)]
    pub fn resident_source_cache(&self) -> Option<GpuLocalParquetSourceCache> {
        self.resident_source_cache.clone()
    }

    /// Returns the backend-owned resident graph cache for this context's
    /// ordinal, if its device profile enabled resident caching.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn resident_graph_cache(&self) -> Option<GpuResidentGraphCache> {
        self.resident_graph_cache.clone()
    }
    pub(super) fn install_root_stream_cleanup_completion(
        &self,
        before_quiescence: impl FnOnce(StreamCleanupOutcome) + Send + 'static,
        after_quiescence: impl FnOnce() + Send + 'static,
    ) -> bool {
        let Some(stream_lease) = &self.stream_lease_guard else {
            return false;
        };
        stream_lease.install_cleanup_completion(before_quiescence, after_quiescence)
    }

    pub(super) fn has_root_stream_lease(&self) -> bool {
        self.stream_lease_guard.is_some()
    }

    /// Returns the cooperative cancellation flag for this execution.
    pub fn cancellation(&self) -> &ExecutionCancellation {
        &self.cancellation
    }

    /// Fails with a structured cancellation error if the execution was cancelled.
    pub fn check_cancellation(&self, operation: &'static str) -> Result<()> {
        self.check_service_epoch(operation)?;
        self.cancellation.check(operation)
    }

    fn check_service_epoch(&self, operation: &'static str) -> Result<()> {
        if let Some(epoch) = &self.service_epoch {
            epoch.check_open(operation)?;
        }
        self.query_epoch.as_ref().map_or(Ok(()), |epoch| epoch.check_open(operation))
    }

    /// CUDA device that runs this context's work.
    pub fn device(&self) -> DeviceId {
        self.runtime.device()
    }

    /// Returns the engine-level memory control policy.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        match (self.memory_control_policy, self.memory_grant.as_ref()) {
            (
                MemoryControlPolicy::Bounded {
                    max_source_chunk_bytes,
                    max_row_groups_per_chunk,
                    max_retry_attempts,
                    min_retry_chunk_row_groups,
                    ..
                },
                Some(grant),
            ) => MemoryControlPolicy::Bounded {
                device_budget_bytes: grant.budget_bytes(),
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            },
            (policy, _) => policy,
        }
    }

    /// Returns the soft governor grant carried by this context, when present.
    pub fn memory_grant(&self) -> Result<Option<MemoryGrant>> {
        let Some(grant) = self.memory_grant.as_ref() else {
            return Ok(None);
        };
        let resource_epoch = self.acquire_resource_epoch("query_engine.runtime.context.memory_grant")?;
        Ok(Some(match resource_epoch {
            Some(resource_epoch) => grant.with_resource_epoch(resource_epoch),
            None => grant.clone(),
        }))
    }

    #[doc(hidden)]
    pub fn retain_resident_output_owner(&self, bytes: u64) -> Result<Option<ResidentOwnerLease>> {
        match self.memory_grant()? {
            Some(grant) => grant.retain_resident_owner(bytes),
            None => Ok(None),
        }
    }

    /// Creates a hard execution-budget token for this context's runtime stream.
    ///
    /// Service-admitted contexts attach a member of their immutable attempt
    /// domain; contexts without one use the runtime's standalone reservation
    /// path.
    pub fn execution_budget_token(&self, request: cudf::ExecutionBudgetRequest) -> Result<cudf::ExecutionBudgetToken> {
        let runtime = self.runtime()?;
        if let Some(lease) = self.attempt_member_lease.as_ref() {
            // One member per stream for the attempt's lifetime; per-stage
            // tokens share it rather than attaching a second member.
            return Ok(cudf::ExecutionBudgetToken::from_shared_lease(
                &runtime,
                request,
                lease.clone(),
                cudf::ExecutionBudgetEnforcement::QueryDomainMember,
            ));
        }
        if let Some(domain) = self.grant_memory_domain.as_ref() {
            return cudf::ExecutionBudgetToken::from_runtime_in_query_domain(&runtime, request, domain.as_ref())
                .map_err(|error| {
                    cudf_query_error(
                        "query_engine.runtime.context.execution_budget_token",
                        "failed to create query-domain execution budget token",
                        error,
                    )
                });
        }
        runtime.execution_budget_token(request).map_err(|error| {
            cudf_query_error(
                "query_engine.runtime.context.execution_budget_token",
                "failed to create execution budget token",
                error,
            )
        })
    }

    /// Returns the immutable domain id for this admitted attempt.
    pub fn attempt_allocation_domain_id(&self) -> Option<u64> {
        self.attempt_allocation_domain.as_ref().map(AttemptAllocationDomain::domain_id)
    }

    /// Immutable grant identity owned by this admitted attempt domain.
    #[cfg(test)]
    pub(crate) fn attempt_allocation_grant_identity(&self) -> Option<GrantIdentity> {
        self.attempt_allocation_domain.as_ref().map(AttemptAllocationDomain::grant_identity)
    }

    /// Samples the allocation-domain current and peak usage.
    pub fn attempt_allocation_domain_usage(&self) -> Result<Option<cudf::QueryMemoryDomainUsage>> {
        self.attempt_allocation_domain.as_ref().map(AttemptAllocationDomain::usage).transpose()
    }

    /// Returns the effective bounded device budget used by reservation tracking.
    pub fn bounded_device_budget_bytes(&self) -> Option<u64> {
        match self.memory_control_policy() {
            MemoryControlPolicy::Unbounded => None,
            MemoryControlPolicy::Bounded { device_budget_bytes, .. } => Some(device_budget_bytes),
        }
    }

    /// Returns source scheduling controls for bounded source execution.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.source_scheduler_policy
    }

    /// Returns the source chunk planning mode.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.source_chunk_planning_mode
    }
    /// Returns the selected execution contract for floating statistical aggregates.
    pub fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.statistical_aggregate_execution_mode
    }

    /// Returns whether this query preserves DataFusion input order for floating
    /// statistical aggregates.
    ///
    /// Native execution derives this per query from the aggregate execution
    /// mode and the plan shape, then carries it on every context clone so
    /// source-read workers on spawned threads observe the same scope as the
    /// caller thread.
    pub fn datafusion_order_tracking(&self) -> bool {
        self.datafusion_order_tracking
    }

    /// Attaches the per-query DataFusion exact-order scope to this context clone.
    pub fn with_datafusion_order_tracking(mut self, enabled: bool) -> Self {
        self.datafusion_order_tracking = enabled;
        self
    }

    /// Returns grouped aggregate strategy execution policy.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.aggregate_strategy_execution_options
    }

    /// Returns expert local-throughput source chunk planning policy.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.source_chunk_planning_options
    }
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
