//! The four lifetimes an execution context is assembled from.
//!
//! A context clone is four reference-count increments. Deriving a context for
//! another stream replaces only its [`StreamBinding`], and a per-plan or
//! per-caller overlay replaces only its [`PlanScope`], so neither can drop a
//! field of the other scopes.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use cudf::{QueryMemoryDomain, Runtime as CudfRuntime, StreamReservationLease};
use query_planning::{
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    correlation::{QueryId, TraceId},
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::{MemoryControlPolicy, SourceChunkPlanningMode},
    mode::StatisticalAggregateExecutionMode,
    policy::PlanningPolicy,
};
use tokio::runtime::Runtime as TokioRuntime;

#[cfg(feature = "cugraph")]
use super::super::{
    allocator_correlation::AttemptCacheScopeAllocatorCorrelation, cache::resident_graph::GpuResidentGraphCache,
};
use super::{
    super::{
        CacheFillByteBudget, GpuExecutionPool, LakehouseCacheFillConfig, MemoryGrant, ServiceStreamPool,
        SourceSchedulerPolicy, StreamLease,
        attempt_domain::{AttemptAllocationDomain, AttemptStreamProviderRoute},
        cache::local_parquet::GpuLocalParquetSourceCache,
        cache_fill::CacheFillConcurrencyGate,
        io_lifecycle::IoLifecycle,
        object_store::SourceObjectStoreRegistry,
    },
    ActivatedDeferredResidentPhase, AdmittedPlanIdentities, AttemptAuxiliaryPermit, AttemptAuxiliarySlots,
    ExecutionCancellation, QueryResourceAnchor, QueryResourceEpoch, QueryServiceEpoch,
};
use crate::{
    attempt::AttemptHandle,
    native_error::cudf_query_error,
    observability::{
        AttemptId, RuntimeExecutionObservationScope, RuntimeObservationHandle, RuntimeObservationOwner,
        RuntimeStreamLeaseScope,
    },
};

/// Backend and device scope shared by every context on one device: worker and
/// stream pools, resident caches, lakehouse cache-fill services, object-store
/// clients, and the execution policies copied from the service config.
///
/// A query service builds one per installed device; a standalone context
/// builds its own.
#[derive(Debug)]
pub(in crate::runtime) struct DeviceRuntime {
    pub(in crate::runtime) gpu_execution_pool: GpuExecutionPool,
    pub(in crate::runtime) service_stream_pool: ServiceStreamPool,
    pub(in crate::runtime) resident_source_cache: Option<GpuLocalParquetSourceCache>,
    #[cfg(feature = "cugraph")]
    pub(in crate::runtime) resident_graph_cache: Option<GpuResidentGraphCache>,
    pub(in crate::runtime) lakehouse_cache_fill_config: LakehouseCacheFillConfig,
    /// Service-wide byte budget for complete-object lakehouse cache fills.
    pub(in crate::runtime) lakehouse_cache_fill_budget: CacheFillByteBudget,
    /// Service-wide operation-count gate for complete-object lakehouse cache fills.
    pub(in crate::runtime) lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate,
    /// Async runtime for object-store control-plane work and cache maintenance.
    pub(in crate::runtime) object_store_control_runtime: Arc<TokioRuntime>,
    pub(in crate::runtime) source_object_stores: Arc<SourceObjectStoreRegistry>,
    pub(in crate::runtime) source_scheduler_policy: SourceSchedulerPolicy,
    pub(in crate::runtime) source_chunk_planning_mode: SourceChunkPlanningMode,
    pub(in crate::runtime) source_chunk_planning_options: SourceChunkPlanningOptions,
    pub(in crate::runtime) statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    pub(in crate::runtime) aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    pub(in crate::runtime) planning_policy: PlanningPolicy,
}

/// Query scope: a standalone query, or the admitted attempt that owns the
/// grant and its allocation domain.
#[derive(Debug, Clone)]
pub(super) enum QueryScope {
    /// No service admitted this query; its cap is its own allocation domain
    /// and nothing revokes it.
    #[cfg_attr(
        not(any(test, feature = "test-hooks")),
        expect(dead_code, reason = "standalone contexts are built only under test-hooks")
    )]
    Standalone(Arc<StandaloneQuery>),
    /// One admitted attempt; every context of the attempt shares this value.
    Admitted(Arc<AdmittedAttempt>),
}

/// A query that no service admitted, with its own immutable allocator domain.
#[derive(Debug)]
pub(super) struct StandaloneQuery {
    pub(super) allocation_domain: QueryMemoryDomain,
    /// Member ids are unique within a domain. A standalone query mints them
    /// here because no admission envelope does it for it.
    pub(super) next_member_id: AtomicU64,
    pub(super) memory_control_policy: MemoryControlPolicy,
    pub(super) cancellation: ExecutionCancellation,
    /// Present when the standalone context owns its runtime observation; the
    /// context then terminalizes the query itself.
    pub(super) observation_owner: Option<RuntimeObservationOwner>,
}

impl StandaloneQuery {
    /// Binds the context's own runtime. Runtimes and outputs derived from it
    /// retain the root member, so callers may keep allocating through
    /// `shared_runtime()` under the same cap after the context drops.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(super) fn bind_root(&self, runtime: Arc<CudfRuntime>) -> Result<StreamBinding> {
        let (member, derived) = self.attach_member(&runtime)?;
        let derived = derived
            .with_retained_lifetime(Arc::new(member.clone()))
            .map_err(|error| cudf_query_error(STANDALONE_BIND, "failed to retain standalone root member", error))?;
        Ok(StreamBinding::Standalone { member, runtime: Arc::new(derived) })
    }

    /// Binds a leased work stream for the lifetime of the work context, as an
    /// attempt binds its work members: the member detaches when the binding
    /// drops, and allocations it made stay charged to the domain until they
    /// are freed. The pooled stream can then join this or another domain.
    pub(super) fn bind_work(&self, runtime: Arc<CudfRuntime>) -> Result<StreamBinding> {
        let (member, derived) = self.attach_member(&runtime)?;
        Ok(StreamBinding::Standalone { member, runtime: Arc::new(derived) })
    }

    fn attach_member(&self, runtime: &CudfRuntime) -> Result<(StreamReservationLease, CudfRuntime)> {
        let limit = usize::try_from(self.memory_control_policy.device_budget_bytes).map_err(|_| {
            QueryError::new(QueryErrorCode::InvalidPlan, STANDALONE_BIND, "standalone member limit exceeds usize")
        })?;
        let member = self
            .allocation_domain
            .reserve_stream(runtime.stream(), self.next_member_id.fetch_add(1, Ordering::Relaxed), limit)
            .map_err(|error| cudf_query_error(STANDALONE_BIND, "failed to attach standalone stream member", error))?;
        let derived = runtime
            .with_memory_resource_override(member.manager().resource(), STANDALONE_BIND)
            .map_err(|error| cudf_query_error(STANDALONE_BIND, "failed to route standalone stream member", error))?;
        Ok((member, derived))
    }
}

const STANDALONE_BIND: &str = "query_engine.runtime.standalone.bind_stream";

/// The admitted binding of one attempt: its epochs, immutable grant, and the
/// allocation domain and auxiliary slots that grant backs.
#[derive(Debug)]
pub(in crate::runtime) struct AdmittedAttempt {
    pub(in crate::runtime) attempt_id: AttemptId,
    /// Revocable service lifetime for runtime and resource acquisition.
    pub(in crate::runtime) service_epoch: QueryServiceEpoch,
    /// Revocable lifetime for this attempt's outward resources.
    pub(in crate::runtime) query_epoch: QueryResourceEpoch,
    /// The attempt's token, shared with `AttemptHandle::cancellation`.
    /// `ExecutionContext::check_cancellation` reads it after both epochs.
    pub(in crate::runtime) cancellation: ExecutionCancellation,
    /// The attempt this context executes under; its plans record their
    /// boundaries on its observation.
    pub(in crate::runtime) attempt: AttemptHandle,
    /// Carrier for exact resident-cache allocator failures, sealed into the
    /// attempt report.
    #[cfg(feature = "cugraph")]
    pub(in crate::runtime) cache_scope_allocator_correlation: AttemptCacheScopeAllocatorCorrelation,
    /// The plan identities this attempt may execute, shared with the query
    /// handle so a fragment that joins the grant reaches every context.
    pub(in crate::runtime) plan_identities: Arc<AdmittedPlanIdentities>,
    /// The service policy with its device budget set to the grant, fixed at admission.
    pub(in crate::runtime) memory_control_policy: MemoryControlPolicy,
    /// Owned by the query handle lifetime and shared by every context of the
    /// attempt, so permits taken on any clone draw from the one `S_d`.
    pub(in crate::runtime) auxiliary_slots: Arc<AttemptAuxiliarySlots>,
    pub(in crate::runtime) allocation_domain: AttemptAllocationDomain,
    /// Declared after the allocation domain so the domain's native resources
    /// release before this grant clone returns capacity to the ledger.
    pub(in crate::runtime) memory_grant: MemoryGrant,
}

/// The stream a context runs its GPU work on, with the attempt-domain state
/// bound to that stream.
///
/// The attempt's fork-stream route is registered on its root stream, and
/// every attempt stream carries a clone of it. Variant fields drop in
/// declaration order: an alias is removed before its member lease detaches
/// the raw stream from the attempt domain and before the route can unregister
/// its native mapping; the member and route clone release before the stream
/// lease, because dropping the root lease runs the attempt's terminal cleanup,
/// which finds any member still attached as residual domain usage.
#[derive(Debug)]
pub(in crate::runtime) enum StreamBinding {
    /// A standalone context's runtime: its own, or a leased stream's.
    Standalone { member: StreamReservationLease, runtime: Arc<CudfRuntime> },
    /// The admitted attempt's root stream, its root domain member, and the
    /// fork-stream route registered on it.
    AttemptRoot {
        runtime: Arc<CudfRuntime>,
        member: StreamReservationLease,
        route: Arc<AttemptStreamProviderRoute>,
        lease: StreamLease,
    },
    /// Attempt work on a leased stream. The alias holds one auxiliary fork of
    /// the route alone for the binding's lifetime.
    AttemptWork {
        runtime: Arc<CudfRuntime>,
        alias: Arc<cudf::AttemptStreamProviderAlias>,
        member: StreamReservationLease,
        route: Arc<AttemptStreamProviderRoute>,
        /// The stream lease and attempt permit of an auxiliary fragment
        /// context, released together; a borrowed lease leaves this empty.
        owned: Option<(StreamLease, AttemptAuxiliaryPermit)>,
    },
}

impl StreamBinding {
    pub(super) fn runtime(&self) -> &Arc<CudfRuntime> {
        match self {
            Self::Standalone { runtime, .. }
            | Self::AttemptRoot { runtime, .. }
            | Self::AttemptWork { runtime, .. } => runtime,
        }
    }

    pub(super) fn member(&self) -> &StreamReservationLease {
        match self {
            Self::Standalone { member, .. } | Self::AttemptRoot { member, .. } | Self::AttemptWork { member, .. } => {
                member
            },
        }
    }

    pub(super) fn route(&self) -> Option<&Arc<AttemptStreamProviderRoute>> {
        match self {
            Self::Standalone { .. } => None,
            Self::AttemptRoot { route, .. } | Self::AttemptWork { route, .. } => Some(route),
        }
    }
}

/// Per-plan and per-caller state layered over one context: the observation
/// handle and scope, stream-lease correlation, order tracking, the activated
/// deferred-resident phase, the outward query anchor, and the correlation ids
/// and I/O lifecycle a caller stamps on its clone.
#[derive(Debug, Clone)]
pub(in crate::runtime) struct PlanScope {
    /// Counted query owner attached only to outward contexts, never the root
    /// template.
    pub(super) query_anchor: Option<Arc<QueryResourceAnchor>>,
    pub(super) query_id: Option<QueryId>,
    pub(super) trace_id: Option<TraceId>,
    /// Server-generation gate for file-backed native I/O.
    pub(super) io_lifecycle: Option<IoLifecycle>,
    pub(super) runtime_observation: RuntimeObservationHandle,
    pub(super) runtime_execution_scope: Option<RuntimeExecutionObservationScope>,
    pub(super) runtime_stream_lease_scope: RuntimeStreamLeaseScope,
    /// Per-query DataFusion exact-order scope for floating statistical
    /// aggregates. Source-read workers run on spawned threads that cannot
    /// observe the caller thread's state, so it travels with the context.
    pub(super) datafusion_order_tracking: bool,
    pub(super) activated_deferred_resident_phase: Option<Arc<ActivatedDeferredResidentPhase>>,
}

impl PlanScope {
    /// The plan scope of a root context, before any plan runs on it.
    pub(in crate::runtime) fn root(
        query_id: Option<QueryId>,
        trace_id: Option<TraceId>,
        runtime_observation: RuntimeObservationHandle,
        runtime_stream_lease_scope: RuntimeStreamLeaseScope,
    ) -> Self {
        Self {
            query_anchor: None,
            query_id,
            trace_id,
            io_lifecycle: None,
            runtime_observation,
            runtime_execution_scope: None,
            runtime_stream_lease_scope,
            datafusion_order_tracking: false,
            activated_deferred_resident_phase: None,
        }
    }
}
