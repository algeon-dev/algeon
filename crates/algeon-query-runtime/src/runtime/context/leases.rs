//! Attempt work leases and stream-lease derivation for execution contexts.

use std::sync::Arc;

use query_planning::error::Result;

use super::{
    super::{AttemptWorkLease, MemoryResourcePolicy, StreamLease},
    AttemptAuxiliaryPermit, ExecutionContext, ExecutionContextParts, LeasedExecutionContext,
    validate_service_stream_pool_device,
};
use crate::observability::{RuntimeTaskId, RuntimeWorkerId};

impl ExecutionContext {
    /// Acquires one attempt work lease: an attempt auxiliary permit, a
    /// service stream lease, and the derived context that owns them.
    ///
    /// This is the single bundle behind every piece of auxiliary attempt work
    /// -- execution lanes and scheduled source reads alike. The derived
    /// context's alias holds one auxiliary fork of the attempt pool alone for
    /// the lease's lifetime, so its native forks never reach a stream another
    /// live holder is launching on. The permit it takes is what bounds those
    /// claims to the pool's width.
    #[doc(hidden)]
    pub fn acquire_attempt_work_lease(
        &self,
        source_worker: Option<(RuntimeTaskId, RuntimeWorkerId)>,
    ) -> Result<AttemptWorkLease> {
        // Admission reserves this bounded auxiliary share per attempt. Hold
        // the local permit for the exact lifetime of the derived stream lease,
        // so another attempt's reservation remains physically reachable.
        let attempt_auxiliary_permit = self
            .attempt_auxiliary_slots
            .as_ref()
            .map(|slots| {
                slots.acquire_with_cancellation(
                    &self.cancellation,
                    "query_engine.runtime.attempt_work_lease.attempt_auxiliary_acquire",
                )
            })
            .transpose()?;
        self.derive_attempt_work_lease(source_worker, attempt_auxiliary_permit)
    }

    /// Acquires one attempt work lease only if this attempt's auxiliary
    /// envelope has a free slot right now.
    ///
    /// Callers that already hold an attempt auxiliary permit -- the root
    /// execution job dispatching lanes among them -- must never block on a
    /// second one, because the holder they would wait for may be waiting on
    /// them. Returns `Ok(None)` when the envelope is fully claimed, so the
    /// caller narrows its own width instead of deadlocking.
    #[doc(hidden)]
    pub fn try_acquire_attempt_work_lease(&self) -> Result<Option<AttemptWorkLease>> {
        let attempt_auxiliary_permit = match self.attempt_auxiliary_slots.as_ref() {
            Some(slots) => {
                match slots.try_acquire("query_engine.runtime.attempt_work_lease.attempt_auxiliary_try_acquire")? {
                    Some(permit) => Some(permit),
                    None => return Ok(None),
                }
            },
            None => None,
        };
        self.derive_attempt_work_lease(None, attempt_auxiliary_permit).map(Some)
    }

    fn derive_attempt_work_lease(
        &self,
        source_worker: Option<(RuntimeTaskId, RuntimeWorkerId)>,
        attempt_auxiliary_permit: Option<AttemptAuxiliaryPermit>,
    ) -> Result<AttemptWorkLease> {
        let mut stream_scope = self.runtime_stream_lease_scope.clone();
        if let Some((task_id, worker_id)) = source_worker {
            stream_scope = stream_scope.with_source_worker(task_id, worker_id);
        }
        let stream_lease = self.service_stream_pool().acquire_observed_with_cancellation(
            &self.cancellation,
            "query_engine.runtime.attempt_work_lease.acquire",
            self.runtime_observation.clone(),
            stream_scope,
        )?;
        let context = self.with_stream_lease(&stream_lease)?;
        Ok(AttemptWorkLease::new(context, stream_lease, attempt_auxiliary_permit))
    }

    /// Whether this context runs on a stream leased for attempt work rather
    /// than on the attempt's own root stream.
    ///
    /// Work issued from such a context already holds an attempt auxiliary
    /// permit, so it must run inline instead of leasing a second one.
    #[doc(hidden)]
    pub fn is_attempt_work_lease_context(&self) -> bool {
        self._attempt_stream_provider_alias.is_some()
    }

    /// Which auxiliary fork of the attempt pool this context's alias holds
    /// alone. A context with no alias -- the attempt root, or a direct context
    /// -- reports `None`; the root runs on the route's reserved fork instead.
    #[doc(hidden)]
    pub fn attempt_claimed_fork_index(&self) -> Option<usize> {
        self._attempt_stream_provider_alias.as_ref().map(|alias| alias.claimed_fork_index())
    }

    /// Builds a short-lived execution context from a borrowed stream lease.
    ///
    /// The returned context shares this context's query policies, caches, and
    /// cancellation flag, but executes GPU work on the leased runtime. The
    /// caller must keep `stream_lease` alive until all work submitted through
    /// the returned context has completed.
    pub fn with_stream_lease(&self, stream_lease: &StreamLease) -> Result<Self> {
        self.check_service_epoch("query_engine.runtime.context.with_stream_lease")?;
        validate_service_stream_pool_device(stream_lease.pool(), self.runtime.device())?;
        let runtime = stream_lease.runtime()?;
        let attempt_member_lease = match self.attempt_allocation_domain.as_ref() {
            Some(domain) => Some(domain.create_member(runtime.stream())?),
            None => None,
        };
        let attempt_stream_provider_alias = self
            .attempt_stream_provider_route
            .as_ref()
            .map(|route| route.attach(runtime.stream()).map(Arc::new))
            .transpose()?;
        Ok(Self::from_parts(ExecutionContextParts {
            service_epoch: self.service_epoch.clone(),
            query_epoch: self.query_epoch.clone(),
            query_anchor: self.query_anchor.clone(),
            io_lifecycle: self.io_lifecycle.clone(),
            query_id: self.query_id.clone(),
            trace_id: self.trace_id.clone(),
            attempt_id: self.attempt_id,
            cache_scope_allocator_correlation: self.cache_scope_allocator_correlation.clone(),
            runtime_observation: self.runtime_observation.clone(),
            runtime_observation_owner: self.runtime_observation_owner.clone(),
            runtime_execution_scope: self.runtime_execution_scope.clone(),
            runtime_stream_lease_scope: self.runtime_stream_lease_scope.clone(),
            runtime: Arc::new(runtime),
            gpu_execution_pool: self.gpu_execution_pool.clone(),
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy,
            memory_grant: self.memory_grant.clone(),
            grant_memory_domain: self.grant_memory_domain.clone(),
            attempt_allocation_domain: self.attempt_allocation_domain.clone(),
            attempt_auxiliary_slots: self.attempt_auxiliary_slots.as_ref().map(Arc::clone),
            admitted_capability_evidence: self.admitted_capability_evidence.clone(),
            attempt_member_lease,
            attempt_stream_provider_route: self.attempt_stream_provider_route.as_ref().map(Arc::clone),
            attempt_stream_provider_alias,
            boundary_policy: self.boundary_policy,
            source_scheduler_policy: self.source_scheduler_policy,
            source_chunk_planning_mode: self.source_chunk_planning_mode,
            resident_source_cache: self.resident_source_cache.clone(),
            #[cfg(feature = "cugraph")]
            resident_graph_cache: self.resident_graph_cache.clone(),
            lakehouse_cache_fill_config: self.lakehouse_cache_fill_config.clone(),
            service_stream_pool: self.service_stream_pool.clone(),
            lakehouse_cache_fill_budget: self.lakehouse_cache_fill_budget.clone(),
            lakehouse_cache_fill_concurrency: self.lakehouse_cache_fill_concurrency.clone(),
            lakehouse_cache_fill_events: self.lakehouse_cache_fill_events.clone(),
            object_store_control_runtime: Arc::clone(&self.object_store_control_runtime),
            source_object_stores: Arc::clone(&self.source_object_stores),
            deployment_io_capability: self.deployment_io_capability.clone(),
            cancellation: self.cancellation.clone(),
            statistical_aggregate_execution_mode: self.statistical_aggregate_execution_mode,
            datafusion_order_tracking: self.datafusion_order_tracking,
            aggregate_strategy_execution_options: self.aggregate_strategy_execution_options,
            source_chunk_planning_options: self.source_chunk_planning_options,
            stream_lease_guard: None,
            auxiliary_stream_permit_guard: None,
            query_service_admitted: self.query_service_admitted,
        }))
    }

    /// Builds a context and keeps the stream lease alive in the returned wrapper.
    pub fn with_owned_stream_lease(&self, stream_lease: StreamLease) -> Result<LeasedExecutionContext> {
        let stream_lease_guard = Arc::new(stream_lease);
        let mut context = self.with_stream_lease(stream_lease_guard.as_ref())?;
        context.stream_lease_guard = Some(stream_lease_guard);
        Ok(LeasedExecutionContext::new(context))
    }

    /// Pairs one admitted auxiliary permit with this context's owned stream.
    ///
    /// The permit is reference-counted with the stream guard, so cloning an
    /// outward context cannot retain the stream while releasing its attempt
    /// envelope share early.
    #[doc(hidden)]
    pub fn with_auxiliary_stream_permit_guard(mut self, permit: AttemptAuxiliaryPermit) -> Self {
        debug_assert!(self.stream_lease_guard.is_some(), "auxiliary permit must be paired with an owned stream lease");
        debug_assert!(
            self.auxiliary_stream_permit_guard.is_none(),
            "an owned auxiliary stream has exactly one attempt permit"
        );
        self.auxiliary_stream_permit_guard = Some(Arc::new(permit));
        self
    }
}
