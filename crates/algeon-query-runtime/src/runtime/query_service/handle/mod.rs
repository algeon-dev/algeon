//! Bounded native query-service admission handle.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    super::context::{AttemptAuxiliaryPermit, QueryServiceResourceEpoch},
    resources::QueryServiceResources,
    *,
};
#[cfg(feature = "cugraph")]
use crate::runtime::query_service::LakehouseCacheFillEventSink;
use crate::{
    metrics::TraceId,
    observability::{AttemptAdmissionWaitFact, RuntimeStreamLeaseScope},
    runtime::{
        GpuExecutorLease, StreamPermitClass,
        gpu::{DeviceContext, GrantIdentity},
        gpu_execution_pool::GpuExecutionJobClass,
        service_stream::StreamSlot,
    },
};

mod lifetime;
mod release;
mod report_commit;

/// An auxiliary context's attempt permit and stream slot, reserved before
/// its context-setup job runs.
#[derive(Debug)]
struct AuxiliaryReservation {
    permit: AttemptAuxiliaryPermit,
    stream: StreamSlot,
}

pub(super) use lifetime::QueryHandleLifetime;
use lifetime::{ContextSetupClaim, RootContextSetupClaimGuard};

/// RAII admission handle for one native query.
#[derive(Debug)]
pub struct QueryHandle {
    pub(super) config: QueryExecutionConfig,
    pub(super) queue_bounds: QueueBounds,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) admission_sequence: u64,
    pub(super) admission_wait: Duration,
    pub(super) admission_wait_fact: Option<AttemptAdmissionWaitFact>,
    pub(super) device_budget_bytes: u64,
    pub(super) device_ordinal: usize,
    pub(super) lifetime: Arc<QueryHandleLifetime>,
}

impl QueryHandle {
    /// Stable service-local request id assigned for this admission attempt.
    pub fn admission_request_id(&self) -> AdmissionRequestId {
        self.admission_request_id
    }

    /// Monotonic service-local admission sequence assigned when this handle was admitted.
    pub fn admission_sequence(&self) -> u64 {
        self.admission_sequence
    }

    /// Time spent waiting in the service admission queue.
    pub fn admission_wait(&self) -> Duration {
        self.admission_wait
    }

    /// Returns the final controller-owned queue state for this queued
    /// admission, when the handle was admitted through the bounded queue.
    /// Direct non-queued admission intentionally has no such fact.
    pub fn admission_wait_fact(&self) -> Option<AttemptAdmissionWaitFact> {
        self.admission_wait_fact
    }

    /// Device-memory budget slice assigned to this query.
    pub fn device_budget_bytes(&self) -> u64 {
        self.device_budget_bytes
    }

    /// CUDA device ordinal selected for this query.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Immutable attempt and ledger generation committed for this admission.
    pub fn grant_identity(&self) -> Result<GrantIdentity> {
        self.lifetime.with_resources(|resources| Ok(resources.envelope.memory_grant().identity()))
    }

    /// Bounded memory policy this handle will use for native execution.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.config.memory_policy_for_admitted_query(self.device_budget_bytes)
    }

    /// Source scheduler policy copied into the admitted execution context.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.config.source_scheduler_policy
    }

    /// Source chunk planning mode copied into the admitted execution context.
    pub fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.config.source_chunk_planning_mode
    }

    /// Floating statistical aggregate execution contract assigned to this admitted query.
    pub fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.config.statistical_aggregate_execution_mode
    }

    /// Join and dynamic-filter thresholds assigned to this admitted query.
    #[cfg(test)]
    pub(crate) fn planning_policy(&self) -> PlanningPolicy {
        self.config.planning_policy
    }

    /// Grouped aggregate strategy execution policy assigned to this admitted query.
    pub fn aggregate_strategy_execution_options(&self) -> AggregateStrategyExecutionOptions {
        self.config.aggregate_strategy_execution_options
    }

    /// Expert source chunk planning policy assigned to this admitted query.
    pub fn source_chunk_planning_options(&self) -> SourceChunkPlanningOptions {
        self.config.source_chunk_planning_options
    }

    /// Reserves an auxiliary stream slot on this query's device without
    /// blocking the caller; the slot becomes a lease inside the context-setup
    /// job.
    async fn reserve_auxiliary_stream(
        &self,
        cancellation: &ExecutionCancellation,
        deadline: Instant,
    ) -> Result<StreamSlot> {
        let operation = "query_engine.query_service.context.stream_acquire";
        let service = self.lifetime.service()?;
        let Some(resources) = &service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "query service was constructed without runtime resources",
            ));
        };
        let pool = resources.devices.context_by_ordinal(self.device_ordinal)?.stream_pool().clone();
        pool.reserve_async(
            cancellation,
            deadline,
            operation,
            self.lifetime.observation.clone(),
            RuntimeStreamLeaseScope::default(),
            StreamPermitClass::Auxiliary,
        )
        .await
    }

    fn context_stream_acquire_deadline(&self, cancellation: &ExecutionCancellation) -> Instant {
        let configured = Instant::now().checked_add(self.queue_bounds.max_wait).unwrap_or_else(Instant::now);
        cancellation.deadline().map_or(configured, |deadline| deadline.min(configured))
    }

    /// Acquires an admitted execution context from the service stream pool.
    pub fn acquire_context(&self) -> Result<AdmittedQueryContext> {
        self.acquire_context_with_cancellation(ExecutionCancellation::new())
    }

    /// Acquires an admitted execution context without blocking the caller's async runtime.
    #[cfg(test)]
    pub(crate) async fn acquire_context_async(self: &Arc<Self>) -> Result<AdmittedQueryContext> {
        self.acquire_context_with_cancellation_async(ExecutionCancellation::new()).await
    }

    /// Acquires an admitted execution context on this query's device executor.
    pub async fn acquire_context_with_cancellation_async(
        self: &Arc<Self>,
        cancellation: ExecutionCancellation,
    ) -> Result<AdmittedQueryContext> {
        let operation = "query_engine.query_service.context";
        let setup_claim = self.lifetime.reserve_context_setup_async(&cancellation, operation).await?;
        let root_setup_claim = (setup_claim == ContextSetupClaim::Root)
            .then(|| RootContextSetupClaimGuard::new(Arc::clone(&self.lifetime)));
        let stream_acquire_deadline = self.context_stream_acquire_deadline(&cancellation);
        // Claim the attempt-owned auxiliary share and its stream slot before
        // submitting an execution-class job, so the GPU worker never waits
        // for either. Otherwise excess work from one attempt can fill every
        // global auxiliary executor permit while it waits for its own local
        // limit.
        let auxiliary_reservation = match setup_claim {
            ContextSetupClaim::Auxiliary => {
                let permit = self
                    .lifetime
                    .attempt_auxiliary_slots
                    .acquire_with_cancellation_deadline_async(
                        &cancellation,
                        stream_acquire_deadline,
                        "query_engine.query_service.context.attempt_auxiliary_acquire",
                    )
                    .await?;
                let stream = self.reserve_auxiliary_stream(&cancellation, stream_acquire_deadline).await?;
                Some(AuxiliaryReservation { permit, stream })
            },
            ContextSetupClaim::Root => None,
            ContextSetupClaim::WaitForRoot => {
                unreachable!("context setup reservation resolves waits before scheduling")
            },
        };
        let pool = self.gpu_execution_pool()?;
        let job_class = match setup_claim {
            ContextSetupClaim::Root => GpuExecutionJobClass::Lifecycle,
            ContextSetupClaim::Auxiliary => GpuExecutionJobClass::Execution,
            ContextSetupClaim::WaitForRoot => {
                unreachable!("context setup reservation resolves waits before scheduling")
            },
        };
        let observation = self.lifetime.observation.clone();
        let handle = Arc::clone(self);
        pool.run_observed_scoped_in_class(
            GpuExecutionJobKind::ContextSetup,
            job_class,
            observation,
            move |stream_scope| async move {
                let _root_setup_claim = root_setup_claim;
                handle.acquire_context_with_cancellation_and_stream_scope(
                    cancellation,
                    stream_scope,
                    setup_claim,
                    stream_acquire_deadline,
                    auxiliary_reservation,
                )
            },
        )
        .await
        .and_then(|result| result)
    }

    /// Opaque lifecycle-bound executor for synchronous GPU work on this
    /// query's selected device.
    pub fn gpu_executor(&self) -> Result<GpuExecutorLease> {
        self.gpu_execution_pool().map(GpuExecutorLease::from_pool)
    }

    pub(crate) fn gpu_execution_pool(&self) -> Result<GpuExecutionPool> {
        let operation = "query_engine.query_service.gpu_execution_pool";
        let query_anchor = self.lifetime.acquire_resource_anchor(operation)?;
        let service = self.lifetime.service()?;
        let Some(resources) = &service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "query service was constructed without runtime resources",
            ));
        };
        let pool = resources.gpu_execution_pool_for_device(self.device_ordinal)?;
        let resource_epoch =
            QueryServiceResourceEpoch::acquire(&service.service_epoch, Some(&query_anchor), operation)?;
        Ok(pool.with_resource_epoch(resource_epoch))
    }

    /// Acquires an admitted execution context that observes caller-supplied
    /// cooperative cancellation.
    pub fn acquire_context_with_cancellation(
        &self,
        cancellation: ExecutionCancellation,
    ) -> Result<AdmittedQueryContext> {
        let operation = "query_engine.query_service.context";
        let setup_claim = self.lifetime.reserve_context_setup_sync(&cancellation, operation)?;
        let _root_setup_claim = (setup_claim == ContextSetupClaim::Root)
            .then(|| RootContextSetupClaimGuard::new(Arc::clone(&self.lifetime)));
        let stream_acquire_deadline = self.context_stream_acquire_deadline(&cancellation);
        self.acquire_context_with_cancellation_and_stream_scope(
            cancellation,
            RuntimeStreamLeaseScope::default(),
            setup_claim,
            stream_acquire_deadline,
            None,
        )
    }

    fn acquire_context_with_cancellation_and_stream_scope(
        &self,
        cancellation: ExecutionCancellation,
        stream_scope: RuntimeStreamLeaseScope,
        setup_claim: ContextSetupClaim,
        stream_acquire_deadline: Instant,
        auxiliary_reservation: Option<AuxiliaryReservation>,
    ) -> Result<AdmittedQueryContext> {
        let operation = "query_engine.query_service.context";
        // This is the first instruction after an asynchronous device-worker
        // queue wait. Do not create a context or consume a root binding after
        // the attempt's absolute deadline has elapsed.
        cancellation.check(operation)?;
        let query_anchor = self.lifetime.acquire_resource_anchor(operation)?;
        let service = self.lifetime.service()?;
        let Some(resources) = &service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.query_service.context",
                "query service was constructed without runtime resources",
            ));
        };
        let device = resources.devices.context_by_ordinal(self.device_ordinal)?;
        let template = self.lifetime.with_resources(|lifetime| Ok(lifetime.root_context.clone()))?;
        if let Some(template) = template {
            if setup_claim != ContextSetupClaim::Auxiliary {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "root context was ready without an auxiliary context-setup reservation",
                )
                .with_fact("reason", "context_setup_claim_mismatch")
                .with_fact("context_setup_claim", "root"));
            }
            // Concurrent fragments under one admitted query must not share the
            // root stream: cuDF rejects a second live query-domain budget token
            // on the same stream. Bind each secondary acquire to a dedicated
            // service stream while preserving the shared grant/domain.
            let context = self.clone_auxiliary_context(
                &cancellation,
                device,
                template,
                stream_scope,
                stream_acquire_deadline,
                auxiliary_reservation,
            )?;
            let context = context.with_query_resource_anchor(query_anchor);
            self.lifetime.query_epoch.check_open(operation).map_err(|_error| self.lifetime.released_error())?;
            return Ok(AdmittedQueryContext { context });
        }
        if setup_claim != ContextSetupClaim::Root {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "auxiliary context setup ran before the admitted root context was ready",
            )
            .with_fact("reason", "context_setup_claim_mismatch")
            .with_fact("context_setup_claim", "auxiliary"));
        }
        debug_assert!(auxiliary_reservation.is_none(), "root context setup cannot carry an auxiliary reservation");
        // The admission transaction already owns the root stream, allocation
        // domain, and root stream member. Consume that binding; no stream or
        // domain is created after admission.
        let context = self.bind_root_execution_context(cancellation, stream_scope, device, &service, resources)?;
        self.lifetime.complete_root_context_setup();
        let context = context.with_query_resource_anchor(query_anchor);
        self.lifetime.query_epoch.check_open(operation).map_err(|_error| self.lifetime.released_error())?;
        Ok(AdmittedQueryContext { context })
    }

    fn clone_auxiliary_context(
        &self,
        cancellation: &ExecutionCancellation,
        device: &DeviceContext,
        template: ExecutionContext,
        stream_scope: RuntimeStreamLeaseScope,
        stream_acquire_deadline: Instant,
        auxiliary_reservation: Option<AuxiliaryReservation>,
    ) -> Result<ExecutionContext> {
        let operation = "query_engine.query_service.context.stream_acquire";
        let (attempt_auxiliary_permit, stream_lease) = match auxiliary_reservation {
            Some(AuxiliaryReservation { permit, stream }) => (permit, stream.into_lease(operation)?),
            None => {
                let permit = self.lifetime.attempt_auxiliary_slots.acquire_with_cancellation_deadline(
                    cancellation,
                    stream_acquire_deadline,
                    "query_engine.query_service.context.attempt_auxiliary_acquire",
                )?;
                let stream_lease = device.stream_pool().acquire_observed_with_cancellation_deadline_and_class(
                    cancellation,
                    stream_acquire_deadline,
                    operation,
                    self.lifetime.observation.clone(),
                    stream_scope,
                    StreamPermitClass::Auxiliary,
                )?;
                (permit, stream_lease)
            },
        };
        validate_service_stream_pool_device(device.stream_pool(), stream_lease.device())?;
        template.with_owned_auxiliary_stream(stream_lease, attempt_auxiliary_permit)
    }

    /// Binds the attempt's root execution context. Only the root setup claim
    /// holder calls this. The lifetime lock is held only to take the root
    /// binding and to store the result, so a release request never waits on
    /// the binding; the query anchor the caller holds keeps the release from
    /// taking the resources in between.
    fn bind_root_execution_context(
        &self,
        cancellation: ExecutionCancellation,
        stream_scope: RuntimeStreamLeaseScope,
        device: &DeviceContext,
        service: &QueryServiceInner,
        resources: &QueryServiceResources,
    ) -> Result<ExecutionContext> {
        let device_runtime = resources.device_runtime(self.device_ordinal)?;
        let (binding, memory_grant, allocation_domain, attempt) = self.lifetime.with_resources(|lifetime| {
            let envelope = &lifetime.envelope;
            // The envelope's worker scope is the admission-time identity
            // evidence: same device, same worker pool, and the lane the
            // attempt has held since the ledger transaction committed.
            if let Some(scope) = envelope.worker_scope() {
                debug_assert_eq!(scope.device_ordinal(), self.device_ordinal);
                debug_assert_eq!(scope.lane_index(), envelope.lane_index().expect("lane matches its scope"));
                debug_assert_eq!(scope.pool_id(), device_runtime.gpu_execution_pool.snapshot().pool_id);
            }
            let binding = envelope.take_root_binding()?;
            let memory_grant = envelope.memory_grant().clone();
            let allocation_domain = envelope.domain().cloned().ok_or_else(|| self.lifetime.released_error())?;
            let attempt = lifetime.attempt().ok_or_else(|| self.lifetime.released_error())?;
            Ok((binding, memory_grant, allocation_domain, attempt))
        })?;
        #[cfg(test)]
        self.lifetime.wait_at_root_context_binding_gate();
        validate_service_stream_pool_device(device.stream_pool(), binding.stream_lease.device())?;
        // The grant, its allocation domain, and the attempt identity come from
        // one admission envelope, so plan validation does not re-check them.
        debug_assert_eq!(allocation_domain.grant_identity(), memory_grant.identity());
        debug_assert_eq!(memory_grant.identity().attempt_id, self.lifetime.attempt_id);
        debug_assert_eq!(memory_grant.device_ordinal(), binding.stream_lease.device().ordinal());
        let runtime = binding.stream_lease.shared_runtime()?;
        let stream_provider_route = Arc::new(
            allocation_domain
                .create_stream_provider_route(runtime.stream(), self.lifetime.attempt_auxiliary_slots.capacity())?,
        );
        let query_id = self.lifetime.observation.query_id().map(|query_id| QueryId::new(query_id.as_str()));
        let trace_id = self.lifetime.observation.trace_id().map(|trace_id| TraceId::new(trace_id.as_str()));
        let attempt = AdmittedAttempt {
            attempt_id: self.lifetime.attempt_id,
            service_epoch: service.service_epoch.clone(),
            query_epoch: self.lifetime.query_epoch.clone(),
            cancellation,
            attempt,
            #[cfg(feature = "cugraph")]
            cache_scope_allocator_correlation: self.lifetime.cache_scope_allocator_correlation(),
            lakehouse_cache_fill_events: LakehouseCacheFillEventSink::default(),
            capability_evidence: self.lifetime.admitted_capability_evidence.clone(),
            memory_control_policy: memory_control_policy_for_grant(self.memory_control_policy(), &memory_grant),
            auxiliary_slots: Arc::clone(&self.lifetime.attempt_auxiliary_slots),
            allocation_domain,
            memory_grant,
        };
        // The root execution stream binds the attempt's root domain member
        // once for the whole attempt lifetime.
        let context = ExecutionContext::admitted_root(
            device_runtime,
            attempt,
            StreamBinding::AttemptRoot {
                runtime,
                member: binding.member_lease,
                route: Arc::clone(&stream_provider_route),
                lease: binding.stream_lease,
            },
            PlanScope::root(query_id, trace_id, self.lifetime.observation.clone(), stream_scope),
        );
        self.lifetime.with_resources(|lifetime| {
            debug_assert!(
                lifetime.attempt_stream_provider_route.is_none(),
                "an admitted attempt can install only one stream-provider route"
            );
            lifetime.attempt_stream_provider_route = Some(Arc::clone(&stream_provider_route));
            lifetime.root_context = Some(context.clone());
            Ok(())
        })?;
        Ok(context)
    }

    /// Requests admission release before dropping the handle.
    ///
    /// The attempt requests this itself once its outcome is terminal, and the
    /// release waits for contexts still running, so work run under an attempt
    /// never calls it. A caller that releases a grant early ends a still-live
    /// attempt as `Dropped`. Runtime-backed services complete GPU resource
    /// destruction on the selected device executor before waking queued
    /// admissions.
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
pub struct AdmittedQueryContext {
    context: ExecutionContext,
}

impl AdmittedQueryContext {
    /// Native execution context for this admitted query.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
}
